"""
This python script is adapted from the origninal script:
    https://github.com/XzwHan/CARD
    CARD: Classification and Regression Diffusion Models by Xizewen Han,
    Huangjie Zheng, and Mingyuan Zhou.
"""

import os
import logging
import time
import gc
#import sys
#import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from scipy.special import logsumexp

from models.dalstm import DALSTMModel
from models.conditional_guided import ConditionalGuidedModel
from utils.diffusion_utils import make_beta_schedule, EMA
from utils.diffusion_utils import q_sample, p_sample_loop
from utils.early_stopping import EarlyStopping
#from model import *
#from utils import *
#from diffusion_utils import *
from utils import get_dataset, get_optimizer


plt.style.use('ggplot')


class Diffusion(object):
    def __init__(self, args, config, device=None, kfold_handler=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device
        self.kfold_handler = kfold_handler
        self.model_var_type = config.model.var_type
        self.num_timesteps = config.diffusion.timesteps
        self.vis_step = config.diffusion.vis_step
        self.num_figs = config.diffusion.num_figs
        self.dataset_object = None
        # get betas based on defined schedule in the config file.
        betas = make_beta_schedule(schedule=config.diffusion.beta_schedule,
                                   num_timesteps=self.num_timesteps,
                                   start=config.diffusion.beta_start,
                                   end=config.diffusion.beta_end)
        betas = self.betas = betas.float().to(self.device)
        #TODO: understand how the followings correspond to the formulas in the original paper.
        self.betas_sqrt = torch.sqrt(betas)
        alphas = 1.0 - betas
        self.alphas = alphas
        self.one_minus_betas_sqrt = torch.sqrt(alphas)
        alphas_cumprod = alphas.cumprod(dim=0)
        self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)
        if config.diffusion.beta_schedule == "cosine":
            # avoid division by 0 for 1/sqrt(alpha_bar_t) during inference
            self.one_minus_alphas_bar_sqrt *= 0.9999  
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.posterior_mean_coeff_1 = (
                betas * torch.sqrt(alphas_cumprod_prev) / (
                    1.0 - alphas_cumprod))
        self.posterior_mean_coeff_2 = (
                torch.sqrt(alphas) * (1 - alphas_cumprod_prev) / (
                    1 - alphas_cumprod))
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = posterior_variance
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()
        self.tau = None  # precision for test NLL computation
        # define loss function for point estimator model
        if self.args.loss_guidance == 'L2':
            self.aux_cost_function = nn.MSELoss()
        else:
            self.aux_cost_function = nn.L1Loss()        
        # initial Prediction model as guided condition
        self.cond_pred_model = None
        if config.diffusion.conditioning_signal == "DALSTM":
            self.cond_pred_model = DALSTMModel(input_size=config.model.x_dim,
                                               hidden_size=config.diffusion.nonlinear_guidance.hidden_size,
                                               n_layers=config.diffusion.nonlinear_guidance.n_layers,
                                               max_len=config.model.max_len,
                                               linear_hidden_size=config.diffusion.nonlinear_guidance.linear_hidden_size,
                                               dropout=config.diffusion.nonlinear_guidance.dropout,
                                               p_fix=config.diffusion.nonlinear_guidance.dropout_rate
                                               ).to(self.device)          
        else:
            #TODO: implementation for ProcessTransformer and PGTNet
            print('Currently only DALSTM model is supported.')
            pass

    # Compute guiding Prediction as diffusion condition
    def compute_guiding_Prediction(self, x):
        # Compute y_0_hat, to be used as the Gaussian mean at time step T.
        y_pred = self.cond_pred_model(x)
        return y_pred

    def evaluate_guidance_model(self, dataset_object, dataset_loader,
                                eval_mode=None):
        """
        Evaluate guidance model: y MAE or y RMSE for train/test set.
        eval_mod control the behavior to work with un/normalized data.
        """
        # option to work with L1/L2 loss.
        if self.args.loss_guidance == 'L2':
            y_se_list = []
        else:
            y_mae_list = []
        if self.config.model.target_norm:
            max_target_value = dataset_object.return_max_target_arrtibute()           
        for batch in dataset_loader:
            x_batch = batch[0].to(self.device)
            y_batch = batch[1].to(self.device)
            y_batch_pred_mean = self.compute_guiding_Prediction(x_batch).cpu().detach().numpy()
            y_batch = y_batch.cpu().detach().numpy()
            if self.config.model.target_norm and eval_mode=="inverse_norm":
                # inverse min-max normalization
                y_batch = max_target_value * y_batch
                y_batch_pred_mean = max_target_value * y_batch_pred_mean
            # option to work with L1/L2 loss.
            if self.args.loss_guidance == 'L2':
                y_se = (y_batch_pred_mean - y_batch) ** 2 # get squared error
                if len(y_se_list) == 0:
                    y_se_list = y_se
                else:
                    y_se_list = np.concatenate([y_se_list, y_se], axis=0)
                y_rmse = np.sqrt(np.mean(y_se_list))
                return y_rmse
            else:
                y_ae = np.abs(y_batch_pred_mean - y_batch) # get absolute error
                if len(y_mae_list) == 0:
                    y_mae_list = y_ae
                else:
                    y_mae_list = np.concatenate([y_mae_list, y_ae], axis=0)
                y_mae = np.mean(y_mae_list)                
                return y_mae
        
    def evaluate_guidance_model_on_both_train_and_test_set(self,
                                                           train_set_object,
                                                           train_loader,
                                                           test_set_object,
                                                           test_loader,
                                                           eval_mode=None):
        # Compute y MAE or y RMSE for both train/test set.
        y_train_loss_aux_model = self.evaluate_guidance_model(train_set_object,
                                                             train_loader,
                                                             eval_mode=eval_mode)
        y_test_loss_aux_model = self.evaluate_guidance_model(test_set_object,
                                                            test_loader,
                                                            eval_mode=eval_mode)
        if self.args.loss_guidance == 'L2':
            logging.info(("{} guidance model y RMSE " +
                          "\n\tof the training set and of the test set are " +
                          "\n\t{:.8f} and {:.8f}, respectively.").format(
                              self.config.diffusion.conditioning_signal,
                              y_train_loss_aux_model, y_test_loss_aux_model))
        else:
            logging.info(("{} guidance model y MAE " +
                          "\n\tof the training set and of the test set are " +
                          "\n\t{:.8f} and {:.8f}, respectively.").format(
                              self.config.diffusion.conditioning_signal,
                              y_train_loss_aux_model, y_test_loss_aux_model))                          
                          
    def nonlinear_guidance_model_train_step(self, x_batch, y_batch,
                                            aux_optimizer):
        """
        One optimization step of the non-linear guidance model that
        predicts y_0_hat.
        """
        y_batch_pred = self.cond_pred_model(x_batch)
        aux_cost = self.aux_cost_function(y_batch_pred, y_batch)
        # update non-linear guidance model
        aux_optimizer.zero_grad()
        aux_cost.backward()
        aux_optimizer.step()
        return aux_cost.cpu().item()

    def nonlinear_guidance_model_train_loop_per_epoch(self, train_batch_loader,
                                                      aux_optimizer, epoch):
        if self.config.diffusion.conditioning_signal == "DALSTM":
            for batch in train_batch_loader:
                x_batch = batch[0].to(self.device)
                y_batch = batch[1].to(self.device)
                aux_loss = self.nonlinear_guidance_model_train_step(x_batch,
                                                                    y_batch,
                                                                    aux_optimizer)
        else:
            #TODO: implementation for ProcessTransformer and PGTNet
            print('Currently only DALSTM model is supported.')
            pass
        if epoch % self.config.diffusion.nonlinear_guidance.logging_interval == 0:
            logging.info(f"epoch: {epoch}, non-linear guidance model \
                         pre-training loss: {aux_loss}")

    def obtain_true_and_pred_y_t(self, cur_t, y_seq, y_T_mean, y_0):
        y_t_p_sample = y_seq[self.num_timesteps - cur_t].detach().cpu()
        y_t_true = q_sample(y_0, y_T_mean,
                            self.alphas_bar_sqrt,
                            self.one_minus_alphas_bar_sqrt,
                            torch.tensor([cur_t - 1])).detach().cpu()
        return y_t_p_sample, y_t_true

    #TODO: check why called unnorm, if it not necessary change the name
    def compute_unnorm_y(self, cur_y, testing, max_targ):
        if testing:
            y_mean = cur_y.cpu().reshape(-1,
                                         self.config.testing.n_z_samples
                                         ).mean(1).reshape(-1, 1)
        else:
            y_mean = cur_y.cpu()
        # TODO: change the following if normalization is conducted differently
        if self.config.data.dataset == "ppm":
            if self.config.model.target_norm:
                y_t_unnorm = y_mean * max_targ
            else:
                y_t_unnorm = y_mean
        return y_t_unnorm

    #TODO: check the resultant plots, and if necessary adjust compute_unnorm_y. is it really unnorm?
    def make_subplot_at_timestep_t(self, cur_t, cur_y, y_i, y_0, axs, ax_idx,
                                   prior=False, testing=True, max_targ=None):
        y_0_unnorm = self.compute_unnorm_y(y_0, testing, max_targ)
        y_t_unnorm = self.compute_unnorm_y(cur_y, testing, max_targ)
        # option to work with RMSE or MAE:
        if self.args.loss_guidance == 'L2':
            kl_unnorm = ((y_0_unnorm - y_t_unnorm) ** 2).mean() ** 0.5
            kl_unnorm_str = 'Unnormed RMSE: {:.2f}'.format(kl_unnorm)
        else:
            kl_unnorm = (abs(y_0_unnorm - y_t_unnorm)).mean()
            kl_unnorm_str = 'Unnormed MAE: {:.2f}'.format(kl_unnorm)
        axs[ax_idx].plot(cur_y, '.', label='pred', c='tab:blue')
        axs[ax_idx].plot(y_i, '.', label='true', c='tab:red')   
        if prior:
            axs[ax_idx].set_title('$p({y}_\mathbf{prior})$',
                                  fontsize=23)
            axs[ax_idx].set_title('$p({y}_\mathbf{prior})$\n' + kl_unnorm_str,
                                  fontsize=23)
            axs[ax_idx].legend()
        else:
            axs[ax_idx].set_title('$p(\mathbf{y}_{' + str(cur_t) + '})$',
                                  fontsize=23)
            axs[ax_idx].set_title('$p(\mathbf{y}_{' + str(cur_t) + '})$\n' + \
                                  kl_unnorm_str, fontsize=23)
                
    def compute_Prediction_SE_AE(self, config, dataset_object, y_batch,
                                 generated_y, return_pred_mean=False):
        # generated_y: has a shape of (current_batch_size, n_z_samples, dim_y)
        #TODO: check functionality of trimmed_mean_range
        # if it is always constant remove it from config file
        low, high = config.testing.trimmed_mean_range
        y_true = y_batch.cpu().detach().numpy()
        y_pred_mean = None  # to be used to compute MAE/RMSE
        if low == 50 and high == 50:
            # use median of samples as the mean Prediction
            y_pred_mean = np.median(generated_y, axis=1) 
        else: 
            """
            Compute trimmed mean
            i.e., discarding certain parts of the samples at both ends
            """
            generated_y.sort(axis=1)
            low_idx = int(low / 100 * config.testing.n_z_samples)
            high_idx = int(high / 100 * config.testing.n_z_samples)
            y_pred_mean = (generated_y[:, low_idx:high_idx]).mean(axis=1)
        if config.model.target_norm:
            max_target_value = dataset_object.return_max_target_arrtibute()
            y_true = max_target_value * y_true
            y_pred_mean = max_target_value * y_pred_mean       
        if return_pred_mean:
            return y_pred_mean
        else:
            if self.args.loss_guidance == 'L2':
                y_se = (y_pred_mean - y_true) ** 2
                return y_se
            else:
                y_ae = abs(y_pred_mean - y_true)
                return y_ae
            
    
    def compute_true_coverage_by_gen_QI(config, dataset_object,
                                        all_true_y, all_generated_y,
                                        verbose=True):
        n_bins = config.testing.n_bins
        quantile_list = np.arange(n_bins + 1) * (100 / n_bins)
        # compute generated y quantiles
        y_pred_quantiles = np.percentile(all_generated_y.squeeze(),
                                         q=quantile_list, axis=1)
        y_true = all_true_y.T
        quantile_membership_array = (
            (y_true - y_pred_quantiles) > 0).astype(int)
        y_true_quantile_membership = quantile_membership_array.sum(axis=0)
        y_true_quantile_bin_count = np.array(
            [(y_true_quantile_membership == v).sum() 
             for v in np.arange(n_bins + 2)])
        if verbose:
            y_true_below_0, y_true_above_100 = y_true_quantile_bin_count[0], \
                                               y_true_quantile_bin_count[-1]
            logging.info(("We have {} true y smaller than min of generated y, " + \
                          "and {} greater than max of generated y."
                          ).format(y_true_below_0, y_true_above_100))
        # combine true y falls outside of 0-100 gen y quantile to the first and last interval
        y_true_quantile_bin_count[1] += y_true_quantile_bin_count[0]
        y_true_quantile_bin_count[-2] += y_true_quantile_bin_count[-1]
        y_true_quantile_bin_count_ = y_true_quantile_bin_count[1:-1]
        # compute true y coverage ratio for each gen y quantile interval
        y_true_ratio_by_bin = y_true_quantile_bin_count_ / dataset_object.test_n_samples
        assert np.abs(
            np.sum(y_true_ratio_by_bin) - 1) < 1e-10, "Sum of quantile\
            coverage ratios shall be 1!"
        qice_coverage_ratio = np.absolute(
            np.ones(n_bins) / n_bins - y_true_ratio_by_bin).mean()
        return y_true_ratio_by_bin, qice_coverage_ratio, y_true

    
    def compute_PICP(config, y_true, all_gen_y, return_CI=False):
        """
        Another coverage metric.
        """
        low, high = config.testing.PICP_range
        CI_y_pred = np.percentile(all_gen_y.squeeze(), 
                                  q=[low, high], axis=1)
        # compute percentage of true y in the range of credible interval
        y_in_range = (y_true >= CI_y_pred[0]) & (y_true <= CI_y_pred[1])
        coverage = y_in_range.mean()
        if return_CI:
            return coverage, CI_y_pred, low, high
        else:
            return coverage, low, high

##############################################################################
################### Train function for card_regression class #################
##############################################################################

    def train(self):
        args = self.args
        config = self.config
        tb_logger = self.config.tb_logger
        kfold_handler = self.kfold_handler
        
        # first obtain test set for pre-trained model evaluation
        logging.info("Test set info:")
        test_set_object, test_set = get_dataset(args, config, test_set=True,
                                                kfold_handler=kfold_handler)
        test_loader = data.DataLoader(test_set,
                                      batch_size=config.testing.batch_size,
                                      num_workers=config.data.num_workers,)
        
        # obtain training set
        logging.info("Training set info:")
        dataset_object, dataset = get_dataset(args, config, test_set=False,
                                              kfold_handler=kfold_handler)
        self.dataset_object = dataset_object
        self.mean_target_value = dataset_object.return_mean_target_arrtibute()
        self.max_target_value = dataset_object.return_max_target_arrtibute()
        train_loader = data.DataLoader(dataset,
                                       batch_size=config.training.batch_size,
                                       shuffle=True,
                                       num_workers=config.data.num_workers,) 
        
        # obtain validation set if necessary
        """
        obtain training (as a subset of original one) and validation set for
        guidance model hyperparameter tuning, the validation set is at least
        used for early stopping. Therefore, we check the attribute 
        apply_early_stopping in the followings.
        """
        # if tere is such an attribute, and its value is True:
        if (hasattr(config.diffusion.nonlinear_guidance,
                    "apply_early_stopping") and 
            config.diffusion.nonlinear_guidance.apply_early_stopping):
            logging.info(("\nSplit original training set into training and \
                          validation set " +
                          "for f_phi hyperparameter tuning or using it for \
                              early stopping..."))
            logging.info("Validation set info:")
            val_set_object, val_set = get_dataset(args, config, test_set=True,
                                                  validation=True)
            val_loader = data.DataLoader(val_set,
                                         batch_size=config.testing.batch_size,
                                         num_workers=config.data.num_workers,)
            logging.info("Training subset info:")
            train_subset_object, train_subset = get_dataset(
                args, config, test_set=False, validation=True,
                kfold_handler=kfold_handler)
            train_subset_loader = data.DataLoader(
                train_subset, batch_size=config.training.batch_size,
                shuffle=True, num_workers=config.data.num_workers,)
        model = ConditionalGuidedModel(config)
        model = model.to(self.device)
               
        # evaluate f_phi(x) on both training and test set
        logging.info("\nBefore pre-training:")
        # check whether normalization is used or not!
        if self.config.model.target_norm:
            self.evaluate_guidance_model_on_both_train_and_test_set(
                dataset_object, train_loader, test_set_object, test_loader,
                eval_mode="inverse_norm")
        else:
            self.evaluate_guidance_model_on_both_train_and_test_set(
                dataset_object, train_loader, test_set_object, test_loader,
                eval_mode="no_inverse_norm")
            
        # apply an optimizer for diffusion model (noise estimate network)   
        optimizer = get_optimizer(self.config.optim, model.parameters())
        # apply an auxiliary optimizer for guidance model that predicts y_0_hat
        aux_optimizer = get_optimizer(
            self.config.aux_optim, self.cond_pred_model.parameters())
        
        # TODO: check what the ema does in the model
        if self.config.model.ema:
            ema_helper = EMA(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None
       
        # pre-train the non-linear guidance model
        if config.diffusion.nonlinear_guidance.pre_train:
            n_guidance_model_pretrain_epochs = \
                config.diffusion.nonlinear_guidance.n_pretrain_epochs
            # set the guidance model to train mode
            self.cond_pred_model.train()
            # check for early stopping option
            if (hasattr(config.diffusion.nonlinear_guidance,
                       "apply_early_stopping") and 
                config.diffusion.nonlinear_guidance.apply_early_stopping):
                early_stopper = EarlyStopping(
                    patience=config.diffusion.nonlinear_guidance.patience,
                    delta=config.diffusion.nonlinear_guidance.delta)
                train_val_start_time = time.time()
                for epoch in range(
                        config.diffusion.nonlinear_guidance.n_pretrain_max_epochs):
                    self.nonlinear_guidance_model_train_loop_per_epoch(
                        train_subset_loader, aux_optimizer, epoch)
                    # in training phase we do not apply any normalization
                    y_val_mae_aux_model = self.evaluate_guidance_model(
                        val_set_object, val_loader, eval_mode="no_inverse_norm")
                    val_cost = y_val_mae_aux_model
                    early_stopper(val_cost=val_cost, epoch=epoch)
                    if early_stopper.early_stop:
                        print(('Obtained best performance on validation set \
                               after Epoch {}; ' + 'early stopping at Epoch {}.'
                               ).format(early_stopper.best_epoch, epoch))
                        break
                train_val_end_time = time.time()
                logging.info(("Tuning for number of epochs to train non-linear \
                              guidance model " + "took {:.4f} minutes.").format(
                              (train_val_end_time - train_val_start_time) / 60))
                logging.info('\nAfter tuning for best total epochs, on \
                             training sebset and validation set:')
                # check whether inverse normalization isrequired or not
                if self.config.model.target_norm:
                    self.evaluate_guidance_model_on_both_train_and_test_set(
                        train_subset_object, train_subset_loader,
                        val_set_object, val_loader, eval_mode='inverse_norm')
                else:
                    self.evaluate_guidance_model_on_both_train_and_test_set(
                        train_subset_object, train_subset_loader,
                        val_set_object, val_loader, eval_mode='no_inverse_norm')
                # reset guidance model weights for re-training on original training set
                logging.info('\nReset guidance model weights...')
                for layer in self.cond_pred_model.children():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
                logging.info("\nRe-training the guidance model on original \
                             training set with {} epochs...".format(
                    early_stopper.best_epoch))
                n_guidance_model_pretrain_epochs = early_stopper.best_epoch
                aux_optimizer = get_optimizer(self.config.aux_optim,
                                              self.cond_pred_model.parameters())
            pretrain_start_time = time.time()
            for epoch in range(n_guidance_model_pretrain_epochs):
                self.nonlinear_guidance_model_train_loop_per_epoch(
                    train_loader, aux_optimizer, epoch)
            pretrain_end_time = time.time()
            logging.info('Pre-training of non-linear guidance model \
                         took {:.4f} minutes.'.format(
                (pretrain_end_time - pretrain_start_time) / 60))
            logging.info('\nAfter pre-training:')
            # check whether inverse normalization isrequired or not
            if self.config.model.target_norm:
                self.evaluate_guidance_model_on_both_train_and_test_set(
                    dataset_object, train_loader, test_set_object, test_loader,
                    eval_mode='inverse_norm')
            else:
                self.evaluate_guidance_model_on_both_train_and_test_set(
                    dataset_object, train_loader, test_set_object, test_loader,
                    eval_mode='no_inverse_norm')                
            # save auxiliary model
            aux_states = [
                self.cond_pred_model.state_dict(),
                aux_optimizer.state_dict(),
            ]
            torch.save(aux_states, os.path.join(self.args.log_path,
                                                'aux_ckpt.pth'))
            
        # train diffusion model
        if not self.args.train_guidance_only:
            start_epoch, step = 0, 0
            if self.args.resume_training:
                states = torch.load(os.path.join(self.args.log_path,
                                                 'ckpt.pth'),
                                    map_location=self.device)
                model.load_state_dict(states[0])

                states[1]['param_groups'][0]['eps'] = self.config.optim.eps
                optimizer.load_state_dict(states[1])
                start_epoch = states[2]
                step = states[3]
                if self.config.model.ema:
                    ema_helper.load_state_dict(states[4])
                # load auxiliary model
                aux_states = torch.load(os.path.join(self.args.log_path,
                                                     'aux_ckpt.pth'),
                                        map_location=self.device)
                self.cond_pred_model.load_state_dict(aux_states[0])
                aux_optimizer.load_state_dict(aux_states[1])
            if config.diffusion.noise_prior:
                if config.diffusion.noise_prior_approach == 'zero':
                    """
                    Apply 0 instead of f_phi(x) as prior mean.
                    However, we used min-max normalization to keep the original
                    distrbution of remining time which often does not follow a
                    normal distribution. (we assume that the minimum value is
                                          zero and thus only use max value)
                    In contrst, in the original
                    implementation of CARD standardization is used, and thus,
                    zero reflect the mean of the target attribute.                     
                    """
                    logging.info('Prior distribution at timestep T \
                                 has a mean of 0.')
                # apply mean instead of f_phi(x) as prior mean
                elif config.diffusion.noise_prior_approach == 'mean':
                    logging.info("Prior distribution at timestep T has a mean\
                                 equivalent to mean of remaining time in\
                                     training set.")
                # apply median instead of f_phi(x) as prior mean
                elif config.diffusion.noise_prior_approach == 'median': 
                    logging.info("Prior distribution at timestep T has a mean\
                                 equivalent to median of remaining time in\
                                     training set.")
                    
            for epoch in range(start_epoch, self.config.training.n_epochs):
                data_start = time.time()
                data_time = 0               
                for i, xy_0 in enumerate(train_loader):
                    if config.diffusion.conditioning_signal == "DALSTM":
                        n = xy_0[0].size(0)
                    data_time += time.time() - data_start
                    model.train()
                    step += 1

                    # antithetic sampling -- low (inclusive) and high (exclusive)
                    t = torch.randint(
                        low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                    ).to(self.device)
                    t = torch.cat([t, self.num_timesteps - 1 - t], dim=0)[:n]
                    
                    #TODO: add necessary code for PGTNET, PT, ...
                    if config.diffusion.conditioning_signal == "NN":
                        xy_0 = xy_0.to(self.device)
                        x_batch = xy_0[:, :-config.model.y_dim]
                        y_batch = xy_0[:, -config.model.y_dim:]  # shape: (batch_size, 1)
                    if config.diffusion.conditioning_signal == "DALSTM":
                        x_batch = xy_0[0].to(self.device)
                        y_batch = xy_0[1].to(self.device)                        
                        
                    y_0_hat_batch = self.compute_guiding_Prediction(x_batch)
                    y_T_mean = y_0_hat_batch
                    
                    if config.diffusion.noise_prior:
                        if config.diffusion.noise_prior_approach == 'zero':
                            # apply 0 instead of f_phi(x) as prior mean
                            y_T_mean = torch.zeros(y_batch.shape[0]).to(
                                y_batch.device)                            
                        elif config.diffusion.noise_prior_approach == 'mean':
                            mean_target_value = \
                                dataset_object.return_mean_target_arrtibute()
                            max_target_value = \
                                dataset_object.return_max_target_arrtibute()
                            normalized_mean_target_value = (
                                mean_target_value/max_target_value)
                            y_T_mean = torch.full(
                                (y_batch.shape[0]),
                                normalized_mean_target_value).to(y_batch.device)                            
                        elif config.diffusion.noise_prior_approach == 'median': 
                            # apply median instead of f_phi(x) as prior mean
                            median_target_value =  \
                                dataset_object.return_median_target_arrtibute()
                            max_target_value = \
                                dataset_object.return_max_target_arrtibute()
                            normalized_median_target_value = (
                                median_target_value/max_target_value)
                            y_T_mean = torch.full(
                                (y_batch.shape[0]),
                                normalized_median_target_value).to(y_batch.device)                     

                    e = torch.randn_like(y_batch).to(y_batch.device)
                    #print(y_batch.size())
                    #print(y_T_mean.size())
                    #print('now call q_sample')
                    y_t_batch = q_sample(y_batch, y_T_mean,
                                         self.alphas_bar_sqrt,
                                         self.one_minus_alphas_bar_sqrt, t,
                                         noise=e)
                    #print(x_batch.size())
                    #print(y_t_batch.size())
                    #print(y_0_hat_batch.size())
                    #print('now exit the program!')
                    #sys.exit()
                    output = model(x_batch, y_t_batch, y_0_hat_batch, t)
                    # noise estimation loss
                    # use the same noise sample e during training to compute loss
                    #TODO: check the meaning of the following to see whether related to L1 OR L2 loss or not
                    loss = (e - output).square().mean()  
                    #if config.diffusion.conditioning_signal == "DALSTM":
                        #loss = (e - output).abs().mean() # use the same noise sample e during training to compute loss

                    tb_logger.add_scalar("loss", loss, global_step=step)

                    if (step % self.config.training.logging_freq == 0 or
                        step == 1):
                        logging.info((f"epoch: {epoch}, step: {step}, \
                                      Noise Estimation loss: {loss.item()}, "
                                      + f"data time: {data_time / (i + 1)}"))

                    # optimize diffusion model that predicts eps_theta
                    optimizer.zero_grad()
                    loss.backward()
                    try:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.optim.grad_clip)
                    except Exception:
                        pass
                    optimizer.step()
                    if self.config.model.ema:
                        ema_helper.update(model)
                    # TODO: check the possibility of joint training!
                    # optimize non-linear guidance model
                    if config.diffusion.nonlinear_guidance.joint_train:
                        self.cond_pred_model.train()
                        aux_loss = \
                            self.nonlinear_guidance_model_train_step(
                                x_batch, y_batch, aux_optimizer)
                        if (step % self.config.training.logging_freq == 0 or
                            step == 1):
                            logging.info(f"meanwhile, non-linear guidance model\
                                         joint-training loss: {aux_loss}")

                    # save diffusion model
                    if (step % self.config.training.snapshot_freq == 0 or
                        step == 1):
                        states = [
                            model.state_dict(),
                            optimizer.state_dict(),
                            epoch,
                            step,
                        ]
                        if self.config.model.ema:
                            states.append(ema_helper.state_dict())
                        if step > 1:  # skip saving the initial ckpt
                            torch.save(
                                states,
                                os.path.join(self.args.log_path,
                                             'ckpt_{}.pth'.format(step)),
                            )
                        torch.save(states, os.path.join(self.args.log_path,
                                                        'ckpt.pth'))

                        # save auxiliary model
                        if hasattr(config.diffusion.nonlinear_guidance,
                                   'joint_train'):
                            if config.diffusion.nonlinear_guidance.joint_train:
                                # TODO: add other implementations in assert!
                                # in the original implementation was NN!
                                # So wee need it separately for PGTNet, PT, ....
                                assert config.diffusion.conditioning_signal ==\
                                    'DALSTM'
                                aux_states = [
                                    self.cond_pred_model.state_dict(),
                                    aux_optimizer.state_dict(),
                                ]
                                if step > 1:  # skip saving the initial ckpt
                                    torch.save(
                                        aux_states,
                                        os.path.join(self.args.log_path,
                                                     'aux_ckpt_{}.pth'.format(step)),
                                    )
                                torch.save(
                                    aux_states, os.path.join(
                                        self.args.log_path, "aux_ckpt.pth"))
                    if (step % self.config.training.validation_freq == 0 or
                        step == 1):
                        # plot Prediction and ground truth
                        with torch.no_grad():
                            if config.diffusion.conditioning_signal == "DALSTM":
                                #TODO: check if possible move squeeze operations to the model!
                                # Need for squeeze!
                                y_p_seq = p_sample_loop(
                                    model, x_batch, y_batch, y_T_mean,
                                    self.num_timesteps, self.alphas,
                                    self.one_minus_alphas_bar_sqrt,
                                    squeeze=True)
                            else:
                                y_p_seq = p_sample_loop(
                                    model, x_batch, y_batch, y_T_mean,
                                    self.num_timesteps, self.alphas,
                                    self.one_minus_alphas_bar_sqrt)                                    
                            fig, axs = plt.subplots(
                                1, (self.num_figs + 1), figsize=(
                                    (self.num_figs + 1) * 8.5, 8.5),
                                clear=True)
                            # plot at timestep 1
                            cur_t = 1
                            cur_y, y_i = self.obtain_true_and_pred_y_t(
                                cur_t, y_p_seq, y_T_mean, y_batch)
                            self.make_subplot_at_timestep_t(
                                cur_t, cur_y, y_i, y_batch, axs, 0,
                                testing=False,
                                max_targ=self.max_target_value) 
                            # plot at vis_step interval
                            for j in range(1, self.num_figs):
                                cur_t = j * self.vis_step
                                cur_y, y_i = self.obtain_true_and_pred_y_t(
                                    cur_t, y_p_seq, y_T_mean, y_batch)
                                self.make_subplot_at_timestep_t(
                                    cur_t, cur_y, y_i, y_batch, axs, j,
                                    testing=False,
                                    max_targ=self.max_target_value)
                            # plot at timestep T
                            cur_t = self.num_timesteps
                            cur_y, y_i = self.obtain_true_and_pred_y_t(
                                cur_t, y_p_seq, y_T_mean, y_batch)
                            self.make_subplot_at_timestep_t(
                                cur_t, cur_y, y_i, y_batch, axs, self.num_figs,
                                prior=True, testing=False,
                                max_targ=self.max_target_value)
                            ax_list = [axs[j] 
                                       for j in range(self.num_figs + 1)]
                            ax_list[0].get_shared_x_axes().join(
                                ax_list[0], *ax_list)
                            ax_list[0].get_shared_y_axes().join(
                                ax_list[0], *ax_list)
                            tb_logger.add_figure('samples', fig, step)
                            fig.savefig(
                                    os.path.join(args.im_path,
                                                 'samples_T{}_{}.png'.format(
                                                     self.num_timesteps, step)))
                        plt.close('all')
                    data_start = time.time()
            plt.close('all')
            # save the model after training is finished
            states = [
                model.state_dict(),
                optimizer.state_dict(),
                epoch,
                step,
            ]
            if self.config.model.ema:
                states.append(ema_helper.state_dict())
            torch.save(states, os.path.join(self.args.log_path, 'ckpt.pth'))
            # save auxiliary model after training is finished
            aux_states = [
                    self.cond_pred_model.state_dict(),
                    aux_optimizer.state_dict(),
                    ]
            torch.save(aux_states, os.path.join(self.args.log_path, 'aux_ckpt.pth'))
            # report training set MAE/RMSE if applied joint training
            if config.diffusion.nonlinear_guidance.joint_train:
                # check for the option of L1/L2 loss
                if self.args.loss_guidance == 'L2':
                    y_rmse_aux_model = self.evaluate_guidance_model(
                        dataset_object, train_loader, 
                        eval_mode="no_inverse_norm")
                    logging.info(
                        "After joint-training, non-linear guidance model \
                            unnormalized y RMSE is {:.8f}.".format(
                            y_rmse_aux_model))
                else:
                    y_mae_aux_model = self.evaluate_guidance_model(
                        dataset_object, train_loader, 
                        eval_mode="no_inverse_norm") 
                    logging.info("After joint-training, non-linear guidance \
                                 model unnormalized y MAE is {:.8f}.".format(
                                 y_mae_aux_model))   
                                     

##############################################################################
################### test function for card_regression class #################
##############################################################################                        

    def test(self):
        """
        Evaluate model on regression tasks on test set.
        """
        kfold_handler = self.kfold_handler
        
        ######################################################################
        #### local functions within the class function scope #################
        ######################################################################
        

        def store_gen_y_at_step_t(config, current_batch_size, idx, y_tile_seq):
            """
            Store generated y from a mini-batch to 
            the array of corresponding time step.
            """
            current_t = self.num_timesteps - idx
            #TODO: check wheteher it is the same for other models: PGTNet, PT,...
            gen_y = y_tile_seq[idx].reshape(current_batch_size,
                                            config.testing.n_z_samples,
                                            config.model.y_dim).cpu().numpy()
            """
            directly modify the dict value by concat np.array instead of append
            np.array gen_y to list reduces a huge amount of memory consumption
            """
            if len(gen_y_by_batch_list[current_t]) == 0:
                gen_y_by_batch_list[current_t] = gen_y
            else:
                gen_y_by_batch_list[current_t] = np.concatenate(
                    [gen_y_by_batch_list[current_t], gen_y], axis=0)
            return gen_y

        def store_y_se_at_step_t(config, idx, dataset_object, y_batch, gen_y):
            
            current_t = self.num_timesteps - idx
            if self.args.loss_guidance == 'L2':            
                # compute sqaured error in each batch
                y_se = self.compute_Prediction_SE_AE(
                    config=config, dataset_object=dataset_object,
                    y_batch=y_batch, generated_y=gen_y)
                if len(y_se_by_batch_list[current_t]) == 0:
                    y_se_by_batch_list[current_t] = y_se
                # TODO: resolve the problem without padding if it is possible!
                else:
                    try:
                        y_se_by_batch_list[current_t] = np.concatenate(
                            [y_se_by_batch_list[current_t], y_se], axis=0)
                    except:
                        # handle exception for the last minibatch by padding
                        pad_value = int(
                            config.testing.batch_size - y_se.shape[1])
                        y_se_resized = np.pad(y_se, ((0, 0), (0, pad_value)))
                        y_se_by_batch_list[current_t] = np.concatenate(
                            [y_se_by_batch_list[current_t], y_se_resized], axis=0)
                        
            else:
                # compute absolute error in each batch
                y_ae = self.compute_Prediction_SE_AE(
                    config=config, dataset_object=dataset_object,
                    y_batch=y_batch, generated_y=gen_y)
                if len(y_ae_by_batch_list[current_t]) == 0:
                    y_ae_by_batch_list[current_t] = y_ae
                # TODO: resolve the problem without padding if it is possible!
                else:
                    try:
                        y_ae_by_batch_list[current_t] = np.concatenate(
                            [y_ae_by_batch_list[current_t], y_ae], axis=0) 
                    except:
                        # handle exception for the last minibatch by padding
                        pad_value = int(
                            config.testing.batch_size - y_ae.shape[1])
                        y_ae_resized = np.pad(y_ae, ((0, 0), (0, pad_value)))                     
                        #print(np.shape(y_ae_by_batch_list[current_t]))
                        #print(np.shape(y_se))
                        y_ae_by_batch_list[current_t] = np.concatenate(
                            [y_ae_by_batch_list[current_t], y_ae_resized], axis=0)
        
        # TODO: the following condition does not work for ppm datasets: should be adjusted
        def set_NLL_global_precision(test_var=True, max_targ=None):
            if test_var:
                # compute test set sample variance
                if self.config.model.target_norm:
                    #TODO: it there is some error: try removinf astype
                    y_test_unnorm = (max_targ * dataset_object.y_test).astype(
                        np.float32)
                else:
                    y_test_unnorm = dataset_object.y_test.astype(np.float32)
                y_test_unnorm = y_test_unnorm if type(y_test_unnorm) is torch.Tensor \
                    else torch.from_numpy(y_test_unnorm)
                self.tau = 1 / (y_test_unnorm.var(unbiased=True).item())
            else:
                self.tau = 1

        def compute_batch_NLL(config, dataset_object, y_batch, generated_y):
            """
            generated_y: has a shape of:
                (current_batch_size, n_z_samples, dim_y)
            NLL computation implementation from MC dropout repo
            https://github.com/yaringal/DropoutUncertaintyExps/blob/master/net/net.py,
            directly from MC Dropout paper Eq. (8).
            """
            #print(generated_y.shape)
            y_true = y_batch.cpu().detach().numpy()
            #batch_size = generated_y.shape[0]  
            if config.model.target_norm:
                max_target_value = dataset_object.return_max_target_arrtibute()
                # unnormalize true y
                y_true = (max_target_value * y_true).astype(np.float32)
                # unnormalize generated y
                generated_y = (max_target_value * generated_y)    
                  
            generated_y = generated_y.swapaxes(0, 1)
            # obtain precision value and compute test batch NLL
            if self.tau is not None:
                tau = self.tau
            else:
                gen_y_var = torch.from_numpy(
                    generated_y).var(dim=0, unbiased=True).numpy()
                tau = 1 / gen_y_var
            nll = -(logsumexp(-0.5 * tau * (y_true[None] - generated_y) ** 2., 0)
                    - np.log(config.testing.n_z_samples)
                    - 0.5 * np.log(2 * np.pi) + 0.5 * np.log(tau))
            return nll

        def store_nll_at_step_t(config, idx, dataset_object, y_batch, gen_y):
            current_t = self.num_timesteps - idx
            # compute negative log-likelihood in each batch
            nll = compute_batch_NLL(config=config,
                                    dataset_object=dataset_object,
                                    y_batch=y_batch, generated_y=gen_y)
            if len(nll_by_batch_list[current_t]) == 0:
                nll_by_batch_list[current_t] = nll
            else:
                try:
                    nll_by_batch_list[current_t] = np.concatenate(
                        [nll_by_batch_list[current_t], nll], axis=0)
                except:
                    # TODO: check the root cause and if possible handle without padding
                    # handle exception for the last minibatch by padding
                    pad_value = int(config.testing.batch_size - nll.shape[1])
                    nll_resized = np.pad(nll, ((0, 0), (0, pad_value)))    
                    nll_by_batch_list[current_t] = np.concatenate(
                        [nll_by_batch_list[current_t], nll_resized], axis=0)

        ######################################################################
        ######################################################################

        args = self.args
        config = self.config
        split = args.split
        log_path = os.path.join(self.args.log_path)
        dataset_object, dataset = get_dataset(args, config, test_set=True,
                                              kfold_handler=kfold_handler)
        test_loader = data.DataLoader(
            dataset,
            batch_size=config.testing.batch_size,
            num_workers=config.data.num_workers,
        )
        self.dataset_object = dataset_object
        self.mean_target_value = dataset_object.return_mean_target_arrtibute()
        self.max_target_value = dataset_object.return_max_target_arrtibute()
        # set global prevision value for NLL computation if needed
        if args.nll_global_var:
            set_NLL_global_precision(test_var=args.nll_test_var,
                                     max_targ=self.max_target_value)

        model = ConditionalGuidedModel(self.config)
        if getattr(self.config.testing, 'ckpt_id', None) is None:
            states = torch.load(os.path.join(log_path, 'ckpt.pth'),
                                map_location=self.device)
            ckpt_id = 'last'
        else:
            states = torch.load(
                os.path.join(log_path, f'ckpt_{self.config.testing.ckpt_id}.pth'),
                map_location=self.device)
            ckpt_id = self.config.testing.ckpt_id
        logging.info(f'Loading from: {log_path}/ckpt_{ckpt_id}.pth')
        model = model.to(self.device)
        model.load_state_dict(states[0], strict=True)

        if self.config.model.ema:
            ema_helper = EMA(mu=self.config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(model)
        else:
            ema_helper = None

        model.eval()

        # load auxiliary model
        aux_states = torch.load(os.path.join(log_path, 'aux_ckpt.pth'),
                                    map_location=self.device)
        self.cond_pred_model.load_state_dict(aux_states[0], strict=True)
        self.cond_pred_model.eval()
        
        # report test set MAE/RMSE with guidance model
        if self.args.loss_guidance == 'L2':
            y_rmse_aux_model = self.evaluate_guidance_model(
                dataset_object, test_loader, eval_mode='inverse_norm')
            logging.info('Test set unnormalized y RMSE on trained {}  \
                         guidance model is {:.8f}.'.format(
                         config.diffusion.conditioning_signal, y_rmse_aux_model)) 
        else:
            y_mae_aux_model = self.evaluate_guidance_model(
                dataset_object, test_loader, eval_mode="inverse_norm")
            logging.info("Test set unnormalized y MAE on trained {} \
                         guidance model is {:.8f}.".format(
                         config.diffusion.conditioning_signal, y_mae_aux_model))        

        ######################################################################
        # sanity check
        ######################################################################
        logging.info("Sanity check of the checkpoint")
        dataset_check = dataset_object.return_dataset(split="train")
        # use the first 50 samples for sanity check
        dataset_check = dataset_check[:50] 
        #TODO: check whether the followings also work for PGTNet, PT, ...
        x_check = dataset_check[0]
        y_check = dataset_check[1]
        y_check = y_check.to(self.device)
        y_0_hat_check = self.compute_guiding_Prediction(x_check.to(self.device))
        y_T_mean_check = y_0_hat_check        
        
        if config.diffusion.noise_prior:
            if config.diffusion.noise_prior_approach == 'zero':
                # apply 0 instead of f_phi(x) as prior mean
                y_T_mean_check = torch.zeros(y_check.shape[0]).to(self.device)
            elif config.diffusion.noise_prior_approach == 'mean':
                # apply mean instead of f_phi(x) as prior mean
                mean_target_value = dataset_object.return_mean_target_arrtibute()
                max_target_value = dataset_object.return_max_target_arrtibute()
                normalized_mean_target_value = (
                    mean_target_value/max_target_value)
                y_T_mean_check = torch.full(
                    (y_check.shape[0]), normalized_mean_target_value).to(
                        self.device)            
            elif config.diffusion.noise_prior_approach == "median":
                # apply median instead of f_phi(x) as prior mean
                median_target_value = dataset_object.return_median_target_arrtibute()                
                max_target_value = dataset_object.return_max_target_arrtibute()
                normalized_median_target_value = (
                    median_target_value/max_target_value)                     
                y_T_mean_check = torch.full(
                    (y_check.shape[0]), normalized_median_target_value).to(
                        self.device)
                        
        with torch.no_grad():
            # TODO: check whether the squeeze=True also works for PGTNET, PT, ..
            y_p_seq = p_sample_loop(
                model, x_check.to(self.device), y_0_hat_check, y_T_mean_check,
                self.num_timesteps, self.alphas, self.one_minus_alphas_bar_sqrt,
                squeeze=True)                       
            fig, axs = plt.subplots(1, (self.num_figs + 1),
                                    figsize=((self.num_figs + 1) * 8.5, 8.5),
                                    clear=True)
            # plot at timestep 1
            cur_t = 1
            cur_y, y_i = self.obtain_true_and_pred_y_t(
                cur_t, y_p_seq, y_T_mean_check, y_check)
            self.make_subplot_at_timestep_t(
                cur_t, cur_y, y_i, y_check, axs, 0, testing=False,
                max_targ=self.max_target_value)
            # plot at vis_step interval
            for i in range(1, self.num_figs):
                cur_t = i * self.vis_step
                cur_y, y_i = self.obtain_true_and_pred_y_t(
                    cur_t, y_p_seq, y_T_mean_check, y_check)
                self.make_subplot_at_timestep_t(
                    cur_t, cur_y, y_i, y_check, axs, i, testing=False,
                    max_targ=self.max_target_value)
            # plot at timestep T
            cur_t = self.num_timesteps
            cur_y, y_i = self.obtain_true_and_pred_y_t(
                cur_t, y_p_seq, y_T_mean_check, y_check)
            self.make_subplot_at_timestep_t(
                cur_t, cur_y, y_i, y_check, axs, self.num_figs, 
                prior=True, testing=False, max_targ=self.max_target_value)
            fig.savefig(os.path.join(args.im_path, 'sanity_check.pdf'))
            plt.close('all')
       ####################################################################### 

        if config.testing.compute_metric_all_steps:
            logging.info("\nWe compute MAE, QICE, PICP and NLL for all steps.\n")
        else:
            mean_idx = self.num_timesteps - config.testing.mean_t
            coverage_idx = self.num_timesteps - config.testing.coverage_t
            nll_idx = self.num_timesteps - config.testing.nll_t
            logging.info(("\nWe pick t={} to compute y mean metric MAE, " + 
                              "and t={} to compute true y coverage metric\
                                  QICE and PICP.\n").format(
                                  config.testing.mean_t,
                                  config.testing.coverage_t))
        
        # define an empty dictionary to collect instance level information
        instance_results = {'GroundTruth': [],
                            'Deterministic_Prediction':[],
                            'Prediction': [],
                            'Total_Uncertainty': [], 
                            'Prefix_length':[],
                            'Absolute_error': []}   
                       
        with torch.no_grad():
            #true_x_by_batch_list = []
            true_x_tile_by_batch_list = []
            true_y_by_batch_list = []
            gen_y_by_batch_list = [[] for _ in range(self.num_timesteps + 1)]
            y_ae_by_batch_list = [[] for _ in range(self.num_timesteps + 1)]
            y_se_by_batch_list = [[] for _ in range(self.num_timesteps + 1)]
            nll_by_batch_list = [[] for _ in range(self.num_timesteps + 1)]
            
            # get mean of remaining time in training set
            mean_target_value = dataset_object.return_mean_target_arrtibute()
            # get max of remaining time in training set
            max_target_value = dataset_object.return_max_target_arrtibute()
            # compute normalized mean of remaining time in training set
            normalized_mean_target_value = (mean_target_value/max_target_value)
            # get median of remaining time in training set 
            median_target_value = dataset_object.return_median_target_arrtibute() 
            normalized_median_target_value = median_target_value/max_target_value 
            
            # get all prefix lengths in test set.
            test_length_list = dataset_object.return_prefix_lengths() 
            # indicator to access relevant prefix lengths in each batch
            index_indicator = 0 

            for step, xy_batch in enumerate(test_loader):
                # TODO: chceck the following works for PGTNet, PT, ...
                current_batch_size = xy_batch[0].shape[0]
                x_batch = xy_batch[0].to(self.device)
                y_batch = xy_batch[1].to(self.device)
                """
                compute y_0_hat as the initial Prediction to 
                guide the reverse diffusion process
                """
                y_0_hat_batch = self.compute_guiding_Prediction(x_batch)
                true_y_by_batch_list.append(y_batch.cpu().numpy()) 
                """
                contain y samples through reverse diffusion 
                -- some pytorch version might not have torch.tile 
                """
                # TODO: check whether the followings work for PGTNet, PT, ...
                y_0_tile = (y_batch.repeat(
                    config.testing.n_z_samples, 1, 1).transpose(0, 1)).to(
                        self.device).flatten(0, 1).view(-1)
                y_0_hat_tile = (y_0_hat_batch.repeat(
                    config.testing.n_z_samples, 1, 1).transpose(0, 1)).to(
                        self.device).flatten(0, 1).view(-1)
                y_T_mean_tile = y_0_hat_tile
                
                """
                If we want to use a noie prior instead of the guidance of 
                point estimator (i.e., pre-trained deterministic model)
                """
                if config.diffusion.noise_prior:  
                    if config.diffusion.noise_prior_approach == 'zero':
                        # apply 0 instead of f_phi(x) as prior mean
                        y_T_mean_tile = torch.zeros(y_0_hat_tile.shape).to(
                            self.device)
                    elif config.diffusion.noise_prior_approach == "mean": 
                        # apply mean instead of f_phi(x) as prior mean
                        y_T_mean_tile = torch.full(
                            (y_0_hat_tile.shape[0]),
                            normalized_mean_target_value).to(self.device)
                    elif config.diffusion.noise_prior_approach == "median":
                        # apply median instead of f_phi(x) as prior mean                        
                        y_T_mean_tile = torch.full_like(
                            y_0_hat_tile, normalized_median_target_value).to(
                                self.device)                            
                x_tile = (x_batch.repeat(
                    config.testing.n_z_samples, 1, 1).transpose(0, 1)).to(
                        self.device).flatten(0, 1).view(
                            -1, config.model.max_len, config.model.x_dim)
                              
                n_samples_gen_y_for_plot = 2 
                if config.testing.plot_gen:
                    # TODO: check when this part is called and revise it like 
                    # the previous lines accordingly!
                    x_repeated = (x_batch.repeat(
                        n_samples_gen_y_for_plot, 1, 1).transpose(0, 1)).to(
                        self.device).flatten(0, 1)
                    true_x_tile_by_batch_list.append(x_repeated.cpu().numpy())
                
                # generate samples from all time steps for current mini-batch
                minibatch_sample_start = time.time()
                # TODO: in the original implementation there was no squeeze=True
                # but we needed it for DALSTM model. Check whether it is also
                # applicable to PGTNet, PT, etc.
                y_tile_seq = p_sample_loop(
                    model, x_tile, y_0_hat_tile,y_T_mean_tile,
                    self.num_timesteps, self.alphas,
                    self.one_minus_alphas_bar_sqrt, squeeze=True)
                minibatch_sample_end = time.time()
                logging.info(
                    "Minibatch {} sampling took {:.4f} seconds.".format(
                        step, (minibatch_sample_end - minibatch_sample_start)))
                
                """
                obtain generated y and compute squared error at:
                    a) all time steps or 
                    b) a particular time step
                """ 
                # a) all time steps or
                if config.testing.compute_metric_all_steps:
                    for idx in range(self.num_timesteps + 1):
                        gen_y = store_gen_y_at_step_t(
                            config=config,
                            current_batch_size=current_batch_size,
                            idx=idx, y_tile_seq=y_tile_seq)
                        store_y_se_at_step_t(config=config, idx=idx,
                                             dataset_object=dataset_object,
                                             y_batch=y_batch, gen_y=gen_y)
                        store_nll_at_step_t(config=config, idx=idx,
                                            dataset_object=dataset_object,
                                            y_batch=y_batch, gen_y=gen_y)                    
                    
                    """
                    For the last gen_y (i.e. t=0):
                        save the result for instance-level analysis.
                    First action:
                        check normalization config, and adjust the followings:
                            1) target attribute
                            2) deterministic point estimates
                            3) generated probabilistic estimations, and thus:
                                both mean and standard deviation in 
                                probabilistic estimation
                    """                        
                    if self.config.model.target_norm:
                        y_batch_unnormalized = y_batch * max_target_value
                        y_0_hat_batch_unnormalized = (y_0_hat_batch * 
                                                      max_target_value)
                        gen_y_unnormalized = gen_y * max_target_value
                    else:
                        y_batch_unnormalized = y_batch
                        y_0_hat_batch_unnormalized = y_0_hat_batch
                        gen_y_unnormalized = gen_y
                    # get ground truth values for remaining time                    
                    groundtruth_values = y_batch_unnormalized.cpu(
                        ).detach().numpy()
                    instance_results['GroundTruth'].extend(groundtruth_values)
                    # get deterministic predictions
                    deterministic_Predictions = y_0_hat_batch_unnormalized.cpu(
                        ).detach().numpy()
                    instance_results['Deterministic_Prediction'].extend(
                        deterministic_Predictions)
                    # get probabilistic prediction:
                    # 1) prediction mean
                    mean_Predictions = np.mean(gen_y_unnormalized, axis=1)
                    mean_Predictions = np.squeeze(mean_Predictions)                  
                    instance_results['Prediction'].extend(mean_Predictions)
                    # 2) prediction std = predicted uncertainty
                    std_Predictions = np.std(gen_y_unnormalized, axis=1)
                    std_Predictions = np.squeeze(std_Predictions)
                    instance_results['Total_Uncertainty'].extend(
                        std_Predictions) 
                    # get relevant prefix length                    
                    relevant_prefix_length = test_length_list[
                        index_indicator:int(current_batch_size)+index_indicator]                    
                    instance_results['Prefix_length'].extend(
                        relevant_prefix_length)                                   
                    index_indicator += int(current_batch_size)
                    # get absolute error
                    absolute_error_values = np.abs(groundtruth_values - 
                                                   mean_Predictions)
                    instance_results['Absolute_error'].extend(
                        absolute_error_values)      
                # b) a particular time step
                else:
                    """
                    store generated y at certain step for MAE/RMSE and 
                    for QICE computation
                    """
                    gen_y = store_gen_y_at_step_t(
                        config=config,
                        current_batch_size=current_batch_size, idx=mean_idx,
                        y_tile_seq=y_tile_seq)
                    store_y_se_at_step_t(config=config, idx=mean_idx,
                                         dataset_object=dataset_object,
                                         y_batch=y_batch, gen_y=gen_y)
                    if coverage_idx != mean_idx:
                        _ = store_gen_y_at_step_t(
                            config=config,
                            current_batch_size=current_batch_size,
                            idx=coverage_idx, y_tile_seq=y_tile_seq)
                    if nll_idx != mean_idx and nll_idx != coverage_idx:
                        _ = store_gen_y_at_step_t(
                            config=config,
                            current_batch_size=current_batch_size, 
                            idx=nll_idx, y_tile_seq=y_tile_seq)
                    store_nll_at_step_t(config=config, idx=nll_idx,
                                        dataset_object=dataset_object,
                                        y_batch=y_batch, gen_y=gen_y)

                # make plot at particular mini-batches
                if step % config.testing.plot_freq == 0: 
                    # plot for every plot_freq-th mini-batch
                    fig, axs = plt.subplots(
                        1, self.num_figs + 1, figsize=((self.num_figs + 1) * 
                                                       8.5, 8.5), clear=True)
                    # plot at timestep 1
                    cur_t = 1
                    cur_y, y_i = self.obtain_true_and_pred_y_t(
                        cur_t, y_tile_seq, y_T_mean_tile, y_0_tile)
                    self.make_subplot_at_timestep_t(
                        cur_t, cur_y, y_i, y_0_tile, axs, 0,
                        max_targ=self.max_target_value)                
                    # plot at vis_step interval
                    for i in range(1, self.num_figs):
                        cur_t = i * self.vis_step
                        cur_y, y_i = self.obtain_true_and_pred_y_t(
                            cur_t, y_tile_seq, y_T_mean_tile, y_0_tile)
                        self.make_subplot_at_timestep_t(
                            cur_t, cur_y, y_i, y_0_tile, axs, i,
                            max_targ=self.max_target_value)
                    # plot at timestep T
                    cur_t = self.num_timesteps
                    cur_y, y_i = self.obtain_true_and_pred_y_t(
                        cur_t, y_tile_seq, y_T_mean_tile, y_0_tile)
                    self.make_subplot_at_timestep_t(
                        cur_t, cur_y, y_i, y_0_tile, axs, self.num_figs,
                        prior=True, max_targ=self.max_target_value)
                    fig.savefig(os.path.join(
                        args.im_path,
                        'samples_T{}_{}.png'.format(self.num_timesteps, step)))
                    plt.close('all')
            
            # save an instance-level dataframe for further analysis
            instance_level_df = pd.DataFrame(instance_results)
            instance_level_df[[
                'Prediction','Total_Uncertainty','GroundTruth',
                'Deterministic_Prediction', 'Absolute_error']
                ] = instance_level_df[['Prediction', 'Total_Uncertainty',
                                       'GroundTruth', 'Deterministic_Prediction',
                                       'Absolute_error' ]].astype(float)
            instance_level_df[['Prefix_length']] = instance_level_df[['Prefix_length']].astype(int)
            # TODO: Save the final result in results folder with specified 
            # UQ method to be used alongside other methods!
            instance_level_df.to_csv(
                os.path.join(self.args.log_path,
                             "instance_level_Predictions.csv"), index=False)   
            
        
        ################## compute metrics on test set ##################
        all_true_y = np.concatenate(true_y_by_batch_list, axis=0)
        if config.testing.plot_gen:
            all_true_x_tile = np.concatenate(true_x_tile_by_batch_list, axis=0)
        if self.args.loss_guidance == 'L2':
            y_rmse_all_steps_list = []
        else:
            y_mae_all_steps_list = []          
        y_qice_all_steps_list = []
        y_picp_all_steps_list = []
        y_nll_all_steps_list = []

        if config.testing.compute_metric_all_steps:
            for idx in range(self.num_timesteps + 1):
                current_t = self.num_timesteps - idx
                if self.args.loss_guidance == 'L2':
                    # compute RMSE
                    y_rmse = np.sqrt(np.mean(y_se_by_batch_list[current_t]))
                    y_rmse_all_steps_list.append(y_rmse)
                else:
                    # compute MAE
                    y_mae = np.mean(y_ae_by_batch_list[current_t])
                    y_mae_all_steps_list.append(y_mae)                
                # compute QICE
                all_gen_y = gen_y_by_batch_list[current_t]
                (y_true_ratio_by_bin,
                 qice_coverage_ratio,
                 y_true) = self.compute_true_coverage_by_gen_QI(
                     config=config, dataset_object=dataset_object,
                     all_true_y=all_true_y, all_generated_y=all_gen_y,
                     verbose=False)
                y_qice_all_steps_list.append(qice_coverage_ratio)
                # compute PICP
                coverage, _, _ = self.compute_PICP(
                    config=config, y_true=y_true, all_gen_y=all_gen_y)
                y_picp_all_steps_list.append(coverage)
                # compute NLL
                y_nll = np.mean(nll_by_batch_list[current_t])
                y_nll_all_steps_list.append(y_nll)
            # make plot for metrics across all timesteps during reverse diffusion
            n_metrics = 4
            fig, axs = plt.subplots(
                n_metrics, 1, figsize=(8.5, n_metrics * 3), clear=True) # W x H
            plt.subplots_adjust(hspace=0.5)
            xticks = np.arange(0, self.num_timesteps + 1,
                               config.diffusion.vis_step)
            # MAE/RMSE
            if self.args.loss_guidance == 'L2':
                axs[0].plot(y_rmse_all_steps_list)
            else:
                axs[0].plot(y_mae_all_steps_list)
            # axs[0].set_title('y RMSE across All Timesteps', fontsize=18)
            axs[0].set_xlabel('timestep', fontsize=12)
            axs[0].set_xticks(xticks)
            axs[0].set_xticklabels(xticks[::-1])
            if self.args.loss_guidance == 'L2':
                axs[0].set_ylabel('y RMSE', fontsize=12)
            else:
                axs[0].set_ylabel('y MAE', fontsize=12)
            # QICE
            axs[1].plot(y_qice_all_steps_list)
            # axs[1].set_title('y QICE across All Timesteps', fontsize=18)
            axs[1].set_xlabel('timestep', fontsize=12)
            axs[1].set_xticks(xticks)
            axs[1].set_xticklabels(xticks[::-1])
            axs[1].set_ylabel('y QICE', fontsize=12)
            # PICP
            picp_ideal = (config.testing.PICP_range[1] - 
                          config.testing.PICP_range[0]) / 100
            axs[2].plot(y_picp_all_steps_list)
            axs[2].axhline(y=picp_ideal, c='b')
            # axs[2].set_title('y PICP across All Timesteps', fontsize=18)
            axs[2].set_xlabel('timestep', fontsize=12)
            axs[2].set_xticks(xticks)
            axs[2].set_xticklabels(xticks[::-1])
            axs[2].set_ylabel('y PICP', fontsize=12)
            # NLL
            axs[3].plot(y_nll_all_steps_list)
            # axs[3].set_title('y NLL across All Timesteps', fontsize=18)
            axs[3].set_xlabel('timestep', fontsize=12)
            axs[3].set_xticks(xticks)
            axs[3].set_xticklabels(xticks[::-1])
            axs[3].set_ylabel('y NLL', fontsize=12)
            # fig.suptitle('y Metrics across All Timesteps')
            fig.savefig(os.path.join(args.im_path,
                                     'metrics_all_timesteps.pdf'))
        else:
            if self.args.loss_guidance == 'L2':               
                # compute RMSE
                y_rmse = np.sqrt(
                    np.mean(y_se_by_batch_list[config.testing.mean_t]))
                y_rmse_all_steps_list.append(y_rmse)
            else: 
                # compute MAE
                y_mae = np.mean(y_ae_by_batch_list[config.testing.mean_t])
                y_mae_all_steps_list.append(y_mae)  
            # compute QICE -- a cover metric
            all_gen_y = gen_y_by_batch_list[config.testing.coverage_t]
            (y_true_ratio_by_bin, 
             qice_coverage_ratio,
             y_true) = self.compute_true_coverage_by_gen_QI(
                 config=config, dataset_object=dataset_object,
                 all_true_y=all_true_y, all_generated_y=all_gen_y, verbose=True)
            y_qice_all_steps_list.append(qice_coverage_ratio)
            logging.info("\nWe generated {} y's given each x.".format(
                config.testing.n_z_samples))
            if self.args.loss_guidance == 'L2':
                logging.info(("\nRMSE between true mean y and the mean of \
                              generated y given each x is " +
                              "{:.8f};\nQICE between true y coverage ratio \
                                  by each generated y " +
                              "quantile interval and optimal ratio is {:.8f}."
                              ).format(y_rmse, qice_coverage_ratio))
            else:
                logging.info(("\nMAE between true mean y and the mean of \
                              generated y given each x is " +
                              "{:.8f};\nQICE between true y coverage ratio \
                                  by each generated y " +
                              "quantile interval and optimal ratio is {:.8f}."
                              ).format(y_mae, qice_coverage_ratio))                
            # compute PICP -- another coverage metric
            coverage, low, high = self.compute_PICP(
                config=config, y_true=y_true, all_gen_y=all_gen_y)
            y_picp_all_steps_list.append(coverage)
            logging.info(("There are {:.4f}% of true test y in the range of " +
                          "the computed {:.0f}% credible interval."
                          ).format(100 * coverage, high - low))
            # compute NLL
            y_nll = np.mean(nll_by_batch_list[config.testing.nll_t])
            y_nll_all_steps_list.append(y_nll)
            logging.info(
                "\nNegative Log-Likelihood on test set is {:.8f}.".format(y_nll))

        if self.args.loss_guidance == 'L2':
            logging.info(f"y RMSE at all steps: {y_rmse_all_steps_list}.\n")
        else:
            logging.info(f"y MAE at all steps: {y_mae_all_steps_list}.\n")
        logging.info(f"y QICE at all steps: {y_qice_all_steps_list}.\n")
        logging.info(f"y PICP at all steps: {y_picp_all_steps_list}.\n\n")
        logging.info(f"y NLL at all steps: {y_nll_all_steps_list}.\n\n")

        # clear the memory
        plt.close('all')
        del true_y_by_batch_list
        if config.testing.plot_gen:
            del all_true_x_tile
        del gen_y_by_batch_list
        if self.args.loss_guidance == 'L2':
            del y_se_by_batch_list
        else:
            del y_ae_by_batch_list
        gc.collect()
        if self.args.loss_guidance == 'L2':
            return (y_rmse_all_steps_list, y_qice_all_steps_list, 
                    y_picp_all_steps_list, y_nll_all_steps_list)
        else:
            return (y_mae_all_steps_list, y_qice_all_steps_list, 
                    y_picp_all_steps_list, y_nll_all_steps_list)