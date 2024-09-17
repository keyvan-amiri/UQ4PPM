"""
This python script is adapted from the origninal script:
    https://github.com/XzwHan/CARD
    CARD: Classification and Regression Diffusion Models by Xizewen Han,
    Huangjie Zheng, and Mingyuan Zhou.
"""

import os
import yaml
import argparse
import shutil
import sys
import json
import logging
from datetime import datetime, timezone, timedelta
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.utils.tensorboard as tb
from utils.utils import set_random_seed
from utils.evaluation import uq_eval
from utils.calibration import calibrated_regression


def card_calibration(args, best_key):
    # Uncertainty quantification evaluation on best experiment results
    only_seed = int(args.seed[0])
    if args.n_splits == 1:            
        test_name = 'CARD_holdout_seed_{}_{}_inference_result_.csv'.format(
                only_seed, best_key)
        val_name = 'CARD_holdout_seed_{}_{}_inference_result_validation_.csv'.format(
                only_seed, best_key)
    else:
        test_name = 'CARD_cv_fold{}_seed_{}_{}_inference_result_.csv'.format(
                        args.split, only_seed, best_key)
        val_name = 'CARD_cv_fold{}_seed_{}_{}_inference_result_validation_.csv'.format(
                args.split, only_seed, best_key)                  
    test_path = os.path.join(args.instance_path, test_name)
    val_path = os.path.join(args.instance_path, val_name)
    _ = uq_eval(test_path, 'CARD', report=True, verbose=True) 
    # now apply calibrated regression   
    recalibration_path = os.path.join(args.instance_path, 'recalibration')
    calibrated_result, recal_model = calibrated_regression(
            calibration_df_path=val_path, test_df_path=test_path, 
            uq_method='CARD', confidence_level=0.95,
            recalibration_path=recalibration_path, report=False)        
    uq_eval(calibrated_result, 'CARD', report=True, verbose=True,
            calibration_mode=True, calibration_type=args.calibration_type,
            recal_model=recal_model)

def card_hpo(args):
    only_seed = int(args.seed[0])
    # TODO: implement it for cross-fold validation as well
    first_experiment = args.exp_dict.get('Exp_1')
    extracted_keys = first_experiment.keys()
    # initilize a dictionary to collect results for all experiments
    hpo_results = {'exp_id': []}
    for key in extracted_keys:
        hpo_results[key] = []
    additional_keys = ['mae', 'rmse', 'nll', 'crps', 'sharp', 'ause', 'aurg',
                       'miscal_area', 'check', 'interval']
    for key in additional_keys:
        hpo_results[key] = []
    
    val_res_lst = []
    for key in args.exp_dict:
        experiment = args.exp_dict.get(key)
        hpo_results['exp_id'].append(key)
        for hpo_key in extracted_keys:
            hpo_results[hpo_key].append(experiment.get(hpo_key))
        # get csv file name for validation using key
        if args.n_splits == 1: 
            csv_name = 'CARD_holdout_seed_{}_{}_inference_result_validation_.csv'.format(
                only_seed, key)
        else:
            csv_name = 'CARD_cv_fold{}_seed_{}_{}_inference_result_validation_.csv'.format(
                args.split, only_seed, key)
        csv_path = os.path.join(args.instance_path, csv_name)    
        val_res_lst.append(csv_path)
        # call UQ evaluation without report option.
        uq_metrics = uq_eval(csv_path, 'CARD')
        hpo_results['mae'].append(uq_metrics.get('accuracy').get('mae'))
        hpo_results['rmse'].append(uq_metrics.get('accuracy').get('rmse'))
        hpo_results['nll'].append(uq_metrics.get('scoring_rule').get('nll'))
        hpo_results['crps'].append(uq_metrics.get('scoring_rule').get('crps'))
        hpo_results['sharp'].append(uq_metrics.get('sharpness').get('sharp'))
        hpo_results['ause'].append(uq_metrics.get(
            'Area Under Sparsification Error curve (AUSE)'))
        hpo_results['aurg'].append(uq_metrics.get(
            'Area Under Random Gain curve (AURG)'))
        hpo_results['miscal_area'].append(
            uq_metrics.get('avg_calibration').get('miscal_area'))                
        hpo_results['check'].append(uq_metrics.get('scoring_rule').get('check'))
        hpo_results['interval'].append(
            uq_metrics.get('scoring_rule').get('interval'))
    hpo_df = pd.DataFrame(hpo_results)
    if args.n_splits == 1: 
        hpo_name = 'CARD_holdout_seed_{}_hpo_result_.csv'.format(only_seed)
    else:
        hpo_name = 'CARD_cv_fold{}_seed_{}_hpo_result_.csv'.format(
            args.split, only_seed)    
    csv_filename = os.path.join(args.instance_path, hpo_name)            
    hpo_df.to_csv(csv_filename, index=False)
    min_exp_id = hpo_df[args.HPO_metric].idxmin()
    best_key = 'Exp_' + str(min_exp_id+1)
    # delet all validation results for inferior experiments
    for csv_path in val_res_lst:
        if best_key not in csv_path:
            os.remove(csv_path)
    return best_key 
    
    

def get_stat(args, config, temp_config, original_doc,
             y_rmse_all_splits_all_steps_list=None,
             y_mae_all_splits_all_steps_list=None,
             y_qice_all_splits_all_steps_list=None,
             y_picp_all_splits_all_steps_list=None,
             y_nll_all_splits_all_steps_list=None):
    n_timesteps = config.diffusion.timesteps
    rmse_idx = n_timesteps - args.rmse_timestep
    qice_idx = n_timesteps - args.qice_timestep
    picp_idx = n_timesteps - args.picp_timestep
    nll_idx = n_timesteps - args.nll_timestep
    if not temp_config.testing.compute_metric_all_steps:
        n_timesteps=rmse_idx=qice_idx=picp_idx=nll_idx=0
    #print(len(y_mae_all_splits_all_steps_list))
    #print(y_mae_all_splits_all_steps_list[0])
    if args.loss_guidance == 'L2':
        y_rmse_all_splits_list = [metric_list[rmse_idx] for metric_list in
                                  y_rmse_all_splits_all_steps_list]
    else:
        y_mae_all_splits_list = [metric_list[rmse_idx] for metric_list in 
                                 y_mae_all_splits_all_steps_list]                   
    y_qice_all_splits_list = [metric_list[qice_idx] for metric_list in 
                              y_qice_all_splits_all_steps_list]
    y_picp_all_splits_list = [metric_list[picp_idx] for metric_list in
                              y_picp_all_splits_all_steps_list]
    y_nll_all_splits_list = [metric_list[nll_idx] for metric_list in
                             y_nll_all_splits_all_steps_list]

    print("\n\n================ Results Across Splits ================")
    if args.loss_guidance == 'L2':
        print(f"y_RMSE mean: {np.mean(y_rmse_all_splits_list)}\
              y_RMSE std: {np.std(y_rmse_all_splits_list)}")
    else:
        print(f"y_MAE mean: {np.mean(y_mae_all_splits_list)} \
              y_MAE std: {np.std(y_mae_all_splits_list)}")
    print(f"QICE mean: {np.mean(y_qice_all_splits_list)}\
          QICE std: {np.std(y_qice_all_splits_list)}")
    print(f"PICP mean: {np.mean(y_picp_all_splits_list)}\
          PICP std: {np.std(y_picp_all_splits_list)}")
    print(f"NLL mean: {np.mean(y_nll_all_splits_list)}\
          NLL std: {np.std(y_nll_all_splits_list)}")

    # plot mean of all metric across all splits at all time steps 
    if args.loss_guidance == 'L2':
        y_rmse_all_splits_all_steps_array = np.array(
            y_rmse_all_splits_all_steps_list)
        y_rmse_mean_all_splits_list = [np.mean(
            y_rmse_all_splits_all_steps_array[:, idx]) 
            for idx in range(n_timesteps + 1)]                 
    else:
        y_mae_all_splits_all_steps_array = np.array(
            y_mae_all_splits_all_steps_list)        
        y_mae_mean_all_splits_list = [np.mean(
            y_mae_all_splits_all_steps_array[:, idx])
            for idx in range(n_timesteps + 1)]  
    y_qice_all_splits_all_steps_array = np.array(
        y_qice_all_splits_all_steps_list)
    y_picp_all_splits_all_steps_array = np.array(
        y_picp_all_splits_all_steps_list)
    y_nll_all_splits_all_steps_array = np.array(
        y_nll_all_splits_all_steps_list)         
    y_qice_mean_all_splits_list = [np.mean(
        y_qice_all_splits_all_steps_array[:, idx]) 
        for idx in range(n_timesteps + 1)]
    y_picp_mean_all_splits_list = [np.mean(
        y_picp_all_splits_all_steps_array[:, idx]) 
        for idx in range(n_timesteps + 1)]
    y_nll_mean_all_splits_list = [np.mean(
        y_nll_all_splits_all_steps_array[:, idx])
        for idx in range(n_timesteps + 1)]
    
    n_metric = 4
    fig, axs = plt.subplots(n_metric, 1, figsize=(8.5, n_metric * 3))  # W x H
    plt.subplots_adjust(hspace=0.5)
    xticks = np.arange(0, n_timesteps + 1, config.diffusion.vis_step)
    # MAE/RMSE
    if args.loss_guidance == 'L2':
        axs[0].plot(y_rmse_mean_all_splits_list)
    else:   
        axs[0].plot(y_mae_mean_all_splits_list)
    axs[0].set_xlabel('timestep', fontsize=12)
    axs[0].set_xticks(xticks)
    axs[0].set_xticklabels(xticks[::-1])
    if args.loss_guidance == 'L2':
        axs[0].set_ylabel('y RMSE', fontsize=12)
    else:
        axs[0].set_ylabel('y MAE', fontsize=12)                    
    # NLL
    axs[1].plot(y_nll_mean_all_splits_list)
    axs[1].set_xlabel('timestep', fontsize=12)
    axs[1].set_xticks(xticks)
    axs[1].set_xticklabels(xticks[::-1])
    axs[1].set_ylabel('y NLL', fontsize=12)
    # QICE
    axs[2].plot(y_qice_mean_all_splits_list)
    axs[2].set_xlabel('timestep', fontsize=12)
    axs[2].set_xticks(xticks)
    axs[2].set_xticklabels(xticks[::-1])
    axs[2].set_ylabel('y QICE', fontsize=12)
    # PICP
    picp_ideal = (config.testing.PICP_range[1]-config.testing.PICP_range[0])/100
    axs[3].plot(y_picp_mean_all_splits_list)
    axs[3].axhline(y=picp_ideal, c='b', label='95 % coverage')
    axs[3].set_xlabel('timestep', fontsize=12)
    axs[3].set_xticks(xticks)
    axs[3].set_xticklabels(xticks[::-1])
    axs[3].set_ylabel('y PICP', fontsize=12)
    axs[3].legend()

    # fig.suptitle('Mean y Metrics of All Splits across All Timesteps')
    im_path = os.path.join(args.exp, config.testing.image_folder, original_doc)
    if not os.path.exists(im_path):
        os.makedirs(im_path)
    fig.savefig(os.path.join(im_path,
                             'mean_metrics_all_splits_all_timesteps.pdf'))

    timestr = datetime.now(timezone(timedelta(hours=+2))).strftime(
        "%Y%m%d-%H%M%S-%f")  # Europe Central time
    res_file_path = os.path.join(args.exp, "logs", original_doc,
                                 "metrics_all_splits")
    if args.loss_guidance == 'L2':
        res_dict = {'split_num': args.n_splits, 'task': original_doc,
                    'y_RMSE mean': float(np.mean(y_rmse_all_splits_list)),
                    'y_RMSE std': float(np.std(y_rmse_all_splits_list)),
                    'QICE mean': float(np.mean(y_qice_all_splits_list)),
                    'QICE std': float(np.std(y_qice_all_splits_list)),
                    'PICP mean': float(np.mean(y_picp_all_splits_list)),
                    'PICP std': float(np.std(y_picp_all_splits_list)),
                    'NLL mean': float(np.mean(y_nll_all_splits_list)),
                    'NLL std': float(np.std(y_nll_all_splits_list))
                    }  
    else:
        res_dict = {'split_num': args.n_splits, 'task': original_doc,
                    'y_MAE mean': float(np.mean(y_mae_all_splits_list)),
                    'y_MAE std': float(np.std(y_mae_all_splits_list)),
                    'QICE mean': float(np.mean(y_qice_all_splits_list)),
                    'QICE std': float(np.std(y_qice_all_splits_list)),
                    'PICP mean': float(np.mean(y_picp_all_splits_list)),
                    'PICP std': float(np.std(y_picp_all_splits_list)),
                    'NLL mean': float(np.mean(y_nll_all_splits_list)),
                    'NLL std': float(np.std(y_nll_all_splits_list))
                    }              
    args_dict = {'task': config.data.dataset, 'loss': args.loss_guidance,
                 'guidance': config.model.type, 'n_timesteps': n_timesteps,
                 'split_num': args.n_splits}
    # save metrics and model hyperparameters to a json file
    if not os.path.exists(res_file_path):
        os.makedirs(res_file_path)
    with open(res_file_path + f"/metrics_{timestr}.json", "w") as outfile:
        json.dump(res_dict, outfile)
        outfile.write('\n\nExperiment arguments:\n')
        json.dump(args_dict, outfile)
    print("\nTest metrics saved in .json file.") 


def set_experiments(args, main_cfg):
    # Also look at parse_config to understad how HPO experiments are conducted
    # create a dictionary for experiments to be done
    # dimension for linear layer used in noise estimation network
    feature_dim_lst = main_cfg.get('hpo').get('feature_dim') 
    if not isinstance(feature_dim_lst, list):
        raise ValueError('Noise estimation network width options is a list.') 
    # whether to concatanate data features (x) to the input of noise estimation
    # network or not    
    cat_x_lst = main_cfg.get('hpo').get('cat_x') 
    if not isinstance(cat_x_lst, list):
        raise ValueError('Options for concatanation should be a list.')
    # if concatanate x, how many events to be included (if noise estimation network is FNN)
    window_cat_x_lst = main_cfg.get('hpo').get('window_cat_x') 
    if not isinstance(window_cat_x_lst, list):
        raise ValueError('Options for window size should be a list.') 
    n_epochs_lst = main_cfg.get('hpo').get('n_epochs') 
    if not isinstance(n_epochs_lst, list):
        raise ValueError('Training epochs options for diffusion model size \
                         should be a list.') 
    beta_start_lst = main_cfg.get('hpo').get('beta_start') 
    if not isinstance(beta_start_lst, list):
        raise ValueError('Beta start options should be a list.') 
    # joint_train: if False: train point estimate, then use it as prior for diffusion
    # model. Otherwise, joint training for backbone and diffusion model.
    joint_train_lst = main_cfg.get('hpo').get('joint_train') 
    if not isinstance(joint_train_lst, list):
        raise ValueError('Joint training options should be a list.')
    beta_schedule_lst = main_cfg.get('hpo').get('beta_schedule') 
    if not isinstance(beta_schedule_lst, list):
        raise ValueError('beta schedule options options should be a list.')

    experiments = []
    for beta_schedule in beta_schedule_lst:
        for dim in feature_dim_lst:
            for n_epochs in n_epochs_lst:
                for beta_start in beta_start_lst:
                    beta_end = 200 * beta_start
                    for joint_train in joint_train_lst:
                        if joint_train:
                            pre_train = False
                        else:
                            pre_train = True
                        for cat_control in cat_x_lst:
                            if cat_control:
                                # Use the full range of window size values
                                for window_size in window_cat_x_lst:
                                    experiments.append({
                                        'feature_dim': dim,
                                        'cat_x': cat_control,
                                        'window_cat_x': window_size,
                                        'beta_start': beta_start,
                                        'beta_end': beta_end,
                                        'n_epochs': n_epochs,
                                        'joint_train': joint_train,
                                        'pre_train': pre_train,
                                        'beta_schedule': beta_schedule})
                            else:
                                # max_depth is None when depth_control is False
                                experiments.append({
                                    'feature_dim': dim,
                                    'cat_x': cat_control,
                                    'window_cat_x': None,
                                    'beta_start': beta_start,
                                    'beta_end': beta_end,
                                    'n_epochs': n_epochs,
                                    'joint_train': joint_train,
                                    'pre_train': pre_train,
                                    'beta_schedule': beta_schedule})
                           
    exp_dict = {}
    for exp_id in range (1, len(experiments)+1):
        key_str = 'Exp_' + str(exp_id)
        exp_dict[key_str] = experiments[exp_id-1]
    args.exp_dict = exp_dict    
    return args    
    

def adjust_arguments(args, main_cfg, test=False, validation=False, 
                     root_path=None):
    """
    a method to add extra information to the parsed arguments for CARD.
    It adjust the following important arguments
    --loss_guidance: loss function for guidance model that predicts y_0_hat.
    If L2 all CARD, results are reported in RMSE otherwise it will be
    reported in MAE.
    --noise_prior: Whether to apply a noise prior distribution at timestep T.
    If True, a noise prior is applied instead of pre-trained guidance model.
    noise_prior_approach (zero, mean, or median) is specified separately.
    --no_cat_f_phi: Whether to not concatenate f_phi as part of eps_theta input.
    If True, new_config.model.cat_y_pred is set to False. This means that f_phi
    (point estimate) is not concatanated in noise estimation network.
    --nll_global_var: Apply global variance for NLL computation.
    --nll_test_var: Apply sample variance of the test set for NLL computation.
    Execution control parameters including:
        --run_all: Whether to run all train test splits.
        --train_guidance_only: Whether to only pre-train the guidance 
        model f_phi. If True, only deterministic model is trained.
        --resume_training: Whether to resume training or strat from scratch.
        --ni: stands for No Interaction. Suitable for Slurm Job launcher.
        --split: which split to use for regression data.
        --init_split: initial split to train for regression data.
    Evaluation metrics arguments including: rmse_timestep, qice_timestep, 
    picp_timestep, nll_timestep
    Documentation arguments including:
        comment: A string for experiment comment
        verbose: Verbose level: info | debug | warning | critical
        i: The folder name of samples.
    """
    # set test and validation modes
    args.test = test
    args.validation = validation
    # Adjust a directory for results of inference
    args.instance_path = os.path.join(
        root_path, 'results', args.dataset, args.model)
    # adjust some arguments based on main configuration file
    args.loss_guidance = main_cfg.get('loss_guidance')
    args.noise_prior = main_cfg.get('noise_prior')
    args.no_cat_f_phi = main_cfg.get('no_cat_f_phi')
    args.nll_global_var = main_cfg.get('nll_global_var')
    args.nll_test_var = main_cfg.get('nll_test_var')
    args.rmse_timestep = main_cfg.get('rmse_timestep')
    args.qice_timestep = main_cfg.get('qice_timestep')
    args.picp_timestep = main_cfg.get('picp_timestep')
    args.nll_timestep = main_cfg.get('nll_timestep')
    args.comment = main_cfg.get('comment')
    args.verbose = main_cfg.get('verbose')
    args.i = main_cfg.get('i')
    args.run_all = main_cfg.get('execution_control').get('run_all')    
    args.train_guidance_only = main_cfg.get('execution_control').get(
        'train_guidance_only')
    args.resume_training = main_cfg.get('execution_control').get(
        'resume_training')
    args.ni = main_cfg.get('execution_control').get('ni')
    args.split = main_cfg.get('execution_control').get('split')
    args.init_split = main_cfg.get('execution_control').get('init_split')
    # metric that is used for hyper-parameter tuning
    args.HPO_metric = main_cfg.get('HPO_metric')
    args.calibration_type = main_cfg.get('calibration_type')
    return args

#TODO: to experiment with other beta schedulers
def make_beta_schedule(schedule="linear", num_timesteps=1000,
                       start=1e-5, end=1e-2):
    if schedule == "linear":
        betas = torch.linspace(start, end, num_timesteps)
    elif schedule == "const":
        betas = end * torch.ones(num_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, num_timesteps) ** 2
    elif schedule == "jsd":
        betas = 1.0 / torch.linspace(num_timesteps, 1, num_timesteps)
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, num_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    elif schedule == "cosine" or schedule == "cosine_reverse":
        max_beta = 0.999
        cosine_s = 0.008
        betas = torch.tensor(
            [min(1 - (math.cos(((i + 1) / num_timesteps + cosine_s) / 
                               (1 + cosine_s) * math.pi / 2) ** 2) / (
                                   math.cos((i / num_timesteps + cosine_s) / 
                                            (1 + cosine_s) * math.pi / 2) ** 2
                                   ), max_beta) for i in range(num_timesteps)])
        if schedule == "cosine_reverse":
            betas = betas.flip(0)  # starts at max_beta then decreases fast
    elif schedule == "cosine_anneal":
        betas = torch.tensor(
            [start + 0.5 * (end - start) * (
                1 - math.cos(t / (num_timesteps - 1) * math.pi)) for t in
             range(num_timesteps)])
    return betas

class EMA(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = type(module)(module.config).to(module.config.device)
        module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

#TODO: understand what does q_sample do?
# Forward functions
def q_sample(y, y_0_hat, alphas_bar_sqrt, one_minus_alphas_bar_sqrt,
             t, noise=None):
    """
    y_0_hat: prediction of pre-trained guidance model;
    can be extended to represent any prior mean setting at timestep T.
    """
    if noise is None:
        noise = torch.randn_like(y).to(y.device)
    sqrt_alpha_bar_t = extract(alphas_bar_sqrt, t, y)
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y)
    # q(y_t | y_0, x)
    y_t = sqrt_alpha_bar_t * y + (
        1 - sqrt_alpha_bar_t) * y_0_hat + sqrt_one_minus_alpha_bar_t * noise
    return y_t


# Reverse function -- sample y_{t-1} given y_t
def p_sample(model, x, y, y_0_hat, y_T_mean, t, alphas,
             one_minus_alphas_bar_sqrt, squeeze=False):
    """
    Reverse diffusion process sampling -- one time step.
    y: sampled y at time step t, y_t.
    y_0_hat: prediction of pre-trained guidance model.
    y_T_mean: mean of prior distribution at timestep T.
    We replace y_0_hat with y_T_mean in the forward process posterior mean
    computation, emphasizing that guidance model prediction y_0_hat = f_phi(x)
    is part of the input to eps_theta network, while in paper we also choose
    to set the prior mean at timestep T y_T_mean = f_phi(x).
    """
    device = next(model.parameters()).device
    z = torch.randn_like(y)  # if t > 1 else torch.zeros_like(y)
    t = torch.tensor([t]).to(device)
    alpha_t = extract(alphas, t, y)
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y)
    sqrt_one_minus_alpha_bar_t_m_1 = extract(one_minus_alphas_bar_sqrt, t - 1, y)
    sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
    sqrt_alpha_bar_t_m_1 = (1 - sqrt_one_minus_alpha_bar_t_m_1.square()).sqrt()
    # y_t_m_1 posterior mean component coefficients
    gamma_0 = (1 - alpha_t) * sqrt_alpha_bar_t_m_1 / (
        sqrt_one_minus_alpha_bar_t.square())
    gamma_1 = (sqrt_one_minus_alpha_bar_t_m_1.square()
               ) * (alpha_t.sqrt()) / (sqrt_one_minus_alpha_bar_t.square())
    gamma_2 = 1 + (sqrt_alpha_bar_t - 1) * (
        alpha_t.sqrt() + sqrt_alpha_bar_t_m_1) / (sqrt_one_minus_alpha_bar_t.square())
    if squeeze:
        eps_theta = model(x, y, y_0_hat, t).squeeze().to(device).detach()
    else:
        eps_theta = model(x, y, y_0_hat, t).to(device).detach()    
    # y_0 reparameterization
    y_0_reparam = 1 / sqrt_alpha_bar_t * (
        y - (1 - sqrt_alpha_bar_t) * y_T_mean - eps_theta * 
        sqrt_one_minus_alpha_bar_t)
    # posterior mean
    y_t_m_1_hat = gamma_0 * y_0_reparam + gamma_1 * y + gamma_2 * y_T_mean
    # posterior variance
    beta_t_hat = (sqrt_one_minus_alpha_bar_t_m_1.square()) / (
        sqrt_one_minus_alpha_bar_t.square()) * (1 - alpha_t)
    y_t_m_1 = y_t_m_1_hat.to(device) + beta_t_hat.sqrt().to(device) * z.to(device)
    return y_t_m_1


# Reverse function -- sample y_0 given y_1
def p_sample_t_1to0(model, x, y, y_0_hat, y_T_mean, one_minus_alphas_bar_sqrt,
                    squeeze=False):
    device = next(model.parameters()).device
    # corresponding to timestep 1 (i.e., t=1 in diffusion models)
    t = torch.tensor([0]).to(device)  
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y)
    sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
    if squeeze:
        eps_theta = model(x, y, y_0_hat, t).squeeze().to(device).detach()
    else:
        eps_theta = model(x, y, y_0_hat, t).to(device).detach()
    # y_0 reparameterization
    y_0_reparam = 1 / sqrt_alpha_bar_t * (
            y - (1 - sqrt_alpha_bar_t) * y_T_mean - eps_theta * 
            sqrt_one_minus_alpha_bar_t)
    y_t_m_1 = y_0_reparam.to(device)
    return y_t_m_1


def p_sample_loop(model, x, y_0_hat, y_T_mean, n_steps, alphas,
                  one_minus_alphas_bar_sqrt, squeeze=False):
    device = next(model.parameters()).device
    z = torch.randn_like(y_T_mean).to(device)
    cur_y = z + y_T_mean  # sample y_T
    y_p_seq = [cur_y]
    for t in reversed(range(1, n_steps)):  # t from T to 2
        y_t = cur_y
        cur_y = p_sample(model, x, y_t, y_0_hat, y_T_mean, t, alphas, 
                         one_minus_alphas_bar_sqrt, squeeze=squeeze)  # y_{t-1}
        y_p_seq.append(cur_y)
    assert len(y_p_seq) == n_steps
    y_0 = p_sample_t_1to0(model, x, y_p_seq[-1], y_0_hat, y_T_mean, 
                          one_minus_alphas_bar_sqrt, squeeze=squeeze)
    y_p_seq.append(y_0)
    return y_p_seq

###############################################################################

# Evaluation with KLD
def kld(y1, y2, grid=(-20, 20), num_grid=400):
    y1, y2 = y1.numpy().flatten(), y2.numpy().flatten()
    p_y1, _ = np.histogram(y1, bins=num_grid, range=[grid[0], grid[1]],
                           density=True)
    p_y1 += 1e-7
    p_y2, _ = np.histogram(y2, bins=num_grid, range=[grid[0], grid[1]],
                           density=True)
    p_y2 += 1e-7
    return (p_y1 * np.log(p_y1 / p_y2)).sum()

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

# return original config before any changes within parse_config method.
def parse_temp_config(task_name=None):
    task_name = task_name + '.yml'
    with open(os.path.join('cfg', task_name), "r") as f:
         temp_config = yaml.safe_load(f)
         temporary_config = dict2namespace(temp_config)
    return temporary_config

# update original config file, and return it alongside the logger.
def parse_config(args=None, key=None):    
    # set log path
    args.log_path = os.path.join(args.exp, 'logs', args.doc)
    # set separate log folder for validation and test results
    if args.validation:
        args.log_path2 = os.path.join(args.exp, 'validation', args.doc)
        
    # parse config file
    with open(os.path.join(args.config), "r") as f:
        if args.test:
            config = yaml.unsafe_load(f)
            new_config = config
        else:
            config = yaml.safe_load(f)
            new_config = dict2namespace(config)
    tb_path = os.path.join(args.exp, 'tensorboard', args.doc)

    if not args.test:
        args.im_path = os.path.join(args.exp, new_config.training.image_folder, args.doc)
        # if noise_prior is not provided by the user the relevant config is set to False.
        new_config.diffusion.noise_prior = True if args.noise_prior else False
        new_config.model.cat_y_pred = False if args.no_cat_f_phi else True
        new_config.model.cat_x = args.exp_dict.get(key).get('cat_x')
        new_config.model.window_cat_x = (
            args.exp_dict.get(key).get('window_cat_x'))
        new_config.model.feature_dim = (
            args.exp_dict.get(key).get('feature_dim'))
        new_config.diffusion.beta_start = (
            args.exp_dict.get(key).get('beta_start'))
        new_config.diffusion.beta_end = args.exp_dict.get(key).get('beta_end')
        new_config.diffusion.beta_schedule = (
            args.exp_dict.get(key).get('beta_schedule'))
        new_config.diffusion.nonlinear_guidance.pre_train = (
            args.exp_dict.get(key).get('pre_train'))
        new_config.diffusion.nonlinear_guidance.joint_train = (
            args.exp_dict.get(key).get('joint_train'))
        new_config.training.n_epochs = args.exp_dict.get(key).get('n_epochs')
        
        if not args.resume_training:
            if not args.timesteps is None:
                new_config.diffusion.timesteps = args.timesteps
            if os.path.exists(args.log_path):
                overwrite = False
                if args.ni:
                    overwrite = True
                else:
                    response = input(
                        'Folder {} already exists. Overwrite? (Y/N)'.format(
                            args.log_path))
                    if response.upper() == "Y":
                        overwrite = True
                if overwrite:
                    shutil.rmtree(args.log_path)
                    shutil.rmtree(tb_path)
                    shutil.rmtree(args.im_path)
                    os.makedirs(args.log_path)
                    os.makedirs(args.im_path)
                    if os.path.exists(tb_path):
                        shutil.rmtree(tb_path)
                else:
                    print('Folder exists. Program halted.')
                    sys.exit(0)
            else:
                os.makedirs(args.log_path)
                if not os.path.exists(args.im_path):
                    os.makedirs(args.im_path)
            #save the updated config the log in result folder
            with open(os.path.join(args.log_path, 'config.yml'), 'w') as f:
                yaml.dump(new_config, f, default_flow_style=False)

        new_config.tb_logger = tb.SummaryWriter(log_dir=tb_path)
        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        # saving training info to a .txt file
        handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        if args.validation:
            # set the image path for inference on validation
            args.im_path = os.path.join(
                args.exp, 'validation', new_config.testing.image_folder,
                args.doc)
        else:
            # set the image path for inference on test set.
            args.im_path = os.path.join(
                args.exp, new_config.testing.image_folder, args.doc)
        
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        # saving test metrics to a .txt file
        if args.validation:
            txt_path = os.path.join(args.log_path2,'testmetrics.txt')
            os.makedirs(os.path.dirname(txt_path), exist_ok=True)
            # set a handler for inference on validation
            open(txt_path, 'w').close()
            handler2 = logging.FileHandler(txt_path)
        else:
            # set a handler for inference on test
            handler2 = logging.FileHandler(
                os.path.join(args.log_path, 'testmetrics.txt'))
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

        os.makedirs(args.im_path, exist_ok=True)

    # add device
    device_name = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set number of threads
    if args.thread > 0:
        torch.set_num_threads(args.thread)
        print('Using {} threads'.format(args.thread))

    # set random seed
    if isinstance(args.seed, list):
        seed = int(args.seed[0])       
    set_random_seed(seed)

    torch.backends.cudnn.benchmark = True

    return new_config, logger