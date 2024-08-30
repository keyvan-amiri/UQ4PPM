"""
To build this python script we use Laplace library:
    https://github.com/aleximmer/Laplace
    Introduced by:
    Laplace Redux – Effortless Bayesian Deep Learning
    Erik Daxberger, Agustinus Kristiadi, Alexander Immer,
    Runa Eschenhagen, Matthias Bauerd, Philipp Hennig.
"""
import os
import argparse
import yaml
import sys
import logging
import random
import shutil
import pickle
import dill
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_
import torch.optim as optim
import torch.utils.tensorboard as tb
from sklearn.ensemble import RandomForestRegressor
from loss.mape import mape
from laplace import Laplace

def post_hoc_laplace(model=None, cfg=None,
                     X_train_path=None, X_val_path=None, X_test_path=None,
                     y_train_path=None, y_val_path=None, y_test_path=None,                  
                     test_original_lengths=None, max_len=None,
                     y_scaler=None, normalization=False,
                     subset_of_weights=None, hessian_structure='kron', 
                     empirical_bayes=False,
                     method=None, grid_size=None,
                     last_layer_name=None,
                     sigma_noise=None, stat_noise=False, 
                     prior_precision=None, temperature=None,                      
                     n_samples=None, link_approx=None, pred_type=None,
                     la_epochs=None, la_lr=None,
                     report_path=None, result_path=None,
                     split=None, fold=None, seed=None, device=None):    

    # get current time (as start) to compute training time
    start=datetime.now()
    # Announcement for fitting Laplace approximation
    if split=='holdout':
        print(f'Fitting Laplace for pre-trained model for {split} data split.')
    else:
        print(f'Fitting Laplace for pre-trained model for {split} data split, fold: {fold}.')
    with open(report_path, 'w') as file:
        file.write('Configurations:\n')
        file.write(str(cfg))
        file.write('\n')
        file.write('\n') 
       
    # Load deterministic pre-trained point estimate
    # path to deterministic point estimate
    if split == 'holdout':
        deterministic_checkpoint_path = os.path.join(
            result_path,
            'deterministic_{}_seed_{}_best_model.pt'.format(split, seed))
    else:
        deterministic_checkpoint_path = os.path.join(
            result_path,
            'deterministic_{}_fold{}_seed_{}_best_model.pt'.format(
                split, fold, seed))
    # check deterministic pre-trained model is available
    if not os.path.isfile(deterministic_checkpoint_path):
        raise FileNotFoundError('Deterministic model must be trained first')
    else:
        # load the checkpoint except the last layer
        checkpoint = torch.load(deterministic_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
    # Path to Laplace model to be saved after training
    deterministic_model_name = os.path.basename(deterministic_checkpoint_path)
    la_name = deterministic_model_name.replace('deterministic_', 'LA_')
    la_path = os.path.join(result_path, la_name) 
    
    # load feature vectors for train, val, test sets
    X_train = torch.load(X_train_path)
    X_val = torch.load(X_val_path)
    X_test = torch.load(X_test_path)
    y_train = torch.load(y_train_path)
    y_val = torch.load(y_val_path)
    y_test = torch.load(y_test_path)
    # create datasets for training and inference with Laplace model
    train_dataset = TensorDataset(X_train, y_train)                        
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    train_val_dataset = ConcatDataset([train_dataset, val_dataset])
    # get the batch size    
    try:
        batch_size = cfg.get('train').get('batch_size')
    except:
        batch_size = max_len  
    try:
        evaluation_batch_size = cfg.get('evaluation').get('batch_size')
    except:
        evaluation_batch_size = max_len 
    # create data loaders for training and inference
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                              shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=evaluation_batch_size,
                             shuffle=False)
    # in some cases we do not any separate validation set.
    train_val_loader = DataLoader(train_val_dataset, batch_size=batch_size,
                                  shuffle=True)
    
    if stat_noise:
        # set sigma noise based on the statistical propertis of target attribute
        y_all = torch.cat([y for _, y in train_val_dataset], dim=0)  
        y_all = y_all.numpy()
        y_std = np.std(y_all)
        y_IQR = np.percentile(y_all, 75) - np.percentile(y_all, 25)
        y_gold = (y_std + 3*y_IQR)/4
        sigma_noise=sigma_noise*y_gold 
    
    # Move the model to device, and set it to training mode
    model = model.to(device)
    model.train() 
    
    ##########################################################################
    ##########  Laplace model, fitting it in a post-hoc fashion   ###########
    ##########################################################################
    optional_args = dict() #empty dict for optional args in Laplace model
    optional_args['last_layer_name'] = last_layer_name 
    la = Laplace(model, likelihood='regression',
                 subset_of_weights=subset_of_weights,
                 hessian_structure=hessian_structure,
                 sigma_noise=sigma_noise,
                 prior_precision=prior_precision,
                 temperature=temperature,
                 **optional_args) 
    # Fit the local Laplace approximation at the parameters of the model.
    # i.e., approximate the posterior of model's weight around MAP estimates.
    # Laplace approximation (LA) is equivalent to fitting a Gaussian with:
        # Mean = MAP estimate
        # covariance matrix = negative inverse Hessian of the loss at the MAP
    # Which data to use?
    # if empirical_bayes=False, method== gridsearch: we need separate val set 
    # else: we use training + validation set to fit the model
    # beccause if method== gridsearch, then validation set is neglected
    if ((not empirical_bayes) and (method == 'gridsearch')):
        la.fit(train_loader)
    else:
        la.fit(train_val_loader)
    # optimnize prior precision 
    # whether to estimate the prior precision and observation noise using
    # empirical Bayes after training or not.
    if not empirical_bayes:
        if method == 'gridsearch':
            la.optimize_prior_precision(pred_type=pred_type,
                                        method=method,
                                        n_steps=la_epochs,
                                        lr=la_lr,
                                        init_prior_prec=prior_precision,
                                        val_loader=val_loader,
                                        grid_size=grid_size,
                                        progress_bar=True)
        else:
            # optimization using marglik method in Laplace library
            la.optimize_prior_precision(pred_type=pred_type,
                                        method=method,
                                        n_steps=la_epochs,
                                        lr=la_lr,
                                        init_prior_prec=prior_precision,
                                        n_samples=n_samples,
                                        link_approx=link_approx,
                                        progress_bar=True)
    else:
        # log of the prior precision 
        log_prior = torch.full((1,), prior_precision, requires_grad=True,
                               dtype=torch.float32)
        #log_prior = torch.ones(1, requires_grad=True)
        # log of the observation noise
        log_sigma = torch.full((1,), sigma_noise,
                               requires_grad=True, dtype=torch.float32)
        #log_sigma = torch.ones(1, requires_grad=True)
        hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=la_lr)
        for i in range(la_epochs):
            hyper_optimizer.zero_grad()
            neg_marglik = - la.log_marginal_likelihood(log_prior.exp(),
                                                       log_sigma.exp())
            neg_marglik.backward()
            hyper_optimizer.step()
            # print the results       
            print(f'Epoch {i + 1}/{la_epochs},', 
                  f'NLL: {neg_marglik}, log prior: {log_prior}, log sigma: {log_sigma}')
            with open(report_path, 'a') as file:
                file.write(
                    'Epoch {}/{} NLL: {}, log prior: {}, log sigma: {}.\n'.format(
                        i+1, la_epochs, neg_marglik, log_prior, log_sigma))     
        
    #save the Laplace model (including deterministic model wrapped in it)
    torch.save(la, la_path, pickle_module=dill)
    
    training_time = (datetime.now()-start).total_seconds()
    print('Fitting and optimizing the Laplace approximation is done')
    with open(report_path, 'a') as file:
        file.write('#######################################################\n')
        file.write('Training time- in seconds: {}\n'.format(
            training_time)) 
        file.write('#######################################################\n')
        file.write('#######################################################\n')
    
    # Bayesian predictive
    if split=='holdout':
        print(f'Start Bayesian inference for {split} data split.')
    else:
        print(f'Start Bayesian inference for {split} data split, fold: {fold}.')
    
    start=datetime.now()
    # empty dictionary to collect inference result in a dataframe
    all_results = {'GroundTruth': [], 'Prediction': [],
                   'Epistemic_Uncertainty': [], 'Prefix_length': [],
                   'Absolute_error': [], 'Absolute_percentage_error': []}   
    # set variabls to zero to collect loss values and length ids
    absolute_error = 0
    absolute_percentage_error = 0
    length_idx = 0
    
    with torch.no_grad():
        # get model prediction and epistemic uncertainty
        for index, test_batch in enumerate(test_loader):
            inputs = test_batch[0].to(device)
            _y_truth = test_batch[1].to(device)
            batch_size = inputs.shape[0] 
            _y_pred, f_var = la(inputs, pred_type=pred_type) 
            #_y_pred = _y_pred.squeeze()  
            # Remove the dimension of size 1 along axis 2
            #f_var = f_var.squeeze()
            f_var = f_var.squeeze(dim=2) 
            # Compute square root element-wise 
            epistemic_std = torch.sqrt(f_var + la.sigma_noise.item()**2)
            #f_std = torch.sqrt(f_var)
            # conduct inverse normalization if required
            if normalization:
                _y_truth = y_scaler * _y_truth
                _y_pred = y_scaler * _y_pred  
                epistemic_std = y_scaler * epistemic_std
            # Compute batch loss
            absolute_error += F.l1_loss(_y_pred, _y_truth).item()
            absolute_percentage_error += mape(_y_pred, _y_truth).item()
            # Detach predictions and ground truths (np arrays)
            _y_truth = _y_truth.detach().cpu().numpy()
            _y_pred = _y_pred.detach().cpu().numpy()
            mae_batch = np.abs(_y_truth - _y_pred)
            mape_batch = (mae_batch/_y_truth*100)
            # collect inference result in all_result dict.
            all_results['GroundTruth'].extend(_y_truth.tolist())
            all_results['Prediction'].extend(_y_pred.tolist())
            pre_lengths = \
                test_original_lengths[length_idx:length_idx+batch_size]
            length_idx+=batch_size
            prefix_lengths = (np.array(pre_lengths).reshape(-1, 1)).tolist()
            all_results['Prefix_length'].extend(prefix_lengths)
            all_results['Absolute_error'].extend(mae_batch.tolist())
            all_results['Absolute_percentage_error'].extend(mape_batch.tolist())
            epistemic_std = epistemic_std.detach().cpu().numpy()
            all_results['Epistemic_Uncertainty'].extend(
                epistemic_std.tolist())
        
        num_test_batches = len(test_loader)    
        absolute_error /= num_test_batches    
        absolute_percentage_error /= num_test_batches
    print('Test - MAE: {:.3f}, '
                  'MAPE: {:.3f}'.format(
                      round(absolute_error, 3),
                      round(absolute_percentage_error, 3))) 
    inference_time = (datetime.now()-start).total_seconds()
    # inference time is reported in milliseconds.
    instance_t = inference_time/len(test_original_lengths)*1000
    with open(report_path, 'a') as file:
        file.write('Inference time- in seconds: {}\n'.format(inference_time))
        file.write(
            'Inference time for each instance- in miliseconds: {}\n'.format(
                instance_t))
        file.write('Test - MAE: {:.3f}, '
                      'MAPE: {:.3f}'.format(
                          round(absolute_error, 3),
                          round(absolute_percentage_error, 3)))
    flattened_list = [item for sublist in all_results['Prefix_length'] 
                      for item in sublist]
    all_results['Prefix_length'] = flattened_list
    results_df = pd.DataFrame(all_results)
    if split=='holdout':
        csv_filename = os.path.join(
            result_path,'LA_{}_seed_{}_inference_result_.csv'.format(split,seed))
    else:
        csv_filename = os.path.join(
            result_path,'LA_{}_fold{}_seed_{}_inference_result_.csv'.format(
                split, fold, seed))         
    results_df.to_csv(csv_filename, index=False)