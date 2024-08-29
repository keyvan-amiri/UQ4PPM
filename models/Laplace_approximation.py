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
                     last_layer_name=None,
                     sigma_noise=None, prior_precision=None, temperature=None,                      
                     n_samples=None, link_approx=None, pred_type=None,
                     la_epochs=None, la_lr=None,
                     report_path=None, result_path=None,
                     split=None, fold=None, seed=None, device=None):    
    
    optional_args = dict() #empty dict for optional args in Laplace model
    optional_args['last_layer_name'] = last_layer_name    

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
        
    # path to random forest model to be saved after training
    deterministic_model_name = os.path.basename(deterministic_checkpoint_path)
    la_name = deterministic_model_name.replace('deterministic_', 'LA_')
    la_name = la_name.replace('.pt', '.pkl')
    la_path = os.path.join(result_path, la_name) 
    
    # create data loaders for training and inference
    X_train = torch.load(X_train_path)
    X_val = torch.load(X_val_path)
    X_test = torch.load(X_test_path)
    y_train = torch.load(y_train_path)
    y_val = torch.load(y_val_path)
    y_test = torch.load(y_test_path)
    train_dataset = TensorDataset(X_train, y_train)                        
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    try:
        batch_size = cfg.get('train').get('batch_size')
    except:
        batch_size = max_len  
    try:
        evaluation_batch_size = cfg.get('evaluation').get('batch_size')
    except:
        evaluation_batch_size = max_len     
    # in case of empirical_bayes: separate validation set to avoid overfitting
    if empirical_bayes:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        train_val_dataset = ConcatDataset([train_dataset, val_dataset])
        train_loader = DataLoader(train_val_dataset, batch_size=batch_size,
                                  shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=evaluation_batch_size,
                             shuffle=False)
   
    # set the model to training mode
    model = model.to(device)
    model.train() 
    
    # define the Laplace model, and conduct post-hoc fitting
    la = Laplace(model, likelihood='regression',
                 subset_of_weights=subset_of_weights,
                 hessian_structure=hessian_structure, sigma_noise=sigma_noise,
                 prior_precision=prior_precision, temperature=temperature,
                 **optional_args)
        
    la.fit(train_loader)
        
    return
