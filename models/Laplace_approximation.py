"""
To build this python script we use Laplace library:
    https://github.com/aleximmer/Laplace
    Introduced by:
    Laplace Redux – Effortless Bayesian Deep Learning
    Erik Daxberger, Agustinus Kristiadi, Alexander Immer,
    Runa Eschenhagen, Matthias Bauerd, Philipp Hennig.
"""
import os
import dill
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from laplace import Laplace
from loss.mape import mape
from utils.utils import add_suffix_to_csv


def post_hoc_laplace(model=None, cfg=None, y_train_val=None, 
                     train_loader=None, val_loader=None, train_val_loader=None,
                     subset_of_weights='last_layer', last_layer_name=None, 
                     hessian_structure='kron', empirical_bayes=False,
                     method='marglik', la_epochs=None, la_lr=None, 
                     sigma_noise=None, stat_noise=False, 
                     prior_precision=None, temperature=None,                      
                     n_samples=None, link_approx='mc', pred_type='glm',
                     report_path=None, result_path=None, split='holdout',
                     fold=None, seed=None, device=None, exp_id=None):    
    """
    A method to fit a Laplace model, and optimize its preior precision.
    
    Parameters
    ----------
    model : Backbone model which provides MAP estimates (point estimate)
    cfg : Congiguration file that is used for training, and inference
    y_train_val : a numpy array containing all remaining time values in train,
    and validation datasets.
    train_loader : A Pytorch data loader containing all training examples.
    val_loader : A Pytorch data loader containing all validation examples.
    train_val_loader : A Pytorch data loader for all training, validation examples.   
    subset_of_weights : type of weights that Hessian is computed for, we always
    apply LA approximation to weights of the last layer of the network in an
    architecture-agnostic manner.
    last_layer_name : Name of the last layer in Backbone point estimate.
    empirical_bayes : whether to estimate the prior precision and observation
    noise using empirical Bayes after training or not.
    hessian_structure : set factorizations assumption that is used for Hessian
    matrix approximation. values: 'full', 'kron', 'diag' 'gp'
    method : method used for prior precision optimization.
    la_epochs : Number of epochs that is used for prior precision optimization.
    la_lr : Learning rate that is used for prior precision optimization.
    sigma_noise : observation noise prior.
    stat_noise : how to use observation noise, if False (default), value of
    sigma_noise is directly used as prior of observation noise, otherwise,
    based on statistical characteristics of remaining time, sigma_noise is 
    used as a coefficient to be multiplied by (y_std + 3*y_IQR)/4 .
    prior_precision : prior precision of a Gaussian prior (= weight decay)
    temperature : controls sharpness of posterior approximation.
    n_samples : Number of Monte Carlo samples.
    link_approx : Type of link approximation, we always apply mc (Monte Carlo).
    pred_type : We only use glm prediction in our setting (see Laplace library).
    report_path : Path to log important insights.
    result_path : Path to the directory in which important results are saved.
    split : Type of the split (holout/cv).
    fold : fold number that is used.
    seed : seed that is used in the experiment.
    device : device that is used in experiment.
    exp_id : an id to specify the experiment.
    """
    print(sigma_noise, prior_precision, temperature, empirical_bayes)
    torch.backends.cudnn.enabled = False
    # get current time (as start) to compute training time
    start=datetime.now()
    # Announcement for fitting Laplace approximation
    if split=='holdout':
        print(f'Fitting Laplace for experiment number: {exp_id}, {split} \
              data split.')
    else:
        print(f'Fitting Laplace for experiment number: {exp_id}, {split} \
              data split, fold: {fold}.')
    with open(report_path, 'w') as file:
        file.write('Configurations:\n')
        file.write(str(cfg))
        file.write('\n')
        file.write('\n')            
    
    # Load deterministic pre-trained point estimate or pretrained MVE model
    if split == 'holdout':
        model_checkpoint_path = os.path.join(
            result_path,
            'deterministic_{}_seed_{}_best_model.pt'.format(split, seed))
        # there are two different types that the model can be saved.
        la_path = os.path.join(
            result_path,
            'LA_{}_seed_{}_exp_{}_best_model.pt'.format(split, seed, exp_id)) 
        #la_path = os.path.join(result_path, 'LA_{}_seed_{}_exp_{}_best_model.bin'.format(split, seed, exp_id)) 
    else:
        model_checkpoint_path = os.path.join(
            result_path,
            'deterministic_{}_fold{}_seed_{}_best_model.pt'.format(
                split, fold, seed))
        # there are two different types that the model can be saved.
        la_path = os.path.join(
            result_path, 'LA_{}_fold{}_seed_{}_exp_{}_best_model.pt'.format(
                split, fold, seed, exp_id)) 
        #la_path = os.path.join(result_path, 'LA_{}_fold{}_seed_{}_exp_{}_best_model.bin'.format(split, fold, seed, exp_id))
        
    # check whether pre-trained model is available
    if not os.path.isfile(model_checkpoint_path):
        raise FileNotFoundError('Deterministic model must be trained first')
    else:
        # load the checkpoint except the last layer
        #checkpoint = torch.load(model_checkpoint_path)
        checkpoint = torch.load(model_checkpoint_path, map_location=cfg['device'])
        
        model.load_state_dict(checkpoint['model_state_dict'])

    if stat_noise:
        # set sigma noise based on the statistical propertis of target attribute
        y_std = np.std(y_train_val)
        y_IQR = np.percentile(y_train_val, 75) - np.percentile(y_train_val, 25)
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
    la = Laplace(
        model, likelihood='regression', subset_of_weights=subset_of_weights,
        hessian_structure=hessian_structure, sigma_noise=sigma_noise,
        prior_precision=prior_precision, temperature=temperature,
        **optional_args) 


    # Fit the local Laplace approximation at the parameters of the model.
    # i.e., approximate the posterior of model's weight around MAP estimates.
    # Laplace approximation (LA) is equivalent to fitting a Gaussian with:
        # Mean = MAP estimate
        # covariance matrix = negative inverse Hessian of the loss at the MAP
    # Which data to use? in order to fit the model and then optimizing prior
    # precision using 'marglik' method. we do not need a separate validation 
    # set, and thus we use all availabel data in training and validation set.
    la.fit(train_val_loader)
    # optimnize prior precision 
    if not empirical_bayes:
        # optimization using marglik method in Laplace library
        la.optimize_prior_precision(
            pred_type=pred_type, method=method, n_steps=la_epochs, lr=la_lr,
            init_prior_prec=prior_precision, n_samples=n_samples,
            link_approx=link_approx)
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
            #print(f'Epoch {i + 1}/{la_epochs},', f'NLL: {neg_marglik}, log prior: {log_prior}, log sigma: {log_sigma}')
            with open(report_path, 'a') as file:
                file.write(
                    'Epoch {}/{} NLL: {}, log prior: {}, log sigma: {}.\n'.format(
                        i+1, la_epochs, neg_marglik, log_prior, log_sigma))     
    
    #save the Laplace model
    torch.save(la, la_path, pickle_module=dill)
    
    training_time = (datetime.now()-start).total_seconds()
    print('Fitting and optimizing the Laplace approximation is done')
    with open(report_path, 'a') as file:
        file.write('#######################################################\n')
        file.write('Training time- in seconds: {}\n'.format(
            training_time)) 
        file.write('#######################################################\n')
        file.write('#######################################################\n')
    
    return la, la_path
    
    
def inference_laplace(la=None, cfg=None, model=None, val_mode=False, test_loader=None, 
                      test_original_lengths=None, y_train_val=None, 
                      y_scaler=None, normalization=False, 
                      subset_of_weights='last_layer', last_layer_name=None, 
                      hessian_structure=None,
                      sigma_noise=None, stat_noise=None, prior_precision=None,
                      temperature=None, pred_type='glm',
                      report_path=None, result_path=None, split='holdout',
                      fold=None, seed=None, device=None, exp_id=None): 
    """
    A method to fit a Laplace model, and optimize its preior precision.
    
    Parameters
    ----------
    la : a Laplace model fitted to data in a post-hoc fashion.
    model : point estimate neural network
    val_mode: Whether the method is called on validation set or not.
    test_loader : A Pytorch data loader containing all test examples. It can
    be applied for both vaidation, and test examples.
    test_original_lengths: list of prefix lengths of event prefixes in test set
    , it facilitates analysis w.r.t different lengths (i.e., earliness).  
    normalization : whether to apply inverse normalization durng inference
    y_scaler : Scaler that is used for inverse normaliation.
    subset_of_weights : type of weights that Hessian is computed for, we always
    apply LA approximation to weights of the last layer of the network in an
    architecture-agnostic manner.
    last_layer_name : Name of the last layer in Backbone point estimate.
    empirical_bayes : whether to estimate the prior precision and observation
    noise using empirical Bayes after training or not.
    hessian_structure : set factorizations assumption that is used for Hessian
    matrix approximation. values: 'full', 'kron', 'diag' 'gp'
    sigma_noise : observation noise prior.
    stat_noise : how to use observation noise, if False (default), value of
    sigma_noise is directly used as prior of observation noise, otherwise,
    based on statistical characteristics of remaining time, sigma_noise is 
    used as a coefficient to be multiplied by (y_std + 3*y_IQR)/4 .
    prior_precision : prior precision of a Gaussian prior (= weight decay)
    pred_type : We only use glm prediction in our setting (see Laplace library).
    report_path : Path to log important insights.
    result_path : Path to the directory in which important results are saved.
    split : Type of the split (holout/cv).
    fold : fold number that is used.
    seed : seed that is used in the experiment.
    device : device that is used in experiment.
    exp_id : an id to specify the experiment.
    """  
    torch.backends.cudnn.enabled = False
    # Bayesian Inference on validation/test set
    if split=='holdout':
        print(f'Start Bayesian inference for experiment number: {exp_id}, \
              {split} data split.')
    else:
        print(f'Start Bayesian inference for experiment number: {exp_id}, \
              {split} data split, fold: {fold}.')
              
    if la==None:
        if split=='holdout':
            la_path = os.path.join(
                result_path, 'LA_{}_seed_{}_exp_{}_best_model.pt'.format(
                split, seed, exp_id))
            model_checkpoint_path = os.path.join(
                result_path,
                'deterministic_{}_seed_{}_best_model.pt'.format(split, seed))
        else:
            la_path = os.path.join(
                result_path, 'LA_{}_fold{}_seed_{}_exp_{}_best_model.pt'.format(
                    split, fold, seed, exp_id))
            model_checkpoint_path = os.path.join(
                result_path,
                'deterministic_{}_fold{}_seed_{}_best_model.pt'.format(
                    split, fold, seed))
        if stat_noise:
            # set sigma noise based on the statistical propertis of target attribute
            y_std = np.std(y_train_val)
            y_IQR = np.percentile(y_train_val, 75) - np.percentile(y_train_val, 25)
            y_gold = (y_std + 3*y_IQR)/4
            sigma_noise=sigma_noise*y_gold

        #checkpoint = torch.load(model_checkpoint_path)
        checkpoint = torch.load(model_checkpoint_path, map_location=cfg['device'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optional_args = dict() #empty dict for optional args in Laplace model
        optional_args['last_layer_name'] = last_layer_name 
        la = Laplace(model, likelihood='regression',
                     subset_of_weights=subset_of_weights,
                     hessian_structure=hessian_structure,
                     sigma_noise=sigma_noise,
                     prior_precision=prior_precision,
                     temperature=temperature,
                     **optional_args) 
        la = torch.load(la_path)
    
    start=datetime.now()
    
    # empty dictionary to collect inference result in a dataframe
    # only capture Epistemic uncertainty by Laplace approximation
    all_results = {'GroundTruth': [], 'Prediction': [],
                   'Epistemic_Uncertainty': [],
                   'Aleatoric_Uncertainty':[], 
                   'Total_Uncertainty':[],
                   'Absolute_error': [],
                   'Absolute_percentage_error': []} 
    # on test set, prefix length is added for earliness analysis
    if not val_mode:
        all_results['Prefix_length'] = []
        
        
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
            #print(f_var)
            #print(la.sigma_noise.item()**2)
            epistemic_std = torch.sqrt(f_var)
            aleatoric_std = la.sigma_noise.item()
            # Compute square root element-wise 
            total_std = torch.sqrt(f_var + la.sigma_noise.item()**2)
            #f_std = torch.sqrt(f_var)
            # conduct inverse normalization if required
            if normalization:
                _y_truth = y_scaler * _y_truth
                _y_pred = y_scaler * _y_pred  
                epistemic_std = y_scaler * epistemic_std
                aleatoric_std = y_scaler * aleatoric_std
                total_std =  y_scaler * total_std
            # Compute batch loss
            #_y_pred = _y_pred.squeeze(dim=1)
            absolute_error += F.l1_loss(_y_pred, _y_truth).item()
            absolute_percentage_error += mape(_y_pred, _y_truth).item()
            # Detach predictions and ground truths (np arrays)
            _y_truth = _y_truth.detach().cpu().numpy()
            _y_pred = _y_pred.detach().cpu().numpy()
            mae_batch = np.abs(_y_truth - _y_pred)
            #mape_batch = (mae_batch/_y_truth*100)
            epsilon = 1e-8
            mape_batch = np.where(
                np.abs(_y_truth) > epsilon, 
                (mae_batch / (_y_truth + epsilon) * 100), 0)
            # collect inference result in all_result dict.
            all_results['GroundTruth'].extend(_y_truth.tolist())
            all_results['Prediction'].extend(_y_pred.tolist())
            if not val_mode:
                pre_lengths = test_original_lengths[length_idx:length_idx+batch_size]
                length_idx+=batch_size
                prefix_lengths = (np.array(pre_lengths).reshape(-1, 1)).tolist()
                all_results['Prefix_length'].extend(prefix_lengths)
            all_results['Absolute_error'].extend(mae_batch.tolist())
            all_results['Absolute_percentage_error'].extend(mape_batch.tolist())
            aleatoric_std_tensor = torch.full_like(epistemic_std, aleatoric_std)
            epistemic_std = epistemic_std.detach().cpu().numpy()
            aleatoric_std_tensor = aleatoric_std_tensor.detach().cpu().numpy()
            total_std = total_std.detach().cpu().numpy()
            all_results['Epistemic_Uncertainty'].extend(epistemic_std.tolist())
            all_results['Aleatoric_Uncertainty'].extend(aleatoric_std_tensor.tolist())
            all_results['Total_Uncertainty'].extend(total_std.tolist())
        
        num_test_batches = len(test_loader)    
        absolute_error /= num_test_batches    
        absolute_percentage_error /= num_test_batches
    print('Test - MAE: {:.3f}, '
                  'MAPE: {:.3f}'.format(
                      round(absolute_error, 3),
                      round(absolute_percentage_error, 3))) 
    inference_time = (datetime.now()-start).total_seconds()
    
    flattened_list = [item for sublist in all_results['GroundTruth']
                      for item in sublist]
    all_results['GroundTruth'] = flattened_list
    
    flattened_list = [item for sublist in all_results['Prediction'] 
                      for item in sublist]
    all_results['Prediction'] = flattened_list
    
    flattened_list = [item for sublist in all_results['Absolute_error'] 
                      for item in sublist]
    all_results['Absolute_error'] = flattened_list
    
    flattened_list = [item for sublist in all_results['Absolute_percentage_error'] 
                      for item in sublist]
    all_results['Absolute_percentage_error'] = flattened_list
    
    flattened_list = [item for sublist in all_results['Epistemic_Uncertainty'] 
                      for item in sublist]
    all_results['Epistemic_Uncertainty'] = flattened_list 
    
    flattened_list = [item for sublist in all_results['Aleatoric_Uncertainty'] 
                      for item in sublist]
    all_results['Aleatoric_Uncertainty'] = flattened_list 
    
    flattened_list = [item for sublist in all_results['Total_Uncertainty'] 
                      for item in sublist]
    all_results['Total_Uncertainty'] = flattened_list     

    
    if not val_mode:
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
        csv_filename = 'LA_{}_seed_{}_exp_{}_inference_result_.csv'.format(
            split, seed, exp_id)
    else:
        csv_filename = 'LA_{}_fold{}_seed_{}_exp_{}_inference_result_.csv'.format(
            split, fold, seed, exp_id)        
    if val_mode:
        csv_filename = add_suffix_to_csv(csv_filename, 
                                         added_suffix='validation_')
    csv_filepath = os.path.join(result_path, csv_filename)        
    results_df.to_csv(csv_filepath, index=False)
    print('inference is done')
    return csv_filepath