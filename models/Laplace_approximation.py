"""
To build this python script we use Laplace library:
    https://github.com/aleximmer/Laplace
    Introduced by:
    Laplace Redux – Effortless Bayesian Deep Learning
    Erik Daxberger, Agustinus Kristiadi, Alexander Immer,
    Runa Eschenhagen, Matthias Bauerd, Philipp Hennig.
"""
import warnings
warnings.filterwarnings(
    "ignore", 
    message="You are using `torch.load` with `weights_only=False`", 
    category=FutureWarning
)
import os
import dill
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
import torch.nn.functional as F
from loss.mape import mape
from laplace import Laplace
from laplace import (LLLaplace, FullLLLaplace, KronLLLaplace, DiagLLLaplace)
from utils.eval_cal_utils import (get_validation_data_and_model_size,
                                  replace_suffix)
from models.dalstm import DALSTMModel





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
    """
    A method to fit a Laplace model, and optimize its preior precision, and
    finally, conduct Bayesian predictive inference with fitted model.
    
    Parameters
    ----------
    model : Backbone model which provides MAP estimates (point estimate)
    cfg : Congiguration file that is used for training, and inference
    X_train_path, X_val_path, X_test_path : Path to the feature vectors of
    event prefixes, previously generated through pre-processing script.
    y_train_path, y_val_path, y_test_path : Path to target attribute.
    test_original_lengths: list of prefix lengths of event prefixes in test set
    , it facilitates analysis w.r.t different lengths (i.e., earliness).
    max_len : Maximum legth of feature vectors that are fed into DALSTM.
    y_scaler : Scaler that is used for inverse normaliation.
    normalization : whether to apply inverse normalization durng inference
    subset_of_weights : type of weights that Hessian is computed for, we always
    apply LA approximation to weights of the last layer of the network in an
    architecture=agnostic manner.
    hessian_structure : set factorizations assumption that is used for Hessian
    matrix approximation. values: 'full', 'kron', 'diag' 'gp'
    empirical_bayes : whether to estimate the prior precision and observation
    noise using empirical Bayes after training or not.
    method : method used for prior precision optimization.
    values: 'marglik', 'gridsearch'
    grid_size : number of values to consider inside the gridsearch interval.
    last_layer_name : Name of the last layer in Backbone point estimate.
    sigma_noise : observation noise prior
    stat_noise : how to use observation noise, if False (default), value of
    sigma_noise is directly used as prior of observation noise, otherwise,
    based on statistical characteristics of remaining time, sigma_noise is 
    used as a coefficient to be multiplied by (y_std + 3*y_IQR)/4 .
    prior_precision : prior precision of a Gaussian prior (= weight decay)
    temperature : controls sharpness of posterior approximation
    n_samples : Number of Monte Carlo samples 
    link_approx : Type of link approximation, we always apply mc (Monte Carlo)
    pred_type : We only use glm prediction in our setting (see Laplace library)
    la_epochs : Number of epochs that is used for prior precision optimization.
    la_lr : Learning rate that is used for prior precision optimization.
    report_path : Path to log important insights.
    result_path : Path to the directory in which important results are saved.
    split : Type of the split (holout/cv)
    fold : fold number that is used
    seed : seed that is used in the experiment.
    device : device that is used in experiment.
    """
    torch.backends.cudnn.enabled = False
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
    # TODO: check type of the save and decide on following line:
    #la_name = la_name.replace('.pt', '.bin')
    la_path = os.path.join(result_path, la_name) 
    
    # load feature vectors for train, val, test sets
    X_train = torch.load(X_train_path)
    X_val = torch.load(X_val_path)
    X_test = torch.load(X_test_path)
    y_train = torch.load(y_train_path)
    y_val = torch.load(y_val_path)
    y_test = torch.load(y_test_path)
    y_train = y_train.unsqueeze(dim=1)
    y_val = y_val.unsqueeze(dim=1)
    y_test = y_test.unsqueeze(dim=1)
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
    """
    la = FullLLLaplace(model, likelihood='regression',
                 sigma_noise=sigma_noise,
                 prior_precision=prior_precision,
                 temperature=temperature,
                 **optional_args) 
    """

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
    if not empirical_bayes:
        if method == 'gridsearch':
            la.optimize_prior_precision(pred_type=pred_type,
                                        method=method,
                                        n_steps=la_epochs,
                                        lr=la_lr,
                                        init_prior_prec=prior_precision,
                                        val_loader=val_loader,
                                        grid_size=grid_size)
        else:
            # optimization using marglik method in Laplace library
            la.optimize_prior_precision(pred_type=pred_type,
                                        method=method,
                                        n_steps=la_epochs,
                                        lr=la_lr,
                                        init_prior_prec=prior_precision,
                                        n_samples=n_samples,
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
            print(f'Epoch {i + 1}/{la_epochs},', 
                  f'NLL: {neg_marglik}, log prior: {log_prior}, log sigma: {log_sigma}')
            with open(report_path, 'a') as file:
                file.write(
                    'Epoch {}/{} NLL: {}, log prior: {}, log sigma: {}.\n'.format(
                        i+1, la_epochs, neg_marglik, log_prior, log_sigma))     
    
    #save the Laplace model
    # TODO: Resolve serialization problem and decide on the following!
    #print(f"Type of 'la': {type(la)}")
    #print(f"Methods of 'la': {dir(la)}")
    #print(la_path)
    #print(la.state_dict()) 
    #torch.save(la.state_dict(), la_path)
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
            #_y_pred = _y_pred.squeeze(dim=1)
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
    results_df = pd.DataFrame(all_results)
    if split=='holdout':
        csv_filename = os.path.join(
            result_path,'LA_{}_seed_{}_inference_result_.csv'.format(split,seed))
    else:
        csv_filename = os.path.join(
            result_path,'LA_{}_fold{}_seed_{}_inference_result_.csv'.format(
                split, fold, seed))         
    results_df.to_csv(csv_filename, index=False)
    
    
def inf_val_laplace(args=None, cfg=None, device=None, 
                    result_path=None, root_path=None,
                    val_inference_path=None, report_path=None):
    
    torch.backends.cudnn.enabled = False
    print('Now: start inference on validation set:')
    start=datetime.now()   
    # get normalization mode
    normalization = cfg.get('data').get('normalization') 
    # get important model dimension, and hyper-parameters
    n_layers = cfg.get('model').get('lstm').get('n_layers')
    hidden_size = cfg.get('model').get('lstm').get('hidden_size')
    dropout = cfg.get('model').get('lstm').get('dropout')
    dropout_prob = cfg.get('model').get('lstm').get('dropout_prob')
    pred_type = 'glm' # define prediction type
    subset_of_weights= 'last_layer' # define name of last layer in Backbone
    last_layer_name = cfg.get('uncertainty').get('laplace').get(
        'last_layer_name')
    hessian_structure = cfg.get('uncertainty').get(
        'laplace').get('hessian_structure')
    sigma_noise = cfg.get('uncertainty').get('laplace').get(
        'sigma_noise')
    prior_precision = cfg.get('uncertainty').get('laplace').get(
        'prior_precision')
    temperature= cfg.get('uncertainty').get('laplace').get(
        'temperature')
    
    # get calibration loader (i.e., validation loader), and some others!
    (calibration_loader, input_size, max_len, y_scaler, _, _
     ) = get_validation_data_and_model_size(args=args, cfg=cfg, 
                                            root_path=root_path)
    # set name and path for checkpoints: Laplace model + Backbone point estimate
    laplace_model_name = replace_suffix(
        args.csv_file, 'inference_result_.csv', 'best_model.pt')
    parts = laplace_model_name.split('_', 1)
    deterministic_checkpoint_name = 'deterministic_' + parts[1]
    deterministic_checkpoint_path = os.path.join(result_path,
                                                 deterministic_checkpoint_name)
    # define the Backbone model, and load its state_dict
    model = DALSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        n_layers=n_layers,
        max_len=max_len,
        dropout=dropout,
        p_fix=dropout_prob,
        return_squeezed=False).to(device)
    model.load_state_dict(torch.load(deterministic_checkpoint_path), strict=False)
    
    # load fitted and optimized Laplace model
    optional_args = dict() #empty dict for optional args in Laplace model
    optional_args['last_layer_name'] = last_layer_name 
    la = Laplace(model, likelihood='regression',
                 subset_of_weights=subset_of_weights,
                 hessian_structure=hessian_structure,
                 sigma_noise=sigma_noise,
                 prior_precision=prior_precision,
                 temperature=temperature,
                 **optional_args) 
    
    laplace_model_path = os.path.join(result_path, laplace_model_name)
    la = torch.load(laplace_model_path)
    # TODO: resolve serialization error!
    #la.load_state_dict(torch.load(laplace_model_path))

    # empty dict to collct predictios on validation (i.e., calibration) set
    res_dict = {'GroundTruth': [], 'Prediction': [], 'Epistemic_Uncertainty': []}
    
    with torch.no_grad():
        # get model prediction and epistemic uncertainty
        for index, cal_batch in enumerate(calibration_loader):
            inputs = cal_batch[0].to(device)
            _y_truth = cal_batch[1].to(device)
            _y_pred, f_var = la(inputs, pred_type=pred_type) 
            # Remove the dimension of size 1 along axis 2
            f_var = f_var.squeeze(dim=2) 
            # Compute square root element-wise 
            epistemic_std = torch.sqrt(f_var + la.sigma_noise.item()**2)
            # conduct inverse normalization if required
            if normalization:
                _y_truth = y_scaler * _y_truth
                _y_pred = y_scaler * _y_pred  
                epistemic_std = y_scaler * epistemic_std
            # Detach predictions and ground truths (np arrays)
            _y_truth = _y_truth.detach().cpu().numpy()
            _y_pred = _y_pred.detach().cpu().numpy()
            # collect inference result in all_result dict.
            res_dict['GroundTruth'].extend(_y_truth.tolist())
            res_dict['Prediction'].extend(_y_pred.tolist())
            epistemic_std = epistemic_std.detach().cpu().numpy()
            res_dict['Epistemic_Uncertainty'].extend(
                epistemic_std.tolist())

    inference_val_time = (datetime.now()-start).total_seconds()
    with open(report_path, 'w') as file:
        file.write('Inference on validation set took  {} seconds. \n'.format(
            inference_val_time)) 
    """        
    flattened_list = [item for sublist in res_dict['GroundTruth'] 
                      for item in sublist]
    res_dict['GroundTruth'] = flattened_list
    """
    flattened_list = [item for sublist in res_dict['Prediction'] 
                      for item in sublist]
    res_dict['Prediction'] = flattened_list
    flattened_list = [item for sublist in res_dict['Epistemic_Uncertainty'] 
                      for item in sublist]
    res_dict['Epistemic_Uncertainty'] = flattened_list    
    calibration_df = pd.DataFrame(res_dict)      
    calibration_df.to_csv(val_inference_path, index=False)

    return calibration_df     
                                                        