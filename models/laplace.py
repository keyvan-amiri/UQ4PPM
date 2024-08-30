from laplace import Laplace
from laplace.curvature import BackPackGGN # backend for Hessian computations
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from itertools import chain
from utils.loss_functions import mape


# Post-hoc Laplace approximation: optimizing the prior, and then inference
def post_hoc_laplace(net, model, train_loader, val_loader, test_loader,
                     test_original_lengths, max_train_val=None, device=None,
                     cfg=None, split_mode=None, fold_idx=None, folds=None,
                     dataset_name=None, test_result_folder=None):
    
    print(f'Post-hoc Laplace- fold: {fold_idx} from {folds} folds is started')
    
    optional_args = dict() #empty dict for optional args in Laplace model
    normalization =  cfg.get('data').get('normalization')
    hessian_structure = cfg.get('uncertainty').get(
        'laplace').get('hessian_structure')
    subset_of_weights = cfg.get('uncertainty').get(
        'laplace').get('subset_of_weights')
    pred_type = cfg.get('uncertainty').get('laplace').get('pred_type')    
    link_approx = cfg.get('uncertainty').get('laplace').get('link_approx')
    n_samples = cfg.get('uncertainty').get('laplace').get('n_samples')
    optimize_prior_precision = cfg.get(
        'uncertainty').get('laplace').get('optimize_prior_precision')
    prior_precision = cfg.get(
        'uncertainty').get('laplace').get('prior_precision')
    empirical_bayes= cfg.get(
        'uncertainty').get('laplace').get('empirical_bayes')

    model = model.to(device)
    model.train()  
    
       
    # define the Laplace model, and conduct post-hoc fitting
    la = Laplace(model, likelihood='regression',
                 subset_of_weights=subset_of_weights,
                 hessian_structure=hessian_structure,
                 backend=BackPackGGN, **optional_args) #subnetwork_indices=subnetwork_indices,
    la.fit(train_loader)
    
    # optimnize prior precision
    if empirical_bayes:
        # TODO: add a option for sigma noise (the line after the next line)
        #log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
        log_prior, log_sigma = torch.ones(1, requires_grad=True),\
            torch.full((1,), 0.2, requires_grad=True, dtype=torch.float32)
        #log_prior = torch.ones(1, requires_grad=True) 
        #log_sigma = torch.ones(1, requires_grad=True)        
        hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
        for i in range(100):
            hyper_optimizer.zero_grad()
            neg_marglik = - la.log_marginal_likelihood(log_prior.exp(),
                                                       log_sigma.exp())
            neg_marglik.backward()
            hyper_optimizer.step()
    else:
        if pred_type == 'glm':
            la.optimize_prior_precision(pred_type=pred_type,
                                        method=optimize_prior_precision,
                                        init_prior_prec=prior_precision,
                                        val_loader=val_loader)
    print('Fitting and optimizing the Laplace approximation is done')
    
    # Bayesian predictive
    print('Now start Bayesian predictive inference')
    print(f'Bayesian predictive for fold: {fold_idx} from {folds} folds')
    all_results = {'GroundTruth': [],
                   'Mean_pred': [],
                   'std_pred': [],
                   'Prefix_length': [], 'Absolute_error': [],
                   'Absolute_percentage_error': [], 'Squared_error': []}
    absolute_error = 0
    squared_error = 0
    absolute_percentage_error = 0
    length_idx = 0
    with torch.no_grad():
        for index, test_batch in enumerate(test_loader):
            inputs = test_batch[0].to(device)
            targets = test_batch[1].to(device)
            batch_size = inputs.shape[0]
            prediction_mean, f_var = la(inputs, pred_type=pred_type)                   
            # Remove the dimension of size 1 along axis 2
            f_var = f_var.squeeze(dim=2) 
            # Compute square root element-wise 
            f_std = torch.sqrt(f_var) 
                        
            #Compute overall loss for the test batch
            absolute_error += F.l1_loss(prediction_mean, targets).item()
            squared_error += F.mse_loss(prediction_mean, targets).item()  
            absolute_percentage_error += mape(prediction_mean, targets).item()
            # detach prediction mean and convert to list           
            prediction_mean = prediction_mean.detach().cpu().numpy()
            #compute standard deviation for predictions
            f_std = f_std.detach().cpu().numpy()
            prediction_std = np.sqrt(f_std**2 + la.sigma_noise.item()**2) 
            # get prefix loss values
            mae_batch = np.abs(targets.detach().cpu().numpy()- prediction_mean)
            mse_batch = ((targets.detach().cpu().numpy() - prediction_mean) ** 2)
            mape_batch = (mae_batch/targets.detach().cpu().numpy()*100)   
            #predictions, ground-truth & prefix lengths: prefix-level inference  
            if normalization and net == 'dalstm':
                targets *= max_train_val
                prediction_mean *= max_train_val
                prediction_std *= max_train_val
                mae_batch *= max_train_val
                mse_batch *= max_train_val ** 2
                
            all_results['GroundTruth'].extend(targets.detach().cpu().numpy().tolist())                               
            all_results['Mean_pred'].extend(prediction_mean.tolist())
            all_results['std_pred'].extend(prediction_std.tolist())
            pre_lengths = test_original_lengths[length_idx:length_idx+batch_size]
            length_idx+=batch_size
            prefix_lengths = (np.array(pre_lengths).reshape(-1, 1)).tolist()
            all_results['Prefix_length'].extend(prefix_lengths)
            all_results['Absolute_error'].extend(mae_batch.tolist())
            all_results['Absolute_percentage_error'].extend(mape_batch.tolist())
            all_results['Squared_error'].extend(mse_batch.tolist()) 
       
        num_test_batches = len(test_loader)
        absolute_error /= num_test_batches
        squared_error /= num_test_batches
        rmse = squared_error ** 0.5
        absolute_percentage_error /= num_test_batches
        if normalization:
            absolute_error *= max_train_val
            rmse *= max_train_val
    print('Test - MAE: {:.3f}, '
              'RMSE: {:.3f}, '
              'MAPE: {:.3f}'.format(
                  round(absolute_error, 3),
                  round(rmse, 3),
                  round(absolute_percentage_error, 3))) 
    all_results_flat = {key: list(chain.from_iterable(value)) 
                        for key, value in all_results.items()}
    #for key, value in all_results_flat.items():
        #print(key, len(value), value[:5])
    results_df = pd.DataFrame(all_results_flat)  
    results_df.to_csv(csv_filename, index=False) 


