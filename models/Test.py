import os
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from loss.mape import mape
from utils.utils import add_suffix_to_csv

# function to handle inference with trained model(s)
def test_model(model=None, models=None, uq_method=None, num_mc_samples=None,
               test_loader=None, test_original_lengths=None, y_scaler=None, 
               processed_data_path=None, report_path=None, val_mode=False,
               data_split='holdout', fold=None, seed=None, device=None,
               normalization=False, ensemble_mode=False, ensemble_size=None,
               sqr_q=None, sqr_factor=None, exp_id=None, deterministic=False,
               std_mean_ratio=None): 
    """
    This function is used for the following UQ techniques:
        1) Deterministic point estimate.
        2) Dropout approximation
        3) Heteroscedastic regression
        combinations of 2,3
        4) Bootstrapping Ensembles 
    """       
    start=datetime.now()
    if data_split=='holdout':
        print(f'Now: start inference experiment number: {exp_id}, \
              data split: {data_split}.')
        # define a list of checkpoint paths if we have ensemble mode
        if ensemble_mode:
            checkpoint_path_list = []
            for model_idx in range(1, ensemble_size+1):
                checkpoint_path = os.path.join(
                    processed_data_path,
                    '{}_{}_seed_{}_exp_{}_member_{}_best_model.pt'.format(
                        uq_method, data_split, seed, exp_id, model_idx))
                checkpoint_path_list.append(checkpoint_path)
        # otherwise: define a single checkpoint path
        else:                
            checkpoint_path = os.path.join(
                processed_data_path,
                '{}_{}_seed_{}_exp_{}_best_model.pt'.format(
                    uq_method, data_split, seed, exp_id)) 
    else:
        print(f'Now: start inference experiment number: {exp_id}, \
              data split: {data_split} ,  fold: {fold}.')
        # define a list of checkpoint paths if we have ensemble mode
        if ensemble_mode:
            checkpoint_path_list = []
            for model_idx in range(1, ensemble_size+1):
                checkpoint_path = os.path.join(
                    processed_data_path,
                    '{}_{}_fold{}_seed_{}_exp_{}_member_{}_best_model.pt'.format(
                        uq_method, data_split, fold, seed, exp_id, model_idx))
                checkpoint_path_list.append(checkpoint_path)
        # otherwise: define a single checkpoint path
        else:       
            checkpoint_path = os.path.join(
                processed_data_path,
                '{}_{}_fold{}_seed_{}_exp_{}_best_model.pt'.format(
                    uq_method, data_split, fold, seed, exp_id)) 
        
    # define the structure of result dataframe based on UQ method   
    if uq_method == 'deterministic':
        all_results = {'GroundTruth': [], 'Prediction': [], 
                       'Absolute_error': [], 'Absolute_percentage_error': []}
    # UQ methods capturing Epistemic Uncertainty
    elif (uq_method == 'DA' or uq_method == 'CDA' or uq_method == 'H' or 
          uq_method == 'BE+H' or uq_method == 'DA+H' or uq_method == 'CDA+H'):
        all_results = {'GroundTruth': [], 'Prediction': [],
                       'Epistemic_Uncertainty': [], 'Aleatoric_Uncertainty': [],
                       'Total_Uncertainty': [], 'Absolute_error': [],
                       'Absolute_percentage_error': []} 
    # on test set, prefix length is added for earliness analysis
    if not val_mode:
        all_results['Prefix_length'] = []
    
    # set variabls to zero to collect loss values and length ids
    absolute_error = 0
    absolute_percentage_error = 0
    length_idx = 0
    
    # load checkpoint(s) and set model(s) to evaluation mode
    if ensemble_mode:
        for model_idx in range(1, ensemble_size+1):
            checkpoint = torch.load(checkpoint_path_list[model_idx-1])
            models[model_idx-1].load_state_dict(checkpoint['model_state_dict'])
            models[model_idx-1].eval()
    else: 
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    
    with torch.no_grad():
        for index, test_batch in enumerate(test_loader):
            inputs = test_batch[0].to(device)
            _y_truth = test_batch[1].to(device)
            batch_size = inputs.shape[0]            
            # get model outputs, and uncertainties if required
            if uq_method == 'deterministic':            
                _y_pred = model(inputs)
                # for numerical stability
                # to ensure predictions are positive (necessary for calibration)
                epsilon = 1e-8
                _y_pred = torch.maximum(_y_pred, torch.tensor(epsilon))                 
            elif (uq_method == 'DA' or uq_method == 'CDA' or
                  uq_method == 'DA+H' or uq_method == 'CDA+H'):
                means_list, logvar_list =[], []
                # conduct Monte Carlo sampling
                for i in range (num_mc_samples): 
                    mean, log_var,_ = model(inputs, stop_dropout=False)
                    means_list.append(mean)
                    logvar_list.append(log_var)
                # Aggregate the results for all samples
                # Compute point estimation and uncertainty
                stacked_means = torch.stack(means_list, dim=0)
                # predited value is the average for all samples
                _y_pred = torch.mean(stacked_means, dim=0)
                # epistemic uncertainty obtained from std for all samples
                epistemic_std = torch.std(stacked_means, dim=0).to(device)
                # normalize epistemic uncertainty if necessary
                if normalization:
                    epistemic_std = y_scaler * epistemic_std
                # now obtain aleatoric uncertainty
                if (uq_method == 'DA+H' or uq_method == 'CDA+H'):
                    stacked_log_var = torch.stack(logvar_list, dim=0)
                    stacked_var = torch.exp(stacked_log_var)
                    mean_var = torch.mean(stacked_var, dim=0)
                    aleatoric_std = torch.sqrt(mean_var).to(device)
                    # normalize aleatoric uncertainty if necessary
                    if normalization:
                        aleatoric_std = y_scaler * aleatoric_std
                else:
                    aleatoric_std = torch.zeros_like(epistemic_std)
                total_std = torch.sqrt(epistemic_std**2 + aleatoric_std**2)
            elif uq_method == 'H':
                _y_pred, log_var = model(inputs)
                aleatoric_std = torch.sqrt(torch.exp(log_var))
                # normalize aleatoric uncertainty if necessary
                if normalization:
                    aleatoric_std = y_scaler * aleatoric_std 
                epistemic_std = torch.zeros_like(aleatoric_std)
                total_std = torch.sqrt(epistemic_std**2 + aleatoric_std**2)
            elif uq_method == 'BE+H':
                # collect prediction means & aleatoric std: all ensemble members
                mean_pred_list, aleatoric_std_list = [], []
                for model_idx in range(1, ensemble_size+1):
                    member_mean, member_log_var = models[model_idx-1](inputs)
                    member_aleatoric_std = torch.sqrt(torch.exp(member_log_var))
                    mean_pred_list.append(member_mean)
                    aleatoric_std_list.append(member_aleatoric_std)
                stacked_mean_pred = torch.stack(mean_pred_list, dim=0)
                stacked_aleatoric = torch.stack(aleatoric_std_list, dim=0)
                # predited value is the average of predictions of all members
                _y_pred = torch.mean(stacked_mean_pred, dim=0)           
                # first part of aleatoric uncertainty = std of predictions of all members
                aleatoric_std1 = torch.std(stacked_mean_pred, dim=0).to(device)
                # second part of aleatoric uncertainty = mean of aleatoric estimates of all members
                aleatoric_std2 = torch.mean(stacked_aleatoric, dim=0)
                # normalize uncertainties if necessary
                if normalization:
                    aleatoric_std1 = y_scaler * aleatoric_std1
                    aleatoric_std2 = y_scaler * aleatoric_std2
                aleatoric_std = torch.sqrt(aleatoric_std1**2 + aleatoric_std2**2)
                epistemic_std = torch.zeros_like(aleatoric_std)
                total_std = torch.sqrt(epistemic_std**2 + aleatoric_std**2)
           
            # convert tragets, outputs in case of normalization
            if normalization:
                _y_truth = y_scaler * _y_truth
                _y_pred = y_scaler * _y_pred  

            # Compute batch loss
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
            # for test set we collect prefix lengths
            if not val_mode:
                pre_lengths = test_original_lengths[
                    length_idx:length_idx+batch_size]
                length_idx+=batch_size
                prefix_lengths = (np.array(pre_lengths).reshape(-1, 1)).tolist()
                all_results['Prefix_length'].extend(prefix_lengths)
            all_results['Absolute_error'].extend(mae_batch.tolist())
            all_results['Absolute_percentage_error'].extend(mape_batch.tolist()) 
            
            # set uncertainty columns based on UQ method
            if (uq_method == 'DA' or uq_method == 'CDA' or uq_method == 'H' or
                uq_method == 'DA+H' or uq_method == 'CDA+H' or 
                uq_method == 'BE+H'):
                epistemic_std = epistemic_std.detach().cpu().numpy()
                aleatoric_std = aleatoric_std.detach().cpu().numpy()
                total_std = total_std.detach().cpu().numpy()                
                all_results['Epistemic_Uncertainty'].extend(epistemic_std.tolist())
                all_results['Aleatoric_Uncertainty'].extend(aleatoric_std.tolist())
                all_results['Total_Uncertainty'].extend(total_std.tolist()) 
        num_test_batches = len(test_loader)    
        absolute_error /= num_test_batches    
        absolute_percentage_error /= num_test_batches
    print('Test - MAE: {:.3f}, '
                  'MAPE: {:.3f}'.format(
                      round(absolute_error, 3),
                      round(absolute_percentage_error, 3))) 
    inference_time = (datetime.now()-start).total_seconds() 
    
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
    if deterministic:
        results_df['Total_Uncertainty']=results_df['Prediction']*std_mean_ratio
        
    if data_split=='holdout':
        csv_filename = '{}_{}_seed_{}_exp_{}_inference_result_.csv'.format(
            uq_method,data_split,seed, exp_id)
    else:
        csv_filename = '{}_{}_fold{}_seed_{}_exp_{}_inference_result_.csv'.format(
            uq_method, data_split, fold, seed, exp_id)
    if val_mode:
        csv_filename = add_suffix_to_csv(csv_filename, added_suffix='validation_')
    csv_filepath = os.path.join(processed_data_path, csv_filename)        
    results_df.to_csv(csv_filepath, index=False)
    return csv_filepath