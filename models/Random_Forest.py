import os
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from utils.utils import add_suffix_to_csv

def fit_rf(model=None, cfg=None, val_loader=None, criterion=None, 
           n_estimators=None, depth_control=None, dataset_path=None, 
           result_path=None, y_val_path=None, report_path=None,      
           split=None, fold=None, seed=None, device=None,
           exp_id=None):
    
    # get current time (as start) to compute training time
    start=datetime.now()
    if split=='holdout':
        print(f'Fitting auxiliary Random Forest for experiment number: {exp_id},\
              {split} data split.')             
    else:
        print(f'Fitting auxiliary Random Forest for experiment number: {exp_id},\
              {split} data split, fold: {fold}.')
    with open(report_path, 'w') as file:
        file.write('Configurations:\n')
        file.write(str(cfg))
        file.write('\n')
        file.write('\n')  
    
    # paths to deterministic point estimate, auxiliary random forest model
    if split == 'holdout':
        deterministic_checkpoint_path = os.path.join(
            result_path,
            'deterministic_{}_seed_{}_best_model.pt'.format(split, seed))
        rf_path = os.path.join(
            result_path, 'RF_{}_seed_{}_exp_{}_best_model.pkl'.format(
                split, seed, exp_id)) 
    else:
        deterministic_checkpoint_path = os.path.join(
            result_path,
            'deterministic_{}_fold{}_seed_{}_best_model.pt'.format(
                split, fold, seed))
        rf_path = os.path.join(
            result_path, 'RF_{}_fold{}_seed_{}_exp_{}_best_model.pkl'.format(
                split, fold, seed, exp_id)) 
        
    # create a path for embedding
    y_val_file = os.path.basename(y_val_path)
    val_emb_filename = y_val_file.replace('_y_val_', '_X_val_emb_')
    val_emb_path = os.path.join(dataset_path, val_emb_filename)     

    # check whether embedding tensor is already available
    if not os.path.isfile(val_emb_path):  
        print('Get the embedding of validation set using pre-trained model.') 
        # check deterministic pre-trained model is available
        if not os.path.isfile(deterministic_checkpoint_path):
            raise FileNotFoundError('Deterministic model must be trained first')
        else:
            # load the checkpoint except the last layer
            checkpoint = torch.load(deterministic_checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        # set deterministic point estimate to evaluation mode
        model.eval()
        # list to collect all embeddings
        all_embeddings = []
        with torch.no_grad():
            for index, val_batch in enumerate(val_loader):
                # Get the embeddings for the current batch
                inputs = val_batch[0].to(device)
                batch_embedding = model(inputs)
                # Move to CPU and add to list to save memory on GPU
                all_embeddings.append(batch_embedding.cpu()) 
        # Concatenate all embeddings into a single tensor
        X_val_emb = torch.cat(all_embeddings, dim=0)
        # save the embedding for validation set
        torch.save(X_val_emb, val_emb_path)
    else:
        print('Embedding for validation set is already created.')
        X_val_emb = torch.load(val_emb_path)
    
    # load remaining time for validation set
    y_val = torch.load(y_val_path)    
    # Convert PyTorch tensors to NumPy arrays
    X_val_emb = X_val_emb.cpu().numpy() if X_val_emb.is_cuda else X_val_emb.numpy()
    y_val = y_val.cpu().numpy() if y_val.is_cuda else y_val.numpy()
    
    if depth_control:
        max_depth = cfg.get('uncertainty').get('union').get('max_depth')
    else:
        max_depth = None
    min_samples_split = (
        cfg.get('uncertainty').get('union').get('min_samples_split'))
    min_samples_leaf = (
        cfg.get('uncertainty').get('union').get('min_samples_leaf'))  
    
    # define a Random Forest Regressor to work on embeddings
    aux_model = RandomForestRegressor(
        n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
        random_state=seed) 
    print('now: fit a random forest to predict remaining time based on embeddings')
    aux_model.fit(X_val_emb, y_val) 
    with open(rf_path, 'wb') as file:
        pickle.dump(aux_model, file)
    training_time = (datetime.now()-start).total_seconds()
    with open(report_path, 'a') as file:
        file.write('Training time- in seconds: {}\n'.format(
            training_time)) 
        file.write('#######################################################\n')
        file.write('#######################################################\n')
    
    return model, aux_model, rf_path

def predict_rf(model=None, model_arch=None, aux_model=None, val_mode=False, 
               val_loader=None, test_loader=None, test_original_lengths=None, 
               y_scaler=None, normalization=False, report_path=None, 
               result_path=None, split=None, fold=None, seed=None, device=None,
               exp_id=None, experiment=None, cfg=None, dataset_path=None,
               y_val_path=None):
    
    if split=='holdout':
        print(f'Now: start inference experiment number: {exp_id}, \
              data split: {split}.')
    else:
        print(f'Now: start inference experiment number: {exp_id}, \
              data split: {split} ,  fold: {fold}.')    
        
    #create a dictionary to collect inference results
    all_results = {'GroundTruth': [], 'Prediction': [],
                   'Epistemic_Uncertainty': [], 'Absolute_error': [],
                   'Absolute_percentage_error': []} 
    # on test set, prefix length is added for earliness analysis
    if not val_mode:
        all_results['Prefix_length'] = []
        # saving a fitted random forest might not work properly using
        # pickle. Therefore, we fit our randome forest with best hyperparameters
        # yet another time.
        print('Fit randome forest with best hyperparameters yet for another time')
        print('best experiment is:', experiment)
        criterion = experiment.get('criterion')
        n_estimators = experiment.get('n_estimators')
        depth_control = experiment.get('depth_control') 
        model, aux_model, _ = fit_rf(
            model=model_arch, cfg=cfg, val_loader=val_loader, criterion=criterion,
            n_estimators=n_estimators, depth_control=depth_control,
            dataset_path=dataset_path, result_path=result_path,
            y_val_path=y_val_path, report_path=report_path,
            split=split, fold=fold, seed=seed, device=device, exp_id=exp_id)

    start=datetime.now()       
    
    # set variabls to zero to collect loss values and length ids
    absolute_error = 0
    absolute_percentage_error = 0
    length_idx = 0
    # set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        for index, test_batch in enumerate(test_loader):
            inputs = test_batch[0].to(device)
            _y_truth = test_batch[1].detach().cpu().numpy()
            batch_size = inputs.shape[0]
            batch_embedding = model(inputs)
            batch_embedding = (batch_embedding.cpu().numpy() 
                               if batch_embedding.is_cuda 
                               else batch_embedding.numpy())
            # Get predictions from each individual tree in the random forest
            tree_pred = np.array([tree.predict(batch_embedding) 
                                  for tree in aux_model.estimators_])
            # Compute the mean prediction across all trees
            _y_pred = np.mean(tree_pred, axis=0)
            # Compute standard devition of predictions (epistemic uncertainty)
            epistemic_std = np.std(tree_pred, axis=0)
            # normalize tragets, outputs, epistemic uncertainty (if necessary)
            if normalization:                
                _y_truth = y_scaler * _y_truth
                _y_pred = y_scaler * _y_pred
                epistemic_std = y_scaler * epistemic_std
            mae_batch = np.abs(_y_truth - _y_pred)
            mape_batch = (mae_batch/_y_truth*100)
            # collect inference result in all_result dict.
            all_results['GroundTruth'].extend(_y_truth.tolist())
            all_results['Prediction'].extend(_y_pred.tolist())
            if not val_mode:
                pre_lengths = (
                    test_original_lengths[length_idx:length_idx+batch_size])
                length_idx+=batch_size
                prefix_lengths = (np.array(pre_lengths).reshape(-1, 1)).tolist()
                all_results['Prefix_length'].extend(prefix_lengths)
            all_results['Absolute_error'].extend(mae_batch.tolist())
            all_results['Absolute_percentage_error'].extend(mape_batch.tolist())
            all_results['Epistemic_Uncertainty'].extend(epistemic_std.tolist())
            # update loss values
            absolute_error += np.sum(mae_batch)
            absolute_percentage_error += np.sum(mape_batch)
        
        if not val_mode:    
            absolute_error /= len(test_original_lengths)    
            absolute_percentage_error /= len(test_original_lengths)
        else:  
            num_test_batches = len(test_loader) 
            eval_batch_size = cfg.get('evaluation').get('batch_size')
            num_test_instances = num_test_batches * eval_batch_size
            absolute_error /= num_test_instances    
            absolute_percentage_error /= num_test_instances
        
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
        
        for key, value in all_results.items():
            print(f"Key: {key}, Length: {len(value)}")
    
    results_df = pd.DataFrame(all_results)
    
    if split=='holdout':
        csv_filename = 'RF_{}_seed_{}_exp_{}_inference_result_.csv'.format(
            split,seed, exp_id)
    else:
        csv_filename = 'RF_{}_fold{}_seed_{}_exp_{}_inference_result_.csv'.format(
            split, fold, seed, exp_id)
    if val_mode:
        csv_filename = add_suffix_to_csv(csv_filename, 
                                         added_suffix='validation_')
    csv_filepath = os.path.join(result_path, csv_filename)        
    results_df.to_csv(csv_filepath, index=False)  

    return csv_filepath