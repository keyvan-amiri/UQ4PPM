import torch
import numpy as np
import random
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_
import pandas as pd
import os
from datetime import datetime
import torch.optim as optim
from loss.mape import mape

##############################################################################
# Utlility functions for training and inference
##############################################################################

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
# function to set the optimizer object
def set_optimizer (model, optimizer_type, base_lr, eps, weight_decay):
    if optimizer_type == 'NAdam':
        optimizer = optim.NAdam(model.parameters(), lr=base_lr, eps=eps,
                                weight_decay=weight_decay)
    elif optimizer_type == 'AdamW':   
        optimizer = optim.AdamW(model.parameters(), lr=base_lr, eps=eps,
                                weight_decay=weight_decay)
    elif optimizer_type == 'Adam':   
        optimizer = optim.Adam(model.parameters(), lr=base_lr, eps=eps,
                               weight_decay=weight_decay)         
    return optimizer

# function to handle training the model
def train_model(model=None, train_loader=None, val_loader=None, criterion=None,
                optimizer=None, scheduler=None, device=None, num_epochs=None,
                early_patience=None, min_delta=None, clip_grad_norm=None,
                clip_value=None, processed_data_path=None, data_split=None,
                seed=None):
    print('Now start training for {} data slit.'.format(data_split))
    checkpoint_path = os.path.join(
        processed_data_path,'{}_seed_{}_best_model.pt'.format(data_split, seed))     
    #Training loop
    current_patience = 0
    best_valid_loss = float('inf')
    for epoch in range(num_epochs):
        # training
        model.train()
        for batch in train_loader:
            # Forward pass
            inputs = batch[0].to(device)
            targets = batch[1].to(device)
            optimizer.zero_grad() # Resets the gradients
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # Backward pass and optimization
            loss.backward()
            if clip_grad_norm: # if True: clips gradient at specified value
                clip_grad_value_(model.parameters(), clip_value=clip_value)
            optimizer.step()        
        # Validation
        model.eval()
        with torch.no_grad():
            total_valid_loss = 0
            for batch in val_loader:
                inputs = batch[0].to(device)
                targets = batch[1].to(device)
                outputs = model(inputs)
                valid_loss = criterion(outputs, targets)
                total_valid_loss += valid_loss.item()                    
            average_valid_loss = total_valid_loss / len(val_loader)
        # print the results       
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Validation Loss: {average_valid_loss}')
        # save the best model
        if average_valid_loss < best_valid_loss - min_delta:
            best_valid_loss = average_valid_loss
            current_patience = 0
            checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
            'best_valid_loss': best_valid_loss            
            }
            torch.save(checkpoint, checkpoint_path)
        else:
            current_patience += 1
            # Check for early stopping
            if current_patience >= early_patience:
                print('Early stopping: Val loss has not improved for {} epochs.'.format(early_patience))
                break
        
        # Update learning rate if there is any scheduler
        if scheduler is not None:
           scheduler.step(average_valid_loss)
           
# function to handle inference with trained model
def test_model(model=None, test_loader=None, test_original_lengths=None,
               y_scaler=None, processed_data_path=None, data_split=None,
               seed=None, device=None, normalization=False):
    start=datetime.now()
    print('Now start inference for {} data slit.'.format(data_split))
    checkpoint_path = os.path.join(
        processed_data_path,'{}_seed_{}_best_model.pt'.format(data_split, seed)) 
    report_path = os.path.join(processed_data_path,
                               '{}_seed_{}_report_.txt'.format(
                                   data_split,seed))    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    all_results = {'GroundTruth': [], 'Prediction': [], 'Prefix_length': [],
                   'Absolute_error': [], 'Absolute_percentage_error': []}
    absolute_error = 0
    absolute_percentage_error = 0
    length_idx = 0 
    model.eval()
    with torch.no_grad():
        for index, test_batch in enumerate(test_loader):
            inputs = test_batch[0].to(device)
            _y_truth = test_batch[1].to(device)
            batch_size = inputs.shape[0]
            _y_pred = model(inputs)
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
        num_test_batches = len(test_loader)    
        absolute_error /= num_test_batches    
        absolute_percentage_error /= num_test_batches
    print('Test - MAE: {:.3f}, '
                  'MAPE: {:.3f}'.format(
                      round(absolute_error, 3),
                      round(absolute_percentage_error, 3))) 
    inference_time = (datetime.now()-start).total_seconds() 
    instance_inference = inference_time / len (test_original_lengths) * 1000
    with open(report_path, 'a') as file:
        file.write('Inference time- in seconds: {}\n'.format(inference_time))
        file.write(
            'Inference time for each instance- in miliseconds: {}\n'.format(
                instance_inference))
        file.write('Test - MAE: {:.3f}, '
                      'MAPE: {:.3f}'.format(
                          round(absolute_error, 3),
                          round(absolute_percentage_error, 3)))
    
    flattened_list = [item for sublist in all_results['Prefix_length'] 
                      for item in sublist]
    all_results['Prefix_length'] = flattened_list
    results_df = pd.DataFrame(all_results)
    csv_filename = os.path.join(
        processed_data_path,'{}_seed_{}_inference_result_.csv'.format(
            data_split,seed)) 
    results_df.to_csv(csv_filename, index=False) 