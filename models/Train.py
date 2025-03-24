import os
from datetime import datetime
import torch
from torch.nn.utils import clip_grad_value_
import torch.optim as optim
from utils.utils import (get_optimizer_params, set_optimizer)

# method to handle training the model
def train_model(model=None, uq_method=None, train_loader=None, val_loader=None,
                train_val_loader=None, criterion=None, optimizer=None,
                scheduler=None, device=None, num_epochs=100, 
                early_patience=50, min_delta=0, early_stop=True,
                clip_grad_norm=None, clip_value=None,
                processed_data_path=None, report_path =None,
                data_split='holdout', fold=None, cfg=None, seed=None,
                model_idx=None, ensemble_mode=False, sqr_q='all',
                sqr_factor=None, exp_id=None):
    """
    This function is used for the following UQ techniques:
        1) Deterministic point estimate.
        2) Dropout approximation (DA, CDA)
        3) Heteroscedastic regression (H)
        combinations of 2,3 (DA+H , CDA+H)
        4) Bootstrapping Ensembles (BE+H)
    """
    
    # get optimizer parameters to be used in retraining with train+val data
    base_lr, eps, weight_decay = get_optimizer_params(optimizer)
    optimizer_type = cfg.get('optimizer').get('type')
    
    # get current time (as start) to compute training time
    start=datetime.now()
    
    # if training is not part of an ensemble
    if not ensemble_mode:  
        print(f'Training for experiment number: {exp_id}, \
              data split: {data_split} , {fold}.')
        # Write the configurations in the report
        with open(report_path, 'w') as file:
            file.write('Configurations:\n')
            file.write(str(cfg))
            file.write('\n')
            file.write('\n')  
            file.write('Training will be done for {} epochs.\n'.format(num_epochs))
        # set the checkpoint path      
        if data_split=='holdout':
            checkpoint_path = os.path.join(
                processed_data_path,
                '{}_{}_seed_{}_exp_{}_best_model.pt'.format(
                    uq_method, data_split, seed, exp_id)) 
        else:
            checkpoint_path = os.path.join(
                processed_data_path,
                '{}_{}_fold{}_seed_{}_exp_{}_best_model.pt'.format(
                    uq_method, data_split, fold, seed, exp_id))   
    else:
        # if we are training a member of an ensemble
        print(f'Training for experiment number: {exp_id}, \
              data split: {data_split} , {fold}, \
              model number:{model_idx} in the ensemble.')
        # Write the configurations in the report
        if model_idx== 1:
            # Write the configurations in the report
            with open(report_path, 'w') as file:
                file.write('Configurations:\n')
                file.write(str(cfg))
                file.write('\n')
                file.write('\n')  
                file.write('Ensemble member index: {}.\n'.format(model_idx))
                file.write('Training will be done for {} epochs.\n'.format(num_epochs))
        else:
            with open(report_path, 'a') as file:
                file.write('\n')
                file.write('\n')
                file.write('Ensemble member index: {}.\n'.format(model_idx))
                file.write('Training will be done for {} epochs.\n'.format(num_epochs))  
        # set the checkpoint path
        if data_split=='holdout':
            checkpoint_path = os.path.join(
                processed_data_path,'{}_{}_seed_{}_exp_{}_member_{}_best_model.pt'.format(
                    uq_method, data_split, seed, exp_id, model_idx))
        else:
            checkpoint_path = os.path.join(
                processed_data_path,
                '{}_{}_fold{}_seed_{}_exp_{}_member_{}_best_model.pt'.format(
                    uq_method, data_split, fold, seed, exp_id, model_idx))               
    print(f'Training will be done for {num_epochs} epochs.')     

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
            if uq_method == 'deterministic':
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            elif (uq_method == 'DA' or uq_method == 'CDA' or
                  uq_method == 'DA+H' or uq_method == 'CDA+H'):
                mean, log_var, regularization = model(inputs) 
                if (uq_method == 'DA+H' or uq_method == 'CDA+H'):
                    loss = criterion(targets, mean, log_var) + regularization
                else:
                    loss = criterion(mean, targets) + regularization
            elif (uq_method == 'H' or uq_method == 'BE+H'):                
                mean, log_var = model(inputs)
                loss = criterion(targets, mean, log_var)                
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
                if uq_method == 'deterministic':
                    outputs = model(inputs)
                    valid_loss = criterion(outputs, targets)
                elif (uq_method == 'DA' or uq_method == 'CDA' or
                      uq_method == 'DA+H' or uq_method == 'CDA+H'):
                    mean, log_var, regularization = model(inputs)
                    if (uq_method == 'DA+H' or uq_method == 'CDA+H'):
                        valid_loss = criterion(targets, mean,
                                               log_var) + regularization
                    else:
                        valid_loss = criterion(mean, targets) + regularization
                elif (uq_method == 'H' or uq_method == 'BE+H'):
                    mean, log_var = model(inputs)
                    valid_loss = criterion(targets, mean, log_var)                    
                total_valid_loss += valid_loss.item()                    
            average_valid_loss = total_valid_loss / len(val_loader)                          
        # print the results       
        print(f'Epoch {epoch + 1}/{num_epochs},',
              f'Loss: {loss.item()}, Validation Loss: {average_valid_loss}')
        with open(report_path, 'a') as file:
            file.write('Epoch {}/{} Loss: {}, Validation Loss: {} .\n'.format(
                epoch + 1, num_epochs, loss.item(), average_valid_loss))            
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
            best_epoch = epoch + 1
        else:
            current_patience += 1
            # Check for early stopping
            if (early_stop and current_patience >= early_patience):
                print('Early stopping No improvement in Val loss for:', 
                      f'{early_patience} epochs.')
                with open(report_path, 'a') as file:
                    file.write(
                        'Early stop- no improvement for: {} epochs.\n'.format(
                            early_patience))  
                break        
        # Update learning rate if there is any scheduler
        if scheduler is not None:
           scheduler.step(average_valid_loss)
    training_time = (datetime.now()-start).total_seconds()
    with open(report_path, 'a') as file:
        file.write('Training time- in seconds: {}\n'.format(
            training_time)) 
        file.write('#######################################################\n')
        file.write('#######################################################\n')
    
    ##########################################################################
    ######## retrain the model on train + val sets.
    ##########################################################################
    if train_val_loader!=None:
        # we do not retrain the model for bootstrapping ensembles
        print('Now start retraining on training + validation sets')
        with open(report_path, 'a') as file:
            file.write('Now start retraining on training + validation sets\n')
            file.write('Retraning epochs: {} .\n'.format(best_epoch)) 
            file.write('###################################################\n')    
        # get current time (as start) to compute training time
        start=datetime.now()   
        # re-initialize model weights
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        # re-initialize optimizer and scheduler
        optimizer = set_optimizer (model, optimizer_type, base_lr, eps, 
                                   weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)
        model.train()
        # Training loop     
        for epoch in range(best_epoch): 
            for batch in train_val_loader:
                # Forward pass
                inputs = batch[0].to(device)
                targets = batch[1].to(device)
                optimizer.zero_grad() # Resets the gradients
                if uq_method == 'deterministic':
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                elif (uq_method == 'DA' or uq_method == 'CDA' or
                      uq_method == 'DA+H' or uq_method == 'CDA+H'):
                    mean, log_var, regularization = model(inputs) 
                    if (uq_method == 'DA+H' or uq_method == 'CDA+H'):
                        loss = criterion(targets, mean, log_var) + regularization
                    else:
                        loss = criterion(mean, targets) + regularization
                elif (uq_method == 'H' or uq_method == 'BE+H'):                
                    mean, log_var = model(inputs)
                    loss = criterion(targets, mean, log_var)                
                # Backward pass and optimization
                loss.backward()
                if clip_grad_norm: # if True: clips gradient at specified value
                    clip_grad_value_(model.parameters(), clip_value=clip_value)
                optimizer.step()
                
            # print the results       
            print(f'Epoch {epoch + 1}/{best_epoch},', f'Loss: {loss.item()}')
            with open(report_path, 'a') as file:
                file.write('Epoch {}/{} Loss: {} .\n'.format(
                    epoch + 1, best_epoch, loss.item()))    
            # Update learning rate if there is any scheduler
            if scheduler is not None:
                scheduler.step(average_valid_loss)
    
        training_time = (datetime.now()-start).total_seconds()
        with open(report_path, 'a') as file:
            file.write('Retraining time- in seconds: {}\n'.format(training_time)) 
            file.write('###################################################\n')
            file.write('###################################################\n')
        checkpoint = {
            'epoch': best_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item()
            }
        torch.save(checkpoint, checkpoint_path)
    
    return checkpoint_path