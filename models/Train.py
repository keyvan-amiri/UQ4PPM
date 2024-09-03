import os
import random
from datetime import datetime
import torch
from torch.nn.utils import clip_grad_value_
from utils.utils import augment

# method to handle training the model
def train_model(model=None, uq_method=None, train_loader=None, val_loader=None,
                criterion=None, optimizer=None, scheduler=None, device=None,
                num_epochs=100, early_patience=20, min_delta=0, early_stop=True,
                clip_grad_norm=None, clip_value=None,
                processed_data_path=None, report_path =None,
                data_split='holdout', fold=None, cfg=None, seed=None,
                model_idx=None, ensemble_mode=False, sqr_q='all',
                sqr_factor=None):
    
    # get current time (as start) to compute training time
    start=datetime.now()
    
    # if training is not part of an ensemble
    if not ensemble_mode:  
        print(f'Training for data split: {data_split} , {fold}.')
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
                processed_data_path,'{}_{}_seed_{}_best_model.pt'.format(
                    uq_method, data_split, seed)) 
        else:
            checkpoint_path = os.path.join(
                processed_data_path, '{}_{}_fold{}_seed_{}_best_model.pt'.format(
                    uq_method, data_split, fold, seed))   
    else:
        # if we are training a member of an ensemble
        print(f'Training for data split: {data_split} , {fold}, \
              model number:{model_idx} in the ensemble')
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
                processed_data_path,'{}_{}_seed_{}_member_{}_best_model.pt'.format(
                    uq_method, data_split, seed, model_idx))
        else:
            checkpoint_path = os.path.join(
                processed_data_path,
                '{}_{}_fold{}_seed_{}_member_{}_best_model.pt'.format(
                    uq_method, data_split, fold, seed, model_idx))               
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
            if (uq_method == 'deterministic' or uq_method == 'en_t' or 
                uq_method == 'en_b'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            elif (uq_method == 'SQR'):
                if sqr_q == 'all':
                    taus = torch.rand(inputs.shape[0], 1) 
                elif isinstance(sqr_q, list):
                    # Generate random values from the list
                    taus = torch.tensor([random.choice(sqr_q) 
                                         for _ in range(inputs.shape[0])]
                                        ).view(-1, 1)   
                else:
                    taus = torch.zeros(inputs.shape[0], 1).fill_(sqr_q)
                taus = taus.to(device)
                outputs = model(
                    augment(inputs, tau=taus, sqr_factor=sqr_factor,
                            aug_type='RNN', device=device))               
                loss = criterion(outputs, targets, taus)
            elif (uq_method == 'DA' or uq_method == 'CDA' or
                  uq_method == 'DA_A' or uq_method == 'CDA_A'):
                mean, log_var, regularization = model(inputs) 
                if (uq_method == 'DA_A' or uq_method == 'CDA_A'):
                    loss = criterion(targets, mean, log_var) + regularization
                else:
                    loss = criterion(mean, targets) + regularization
            elif (uq_method == 'mve' or uq_method == 'en_t_mve' or
                  uq_method == 'en_b_mve'):                
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
                if (uq_method == 'deterministic' or uq_method == 'en_t' or
                    uq_method == 'en_b'):
                    outputs = model(inputs)
                    valid_loss = criterion(outputs, targets)
                elif (uq_method == 'SQR'):
                    if sqr_q == 'all':
                        taus = torch.rand(inputs.shape[0], 1)
                    elif isinstance(sqr_q, list):
                        # Generate random values from the list
                        taus = torch.tensor([random.choice(sqr_q) 
                                             for _ in range(inputs.shape[0])]
                                            ).view(-1, 1)                        
                    else:
                        taus = torch.zeros(inputs.shape[0], 1).fill_(sqr_q)
                    outputs = model(
                        augment(inputs, tau=taus, sqr_factor=sqr_factor,
                                aug_type='RNN', device=device)) 
                    taus = taus.to(device)
                    valid_loss = criterion(outputs, targets, taus)
                elif (uq_method == 'DA' or uq_method == 'CDA' or
                      uq_method == 'DA_A' or uq_method == 'CDA_A'):
                    mean, log_var, regularization = model(inputs)
                    if (uq_method == 'DA_A' or uq_method == 'CDA_A'):
                        valid_loss = criterion(targets, mean,
                                               log_var) + regularization
                    else:
                        valid_loss = criterion(mean, targets) + regularization
                elif (uq_method == 'mve' or uq_method == 'en_t_mve' or
                      uq_method == 'en_b_mve'):
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