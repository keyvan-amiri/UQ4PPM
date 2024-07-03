import os
import argparse
import yaml
import sys
import logging
import random
import shutil
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_
import torch.optim as optim
import torch.utils.tensorboard as tb
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
  
# TODO: combine these two methods for better code structure
# functions to set the optimizer object
def set_optimizer (model, optimizer_type, base_lr, eps, weight_decay):
    eps = float(eps) #ensure to having a floating number
    if optimizer_type == 'NAdam':
        optimizer = optim.NAdam(model.parameters(), lr=base_lr, eps=eps,
                                weight_decay=weight_decay)
    elif optimizer_type == 'AdamW':   
        optimizer = optim.AdamW(model.parameters(), lr=base_lr, eps=eps,
                                weight_decay=weight_decay)
    elif optimizer_type == 'Adam':   
        optimizer = optim.Adam(model.parameters(), lr=base_lr, eps=eps,
                               weight_decay=weight_decay) 
    elif optimizer_type == 'RAdam':
        optimizer = optim.RAdam(model.parameters(), lr=base_lr, eps=eps,
                               weight_decay=weight_decay)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=base_lr,
                              weight_decay=weight_decay)
    else:
        print(f'The optimizer {optimizer_type} is not supported')
    return optimizer

def get_optimizer(config_optim, parameters):
    if config_optim.optimizer == 'Adam':
        return optim.Adam(parameters, lr=config_optim.lr,
                          weight_decay=config_optim.weight_decay,
                          betas=(config_optim.beta1, 0.999),
                          amsgrad=config_optim.amsgrad,
                          eps=config_optim.eps)
    elif config_optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config_optim.lr,
                             weight_decay=config_optim.weight_decay)
    elif config_optim.optimizer == 'SGD':
        return optim.SGD(parameters, lr=config_optim.lr, momentum=0.9)
    elif config_optim.optimizer == 'NAdam':
        return optim.NAdam(parameters, lr=config_optim.lr, eps=1e-07)  
    else:
        raise NotImplementedError(
            'Optimizer {} not understood.'.format(config_optim.optimizer))

# function to handle training the model
def train_model(model=None, uq_method=None, heteroscedastic=None, 
                train_loader=None, val_loader=None,
                criterion=None, optimizer=None, scheduler=None, device=None,
                num_epochs=100, early_patience=20, min_delta=0,
                clip_grad_norm=None, clip_value=None,
                processed_data_path=None, report_path =None,
                data_split='holdout', fold=None, cfg=None, seed=None):
    print(f'Training for data split: {data_split} , {fold}.')
    start=datetime.now()
    # Write the configurations in the report
    with open(report_path, 'w') as file:
        file.write('Configurations:\n')
        file.write(str(cfg))
        file.write('\n')
        file.write('\n')  
        file.write('Training is done for {} epochs.\n'.format(num_epochs))
    print(f'Training is done for {num_epochs} epochs.') 
    # set the checkpoint name      
    if data_split=='holdout':
        checkpoint_path = os.path.join(processed_data_path,
                                       '{}_{}_seed_{}_best_model.pt'.format(
                                           uq_method, data_split, seed)) 
    else:
        checkpoint_path = os.path.join(
            processed_data_path, '{}_{}_fold{}_seed_{}_best_model.pt'.format(
                uq_method, data_split, fold, seed))   
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
                  uq_method == 'DA_A' or uq_method == 'CDA_A'):
                mean, log_var, regularization = model(inputs)                
                if heteroscedastic:
                    loss = criterion(targets, mean, log_var) + regularization
                else:
                    loss = criterion(mean, targets) + regularization
            elif uq_method == 'mve':
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
                      uq_method == 'DA_A' or uq_method == 'CDA_A'):
                    mean, log_var, regularization = model(inputs)
                    if heteroscedastic:
                        valid_loss = criterion(targets, mean,
                                               log_var) + regularization
                    else:
                        valid_loss = criterion(mean, targets) + regularization
                elif uq_method == 'mve':
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
            if current_patience >= early_patience:
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
                
           
# function to handle inference with trained model
def test_model(model=None, uq_method=None, heteroscedastic=None,
               num_mc_samples=None, test_loader=None,
               test_original_lengths=None, y_scaler=None, 
               processed_data_path=None, report_path=None,
               data_split=None, fold=None, seed=None, device=None,
               normalization=False): 
    
    start=datetime.now()
    if data_split=='holdout':
        print(f'Now: start inference- data split: {data_split}.')
        checkpoint_path = os.path.join(processed_data_path,
                                       '{}_{}_seed_{}_best_model.pt'.format(
                                           uq_method, data_split, seed)) 
    else:
        print(f'Now: start inference- data split: {data_split} ,  fold: {fold}.')
        checkpoint_path = os.path.join(
            processed_data_path, '{}_{}_fold{}_seed_{}_best_model.pt'.format(
                uq_method, data_split, fold, seed)) 
        
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    # define the columns in csv file for the results
    if uq_method == 'deterministic':
        all_results = {'GroundTruth': [], 'Prediction': [],
                       'Prefix_length': [], 'Absolute_error': [],
                       'Absolute_percentage_error': []}
    elif (uq_method == 'DA' or uq_method == 'CDA'):
        all_results = {'GroundTruth': [], 'Prediction': [],
                       'Epistemic_Uncertainty': [], 'Prefix_length': [],
                       'Absolute_error': [], 'Absolute_percentage_error': []}
    elif (uq_method == 'DA_A' or uq_method == 'CDA_A'):
        all_results = {'GroundTruth': [], 'Prediction': [],
                       'Epistemic_Uncertainty': [], 'Aleatoric_Uncertainty': [],
                       'Total_Uncertainty': [], 'Prefix_length': [],
                       'Absolute_error': [], 'Absolute_percentage_error': []} 
    elif uq_method == 'mve':
        all_results = {'GroundTruth': [], 'Prediction': [],
                       'Aleatoric_Uncertainty': [], 'Prefix_length': [],
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
            
            # get model outputs, and uncertainties if required
            if uq_method == 'deterministic':            
                _y_pred = model(inputs)
            elif (uq_method == 'DA' or uq_method == 'CDA' or
                  uq_method == 'DA_A' or uq_method == 'CDA_A'):
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
                if heteroscedastic:
                    stacked_log_var = torch.stack(logvar_list, dim=0)
                    stacked_var = torch.exp(stacked_log_var)
                    mean_var = torch.mean(stacked_var, dim=0)
                    aleatoric_std = torch.sqrt(mean_var).to(device)
                    # normalize aleatoric uncertainty if necessary
                    if normalization:
                        aleatoric_std = y_scaler * aleatoric_std
                    total_std = epistemic_std + aleatoric_std
            elif uq_method == 'mve':
                _y_pred, log_var = model(inputs)
                aleatoric_std = torch.sqrt(torch.exp(log_var))
                # normalize aleatoric uncertainty if necessary
                if normalization:
                    aleatoric_std = y_scaler * aleatoric_std                    
            
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
            if (uq_method == 'DA' or uq_method == 'CDA' or
                  uq_method == 'DA_A' or uq_method == 'CDA_A'):
                epistemic_std = epistemic_std.detach().cpu().numpy()
                all_results['Epistemic_Uncertainty'].extend(
                    epistemic_std.tolist()) 
                if heteroscedastic:
                    aleatoric_std = aleatoric_std.detach().cpu().numpy()
                    total_std = total_std.detach().cpu().numpy()
                    all_results['Aleatoric_Uncertainty'].extend(
                        aleatoric_std.tolist())
                    all_results['Total_Uncertainty'].extend(
                        total_std.tolist()) 
            elif uq_method == 'mve':
                aleatoric_std = aleatoric_std.detach().cpu().numpy()
                all_results['Aleatoric_Uncertainty'].extend(
                    aleatoric_std.tolist())                
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
    if data_split=='holdout':
        csv_filename = os.path.join(
            processed_data_path,'{}_{}_seed_{}_inference_result_.csv'.format(
                uq_method,data_split,seed))
    else:
        csv_filename = os.path.join(
            processed_data_path,
            '{}_{}_fold{}_seed_{}_inference_result_.csv'.format(
                uq_method, data_split, fold, seed))         
    results_df.to_csv(csv_filename, index=False)

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace
    
# return original config before any changes within parse_config method.
def parse_temp_config(task_name=None):
    task_name = task_name + '.yml'
    with open(os.path.join('cfg', task_name), "r") as f:
         temp_config = yaml.safe_load(f)
         temporary_config = dict2namespace(temp_config)
    return temporary_config

# update original config file, and return it alongside the logger.
def parse_config(args=None):    
    # set log path
    args.log_path = os.path.join(args.exp, 'logs', args.doc)
    # set separate log folder for recalibration results, and reports
    if args.recalibration:
        args.log_path2 = os.path.join(args.exp, 'recalibration', args.doc)
        
    # parse config file
    with open(os.path.join(args.config), "r") as f:
        if args.test:
            config = yaml.unsafe_load(f)
            new_config = config
        else:
            config = yaml.safe_load(f)
            new_config = dict2namespace(config)
    tb_path = os.path.join(args.exp, 'tensorboard', args.doc)

    if not args.test:
        args.im_path = os.path.join(args.exp, new_config.training.image_folder, args.doc)
        # if noise_prior is not provided by the user the relevant config is set to False.
        new_config.diffusion.noise_prior = True if args.noise_prior else False
        new_config.model.cat_y_pred = False if args.no_cat_f_phi else True
        if not args.resume_training:
            if not args.timesteps is None:
                new_config.diffusion.timesteps = args.timesteps
            if os.path.exists(args.log_path):
                overwrite = False
                if args.ni:
                    overwrite = True
                else:
                    response = input(
                        'Folder {} already exists. Overwrite? (Y/N)'.format(
                            args.log_path))
                    if response.upper() == "Y":
                        overwrite = True
                if overwrite:
                    shutil.rmtree(args.log_path)
                    shutil.rmtree(tb_path)
                    shutil.rmtree(args.im_path)
                    os.makedirs(args.log_path)
                    os.makedirs(args.im_path)
                    if os.path.exists(tb_path):
                        shutil.rmtree(tb_path)
                else:
                    print('Folder exists. Program halted.')
                    sys.exit(0)
            else:
                os.makedirs(args.log_path)
                if not os.path.exists(args.im_path):
                    os.makedirs(args.im_path)
            #save the updated config the log in result folder
            with open(os.path.join(args.log_path, 'config.yml'), 'w') as f:
                yaml.dump(new_config, f, default_flow_style=False)

        new_config.tb_logger = tb.SummaryWriter(log_dir=tb_path)
        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        # saving training info to a .txt file
        handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        if args.recalibration:
            # set the image path for inference on validation (i.e., recalibration)
            args.im_path = os.path.join(
                args.exp, 'recalibration', new_config.testing.image_folder,
                args.doc)
        else:
            # set the image path for inference on test set.
            args.im_path = os.path.join(
                args.exp, new_config.testing.image_folder, args.doc)
        
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        # saving test metrics to a .txt file
        if args.recalibration:
            txt_path = os.path.join(args.log_path2,'testmetrics.txt')
            os.makedirs(os.path.dirname(txt_path), exist_ok=True)
            # set a handler for recalibration
            open(txt_path, 'w').close()
            handler2 = logging.FileHandler(txt_path)
        else:
            # set a handler for inference on test
            handler2 = logging.FileHandler(
                os.path.join(args.log_path, 'testmetrics.txt'))
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

        os.makedirs(args.im_path, exist_ok=True)

    # add device
    device_name = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set number of threads
    if args.thread > 0:
        torch.set_num_threads(args.thread)
        print('Using {} threads'.format(args.thread))

    # set random seed
    if isinstance(args.seed, list):
        seed = int(args.seed[0])       
    set_random_seed(seed)

    torch.backends.cudnn.benchmark = True

    return new_config, logger

# optimize disk usage: remove unnecessary data inputs
# TODO: this method should be changed data_root is removed from configurations!
def delete_preprocessd_tensors (config):
    _DATA_DIRECTORY_PATH = os.path.join(config.data.data_root,
                                        config.data.dir, "data")      
    _DATA_TRAIN_FILE_PATH = os.path.join(_DATA_DIRECTORY_PATH,
                                         "x_train.pt")
    _TARGET_TRAIN_FILE_PATH = os.path.join(_DATA_DIRECTORY_PATH,
                                           "y_train.pt")
    _DATA_TEST_FILE_PATH = os.path.join(_DATA_DIRECTORY_PATH,
                                        "x_test.pt")
    _TARGET_TEST_FILE_PATH = os.path.join(_DATA_DIRECTORY_PATH,
                                          "y_test.pt")
    pre_processing_paths = [_DATA_TRAIN_FILE_PATH,
                            _TARGET_TRAIN_FILE_PATH,
                            _DATA_TEST_FILE_PATH,
                            _TARGET_TEST_FILE_PATH]
    for file_path in pre_processing_paths:
        os.remove(file_path)
        
# A method to get booleans from strings
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')