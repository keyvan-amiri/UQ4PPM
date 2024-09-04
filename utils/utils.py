import os
import argparse
import yaml
import sys
import logging
import random
import shutil
import itertools
import numpy as np
import torch
import torch.optim as optim
import torch.utils.tensorboard as tb
from scipy.stats import norm
from models.dalstm import DALSTMModel, DALSTMModelMve, dalstm_init_weights
from models.stochastic_dalstm import StochasticDALSTM


##############################################################################
# Utlility functions for training and inference
##############################################################################
# a method to define exeriments for hyper-parameter tuning
def get_exp(uq_method=None, cfg=None, random=False):
    if (uq_method == 'en_b' or uq_method == 'en_b_mve' or
        uq_method == 'en_t' or uq_method == 'en_t_mve'): 
        num_model_lst = cfg.get('uncertainty').get('ensemble').get('num_models')
        if not isinstance(num_model_lst, list):
            raise ValueError('number of models should be a list.')
        else:
            max_model_num = max(num_model_lst)
        boot_ratio_lst = cfg.get('uncertainty').get('ensemble'
                                                    ).get('Bootstrapping_ratio')
        if not isinstance(boot_ratio_lst, list):
            raise ValueError('Bootstrapping ratio should be a list.')
        hyperparameters = {'num_models': num_model_lst, 
                           'Bootstrapping_ratio': boot_ratio_lst}
        keys = hyperparameters.keys()
        values = hyperparameters.values()
        combinations = list(itertools.product(*values))
        experiments = [dict(zip(keys, combination)) 
                       for combination in combinations]
    return experiments, max_model_num


# a method to get model based on UQ technique selected for experiment
def get_model(uq_method=None, input_size=None, hidden_size=None, n_layers=None,
              max_len=None, dropout=None, dropout_prob=None, num_models=None,
              concrete_dropout=None, weight_regularizer=None,
              dropout_regularizer=None, Bayes=None, device=None):    

    # deterministic model 
    if uq_method == 'deterministic':
        model = DALSTMModel(input_size=input_size, hidden_size=hidden_size,
                            n_layers=n_layers, max_len=max_len, dropout=dropout,
                            p_fix=dropout_prob).to(device) 
        return model
    elif uq_method == 'SQR':
        model = DALSTMModel(input_size=input_size+1, hidden_size=hidden_size,
                            n_layers=n_layers, max_len=max_len, dropout=dropout,
                            p_fix=dropout_prob).to(device) 
        return model
    elif uq_method == 'RF':
        model = DALSTMModel(input_size=input_size, hidden_size=hidden_size,
                            n_layers=n_layers, max_len=max_len, dropout=dropout,
                            p_fix=dropout_prob, exclude_last_layer=True).to(device) 
        return model
    elif (uq_method == 'LA' or uq_method == 'LA_H'):
        model = DALSTMModel(input_size=input_size, hidden_size=hidden_size,
                            n_layers=n_layers, max_len=max_len, dropout=dropout,
                            p_fix=dropout_prob, return_squeezed=False).to(device)
        return model
    # Stochastic model
    elif (uq_method == 'DA' or uq_method == 'CDA'):
        model = StochasticDALSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            max_len=max_len,
            dropout=dropout,
            concrete=concrete_dropout,
            p_fix=dropout_prob,
            weight_regularizer=weight_regularizer,
            dropout_regularizer=dropout_regularizer,
            hs=False,
            Bayes=Bayes,
            device=device).to(device)
        return model
    # dropout approximation with heteroscedastic regression
    elif (uq_method == 'DA_A' or uq_method == 'CDA_A'):
        model = StochasticDALSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            max_len=max_len,
            dropout=dropout,
            concrete=concrete_dropout,
            p_fix=dropout_prob,
            weight_regularizer=weight_regularizer,
            dropout_regularizer=dropout_regularizer,
            hs=True,
            Bayes=Bayes,
            device=device).to(device)
        return model
    # heteroscedastic regression also known as mean variance estimation
    elif uq_method == 'mve':                
        model = DALSTMModelMve(input_size=input_size, hidden_size=hidden_size,
                               n_layers=n_layers, max_len=max_len,
                               dropout=dropout, p_fix=dropout_prob).to(device)
        return model
    
    # b: Bootstrapping ensemble: multiple models, same initialization.            
    elif (uq_method == 'en_b' or uq_method == 'en_b_mve'):
        # empty lists (ensemble of) models, optimizers, schedulers
        models = []
        for i in range(num_models):
            if uq_method == 'en_b':
                # each ensemble member is a deterministic model
                model = DALSTMModel(
                            input_size=input_size,
                            hidden_size=hidden_size,
                            n_layers=n_layers,
                            max_len=max_len,
                            dropout=dropout,
                            p_fix=dropout_prob).to(device)
            else:
                # last layer include log variance estimation
                model = DALSTMModelMve(
                            input_size=input_size,
                            hidden_size=hidden_size,
                            n_layers=n_layers,
                            max_len=max_len,
                            dropout=dropout,
                            p_fix=dropout_prob).to(device)
            models.append(model) 
        return models
    # t: Traditional ensemble: multiple models, different initialization.
    elif (uq_method == 'en_t' or uq_method == 'en_t_mve'):
        # empty lists (ensemble of) models, optimizers, schedulers
        models = []      
        # Original random state (before initializing models)
        original_rng_state = torch.get_rng_state()
        for i in range(num_models):
            # Set a unique seed for each model's initialization
            unique_seed = i + 100  
            torch.manual_seed(unique_seed)
            if uq_method == 'en_t':
                # each ensemble member is a deterministic model
                model = DALSTMModel(
                            input_size=input_size,
                            hidden_size=hidden_size,
                            n_layers=n_layers,
                            max_len=max_len,
                            dropout=dropout,
                            p_fix=dropout_prob).to(device)
            else:
                # last layer include log variance estimation
                model = DALSTMModelMve(
                            input_size=input_size,
                            hidden_size=hidden_size,
                            n_layers=n_layers,
                            max_len=max_len,
                            dropout=dropout,
                            p_fix=dropout_prob).to(device)                    
                # Apply weight initialization function
                model.apply(dalstm_init_weights)
            models.append(model)
        # Restore the original random state
        torch.set_rng_state(original_rng_state)
        return models

# a method to get optimizer and scheduler
def get_optimizer_scheduler(model=None, models=None, cfg=None,
                            ensemble_mode=False, num_models=None): 
    if not ensemble_mode:
        # get number of model parameters
        total_params = sum(p.numel() for p in model.parameters() 
                           if p.requires_grad) 
        # define optimizer
        optimizer = set_optimizer(
            model, cfg.get('optimizer').get('type'),
            cfg.get('optimizer').get('base_lr'), 
            cfg.get('optimizer').get('eps'), 
            cfg.get('optimizer').get('weight_decay'))
        # define scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5)  
        print(f'Total model parameters: {total_params}')    
        return optimizer, scheduler  
    else:
        # get number of parameters for the first model
        total_params = sum(p.numel() for p in models[0].parameters()
                           if p.requires_grad) 
        # define list of optimizeers and schedulers
        optimizers, schedulers = [], []
        for i in range(num_models):
            current_optimizer = set_optimizer(
                models[i], cfg.get('optimizer').get('type'),
                cfg.get('optimizer').get('base_lr'),
                cfg.get('optimizer').get('eps'),
                cfg.get('optimizer').get('weight_decay'))
            optimizers.append(current_optimizer)
            current_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizers[i], factor=0.5)
            schedulers.append(current_scheduler)                
        print(f'Total model parameters: {total_params}') 
        return optimizers, schedulers

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
        
        
def augment(x, tau=None, sqr_factor=12, aug_type=None, device=None):
    """
    Augment the input tensor `x` with the quantile `tau` as additional feature.
    Parameters:
    - x: Input tensor 
        if feature vector: batch_size*sequence_length*feature_size
    - tau: Quantile values. If None, a tensor filled with 0.5 is created.
           If a float, a tensor filled with this value is created.
    Returns:
    - Augmented tensor with tau appended as an additional feature.
    """
    if aug_type=='RNN':
        batch_size, sequence_length, feature_size = x.size()
        if tau is None:
            tau = torch.zeros(batch_size, 1).fill_(0.5)  # (batch_size, 1)
        elif isinstance(tau, float):
            tau = torch.zeros(batch_size, 1).fill_(tau)  # (batch_size, 1)    
        # Expand tau to match the sequence length
        # (batch_size, sequence_length, 1)
        tau = tau.unsqueeze(1).repeat(1, sequence_length, 1) 
        # Center and scale tau
        tau = (tau - 0.5) * sqr_factor  # (batch_size, sequence_length, 1)
        tau = tau.to(device)
        # Concatenate tau with the input features
        # (batch_size, sequence_length, feature_size + 1)
        augmented_x = torch.cat((x, tau), dim=2)  
    else:
        raise ValueError('Augmentation type has to be RNN') 
    return augmented_x

# method to provide z value for a confidence interval
def get_z_alpha_half(confidence_level):
    # Calculate alpha
    alpha = 1 - confidence_level    
    # Calculate the critical value z_alpha/2
    z_alpha_half = norm.ppf(1 - alpha / 2)    
    return z_alpha_half

# A method to add a customized suffix to a csv file name.
def add_suffix_to_csv(csv_file, added_suffix=None):
    # Check if the file name ends with .csv
    if csv_file.endswith('.csv'):
        # Insert the suffix before the .csv extension
        new_csv_file = csv_file[:-4] + added_suffix + '.csv'
        return new_csv_file
    else:
        raise ValueError("The file name does not end with .csv")
        
# Get prediction means, standard deviations, ground truth from inference result        
def get_mean_std_truth (df=None, uq_method=None):
    pred_mean = df['Prediction'].values 
    y_true = df['GroundTruth'].values
    if (uq_method=='DA_A' or uq_method=='CDA_A' or
        uq_method == 'en_t_mve' or uq_method == 'en_b_mve'):
        pred_std = df['Total_Uncertainty'].values 
    elif (uq_method=='CARD' or uq_method=='mve' or uq_method=='SQR'):
        pred_std = df['Aleatoric_Uncertainty'].values
    elif (uq_method=='DA' or uq_method=='CDA' or uq_method == 'en_t' or
          uq_method == 'en_b' or uq_method == 'RF' or uq_method == 'LA'):
        pred_std = df['Epistemic_Uncertainty'].values
    else:
        raise NotImplementedError(
            'Uncertainty quantification {} not understood.'.format(uq_method))
    return (pred_mean, pred_std, y_true)