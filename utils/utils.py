import os
import re
import argparse
import random
import itertools
import numpy as np
from scipy.stats import norm, spearmanr, pearsonr
from sklearn.feature_selection import mutual_info_regression
import torch
import torch.optim as optim
from models.dalstm import DALSTMModel, DALSTMModelMve, dalstm_init_weights
from models.stochastic_dalstm import StochasticDALSTM
#import pandas as pd


# a method to define exeriments for hyper-parameter tuning
def get_exp(uq_method=None, cfg=None, is_random=False, random_ratio=0.1):
    
    if (uq_method == 'BE' or uq_method == 'BE+H' or
        uq_method == 'DE' or uq_method == 'DE+H'): 
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

    elif (uq_method == 'DA' or uq_method == 'DA+H' or uq_method == 'CDA'
          or uq_method == 'CDA+H' or uq_method == 'H'):
        num_mc_lst = cfg.get('uncertainty').get('dropout_approximation').get(
            'num_stochastic_forward_path')
        print(type(num_mc_lst))
        if not isinstance(num_mc_lst, list):
            raise ValueError('Monte Carlo Samples should be packed in a list.')
        early_stop_lst = cfg.get('train').get('early_stop')
        if not isinstance(early_stop_lst, list):
            raise ValueError('Early stop possibilities should be packed \
                             in a list.')
        hyperparameters = {'num_mcmc': num_mc_lst, 
                           'early_stop': early_stop_lst} 
    if uq_method == 'deterministic':         
        deterministc_lst = [True] 
        early_stop_lst = [True]
        hyperparameters = {'deterministic': deterministc_lst, 
                           'early_stop': early_stop_lst} 
    if uq_method == 'E-RF': 
        # define loss funciton options for fitting the auxiliary model
        loss_lst = cfg.get('uncertainty').get('union').get('loss_function')
        if not isinstance(loss_lst, list):
            raise ValueError('Loss Function possibilities should be packed \
                             in a list.')
        # define number of trees in the forest
        n_est_lst = cfg.get('uncertainty').get('union').get('n_estimators')
        if not isinstance(n_est_lst, list):
            raise ValueError('number of trees should be packed in a list.')
        # whether to use a maximum length for Random Forest or not
        depth_control_lst = cfg.get('uncertainty').get('union').get(
            'depth_control')
        if not isinstance(depth_control_lst, list):
            raise ValueError('depth contor options should be packed in a list.')
        # whether to use a maximum length for Random Forest or not
        max_depth_lst = cfg.get('uncertainty').get('union').get('max_depth')
        if not isinstance(max_depth_lst, list):
            raise ValueError('Max depth options should be packed in a list.')
            
        hyperparameters = {'criterion': loss_lst,
                           'n_estimators': n_est_lst,
                           'depth_control': depth_control_lst,
                           'max_depth': max_depth_lst} 
    if uq_method == 'LA':
        hessian_structure_lst = cfg.get('uncertainty').get('laplace').get(
            'hessian_structure')
        if not isinstance(hessian_structure_lst, list):
            raise ValueError('Hessian structure possibilities should be packed\
                             in a list.')
        # whether to estimate the prior precision and observation noise 
        # using empirical Bayes after training or not
        empirical_bayes_lst = cfg.get('uncertainty').get('laplace').get(
               'empirical_bayes')
        if not isinstance(empirical_bayes_lst, list):
            raise ValueError('Empirical Bayes possibilities should be packed\
                             in a list.')
        # number of epochs in case of empirical Bayes optimization
        la_epochs_lst = cfg.get('uncertainty').get('laplace').get('epochs')
        if not isinstance(la_epochs_lst, list):
            raise ValueError('Empirical Bayes optimization epoch options should\
                             be packed in a list.')
        # learning rate in case of empirical Bayes optimization
        la_lr_lst = cfg.get('uncertainty').get('laplace').get('lr')
        if not isinstance(la_lr_lst, list):
            raise ValueError('Empirical Bayes learning rate options should be\
                             packed in a list.')
        # amount of observation noise that is considered                     
        sigma_noise_lst = cfg.get('uncertainty').get('laplace').get(
                                 'sigma_noise')
        if not isinstance(sigma_noise_lst, list):
            raise ValueError('Sigma noise options should be packed in a list.')
        # whether to apply statistical adjustment to observation noise or not
        stat_noise_lst = cfg.get('uncertainty').get('laplace').get('stat_noise')
        if not isinstance(stat_noise_lst, list):
            raise ValueError('Statistical noise options should be packed in a \
                             list.')
        # prior precision of a Gaussian prior (= weight decay); it is a scalar                     
        prior_precision_lst = cfg.get('uncertainty').get('laplace').get(
                                 'prior_precision')
        if not isinstance(prior_precision_lst, list):
            raise ValueError('Prior precision options should be packed in a \
                             list.')
        # temperature of the likelihood; lower temperature leads to more
        # concentrated posterior and vice versa.                     
        temperature_lst = cfg.get('uncertainty').get('laplace').get(
            'temperature')
        if not isinstance(temperature_lst, list):
            raise ValueError('Temperature values should be packed in a list.')
        # num of MC samples for approx. posterior predictive distribution
        n_samples_lst = cfg.get('uncertainty').get('laplace').get('n_samples')
        if not isinstance(n_samples_lst, list):
            raise ValueError('Number of samples for inference should be packed\
                             in a list.')
        hyperparameters = {'hessian_structure': hessian_structure_lst,
                           'empirical_bayes': empirical_bayes_lst,
                           'la_epochs': la_epochs_lst, 'la_lr': la_lr_lst,
                           'sigma_noise': sigma_noise_lst,
                           'stat_noise': stat_noise_lst,
                           'prior_precision': prior_precision_lst,
                           'temperature': temperature_lst,
                           'n_samples': n_samples_lst} 
    if uq_method == 'SQR':
        tau_lst = cfg.get('uncertainty').get('sqr').get('tau')
        if not isinstance(tau_lst, list):
            raise ValueError('tau options should be packed in a list.')
        max_epochs_lst = cfg.get('train').get('max_epochs')
        if not isinstance(max_epochs_lst, list):
            raise ValueError('Maximum epoch options should be packed in a list.')
        sqr_factor_lst = cfg.get('uncertainty').get('sqr').get('scaling_factor')
        if not isinstance(sqr_factor_lst, list):
            raise ValueError('Scaling factor options should be packed in a list.')           
        hyperparameters = {'tau': tau_lst, 'max_epochs': max_epochs_lst,
                           'sqr_factor': sqr_factor_lst} 
    # get experiment list based on hyper-parameter combinations               
    if uq_method != 'E-RF':
        keys = hyperparameters.keys()
        values = hyperparameters.values()
        combinations = list(itertools.product(*values))
        experiments = [dict(zip(keys, combination)) 
                       for combination in combinations]
    else:
        experiments = []
        for criterion in loss_lst:
            for n_estimators in n_est_lst:
                for depth_control in depth_control_lst:
                    if depth_control:
                        # Use the full range of max_depth values
                        for max_depth in max_depth_lst:
                            experiments.append({
                                'criterion': criterion,
                                'n_estimators': n_estimators,
                                'depth_control': depth_control,
                                'max_depth': max_depth})
                    else:
                        # Set max_depth to None when depth_control is False
                        experiments.append({
                            'criterion': criterion,
                            'n_estimators': n_estimators,
                            'depth_control': depth_control,
                            'max_depth': None})
    # sample some of the experiments to save computation resources
    if is_random:
        # Randomly select the desired number of experiments
        random_selected_experiments = random.sample(
            experiments, int(len(experiments)*random_ratio))
        experiments = random_selected_experiments
        
    if (uq_method == 'BE' or uq_method == 'BE+H' or
        uq_method == 'DE' or uq_method == 'DE+H'): 
        return experiments, max_model_num
    else:
        return experiments
        

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
    elif uq_method == 'E-RF':
        model = DALSTMModel(input_size=input_size, hidden_size=hidden_size,
                            n_layers=n_layers, max_len=max_len, dropout=dropout,
                            p_fix=dropout_prob, exclude_last_layer=True).to(device) 
        return model
    elif uq_method == 'LA':
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
    elif (uq_method == 'DA+H' or uq_method == 'CDA+H'):
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
    elif uq_method == 'H':                
        model = DALSTMModelMve(input_size=input_size, hidden_size=hidden_size,
                               n_layers=n_layers, max_len=max_len,
                               dropout=dropout, p_fix=dropout_prob).to(device)
        return model
    
    # b: Bootstrapping ensemble: multiple models, same initialization.            
    elif (uq_method == 'BE' or uq_method == 'BE+H'):
        # empty lists (ensemble of) models, optimizers, schedulers
        models = []
        for i in range(num_models):
            if uq_method == 'BE':
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
    elif (uq_method == 'DE' or uq_method == 'DE+H'):
        # empty lists (ensemble of) models, optimizers, schedulers
        models = []      
        # Original random state (before initializing models)
        original_rng_state = torch.get_rng_state()
        for i in range(num_models):
            # Set a unique seed for each model's initialization
            unique_seed = i + 100  
            torch.manual_seed(unique_seed)
            if uq_method == 'DE':
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
  
# combine these two methods for better code structure
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

def get_optimizer_params(optimizer):
    # Access the first param group 
    param_group = optimizer.param_groups[0]
    # Extract optimizer parameters 
    base_lr = param_group['lr']
    eps = param_group.get('eps', None)
    weight_decay = param_group.get('weight_decay', None)
    return base_lr, eps, weight_decay

# optimize disk usage: remove unnecessary data inputs
# this method should be changed data_root is removed from configurations!
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
    if (uq_method=='DA+H' or uq_method=='CDA+H' or
        uq_method == 'DE+H' or uq_method == 'BE+H' or uq_method == 'LA' or
        uq_method=='deterministic' or uq_method=='GMM' or uq_method=='GMMD'):
        pred_std = df['Total_Uncertainty'].values 
    elif (uq_method=='CARD' or uq_method=='H' or uq_method=='SQR'):
        pred_std = df['Aleatoric_Uncertainty'].values
    elif (uq_method=='DA' or uq_method=='CDA' or uq_method == 'DE' or
          uq_method == 'BE' or uq_method == 'E-RF'):
        pred_std = df['Epistemic_Uncertainty'].values
    else:
        raise NotImplementedError(
            'Uncertainty quantification {} not understood.'.format(uq_method))
    return (pred_mean, pred_std, y_true)

# get label for plots
def uq_label_plot (uq_method=None):
    if uq_method=='DA+H':
        uq_label = 'Dropout + Heteroscedastic'
    elif uq_method=='CDA+H':
        uq_label = 'Concrete Dropout + Heteroscedastic'
    elif uq_method=='DE+H':
        uq_label = 'Traditional Ensemble + Heteroscedastic'
    elif uq_method == 'BE+H':
        uq_label = 'Bootstrapping Ensemble + Heteroscedastic'
    if uq_method=='DA':
        uq_label = 'Dropout'
    elif uq_method=='CDA':
        uq_label = 'Concrete Dropout'
    elif uq_method=='DE':
        uq_label = 'Traditional Ensemble'
    elif uq_method == 'BE':
        uq_label = 'Bootstrapping Ensemble'
    if uq_method=='E-RF':
        uq_label = 'Random Forest On Embedding'
    elif uq_method=='LA':
        uq_label = 'Laplace Approximation'
    elif uq_method=='H':
        uq_label = 'Heteroscedastic'
    elif uq_method == 'CARD':
        uq_label = 'CARD Model'
    elif uq_method == 'SQR':
        uq_label = 'Simultaneous Quantile Regression'
    elif uq_method == 'deterministic':
        uq_label = 'deterministic'
    elif uq_method == 'GMM':
        uq_label = 'GMM-uniform'
    elif uq_method == 'GMMD':
        uq_label = 'GMM-dynamic'
    return uq_label

def get_statistics(df=None, error_col=None, uncertainty_col=None):
    corr, p_value = spearmanr(df[error_col], df[uncertainty_col])
    pear_corr, pear_p_value = pearsonr(df[error_col], df[uncertainty_col])
    mi = float(mutual_info_regression(df[error_col].to_numpy().reshape(-1, 1),
                                      df[uncertainty_col].to_numpy())[0] )
    return (corr, p_value, pear_corr, pear_p_value, mi)

def adjust_model_name(all_checkpoints=None):
    file_path = all_checkpoints[0]
    filename = os.path.basename(file_path)
    dir_name = os.path.dirname(file_path)
    modified_filename = re.sub(r'_exp_\d+', '', filename)
    modified_path = os.path.join(dir_name, modified_filename)
    os.rename(file_path, modified_path)