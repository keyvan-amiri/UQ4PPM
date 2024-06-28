import glob
import re
import os
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader
from models.dalstm import DALSTMModelMve
from models.stochastic_dalstm import StochasticDALSTM
from loss.loss_handler import set_loss

# A mothod to retrieve all results for a combination of dataset-model
def get_csv_files(folder_path):
    # Get all csv files in the folder
    all_csv_files = glob.glob(os.path.join(folder_path, '*.csv'))    
    # Filter out files containing 'deterministic' in their names
    filtered_csvs = [f for f in all_csv_files if 
                          'deterministic' not in os.path.basename(f).lower()]
    filtered_csv_files = [f for f in filtered_csvs if 
                          'validation_calibrated' not in os.path.basename(f).lower()]
    # Collect name of uncertainty quantification approaches
    prefixes = []
    pattern = re.compile(r'(.*?)(_holdout_|_cv_)')
    for file_path in filtered_csv_files:
        file_name = os.path.basename(file_path)
        match = pattern.match(file_name)
        if match:
            prefixes.append(match.group(1))   
    return filtered_csv_files, prefixes


def extract_info_from_cfg(cfg_file):
    # Define the regex pattern
    pattern = r"^(?P<model>[a-zA-Z0-9]+)_(?P<dataset>[a-zA-Z0-9]+)(?:_(?P<uq_method>[a-zA-Z0-9]+))?\.(?:yaml|yml)$"    
    # Match the pattern against the cfg_file
    match = re.match(pattern, cfg_file)    
    if match:
        model = match.group('model')
        dataset = match.group('dataset')
        uq_method = match.group('uq_method')  
        if uq_method!=None:
            uq_method=uq_method.upper()
        return model, dataset, uq_method
    else:
        raise ValueError('The configuration file name does not match the expected pattern.')
        
def get_uq_method(csv_file):
    # Define the regex pattern to capture uq_method
    pattern = r"^(?P<uq_method>.+?)_(holdout|cv)_.*\.csv$"
    
    # Match the pattern against the csv_file
    match = re.match(pattern, csv_file)
    
    if match:
        uq_method = match.group('uq_method')
        return uq_method
    else:
        raise ValueError("The CSV file name does not match the expected pattern.")
        
def replace_suffix(input_string, old_suffix, new_suffix):
    if input_string.endswith(old_suffix):
        return input_string[:-len(old_suffix)] + new_suffix
    else:
        raise ValueError(
            'The input string does not end with the specified old suffix.')
        
def extract_fold(csv_file):
    # Define the regex pattern with fold
    pattern_with_fold = r"^[^_]+_[^_]+_cv_fold(?P<fold>\d+)_seed_\d+_inference_result_\.csv$"
    
    # Match the pattern against the csv_file
    match = re.match(pattern_with_fold, csv_file)
    
    if match:
        fold = match.group('fold')
        return fold
    else:
        return None

def add_suffix_to_csv(csv_file, added_suffix=None):
    # Check if the file name ends with .csv
    if csv_file.endswith('.csv'):
        # Insert the suffix before the .csv extension
        new_csv_file = csv_file[:-4] + added_suffix + '.csv'
        return new_csv_file
    else:
        raise ValueError("The file name does not end with .csv")
    
def get_validation_data_and_model_size(args=None, cfg=None, root_path=None):
    # check holdout/cv data split and get the fold
    fold = extract_fold(args.csv_file)    
    if args.model == 'dalstm':
        dataset_class = 'DALSTM_'+ args.dataset
        daaset_path = os.path.join(root_path, 'datasets', dataset_class)
        if fold == None:
            X_val_path = os.path.join(
                daaset_path, 'DALSTM_X_val_'+args.dataset+'.pt')
            y_val_path = os.path.join(
                daaset_path, 'DALSTM_y_val_'+args.dataset+'.pt')
        else:
            X_val_path = os.path.join(
                daaset_path, 'DALSTM_X_val_fold_'+fold+args.dataset+'.pt')
            y_val_path = os.path.join(
                daaset_path, 'DALSTM_y_val_fold_'+fold+args.dataset+'.pt')
        # load validation data
        X_val = torch.load(X_val_path)
        y_val = torch.load(y_val_path)
        # load model's dimensions
        input_size_path = os.path.join(
            daaset_path, 'DALSTM_input_size_'+args.dataset+'.pkl')
        max_len_path = os.path.join(
            daaset_path, 'DALSTM_max_len_'+args.dataset+'.pkl')
        max_train_val_path = os.path.join(
            daaset_path, 'DALSTM_max_train_val_'+args.dataset+'.pkl')
        mean_train_val_path = os.path.join(
            daaset_path, 'DALSTM_mean_train_val_'+args.dataset+'.pkl')
        median_train_val_path = os.path.join(
            daaset_path, 'DALSTM_median_train_val_'+args.dataset+'.pkl')
        with open(input_size_path, 'rb') as f:
            input_size =  pickle.load(f)
        with open(max_len_path, 'rb') as f:
            max_len =  pickle.load(f) 
        with open(max_train_val_path, 'rb') as f:
            max_train_val =  pickle.load(f) 
        with open(mean_train_val_path, 'rb') as f:
            mean_train_val =  pickle.load(f)
        with open(median_train_val_path, 'rb') as f:
            median_train_val =  pickle.load(f)
        # get relevant batch size
        try:
            calibration_batch_size = cfg.get('evaluation').get('batch_size')
        except:
            calibration_batch_size = max_len 
        # create validation set
        calibration_dataset = TensorDataset(X_val, y_val)
        calibration_loader = DataLoader(calibration_dataset,
                                        batch_size=calibration_batch_size,
                                        shuffle=False)
        
        return (calibration_loader, input_size, max_len, max_train_val,
                mean_train_val, median_train_val)
     
    #TODO: to complete this part for other architectures
    elif args.model == 'pgtnet':
        dataset_class = 'PGTNet_'+ args.dataset

        return None


def get_model_and_loss(args=None, cfg=None, input_size=None, max_len=None, 
                       device=None):

    # get other model characteristics
    hidden_size = cfg.get('model').get('lstm').get('hidden_size')
    n_layers = cfg.get('model').get('lstm').get('n_layers')
    dropout = cfg.get('model').get('lstm').get('dropout')  
    dropout_prob = cfg.get('model').get('lstm').get('dropout_prob')
    normalization = cfg.get('data').get('normalization')
    
    # define model, loss function, and other important variables                            
    if args.UQ == 'mve':
        model = DALSTMModelMve(input_size=input_size, hidden_size=hidden_size,
                               n_layers=n_layers, max_len=max_len,
                               dropout=dropout, p_fix=dropout_prob).to(device)
        # use heteroscedastic loss funciton for MVE approach
        criterion = set_loss(loss_func=cfg.get('train').get('loss_function'),
                             heteroscedastic=True)  
        heteroscedastic = False 
        num_mcmc = None
    elif (args.UQ == 'DA' or args.UQ == 'CDA' or args.UQ == 'DA_A' or 
          args.UQ == 'CDA_A'):
        dropout = True
        Bayes = True
        num_mcmc = cfg.get('uncertainty').get('dropout_approximation').get(
            'num_stochastic_forward_path')
        weight_regularizer = cfg.get('uncertainty').get(
            'dropout_approximation').get('weight_regularizer')
        dropout_regularizer = cfg.get('uncertainty').get(
            'dropout_approximation').get('dropout_regularizer')
        # Set the parameter for concrete dropout
        if (args.UQ == 'DA' or args.UQ == 'DA_A'):
            concrete_dropout = False
        else:
            concrete_dropout = True
        # Set the loss function (heteroscedastic/homoscedastic)
        if (args.UQ == 'DA' or args.UQ == 'CDA'):
            heteroscedastic = False
            criterion = set_loss(loss_func=cfg.get('train').get('loss_function'))
        else:
            heteroscedastic = True
            criterion = set_loss(loss_func=cfg.get('train').get('loss_function'),
                                 heteroscedastic=True) 
            model = StochasticDALSTM(input_size=input_size, hidden_size=hidden_size,
                                     n_layers=n_layers, max_len=max_len,
                                     dropout=dropout, concrete=concrete_dropout,
                                     p_fix=dropout_prob, 
                                     weight_regularizer=weight_regularizer,
                                     dropout_regularizer=dropout_regularizer,
                                     hs=heteroscedastic, Bayes=Bayes, 
                                     device=device).to(device)
            
    return (model, criterion, heteroscedastic, num_mcmc, normalization)