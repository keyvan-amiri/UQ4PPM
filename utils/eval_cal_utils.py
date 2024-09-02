import re
import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import uncertainty_toolbox as uct
import torch
from torch.utils.data import TensorDataset, DataLoader
from models.dalstm import DALSTMModel, DALSTMModelMve
from models.stochastic_dalstm import StochasticDALSTM
from loss.loss_handler import set_loss
from evaluation import evaluate_coverage



def inf_val_union(args=None, cfg=None, result_path=None,
                  root_path=None, val_inference_path=None, report_path=None):
    """
    Handle inference on validation set for union-based approaches.
    Parameters
    ----------
    args : arguments provided by user for recalibration.py script
    cfg : configuration file used for training, and inference
    result_path : result folder for a model (e.g., DALSTM), and a dataset
    root_path : folder that main scripts are located in.
    val_inference_path : path to the dataframe which includes predictions for
    validation set.
    report_path : path to a .txt file for reporting recalibration time.
    Raises
    ------
    FileNotFoundError: in case embedding of the validation set is not saved
    during training in advance.
    Returns
    -------
    calibration_df : a dataframe includes all validation examples, with these 
    columns: 'GroundTruth', 'Prediction', 'Epistemic_Uncertainty'
    """
    
    print('Now: start inference on validation set:')
    start=datetime.now()   
    # get normalization mode
    normalization = cfg.get('data').get('normalization')    
    # check holdout/cv data split and get the fold
    fold = extract_fold(args.csv_file) 
    if args.model == 'dalstm':
        # get the path to dataset folder
        dataset_class = 'DALSTM_'+ args.dataset
        daaset_path = os.path.join(root_path, 'datasets', dataset_class)
        # get the normalization factor
        max_train_val_path = os.path.join(
            daaset_path, 'DALSTM_max_train_val_'+args.dataset+'.pkl')
        with open(max_train_val_path, 'rb') as f:
            y_scaler =  pickle.load(f)
        # get maximum length of feature vectors (possibly to use as batch size)
        max_len_path = os.path.join(
            daaset_path, 'DALSTM_max_len_'+args.dataset+'.pkl')
        with open(max_len_path, 'rb') as f:
            max_len =  pickle.load(f) 
        # get embdding, ground truth for validation set (based on data split)
        if fold == None:
            X_val_emb_path = os.path.join(
                daaset_path, 'DALSTM_X_val_emb_'+args.dataset+'.pt')
            y_val_path = os.path.join(
                daaset_path, 'DALSTM_y_val_'+args.dataset+'.pt')            
        else:
            X_val_emb_path = os.path.join(
                daaset_path, 'DALSTM_X_val_emb_fold_'+fold+args.dataset+'.pt')
            y_val_path = os.path.join(
                daaset_path, 'DALSTM_y_val_fold_'+fold+args.dataset+'.pt')
        # check whether embedding is already obtained
        if os.path.isfile(X_val_emb_path):
            X_val_emb = torch.load(X_val_emb_path)
            y_val = torch.load(y_val_path)
            # get relevant batch size
            try:
                calibration_batch_size = cfg.get('evaluation').get('batch_size')
            except:
                calibration_batch_size = max_len 
            # create validation set
            calibration_dataset = TensorDataset(X_val_emb, y_val)            
            calibration_loader = DataLoader(calibration_dataset,
                                            batch_size=calibration_batch_size,
                                            shuffle=False)
        else:
            raise FileNotFoundError('Random Forest must be fitted first')
   
    # load fitted random forest regressor
    aux_model_name = replace_suffix(
        args.csv_file, 'inference_result_.csv', 'best_model.pkl')
    aux_model_path = os.path.join(result_path, aux_model_name)
    with open(aux_model_path, 'rb') as file:
        aux_model = pickle.load(file) 
        
    # Now: inference on validation set    
    # create a dictionary to collect inference results
    res_dict = {'GroundTruth': [], 'Prediction': [], 
                'Epistemic_Uncertainty': []}
    for index, calib_batch in enumerate(calibration_loader):
        batch_embedding = calib_batch[0].detach().cpu().numpy()
        _y_truth = calib_batch[1].detach().cpu().numpy()
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
        res_dict['GroundTruth'].extend(_y_truth.tolist())
        res_dict['Prediction'].extend(_y_pred.tolist())
        res_dict['Epistemic_Uncertainty'].extend(epistemic_std.tolist())
    calibration_df = pd.DataFrame(res_dict)  
    inference_val_time = (datetime.now()-start).total_seconds()
    with open(report_path, 'w') as file:
        file.write('Inference on validation set took  {} seconds. \n'.format(
            inference_val_time))        
    calibration_df.to_csv(val_inference_path, index=False)    
    return calibration_df

# A method to extract number of ensemble members from the report .txt file
def get_num_models_from_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            # Check if the line contains the 'num_models' key
            if "'num_models':" in line:
                # Extract the part after 'num_models':
                try:
                    # Split the line at 'num_models' and then,
                    # split at the first occurrence of either ',' or '}'
                    num_str = line.split(
                        "'num_models':")[1].split(',')[0].split('}')[0].strip()
                    num_models = int(num_str)
                    return num_models
                except (ValueError, IndexError):
                    raise ValueError('Cannot find a valid num_models value.')
    raise ValueError('num_models not found in the file.')


# Extract model type, dataset, and Uncertainty Quantification from cfg file
def extract_info_from_cfg(cfg_file):
    """
    Note: it can only detect CARD model, for other UQ techniques a separate
    method (i.e., get_uq_method) is used.
    """
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
        raise ValueError('The configuration file name does not match \
                         the expected pattern.')

# A method t extract Uncertainty Quantification approach from csv file       
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

# A method to replace a customized suffix for any file        
def replace_suffix(input_string, old_suffix, new_suffix,
                   ensemble_mode=False, model_idx=None):
    """
    In case of ensemble models, it will add this string between the base_name
    and the new suffix: 'member_' + str(model_idx) + '_' this means:
    'member_1_', 'member_1_', and so on.
    """
    if input_string.endswith(old_suffix):
        if ensemble_mode:
            return input_string[:-len(old_suffix)] + 'member_' + str(model_idx) + '_' + new_suffix
        else:
            return input_string[:-len(old_suffix)] + new_suffix
    else:
        raise ValueError(
            'The input string does not end with the specified old suffix.')

# Extract fold number from a csv file name. (for cross-fold validation usage)        
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

# A method to add a customized suffix to a csv file name.
def add_suffix_to_csv(csv_file, added_suffix=None):
    # Check if the file name ends with .csv
    if csv_file.endswith('.csv'):
        # Insert the suffix before the .csv extension
        new_csv_file = csv_file[:-4] + added_suffix + '.csv'
        return new_csv_file
    else:
        raise ValueError("The file name does not end with .csv")

# A method to extract processed data and model dimensions.
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

# A method to extract model and loss type.
def get_model_and_loss(args=None, cfg=None, input_size=None, max_len=None, 
                       device=None, ensemble_mode=None, num_models=None):
    """
    Parameters
    ----------
    args : Argument for recalibration step.
    cfg : cfg file that is used for training and inference.
    input_size : Number of features. 
    max_len : Length of the sequences.
    device : Device that executes this script (e.g., GPU number)
    ensemble_mode : Whether UQ techniques is an ensemble approach or not.
    num_models : Number of ensembl members (if any)

    Returns
    -------
    model : In case of non-ensembl approaches (including all variations of 
    drop out approaximation, and MVE approach), it returns a predictive model
    that is used for training and inference. This model will be used to load 
    a checkpoint file (pre-trained model that is ready for inference.)
    criterion : The loss function that is used in Backpropagation.
    num_mcmc : Number of Monte-carlo samples that is used for inference with
    dropout approximation.
    normalization : Whether normalization is done on target attribute or not.
    model_list : In case of ensemble models, instead of one single pre-trained
    model a list of models (one for each ensemble member) is created to be
    later loaded with the relevant checkpoint file created during training.
    """

    # get other model characteristics
    hidden_size = cfg.get('model').get('lstm').get('hidden_size')
    n_layers = cfg.get('model').get('lstm').get('n_layers')
    dropout = cfg.get('model').get('lstm').get('dropout')  
    dropout_prob = cfg.get('model').get('lstm').get('dropout_prob')
    normalization = cfg.get('data').get('normalization')
    
    # set parameters for dropout approximation
    if (args.UQ == 'DA' or args.UQ == 'CDA' or args.UQ == 'DA_A' or 
          args.UQ == 'CDA_A'): 
        num_mcmc = cfg.get('uncertainty').get('dropout_approximation').get(
            'num_stochastic_forward_path')
        weight_regularizer = cfg.get('uncertainty').get(
            'dropout_approximation').get('weight_regularizer')
        dropout_regularizer = cfg.get('uncertainty').get(
            'dropout_approximation').get('dropout_regularizer')
        if (args.UQ == 'DA' or args.UQ == 'DA_A'):
            concrete_dropout = False
        else:
            concrete_dropout = True
    else:
        # to use the same execution path
        num_mcmc = None
    
    # define loss function (heteroscedastic/homoscedastic):
    if (args.UQ == 'DA' or args.UQ == 'CDA' or args.UQ == 'en_t' or
        args.UQ == 'en_b'):
        criterion = set_loss(loss_func=cfg.get('train').get('loss_function'))          
    elif (args.UQ == 'DA_A' or args.UQ == 'CDA_A' or args.UQ == 'mve' or
        args.UQ == 'en_t_mve' or args.UQ == 'en_b_mve'):
        criterion = set_loss(loss_func=cfg.get('train').get('loss_function'),
                             heteroscedastic=True)    

    # define model(s) based on UQ method
    # an emtpy list for ensemble of models
    model_list = []  # in case of single models remains empty
    if not ensemble_mode:
        if args.UQ == 'mve':
            model = DALSTMModelMve(
                input_size=input_size, hidden_size=hidden_size,
                n_layers=n_layers, max_len=max_len, dropout=dropout,
                p_fix=dropout_prob).to(device)
        elif (args.UQ == 'DA' or args.UQ == 'CDA'):
            # hs (heteroscedastic) is set to False
            model = StochasticDALSTM(
                input_size=input_size, hidden_size=hidden_size,
                n_layers=n_layers, max_len=max_len, dropout=True,
                concrete=concrete_dropout, p_fix=dropout_prob, 
                weight_regularizer=weight_regularizer,
                dropout_regularizer=dropout_regularizer, 
                hs=False, Bayes=True, device=device).to(device)
        elif (args.UQ == 'DA_A' or args.UQ == 'CDA_A'):
            # hs (heteroscedastic) is set to True
            model = StochasticDALSTM(
                input_size=input_size, hidden_size=hidden_size, 
                n_layers=n_layers, max_len=max_len, dropout=True,
                concrete=concrete_dropout, p_fix=dropout_prob, 
                weight_regularizer=weight_regularizer,
                dropout_regularizer=dropout_regularizer,
                hs=True, Bayes=True, device=device).to(device)
    else:
        #print(num_models)
        for i in range(num_models):
            if (args.UQ == 'en_t' or args.UQ == 'en_b'):
                model = DALSTMModel(
                    input_size=input_size, hidden_size=hidden_size,
                    n_layers=n_layers, max_len=max_len, dropout=dropout,
                    p_fix=dropout_prob).to(device)
            elif (args.UQ == 'en_t_mve' or args.UQ == 'en_b_mve'):
                model = DALSTMModelMve(
                    input_size=input_size, hidden_size=hidden_size,
                    n_layers=n_layers, max_len=max_len, dropout=dropout,
                    p_fix=dropout_prob).to(device)
            model_list.append(model)
    return (model, criterion, num_mcmc, normalization, model_list)


# inference for validation for DA, CDA,DA_A, CDA_A, mve approaches
def inference_on_validation(args=None, model=None, model_list=None,
                            checkpoint_path=None, checkpoint_paths_list=None,
                            calibration_loader=None, num_mc_samples=None,
                            normalization=False, y_scaler=None, device=None,
                            report_path=None, recalibration_path=None,
                            ensemble_mode=False, num_models=None):
        
    print('Now: start inference on validation set:')
    start=datetime.now()
    
    # setstructure of instance-level results
    if (args.UQ == 'DA' or args.UQ == 'CDA' or 
        args.UQ == 'en_t' or args.UQ == 'en_b'):
        res_dict = {'GroundTruth': [], 'Prediction': [],
                    'Epistemic_Uncertainty': []}
    elif (args.UQ == 'DA_A' or args.UQ == 'CDA_A' or
          args.UQ == 'en_t_mve' or args.UQ == 'en_b_mve'):
        res_dict = {'GroundTruth': [], 'Prediction': [], 
                    'Epistemic_Uncertainty': [], 'Aleatoric_Uncertainty': [],
                    'Total_Uncertainty': []} 
    elif args.UQ == 'mve':
        res_dict = {'GroundTruth': [], 'Prediction': [],
                    'Aleatoric_Uncertainty': []}
    
    # load the checkpoint(s) 
    if not ensemble_mode:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        # set model to evaluation mode
        model.eval()
    else:
        for i in range(num_models):
            # load one checkpoint in each iteration
            checkpoint = torch.load(checkpoint_paths_list[i])
            model_list[i].load_state_dict(checkpoint['model_state_dict'])
            # set the relevant model to evaluation mode
            model_list[i].eval()          

    # get instance-level results on validation set    
    with torch.no_grad():
        for index, calibration_batch in enumerate(calibration_loader):
            # get batch data
            inputs = calibration_batch[0].to(device)
            _y_truth = calibration_batch[1].to(device)  
            # get model outputs, and uncertainties
            if (args.UQ == 'DA' or args.UQ == 'CDA' or args.UQ == 'DA_A' or
                args.UQ == 'CDA_A'):
                means_list, logvar_list =[], []
                # conduct Monte Carlo sampling
                for i in range (num_mc_samples): 
                    # TODO: remove stop_dropout since for deterministic version we have a separate model
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
                if (args.UQ == 'DA_A' or args.UQ == 'CDA_A'):
                    stacked_log_var = torch.stack(logvar_list, dim=0)
                    stacked_var = torch.exp(stacked_log_var)
                    mean_var = torch.mean(stacked_var, dim=0)
                    aleatoric_std = torch.sqrt(mean_var).to(device)
                    # normalize aleatoric uncertainty if necessary
                    if normalization:
                        aleatoric_std = y_scaler * aleatoric_std
                    total_std = epistemic_std + aleatoric_std
            elif args.UQ == 'mve':
                _y_pred, log_var = model(inputs)
                aleatoric_std = torch.sqrt(torch.exp(log_var))
                # normalize aleatoric uncertainty if necessary
                if normalization:
                    aleatoric_std = y_scaler * aleatoric_std 
            elif (args.UQ == 'en_t' or args.UQ == 'en_b'):
                # empty list to collect predictions of all members of ensemble
                prediction_list = []
                for model_idx in range(num_models):
                    member_prediciton = model_list[model_idx](inputs)
                    prediction_list.append(member_prediciton)
                stacked_predictions = torch.stack(prediction_list, dim=0)
                # predited value is the average of predictions of all members
                _y_pred = torch.mean(stacked_predictions, dim=0)
                # epistemic uncertainty = std of predictions of all members
                epistemic_std = torch.std(stacked_predictions, dim=0).to(device)
                # normalize epistemic uncertainty if necessary
                if normalization:
                    epistemic_std = y_scaler * epistemic_std
            elif (args.UQ == 'en_t_mve' or args.UQ == 'en_b_mve'):
                # collect prediction means & aleatoric std: all ensemble members
                mean_pred_list, aleatoric_std_list = [], []
                for model_idx in range(num_models):
                    member_mean, member_log_var = model_list[model_idx](inputs)
                    member_aleatoric_std = torch.sqrt(torch.exp(member_log_var))
                    mean_pred_list.append(member_mean)
                    aleatoric_std_list.append(member_aleatoric_std)
                stacked_mean_pred = torch.stack(mean_pred_list, dim=0)
                stacked_aleatoric = torch.stack(aleatoric_std_list, dim=0)
                # predited value is the average of predictions of all members
                _y_pred = torch.mean(stacked_mean_pred, dim=0)
                # epistemic uncertainty = std of predictions of all members
                epistemic_std = torch.std(stacked_mean_pred, dim=0).to(device)
                # epistemic uncertainty = mean of aleatoric estimates of all members
                aleatoric_std = torch.mean(stacked_aleatoric, dim=0)
                # normalize uncertainties if necessary
                if normalization:
                    epistemic_std = y_scaler * epistemic_std
                    aleatoric_std = y_scaler * aleatoric_std
                total_std = epistemic_std + aleatoric_std
                
            # convert tragets, outputs in case of normalization
            if normalization:
                _y_truth = y_scaler * _y_truth
                _y_pred = y_scaler * _y_pred
            # Detach predictions and ground truths (np arrays)
            _y_truth = _y_truth.detach().cpu().numpy()
            _y_pred = _y_pred.detach().cpu().numpy()
            # collect inference result in all_result dict.
            res_dict['GroundTruth'].extend(_y_truth.tolist())
            res_dict['Prediction'].extend(_y_pred.tolist())
            if (args.UQ == 'DA' or args.UQ == 'CDA' or args.UQ == 'DA_A' or 
                args.UQ == 'CDA_A' or args.UQ == 'en_t' or args.UQ == 'en_b' or
                args.UQ == 'en_t_mve' or args.UQ == 'en_b_mve'):
                epistemic_std = epistemic_std.detach().cpu().numpy()
                res_dict['Epistemic_Uncertainty'].extend(epistemic_std.tolist()) 
                if (args.UQ == 'DA_A' or args.UQ == 'CDA_A' or 
                    args.UQ == 'en_t_mve' or args.UQ == 'en_b_mve'):
                    aleatoric_std = aleatoric_std.detach().cpu().numpy()
                    total_std = total_std.detach().cpu().numpy()
                    res_dict['Aleatoric_Uncertainty'].extend(aleatoric_std.tolist())
                    res_dict['Total_Uncertainty'].extend(total_std.tolist()) 
            elif args.UQ == 'mve':
                aleatoric_std = aleatoric_std.detach().cpu().numpy()
                res_dict['Aleatoric_Uncertainty'].extend(aleatoric_std.tolist())
    validation_df = pd.DataFrame(res_dict)
    val_csv_name = add_suffix_to_csv(args.csv_file, added_suffix='validation_')
    val_csv_path = os.path.join(recalibration_path, val_csv_name)
    validation_df.to_csv(val_csv_path, index=False)
    inference_val_time = (datetime.now()-start).total_seconds()
    with open(report_path, 'w') as file:
        file.write('Inference on validation set took  {} seconds. \n'.format(
            inference_val_time))    
    return validation_df

# utility method to compute PICP for recalibration using isotonic regression
def calculate_picp(df):
    in_interval = np.logical_and(df['GroundTruth'] >= df['confidence_lower'],
                                 df['GroundTruth'] <= df['confidence_upper'])
    picp = np.mean(in_interval)
    return picp

# utility method to compute MPIW for recalibration using isotonic regression
def calculate_mpiw(df):
    interval_widths = df['confidence_upper'] - df['confidence_lower']
    mpiw = np.mean(interval_widths)
    return mpiw


# A method for performance evaluation of recalibration techniques
def recalibration_evaluation (args=None, calibrated_test_def=None,
                              recal_model=None, recalibration_plot_path=None,
                              recalibration_result_path=None):
    
    # get name of the calibrated csv file witouht extension
    recal_name = add_suffix_to_csv(args.csv_file, added_suffix='recalibrated_')
    base_recal_name = os.path.splitext(recal_name)[0]   
    # get prediction mean and ground truth
    pred_mean = calibrated_test_def['Prediction'].values
    y_true = calibrated_test_def['GroundTruth'].values
    # get prediction standard deviation before recalibration
    if (args.UQ=='DA_A' or args.UQ=='CDA_A' or 
        args.UQ == 'en_t_mve' or args.UQ == 'en_b_mve'):
        pred_std = calibrated_test_def['Total_Uncertainty'].values 
    elif (args.UQ=='CARD' or args.UQ=='mve'):
        pred_std = calibrated_test_def['Aleatoric_Uncertainty'].values
    elif (args.UQ=='DA' or args.UQ=='CDA' or args.UQ == 'en_t' or 
          args.UQ == 'en_b' or args.UQ == 'RF' or args.UQ == 'LA'):
        pred_std = calibrated_test_def['Epistemic_Uncertainty'].values
         
    # Non-Gaussian calibration: expected proportions and observed proportions
    exp_props, obs_props = uct.metrics_calibration.get_proportion_lists_vectorized(
        pred_mean, pred_std, y_true, recal_model=recal_model) 
    
    # Create average calibration plot for recalibrated predictions
    uct.viz.plot_calibration(pred_mean, pred_std, y_true, exp_props=exp_props,
                             obs_props=obs_props)
    plt.gcf().set_size_inches(10, 10)
    new_file_name = base_recal_name + 'miscalibrated_area_isotonic_regression' + '.pdf'
    new_file_path = os.path.join(recalibration_plot_path, new_file_name)
    plt.savefig(new_file_path, format='pdf')
    plt.clf()
    
    # sort the calibrated predictions based on absolute error
    sorted_df = calibrated_test_def.sort_values(by='Absolute_error')
    sorted_pred_mean = sorted_df['Prediction'].values
    sorted_errors = sorted_df['Absolute_error'].values
    if (args.UQ=='DA_A' or args.UQ=='CDA_A' or 
        args.UQ == 'en_t_mve' or args.UQ == 'en_b_mve'):
        sorted_pred_std = sorted_df['Total_Uncertainty'].values 
    elif (args.UQ=='CARD' or args.UQ=='mve'):
        sorted_pred_std = sorted_df['Aleatoric_Uncertainty'].values
    elif (args.UQ=='DA' or args.UQ=='CDA' or args.UQ == 'en_t' or 
          args.UQ == 'en_b' or args.UQ == 'RF' or args.UQ == 'LA'):
        sorted_pred_std = sorted_df['Epistemic_Uncertainty'].values
    # now compare confidence intervals before and after calibration
    orig_bounds = uct.metrics_calibration.get_prediction_interval(
        sorted_pred_mean, sorted_pred_std, 0.95, None)    
    recal_bounds = uct.metrics_calibration.get_prediction_interval(
        sorted_pred_mean, sorted_pred_std, 0.95, recal_model)      
    plt.fill_between(sorted_errors, recal_bounds.lower, recal_bounds.upper,
                     color='orange', alpha=0.4, label='Recalibrated',
                     hatch='//', edgecolor='orange', zorder=1)
    plt.fill_between(sorted_errors, orig_bounds.lower, orig_bounds.upper,
                     color='blue', alpha=0.6, label='Before Calibration',
                     hatch='\\\\', edgecolor='blue', zorder=2)    
    plt.xlabel('Sorted Absolute Errors') 
    plt.ylabel('Confidence Intervals (95%)')    
    plt.legend()
    plt.gcf().set_size_inches(10, 10)
    plt.title('95% Centered Prediction Interval')
    new_file_name = (base_recal_name +
                     'confidence_intervals_isotonic_regression_error_based' + 
                     '.pdf')
    new_file_path = os.path.join(recalibration_plot_path, new_file_name)
    plt.savefig(new_file_path, format='pdf')
    plt.clf()
    
    # sort the calibrated predictions based on prefix length
    sorted_df = calibrated_test_def.sort_values(by='Prefix_length')
    sorted_pred_mean = sorted_df['Prediction'].values
    sorted_lengths = sorted_df['Prefix_length'].values
    if (args.UQ=='DA_A' or args.UQ=='CDA_A' or 
        args.UQ == 'en_t_mve' or args.UQ == 'en_b_mve'):
        sorted_pred_std = sorted_df['Total_Uncertainty'].values 
    elif (args.UQ=='CARD' or args.UQ=='mve'):
        sorted_pred_std = sorted_df['Aleatoric_Uncertainty'].values
    elif (args.UQ=='DA' or args.UQ=='CDA' or args.UQ == 'en_t' or
          args.UQ == 'en_b' or args.UQ == 'RF' or args.UQ == 'LA'):
        sorted_pred_std = sorted_df['Epistemic_Uncertainty'].values
    # now compare confidence intervals before and after calibration
    orig_bounds = uct.metrics_calibration.get_prediction_interval(
        sorted_pred_mean, sorted_pred_std, 0.95, None)    
    recal_bounds = uct.metrics_calibration.get_prediction_interval(
        sorted_pred_mean, sorted_pred_std, 0.95, recal_model)      
    plt.fill_between(sorted_lengths, recal_bounds.lower, recal_bounds.upper,
                     color='orange', alpha=0.4, label='Recalibrated',
                     hatch='//', edgecolor='orange', zorder=1)
    plt.fill_between(sorted_lengths, orig_bounds.lower, orig_bounds.upper,
                     color='blue', alpha=0.6, label='Before Calibration',
                     hatch='\\\\', edgecolor='blue', zorder=2)    
    plt.xlabel('Sorted Prefix Lengths') 
    plt.ylabel('Confidence Intervals (95%)')    
    plt.legend()
    plt.gcf().set_size_inches(10, 10)
    plt.title('95% Centered Prediction Interval')
    new_file_name = (base_recal_name +
                     'confidence_intervals_isotonic_regression_length_based' + 
                     '.pdf')
    new_file_path = os.path.join(recalibration_plot_path, new_file_name)
    plt.savefig(new_file_path, format='pdf')
    plt.clf()
    
    # sort the calibrated predictions based on ground truth (remaining time)
    sorted_df = calibrated_test_def.sort_values(by='GroundTruth')
    sorted_pred_mean = sorted_df['Prediction'].values
    sorted_rem_time = sorted_df['GroundTruth'].values
    if (args.UQ=='DA_A' or args.UQ=='CDA_A' or 
        args.UQ == 'en_t_mve' or args.UQ == 'en_b_mve'):
        sorted_pred_std = sorted_df['Total_Uncertainty'].values 
    elif (args.UQ=='CARD' or args.UQ=='mve'):
        sorted_pred_std = sorted_df['Aleatoric_Uncertainty'].values
    elif (args.UQ=='DA' or args.UQ=='CDA' or args.UQ == 'en_t' or 
          args.UQ == 'en_b' or args.UQ == 'RF' or args.UQ == 'LA'):
        sorted_pred_std = sorted_df['Epistemic_Uncertainty'].values
    # now compare confidence intervals before and after calibration
    orig_bounds = uct.metrics_calibration.get_prediction_interval(
        sorted_pred_mean, sorted_pred_std, 0.95, None)    
    recal_bounds = uct.metrics_calibration.get_prediction_interval(
        sorted_pred_mean, sorted_pred_std, 0.95, recal_model)      
    plt.fill_between(sorted_rem_time, recal_bounds.lower, recal_bounds.upper,
                     color='orange', alpha=0.4, label='Recalibrated',
                     hatch='//', edgecolor='orange', zorder=1)
    plt.fill_between(sorted_rem_time, orig_bounds.lower, orig_bounds.upper,
                     color='blue', alpha=0.6, label='Before Calibration',
                     hatch='\\\\', edgecolor='blue', zorder=2)    
    plt.xlabel('Sorted Remaining Times') 
    plt.ylabel('Confidence Intervals (95%)')    
    plt.legend()
    plt.gcf().set_size_inches(10, 10)
    plt.title('95% Centered Prediction Interval')
    new_file_name = (base_recal_name +
                     'confidence_intervals_isotonic_regression_remainingtime_based' + 
                     '.pdf')
    new_file_path = os.path.join(recalibration_plot_path, new_file_name)
    plt.savefig(new_file_path, format='pdf')
    plt.clf()
    
    # Compute PICP and MPIW for isotonic regression calibration
    picp = calculate_picp(calibrated_test_def)
    mpiw = calculate_mpiw(calibrated_test_def)
    new_file_name = base_recal_name + 'pcip_mpiw_isotonic_regression' + '.txt'
    new_file_path = os.path.join(recalibration_result_path, new_file_name)
    with open(new_file_path, 'w') as file:
        file.write(f"Prediction Interval Coverage Probability (PICP): {picp}\n")
        file.write(f"Mean Prediction Interval Width (MPIW): {mpiw}\n")   
    
    # Now plot miscalibration for Gaussian calibrations
    pred_std_miscal = calibrated_test_def['calibrated_std_miscal'].values
    uct.viz.plot_calibration(pred_mean, pred_std_miscal, y_true)
    plt.gcf().set_size_inches(10, 10)
    new_file_name = base_recal_name + 'miscalibrated_area_std_miscal' + '.pdf'
    new_file_path = os.path.join(recalibration_plot_path, new_file_name)
    plt.savefig(new_file_path, format='pdf')
    plt.clf()
    pred_std_rms_cal = calibrated_test_def['calibrated_std_rms_cal'].values
    uct.viz.plot_calibration(pred_mean, pred_std_rms_cal, y_true)
    plt.gcf().set_size_inches(10, 10)
    new_file_name = base_recal_name + 'miscalibrated_area_std_rms_cal' + '.pdf'
    new_file_path = os.path.join(recalibration_plot_path, new_file_name)
    plt.savefig(new_file_path, format='pdf')
    plt.clf()
    pred_std_ma_cal = calibrated_test_def['calibrated_std_ma_cal'].values
    uct.viz.plot_calibration(pred_mean, pred_std_ma_cal, y_true)
    plt.gcf().set_size_inches(10, 10)
    new_file_name = base_recal_name + 'miscalibrated_area_std_ma_cal' + '.pdf'
    new_file_path = os.path.join(recalibration_plot_path, new_file_name)
    plt.savefig(new_file_path, format='pdf')
    plt.clf()
    
    # Plot adversarial group calibration for Gaussian calibrations
    uct.viz.plot_adversarial_group_calibration(pred_mean, pred_std_miscal, y_true)
    plt.gcf().set_size_inches(10, 6)
    new_file_name = base_recal_name + 'adversarial_group_calibration_std_miscal' + '.pdf'
    new_file_path = os.path.join(recalibration_plot_path, new_file_name)
    plt.savefig(new_file_path, format='pdf')
    plt.clf()
    uct.viz.plot_adversarial_group_calibration(pred_mean, pred_std_rms_cal, y_true)
    plt.gcf().set_size_inches(10, 6)
    new_file_name = base_recal_name + 'adversarial_group_calibration_std_rms_cal' + '.pdf'
    new_file_path = os.path.join(recalibration_plot_path, new_file_name)
    plt.savefig(new_file_path, format='pdf')
    plt.clf()
    uct.viz.plot_adversarial_group_calibration(pred_mean, pred_std_ma_cal, y_true)
    plt.gcf().set_size_inches(10, 6)
    new_file_name = base_recal_name + 'adversarial_group_calibration_std_ma_cal' + '.pdf'
    new_file_path = os.path.join(recalibration_plot_path, new_file_name)
    plt.savefig(new_file_path, format='pdf')
    plt.clf()
    
    # Plot ordered prediction intervals for Gaussian calibrations
    uct.viz.plot_intervals_ordered(pred_mean, pred_std_miscal, y_true)
    plt.gcf().set_size_inches(10, 10)
    # define name of the plot to be saved
    new_file_name = base_recal_name + 'ordered_prediction_intervals_std_miscal' + '.pdf'
    new_file_path = os.path.join(recalibration_plot_path, new_file_name)
    plt.savefig(new_file_path, format='pdf')
    plt.clf()
    uct.viz.plot_intervals_ordered(pred_mean, pred_std_rms_cal, y_true)
    plt.gcf().set_size_inches(10, 10)
    # define name of the plot to be saved
    new_file_name = base_recal_name + 'ordered_prediction_intervals_std_rms_cal' + '.pdf'
    new_file_path = os.path.join(recalibration_plot_path, new_file_name)
    plt.savefig(new_file_path, format='pdf')
    plt.clf()
    uct.viz.plot_intervals_ordered(pred_mean, pred_std_ma_cal, y_true)
    plt.gcf().set_size_inches(10, 10)
    # define name of the plot to be saved
    new_file_name = base_recal_name + 'ordered_prediction_intervals_std_ma_cal' + '.pdf'
    new_file_path = os.path.join(recalibration_plot_path, new_file_name)
    plt.savefig(new_file_path, format='pdf')
    plt.clf()
    
    # Get all uncertainty quantification metrics for std_miscal
    uq_metrics = uct.metrics.get_all_metrics(pred_mean, pred_std_miscal, y_true)
    new_file_name = base_recal_name + 'uq_metrics_std_miscal' + '.txt'
    new_file_path = os.path.join(recalibration_result_path, new_file_name)
    with open(new_file_path, 'w') as file:
        # Iterate over the dictionary items and write them to the file
        for key, value in uq_metrics.items():
            file.write(f"{key}: {value}\n")
    # get PICP for all uncertainty quantfaction approaches
    picp, mpiw, qice, y_b_0, y_a_100 = evaluate_coverage(
        y_true=y_true, pred_mean=pred_mean, pred_std=pred_std_miscal,
        low_percentile=2.5, high_percentile=97.5, num_samples= 50,
        n_bins=10)
    with open(new_file_path, 'a') as file:
        file.write(f"Prediction Interval Coverage Probability (PICP): {picp}\n")
        file.write(f"Mean Prediction Interval Width (MPIW): {mpiw}\n")
        file.write(f"Quantile Interval Coverage Error (QICE): {qice}\n")
        file.write(
            f"We have {y_b_0} true remaining times smaller than predicted"
            f"lower bound.\n")
        file.write(
            f"We have {y_a_100} true remaining times greater than predicted"
            f"upper bound.\n") 

    # Get all uncertainty quantification metrics for std_rms_cal
    uq_metrics = uct.metrics.get_all_metrics(pred_mean, pred_std_rms_cal, y_true)
    new_file_name = base_recal_name + 'uq_metrics_std_rms_cal' + '.txt'
    new_file_path = os.path.join(recalibration_result_path, new_file_name)
    with open(new_file_path, 'w') as file:
        # Iterate over the dictionary items and write them to the file
        for key, value in uq_metrics.items():
            file.write(f"{key}: {value}\n")
    # get PICP for all uncertainty quantfaction approaches
    picp, mpiw, qice, y_b_0, y_a_100 = evaluate_coverage(
        y_true=y_true, pred_mean=pred_mean, pred_std=pred_std_rms_cal,
        low_percentile=2.5, high_percentile=97.5, num_samples= 50,
        n_bins=10)
    with open(new_file_path, 'a') as file:
        file.write(f"Prediction Interval Coverage Probability (PICP): {picp}\n")
        file.write(f"Mean Prediction Interval Width (MPIW): {mpiw}\n")
        file.write(f"Quantile Interval Coverage Error (QICE): {qice}\n")
        file.write(
            f"We have {y_b_0} true remaining times smaller than predicted"
            f"lower bound.\n")
        file.write(
            f"We have {y_a_100} true remaining times greater than predicted"
            f"upper bound.\n") 

    # Get all uncertainty quantification metrics for std_ma_cal
    uq_metrics = uct.metrics.get_all_metrics(pred_mean, pred_std_ma_cal, y_true)
    new_file_name = base_recal_name + 'uq_metrics_std_ma_cal' + '.txt'
    new_file_path = os.path.join(recalibration_result_path, new_file_name)
    with open(new_file_path, 'w') as file:
        # Iterate over the dictionary items and write them to the file
        for key, value in uq_metrics.items():
            file.write(f"{key}: {value}\n")
    # get PICP for all uncertainty quantfaction approaches
    picp, mpiw, qice, y_b_0, y_a_100 = evaluate_coverage(
        y_true=y_true, pred_mean=pred_mean, pred_std=pred_std_ma_cal,
        low_percentile=2.5, high_percentile=97.5, num_samples= 50,
        n_bins=10)
    with open(new_file_path, 'a') as file:
        file.write(f"Prediction Interval Coverage Probability (PICP): {picp}\n")
        file.write(f"Mean Prediction Interval Width (MPIW): {mpiw}\n")
        file.write(f"Quantile Interval Coverage Error (QICE): {qice}\n")
        file.write(
            f"We have {y_b_0} true remaining times smaller than predicted"
            f"lower bound.\n")
        file.write(
            f"We have {y_a_100} true remaining times greater than predicted"
            f"upper bound.\n") 

# Get prediction means, standard deviations, ground truth from inference result        
def get_mean_std_truth (df=None, uq_method=None):
    pred_mean = df['Prediction'].values 
    y_true = df['GroundTruth'].values
    if (uq_method=='DA_A' or uq_method=='CDA_A' or
        uq_method == 'en_t_mve' or uq_method == 'en_b_mve'):
        pred_std = df['Total_Uncertainty'].values 
    elif (uq_method=='CARD' or uq_method=='mve'):
        pred_std = df['Aleatoric_Uncertainty'].values
    elif (uq_method=='DA' or uq_method=='CDA' or uq_method == 'en_t' or
          uq_method == 'en_b' or uq_method == 'RF' or uq_method == 'LA'):
        pred_std = df['Epistemic_Uncertainty'].values
    else:
        raise NotImplementedError(
            'Uncertainty quantification {} not understood.'.format(uq_method))
    return (pred_mean, pred_std, y_true)

# A method to prepare arguments for recalibration on CARD model
def prepare_args (args=None, result_path=None, root_path=None):
    # set remaining arguments for smooth utilization of CARD model
    torch.set_printoptions(sci_mode=False)
    if args.model == 'pgtnet':
        torch.backends.cuda.matmul.allow_tf32 = True 
        torch.backends.cudnn.allow_tf32 = True 
    args.test = True
    args.recalibration = True
    args.thread = 4
    args.instance_path = result_path
    exp_path = os.path.join(result_path, 'card')
    args.exp = exp_path 
    args.doc =  args.model + '_' + args.dataset + '_card' 
    config_path = os.path.join(exp_path, 'logs/')
    args.config = config_path        
    pattern = r'_(holdout|cv)(?:_fold(\d+))?_seed_(\d+)_'                  
    match = re.search(pattern, args.csv_file)
    if match:
        # get the data split type
        args.split_mode = match.group(1)
        if args.split_mode == 'holdout':
            args.n_splits = 1
        else:
            args.n_splits = 5
        # get the relevant split number
        split_value = match.group(2)
        if split_value is not None:
            args.split = int(split_value)
        else:
            args.split = 0
        seed_value = match.group(3) 
        args.seed = [seed_value]
    # load dimensions of the model and add them to args
    if args.model == 'dalstm':
        dalstm_class = 'DALSTM_' + args.dataset
        x_dim_path = os.path.join(root_path, 'datasets', dalstm_class, 
                                  f'DALSTM_input_size_{args.dataset}.pkl')            
        with open(x_dim_path, 'rb') as file:
            args.x_dim = pickle.load(file)            
        max_len_path = os.path.join(root_path, 'datasets', dalstm_class,
                                f'DALSTM_max_len_{args.dataset}.pkl')
        with open(max_len_path, 'rb') as file:
            args.max_len = pickle.load(file) 
    # TODO: get all necessary sizes for pgtnet model
    if args.model == 'pgtnet':
        pass
    original_doc = args.doc
    original_config = args.config
    args.doc = original_doc + '/split_' + str(args.split)
    args.config = original_config + args.doc + '/config.yml'
    
    return args