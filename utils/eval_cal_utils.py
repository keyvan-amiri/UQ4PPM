import glob
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
from models.dalstm import DALSTMModelMve
from models.stochastic_dalstm import StochasticDALSTM
from loss.loss_handler import set_loss
from evaluation import evaluate_coverage



# A mothod to retrieve all results for a combination of dataset-model
def get_csv_files(folder_path):
    # Get all csv files in the folder
    all_csv_files = glob.glob(os.path.join(folder_path, '*.csv'))    
    # Filter out files containing 'deterministic' in their names
    filtered_csv_files = [f for f in all_csv_files if 
                          'deterministic' not in os.path.basename(f).lower()]
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

def inference_on_validation(args=None, model=None, checkpoint_path=None,
                            calibration_loader=None, heteroscedastic=None,
                            num_mc_samples=None, normalization=False, 
                            y_scaler=None, device=None, report_path=None,
                            recalibration_path=None):
        
    print('Now: start inference on validation set:')
    start=datetime.now()
    
    # setstructure of instance-level results
    if (args.UQ == 'DA' or args.UQ == 'CDA'):
        res_dict = {'GroundTruth': [], 'Prediction': [],
                    'Epistemic_Uncertainty': []}
    elif (args.UQ == 'DA_A' or args.UQ == 'CDA_A'):
        res_dict = {'GroundTruth': [], 'Prediction': [], 
                    'Epistemic_Uncertainty': [], 'Aleatoric_Uncertainty': [],
                    'Total_Uncertainty': []} 
    elif args.UQ == 'mve':
        res_dict = {'GroundTruth': [], 'Prediction': [],
                    'Aleatoric_Uncertainty': []}
    
    # load the checkpoint  
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # get instance-level results on validation set
    model.eval()
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
            elif args.UQ == 'mve':
                _y_pred, log_var = model(inputs)
                aleatoric_std = torch.sqrt(torch.exp(log_var))
                # normalize aleatoric uncertainty if necessary
                if normalization:
                    aleatoric_std = y_scaler * aleatoric_std            
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
                args.UQ == 'CDA_A'):
                epistemic_std = epistemic_std.detach().cpu().numpy()
                res_dict['Epistemic_Uncertainty'].extend(epistemic_std.tolist()) 
                if heteroscedastic:
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
    if (args.UQ=='DA_A' or args.UQ=='CDA_A'):
        pred_std = calibrated_test_def['Total_Uncertainty'].values 
    elif (args.UQ=='CARD' or args.UQ=='mve'):
        pred_std = calibrated_test_def['Aleatoric_Uncertainty'].values
    elif (args.UQ=='DA' or args.UQ=='CDA'):
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
    if (args.UQ=='DA_A' or args.UQ=='CDA_A'):
        sorted_pred_std = sorted_df['Total_Uncertainty'].values 
    elif (args.UQ=='CARD' or args.UQ=='mve'):
        sorted_pred_std = sorted_df['Aleatoric_Uncertainty'].values
    elif (args.UQ=='DA' or args.UQ=='CDA'):
        sorted_pred_std = sorted_df['Epistemic_Uncertainty'].values
    # now compare confidence intervals before and after calibration
    orig_bounds = uct.metrics_calibration.get_prediction_interval(
        sorted_pred_mean, sorted_pred_std, 0.95, None)    
    recal_bounds = uct.metrics_calibration.get_prediction_interval(
        sorted_pred_mean, sorted_pred_std, 0.95, recal_model)    
    plt.fill_between(sorted_errors, orig_bounds.lower, orig_bounds.upper,
                     color= 'blue', alpha=0.6, label='Before Calibration')
    plt.fill_between(sorted_errors, recal_bounds.lower, recal_bounds.upper,
                     color= 'orange', alpha=0.4, label='Recalibrated')
    plt.xlabel('Sorted Absolute Errors') 
    plt.ylabel('Confidence Intervals (95%)')    
    plt.legend()
    plt.gcf().set_size_inches(10, 10)
    plt.title('95% Centered Prediction Interval')
    new_file_name = base_recal_name + 'confidence_intervals_isotonic_regression' + '.pdf'
    new_file_path = os.path.join(recalibration_plot_path, new_file_name)
    plt.savefig(new_file_path, format='pdf')
    plt.clf()
    
    # Compute PICP and MPIW for isotonic regression calibration
    picp = calculate_picp(calibrated_test_def)
    mpiw = calculate_mpiw(calibrated_test_def)
    new_file_name = base_recal_name + 'pcip_mpiw_isotonic_regression' + '.txt'
    new_file_path = os.path.join(recalibration_plot_path, new_file_name)
    with open(new_file_path, 'w') as file:
        file.write(f"Prediction Interval Coverage Probability (PICP): {picp}\n")
        file.write(f"Mean Prediction Interval Width (MPIW): {mpiw}\n")   
    
    # Now average calibration for Gaussian calibrations
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
    new_file_path = os.path.join(recalibration_plot_path, new_file_name)
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
            f"We have {y_b_0} true remaining times smaller than min of "
            f"generated remaining time predictions.\n")
        file.write(
            f"We have {y_a_100} true remaining times greater than max of "
            f"generated remaining time predictions.\n") 

    # Get all uncertainty quantification metrics for std_rms_cal
    uq_metrics = uct.metrics.get_all_metrics(pred_mean, pred_std_rms_cal, y_true)
    new_file_name = base_recal_name + 'uq_metrics_std_rms_cal' + '.txt'
    new_file_path = os.path.join(recalibration_plot_path, new_file_name)
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
            f"We have {y_b_0} true remaining times smaller than min of "
            f"generated remaining time predictions.\n")
        file.write(
            f"We have {y_a_100} true remaining times greater than max of "
            f"generated remaining time predictions.\n") 

    # Get all uncertainty quantification metrics for std_ma_cal
    uq_metrics = uct.metrics.get_all_metrics(pred_mean, pred_std_ma_cal, y_true)
    new_file_name = base_recal_name + 'uq_metrics_std_ma_cal' + '.txt'
    new_file_path = os.path.join(recalibration_plot_path, new_file_name)
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
            f"We have {y_b_0} true remaining times smaller than min of "
            f"generated remaining time predictions.\n")
        file.write(
            f"We have {y_a_100} true remaining times greater than max of "
            f"generated remaining time predictions.\n")

# Get prediction means, standard deviations, ground truth from inference result        
def get_mean_std_truth (df=None, uq_method=None):
    pred_mean = df['Prediction'].values 
    y_true = df['GroundTruth'].values
    if (uq_method=='DA_A' or uq_method=='CDA_A'):
        pred_std = df['Total_Uncertainty'].values 
    elif (uq_method=='CARD' or uq_method=='mve'):
        pred_std = df['Aleatoric_Uncertainty'].values
    elif (uq_method=='DA' or uq_method=='CDA'):
        pred_std = df['Epistemic_Uncertainty'].values
    else:
        raise NotImplementedError(
            'Uncertainty quantification {} not understood.'.format(uq_method))
    return (pred_mean, pred_std, y_true)

