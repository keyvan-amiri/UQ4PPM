"""
To prepare this script we used uncertainty tool-box which can be find in:
    https://uncertainty-toolbox.github.io/about/
"""

import argparse
import os
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import torch
import uncertainty_toolbox as uct
from utils.eval_cal_utils import (extract_info_from_cfg, replace_suffix,
                                  get_validation_data_and_model_size,
                                  get_model_and_loss, get_uq_method,
                                  add_suffix_to_csv, get_mean_std_truth)
from utils.eval_cal_utils import inference_on_validation
from utils.eval_cal_utils import recalibration_evaluation



def recalibration_on_test(args=None, calibration_df=None, test_df=None,
                          confidence_level=0.95, report_path=None,
                          recalibration_path=None):
    """
    recalibration is done based on two approaches:
    1) Isotonic regression remaps quantiles of the original distribution. The
    recalibrated distribution is unikely to be Gaussian. But, it still can
    provide confidence intervals (confidence_level).
    2) Scaling factor for the standard deviation is computed based on three
    different metrics: constrains the recalibrated distribution to be Gaussian.
    """
    print('Now: start recalibration:')
    start=datetime.now()
    # get prediction means, standard deviations, and ground truths for val set
    (pred_mean, pred_std, y_true
     ) = get_mean_std_truth(df=calibration_df, uq_method=args.UQ)
    
    # Gaussian calibration on validation set
    # Compute scaling factor for the standard deviation
    miscal_std_scaling = uct.recalibration.optimize_recalibration_ratio(
      pred_mean, pred_std, y_true, criterion="miscal")
    rms_cal_std_scaling = uct.recalibration.optimize_recalibration_ratio(
      pred_mean, pred_std, y_true, criterion="rms_cal")
    ma_cal_std_scaling = uct.recalibration.optimize_recalibration_ratio(
      pred_mean, pred_std, y_true, criterion="ma_cal")
    # get prediction means, standard deviations, and ground truths for test set
    (test_pred_mean, test_pred_std, test_y_true
     ) = get_mean_std_truth(df=test_df, uq_method=args.UQ)
    # Apply the scaling factors to get recalibrated standard deviations
    miscal_test_pred_std = miscal_std_scaling * test_pred_std
    rms_cal_test_pred_std = rms_cal_std_scaling * test_pred_std
    ma_cal_test_pred_std = ma_cal_std_scaling * test_pred_std
    test_df['calibrated_std_miscal'] = miscal_test_pred_std 
    test_df['calibrated_std_rms_cal'] = rms_cal_test_pred_std
    test_df['calibrated_std_ma_cal'] = ma_cal_test_pred_std
    
    # Gaussian calibration on validation set
    # Get the expected proportions and observed proportions on calibration set
    exp_props, obs_props = uct.metrics_calibration.get_proportion_lists_vectorized(
        pred_mean, pred_std, y_true)
    # Train a recalibration model.
    recal_model = uct.recalibration.iso_recal(exp_props, obs_props) 
    # Get prediction intervals
    recal_bounds = uct.metrics_calibration.get_prediction_interval(
        test_pred_mean, test_pred_std, confidence_level, recal_model)
    test_df['confidence_lower'] = recal_bounds.lower
    test_df['confidence_upper'] = recal_bounds.upper
    recal_name = add_suffix_to_csv(args.csv_file, added_suffix='recalibrated_')
    recalibrated_test_path = os.path.join(recalibration_path, recal_name)
    test_df.to_csv(recalibrated_test_path, index=False)
    calibration_time = (datetime.now()-start).total_seconds()
    with open(report_path, 'a') as file:
        file.write('Calibration took  {} seconds. \n'.format(calibration_time))
        
    return (test_df, recal_model)
    
    
def main():   
    
    # Parse arguments 
    parser = argparse.ArgumentParser(
        description='Recalibration for specified models')
    parser.add_argument('--device', type=int, default=0, help='GPU device id')
    parser.add_argument('--csv_file', help='results to be recalibrated')
    parser.add_argument('--cfg_file', help='configuration used for training')
    args = parser.parse_args()

    # Define the device
    device_name = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    # Define path for cfg and csv files
    root_path = os.getcwd()
    cfg_path = os.path.join(root_path, 'cfg', args.cfg_file)    
    # Get model, dataset, and uq_method based on configuration file name
    args.model, args.dataset, args.UQ = extract_info_from_cfg(args.cfg_file)    
    result_path = os.path.join(root_path, 'results', args.dataset, args.model)
    plot_path = os.path.join(root_path, 'plots', args.dataset, args.model)
    # path to inference results on test set
    csv_path = os.path.join(result_path, args.csv_file)
    # create recalibration folder in result folder
    recalibration_path = os.path.join(result_path, 'recalibration')
    recalibration_plot_path = os.path.join(plot_path, 'recalibration')
    if not os.path.exists(recalibration_path):
        os.makedirs(recalibration_path)
    if not os.path.exists(recalibration_plot_path):
        os.makedirs(recalibration_plot_path)
    # define a path for report .txt to add recalibration time
    report_path = os.path.join(recalibration_path, 'recalibration_report.txt')
    # define a path for inference on validation (calibration set)
    val_inference_name = add_suffix_to_csv(args.csv_file, added_suffix='validation_')    
    val_inference_path = os.path.join(recalibration_path, val_inference_name)    

    # a separate execution path for CARD model
    if args.UQ != 'CARD':
        args.UQ = get_uq_method(args.csv_file)
        # load cfg file used for training
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        # check whether inference is already done on validation set or not
        if os.path.exists(val_inference_path):
            print('Inference is already done on validation set.')
            calibration_df = pd.read_csv(val_inference_path)
        else:
            # define name of the check point (best model)
            checkpoint_name = replace_suffix(
                args.csv_file, 'inference_result_.csv', 'best_model.pt')
            checkpoint_path = os.path.join(result_path, checkpoint_name)
            # Get calibration loader, model dimensions, normalization ratios
            (calibration_loader, input_size, max_len, max_train_val,
             mean_train_val, median_train_val
             ) = get_validation_data_and_model_size(args=args, cfg=cfg,
                                                    root_path=root_path)
            # define model and loss function
            (model, criterion, heteroscedastic, num_mcmc, normalization
             ) = get_model_and_loss(args=args, cfg=cfg, input_size=input_size,
                                    max_len=max_len, device=device)
            # execute inference on validation set
            calibration_df = inference_on_validation(
                args=args, model=model, checkpoint_path=checkpoint_path,
                calibration_loader=calibration_loader,
                heteroscedastic=heteroscedastic, num_mc_samples=num_mcmc,
                normalization=normalization, y_scaler=mean_train_val,
                device=device, report_path=report_path,
                recalibration_path=recalibration_path)
        # load uncalibrated test dataframe
        test_df = pd.read_csv(csv_path)
        # get recalibrated predicitons
        (recalibrated_test_df, calibration_model) = recalibration_on_test(
            args=args, calibration_df=calibration_df, test_df=test_df,
            confidence_level=0.95, report_path=report_path, 
            recalibration_path=recalibration_path)
        recalibration_evaluation (
            args=args, calibrated_test_def=recalibrated_test_df,
            recal_model=calibration_model,
            recalibration_plot_path=recalibration_plot_path,
            recalibration_result_path=recalibration_path)      

        
    else:
        # TODO: access the relevant configuration file
        # TODO: access the model checkpoint
        # TODO: handle report .txt for time!
        pass
           

        
if __name__ == '__main__':
    main()
    