"""
To prepare this script we used uncertainty tool-box which can be find in:
    https://uncertainty-toolbox.github.io/about/
"""

import argparse
import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import uncertainty_toolbox as uct
from utils.eval_cal_utils import (extract_info_from_cfg, replace_suffix,
                                  get_validation_data_and_model_size,
                                  get_model_and_loss, get_uq_method,
                                  add_suffix_to_csv, inference_on_validation)
from evaluation import evaluate_coverage

def calculate_picp(df):
    in_interval = np.logical_and(df['GroundTruth'] >= df['confidence_lower'],
                                 df['GroundTruth'] <= df['confidence_upper'])
    picp = np.mean(in_interval)
    return picp

def calculate_mpiw(df):
    interval_widths = df['confidence_upper'] - df['confidence_lower']
    mpiw = np.mean(interval_widths)
    return mpiw


def recalibration_evaluation (args=None, calibrated_test_def=None,
                              recal_model=None, recalibration_plot_path=None):
    
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
                     alpha=0.6, label='Before Calibration')
    plt.fill_between(sorted_errors, recal_bounds.lower, recal_bounds.upper,
                     alpha=0.4, label='Recalibrated')
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
    pred_std_miscal = calibrated_test_def['calibrated_std_miscal']
    uct.viz.plot_calibration(pred_mean, pred_std_miscal, y_true)
    plt.gcf().set_size_inches(10, 10)
    new_file_name = base_recal_name + 'miscalibrated_area_std_miscal' + '.pdf'
    new_file_path = os.path.join(recalibration_plot_path, new_file_name)
    plt.savefig(new_file_path, format='pdf')
    plt.clf()
    pred_std_rms_cal = calibrated_test_def['calibrated_std_rms_cal']
    uct.viz.plot_calibration(pred_mean, pred_std_rms_cal, y_true)
    plt.gcf().set_size_inches(10, 10)
    new_file_name = base_recal_name + 'miscalibrated_area_std_rms_cal' + '.pdf'
    new_file_path = os.path.join(recalibration_plot_path, new_file_name)
    plt.savefig(new_file_path, format='pdf')
    plt.clf()
    pred_std_ma_cal = calibrated_test_def['calibrated_std_ma_cal']
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
            recalibration_plot_path=recalibration_plot_path)      

        
    else:
        # TODO: access the relevant configuration file
        # TODO: access the model checkpoint
        # TODO: handle report .txt for time!
        pass
           

        
if __name__ == '__main__':
    main()
    