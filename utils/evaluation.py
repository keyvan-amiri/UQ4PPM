"""
To prepare this script we used uncertainty tool-box which can be find in:
    https://uncertainty-toolbox.github.io/about/
For QICE metric we used the source code from:
    https://github.com/XzwHan/CARD
"""

import os
import pandas as pd
import numpy as np
import uncertainty_toolbox as uct
import matplotlib.pyplot as plt
import pickle
from utils.utils import get_mean_std_truth, uq_label_plot, get_statistics


# A method to conduct evaluation on prediction dataframe
def uq_eval(csv_file, prefix, report=False, verbose=False,
            calibration_mode=False, calibration_type=None, recal_model=None):
    """    
    Parameters
    ----------
    csv_file : Inference results (predictions: point estimates + uncertainty)
    prefix : the uncertainty quantification technique used for inference
    report : whether to create a report for UQ metrics
    verbose : whether to report staistical metrics for correlation between
    model's confidence and error. it also applies for plots for:
        adversarial group calibration
        ordered prediction intervals
    verbose = True is only effective for report=True.
    calibration_mode : whether evaluation is conducted for recalibrated 
    predictions or not (generate extra reports, and plots.)
    calibration_type: method that is used for calibration, scaling or isotonic
    regression. scaling can be done w.r.t accuracy or miscalibration area.
    recal_model : in case of evalution for recalibrated result, a recal model
    is required (this is a Non-Gaussian isotonic regression model)
    Returns
    -------
    uq_metrics : a dictionary for all uncertainty quantification metrics

    """
    df = pd.read_csv(csv_file)    
    # get ground truth, posterior mean and standard deviation (uncertainty)
    pred_mean, pred_std, y_true = get_mean_std_truth(df=df, uq_method=prefix)    
    if calibration_mode:
        pred_std_miscal = df['calibrated_std_miscal'].values
        pred_std_rms_cal = df['calibrated_std_rms_cal'].values
        pred_std_ma_cal = df['calibrated_std_ma_cal'].values
        
    # Get all uncertainty quantification metrics
    if calibration_mode:
        # if evaluation is done for calibrated predictions
        if (calibration_type == 'miscal' or calibration_type == 'all'):
            # for numerical stability
            pred_std_miscal = np.maximum(pred_std_miscal, 1e-6) 
            uq_metrics1 = uct.metrics.get_all_metrics(
                pred_mean, pred_std_miscal, y_true)
        if (calibration_type == 'rms' or calibration_type == 'all'):
            # for numerical stability
            pred_std_rms_cal = np.maximum(pred_std_rms_cal, 1e-6) 
            uq_metrics2 = uct.metrics.get_all_metrics(
                pred_mean, pred_std_rms_cal, y_true)
        if (calibration_type == 'ma' or calibration_type == 'all'):
            # for numerical stability
            pred_std_ma_cal = np.maximum(pred_std_ma_cal, 1e-6) 
            uq_metrics3 = uct.metrics.get_all_metrics(
                pred_mean, pred_std_ma_cal, y_true)
        if (calibration_type == 'isotonic' or calibration_type == 'all'):
            # Compute PICP and MPIW for isotonic regression calibration
            # since this calibration is Non-Gaussian we only provide few
            # unertainty quantification metrics
            uq_metrics4={}
            picp = calculate_picp(df)
            mpiw = calculate_mpiw(df)
            uq_metrics4['Mean Prediction Interval Width (MPIW)'] = mpiw
            uq_metrics4['Prediction Interval Coverage Probability (PICP)-0.95'] = picp     
    else:
        # if evaluation is done before calibrated regression.
        pred_std = np.maximum(pred_std, 1e-6) # for numerical stability
        uq_metrics = uct.metrics.get_all_metrics(pred_mean, pred_std, y_true)
    
    if report:        
        # set the path to report UQ analysis in a .txt file
        base_name, _ = os.path.splitext(os.path.basename(csv_file))
        result_path = os.path.dirname(csv_file)
        if not calibration_mode:
            report_name = base_name + 'uq_metrics' + '.txt'
            uq_dict_name = base_name + 'uq_metrics' + '.pkl'
            report_path = os.path.join(result_path, report_name)
            uq_dict_path = os.path.join(result_path, uq_dict_name)
        else:
            report_name1 = base_name + 'uq_metrics_std_miscal' + '.txt'
            report_name2 = base_name + 'uq_metrics_std_rms_cal' + '.txt'
            report_name3 = base_name + 'uq_metrics_std_ma_cal' + '.txt'
            report_name4 = base_name + 'pcip_mpiw_isotonic_regression' + '.txt'
            uq_dict_name1 = base_name + 'uq_metrics_std_miscal' + '.pkl'
            uq_dict_name2 = base_name + 'uq_metrics_std_rms_cal' + '.pkl'
            uq_dict_name3 = base_name + 'uq_metrics_std_ma_cal' + '.pkl'
            uq_dict_name4 = base_name + 'pcip_mpiw_isotonic_regression' + '.pkl'
            report_path1 = os.path.join(result_path, report_name1)
            report_path2 = os.path.join(result_path, report_name2)
            report_path3 = os.path.join(result_path, report_name3)
            report_path4 = os.path.join(result_path, report_name4)
            uq_dict_path1 = os.path.join(result_path, uq_dict_name1)
            uq_dict_path2 = os.path.join(result_path, uq_dict_name2)
            uq_dict_path3 = os.path.join(result_path, uq_dict_name3)
            uq_dict_path4 = os.path.join(result_path, uq_dict_name4)
                
        # UQ method name as text in plots.
        plot_label = uq_label_plot(uq_method=prefix)
        
        # Plot sparsification error only before calibration. 
        # Note that this plot is not affected by calibration, since Gaussian 
        # valibration is simply scaling the standard deviation.
        if not calibration_mode:
            try:
                n_samples = len(pred_mean)
                n_steps = 100
                step_size = int(n_samples / n_steps)
                np.random.seed(42)
                # Compute Oracle curve        
                mae_per_sample = np.abs(pred_mean - y_true) # MAE for each sample
                sorted_indices_by_mae = np.argsort(
                    mae_per_sample)[::-1]  # Sort by MAE descending
                mae_oracle = []
                for i in range(n_steps):
                    remaining_indices = sorted_indices_by_mae[i * step_size:]
                    mae_oracle.append(compute_mae(y_true[remaining_indices],
                                              pred_mean[remaining_indices]))
                # Compute Random curve
                mae_random = []
                for i in range(n_steps):
                    remaining_indices = np.random.choice(
                        n_samples, n_samples - i * step_size, replace=False)
                    mae_random.append(
                        compute_mae(y_true[remaining_indices],
                                    pred_mean[remaining_indices]))
                # Compute UQ curve
                sorted_indices_by_uncertainty = np.argsort(
                    pred_std)[::-1]  # Sort by uncertainty descending
                mae_uq = []
                for i in range(n_steps):
                    remaining_indices = sorted_indices_by_uncertainty[i * step_size:]
                    mae_uq.append(compute_mae(y_true[remaining_indices],
                                              pred_mean[remaining_indices]))
                # Plot the curves
                x = np.linspace(0, 1, n_steps)
                plt.figure(figsize=(10, 6))
                plt.plot(x, mae_oracle, label='Oracle', color='gray')
                plt.plot(x, mae_random, label='Random', color='red')
                plt.plot(x, mae_uq, label=plot_label, color='blue')
                plt.xlabel('Fraction of Removed Samples')
                plt.ylabel('MAE')
                plt.title('Sparsification Plot')
                plt.legend()
                # Compute areas for AUSE and AURG
                ause = np.trapz(mae_uq, x) - np.trapz(mae_oracle, x)
                aurg = np.trapz(mae_random, x) - np.trapz(mae_uq, x)
                uq_metrics['Area Under Sparsification Error curve (AUSE)'] = ause
                uq_metrics['Area Under Random Gain curve (AURG)'] = aurg
                # Display AUSE and AURG on the plot
                plt.text(0.6, max(mae_oracle) * 0.9, f'AUSE: {ause:.4f}',
                         color='black', fontsize=12)
                plt.text(0.6, max(mae_oracle) * 0.85, f'AURG: {aurg:.4f}',
                         color='black', fontsize=12)
                new_file_name = base_name + 'sparsification_plot' + '.pdf'
                new_file_path = os.path.join(result_path, new_file_name)
                plt.savefig(new_file_path, format='pdf')
                plt.clf()
                plt.close()            
            except:
                print('Plotting sparsfication is not possible', prefix) 
            
            # Plot average calibration
            try:
                uct.viz.plot_calibration(pred_mean, pred_std, y_true)
                plt.text(x=0.95, y=0.03, s=plot_label, 
                         fontsize='small', ha='right', va='bottom')            
                plt.gcf().set_size_inches(10, 10)
                new_file_name = base_name + 'miscalibrated_area' + '.pdf'
                new_file_path = os.path.join(result_path, new_file_name)
                plt.savefig(new_file_path, format='pdf')
                plt.clf()
                plt.close()
            except:
                print('Plotting the average calibration is not possible', prefix)
                
            # create earliness analysis plot
            if (prefix=='DA_A' or prefix=='CDA_A' or prefix == 'en_t_mve' or 
                prefix == 'en_b_mve' or prefix=='deterministic' or 
                prefix=='GMM' or prefix=='GMMD'):
                uncertainty_col = 'Total_Uncertainty'
            elif (prefix=='CARD' or prefix=='mve' or prefix=='SQR'):
                uncertainty_col = 'Aleatoric_Uncertainty'
            elif (prefix=='DA' or prefix=='CDA' or prefix == 'en_t' or 
                  prefix == 'en_b' or prefix == 'RF' or prefix == 'LA'):
                uncertainty_col = 'Epistemic_Uncertainty'
            
            df_filtered = df[df['Prefix_length'] >= 2]
            prefix_90_percentile = np.percentile(df_filtered['Prefix_length'], 
                                                 90)
            if prefix_90_percentile < 10:
                df_limited = df_filtered
            else:
                df_limited = df_filtered[df_filtered['Prefix_length'] 
                                         <= prefix_90_percentile]
            mean_abs_error = df_limited.groupby('Prefix_length')['Absolute_error'].mean()
            mean_std = df_limited.groupby('Prefix_length')[uncertainty_col].mean()
            mean_pred = df_limited.groupby('Prefix_length')['Prediction'].mean()            
            plt.figure(figsize=(10, 6))  
            plt.plot(mean_abs_error.index, mean_abs_error.values, marker='o', 
                     linestyle='-', color='b', label='Mean Absolute Error')
            plt.plot(mean_pred.index, mean_pred.values, marker='^', 
                     linestyle='--', color='g', 
                     label='Mean Predicted Value')
            plt.plot(mean_std.index, mean_std.values, marker='s', 
                     linestyle=':', color='r', 
                     label='Mean Posterior Standard Deviation')
            plt.title('Mean Absolute Error and Mean Uncertainty vs Prefix Length')
            plt.xlabel('Prefix Length')
            plt.ylabel('MAE / Mean Uncertainty / Mean predictions')
            plt.legend()
            plt.text(0.95, 0.05, s=plot_label, 
                     fontsize='small', ha='right', va='bottom',
                     transform=plt.gca().transAxes)
            new_file_name = (base_name + 'Earliness_analysis' + '.pdf')
            new_file_path = os.path.join(result_path, new_file_name)
            plt.savefig(new_file_path, format='pdf')
            plt.clf()
            plt.close()
        else:
            # Now plot miscalibration for Gaussian calibrations
            if (calibration_type == 'miscal' or calibration_type == 'all'):
                uct.viz.plot_calibration(pred_mean, pred_std_miscal, y_true)
                plt.text(x=0.95, y=0.03, s=plot_label, 
                         fontsize='small', ha='right', va='bottom')  
                plt.gcf().set_size_inches(10, 10)
                new_file_name = (
                    base_name + 'miscalibrated_area_std_miscal' + '.pdf')
                new_file_path = os.path.join(result_path, new_file_name)
                plt.savefig(new_file_path, format='pdf')
                plt.clf()
                plt.close()
            elif (calibration_type == 'rms' or calibration_type == 'all'):
                uct.viz.plot_calibration(pred_mean, pred_std_rms_cal, y_true)
                plt.text(x=0.95, y=0.03, s=plot_label, 
                         fontsize='small', ha='right', va='bottom')  
                plt.gcf().set_size_inches(10, 10)
                new_file_name = (
                    base_name + 'miscalibrated_area_std_rms_cal' + '.pdf')
                new_file_path = os.path.join(result_path, new_file_name)
                plt.savefig(new_file_path, format='pdf')
                plt.clf()
                plt.close()
            elif (calibration_type == 'ma' or calibration_type == 'all'):
                uct.viz.plot_calibration(pred_mean, pred_std_ma_cal, y_true)
                plt.text(x=0.95, y=0.03, s=plot_label, 
                         fontsize='small', ha='right', va='bottom')  
                plt.gcf().set_size_inches(10, 10)
                new_file_name = (
                    base_name + 'miscalibrated_area_std_ma_cal' + '.pdf')
                new_file_path = os.path.join(result_path, new_file_name)
                plt.savefig(new_file_path, format='pdf')
                plt.clf()
                plt.close()
            # Non-Gaussian calibration: 
            # expected proportions and observed proportions
            exp_props, obs_props = (
                uct.metrics_calibration.get_proportion_lists_vectorized(
                pred_mean, pred_std, y_true, recal_model=recal_model)) 
            
            # Create average calibration plot for recalibrated predictions
            uct.viz.plot_calibration(pred_mean, pred_std, y_true,
                                     exp_props=exp_props,
                                     obs_props=obs_props)
            plt.text(x=0.95, y=0.03, s=plot_label, 
                     fontsize='small', ha='right', va='bottom')  
            plt.gcf().set_size_inches(10, 10)
            new_file_name = (
                base_name + 'miscalibrated_area_isotonic_regression' + '.pdf')
            new_file_path = os.path.join(result_path, new_file_name)
            plt.savefig(new_file_path, format='pdf')
            plt.clf()
            plt.close()
            
            # sort the calibrated predictions based on absolute error
            sorted_df = df.sort_values(by='Absolute_error')
            sorted_pred_mean = sorted_df['Prediction'].values           
            sorted_errors = sorted_df['Absolute_error'].values
            if (prefix=='DA_A' or prefix=='CDA_A' or 
                prefix == 'en_t_mve' or prefix == 'en_b_mve' or
                prefix=='deterministic' or prefix=='GMM' or prefix=='GMMD'):
                sorted_pred_std = sorted_df['Total_Uncertainty'].values 
            elif (prefix=='CARD' or prefix=='mve' or prefix=='SQR'):
                sorted_pred_std = sorted_df['Aleatoric_Uncertainty'].values
            elif (prefix=='DA' or prefix=='CDA' or prefix == 'en_t' or 
                  prefix == 'en_b' or prefix == 'RF' or prefix == 'LA'):
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
            plt.text(x=0.95, y=0.03, s=plot_label, 
                     fontsize='small', ha='right', va='bottom')  
            plt.gcf().set_size_inches(10, 10)
            plt.title('95% Centered Prediction Interval')
            new_file_name = (base_name +
                             'confidence_intervals_isotonic_regression_error_based' + 
                             '.pdf')
            new_file_path = os.path.join(result_path, new_file_name)
            plt.savefig(new_file_path, format='pdf')
            plt.clf()
            plt.close()
            
            # sort the calibrated predictions based on prefix length
            sorted_df = df.sort_values(by='Prefix_length')
            sorted_pred_mean = sorted_df['Prediction'].values
            sorted_lengths = sorted_df['Prefix_length'].values
            if (prefix=='DA_A' or prefix=='CDA_A' or 
                prefix == 'en_t_mve' or prefix == 'en_b_mve' or 
                prefix=='deterministic' or prefix=='GMM' or prefix=='GMMD'):
                sorted_pred_std = sorted_df['Total_Uncertainty'].values 
            elif (prefix=='CARD' or prefix=='mve' or prefix=='SQR'):
                sorted_pred_std = sorted_df['Aleatoric_Uncertainty'].values
            elif (prefix=='DA' or prefix=='CDA' or prefix == 'en_t' or
                  prefix == 'en_b' or prefix == 'RF' or prefix == 'LA'):
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
            plt.text(x=0.95, y=0.03, s=plot_label, 
                     fontsize='small', ha='right', va='bottom')  
            plt.gcf().set_size_inches(10, 10)
            plt.title('95% Centered Prediction Interval')
            new_file_name = (base_name +
                             'confidence_intervals_isotonic_regression_length_based' + 
                             '.pdf')
            new_file_path = os.path.join(result_path, new_file_name)
            plt.savefig(new_file_path, format='pdf')
            plt.clf()
            plt.close()
            
            # sort the calibrated predictions based on ground truth (remaining time)
            sorted_df = df.sort_values(by='GroundTruth')
            sorted_pred_mean = sorted_df['Prediction'].values
            sorted_rem_time = sorted_df['GroundTruth'].values
            if (prefix=='DA_A' or prefix=='CDA_A' or 
                prefix == 'en_t_mve' or prefix == 'en_b_mve' or 
                prefix=='deterministic' or prefix=='GMM' or prefix=='GMMD'):
                sorted_pred_std = sorted_df['Total_Uncertainty'].values 
            elif (prefix=='CARD' or prefix=='mve' or prefix=='SQR'):
                sorted_pred_std = sorted_df['Aleatoric_Uncertainty'].values
            elif (prefix=='DA' or prefix=='CDA' or prefix == 'en_t' or 
                  prefix == 'en_b' or prefix == 'RF' or prefix == 'LA'):
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
            plt.text(x=0.95, y=0.03, s=plot_label, 
                     fontsize='small', ha='right', va='bottom')  
            plt.gcf().set_size_inches(10, 10)
            plt.title('95% Centered Prediction Interval')
            new_file_name = (base_name +
                             'confidence_intervals_isotonic_regression_remainingtime_based' + 
                             '.pdf')
            new_file_path = os.path.join(result_path, new_file_name)
            plt.savefig(new_file_path, format='pdf')
            plt.clf()
            plt.close()
                
            
        if verbose:
            if not calibration_mode:
                # Plot adversarial group calibration
                try:
                    uct.viz.plot_adversarial_group_calibration(
                        pred_mean, pred_std, y_true)
                    plt.text(x=0.95, y=0.03, s=plot_label, 
                             fontsize='small', ha='right', va='bottom')  
                    plt.gcf().set_size_inches(10, 6)
                    new_file_name = (
                        base_name + 'adversarial_group_calibration' + '.pdf')
                    new_file_path = os.path.join(result_path, new_file_name)
                    plt.savefig(new_file_path, format='pdf')
                    plt.clf()
                    plt.close()
                except:
                    print('Plotting the adversarial group calibration is not \
                          possible', prefix)
            else:
                # Plot adversarial group calibration for Gaussian calibrations
                if (calibration_type == 'miscal' or calibration_type == 'all'):
                    uct.viz.plot_adversarial_group_calibration(
                        pred_mean, pred_std_miscal, y_true)
                    plt.text(x=0.95, y=0.03, s=plot_label, 
                             fontsize='small', ha='right', va='bottom')  
                    plt.gcf().set_size_inches(10, 6)
                    new_file_name = (
                        base_name + 
                        'adversarial_group_calibration_std_miscal' + '.pdf')
                    new_file_path = os.path.join(result_path, new_file_name)
                    plt.savefig(new_file_path, format='pdf')
                    plt.clf()
                    plt.close()
                if (calibration_type == 'rms' or calibration_type == 'all'):                 
                    uct.viz.plot_adversarial_group_calibration(
                        pred_mean, pred_std_rms_cal, y_true)
                    plt.text(x=0.95, y=0.03, s=plot_label, 
                             fontsize='small', ha='right', va='bottom')  
                    plt.gcf().set_size_inches(10, 6)
                    new_file_name = (
                        base_name +
                        'adversarial_group_calibration_std_rms_cal' + '.pdf')
                    new_file_path = os.path.join(result_path, new_file_name)
                    plt.savefig(new_file_path, format='pdf')
                    plt.clf()
                    plt.close()
                if (calibration_type == 'ma' or calibration_type == 'all'):                    
                    uct.viz.plot_adversarial_group_calibration(
                        pred_mean, pred_std_ma_cal, y_true)
                    plt.text(x=0.95, y=0.03, s=plot_label, 
                             fontsize='small', ha='right', va='bottom')  
                    plt.gcf().set_size_inches(10, 6)
                    new_file_name = (
                        base_name + 
                        'adversarial_group_calibration_std_ma_cal' + '.pdf')
                    new_file_path = os.path.join(result_path, new_file_name)
                    plt.savefig(new_file_path, format='pdf')
                    plt.clf()
                    plt.close()
            
            # Plot ordered prediction intervals
            if not calibration_mode:
                try:
                    uct.viz.plot_intervals_ordered(pred_mean, pred_std, y_true)
                    plt.text(x=0.95, y=0.03, s=plot_label, 
                             fontsize='small', ha='right', va='bottom')  
                    plt.gcf().set_size_inches(10, 10)
                    # define name of the plot to be saved
                    new_file_name = (
                        base_name + 'ordered_prediction_intervals' + '.pdf')
                    new_file_path = os.path.join(result_path, new_file_name)
                    plt.savefig(new_file_path, format='pdf')
                    plt.clf()
                    plt.close()
                except:
                    print('Plotting the ordered prediction intervals is not \
                          possible', prefix)
            else:
                if (calibration_type == 'miscal' or calibration_type == 'all'):
                    # Plot ordered prediction intervals for Gaussian calibrations
                    uct.viz.plot_intervals_ordered(
                        pred_mean, pred_std_miscal, y_true)
                    plt.text(x=0.95, y=0.03, s=plot_label, 
                             fontsize='small', ha='right', va='bottom')  
                    plt.gcf().set_size_inches(10, 10)
                    # define name of the plot to be saved
                    new_file_name = (
                        base_name + 
                        'ordered_prediction_intervals_std_miscal' + '.pdf')
                    new_file_path = os.path.join(result_path, new_file_name)
                    plt.savefig(new_file_path, format='pdf')
                    plt.clf()
                    plt.close()
                if (calibration_type == 'rms' or calibration_type == 'all'):
                    uct.viz.plot_intervals_ordered(
                        pred_mean, pred_std_rms_cal, y_true)
                    plt.text(x=0.95, y=0.03, s=plot_label, 
                             fontsize='small', ha='right', va='bottom')  
                    plt.gcf().set_size_inches(10, 10)
                    # define name of the plot to be saved
                    new_file_name = (
                        base_name + 
                        'ordered_prediction_intervals_std_rms_cal' + '.pdf')
                    new_file_path = os.path.join(result_path, new_file_name)
                    plt.savefig(new_file_path, format='pdf')
                    plt.clf()
                    plt.close()
                if (calibration_type == 'ma' or calibration_type == 'all'):    
                    uct.viz.plot_intervals_ordered(
                        pred_mean, pred_std_ma_cal, y_true)
                    plt.text(x=0.95, y=0.03, s=plot_label, 
                             fontsize='small', ha='right', va='bottom')  
                    plt.gcf().set_size_inches(10, 10)
                    # define name of the plot to be saved
                    new_file_name = (
                        base_name + 
                        'ordered_prediction_intervals_std_ma_cal' + '.pdf')
                    new_file_path = os.path.join(result_path, new_file_name)
                    plt.savefig(new_file_path, format='pdf')
                    plt.clf()
                    plt.close()                

        # get PICP for all uncertainty quantfaction approaches
        if not calibration_mode:
            picp, mpiw, qice, y_l, y_u = evaluate_coverage(
                y_true=y_true, pred_mean=pred_mean, pred_std=pred_std,
                low_percentile=2.5, high_percentile=97.5, num_samples= 50,
                n_bins=10)
            uq_metrics['Mean Prediction Interval Width (MPIW)'] = mpiw
            uq_metrics['Prediction Interval Coverage Probability (PICP)-0.95'] = picp 
            uq_metrics['Quantile Interval Coverage Error (QICE)'] = qice
            uq_metrics['Test_instance_below_lower_bound'] = y_l
            uq_metrics['Test_instance_morethan_upper_bound'] = y_u
        else:
            if (calibration_type == 'miscal' or calibration_type == 'all'):
                picp, mpiw, qice, y_l, y_u = evaluate_coverage(
                    y_true=y_true, pred_mean=pred_mean, pred_std=pred_std_miscal,
                    low_percentile=2.5, high_percentile=97.5, num_samples= 50,
                    n_bins=10)
                uq_metrics1['Mean Prediction Interval Width (MPIW)'] = mpiw
                uq_metrics1['Prediction Interval Coverage Probability (PICP)-0.95'] = picp 
                uq_metrics1['Quantile Interval Coverage Error (QICE)'] = qice
                uq_metrics1['Test_instance_below_lower_bound'] = y_l
                uq_metrics1['Test_instance_morethan_upper_bound'] = y_u
            if (calibration_type == 'rms' or calibration_type == 'all'):
                picp, mpiw, qice, y_l, y_u = evaluate_coverage(
                    y_true=y_true, pred_mean=pred_mean, pred_std=pred_std_rms_cal,
                    low_percentile=2.5, high_percentile=97.5, num_samples= 50,
                    n_bins=10)
                uq_metrics2['Mean Prediction Interval Width (MPIW)'] = mpiw
                uq_metrics2['Prediction Interval Coverage Probability (PICP)-0.95'] = picp 
                uq_metrics2['Quantile Interval Coverage Error (QICE)'] = qice
                uq_metrics2['Test_instance_below_lower_bound'] = y_l
                uq_metrics2['Test_instance_morethan_upper_bound'] = y_u           
            if (calibration_type == 'ma' or calibration_type == 'all'):
                picp, mpiw, qice, y_l, y_u = evaluate_coverage(
                    y_true=y_true, pred_mean=pred_mean, pred_std=pred_std_ma_cal,
                    low_percentile=2.5, high_percentile=97.5, num_samples= 50,
                    n_bins=10)
                uq_metrics3['Mean Prediction Interval Width (MPIW)'] = mpiw
                uq_metrics3['Prediction Interval Coverage Probability (PICP)-0.95'] = picp 
                uq_metrics3['Quantile Interval Coverage Error (QICE)'] = qice
                uq_metrics3['Test_instance_below_lower_bound'] = y_l
                uq_metrics3['Test_instance_morethan_upper_bound'] = y_u                 


        # Add statistical information for correlation of confidence and error
        if verbose:
            if not calibration_mode:
                uq_metrics = correlation_stats(df, prefix, uq_metrics)  
            else:
                if (calibration_type == 'miscal' or calibration_type == 'all'):
                    uq_metrics1 = correlation_stats(
                        df, prefix, uq_metrics1,calibration_mode=True,
                        calibration_type='miscal')
                if (calibration_type == 'rms' or calibration_type == 'all'):
                    uq_metrics2 = correlation_stats(
                        df, prefix, uq_metrics2,calibration_mode=True,
                        calibration_type='rms')
                if (calibration_type == 'ma' or calibration_type == 'all'):
                    uq_metrics3 = correlation_stats(
                        df, prefix, uq_metrics3,calibration_mode=True,
                        calibration_type='ma')
                    
        if not calibration_mode:
            # write uq_metrics dictionary into a .txt file
            with open(report_path, 'w') as file:
                # Iterate over the dictionary items and write them to the file
                for key, value in uq_metrics.items():
                    file.write(f"{key}: {value}\n") 
            with open(uq_dict_path, 'wb') as file:
                pickle.dump(uq_metrics, file)         
        else:
            if (calibration_type == 'miscal' or calibration_type == 'all'):
                with open(report_path1, 'w') as file:
                    for key, value in uq_metrics1.items():
                        file.write(f"{key}: {value}\n")
                with open(uq_dict_path1, 'wb') as file:
                    pickle.dump(uq_metrics1, file)   
            if (calibration_type == 'rms' or calibration_type == 'all'):
                with open(report_path2, 'w') as file:
                    for key, value in uq_metrics2.items():
                        file.write(f"{key}: {value}\n")
                with open(uq_dict_path2, 'wb') as file:
                    pickle.dump(uq_metrics2, file)   
            if (calibration_type == 'ma' or calibration_type == 'all'):
                with open(report_path3, 'w') as file:
                    for key, value in uq_metrics3.items():
                        file.write(f"{key}: {value}\n")
                with open(uq_dict_path3, 'wb') as file:
                    pickle.dump(uq_metrics3, file)   
            if (calibration_type == 'isotonic' or calibration_type == 'all'):
                with open(report_path4, 'w') as file:
                    for key, value in uq_metrics4.items():
                        file.write(f"{key}: {value}\n") 
                with open(uq_dict_path4, 'wb') as file:
                    pickle.dump(uq_metrics4, file)   
    if (not calibration_mode and not report):
        # we are in HPO situation!
        (uq_metrics['Area Under Sparsification Error curve (AUSE)'],
         uq_metrics['Area Under Random Gain curve (AURG)']
         ) = get_sparsification(
             pred_mean=pred_mean, y_true=y_true, pred_std=pred_std)
    if not calibration_mode:
        return uq_metrics
          

# A helper function for sparsification plot (computes MAE)
def compute_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# A helper function to compute PICP, MPIW, QICE 
def evaluate_coverage(y_true=None, pred_mean=None, pred_std=None,
                      low_percentile=2.5, high_percentile=97.5,
                      num_samples= 50, n_bins=10):
    """
    A method to compute PICP, MPIW, QICE and number of instances out of 
    confidence interval. With the exception of QICE all other metrics are 
    computed based on the confidence interval of high_percentile-low_percentile
    for instance, a confidence interval of 0.95 is equivalent to 97.5-2.5
    Arguments:
        y_true: a numpy array representing ground truth for remaining time.
        pred_mean: a numpy array representing mean of predictions.
        pred_std: a numpy array representing standard deviation of predictions
        low_percentile, high_percentile: ranges for low and high percetiles,
        for instance 2.5 , 97.5 is equivalent to confidence interval of 95%.
        num_samples: number of samples to generate for predictions.
        n_bins (int): Number of quantile bins.        
    """  
    # Generate prediction samples
    pred_samples = np.random.normal(loc=pred_mean[:, np.newaxis],
                                    scale=pred_std[:, np.newaxis],
                                    size=(pred_mean.shape[0], num_samples))
    # Compute the prediction intervals
    CI_lower = np.percentile(pred_samples, low_percentile, axis=1)
    CI_upper = np.percentile(pred_samples, high_percentile, axis=1)
    
    # Compute the Prediction Interval Coverage Probability (PICP)
    y_in_range = (y_true >= CI_lower) & (y_true <= CI_upper)
    PICP = y_in_range.mean()
    
    # Compute the Mean Prediction Interval Width (MPIW)
    interval_widths = CI_upper - CI_lower
    MPIW = interval_widths.mean()
    
    # Compute Quantile Interval Coverage Error (QICE)
    # 1) Create a list of quantiles based on the number of bins
    quantile_list = np.arange(n_bins + 1) * (100 / n_bins)    
    # 2) Compute predicted quantiles
    y_pred_quantiles = np.percentile(pred_samples, q=quantile_list, axis=1)    
    # 3) Determine which quantile each true value falls into
    y_true = y_true.T
    quantile_membership_array = (y_true - y_pred_quantiles > 0).astype(int)
    y_true_quantile_membership = quantile_membership_array.sum(axis=0)    
    # 4) Count the number of true values in each quantile bin
    y_true_quantile_bin_count = np.array([
        (y_true_quantile_membership == v).sum() for v in np.arange(n_bins + 2)])
    y_true_below_0 = y_true_quantile_bin_count[0]
    y_true_above_100 = y_true_quantile_bin_count[-1]    
    y_true_quantile_bin_count[1] += y_true_quantile_bin_count[0]
    y_true_quantile_bin_count[-2] += y_true_quantile_bin_count[-1]
    y_true_quantile_bin_count_ = y_true_quantile_bin_count[1:-1]    
    # 5) Compute true y coverage ratio for each gen y quantile interval
    y_true_ratio_by_bin = y_true_quantile_bin_count_ / len(y_true)
    # Sum of quantile coverage ratios shall be 1!
    assert np.abs(np.sum(y_true_ratio_by_bin) - 1) < 1e-10     
    QICE = np.absolute(np.ones(n_bins) / n_bins - y_true_ratio_by_bin).mean()
    
    return PICP, MPIW, QICE, y_true_below_0, y_true_above_100


def correlation_stats (df, prefix, uq_dict, calibration_mode=False,
                       calibration_type=None):
    if not calibration_mode:
        if (prefix=='DA_A' or prefix=='CDA_A' or prefix=='en_b_mve' or
            prefix=='en_t_mve' or prefix=='deterministic' or prefix=='GMM'
            or prefix=='GMMD'):
            uncertainty_col = 'Total_Uncertainty'
        elif (prefix=='DA' or prefix=='CDA' or prefix=='en_t' or 
              prefix=='en_b' or prefix=='RF' or prefix=='LA'):
            uncertainty_col = 'Epistemic_Uncertainty'
        elif (prefix=='CARD' or prefix=='mve' or prefix=='SQR'):
            uncertainty_col = 'Aleatoric_Uncertainty'
        else:
            raise NotImplementedError(
                'Uncertainty quantification {} not understood.'.format(prefix)) 
    else:
        if calibration_type == 'miscal':
            uncertainty_col = 'calibrated_std_miscal'
        elif calibration_type == 'rms':
            uncertainty_col = 'calibrated_std_rms_cal'
        elif calibration_type == 'ma':
            uncertainty_col = 'calibrated_std_ma_cal'
    # Uncertainty vs. MAE
    error_col = 'Absolute_error'
    (corr, p_value, pear_corr, pear_p_value, mi) = get_statistics(
        df=df, error_col=error_col, uncertainty_col=uncertainty_col)        
    uq_dict['Spearman rank correlation coefficient: Uncertainty vs. MAE'] = corr
    uq_dict['Spearman rank correlation p_value: Uncertainty vs. MAE'] = p_value 
    uq_dict['Pearson Correlation Coefficient: Uncertainty vs. MAE'] = pear_corr
    uq_dict['Pearson Correlation p_value: Uncertainty vs. MAE'] = pear_p_value 
    uq_dict['Mutual Information: Uncertainty and MAE'] = mi
    
    # Uncertainty vs. MAPE
    error_col = 'Absolute_percentage_error'
    if not 'Absolute_percentage_error' in df.columns:
        df['Absolute_percentage_error'] = (df['Absolute_error']/
                                           df['GroundTruth'])
    (corr, p_value, pear_corr, pear_p_value, mi) = get_statistics(
        df=df, error_col=error_col, uncertainty_col=uncertainty_col)            
    uq_dict['Spearman rank correlation coefficient: Uncertainty vs. MAPE'] = corr
    uq_dict['Spearman rank correlation p_value: Uncertainty vs. MAPE'] = p_value 
    uq_dict['Pearson Correlation Coefficient: Uncertainty vs. MAPE'] = pear_corr
    uq_dict['Pearson Correlation p_value: Uncertainty vs. MAPE'] = pear_p_value 
    uq_dict['Mutual Information: Uncertainty and MAPE'] = mi
    
    # Uncertainty vs. MAPE: 
        # excluding examples with short remaining times
        # only include examples with large remainign times
    quantile_ratio1 = 0.2
    quantile_ratio2 = 0.8        
    # get only prefixes of length 2 to estimate cycle time
    #aux_df1 = df[df['Prefix_length'] < 3] 
    #mean_ground = aux_df1['GroundTruth'].mean()
    #lower_bound = mean_ground * filter_ratio
    #filtered_df = df[df['GroundTruth'] > lower_bound]  
    percentile1 = df['GroundTruth'].quantile(quantile_ratio1)
    percentile2 = df['GroundTruth'].quantile(quantile_ratio2)
    filtered_df1 = df[df['GroundTruth'] >= percentile1]  
    filtered_df2 = df[df['GroundTruth'] >= percentile2]
    
    (corr1, p_value1, pear_corr1, pear_p_value1, mi1) = get_statistics(
        df=filtered_df1, error_col=error_col, uncertainty_col=uncertainty_col)
    (corr2, p_value2, pear_corr2, pear_p_value2, mi2) = get_statistics(
        df=filtered_df2, error_col=error_col, uncertainty_col=uncertainty_col)    
    uq_dict['Spearman coefficient: Uncertainty vs. MAPE (excl. small remaining times)'] = corr1
    uq_dict['Spearman p_value: Uncertainty vs. MAPE (excl. small remaining times)'] = p_value1 
    uq_dict['Pearson Coefficient: Uncertainty vs. MAPE (excl. small remaining times)'] = pear_corr1
    uq_dict['Pearson p_value: Uncertainty vs. MAPE (excl. small remaining times)'] = pear_p_value1 
    uq_dict['Mutual Information: Uncertainty and MAPE (excl. small remaining times)'] = mi1
    uq_dict['Spearman coefficient: Uncertainty vs. MAPE (only large remaining times)'] = corr2
    uq_dict['Spearman p_value: Uncertainty vs. MAPE (only large remaining times)'] = p_value2 
    uq_dict['Pearson Coefficient: Uncertainty vs. MAPE (only large remaining times)'] = pear_corr2
    uq_dict['Pearson p_value: Uncertainty vs. MAPE (only large remaining times)'] = pear_p_value1 
    uq_dict['Mutual Information: Uncertainty and MAPE (only large remaining times)'] = mi2          
    return uq_dict

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

def get_sparsification(pred_mean=None, y_true=None, pred_std=None):
    n_samples = len(pred_mean)
    n_steps = 100
    step_size = int(n_samples / n_steps)
    np.random.seed(42)
    # Compute Oracle curve        
    mae_per_sample = np.abs(pred_mean - y_true) # MAE for each sample
    # Sort by MAE descending
    sorted_indices_by_mae = np.argsort(mae_per_sample)[::-1]  
    mae_oracle = []
    for i in range(n_steps):
        remaining_indices = sorted_indices_by_mae[i * step_size:]
        mae_oracle.append(compute_mae(y_true[remaining_indices],
                                              pred_mean[remaining_indices]))
    # Compute Random curve
    mae_random = []
    for i in range(n_steps):
        remaining_indices = np.random.choice(
                        n_samples, n_samples - i * step_size, replace=False)
        mae_random.append(
                        compute_mae(y_true[remaining_indices],
                                    pred_mean[remaining_indices]))
    # Compute UQ curve
    sorted_indices_by_uncertainty = np.argsort(
                    pred_std)[::-1]  # Sort by uncertainty descending
    mae_uq = []
    for i in range(n_steps):
        remaining_indices = sorted_indices_by_uncertainty[i * step_size:]
        mae_uq.append(compute_mae(y_true[remaining_indices],
                                              pred_mean[remaining_indices]))
    # Plot the curves
    x = np.linspace(0, 1, n_steps)
    # Compute areas for AUSE and AURG
    ause = np.trapz(mae_uq, x) - np.trapz(mae_oracle, x)
    aurg = np.trapz(mae_random, x) - np.trapz(mae_uq, x)
    return (ause, aurg)