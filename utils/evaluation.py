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
from scipy.stats import spearmanr, pearsonr
from sklearn.feature_selection import mutual_info_regression


# A method to conduct evaluation on prediction dataframe
def uq_eval(csv_file, prefix, report=False, verbose=False):
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
    Returns
    -------
    uq_metrics : a dictionary for all uncertainty quantification metrics

    """
    df = pd.read_csv(csv_file)    
    # get ground truth as well as mean and std for predictions
    pred_mean = df['Prediction'].values 
    y_true = df['GroundTruth'].values
    
    if (prefix=='DA_A' or prefix=='CDA_A' or prefix=='en_b_mve' or
        prefix=='en_t_mve'):
        pred_std = df['Total_Uncertainty'].values 
    
    elif (prefix=='CARD' or prefix=='mve' or prefix=='SQR'):
        pred_std = df['Aleatoric_Uncertainty'].values
    
    elif (prefix=='DA' or prefix=='CDA' or prefix=='en_t' or 
          prefix=='en_b' or prefix=='RF' or prefix=='LA'):
        pred_std = df['Epistemic_Uncertainty'].values
    
    else:
        raise NotImplementedError(
                'Uncertainty quantification {} not understood.'.format(prefix))
        
    # Get all uncertainty quantification metrics
    uq_metrics = uct.metrics.get_all_metrics(pred_mean, pred_std, y_true)
    
    if report:
        
        # set the path to report UQ analysis in a .txt file
        base_name, _ = os.path.splitext(os.path.basename(csv_file))
        result_path = os.path.dirname(csv_file)
        report_name = base_name + 'uq_metrics' + '.txt'
        report_path = os.path.join(result_path, report_name)
                
        # Plot sparsification error
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
                mae_random.append(compute_mae(y_true[remaining_indices],
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
            plt.plot(x, mae_uq, label=prefix, color='blue')
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
            plt.gcf().set_size_inches(10, 10)
            new_file_name = base_name + 'miscalibrated_area' + '.pdf'
            new_file_path = os.path.join(result_path, new_file_name)
            plt.savefig(new_file_path, format='pdf')
            plt.clf()
            plt.close()
        except:
            print('Plotting the average calibration is not possible', prefix)
            
        # Plot adversarial group calibration
        if verbose:
            try:
                uct.viz.plot_adversarial_group_calibration(
                    pred_mean, pred_std, y_true)
                plt.gcf().set_size_inches(10, 6)
                new_file_name = (
                    base_name + 'adversarial_group_calibration' + '.pdf')
                new_file_path = os.path.join(result_path, new_file_name)
                plt.savefig(new_file_path, format='pdf')
                plt.clf()
                plt.close()
            except:
                print('Plotting the adversarial group calibration is not possible',
                      prefix)
            
            # Plot ordered prediction intervals
            try:
                uct.viz.plot_intervals_ordered(pred_mean, pred_std, y_true)
                plt.gcf().set_size_inches(10, 10)
                # define name of the plot to be saved
                new_file_name = (
                    base_name + 'ordered_prediction_intervals' + '.pdf')
                new_file_path = os.path.join(result_path, new_file_name)
                plt.savefig(new_file_path, format='pdf')
                plt.clf()
                plt.close()
            except:
                print('Plotting the ordered prediction intervals is not possible',
                      prefix)

        # get PICP for all uncertainty quantfaction approaches
        picp, mpiw, qice, y_l, y_u = evaluate_coverage(
            y_true=y_true, pred_mean=pred_mean, pred_std=pred_std,
            low_percentile=2.5, high_percentile=97.5, num_samples= 50,
            n_bins=10)
        uq_metrics['Mean Prediction Interval Width (MPIW)'] = mpiw
        uq_metrics['Prediction Interval Coverage Probability (PICP)-0.95'] = picp 
        uq_metrics['Quantile Interval Coverage Error (QICE)'] = qice
        uq_metrics['Test_instance_below_lower_bound'] = y_l
        uq_metrics['Test_instance_morethan_upper_bound'] = y_u
        
        # Add statistical information for correlation of confidence and error
        if verbose:
            uq_metrics = correlation_stats(df, prefix, uq_metrics)      

        # write uq_metrics dictionary into a .txt file
        with open(report_path, 'w') as file:
            # Iterate over the dictionary items and write them to the file
            for key, value in uq_metrics.items():
                file.write(f"{key}: {value}\n")     
                
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


def correlation_stats (df, prefix, uq_dict):
    
    # Uncertainty vs. MAE
    if (prefix=='DA_A' or prefix=='CDA_A' or prefix=='en_b_mve' or
        prefix=='en_t_mve'):
        corr, p_value = spearmanr(
            df['Absolute_error'], df['Total_Uncertainty'])
        pear_corr, pear_p_value = pearsonr(
            df['Absolute_error'], df['Total_Uncertainty'])
        mi = mutual_info_regression(
            df['Absolute_error'].to_numpy().reshape(-1, 1), 
            df['Total_Uncertainty'].to_numpy())
    elif (prefix=='DA' or prefix=='CDA' or prefix=='en_t' or 
          prefix=='en_b' or prefix=='RF' or prefix=='LA'):
        corr, p_value = spearmanr(
            df['Absolute_error'], df['Epistemic_Uncertainty'])
        pear_corr, pear_p_value = pearsonr(
            df['Absolute_error'], df['Epistemic_Uncertainty'])
        mi = mutual_info_regression(
            df['Absolute_error'].to_numpy().reshape(-1, 1), 
            df['Epistemic_Uncertainty'].to_numpy())
    elif (prefix=='CARD' or prefix=='mve' or prefix=='SQR'):
        corr, p_value = spearmanr(
            df['Absolute_error'], df['Aleatoric_Uncertainty'])
        pear_corr, pear_p_value = pearsonr(
            df['Absolute_error'], df['Aleatoric_Uncertainty'])
        mi = mutual_info_regression(
            df['Absolute_error'].to_numpy().reshape(-1, 1), 
            df['Aleatoric_Uncertainty'].to_numpy())
    else:
        raise NotImplementedError(
                'Uncertainty quantification {} not understood.'.format(prefix)) 
    uq_dict['Spearman rank correlation coefficient: Uncertainty vs. MAE'] = corr
    uq_dict['Spearman rank correlation p_value: Uncertainty vs. MAE'] = p_value 
    uq_dict['Pearson Correlation Coefficient: Uncertainty vs. MAE'] = pear_corr
    uq_dict['Pearson Correlation p_value: Uncertainty vs. MAE'] = pear_p_value 
    uq_dict['Mutual Information: Uncertainty and MAE'] = mi
    
    # Uncertainty vs. MAPE
    if not 'Absolute_percentage_error' in df.columns:
        df['Absolute_percentage_error'] = (df['Absolute_error']/
                                           df['GroundTruth'])
    if (prefix=='DA_A' or prefix=='CDA_A' or prefix=='en_b_mve' or
        prefix=='en_t_mve'):
        corr, p_value = spearmanr(
            df['Absolute_percentage_error'], df['Total_Uncertainty'])
        pear_corr, pear_p_value = pearsonr(
            df['Absolute_percentage_error'], df['Total_Uncertainty'])
        mi = mutual_info_regression(
            df['Absolute_percentage_error'].to_numpy().reshape(-1, 1), 
            df['Total_Uncertainty'].to_numpy())
    elif (prefix=='DA' or prefix=='CDA' or prefix=='en_t' or 
          prefix=='en_b' or prefix=='RF' or prefix=='LA'):
        corr, p_value = spearmanr(
            df['Absolute_percentage_error'], df['Epistemic_Uncertainty'])
        pear_corr, pear_p_value = pearsonr(
            df['Absolute_percentage_error'], df['Epistemic_Uncertainty'])
        mi = mutual_info_regression(
            df['Absolute_percentage_error'].to_numpy().reshape(-1, 1), 
            df['Epistemic_Uncertainty'].to_numpy())
    elif (prefix=='CARD' or prefix=='mve' or prefix=='SQR'):
        corr, p_value = spearmanr(
            df['Absolute_percentage_error'], df['Aleatoric_Uncertainty'])
        pear_corr, pear_p_value = pearsonr(
            df['Absolute_percentage_error'], df['Aleatoric_Uncertainty'])
        mi = mutual_info_regression(
            df['Absolute_percentage_error'].to_numpy().reshape(-1, 1), 
            df['Aleatoric_Uncertainty'].to_numpy())
    else:
        raise NotImplementedError(
                'Uncertainty quantification {} not understood.'.format(prefix)) 
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
     
    if not 'Absolute_percentage_error' in df.columns:
        df['Absolute_percentage_error'] = (df['Absolute_error']/
                                           df['GroundTruth'])
    if (prefix=='DA_A' or prefix=='CDA_A' or prefix=='en_b_mve' or
        prefix=='en_t_mve'):
        corr1, p_value1 = spearmanr(
            filtered_df1['Absolute_percentage_error'],
            filtered_df1['Total_Uncertainty'])
        pear_corr1, pear_p_value1 = pearsonr(
            filtered_df1['Absolute_percentage_error'],
            filtered_df1['Total_Uncertainty'])
        mi1 = mutual_info_regression(
            filtered_df1['Absolute_percentage_error'].to_numpy().reshape(-1, 1), 
            filtered_df1['Total_Uncertainty'].to_numpy())
        corr2, p_value2 = spearmanr(
            filtered_df2['Absolute_percentage_error'],
            filtered_df2['Total_Uncertainty'])
        pear_corr2, pear_p_value2 = pearsonr(
            filtered_df2['Absolute_percentage_error'],
            filtered_df2['Total_Uncertainty'])
        mi2 = mutual_info_regression(
            filtered_df2['Absolute_percentage_error'].to_numpy().reshape(-1, 1), 
            filtered_df2['Total_Uncertainty'].to_numpy())
    elif (prefix=='DA' or prefix=='CDA' or prefix=='en_t' or 
          prefix=='en_b' or prefix=='RF' or prefix=='LA'):
        corr1, p_value1 = spearmanr(
            filtered_df1['Absolute_percentage_error'],
            filtered_df1['Epistemic_Uncertainty'])
        pear_corr1, pear_p_value1 = pearsonr(
            filtered_df1['Absolute_percentage_error'],
            filtered_df1['Epistemic_Uncertainty'])
        mi1 = mutual_info_regression(
            filtered_df1['Absolute_percentage_error'].to_numpy().reshape(-1, 1), 
            filtered_df1['Epistemic_Uncertainty'].to_numpy())
        corr2, p_value2 = spearmanr(
            filtered_df2['Absolute_percentage_error'],
            filtered_df2['Epistemic_Uncertainty'])
        pear_corr2, pear_p_value1 = pearsonr(
            filtered_df2['Absolute_percentage_error'],
            filtered_df2['Epistemic_Uncertainty'])
        mi2 = mutual_info_regression(
            filtered_df2['Absolute_percentage_error'].to_numpy().reshape(-1, 1), 
            filtered_df2['Epistemic_Uncertainty'].to_numpy())
    elif (prefix=='CARD' or prefix=='mve' or prefix=='SQR'):
        corr1, p_value1 = spearmanr(
            filtered_df1['Absolute_percentage_error'],
            filtered_df1['Aleatoric_Uncertainty'])
        pear_corr1, pear_p_value1 = pearsonr(
            filtered_df1['Absolute_percentage_error'],
            filtered_df1['Aleatoric_Uncertainty'])
        mi1 = mutual_info_regression(
            filtered_df1['Absolute_percentage_error'].to_numpy().reshape(-1, 1), 
            filtered_df1['Aleatoric_Uncertainty'].to_numpy())
        corr2, p_value2 = spearmanr(
            filtered_df2['Absolute_percentage_error'],
            filtered_df2['Aleatoric_Uncertainty'])
        pear_corr2, pear_p_value2 = pearsonr(
            filtered_df2['Absolute_percentage_error'], 
            filtered_df2['Aleatoric_Uncertainty'])
        mi2 = mutual_info_regression(
            filtered_df2['Absolute_percentage_error'].to_numpy().reshape(-1, 1), 
            filtered_df2['Aleatoric_Uncertainty'].to_numpy())
    else:
        raise NotImplementedError(
                'Uncertainty quantification {} not understood.'.format(prefix)) 
    uq_dict['Spearman rank correlation coefficient: Uncertainty vs. MAPE \
            (excl. small remaining times)'] = corr1
    uq_dict['Spearman rank correlation p_value: Uncertainty vs. MAPE \
            (excl. small remaining times)'] = p_value1 
    uq_dict['Pearson Correlation Coefficient: Uncertainty vs. MAPE \
            (excl. small remaining times)'] = pear_corr1
    uq_dict['Pearson Correlation p_value: Uncertainty vs. MAPE \
            (excl. small remaining times)'] = pear_p_value1 
    uq_dict['Mutual Information: Uncertainty and MAPE \
            (excl. small remaining times)'] = mi1
    uq_dict['Spearman rank correlation coefficient: Uncertainty vs. MAPE \
            (only large remaining times)'] = corr2
    uq_dict['Spearman rank correlation p_value: Uncertainty vs. MAPE \
            (only large remaining times)'] = p_value2 
    uq_dict['Pearson Correlation Coefficient: Uncertainty vs. MAPE \
            (only large remaining times)'] = pear_corr2
    uq_dict['Pearson Correlation p_value: Uncertainty vs. MAPE \
            (only large remaining times)'] = pear_p_value1 
    uq_dict['Mutual Information: Uncertainty and MAPE \
            (only large remaining times)'] = mi2          
    return uq_dict