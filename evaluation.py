"""
To prepare this script we used uncertainty tool-box which can be find in:
    https://uncertainty-toolbox.github.io/about/
For QICE metric we used the source code from:
    https://github.com/XzwHan/CARD
"""

import argparse
import os
import glob
import re
import pandas as pd
import numpy as np
import uncertainty_toolbox as uct
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


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

# a helper function for sparsification plot (computes MAE)
def compute_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def evaluate_coverage(y_true=None, pred_mean=None, pred_std=None,
                      low_percentile=2.5, high_percentile=97.5,
                      num_samples= 50, n_bins=10):
    """
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

def main():   
    
    # Parse arguments 
    parser = argparse.ArgumentParser(
        description='Uncertainy quantification evaluation')
    parser.add_argument('--dataset', default='HelpDesk',
                        help='Datasets that is used for predictions')
    parser.add_argument('--model', default='dalstm',
                        help='Type of the predictive model')
    args = parser.parse_args()
    
    # define input and output directories for evaluation
    root_path = os.getcwd()
    source_path = os.path.join(root_path, 'results', args.dataset, args.model)
    target_path = os.path.join(root_path, 'plots', args.dataset, args.model)
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    # find all csv files created by predictive model
    csv_files, uncertainty_methods = get_csv_files(source_path)
    
    # iterate over results, and create evaluation/visualization
    for i in range (len(csv_files)):
        # Extract the file name and extension
        base_name, ext = os.path.splitext(os.path.basename(csv_files[i]))
        # read csv file, and find the uncertainty quantification method used
        df = pd.read_csv(csv_files[i])
        prefix = uncertainty_methods[i]
        
        # get ground truth as well as mean and std for predictions
        pred_mean = df['Prediction'].values 
        y_true = df['GroundTruth'].values
        if (prefix=='DA_A' or prefix=='CDA_A' or prefix=='en_b_mve' or
            prefix=='en_t_mve'):
            pred_std = df['Total_Uncertainty'].values 
        elif (prefix=='CARD' or prefix=='mve'):
            pred_std = df['Aleatoric_Uncertainty'].values
        elif (prefix=='DA' or prefix=='CDA' or prefix=='en_t' or 
              prefix=='en_b' or prefix=='RF' or prefix=='LA'):
            pred_std = df['Epistemic_Uncertainty'].values
        else:
            raise NotImplementedError(
                'Uncertainty quantification {} not understood.'.format(prefix))
            
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
            # Display AUSE and AURG on the plot
            plt.text(0.6, max(mae_oracle) * 0.9, f'AUSE: {ause:.4f}',
                     color='black', fontsize=12)
            plt.text(0.6, max(mae_oracle) * 0.85, f'AURG: {aurg:.4f}',
                     color='black', fontsize=12)
            new_file_name = base_name + 'sparsification_plot' + '.pdf'
            new_file_path = os.path.join(target_path, new_file_name)
            plt.savefig(new_file_path, format='pdf')
            plt.clf()
        except:
            print('Plotting sparsfication is not possible', prefix)       
            
        # Plot average calibration
        try:
            uct.viz.plot_calibration(pred_mean, pred_std, y_true)
            plt.gcf().set_size_inches(10, 10)
            new_file_name = base_name + 'miscalibrated_area' + '.pdf'
            new_file_path = os.path.join(target_path, new_file_name)
            plt.savefig(new_file_path, format='pdf')
            plt.clf()
        except:
            print('Plotting the average calibration is not possible', prefix)
            
        # Plot adversarial group calibration
        try:
            uct.viz.plot_adversarial_group_calibration(pred_mean, pred_std, y_true)
            plt.gcf().set_size_inches(10, 6)
            new_file_name = base_name + 'adversarial_group_calibration' + '.pdf'
            new_file_path = os.path.join(target_path, new_file_name)
            plt.savefig(new_file_path, format='pdf')
            plt.clf()
        except:
            print('Plotting the adversarial group calibration is not possible',
                  prefix)
            
        # Plot ordered prediction intervals
        try:
            uct.viz.plot_intervals_ordered(pred_mean, pred_std, y_true)
            plt.gcf().set_size_inches(10, 10)
            # define name of the plot to be saved
            new_file_name = base_name + 'ordered_prediction_intervals' + '.pdf'
            new_file_path = os.path.join(target_path, new_file_name)
            plt.savefig(new_file_path, format='pdf')
            plt.clf()
        except:
            print('Plotting the ordered prediction intervals is not possible',
                  prefix)
                       
        # Get all uncertainty quantification metrics
        uq_metrics = uct.metrics.get_all_metrics(pred_mean, pred_std, y_true)
        new_file_name = base_name + 'uq_metrics' + '.txt'
        new_file_path = os.path.join(source_path, new_file_name)
        with open(new_file_path, 'w') as file:
            # Iterate over the dictionary items and write them to the file
            for key, value in uq_metrics.items():
                file.write(f"{key}: {value}\n")
                
        # Get Spearman's rank correlation coefficient (using MAE)
        if (prefix=='DA_A' or prefix=='CDA_A' or prefix=='en_b_mve' or
            prefix=='en_t_mve'):
            corr, p_value = spearmanr(df['Absolute_error'],
                                      df['Total_Uncertainty'])
        elif (prefix=='DA' or prefix=='CDA' or prefix=='en_t' or
              prefix=='en_b' or prefix=='RF' or prefix=='LA'):
            corr, p_value = spearmanr(df['Absolute_error'],
                                      df['Epistemic_Uncertainty'])
        elif (prefix=='CARD' or prefix=='mve'):
            corr, p_value = spearmanr(df['Absolute_error'],
                                      df['Aleatoric_Uncertainty'])
        else:
            raise NotImplementedError(
                'Uncertainty quantification {} not understood.'.format(prefix))                
        with open(new_file_path, 'a') as file:
            file.write("Spearman's rank correlation coefficient uncertainty w.r.t MAE: \n")
            file.write(f"Spearman's rank correlation coefficient: {corr}\t \t")
            file.write(f"P-value: {p_value}\n")  
        
        # Get Spearman's rank correlation coefficient (using MAPE)
        if not 'Absolute_percentage_error' in df.columns:
            df['Absolute_percentage_error'] = (df['Absolute_error']/
                                               df['GroundTruth'])
        if (prefix=='DA_A' or prefix=='CDA_A' or prefix=='en_b_mve' or
            prefix=='en_t_mve'):
            corr, p_value = spearmanr(df['Absolute_percentage_error'],
                                      df['Total_Uncertainty'])
        elif (prefix=='DA' or prefix=='CDA' or prefix=='en_t' or 
              prefix=='en_b' or prefix=='RF' or prefix=='LA'):
            corr, p_value = spearmanr(df['Absolute_percentage_error'],
                                      df['Epistemic_Uncertainty'])
        elif (prefix=='CARD' or prefix=='mve'):
            corr, p_value = spearmanr(df['Absolute_percentage_error'],
                                      df['Aleatoric_Uncertainty'])
        else:
            raise NotImplementedError(
                'Uncertainty quantification {} not understood.'.format(prefix))                
        with open(new_file_path, 'a') as file:
            file.write("Spearman's rank correlation coefficient uncertainty w.r.t MAPE: \n")
            file.write(f"Spearman's rank correlation coefficient: {corr}\t \t")
            file.write(f"P-value: {p_value}\n") 
            
        
        # Get Spearman's rank correlation coefficient (using filtered MAPE)
        # first conduct the filtering on test dataset
        filter_ratio = 0.1
        # get only prefixes of length 2 to estimate cycle time
        aux_df1 = df[df['Prefix_length'] < 3] 
        # average remaining time * coefficient == lower bound for remaining time
        mean_ground = aux_df1['GroundTruth'].mean()
        lower_bound = mean_ground * filter_ratio
        filtered_df = df[df['GroundTruth'] > lower_bound]       
        removal_percentage = (len(df)-len(filtered_df))/len(df)*100
        if (prefix=='DA_A' or prefix=='CDA_A' or prefix=='en_b_mve' or
            prefix=='en_t_mve'):
            corr, p_value = spearmanr(filtered_df['Absolute_percentage_error'],
                                      filtered_df['Total_Uncertainty'])
        elif (prefix=='DA' or prefix=='CDA' or prefix=='en_t' or 
              prefix=='en_b' or prefix=='RF' or prefix=='LA'):
            corr, p_value = spearmanr(filtered_df['Absolute_percentage_error'],
                                      filtered_df['Epistemic_Uncertainty'])
        elif (prefix=='CARD' or prefix=='mve'):
            corr, p_value = spearmanr(filtered_df['Absolute_percentage_error'],
                                      filtered_df['Aleatoric_Uncertainty'])
        else:
            raise NotImplementedError(
                'Uncertainty quantification {} not understood.'.format(prefix))                
        with open(new_file_path, 'a') as file:
            file.write(f"For filtered dataframe, {removal_percentage} percent of prefixes are removed. \n")  
            file.write("Spearman's rank correlation coefficient uncertainty w.r.t filtered MAPE: \n")
            file.write(f"Spearman's rank correlation coefficient: {corr}\t \t")
            file.write(f"P-value: {p_value}\n")  
            
        # get PICP for all uncertainty quantfaction approaches
        picp, mpiw, qice, y_b_0, y_a_100 = evaluate_coverage(
            y_true=y_true, pred_mean=pred_mean, pred_std=pred_std,
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

if __name__ == '__main__':
    main()
    
    
    