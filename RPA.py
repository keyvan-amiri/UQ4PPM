import os
import argparse
import pandas as pd
import seaborn as sns
import numpy as np
import pickle
import yaml
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import uncertainty_toolbox as uct
from scipy.optimize import minimize
from utils.evaluation import evaluate_coverage

def learn_transformation_function(y_val_true, y_val_pred, kernel='gaussian'):
    # Sort ground truths and compute empirical CDF for ground truth 
    gt_sorted = np.sort(y_val_true)
    gt_cdf = np.arange(1, len(gt_sorted) + 1) / len(gt_sorted)
    kde = KernelDensity(kernel=kernel).fit(y_val_pred.reshape(-1, 1))    

    def transformation_function(prediction):
        # Step 4: Compute the CDF value of the prediction using the KDE
        prob = np.exp(kde.score_samples([[prediction]]))[0]

        # Step 5: Interpolate the CDF value to match the ground truth CDF
        return np.interp(prob, gt_cdf, gt_sorted)
    
    return transformation_function

def apply_binning(data, num_bins, method='same_distance'):
    if  method=='same_distance':
        bin_edges = np.linspace(np.min(data), np.max(data), num_bins + 1)
    elif method=='quantile':
        bin_edges = np.percentile(data, np.linspace(0, 100, num_bins + 1))
    bin_indices = np.digitize(data, bin_edges) - 1
    return bin_indices, bin_edges

def MeanLT(y_hat, y):    
    def mae_loss(params):
        a, b = params
        transformed_y_hat = a * y_hat + b
        return np.mean(np.abs(transformed_y_hat - y))    
    initial_guess = [1.0, 0.0]
    result = minimize(mae_loss, initial_guess, method='Nelder-Mead')
    a_opt, b_opt = result.x
    return a_opt, b_opt

# Function to apply linear transformation
def apply_transform(data, transform_dict):
    transformed_data = np.zeros_like(data, dtype=float)
    # Loop over each element in the data
    for i, element in enumerate(data):
        # Find the appropriate boundary in the dictionary
        for (lower_bound, upper_bound), (a, b) in transform_dict.items():
            if lower_bound <= element < upper_bound:
                # Apply the linear transformation
                transformed_data[i] = a * element + b
                break
        else:
            # Optional: Handle elements outside any boundaries
            transformed_data[i] = element  # No transformation if no interval is found

    return transformed_data

def compute_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def sparsification_analysis(pred_mean=None, y_true=None, pred_std=None,
                            pred_mean_adj = None, pred_std_adj = None,
                            rpa_path = None, method=None, kernel=None):
    n_samples = len(pred_mean)
    n_steps = 100
    step_size = int(n_samples / n_steps)
    np.random.seed(42)
    
    # Compute Oracle curves       
    mae_per_sample = np.abs(pred_mean - y_true) # MAE for each sample
    mae_per_sample_adj = np.abs(pred_mean_adj - y_true) # MAE for each sample
    # Sort by MAE descending
    sorted_indices_by_mae = np.argsort(mae_per_sample)[::-1]  
    sorted_indices_by_mae_adj = np.argsort(mae_per_sample_adj)[::-1]
    mae_oracle,  mae_oracle_adj= [], []
    for i in range(n_steps):
        remaining_indices = sorted_indices_by_mae[i * step_size:]
        mae_oracle.append(compute_mae(y_true[remaining_indices],
                                              pred_mean[remaining_indices]))
    for i in range(n_steps):
        remaining_indices = sorted_indices_by_mae_adj[i * step_size:]
        mae_oracle_adj.append(compute_mae(y_true[remaining_indices],
                                              pred_mean_adj[remaining_indices]))
    # Compute Random curves
    mae_random, mae_random_adj = [], []
    for i in range(n_steps):
        remaining_indices = np.random.choice(
                        n_samples, n_samples - i * step_size, replace=False)
        mae_random.append(
                        compute_mae(y_true[remaining_indices],
                                    pred_mean[remaining_indices]))
        mae_random_adj.append(
                        compute_mae(y_true[remaining_indices],
                                    pred_mean_adj[remaining_indices]))
    # Compute UQ curves
    sorted_indices_by_uncertainty = np.argsort(
        pred_std)[::-1]  # Sort by uncertainty descending
    sorted_indices_by_uncertainty_adj = np.argsort(
        pred_std_adj)[::-1]  # Sort by uncertainty descending
    mae_uq, mae_uq_adj = [], []
    for i in range(n_steps):
        remaining_indices = sorted_indices_by_uncertainty[i * step_size:]
        mae_uq.append(compute_mae(y_true[remaining_indices],
                                              pred_mean[remaining_indices]))
    for i in range(n_steps):
        remaining_indices = sorted_indices_by_uncertainty_adj[i * step_size:]
        mae_uq_adj.append(compute_mae(y_true[remaining_indices],
                                              pred_mean_adj[remaining_indices]))
        
    # Plot the curves
    x = np.linspace(0, 1, n_steps)
    plt.figure(figsize=(10, 6))
    plt.plot(x, mae_oracle, label='Oracle', color='gray', linestyle='--')
    plt.plot(x, mae_oracle_adj, label='Oracle-adjusted', color='black')    
    plt.plot(x, mae_random, label='Random', color='orange', linestyle='--')
    plt.plot(x, mae_random_adj, label='Random-adjusted', color='red')    
    plt.plot(x, mae_uq, label=method, color='blue', linestyle='--')
    plt.plot(x, mae_uq_adj, label=method+'-adjusted', color='olive') 
    plt.xlabel('Fraction of Removed Samples')
    plt.ylabel('MAE')
    plt.title('Sparsification Plot')
    plt.legend()
    # Compute areas for AUSE and AURG
    ause = np.trapz(mae_uq, x) - np.trapz(mae_oracle, x)
    aurg = np.trapz(mae_random, x) - np.trapz(mae_uq, x)
    ause_adj = np.trapz(mae_uq_adj, x) - np.trapz(mae_oracle_adj, x)
    aurg_adj = np.trapz(mae_random_adj, x) - np.trapz(mae_uq_adj, x)
    #plt.text(0.6, max(mae_oracle) * 0.9, f'AUSE: {ause:.4f}', color='black', fontsize=12)
    #plt.text(0.6, max(mae_oracle) * 0.85, f'AURG: {aurg:.4f}', color='black', fontsize=12)
    #plt.text(0.6, max(mae_oracle) * 0.80, f'AUSE-adjusted: {ause_adj:.4f}', color='black', fontsize=12)
    #plt.text(0.6, max(mae_oracle) * 0.75, f'AURG-adjusted: {aurg_adj:.4f}', color='black', fontsize=12)    
    new_file_name = method + '_' + kernel + '_sparsification_plot' + '.pdf'
    new_file_path = os.path.join(rpa_path, new_file_name)
    plt.savefig(new_file_path, format='pdf')
    plt.clf()
    plt.close()   
    
    return (ause, aurg, ause_adj, aurg_adj)


def apply_rpa(args, rpa_path):
    val_df_names = [f for f in os.listdir(args.result_path)
                    if f.endswith('inference_result_validation_.csv')]
    test_df_names = []
    for val_name in val_df_names:
        modified_string = val_name.replace('validation_.csv', '.csv') 
        test_df_names.append(modified_string)
    # Collect UQ technique names 
    methods = [f.split('_'+args.split)[0] for f in val_df_names]
       
    # Collect a list of dataframes for each val_df_path
    val_df_lst = []
    for val_name in val_df_names:
        val_df = pd.read_csv(os.path.join(args.result_path, val_name))    
        val_df_lst.append(val_df)
    # Collect a list of dataframes for each test_df_names
    test_df_lst = []
    for test_name in test_df_names:
        df = pd.read_csv(os.path.join(args.result_path, test_name))  
        test_df_lst.append(df)
    
    uq_adj_lst, uq_lst = [], []
    for i in range (len(methods)):
        method = methods[i]
        df_val = val_df_lst[i]
        df = test_df_lst[i]
        rpa_test_name = test_df_names[i].replace('.csv', 'rpa_.csv')
        report_name = rpa_test_name.replace('.csv', 'uq_metrics.txt')
        uq_dict_name = rpa_test_name.replace('.csv', 'uq_metrics.pkl')
        before_uq_dict_name = uq_dict_name.replace('rpa_uq_metrics.pkl',
                                                   'uq_metrics.pkl')
        before_uq_dict_path = os.path.join(args.result_path, before_uq_dict_name)
        report_path = os.path.join(rpa_path, report_name)
        uq_dict_path = os.path.join(rpa_path, uq_dict_name)
        uq_adj_lst.append(uq_dict_path)
        uq_lst.append(before_uq_dict_path)
        
        # set the uncertainty column  
        # TODO: how to deal with LA?
        if (method=='DA+H' or method=='CDA+H' or method == 'DE+H' or 
            method == 'BE+H' or method == 'LA' or method=='deterministic' or 
            method=='GMM_uniform' or method=='GMM_dynamic'):
            uncertainty_col = 'Total_Uncertainty'
        elif (method=='CARD' or method=='H' or method=='SQR'):
            uncertainty_col = 'Aleatoric_Uncertainty'
        elif (method=='DA' or method=='CDA' or method == 'DE' or 
              method == 'BE' or method == 'E-RF'):
            uncertainty_col = 'Epistemic_Uncertainty'

        y_val_pred = df_val['Prediction'].values
        y_val_true = df_val['GroundTruth'].values
        y_val_std = df_val[uncertainty_col].values
        y_test_pred = df['Prediction'].values
        y_test_std = df[uncertainty_col].values
        
        if args.style == 'kde':
            # Learn the transformation function from the validation set
            transformation_func = learn_transformation_function(
                y_val_true, y_val_pred, kernel=args.kernel)
            # Apply the transformation to each test prediction
            y_test_transformed = np.array([transformation_func(pred) for pred in y_test_pred])
        elif args.style == 'linear':
            bin_indices, bin_edges = apply_binning(y_val_pred, args.num_bins,
                                                   method=args.bin_method)            
            # Create a dictionary to collect values for each bin
            bin_data = {i: {'y_val_pred': [], 'y_val_std': [], 'y_val_true': []
                            } for i in range(args.num_bins)}
            # Populate the dictionary with values for each bin
            for i, bin_idx in enumerate(bin_indices):
                if 0 <= bin_idx < args.num_bins:  # Only consider valid bins
                    bin_data[bin_idx]['y_val_pred'].append(y_val_pred[i])
                    bin_data[bin_idx]['y_val_std'].append(y_val_std[i])
                    bin_data[bin_idx]['y_val_true'].append(y_val_true[i])
            for bin_idx, data in bin_data.items():
                bin_data[bin_idx]['y_val_pred'] = np.asarray(data['y_val_pred'])
                bin_data[bin_idx]['y_val_std'] = np.asarray(data['y_val_std'])
                bin_data[bin_idx]['y_val_true'] = np.asarray(data['y_val_true'])
            trans_dict = {}
            for bin_idx, data in bin_data.items():
                y_hat = data['y_val_pred']
                y = data['y_val_true']
                a_opt, b_opt = MeanLT(y_hat, y)
                trans_dict[(bin_edges[bin_idx], bin_edges[bin_idx+1])] = (a_opt, b_opt)
            #print(trans_dict)
            # Apply transformation on test set to adjust predicted expected values
            y_test_transformed = apply_transform(y_test_pred, trans_dict)
            y_test_transformed = np.where(y_test_transformed <= 0, 1e-6, y_test_transformed)
            
    
        df['Transformed_Prediction'] = y_test_transformed
        # Apply calibration regression after adjusting the mean.
        # Apply the same transformation on validation set
        y_val_transformed = apply_transform(y_val_pred, trans_dict)
        y_val_transformed = np.where(y_val_transformed <= 0, 1e-6, y_val_transformed)
        try:
            miscal_std_scaling = uct.recalibration.optimize_recalibration_ratio(
                y_val_transformed, y_val_std, y_val_true, criterion="miscal")
        except:
            y_val_std = np.where(y_val_std <= 0, 1e-6, y_val_std)
            miscal_std_scaling = uct.recalibration.optimize_recalibration_ratio(
                y_val_transformed, y_val_std, y_val_true, criterion="miscal") 
        df['Transformed_uncertainty']  = miscal_std_scaling * y_test_std       
        
        df.to_csv(os.path.join(rpa_path, rpa_test_name), index=False)
        df['adjusted_error'] = (df['Transformed_Prediction'] - df['GroundTruth']).abs()
        
        ground_truth = df['GroundTruth'].values
        predictions = df['Prediction'].values
        adjusted_predictions = df['Transformed_Prediction'].values
        uncertainties =  df[uncertainty_col].values
        adjusted_uncertainties = df['Transformed_uncertainty'].values      
        
        # Visualizations for PDF and CDF distributions
        mae1 = mean_absolute_error(df['GroundTruth'], df['Prediction'])
        mae2 = mean_absolute_error(df['GroundTruth'], df['Transformed_Prediction'])       
        # Visualization: PDF for ground truth, prediction, adjusted prediction
        plt.figure(figsize=(10, 6))
        sns.kdeplot(ground_truth, label='Ground Truth', color='blue', fill=True, alpha=0.5)
        sns.kdeplot(predictions, label='Prediction', color='orange', fill=True, alpha=0.5)
        sns.kdeplot(adjusted_predictions, label='Prediction-adjusted', color='red', fill=True, alpha=0.5)       
        plt.text(x=0.05, y=0.95, s=f'MAE: {mae1:.2f}', 
                transform=plt.gca().transAxes,  # Use axes coordinates (0, 1) for positioning
                fontsize=12, color='black', verticalalignment='top')
        plt.text(x=0.05, y=0.90, s=f'MAE-Adjusted: {mae2:.2f}',
                transform=plt.gca().transAxes,  # Use axes coordinates (0, 1) for positioning
                fontsize=12, color='black', verticalalignment='top')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title(f'Distribution of Ground Truth and Predictions for {method}')
        plt.legend(loc='best')
        adj_name = method + '_' + args.kernel + '_Prediction_PDF.pdf'
        adj_path = os.path.join(rpa_path, adj_name)
        plt.savefig(adj_path, format='pdf')
        plt.close()
        # Visualization: PDF for uncertainty, adjusted uncertainty
        plt.figure(figsize=(10, 6))
        sns.kdeplot(uncertainties, label='Estimated Uncertainty', color='blue', fill=True, alpha=0.5)
        sns.kdeplot(adjusted_uncertainties, label='Adjusted Uncertainty', color='orange', fill=True, alpha=0.5)      
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title(f'Distribution of Estimated Uncertainty and its adjusted version for {method}')
        plt.legend(loc='best')
        adj_std_name = method + '_' + args.kernel + '_Uncertainty_PDF.pdf'
        adj_std_path = os.path.join(rpa_path, adj_std_name)
        plt.savefig(adj_std_path, format='pdf')
        plt.close()
        # Visualization: CDF for ground truth, prediction, adjusted prediction
        # Create a figure for each file
        plt.figure(figsize=(10, 6))       
        sns.ecdfplot(data=df, x='GroundTruth', label='Ground Truth', color='blue')
        sns.ecdfplot(data=df, x='Prediction', label='Prediction', color='orange')
        sns.ecdfplot(data=df, x='Transformed_Prediction', label='Prediction-adjusted', color='red')
        plt.xlabel('Value')
        plt.ylabel('Cumulative Probability')
        plt.title(f'Cumulative Distribution of Ground Truth and Predictions for {method}')
        plt.legend(loc='best')
        plt.grid()
        cdf_name = method + '_' + args.kernel + '_Prediction_CDF.pdf'
        cdf_path = os.path.join(rpa_path, cdf_name)
        plt.savefig(cdf_path, format='pdf')
        plt.close()  
        # Visualization: earliness of predictions
        df_filtered = df[df['Prefix_length'] >= 2]
        prefix_90_percentile = np.percentile(df_filtered['Prefix_length'], 90)
        if prefix_90_percentile < 10:
            df_limited = df_filtered
        else:
            df_limited = df_filtered[df_filtered['Prefix_length'] <= prefix_90_percentile]            
        mean_abs_error = df_limited.groupby('Prefix_length')['Absolute_error'].mean()
        mean_std = df_limited.groupby('Prefix_length')[uncertainty_col].mean()
        mean_pred = df_limited.groupby('Prefix_length')['Prediction'].mean() 
        mean_error_adj = df_limited.groupby('Prefix_length')['adjusted_error'].mean()
        mean_std_adj = df_limited.groupby('Prefix_length')['Transformed_uncertainty'].mean()
        mean_pred_adj = df_limited.groupby('Prefix_length')['Transformed_Prediction'].mean()
        plt.figure(figsize=(10, 6)) 
        plt.plot(mean_pred.index, mean_pred.values, linestyle='--', color='green', 
                 label='Mean Predicted Value')
        plt.plot(mean_pred_adj.index, mean_pred_adj.values, linestyle='-',
                 color='olive', label='Mean Predicted Value- after RPA')        
        plt.plot(mean_abs_error.index, mean_abs_error.values, linestyle='--', 
                 color='blue', label='Mean Absolute Error')
        plt.plot(mean_error_adj.index, mean_error_adj.values, linestyle='-', 
                 color='cyan', label='Mean Absolute Error- after RPA')
        plt.plot(mean_std.index, mean_std.values, linestyle='--', color='purple', 
                 label='Mean Posterior Standard Deviation')
        plt.plot(mean_std_adj.index, mean_std_adj.values, linestyle='-', color='pink', 
                 label='Mean Posterior Standard Deviation- after RPA')
        plt.title(f'{method}: Mean Absolute Error, Mean Uncertainty vs Prefix Length')
        plt.xlabel('Prefix Length')
        plt.ylabel('MAE / Mean Uncertainty / Mean predictions')
        plt.legend()
        early_name = method + '_' + args.kernel + '_Earliness.pdf'
        early_path = os.path.join(rpa_path, early_name)
        plt.savefig(early_path, format='pdf')
        plt.clf()
        plt.close()
        # sparsification analysis, and plotting      
        (ause, aurg, ause_adj, aurg_adj) = sparsification_analysis(
            pred_mean=predictions, y_true=ground_truth, pred_std=uncertainties,
            pred_mean_adj = adjusted_predictions, 
            pred_std_adj = adjusted_uncertainties, rpa_path = rpa_path, 
            method=method, kernel=args.kernel)
        # get all metrics after adjustment
        adjusted_uncertainties = np.maximum(adjusted_uncertainties, 1e-6)        
        uq_metrics_adj = uct.metrics.get_all_metrics(
            adjusted_predictions, adjusted_uncertainties, ground_truth)
        # add sparsification metrics
        uq_metrics_adj['Area Under Sparsification Error curve (AUSE)'] = ause_adj
        uq_metrics_adj['Area Under Random Gain curve (AURG)'] = aurg_adj
        picp, mpiw, qice, y_l, y_u = evaluate_coverage(
            y_true=ground_truth, pred_mean=adjusted_predictions,
            pred_std=adjusted_uncertainties, low_percentile=2.5,
            high_percentile=97.5, num_samples= 50, n_bins=10)
        uq_metrics_adj['Mean Prediction Interval Width (MPIW)'] = mpiw
        uq_metrics_adj['Prediction Interval Coverage Probability (PICP)-0.95'] = picp
        uq_metrics_adj['Quantile Interval Coverage Error (QICE)'] = qice
        uq_metrics_adj['Test_instance_below_lower_bound'] = y_l
        uq_metrics_adj['Test_instance_morethan_upper_bound'] = y_u
        with open(report_path, 'w') as file:
            # Iterate over the dictionary items and write them to the file
            for key, value in uq_metrics_adj.items():
                file.write(f"{key}: {value}\n") 
        with open(uq_dict_path, 'wb') as file:
            pickle.dump(uq_metrics_adj, file) 
            
    return methods, uq_adj_lst, uq_lst


def main():
    parser = argparse.ArgumentParser(
        description='Use Prediction Distribution Adjustment')
    parser.add_argument('--dataset', help='dataset that is used.')
    parser.add_argument('--cfg', help='configuration for RPA.')
    args = parser.parse_args()
    root_path = os.getcwd()
    # read the relevant cfg file
    cfg_file = os.path.join(root_path, 'cfg', args.cfg)
    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f) 
    args.model = cfg.get('model')
    args.split = cfg.get('split')
    args.seed = cfg.get('seed')    
    args.style = cfg.get('style')
    args.kernel = cfg.get('kernel')
    args.num_bins = cfg.get('num_bins')
    args.bin_method = cfg.get('bin_method')
    args.result_path = os.path.join(root_path, 'results', args.dataset, args.model)
    rpa_path = os.path.join(args.result_path, 'RPA')
    if not os.path.exists(rpa_path):
        os.makedirs(rpa_path)
    
    methods, uq_adj_lst, uq_lst = apply_rpa(args, rpa_path)
        
if __name__ == '__main__':
    main()



