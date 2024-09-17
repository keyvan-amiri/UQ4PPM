import argparse
import os
import re
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


def get_val_dataframes(args, result_path):
    columns_to_keep = ['GroundTruth', 'Prediction', 'std']
    val_df_lst = []
    for technique in args.techniques:
        pattern = re.compile(rf'{re.escape(technique)}_{re.escape(args.split)}_seed_{re.escape(str(args.seed))}_.*\.csv') 
        matching_files = [f for f in os.listdir(result_path) if pattern.match(f)]
        for file in matching_files:
            if file.endswith('inference_result_validation_.csv'):
                file_path = os.path.join(result_path, file)
                df = pd.read_csv(file_path) 
                if (technique=='DA_A' or technique=='CDA_A' or 
                    technique == 'en_t_mve' or technique == 'en_b_mve' or 
                    technique=='deterministic'):
                    df = df.rename(columns={'Total_Uncertainty': 'std'})
                elif (technique=='CARD' or technique=='mve' or 
                      technique=='SQR'):
                    df = df.rename(columns={'Aleatoric_Uncertainty': 'std'})
                elif (technique=='DA' or technique=='CDA' or 
                      technique == 'en_t' or technique == 'en_b' or 
                      technique == 'RF' or technique == 'LA'):
                    df = df.rename(columns={'Epistemic_Uncertainty': 'std'})
                df = df[columns_to_keep]                
                val_df_lst.append(df)        
    return val_df_lst
    

def main():
    parser = argparse.ArgumentParser(
        description='Use mixture of Gaussians to combine UQ techniques')
    parser.add_argument('--dataset', default='HelpDesk',
                        help='Dataset to be analysed') 
    parser.add_argument('--model', default='dalstm',
                        help='Type of the predictive model')
    parser.add_argument(
    '--techniques', 
    nargs='+',  # '+' means one or more values
    required=True,  # Makes the argument mandatory
    help='List of UQ techniques to be used'
    )
    parser.add_argument('--split', default='holdout',
                        help='The data split that is used')
    parser.add_argument('--seed', type=int, default=42,
                        help='seed number to be used')
    
    args = parser.parse_args()
    root_path = os.getcwd()
    result_path = os.path.join(root_path, 'results', args.dataset, args.model) 
    val_df_lst = get_val_dataframes(args, result_path)
    ground_truth = val_df_lst[0]['GroundTruth']
    predictions_std = pd.concat(
    [df[['Prediction', 'std']].rename(
        columns={'Prediction': f'Prediction_{i}', 'std': f'std_{i}'}) 
        for i, df in enumerate(val_df_lst)], axis=1)
    result_df = pd.concat([ground_truth, predictions_std], axis=1)
    
    
    prediction_columns = [col for col in result_df.columns 
                          if col.startswith('Prediction_')]
    predictions_df = result_df[prediction_columns]
    means = predictions_df.to_numpy()
    std_columns = [col for col in result_df.columns if col.startswith('std_')]
    stds_df = result_df[std_columns]
    stds = stds_df.to_numpy()
    targets = result_df['GroundTruth'].to_numpy()
    # Number of techniques and validation samples
    num_techniques = means.shape[1]
    num_samples = means.shape[0]
    # Weights for combining the different objectives
    alpha_nll = 0.5   # Weight for NLL
    alpha_mse = 0.3   # Weight for accuracy
    alpha_sharp = 0.2 # Weight for sharpness

    # Negative log-likelihood function
    def negative_log_likelihood(params):
        # Apply softmax to ensure the weights sum to 1
        weights = np.exp(params) / np.sum(np.exp(params))

        nll = 0  # Accumulate negative log-likelihood

        for i in range(num_samples):
            mixture_prob = 0
        
            # Mixture of Gaussians probability for each sample
            for j in range(num_techniques):
                gaussian_prob = norm.pdf(targets[i], loc=means[i, j], scale=stds[i, j])
                mixture_prob += weights[j] * gaussian_prob

            # Add to the total negative log-likelihood
            nll += -np.log(mixture_prob)
        return nll
    
    # Combined loss function
    def combined_loss(params):
        # Apply softmax to ensure the weights sum to 1
        weights = np.exp(params) / np.sum(np.exp(params))    
        nll, mse, sharpness = 0, 0, 0  # Initialize metrics    
        for i in range(num_samples):
            mixture_mean = np.sum(weights * means[i, :])
            mixture_std = np.sqrt(np.sum(
                weights * (stds[i, :]**2 + (means[i, :]-mixture_mean)**2)))  
        
            # NLL: Negative Log-Likelihood
            mixture_prob = 0
            for j in range(num_techniques):
                gaussian_prob = norm.pdf(targets[i], loc=means[i, j], scale=stds[i, j])
                mixture_prob += weights[j] * gaussian_prob
            nll += -np.log(mixture_prob)
        
            # MSE: Mean Squared Error
            mse += (targets[i] - mixture_mean) ** 2
        
            # Sharpness: Average uncertainty (use variance as proxy)
            sharpness += np.mean(mixture_std)

        # Combined objective
        loss = alpha_nll * nll + alpha_mse * mse + alpha_sharp * sharpness
    
        return loss

    # Initial parameters (log-space for softmax)
    initial_params = np.zeros(num_techniques)

    # Optimize using L-BFGS-B without explicit constraints
    #result = minimize(negative_log_likelihood, initial_params, method='L-BFGS-B')
    result = minimize(combined_loss, initial_params, method='L-BFGS-B')

    # Softmax the result to get the final weights
    optimized_weights = np.exp(result.x) / np.sum(np.exp(result.x))

    print("Learned Weights:", optimized_weights)

    csv_filepath = os.path.join(result_path, 'combined.csv')        
    result_df.to_csv(csv_filepath, index=False)

   
# A generic class for Gaussian Mixture Models to combine UQ technique
class GMM ():
    def __init__ (self, args=None, root_path=None): 
        self.args = args
        self.dataset = args.dataset
        self.model = args.model
        self.calibration = args.calibration
        self.result_path = os.path.join(root_path, 'results', args.dataset,
                                        args.model)
        self.cal_path = os.path.join(root_path, 'results', args.dataset,
                                        args.model, 'recalibration')
        self.comp_path = os.path.join(root_path, 'results', args.dataset,
                                        args.model, 'comparison')
        if not os.path.exists(self.comp_path):
            os.makedirs(self.comp_path)
        
        # comparison for results before calibrated regression
        self.uq_df_path = os.path.join(self.comp_path, 
                                       self.dataset + '_uq_metrics.csv')   
        self.uq_dict_lst, self.init_uq_method_lst = self.load_uq_dicts(
            self.result_path)        
        (self.mae_lst, self.rmse_lst, self.mis_cal_lst, self.sharpness_lst,
         self.nll_lst, self.crps_lst, self.ause_lst, self.aurg_lst, 
         self.mpiw_lst, self.picp_lst, self.qice_lst, self.below_lst, 
         self.more_than_lst) = self.extract_metrics(self.uq_dict_lst)
        self.uq_method_lst = self.replace_techniques()
        self.uq_df = self.lists_to_dataframe(
            self.uq_method_lst, mae=self.mae_lst, rmse=self.rmse_lst, 
            mis_cal=self.mis_cal_lst, sharpness=self.sharpness_lst,
            nll=self.nll_lst, crps=self.crps_lst, ause=self.ause_lst,
            aurg=self.aurg_lst, mpiw=self.mpiw_lst, picp=self.picp_lst,
            qice=self.qice_lst, below=self.below_lst, 
            more_than=self.more_than_lst)
        for column in self.uq_df.columns:
            if not column in ['aurg', 'picp']:
                self.plot_metrics(self.uq_df, column)
            elif column=='picp':
                self.plot_picp(self.uq_df)
            elif column=='aurg':
                self.plot_aurg()
        
        # comparison for results after calibrated regression
        self.uq_df_path = os.path.join(
            self.comp_path, self.dataset + '_uq_metrics_calibrated.csv')   
        self.uq_dict_lst, self.init_uq_method_lst = self.load_uq_dicts(
            self.cal_path, calibration=True)           
        (self.mae_lst, self.rmse_lst, self.mis_cal_lst, self.sharpness_lst,
         self.nll_lst, self.crps_lst, self.mpiw_lst, self.picp_lst, 
         self.qice_lst, self.below_lst,
         self.more_than_lst) = self.extract_metrics(self.uq_dict_lst,
                                                    calibration=True)
        self.uq_method_lst = self.replace_techniques()
        self.uq_df_cal = self.lists_to_dataframe(
            self.uq_method_lst, mae=self.mae_lst, rmse=self.rmse_lst, 
            mis_cal=self.mis_cal_lst, sharpness=self.sharpness_lst,
            nll=self.nll_lst, crps=self.crps_lst, mpiw=self.mpiw_lst,
            picp=self.picp_lst, qice=self.qice_lst, below=self.below_lst, 
            more_than=self.more_than_lst)
        for column in self.uq_df_cal.columns:
            if not column in ['aurg', 'mae', 'rmse', 'picp']:
                self.plot_metrics(self.uq_df_cal, column, calibration=True)
            elif column=='picp':
                self.plot_picp(self.uq_df_cal, calibration=True)
        
        # visualize calibration effect
        self.plot_metric_comparison()
                
                
                
        
    def load_uq_dicts(self, folder_path, calibration=False):
        # set appropraie string to search for files
        if not calibration:
            search_str = 'uq_metrics.pkl'
        else:
            if self.calibration == 'ma_cal':
                search_str = 'uq_metrics_std_ma_cal.pkl'
            elif self.calibration == 'rms_cal':
                search_str = 'uq_metrics_std_rms_cal.pkl'
            else:
                search_str = 'uq_metrics_std_miscal.pkl'              
                        
        data_list = []
        techniques = []
        # Iterate over the files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(search_str):
                # Extract the technique from the filename
                # Split by 'holdout' or 'cv' and take the first part
                technique = filename.split('_holdout')[0].split('_cv')[0]
                techniques.append(technique)
            
                # Full path to the pickle file
                file_path = os.path.join(folder_path, filename)
            
                # Load the pickle file and add to data_list
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    data_list.append(data)    
        return data_list, techniques
    

    def replace_techniques(self):
        # Define the replacements in a dictionary
        replacements = {
        'en_b_mve': 'En (B) + H',
        'en_t': 'En (T)',
        'DA_A': 'DA + H',
        'CDA_A': 'CDA + H',
        'en_t_mve': 'En (T) + H',
        'en_b': 'En (B)',
        'mve': 'H'
        }    
        # Use list comprehension to replace elements in the list
        updated_list = [replacements.get(item, item) for 
                        item in self.init_uq_method_lst]    
        return updated_list

    def extract_metrics(self, dict_list, calibration=False):
        mae_lst = [d['accuracy']['mae'] for d in dict_list]
        rmse_lst = [d['accuracy']['rmse'] for d in dict_list]
        mis_cal_lst = [d['avg_calibration']['miscal_area'] for d in dict_list]
        sharpness_lst = [d['sharpness']['sharp'] for d in dict_list]
        nll_lst = [d['scoring_rule']['nll'] for d in dict_list]
        crps_lst = [d['scoring_rule']['crps'] for d in dict_list]
        if not calibration:
            ause_lst = [d['Area Under Sparsification Error curve (AUSE)']
                        for d in dict_list]
            aurg_lst = [d['Area Under Random Gain curve (AURG)']
                        for d in dict_list]
        mpiw_lst = [d['Mean Prediction Interval Width (MPIW)']
                    for d in dict_list]
        picp_lst = [d['Prediction Interval Coverage Probability (PICP)-0.95']
                    for d in dict_list]
        qice_lst = [d['Quantile Interval Coverage Error (QICE)']
                    for d in dict_list]
        below_lst = [d['Test_instance_below_lower_bound']
                    for d in dict_list]
        more_than_lst = [d['Test_instance_morethan_upper_bound']
                    for d in dict_list]
        if not calibration:
            return (mae_lst, rmse_lst, mis_cal_lst, sharpness_lst, nll_lst,
                    crps_lst, ause_lst, aurg_lst, mpiw_lst, picp_lst, qice_lst,
                    below_lst, more_than_lst)
        else:
            return (mae_lst, rmse_lst, mis_cal_lst, sharpness_lst, nll_lst,
                    crps_lst, mpiw_lst, picp_lst, qice_lst,
                    below_lst, more_than_lst)
            
    def lists_to_dataframe(self, techniques, **metrics):
        df = pd.DataFrame(metrics, index=techniques)
        df.index.name = 'technique'  # Set index name for clarity
        df.to_csv(self.uq_df_path)
        return df
    
    def plot_metrics(self, df, metric, smaller_is_better=True, calibration=False):
        str_metric = str(metric).upper()
        df_filtered = df[df.index != 'deterministic']
        # Get the value for the 'deterministic' technique
        deterministic_val = df.loc['deterministic', metric]
    
        # Define colors based on whether smaller or larger values are better
        if smaller_is_better:
            colors = ['green' if val <= deterministic_val else 'orange' for 
                      val in df_filtered[metric]]
        else:
            colors = ['green' if val > deterministic_val else 'orange' for val 
                      in df_filtered[metric]]
        
        # Plot the bar chart for techniques excluding 'deterministic'
        plt.figure(figsize=(10, 6))
        bars = plt.bar(df_filtered.index, df_filtered[metric], color=colors)
        #plt.bar(df_filtered.index, df_filtered[metric], color=colors)

        # Add the dashed horizontal line for 'deterministic' metric value
        plt.axhline(deterministic_val, color='k', linestyle='--', 
                    label=f'Deterministic {str_metric} ({deterministic_val:.2f})')

        # Customize the plot
        plt.xlabel('UQ Technique')
        plt.ylabel(str_metric)
        plt.title(f'{self.dataset}: {str_metric} for different UQ Techniques')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
    
        # Save the plot as a PDF file
        if not calibration:
            new_file_name = f'{self.dataset}_{str_metric}_comparison.pdf'
        else:
            new_file_name = f'{self.dataset}_{str_metric}_comparison_calibrated_{self.calibration}.pdf'
        new_file_path = os.path.join(self.comp_path, new_file_name)
        plt.tight_layout()
        plt.savefig(new_file_path, format='pdf')
    
        # Clear and close the plot
        plt.clf()
        plt.close()
        
    def plot_picp(self, df, calibration=False):
        # Ideal value
        ideal_value = 0.95
        # Calculate distances from the ideal value
        df['distance'] = np.abs(df['picp'] - ideal_value)    
        # Normalize distances to [0, 1]
        norm = mcolors.Normalize(vmin=df['distance'].min(), 
                                 vmax=df['distance'].max())
        cmap = plt.get_cmap('RdYlGn_r')  # Red to Green colormap, reversed
        # Plot the bar chart for picp
        plt.figure(figsize=(10, 6))
        bars = plt.bar(df.index, df['picp'], 
                       color=cmap(norm(df['distance'])))
    
        # Add the horizontal dashed line for ideal value
        plt.axhline(ideal_value, color='k', linestyle='--', 
                    label=f'Ideal PICP ({ideal_value:.2f})')
    
        # Customize the plot
        plt.xlabel('UQ Technique')
        plt.ylabel('PICP')
        plt.title(f'{self.dataset}: PICP for Different UQ Techniques')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        
        # Save the plot as a PDF file
        if not calibration:
            new_file_name = f'{self.dataset}_PICP_comparison.pdf'
        else:
            new_file_name = f'{self.dataset}_PICP_comparison_calibrated_{self.calibration}.pdf'
            
        
        new_file_path = os.path.join(self.comp_path, new_file_name)
        plt.tight_layout()
        plt.savefig(new_file_path, format='pdf')
    
        # Clear and close the plot
        plt.clf()
        plt.close()
    
    def plot_aurg(self):
        # Create a list of colors: green for positive, red for negative values
        colors = ['green' if val > 0 else 'red' for val in self.uq_df['aurg']]

        # Plot the bar chart for aurg
        plt.figure(figsize=(10, 6))
        bars = plt.bar(self.uq_df.index, self.uq_df['aurg'], color=colors)
    
        # Customize the plot
        plt.xlabel('Technique')
        plt.ylabel('AURG')
        plt.title(f'{self.dataset}: AURG for Different UQ Techniques')
        plt.xticks(rotation=45, ha='right')

        # Save the plot as a PDF file
        new_file_name = f'{self.dataset}_AURG_comparison.pdf'
        new_file_path = os.path.join(self.comp_path, new_file_name)
        plt.tight_layout()  # Adjust layout to fit labels
        plt.savefig(new_file_path, format='pdf')    
        # Clear and close the plot
        plt.clf()
        plt.close()
        
    def plot_metric_comparison(self):
        metrics = self.uq_df.columns  # Get the list of metrics (columns)
        techniques = self.uq_df.index  # Get the list of techniques (index)
        for metric in metrics:
            if not metric in ['aurg', 'mae', 'rmse', 'ause', 'distance']:
                if metric not in self.uq_df_cal.columns:
                    print(f"Warning: '{metric}' not found in uq_df_cal. Skipping...")
                    continue
                str_metric = str(metric).upper()
                plt.figure(figsize=(10, 6))
                # Get values before and after calibration for the current metric
                before_calibration = self.uq_df[metric]
                after_calibration = self.uq_df_cal[metric]
                # Set positions for the bars
                x = np.arange(len(techniques))
                width = 0.35  # Width of the bars
                # Plot bars for the metric
                plt.bar(x - width/2, before_calibration, width, 
                        label='Before Calibration', color='orange')
                plt.bar(x + width/2, after_calibration, width, 
                        label='After Calibration', color='blue')

                # Add labels and title
                plt.xlabel('Techniques')
                plt.ylabel(metric)
                plt.title(f'{self.dataset} Comparison of {metric} Before and After Calibration')
                plt.xticks(x, techniques, rotation=45, ha='right')
                plt.legend()

                # Show the plot
                plt.tight_layout()
                # Save the plot as a PDF file
                new_file_name = f'{self.dataset}_{str_metric}_calibration_effect.pdf'
                new_file_path = os.path.join(self.comp_path, new_file_name)
                plt.tight_layout()  # Adjust layout to fit labels
                plt.savefig(new_file_path, format='pdf')    
                # Clear and close the plot
                plt.clf()
                plt.close()

        
if __name__ == '__main__':
    main()