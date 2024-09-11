import argparse
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='UQ Comparison')
    parser.add_argument('--dataset', default='HelpDesk',
                        help='Dataset to be analysed') 
    parser.add_argument('--model', default='dalstm',
                        help='Type of the predictive model')
    
    args = parser.parse_args()
    root_path = os.getcwd()
    UQ_Comparison(args=args, root_path=root_path)
    
    
    # set the folder to analyse different UQ metrics saved there


# A generic class for comparing results of different UQ techniques
class UQ_Comparison ():
    def __init__ (self, args=None, root_path=None): 
        self.args = args
        self.dataset = args.dataset
        self.model = args.model
        self.result_path = os.path.join(root_path, 'results', args.dataset,
                                        args.model)
        self.cal_path = os.path.join(root_path, 'results', args.dataset,
                                        args.model, 'recalibration')
        self.comp_path = os.path.join(root_path, 'results', args.dataset,
                                        args.model, 'comparison')
        if not os.path.exists(self.comp_path):
            os.makedirs(self.comp_path)
        self.uq_df_path = os.path.join(self.comp_path, 'uq_metrics.csv')   
        self.uq_dict_lst, self.uq_method_lst = self.load_uq_dicts(
            self.result_path)
        (self.mae_lst, self.rmse_lst, self.mis_cal_lst, self.sharpness_lst,
         self.nll_lst, self.crps_lst, self.ause_lst, self.aurg_lst, 
         self.mpiw_lst, self.picp_lst, self.qice_lst, self.below_lst, 
         self.more_than_lst) = self.extract_metrics(self.uq_dict_lst)
        self.uq_df = self.lists_to_dataframe(
            self.uq_method_lst, mae=self.mae_lst, rmse=self.rmse_lst, 
            mis_cal=self.mis_cal_lst, sharpness=self.sharpness_lst,
            nll=self.nll_lst, crps=self.crps_lst, ause=self.ause_lst,
            aurg=self.aurg_lst, mpiw=self.mpiw_lst, picp=self.picp_lst,
            qice=self.qice_lst, below=self.below_lst, 
            more_than=self.more_than_lst)
        for column in self.uq_df.columns:
            if not column in ['aurg', 'picp']:
                self.plot_metrics(column)
            elif column=='picp':
                self.plot_picp()
            elif column=='aurg':
                self.plot_aurg()
        
    def load_uq_dicts(self, folder_path):
        data_list = []
        techniques = []
        # Iterate over the files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('uq_metrics.pkl'):
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
    
    def extract_metrics(self, dict_list):
        mae_lst = [d['accuracy']['mae'] for d in dict_list]
        rmse_lst = [d['accuracy']['rmse'] for d in dict_list]
        mis_cal_lst = [d['avg_calibration']['miscal_area'] for d in dict_list]
        sharpness_lst = [d['sharpness']['sharp'] for d in dict_list]
        nll_lst = [d['scoring_rule']['nll'] for d in dict_list]
        crps_lst = [d['scoring_rule']['crps'] for d in dict_list]
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
        return (mae_lst, rmse_lst, mis_cal_lst, sharpness_lst, nll_lst,
                crps_lst, ause_lst, aurg_lst, mpiw_lst, picp_lst, qice_lst,
                below_lst, more_than_lst)
    
    def lists_to_dataframe(self, techniques, **metrics):
        df = pd.DataFrame(metrics, index=techniques)
        df.index.name = 'technique'  # Set index name for clarity
        df.to_csv(self.uq_df_path)
        return df
    
    def plot_metrics(self, metric, smaller_is_better=True):
        str_metric = str(metric).upper()
        df_filtered = self.uq_df[self.uq_df.index != 'deterministic']
        # Get the value for the 'deterministic' technique
        deterministic_val = self.uq_df.loc['deterministic', metric]
    
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
        plt.title(f'{str_metric} for different UQ Techniques')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
    
        # Save the plot as a PDF file
        new_file_name = f'{self.dataset}_{str_metric}_comparison.pdf'
        new_file_path = os.path.join(self.comp_path, new_file_name)
        plt.tight_layout()
        plt.savefig(new_file_path, format='pdf')
    
        # Clear and close the plot
        plt.clf()
        plt.close()
        
    def plot_picp(self):
        # Ideal value
        ideal_value = 0.95
        # Calculate distances from the ideal value
        self.uq_df['distance'] = np.abs(self.uq_df['picp'] - ideal_value)    
        # Normalize distances to [0, 1]
        norm = mcolors.Normalize(vmin=self.uq_df['distance'].min(), 
                                 vmax=self.uq_df['distance'].max())
        cmap = plt.get_cmap('RdYlGn_r')  # Red to Green colormap, reversed
        # Plot the bar chart for picp
        plt.figure(figsize=(10, 6))
        bars = plt.bar(self.uq_df.index, self.uq_df['picp'], 
                       color=cmap(norm(self.uq_df['distance'])))
    
        # Add the horizontal dashed line for ideal value
        plt.axhline(ideal_value, color='k', linestyle='--', 
                    label=f'Ideal PICP ({ideal_value:.2f})')
    
        # Customize the plot
        plt.xlabel('UQ Technique')
        plt.ylabel('PICP')
        plt.title('PICP for Different UQ Techniques')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        
        # Save the plot as a PDF file
        new_file_name = f'{self.dataset}_PICP_comparison.pdf'
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
        plt.title('AURG for Different UQ Techniques')
        plt.xticks(rotation=45, ha='right')
    
        # Save the plot as a PDF file
        new_file_name = f'{self.dataset}_AURG_comparison.pdf'
        new_file_path = os.path.join(self.comp_path, new_file_name)
        plt.tight_layout()  # Adjust layout to fit labels
        plt.savefig(new_file_path, format='pdf')    
        # Clear and close the plot
        plt.clf()
        plt.close()
    

        
if __name__ == '__main__':
    main()