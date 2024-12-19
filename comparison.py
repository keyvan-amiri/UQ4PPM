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
    parser.add_argument('--calibration', default='miscal',
                        help='Type of the calibrated regression used') 
    args = parser.parse_args()
    root_path = os.getcwd()
    UQ_Comparison(args=args, root_path=root_path)
    


# A generic class for comparing results of different UQ techniques
class UQ_Comparison ():
    def __init__ (self, args=None, root_path=None): 
        self.args = args
        self.dataset = args.dataset
        self.model = args.model
        self.calibration = args.calibration
        self.result_path = os.path.join(root_path, 'results', args.dataset,
                                        args.model)
        self.cal_path = os.path.join(root_path, 'results', args.dataset,
                                        args.model, 'recalibration')
        self.rpa_path = os.path.join(root_path, 'results', args.dataset,
                                        args.model, 'RPA')
        self.comp_path = os.path.join(root_path, 'results', args.dataset,
                                        args.model, 'comparison')
        if not os.path.exists(self.comp_path):
            os.makedirs(self.comp_path)
        
        # comparison for results before calibrated regression and RPA
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
                
        # comparison for results after RPA
        self.uq_df_path = os.path.join(self.comp_path, 
                                       self.dataset + '_uq_metrics_rpa.csv')   
        self.uq_dict_lst, self.init_uq_method_lst = self.load_uq_dicts(
            self.rpa_path)         
        (self.mae_lst, self.rmse_lst, self.mis_cal_lst, self.sharpness_lst,
         self.nll_lst, self.crps_lst, self.ause_lst, self.aurg_lst, 
         self.mpiw_lst, self.picp_lst, self.qice_lst, self.below_lst, 
         self.more_than_lst) = self.extract_metrics(self.uq_dict_lst)        
        self.uq_method_lst = self.replace_techniques()
        self.uq_df_rpa = self.lists_to_dataframe(
            self.uq_method_lst, mae=self.mae_lst, rmse=self.rmse_lst, 
            mis_cal=self.mis_cal_lst, sharpness=self.sharpness_lst,
            nll=self.nll_lst, crps=self.crps_lst, ause=self.ause_lst,
            aurg=self.aurg_lst, mpiw=self.mpiw_lst, picp=self.picp_lst,
            qice=self.qice_lst, below=self.below_lst, 
            more_than=self.more_than_lst)
        for column in self.uq_df_rpa.columns:
            if not column in ['aurg', 'picp']:
                self.plot_metrics(self.uq_df_rpa, column, rpa=True)
            elif column=='picp':
                self.plot_picp(self.uq_df_rpa, rpa=True)
            elif column=='aurg':
                self.plot_aurg(rpa=True)
        
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
        self.ause_lst = self.uq_df['ause'].tolist()
        self.aurg_lst = self.uq_df['aurg'].tolist()
        self.uq_method_lst = self.replace_techniques()        
        self.uq_df_cal = self.lists_to_dataframe_calibrated(
            self.uq_method_lst, self.ause_lst, self.aurg_lst, mae=self.mae_lst,
            rmse=self.rmse_lst, mis_cal=self.mis_cal_lst, 
            sharpness=self.sharpness_lst, nll=self.nll_lst, crps=self.crps_lst,
            mpiw=self.mpiw_lst, picp=self.picp_lst, qice=self.qice_lst,
            below=self.below_lst, more_than=self.more_than_lst)
        for column in self.uq_df_cal.columns:
            if not column in ['aurg', 'mae', 'rmse', 'picp', 'ause']:
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
        'BE+H': 'BE+H',
        'DE': 'DE',
        'DA+H': 'DA+H',
        'CDA+H': 'CDA+H',
        'DE+H': 'DE+H',
        'BE': 'BE',
        'H': 'H'
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
        df_sorted = df.sort_index()
        df_sorted.to_csv(self.uq_df_path)
        return df_sorted
    
    def lists_to_dataframe_calibrated(self, techniques, ause_list, aurg_list, 
                                      **metrics):
        df = pd.DataFrame(metrics, index=techniques)
        df.index.name = 'technique'  # Set index name for clarity
        df_sorted = df.sort_index()
        df_sorted['ause'] = ause_list
        df_sorted['aurg'] = aurg_list        
        df_sorted.to_csv(self.uq_df_path)
        return df_sorted
    
    
    
    def plot_metrics(self, df, metric, smaller_is_better=True,
                     calibration=False, rpa=False):
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
            if not rpa:
                new_file_name = f'{self.dataset}_{str_metric}_comparison.pdf'
            else:
                new_file_name = f'{self.dataset}_{str_metric}_comparison_adjusted.pdf'
        else:
            new_file_name = f'{self.dataset}_{str_metric}_comparison_calibrated_{self.calibration}.pdf'
        new_file_path = os.path.join(self.comp_path, new_file_name)
        plt.tight_layout()
        plt.savefig(new_file_path, format='pdf')
    
        # Clear and close the plot
        plt.clf()
        plt.close()
        
    def plot_picp(self, df, calibration=False, rpa=False):
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
            if not rpa:
                new_file_name = f'{self.dataset}_PICP_comparison.pdf'
            else:
                new_file_name = f'{self.dataset}_PICP_comparison_adjusted.pdf'
        else:
            new_file_name = f'{self.dataset}_PICP_comparison_calibrated_{self.calibration}.pdf'
            
        
        new_file_path = os.path.join(self.comp_path, new_file_name)
        plt.tight_layout()
        plt.savefig(new_file_path, format='pdf')
    
        # Clear and close the plot
        plt.clf()
        plt.close()
    
    def plot_aurg(self, rpa = False):
        # Create a list of colors: green for positive, red for negative values
        colors = ['green' if val > 0 else 'red' for val in self.uq_df['aurg']]

        # Plot the bar chart for aurg
        plt.figure(figsize=(10, 6))
        if not rpa:
            bars = plt.bar(self.uq_df.index, self.uq_df['aurg'], color=colors)
        else:
            bars = plt.bar(self.uq_df_rpa.index, self.uq_df_rpa['aurg'], color=colors)
            
        # Customize the plot
        plt.xlabel('Technique')
        plt.ylabel('AURG')
        plt.title(f'{self.dataset}: AURG for Different UQ Techniques')
        plt.xticks(rotation=45, ha='right')

        # Save the plot as a PDF file
        if not rpa:
            new_file_name = f'{self.dataset}_AURG_comparison.pdf'
        else:
            new_file_name = f'{self.dataset}_AURG_comparison_adjusted.pdf'
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
            if not metric in ['distance']:
                if metric not in self.uq_df_cal.columns:
                    print(f"Warning: '{metric}' not found in uq_df_cal. Skipping...")
                    continue
                str_metric = str(metric).upper()
                plt.figure(figsize=(10, 6))
                # Get values before and after calibration for the current metric
                initial_result = self.uq_df[metric]
                if not metric in ['aurg', 'mae', 'rmse', 'ause']:                
                    after_calibration = self.uq_df_cal[metric]
                else:
                    after_calibration = initial_result
                after_rpa = self.uq_df_rpa[metric]
                # Set positions for the bars
                x = np.arange(len(techniques))
                width = 0.24  # Width of the bars
                # Plot bars for the metric
                plt.bar(x - width, initial_result, width, 
                        label='Initial Result', color='orange')                
                plt.bar(x, after_calibration, width, 
                        label='After CR', color='blue')
                plt.bar(x + width, after_rpa, width, 
                        label='After RPA', color='olive') 

                # Add labels and title
                plt.xlabel('Techniques')
                plt.ylabel(metric)
                plt.title(f'{self.dataset} Comparison of {metric} Before and After Post-hoc improvement')
                plt.xticks(x, techniques, rotation=45, ha='right')
                plt.legend()

                # Show the plot
                plt.tight_layout()
                # Save the plot as a PDF file
                new_file_name = f'{self.dataset}_{str_metric}_post-hoc_effect.pdf'
                new_file_path = os.path.join(self.comp_path, new_file_name)
                plt.tight_layout()  # Adjust layout to fit labels
                plt.savefig(new_file_path, format='pdf')    
                # Clear and close the plot
                plt.clf()
                plt.close()

        
if __name__ == '__main__':
    main()