"""
To prepare this script we used uncertainty tool-box which can be find in:
    https://uncertainty-toolbox.github.io/about/
"""
import argparse
import os
import pandas as pd
import uncertainty_toolbox as uct
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from utils.evaluation_utils import get_csv_files


def main():   
    
    # Parse arguments 
    parser = argparse.ArgumentParser(
        description='Uncertainy quantification evaluation')
    parser.add_argument('--dataset', default='HelpDesk',
                        help='Datasets that is used for predictions')
    parser.add_argument('--model', default='pgtnet',
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
        y = df['GroundTruth'].values
        # TODO: add CARD model the first condition!
        if (prefix=='DA_A' or prefix=='CDA_A'):
            pred_std = df['Total_Uncertainty'].values 
        else:
            pred_std = df['Epistemic_Uncertainty'].values
        # Plot ordered prediction intervals
        uct.viz.plot_intervals_ordered(pred_mean, pred_std, y)
        plt.gcf().set_size_inches(10, 10)
        # define name of the plot to be saved
        new_file_name = base_name + 'ordered_prediction_intervals' + '.pdf'
        new_file_path = os.path.join(target_path, new_file_name)
        plt.savefig(new_file_path, format='pdf')
        plt.clf()  
        # Plot average calibration
        uct.viz.plot_calibration(pred_mean, pred_std, y)
        plt.gcf().set_size_inches(10, 10)
        new_file_name = base_name + 'miscalibrated_area' + '.pdf'
        new_file_path = os.path.join(target_path, new_file_name)
        plt.savefig(new_file_path, format='pdf')
        plt.clf()
        # Plot adversarial group calibration
        uct.viz.plot_adversarial_group_calibration(pred_mean, pred_std, y)
        plt.gcf().set_size_inches(10, 6)
        new_file_name = base_name + 'adversarial_group_calibration' + '.pdf'
        new_file_path = os.path.join(target_path, new_file_name)
        plt.savefig(new_file_path, format='pdf')
        plt.clf()
        # Get all uncertainty quantification metrics
        uq_metrics = uct.metrics.get_all_metrics(pred_mean, pred_std, y)
        new_file_name = base_name + 'uq_metrics' + '.txt'
        new_file_path = os.path.join(source_path, new_file_name)
        with open(new_file_path, 'w') as file:
            # Iterate over the dictionary items and write them to the file
            for key, value in uq_metrics.items():
                file.write(f"{key}: {value}\n")  
        # Get Spearman's rank correlation coefficient 
        if (prefix=='DA_A' or prefix=='CDA_A'):
            corr, p_value = spearmanr(df['Absolute_error'],
                                      df['Total_Uncertainty'])
        else:
            corr, p_value = spearmanr(df['Absolute_error'],
                                      df['Epistemic_Uncertainty'])
        with open(new_file_path, 'a') as file:
            file.write(f"Spearman's rank correlation coefficient: {corr}\n")
            file.write(f"P-value: {p_value}\n")        
 
if __name__ == '__main__':
    main()