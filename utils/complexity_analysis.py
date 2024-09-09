import os
import torch
from scipy import stats
from scipy.stats import wasserstein_distance
import numpy as np
import pandas as pd
from Complexity import (generate_pm4py_log, generate_log, build_graph,
                              log_complexity)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning,
                        message="ISO8601 strings are not fully supported")


def normalize(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))

"""
dataset_list = ['BPIC12', 'BPIC13I', 'BPIC15_1', 'BPIC20DD', 'BPIC20ID', 
                'BPIC20PTC', 'BPIC20RFP', 'BPIC20TPD', 'HelpDesk', 'Sepsis']
"""

# a dictionary to collect different complexity measures
# NSE: Normalized Sequence Entropy
# EMD: Earth Mover Distance (Normalized target attribute)
comlexity_dict = {'dataset': [], 'train_val_size':[], 'NSE': [], 'EMD': []}

# get Kolmogorov–Smirnov (KS) statistics
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dalstm_dir = os.path.join(root_path, 'datasets') 
log_dir = os.path.join(dalstm_dir, 'event_logs')
complexity_result_path = os.path.join(dalstm_dir, 'complexity.csv')
# Get a list of all files in the log directory
all_files = os.listdir(log_dir)
# Filter files with .xes extension and remove the extension from names
dataset_list = [os.path.splitext(file)[0] 
                for file in all_files if file.endswith('.xes')]


if not os.path.exists(complexity_result_path):
    print('Compute complexity measures for all event logs')
    for dataset in dataset_list:
    
        comlexity_dict['dataset'].append(dataset)  
    
        dalstm_class = 'DALSTM_' + dataset
        dataset_path =  os.path.join(dalstm_dir, dalstm_class)
        y_train_path = os.path.join(
            dataset_path, "DALSTM_y_train_"+dataset+".pt")
        y_val_path = os.path.join(dataset_path, "DALSTM_y_val_"+dataset+".pt")
        y_test_path = os.path.join(dataset_path, "DALSTM_y_test_"+dataset+".pt")
        y_train = torch.load(y_train_path, weights_only=True).numpy()
        y_val = torch.load(y_val_path, weights_only=True).numpy()
        y_test = torch.load(y_test_path, weights_only=True).numpy()
    
        # get number of training and validation sets.
        num_train_val = len(y_train) + len(y_val)
        comlexity_dict['train_val_size'].append(num_train_val)
        y_train_val = np.concatenate([y_train, y_val])
    
        # Compute KS statistic
        ks_statistic, p_value = stats.ks_2samp(y_train_val, y_test)
        y_train_val_normalized = normalize(y_train_val)
        y_test_normalized = normalize(y_test)
        # Compute Earth Mover's Distance (EMD)
        emd = wasserstein_distance(y_train_val_normalized, y_test_normalized)
        comlexity_dict['EMD'].append(emd)
    
        xes_name = dataset+'.xes'
        log_path = os.path.join(log_dir, xes_name)
        pm4py_log = generate_pm4py_log(log_path)
        # Transform PM4Py log into plain log
        log = generate_log(pm4py_log) 
        # Build EPA
        epa = build_graph(log)
        _,nse  = log_complexity(epa)
        comlexity_dict['NSE'].append(nse)
        
    comlexity_df = pd.DataFrame(comlexity_dict)
    comlexity_df.to_csv(complexity_result_path, index=False)
else:
    comlexity_df = pd.read_csv(complexity_result_path)

# plot the complexity analysis results
nse = comlexity_df['NSE']
emd = comlexity_df['EMD']
train_val_size = comlexity_df['train_val_size']
datasets = comlexity_df['dataset']
# Define plot limits
nse_min, nse_max = nse.min(), nse.max()
emd_min, emd_max = emd.min(), emd.max()
# Calculate midpoints for quadrants
nse_mid = (nse_min + nse_max) / 2
emd_mid = (emd_min + emd_max) / 2
# Define threshold for dataset size
size_threshold = train_val_size.median()
# Create the plot
plt.figure(figsize=(10, 8))
# Plot large datasets with blue circles and small datasets with red circles
colors = ['blue' if size > size_threshold else 'red' for size in train_val_size]
# Normalize sizes for better visualization
sizes = train_val_size / train_val_size.max() * 200

plt.scatter(nse, emd, c=colors, s=sizes, alpha=0.7, edgecolors='w', linewidth=0.5)
# Plot points
#plt.scatter(nse, emd)
# Add annotations with offset to avoid collision
for i, dataset in enumerate(datasets):
    plt.annotate(dataset, (nse.iloc[i], emd.iloc[i]),
                 textcoords="offset points", 
                 xytext=(5,5), 
                 ha='center')
# Add lines to divide the plot into quadrants
plt.axvline(x=nse_mid, color='gray', linestyle='--')
plt.axhline(y=emd_mid, color='gray', linestyle='--')
# Set labels and title
plt.xlabel('Normalized Sequence Entropy')
plt.ylabel('Earth Mover Distance (normalized remaining time train vs. test)')
plt.title('Event Complexity Plot')
# Set axis limits for better visualization
plt.xlim(nse_min - 0.05, nse_max + 0.05)
plt.ylim(emd_min - 0.05, emd_max + 0.05)
new_file_name = 'complexity_plot' + '.pdf'
new_file_path = os.path.join(dalstm_dir, new_file_name)
plt.savefig(new_file_path, format='pdf')
plt.clf()
plt.close() 
    
    
    

    
    



