import os
import pandas as pd
import numpy as np
#import scipy.stats as stats
#from scipy.optimize import minimize
#from scipy.stats import gaussian_kde
import pickle
import uncertainty_toolbox as uct
#import matplotlib.pyplot as plt
#from matplotlib.backends.backend_pdf import PdfPages
#import seaborn as sns
import random
import warnings
from scipy.stats import wilcoxon
warnings.simplefilter("ignore", category=RuntimeWarning)

# method to get the path and csv file for validation and test result
def get_result(dataset, model, mode, folder):
    dataset_folder = os.path.join(folder, dataset)
    for file in os.listdir(dataset_folder):
        if file.startswith(model + "_holdout"):
            if mode == "val" and file.endswith("validation_.csv"):
                csv_path = os.path.join(dataset_folder, file)
                df = pd.read_csv(csv_path)
                return df, csv_path
            elif mode == "test" and file.endswith("result_.csv"):
                csv_path = os.path.join(dataset_folder, file)
                df = pd.read_csv(csv_path)
                return df, csv_path    
    return None, None

# method to augment dataframes by prefix length and time since start
def add_val_length(folder, dataset, df):
    length_path = os.path.join(folder, dataset, 'DALSTM_val_length_list_'+dataset+'.pkl')
    with open(length_path, 'rb') as f:
        val_lengths =  pickle.load(f)
    df["Prefix_length"] = val_lengths.astype(int)       
    return df

def correct_uncertainty (df_org, model):
    df = df_org.copy()
    if model in ['CARD', 'mve']:
        df['Total_Uncertainty'] = df['Aleatoric_Uncertainty']
        df['Epistemic_Uncertainty'] = 0
    elif model in ['CDA_A', 'DA_A']:  
        df['Total_Uncertainty'] = np.sqrt(df['Aleatoric_Uncertainty']**2 + df['Epistemic_Uncertainty']**2)
    elif model in ['en_b_mve']:   
        df['Total_Uncertainty'] = np.sqrt(df['Aleatoric_Uncertainty']**2 + df['Epistemic_Uncertainty']**2)
        df['Aleatoric_Uncertainty'] = df['Total_Uncertainty']
        df['Epistemic_Uncertainty'] = 0
    return df 


def load_results(dataset, model, folder, mode = 'val'): 
    
    df, _ = get_result(dataset, model, mode, folder)
    
    if mode == 'val':        
        df = add_val_length(folder, dataset, df)
        
    # add trace lengths to the dataframe
    df['Case_Group'] = (df['Prefix_length'] == 2).cumsum()
    df['Trace_length'] = df.groupby('Case_Group')['Prefix_length'].transform('max')
    df['Trace_length'] = df['Trace_length'] + 1
    df.drop(columns=['Case_Group'], inplace=True)

    df_adj = correct_uncertainty (df, model)  
    
    #return df_adj, _
    #df_test.to_csv(df_val_path, index=False)
    
    return df_adj  

def compute_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


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

def compare_models_wilcoxon(model1_scores, model2_scores, higher_is_better=True):
    """
    Compare two models using the Wilcoxon signed-rank test at three confidence levels.
    
    Parameters:
    - model1_scores (list): Performance scores of model 1 across datasets.
    - model2_scores (list): Performance scores of model 2 across datasets.
    - higher_is_better (bool): If True, higher values indicate better performance;
                               if False, lower values indicate better performance.
    """
    if not higher_is_better:
        # Invert the values so that higher values always mean better performance
        model1_scores = [-x for x in model1_scores]
        model2_scores = [-x for x in model2_scores]
    
    # Perform Wilcoxon signed-rank test
    stat, p_value = wilcoxon(model1_scores, model2_scores)
    
    # Print results for different confidence levels
    print("Wilcoxon Signed-Rank Test Results:")
    print(f"Statistic: {stat:.4f}, p-value: {p_value:.4f}")
    
    for alpha in [0.05, 0.10, 0.20]:
        confidence = 1 - alpha
        if p_value < alpha:
            print(f"At {confidence*100:.0f}% confidence, there is a significant difference between the models.")
        else:
            print(f"At {confidence*100:.0f}% confidence, there is NO significant difference between the models.")
            
def group_prefix_lengths(df_init, max_prefix_length, K=100, seed=42):
    df = df_init.copy()
    random.seed(seed)
    
    existing_lengths = sorted(df['Prefix_length'].unique()) # unique prefix lengths
    all_lengths = set(range(2, max_prefix_length + 1))    
    missing_lengths = list(all_lengths - set(existing_lengths)) # missing prefix lengths

    # Step 1: Group existing lengths into at most K groups
    groups = []
    temp_group = []

    for i, length in enumerate(existing_lengths):
        temp_group.append(length)
        subset = df[df['Prefix_length'].isin(temp_group)]
        if len(subset) > (len(df)/K) or i == len(existing_lengths) - 1:        
            groups.append(temp_group)
            temp_group = []
    
    # Step 2: Assign missing lengths to the nearest group
    grouped_dict = {length: i for i, group in enumerate(groups) for length in group}
    for length in missing_lengths:
        # Find the closest existing prefix length
        distances = {group_idx: min(abs(length - l) for l in group) for group_idx, group in enumerate(groups)}
        min_distance = min(distances.values())
        # Get groups with the minimum distance
        closest_groups = [g for g, d in distances.items() if d == min_distance]
        # Randomly choose one if multiple groups have the same distance
        assigned_group = random.choice(closest_groups)
        grouped_dict[length] = assigned_group
        groups[assigned_group].append(length)
        
    return groups

def learn_interval_calibration(df_init, groups):
    df = df_init.copy()
    model_dict = {}
    for group in groups:
        subset = df[df['Prefix_length'].isin(group)]
        pred_mean = subset['Prediction'].to_numpy()
        pred_std = subset["Total_Uncertainty"].to_numpy()
        y_true = subset["GroundTruth"].to_numpy()
        try:
            interval_recalibrator = uct.recalibration.get_interval_recalibrator(pred_mean, pred_std, y_true)
        except:
            interval_recalibrator = prev_calibrator
        for length in group:
            model_dict[length] = interval_recalibrator
            prev_calibrator = interval_recalibrator
    return model_dict

def istonic_pl_based(df_init, model_dict, quantile = 0.95, z_double = 3.92):
    df_list = []
    df = df_init.copy()
    for length, interval_recalibrator in model_dict.items():
        subset = df[df['Prefix_length'] == length]
        pred_mean = subset["Prediction"].to_numpy()
        pred_std = subset["Total_Uncertainty"].to_numpy()
        y_true = subset["GroundTruth"].to_numpy()
        recalibrated_interval = interval_recalibrator(pred_mean, pred_std, quantile)
        upper = np.array(recalibrated_interval.upper)  # Extract upper bound as numpy array
        lower = np.array(recalibrated_interval.lower)  # Extract lower bound as numpy array
        subset_copy = subset.copy()
        subset_copy["Prediction"] = (lower+upper)/2
        subset_copy["new_std"] = (upper-lower)/z_double
        subset_copy["quantile_scale"] = subset["Total_Uncertainty"]/subset_copy["new_std"]
        #print(subset_copy["quantile_scale"].mean())
        subset_copy["Epistemic_Uncertainty"] = subset["Epistemic_Uncertainty"]*subset_copy["quantile_scale"]
        subset_copy["Aleatoric_Uncertainty"] = subset["Aleatoric_Uncertainty"]*subset_copy["quantile_scale"]
        subset_copy["Total_Uncertainty"] = subset_copy["new_std"]
        df_list.append(subset_copy)
    merged_df = pd.concat(df_list, ignore_index=True) 
    return merged_df

def group_length_calibration(df_init, groups):
    df = df_init.copy()
    scale_dict = {}
    for group in groups:
        subset = df[df['Prefix_length'].isin(group)]
        pred_mean = subset['Prediction'].to_numpy()
        pred_std = subset["Total_Uncertainty"].to_numpy()
        y_true = subset["GroundTruth"].to_numpy()
        try:
            miscal_scale = uct.recalibration.optimize_recalibration_ratio(pred_mean, pred_std, y_true, criterion="miscal")
        except:
            miscal_scale = prev_scale
        for length in group:
            scale_dict[length] = miscal_scale
        prev_scale = miscal_scale
    return scale_dict

def scale_uncertainties(scale_dict, df_init, avoid_negative = False):
    df = df_init.copy()
    def get_scale_value(prefix):
        if prefix in scale_dict:
            return scale_dict[prefix]
        keys = np.array(list(scale_dict.keys()))
        lower_keys = keys[keys < prefix]
        upper_keys = keys[keys > prefix]
        if len(lower_keys) == 0 or len(upper_keys) == 0:
            return np.mean(list(scale_dict.values()))  # Fallback if no surrounding keys
        lower_key = lower_keys.max()
        upper_key = upper_keys.min()
        return (scale_dict[lower_key] + scale_dict[upper_key]) / 2
    
    scale_values = df['Prefix_length'].apply(get_scale_value)
    df[["Epistemic_Uncertainty", "Aleatoric_Uncertainty", "Total_Uncertainty"]] *= scale_values.to_numpy()[:, None]
    if avoid_negative:
        df["Total_Uncertainty"] = np.minimum(df["Total_Uncertainty"], (1/3) * df["Prediction"])
        df["Aleatoric_Uncertainty"] = np.sqrt(df["Total_Uncertainty"]**2-df["Epistemic_Uncertainty"]**2)
    return df

            
