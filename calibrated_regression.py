import os
import argparse
import pickle
import random
import torch
import numpy as np
import pandas as pd
import uncertainty_toolbox as uct
import warnings
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
    length_path = os.path.join(folder, dataset,
                               'DALSTM_val_length_list_'+dataset+'.pkl')
    with open(length_path, 'rb') as f:
        val_lengths =  pickle.load(f)
    df["Prefix_length"] = val_lengths.astype(int)       
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
   
    return df

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

def group_prefix_lengths(df_init, max_prefix_length, K=100, seed=42):
    df = df_init.copy()
    random.seed(seed)
    # unique prefix lengths
    existing_lengths = sorted(df['Prefix_length'].unique()) 
    all_lengths = set(range(2, max_prefix_length + 1))  
    # missing prefix lengths
    missing_lengths = list(all_lengths - set(existing_lengths)) 

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
    grouped_dict = {length: i for i, group in enumerate(groups) 
                    for length in group}
    for length in missing_lengths:
        # Find the closest existing prefix length
        distances = {group_idx: min(abs(length - l) for l in group) 
                     for group_idx, group in enumerate(groups)}
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
    model_dict, prev_calibrator = {}, {}
    for group in groups:
        subset = df[df['Prefix_length'].isin(group)]
        pred_mean = subset['Prediction'].to_numpy()
        pred_std = subset["Total_Uncertainty"].to_numpy()
        y_true = subset["GroundTruth"].to_numpy()
        try:
            interval_recalibrator = uct.recalibration.get_interval_recalibrator(
                pred_mean, pred_std, y_true)
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
        recalibrated_interval = interval_recalibrator(
            pred_mean, pred_std, quantile)
        # Extract upper bound as numpy array
        upper = np.array(recalibrated_interval.upper)  
        # Extract lower bound as numpy array
        lower = np.array(recalibrated_interval.lower)  
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
    scale_dict, prev_scale = {}, {}
    for group in groups:
        subset = df[df['Prefix_length'].isin(group)]
        pred_mean = subset['Prediction'].to_numpy()
        pred_std = subset["Total_Uncertainty"].to_numpy()
        y_true = subset["GroundTruth"].to_numpy()
        try:
            miscal_scale = uct.recalibration.optimize_recalibration_ratio(
                pred_mean, pred_std, y_true, criterion="miscal")
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
            # Fallback if no surrounding keys
            return np.mean(list(scale_dict.values()))  
        lower_key = lower_keys.max()
        upper_key = upper_keys.min()
        return (scale_dict[lower_key] + scale_dict[upper_key]) / 2
    
    scale_values = df['Prefix_length'].apply(get_scale_value)
    df[["Epistemic_Uncertainty", "Aleatoric_Uncertainty", "Total_Uncertainty"]
       ] *= scale_values.to_numpy()[:, None]
    if avoid_negative:
        df["Total_Uncertainty"] = np.minimum(
            df["Total_Uncertainty"], (1/3) * df["Prediction"])
        df["Aleatoric_Uncertainty"] = np.sqrt(
            df["Total_Uncertainty"]**2-df["Epistemic_Uncertainty"]**2)
    return df


def main():
    # Parse arguments     
    parser = argparse.ArgumentParser(
        description='Process_Aware Calibrated Regression') 
    parser.add_argument('--dataset', help='Dataset used by model')
    
    args = parser.parse_args()    
    root_path = os.getcwd()
    folder = os.path.join(root_path, 'results', args.dataset)
    
    # models included in our experiments
    models = ['DA+H', 'CDA+H', 'LA', 'BE+H', 'H', 'CARD']
    # model selected for Calibrated Regression
    sub_models = ['LA']
    # frequencies for partitioning algorithm
    # e.g., each prefix group should at least cover 0.01 of the data if K=1.
    frequencies = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] 
    # metric to choose best partition
    focus = 'aurg'
    csv_path =  os.path.join(folder, args.dataset,
                             args.dataset + '_overal_results.csv')
    performance_dict = {'Model': [], 'MAE': [], 'MA': [], 'Sharp':[], 'AURG':[]} 
    
    # get the length of prefixes in the validation set
    dalstm_name = 'DALSTM_' + args.dataset
    data_path = os.path.join(root_path, 'datasets', dalstm_name)
    X_path = os.path.join(data_path, 'DALSTM_X_val_'+args.dataset+'.pt')
    length_path = os.path.join(
        data_path, 'DALSTM_val_length_list_'+args.dataset+'.pkl') 
    X = torch.load(X_path)
    padding_value = 0.0    
    lengths = (X.abs().sum(dim=2) != padding_value).sum(dim=1)
    lengths = lengths.numpy()
    with open(length_path, 'wb') as file:
        pickle.dump(lengths, file)
    
    for model in models:
        df_test = load_results(args.dataset, model, folder, mode = 'test')
        # Compute metrics for test set
        pred_mean = df_test["Prediction"].to_numpy()
        pred_std = df_test["Total_Uncertainty"].to_numpy()
        y_true = df_test["GroundTruth"].to_numpy()  
        mae = compute_mae(y_true, pred_mean)
        ma = uct.metrics_calibration.miscalibration_area(
            pred_mean, pred_std, y_true)
        sharp = uct.metrics_calibration.sharpness(pred_std)
        _, aurg = get_sparsification(pred_mean, y_true, pred_std)
        # Add metrics to dictionary
        performance_dict['Model'].append(model)
        performance_dict['MAE'].append(mae)
        performance_dict['MA'].append(ma)
        performance_dict['Sharp'].append(sharp)
        performance_dict['AURG'].append(aurg)    
        if model in sub_models:
            df_cal = load_results(args.dataset, model, folder, mode = 'val')
            # group prefix lengths together
            max_pl = max(df_test['Prefix_length'].max(), 
                         df_cal['Prefix_length'].max())
            k_selected = []
            subset_selection = []
            for k in frequencies:
                pl_groups = group_prefix_lengths(
                    df_cal, max_prefix_length=max_pl, K=k)
                if pl_groups in subset_selection:
                    continue
                subset_selection.append(pl_groups)
                k_selected.append(k)
            
            # Isotonic regression
            # Choose the best groups
            ma_lst, sharp_lst, aurg_lst, k_list = [], [], [], []
            for index, pl_groups in enumerate(subset_selection):    
                model_dict = learn_interval_calibration(df_cal, pl_groups)
                df_calibrated_pl = istonic_pl_based(df_cal, model_dict)    
                pred_mean = df_calibrated_pl["Prediction"].to_numpy()
                pred_std = df_calibrated_pl["Total_Uncertainty"].to_numpy()
                y_true = df_calibrated_pl["GroundTruth"].to_numpy()
                ma = uct.metrics_calibration.miscalibration_area(
                    pred_mean, pred_std, y_true)
                sharp = uct.metrics_calibration.sharpness(pred_std)
                _, aurg = get_sparsification(pred_mean, y_true, pred_std)
                ma_lst.append(ma)
                sharp_lst.append(sharp)
                aurg_lst.append(aurg)
                k_list.append(k_selected[index])
            if focus == 'aurg':
                max_index = aurg_lst.index(max(aurg_lst))
                #print(k_list[max_index])
            elif focus == 'ma':
                max_index = ma_lst.index(min(ma_lst))
            elif focus == 'sharp':
                max_index = sharp_lst.index(min(sharp_lst))  
            pl_groups = subset_selection[max_index]        
            # Apply isotonic regression for the best group
            model_dict = learn_interval_calibration(df_cal, pl_groups)
            df_calibrated_pl = istonic_pl_based(df_test, model_dict)
            pred_mean = df_calibrated_pl["Prediction"].to_numpy()
            pred_std = df_calibrated_pl["Total_Uncertainty"].to_numpy()
            y_true = df_calibrated_pl["GroundTruth"].to_numpy()
            ma = uct.metrics_calibration.miscalibration_area(
                pred_mean, pred_std, y_true)
            sharp = uct.metrics_calibration.sharpness(pred_std)
            _, aurg = get_sparsification(pred_mean, y_true, pred_std)
            model_name = model + '+I'
            performance_dict['Model'].append(model_name)
            performance_dict['MAE'].append(mae)
            performance_dict['MA'].append(ma)
            performance_dict['Sharp'].append(sharp)
            performance_dict['AURG'].append(aurg)
            
            # Scaling-based CR
            # Choose the best groups
            ma_lst, sharp_lst, aurg_lst, k_list = [], [], [], []
            for index, pl_groups in enumerate(subset_selection): 
                scale_dict = group_length_calibration(df_cal, pl_groups)
                df_calibrated = scale_uncertainties(scale_dict, df_test)
                pred_mean = df_calibrated["Prediction"].to_numpy()
                pred_std = df_calibrated["Total_Uncertainty"].to_numpy()
                y_true = df_calibrated["GroundTruth"].to_numpy()
                ma = uct.metrics_calibration.miscalibration_area(
                    pred_mean, pred_std, y_true)
                sharp = uct.metrics_calibration.sharpness(pred_std)
                _, aurg = get_sparsification(pred_mean, y_true, pred_std)
                ma_lst.append(ma)
                sharp_lst.append(sharp)
                aurg_lst.append(aurg)
                k_list.append(k_selected[index])
            if focus == 'aurg':
                max_index = aurg_lst.index(max(aurg_lst))
                #print(k_list[max_index])
            elif focus == 'ma':
                max_index = ma_lst.index(min(ma_lst))
            elif focus == 'sharp':
                max_index = sharp_lst.index(min(sharp_lst)) 
            pl_groups = subset_selection[max_index] 
            # Apply scaling based CR for the best group
            scale_dict = group_length_calibration(df_cal, pl_groups)
            df_calibrated = scale_uncertainties(scale_dict, df_test)
            pred_mean = df_calibrated["Prediction"].to_numpy()
            pred_std = df_calibrated["Total_Uncertainty"].to_numpy()
            y_true = df_calibrated["GroundTruth"].to_numpy()
            ma = uct.metrics_calibration.miscalibration_area(
                pred_mean, pred_std, y_true)
            sharp = uct.metrics_calibration.sharpness(pred_std)
            _, aurg = get_sparsification(pred_mean, y_true, pred_std)
            model_name = model + '+S'
            performance_dict['Model'].append(model_name)
            performance_dict['MAE'].append(mae)
            performance_dict['MA'].append(ma)
            performance_dict['Sharp'].append(sharp)
            performance_dict['AURG'].append(aurg)
    results_df = pd.DataFrame(performance_dict)
    results_df.to_csv(csv_path, index=False)
            
    
if __name__ == '__main__':
    main()   