import argparse
import os
import yaml
import re
import pickle
import pandas as pd
import torch
import itertools
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import uncertainty_toolbox as uct
from uncertainty_toolbox.metrics_scoring_rule import nll_gaussian
from uncertainty_toolbox.metrics_calibration import (miscalibration_area,
                                                     sharpness)
from utils.utils import get_val_dataframes, get_mean_std_truth
from utils.evaluation import get_sparsification, uq_eval
from utils.calibration import calibrated_regression
from loss.GMMLoss import GMM_Loss
    

def mixture_inference(args, best_combination, techniques, test_df_lst,
                      validation=True):
    # TODO: implement GMM for cross-fold validation
    csv_val_name = 'GMM_{}_{}_seed_{}_inference_result_validation_.csv'.format(
        args.style, args.split,args.seed)
    csv_val_path = os.path.join(args.result_path, csv_val_name)
    csv_test_name = 'GMM_{}_{}_seed_{}_inference_result_.csv'.format(
        args.style, args.split,args.seed)
    csv_test_path = os.path.join(args.result_path, csv_test_name)
    if not validation:
        test_lengths = test_df_lst[0]['Prefix_length']
           
    pred_mean_dict = {}
    pred_std_dict = {}
    weight_dict = {}
    for technique in best_combination.keys():
        # Get the index of the technique in the list
        index = techniques.index(technique)
        df = test_df_lst[index]
        weight = best_combination.get(technique)
        pred_mean, pred_std, y_true = get_mean_std_truth(
            df=df, uq_method=technique)
        pred_mean_dict[technique] = pred_mean
        pred_std_dict[technique] = pred_std
        weight_dict[technique] = weight
    # get the prediction means for the mixture of Gaussians
    mixture_mean = 0
    for technique in pred_mean_dict:
        weight = weight_dict[technique]
        pred_mean = pred_mean_dict[technique]
        mixture_mean += pred_mean * weight
    # get the prediction std for the mixture of Gaussians
    mixture_var = 0
    for technique in pred_mean_dict:
        weight = weight_dict[technique]
        pred_mean = pred_mean_dict[technique]
        pred_std = pred_std_dict[technique]      
        mixture_var += (pred_std**2 + (pred_mean-mixture_mean)**2)* weight
    mixture_std = np.sqrt(mixture_var)
    # now create the inference dataframe using the mixture of Gaussians
    if not validation:
        mixture_df = pd.DataFrame(
            {'GroundTruth': y_true, 'Prediction': mixture_mean, 
             'Total_Uncertainty': mixture_std, 'Prefix_length': test_lengths})
    else:
        mixture_df = pd.DataFrame(
            {'GroundTruth': y_true, 'Prediction': mixture_mean, 
             'Total_Uncertainty': mixture_std})
    mixture_df['Absolute_error'] = (
        mixture_df['GroundTruth'] - mixture_df['Prediction']).abs()
    # for numerical stability
    threshold=1e-6
    mixture_df['Absolute_percentage_error'] = np.where(
    np.abs(mixture_df['GroundTruth']) > threshold,
    (mixture_df['Absolute_error'] / mixture_df['GroundTruth']) * 100,
    0)
    if validation:
        mixture_df.to_csv(csv_val_path, index=False)
    else:
        mixture_df.to_csv(csv_test_path, index=False)
        _ = uq_eval(csv_test_path, 'GMM', report=True, verbose=True)  
        recalibration_path = os.path.join(args.result_path, 'recalibration')
        calibrated_result, recal_model = calibrated_regression(
            calibration_df_path=csv_val_path, test_df_path=csv_test_path, 
            uq_method='GMM', confidence_level=0.95,
            recalibration_path=recalibration_path, report=False)  
        uq_eval(calibrated_result, 'GMM', report=True, verbose=True,
                calibration_mode=True, calibration_type=args.calibration_type,
                recal_model=recal_model)


def get_best_combination(args, experiments, techniques, df_lst):    
    # compute NLL for all combinations of techniques
    exp_dict = {}
    for experiment in experiments:
        if args.style == 'uniform':
            NLL_exp, weight_dict = static_mixture(
                experiment, techniques, df_lst)
            exp_dict[tuple(experiment)] = {
                'score': NLL_exp, 'weights': weight_dict}
        elif args.style == 'dynamic':
            Loss_exp, weight_dict = dynamic_mixture(
                args, experiment, techniques, df_lst)
            exp_dict[tuple(experiment)] = {
                'score': Loss_exp, 'weights': weight_dict}
        else:
            raise ValueError('Mixture style is not supported.')
    # find the combination with lowes NLL
    lowest_score = float('inf')
    #lowest_score_key = None
    lowest_score_value = None
    for key, value in exp_dict.items():
        score = value['score']
        if score < lowest_score:
            lowest_score = score
            #lowest_score_key = key
            lowest_score_value = value
    return lowest_score_value.get('weights')


def dynamic_mixture(args, experiment, techniques, df_lst):
    # collect predictions for all components
    # Initialize lists to store tensors
    pred_mean_list = []
    pred_std_list = []
    for index in experiment:
        df = df_lst[index]
        technique = techniques[index]
        pred_mean, pred_std, y_true = get_mean_std_truth(
            df=df, uq_method=technique)
        pred_mean_list.append(torch.tensor(pred_mean))
        pred_std_list.append(torch.tensor(pred_std))
    # stack the result for all experiments
    pred_mean_tensor = torch.stack(pred_mean_list, dim=1)
    pred_std_tensor = torch.stack(pred_std_list, dim=1)
    y_true_tensor=torch.tensor(y_true)
    #print(pred_mean_tensor.size())
    num_tech = len(experiment)
    # initial uniform weights for starting point
    uniform_weight = 1 / num_tech
    weights = torch.full((num_tech,), uniform_weight, dtype=torch.double, 
                         requires_grad=True)
    #print(weights.size())
    gmm_loss_fn = GMM_Loss()
    hyper_optimizer = torch.optim.Adam([weights], lr=args.gmm_lr)
    for epoch in range(args.gmm_epochs):
        hyper_optimizer.zero_grad()
        # Ensure the weights are normalized (sum to 1)
        weights = weights / weights.sum()
        # get the prediction means for the mixture of Gaussians
        mixture_mean = torch.matmul(pred_mean_tensor, weights)
        mixture_mean_expanded = mixture_mean.unsqueeze(1)  # (num_samples, 1)
        mean_diff_squared = (pred_mean_tensor - mixture_mean_expanded) ** 2  
        std_squared = pred_std_tensor ** 2 
        mixture_var = torch.matmul(std_squared + mean_diff_squared, weights)
        mixture_std = torch.sqrt(mixture_var)
        #print(mixture_mean.size())
        #print(mixture_std.size())
        loss = gmm_loss_fn(mixture_mean, mixture_std, y_true_tensor, args.alpha)
        loss.backward()
        hyper_optimizer.step()

    weights = weights / weights.sum()
    mixture_mean = torch.matmul(pred_mean_tensor, weights)
    mixture_mean_expanded = mixture_mean.unsqueeze(1)  # (num_samples, 1)
    mean_diff_squared = (pred_mean_tensor - mixture_mean_expanded) ** 2  
    std_squared = pred_std_tensor ** 2 
    mixture_var = torch.matmul(std_squared + mean_diff_squared, weights)
    mixture_std = torch.sqrt(mixture_var)
    final_loss = gmm_loss_fn(mixture_mean, mixture_std, y_true_tensor, args.alpha)
    weights = weights.cpu().detach().numpy()
    weight_dict = {}
    for i, index in enumerate(experiment):
        technique = techniques[index]
        weight_dict[technique] = weights[i]
    
    return final_loss, weight_dict
   
def static_mixture(experiment, techniques, df_lst):
    # get uniform weights for models
    num_tech = len(experiment)
    uniform_weight = 1 / num_tech
    # collect predictions for all components
    pred_mean_dict = {}
    pred_std_dict = {}
    weight_dict = {}
    for index in experiment:
        df = df_lst[index]
        technique = techniques[index]
        pred_mean, pred_std, y_true = get_mean_std_truth(
            df=df, uq_method=technique)
        pred_mean_dict[technique] = pred_mean
        pred_std_dict[technique] = pred_std
        weight_dict[technique] = uniform_weight
    # get the prediction means for the mixture of Gaussians
    mixture_mean = 0
    for technique in pred_mean_dict:
        weight = weight_dict[technique]
        pred_mean = pred_mean_dict[technique]
        mixture_mean += pred_mean * weight
    # get the prediction std for the mixture of Gaussians
    mixture_var = 0
    for technique in pred_mean_dict:
        weight = weight_dict[technique]
        pred_mean = pred_mean_dict[technique]
        pred_std = pred_std_dict[technique]      
        mixture_var += (pred_std**2 + (pred_mean-mixture_mean)**2)* weight
    mixture_std = np.sqrt(mixture_var)
    # get NLL score for the selected combinations of techniques
    NLL_exp =  nll_gaussian(mixture_mean, mixture_std, y_true, scaled=True)       
    return NLL_exp, weight_dict

def main():
    parser = argparse.ArgumentParser(
        description='Use mixture of Gaussians to combine UQ techniques')
    parser.add_argument('--cfg', help='configuration for fitting GMM model.')
    parser.add_argument('--style', default='uniform',
                        help='weights for mixture components: uniform/dynamic')
    args = parser.parse_args()   
    root_path = os.getcwd()
    # read the relevant cfg file
    cfg_file = os.path.join(root_path, 'cfg', args.cfg)
    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)
    # set arguments
    args.dataset = cfg.get('dataset')
    args.model = cfg.get('model')
    args.split = cfg.get('split')
    args.seed = cfg.get('seed')
    args.calibration_type = cfg.get('calibration_type')
    args.num_acc = cfg.get('num_models').get('accuracy')
    args.num_cal = cfg.get('num_models').get('calibration')
    args.num_spa = cfg.get('num_models').get('sparsification')
    args.num_sha = cfg.get('num_models').get('sharpness')
    args.gmm_lr = cfg.get('gmm_lr')
    args.gmm_epochs = cfg.get('gmm_epochs')
    args.alpha = cfg.get('accuracy_emphasize')
    args.result_path = os.path.join(
        root_path, 'results', args.dataset, args.model)
    # get all validation dataframes created for the model
    val_df_names = [f for f in os.listdir(args.result_path)
                    if f.endswith('inference_result_validation_.csv')
                    and not f.startswith('GMM')]
    # Collect UQ technique names 
    techniques = [f.split('_'+args.split)[0] for f in val_df_names]
    # Collect a list of dataframes for each val_df_path
    df_lst = []
    for val_name in val_df_names:
        df = pd.read_csv(os.path.join(args.result_path, val_name))    
        df_lst.append(df)
    # Get sparsification, miscalibration area, sharpness scores.
    SP_score_lst, miscal_lst, sharp_lst, accuracy_lst = [], [], [], []
    for df, technique in zip(df_lst, techniques):
        # get mean, standard deviation for predicted values, plus ground truths
        pred_mean, pred_std, y_true = get_mean_std_truth(
            df=df, uq_method=technique)
        # get AUSE and AURG for each technique
        (ause, aurg) = get_sparsification(pred_mean=pred_mean, y_true=y_true, 
                                          pred_std=pred_std)
        # compute sparsification score
        if aurg > 0:
            SP_score = aurg/(ause+aurg)
        else:
            SP_score = 0
        SP_score_lst.append(SP_score)   
        miscal_lst.append(
            miscalibration_area(pred_mean, pred_std, y_true, num_bins=100))
        sharp_lst.append(sharpness(pred_std))
        accuracy_lst.append(np.mean(np.abs(y_true - pred_mean)))   
    # Get indices for best performing UQ techniques based on sparcification
    positive_indices = [i for i, score in enumerate(SP_score_lst) if score > 0]
    if len(positive_indices) > args.num_spa:
        # Sort indices by corresponding scores (descending) and get the top 4
        top_sp_indices = sorted(
            positive_indices, key=lambda i: SP_score_lst[i],
            reverse=True)[:args.num_spa]
    else:
        top_sp_indices = positive_indices    
    final_indices = top_sp_indices.copy()
    # if the best performing model in terms of calibration and sharpness
    # is not in the list we add them
    best_cal_index = miscal_lst.index(min(miscal_lst))
    best_sharp_index = sharp_lst.index(min(sharp_lst))
    best_acc_index = accuracy_lst.index(min(accuracy_lst))
    if best_cal_index not in final_indices:    
        final_indices.append(best_cal_index) 
    if best_sharp_index not in final_indices:    
        final_indices.append(best_sharp_index)
    if best_acc_index not in final_indices:    
        final_indices.append(best_acc_index) 
    #print([techniques[i] for i in final_indices])
    #print(final_indices)
    # get all the combinations with at least two components
    subsets = []
    for r in range(2, len(final_indices) + 1):
        subsets.extend(itertools.combinations(final_indices, r))
    # Convert tuple of indices to lists for compuation
    experiments = [list(subset) for subset in subsets]
    #print(len(experiments))
    #print(experiments)
    best_combination = get_best_combination(args, experiments, techniques, df_lst)
    print(best_combination)

    # inference on validation set (for calibrated regression)
    mixture_inference(args, best_combination, techniques, df_lst)
    # inference on test set    
    test_df_names = [f for f in os.listdir(args.result_path)
                     if f.endswith('inference_result_.csv') 
                     and not f.startswith('GMM')]
    # Collect UQ technique names 
    techniques = [f.split('_'+args.split)[0] for f in test_df_names]
    # Collect a list of dataframes for each val_df_path
    test_df_lst = []
    for test_name in test_df_names:
        df = pd.read_csv(os.path.join(args.result_path, test_name))  
        test_df_lst.append(df)
    mixture_inference(args, best_combination, techniques, test_df_lst,
                      validation=False)

    


    
    
    
    
    

    
    """
    # create a combined dataframe based on selected techniques

    val_df_lst = get_val_dataframes(args, result_path)
    ground_truth = val_df_lst[0]['GroundTruth']
    predictions_std = pd.concat(
    [df[['Prediction', 'std']].rename(
        columns={'Prediction': f'Prediction_{i}', 'std': f'std_{i}'}) 
        for i, df in enumerate(val_df_lst)], axis=1)
    result_df = pd.concat([ground_truth, predictions_std], axis=1)
    if args.style == 'uniform':
        updated_df = static_GMM(result_df)
        
        
        
    # get means, and standard deviations for different models
    prediction_columns = [col for col in result_df.columns 
                          if col.startswith('Prediction_')]
    predictions_df = result_df[prediction_columns]
    means = predictions_df.to_numpy()
    std_columns = [col for col in result_df.columns if col.startswith('std_')]
    stds_df = result_df[std_columns]
    stds = stds_df.to_numpy()
    # get ground truth values for validation set
    targets = result_df['GroundTruth'].to_numpy()
    # Number of techniques and validation samples
    num_techniques = means.shape[1]
    num_samples = means.shape[0]
    """

    

def static_GMM(df, weights=None):
    prediction_cols = [col for col in df.columns if col.startswith('Prediction_')]
    std_cols = [col for col in df.columns if col.startswith('std_')]
    if weights is None: 
        num_techniques = len(prediction_cols)
        weights = [1 / num_techniques] * num_techniques
    else:
        if len(weights) != len(prediction_cols):
            raise ValueError('Number of weights must match number of techniques.')
    df['Prediction'] = sum(df[col] * weight 
                           for col, weight in zip(prediction_cols, weights))
    # Initialize an empty column for the combined uncertainty
    df['Total_Uncertainty'] = 0
    # Calculate the new column based on the given formula
    for i, (pred_col, std_col) in enumerate(zip(prediction_cols, std_cols)):
        weight = weights[i]
        df['Total_Uncertainty'] += (
            weight * (df[std_col]**2 + (df[pred_col] - df['Prediction'])**2))
    # Take the square root of the sum to complete the formula
    df['Total_Uncertainty'] = np.sqrt(df['Total_Uncertainty'])
    return df



    
"""

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
"""
        
if __name__ == '__main__':
    main()