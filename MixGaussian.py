import argparse
import os
import yaml
import pandas as pd
import torch
import itertools
import numpy as np
from uncertainty_toolbox.metrics_scoring_rule import nll_gaussian
from uncertainty_toolbox.metrics_calibration import (miscalibration_area,
                                                     sharpness)
from utils.utils import get_mean_std_truth
from utils.evaluation import get_sparsification, uq_eval
from utils.calibration import calibrated_regression
from loss.GMMLoss import GMM_Loss
    

def mixture_inference(args, best_combination, techniques, test_df_lst,
                      validation=True):
    # TODO: implement GMM for cross-fold validation
    csv_val_name = 'GMM_{}_{}_seed_{}_inference_result_validation_.csv'.format(
        args.style, args.split, args.seed)
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
        if args.style == 'dynamic':
            gmm_method = 'GMMD'
        else:
            gmm_method = 'GMM'
        mixture_df.to_csv(csv_test_path, index=False)
        _ = uq_eval(csv_test_path, gmm_method, report=True, verbose=True,
                    mixture_mode=True, mixture_info=best_combination)  
        recalibration_path = os.path.join(args.result_path, 'recalibration')
        calibrated_result, recal_model = calibrated_regression(
            calibration_df_path=csv_val_path, test_df_path=csv_test_path, 
            uq_method=gmm_method, confidence_level=0.95,
            recalibration_path=recalibration_path, report=False)  
        uq_eval(calibrated_result, gmm_method, report=True, verbose=True,
                calibration_mode=True, calibration_type=args.calibration_type,
                recal_model=recal_model, mixture_mode=True, 
                mixture_info=best_combination)


def get_best_combination(args, experiments, techniques, df_lst):    
    # compute NLL for all combinations of techniques
    exp_dict = {}
    for experiment in experiments:
        if args.style == 'uniform' or args.style == 'default':
            NLL_exp, weight_dict = uniform_mixture(
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
    pred_mean_list = [t.to(torch.float64).requires_grad_() for t in pred_mean_list]
    pred_std_list = [t.to(torch.float64).requires_grad_() for t in pred_std_list]
    pred_mean_tensor = torch.stack(pred_mean_list, dim=1)    
    pred_std_tensor = torch.stack(pred_std_list, dim=1)
    y_true_tensor=torch.tensor(y_true, requires_grad=True)
    #print(pred_mean_tensor.size())
    num_tech = len(experiment)
    weights = torch.randn((num_tech,), dtype=torch.double, requires_grad=True)
    """
    # initial uniform weights for starting point
    uniform_weight = 1 / num_tech
    weights = torch.full((num_tech,), uniform_weight, dtype=torch.double, 
                         requires_grad=True)
    """
    #print(weights.size())
    gmm_loss_fn = GMM_Loss()
    hyper_optimizer = torch.optim.Adam([weights], lr=args.gmm_lr)
    for epoch in range(args.gmm_epochs):
        hyper_optimizer.zero_grad()
        #weights.data /= weights.data.sum()
        # get the prediction means for the mixture of Gaussians
        mixture_mean = torch.matmul(pred_mean_tensor, weights)/(weights.sum())
        mixture_mean_expanded = mixture_mean.unsqueeze(1)  # (num_samples, 1)
        mean_diff_squared = (pred_mean_tensor - mixture_mean_expanded) ** 2  
        std_squared = pred_std_tensor ** 2 
        mixture_var = torch.matmul(std_squared + mean_diff_squared, weights)/(weights.sum())
        mixture_std = torch.sqrt(mixture_var)
        #print(mixture_mean.size())
        #print(mixture_std.size())
        loss = gmm_loss_fn(mixture_mean, mixture_std, y_true_tensor, args.alpha)
        loss.backward()
        hyper_optimizer.step()
    
    with torch.no_grad():   
        #weights = weights / weights.sum()
        weights = torch.softmax(weights, dim=0)
        mixture_mean = torch.matmul(pred_mean_tensor, weights)
        mixture_mean_expanded = mixture_mean.unsqueeze(1)  # (num_samples, 1)
        mean_diff_squared = (pred_mean_tensor - mixture_mean_expanded) ** 2  
        std_squared = pred_std_tensor ** 2 
        mixture_var = torch.matmul(std_squared + mean_diff_squared, weights)
        mixture_std = torch.sqrt(mixture_var)
        final_loss = gmm_loss_fn(mixture_mean, mixture_std, y_true_tensor,
                                 args.alpha)
    
    weights = weights.cpu().detach().numpy()
    weight_dict = {}
    for i, index in enumerate(experiment):
        technique = techniques[index]
        weight_dict[technique] = weights[i]
    
    
    return final_loss, weight_dict
   
def uniform_mixture(experiment, techniques, df_lst):
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
    parser.add_argument('--dataset', help='dataset used fitting GMM model.')
    parser.add_argument('--cfg', help='configuration for fitting GMM model.')
    parser.add_argument('--selection', default='remove_worst',
                        help='remove worst models or keep best models')
    parser.add_argument('--style', default='default',
                        help='weights for mixture components: uniform/dynamic/default')
    args = parser.parse_args()   
    root_path = os.getcwd()
    # read the relevant cfg file
    cfg_file = os.path.join(root_path, 'cfg', args.cfg)
    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)
    # set arguments
    args.model = cfg.get('model')
    args.split = cfg.get('split')
    args.seed = cfg.get('seed')
    args.calibration_type = cfg.get('calibration_type')
    args.gmm_lr = cfg.get('gmm_lr')
    args.gmm_epochs = cfg.get('gmm_epochs')
    args.alpha = cfg.get('accuracy_emphasize')
    args.search_num = cfg.get('search_num')
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
    if args.style != 'default':   
        # Get sparsification, miscalibration area, accuracy, sharpness scores.
        ause_lst, aurg_lst, miscal_lst, accuracy_lst = [], [], [], []
        for df, technique in zip(df_lst, techniques):
            # get mean, standard deviation for predicted values, plus ground truths
            pred_mean, pred_std, y_true = get_mean_std_truth(
                df=df, uq_method=technique)   
            # get AUSE and AURG for each technique
            (ause, aurg) = get_sparsification(pred_mean=pred_mean, y_true=y_true, 
                                          pred_std=pred_std)
            ause_lst.append(ause) 
            aurg_lst.append(aurg)
            miscal_lst.append(
                miscalibration_area(pred_mean, pred_std, y_true, num_bins=100))
            accuracy_lst.append(np.mean(np.abs(y_true - pred_mean)))    
        ause_indices = sorted(range(len(ause_lst)), key=lambda i: ause_lst[i])
        miscal_indices = sorted(range(len(miscal_lst)), key=lambda i: miscal_lst[i])
        accuracy_indices = sorted(range(len(accuracy_lst)), key=lambda i: accuracy_lst[i])
        performance_lists = [ause_indices, miscal_indices, accuracy_indices]  
        # remove negative AURG (sparsification worse than random guess)
        no_use_indices = [index for index, value in enumerate(aurg_lst) if value < 0]
        if args.selection == 'remove_worst':
            all_indices = list(range(len(techniques)))
            filtered_indices = [item for item in all_indices if item not in no_use_indices]       
            while len(filtered_indices) > args.search_num:
                for lst in performance_lists:
                    worst_index = lst[-1]
                    if worst_index in filtered_indices:
                        filtered_indices.remove(worst_index)
                    ause_indices.remove(worst_index)
                    miscal_indices.remove(worst_index)
                    accuracy_indices.remove(worst_index)
                    if len(filtered_indices) == args.search_num:
                        break
            final_indices = filtered_indices
        else:
            final_indices = []
            while len(final_indices) < args.search_num:
                for lst in performance_lists:
                    best_index = lst[0]
                    if ((best_index not in final_indices) and
                        (best_index not in no_use_indices)):
                        final_indices.append(best_index)
                    ause_indices.remove(best_index)
                    miscal_indices.remove(best_index)
                    accuracy_indices.remove(best_index)
                    if len(final_indices) == args.search_num:
                        break                
        #removed_techniques = [techniques[i] for i in no_use_indices]
        #selected_techniques = [techniques[i] for i in final_indices]    
        #print(selected_techniques)
        #print(removed_techniques)
    else:
        light_weight_models = ['deterministic', 'LA', 'RF', 'mve']
        final_indices = [i for i, x in enumerate(techniques) if x in light_weight_models]
    
    # get all the combinations with at least two components
    subsets = []
    for r in range(1, len(final_indices) + 1):
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
    # Collect a list of dataframes for each test_df_names
    test_df_lst = []
    for test_name in test_df_names:
        df = pd.read_csv(os.path.join(args.result_path, test_name))  
        test_df_lst.append(df)
    mixture_inference(args, best_combination, techniques, test_df_lst,
                      validation=False)
        
if __name__ == '__main__':
    main()