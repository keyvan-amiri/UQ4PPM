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
        if args.style == 'uniform':
            gmm_method = 'GMM'
        else:
            gmm_method = 'GMMD'
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
    # Collect a list of dataframes for each test_df_names
    test_df_lst = []
    for test_name in test_df_names:
        df = pd.read_csv(os.path.join(args.result_path, test_name))  
        test_df_lst.append(df)
    mixture_inference(args, best_combination, techniques, test_df_lst,
                      validation=False)

        
if __name__ == '__main__':
    main()