import argparse
import os
import yaml
import pickle
import pandas as pd
from datetime import datetime
import torch
from torch.utils.data import TensorDataset, DataLoader
import uncertainty_toolbox as uct
from utils.eval_cal_utils import get_csv_files
from utils.eval_cal_utils import (extract_info_from_cfg, replace_suffix,
                                  get_validation_data_and_model_size,
                                  get_model_and_loss, get_uq_method,
                                  add_suffix_to_csv)


def calibration_on_validation(args=None, model=None, checkpoint_path=None,
                              calibration_loader=None, heteroscedastic=None,
                              num_mc_samples=None, normalization=False, 
                              y_scaler=None, device=None, result_path=None):
        
    print('Now: start recalibration:')
    start=datetime.now()
    
    # setstructure of instance-level results
    if (args.UQ == 'DA' or args.UQ == 'CDA'):
        res_dict = {'GroundTruth': [], 'Prediction': [],
                    'Epistemic_Uncertainty': []}
    elif (args.UQ == 'DA_A' or args.UQ == 'CDA_A'):
        res_dict = {'GroundTruth': [], 'Prediction': [], 
                    'Epistemic_Uncertainty': [], 'Aleatoric_Uncertainty': [],
                    'Total_Uncertainty': []} 
    elif args.UQ == 'mve':
        res_dict = {'GroundTruth': [], 'Prediction': [],
                    'Aleatoric_Uncertainty': []}
    
    # load the checkpoint  
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # get instance-level results on validation set
    model.eval()
    with torch.no_grad():
        for index, calibration_batch in enumerate(calibration_loader):
            # get batch data
            inputs = calibration_batch[0].to(device)
            _y_truth = calibration_batch[1].to(device)  
            # get model outputs, and uncertainties
            if (args.UQ == 'DA' or args.UQ == 'CDA' or args.UQ == 'DA_A' or
                args.UQ == 'CDA_A'):
                means_list, logvar_list =[], []
                # conduct Monte Carlo sampling
                for i in range (num_mc_samples): 
                    mean, log_var,_ = model(inputs, stop_dropout=False)
                    means_list.append(mean)
                    logvar_list.append(log_var)
                # Aggregate the results for all samples
                # Compute point estimation and uncertainty
                stacked_means = torch.stack(means_list, dim=0)
                # predited value is the average for all samples
                _y_pred = torch.mean(stacked_means, dim=0)
                # epistemic uncertainty obtained from std for all samples
                epistemic_std = torch.std(stacked_means, dim=0).to(device)
                # normalize epistemic uncertainty if necessary
                if normalization:
                    epistemic_std = y_scaler * epistemic_std
                # now obtain aleatoric uncertainty
                if heteroscedastic:
                    stacked_log_var = torch.stack(logvar_list, dim=0)
                    stacked_var = torch.exp(stacked_log_var)
                    mean_var = torch.mean(stacked_var, dim=0)
                    aleatoric_std = torch.sqrt(mean_var).to(device)
                    # normalize aleatoric uncertainty if necessary
                    if normalization:
                        aleatoric_std = y_scaler * aleatoric_std
                    total_std = epistemic_std + aleatoric_std
            elif args.UQ == 'mve':
                _y_pred, log_var = model(inputs)
                aleatoric_std = torch.sqrt(torch.exp(log_var))
                # normalize aleatoric uncertainty if necessary
                if normalization:
                    aleatoric_std = y_scaler * aleatoric_std            
            # convert tragets, outputs in case of normalization
            if normalization:
                _y_truth = y_scaler * _y_truth
                _y_pred = y_scaler * _y_pred
            # Detach predictions and ground truths (np arrays)
            _y_truth = _y_truth.detach().cpu().numpy()
            _y_pred = _y_pred.detach().cpu().numpy()
            # collect inference result in all_result dict.
            res_dict['GroundTruth'].extend(_y_truth.tolist())
            res_dict['Prediction'].extend(_y_pred.tolist())
            if (args.UQ == 'DA' or args.UQ == 'CDA' or args.UQ == 'DA_A' or 
                args.UQ == 'CDA_A'):
                epistemic_std = epistemic_std.detach().cpu().numpy()
                res_dict['Epistemic_Uncertainty'].extend(epistemic_std.tolist()) 
                if heteroscedastic:
                    aleatoric_std = aleatoric_std.detach().cpu().numpy()
                    total_std = total_std.detach().cpu().numpy()
                    res_dict['Aleatoric_Uncertainty'].extend(aleatoric_std.tolist())
                    res_dict['Total_Uncertainty'].extend(total_std.tolist()) 
            elif args.UQ == 'mve':
                aleatoric_std = aleatoric_std.detach().cpu().numpy()
                res_dict['Aleatoric_Uncertainty'].extend(aleatoric_std.tolist())
    calibration_df = pd.DataFrame(res_dict)
    pred_mean, pred_std, y_true = get_mean_std_truth(
        calibration_df=calibration_df, uq_method=args.UQ)
    if args.method == 'std_ratio':
        recalibrator = uct.recalibration.get_std_recalibrator(
            pred_mean, pred_std, y_true, criterion=args.criterion)
        # Apply the recalibrator to get recalibrated standard deviations
        recalibrated_std = recalibrator(pred_std)
        calibration_df['calibrated_std'] = recalibrated_std 
    else:
        # TODO: implement any other recalibration approach
        pass
    calibration_time = (datetime.now()-start).total_seconds()
    new_csv_name = add_suffix_to_csv(args.csv_file,
                                     added_suffix='validation_calibrated_')
    new_csv_path = os.path.join(result_path, new_csv_name)
    calibration_df.to_csv(new_csv_path, index=False)


def get_mean_std_truth (calibration_df=None, uq_method=None):
    pred_mean = calibration_df['Prediction'].values 
    y_true = calibration_df['GroundTruth'].values
    if (uq_method=='DA_A' or uq_method=='CDA_A'):
        pred_std = calibration_df['Total_Uncertainty'].values 
    elif (uq_method=='CARD' or uq_method=='mve'):
        pred_std = calibration_df['Aleatoric_Uncertainty'].values
    elif (uq_method=='DA' or uq_method=='CDA'):
        pred_std = calibration_df['Epistemic_Uncertainty'].values
    else:
        raise NotImplementedError(
            'Uncertainty quantification {} not understood.'.format(uq_method))
    return pred_mean, pred_std, y_true

def main():   
    
    # Parse arguments 
    parser = argparse.ArgumentParser(
        description='Recalibration for specified models')
    parser.add_argument('--device', type=int, default=0, help='GPU device id')
    parser.add_argument('--csv_file', help='results to be recalibrated')
    parser.add_argument('--cfg_file', help='configuration used for training')
    parser.add_argument('--method', default='std_ratio',
                        help='recalibration criterion to be used')
    parser.add_argument('--criterion', default='miscal',
                        help='recalibration criterion to be used')
    args = parser.parse_args()
    
    # Ensure criterion is one of the allowed values
    allowed_criteria = {'ma_cal', 'rms_cal', 'miscal'}
    assert args.criterion in allowed_criteria  
    # Define the device
    device_name = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    # Define path for cfg and csv files
    root_path = os.getcwd()
    cfg_path = os.path.join(root_path, 'cfg', args.cfg_file)
    
    # Get model, dataset, and uq_method based on configuration file name
    args.model, args.dataset, args.UQ = extract_info_from_cfg(args.cfg_file)    
    result_path = os.path.join(root_path, 'results', args.dataset, args.model)
    csv_path = os.path.join(result_path, args.csv_file)
    if args.UQ != 'CARD':
        args.UQ = get_uq_method(args.csv_file)
        # load cfg file used for training
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f) 
        # define path for report .txt to add recalibration time
        report_name = replace_suffix(args.csv_file, 'inference_result_.csv',
                                     'report_.txt')
        report_path = os.path.join(root_path, 'results', args.dataset,
                                   args.model, report_name)
        # define name of the check point (best model)
        checkpoint_name = replace_suffix(args.csv_file, 'inference_result_.csv',
                                         'best_model.pt')
        checkpoint_path = os.path.join(result_path, checkpoint_name)
        
    else:
        # TODO: access the relevant configuration file
        # TODO: access the model checkpoint
        # TODO: handle report .txt for time!
        pass
           
    # TODO: adjust when it is done in the past
    calibration_history = False
    
    if not calibration_history:
        # Separate execution path for CARD approach   
        if args.UQ == 'CARD':
            pass
        else:
            # Get calibration loader, model dimensions, normalization ratios
            (calibration_loader, input_size, max_len, max_train_val,
             mean_train_val, median_train_val
             ) = get_validation_data_and_model_size(args=args, cfg=cfg,
                                                    root_path=root_path)
            # define model and loss function
            (model, criterion, heteroscedastic, num_mcmc, normalization
             ) = get_model_and_loss(args=args, cfg=cfg, input_size=input_size,
                                    max_len=max_len, device=device)
            # execute calibration on validation set
            calibration_on_validation(
                args=args, model=model, checkpoint_path=checkpoint_path,
                calibration_loader=calibration_loader,
                heteroscedastic=heteroscedastic, num_mc_samples=num_mcmc,
                normalization=normalization, y_scaler=mean_train_val,
                device=device, result_path=result_path)

            
        
            
 
                


    
    """
   
    # get a dataframe containint test results to be recalibrated    
    test_df_path = os.path.join(root_path, 'results', args.csv_file)
    test_df = pd.read_csv(test_df_path)
    
        if (prefix=='DA_A' or prefix=='CDA_A'):
            pred_std = df['Total_Uncertainty'] = recalibrated_std
        elif (prefix=='CARD' or prefix=='mve'):
            pred_std = df['Aleatoric_Uncertainty'] = recalibrated_std
        elif (prefix=='DA' or prefix=='CDA'):
            pred_std = df['Epistemic_Uncertainty'] = recalibrated_std  
        
        new_csv_path = os.path.join(model_path, base_name + 'recalibrated_.csv')
        df.to_csv(new_csv_path, index=False)
        """
        
if __name__ == '__main__':
    main()        
        
    