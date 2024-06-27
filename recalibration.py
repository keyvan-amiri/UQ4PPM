import argparse
import os
import pandas as pd
import uncertainty_toolbox as uct
from utils.eval_cal_utils import get_csv_files


def main():   
    
    # Parse arguments 
    parser = argparse.ArgumentParser(
        description='Recalibration for specified models')
    parser.add_argument('--dataset', default='HelpDesk',
                        help='Datasets that is used for recalibration')
    parser.add_argument('--model', default='dalstm',
                        help='Type of the predictive model')
    parser.add_argument('--criterion', default='miscal',
                        help='miscalibration criterion to be used')
    args = parser.parse_args()
    
    # Ensure criterion is one of the allowed values
    allowed_criteria = {'ma_cal', 'rms_cal', 'miscal'}
    assert args.criterion in allowed_criteria
    
    # define model directory for recalibration
    root_path = os.getcwd()
    model_path = os.path.join(root_path, 'results', args.dataset, args.model)
    
    # find all csv files created by predictive model
    csv_files, uncertainty_methods = get_csv_files(model_path)
    
    # iterate over results, and recalibrate predictions
    for i in range (len(csv_files)):
        # Extract the file name and extension
        base_name, ext = os.path.splitext(os.path.basename(csv_files[i]))
        # read csv file, and find the uncertainty quantification method used
        df = pd.read_csv(csv_files[i])
        prefix = uncertainty_methods[i]
        # get ground truth as well as mean and std for predictions
        pred_mean = df['Prediction'].values 
        y_true = df['GroundTruth'].values
        if (prefix=='DA_A' or prefix=='CDA_A'):
            pred_std = df['Total_Uncertainty'].values 
        elif (prefix=='CARD' or prefix=='mve'):
            pred_std = df['Aleatoric_Uncertainty'].values
        elif (prefix=='DA' or prefix=='CDA'):
            pred_std = df['Epistemic_Uncertainty'].values
        else:
            raise NotImplementedError(
                'Uncertainty quantification {} not understood.'.format(prefix))
            
        recalibrator = uct.recalibration.get_std_recalibrator(
            pred_mean, pred_std, y_true, criterion=args.criterion)
        # Apply the recalibrator to get recalibrated standard deviations
        recalibrated_std = recalibrator(pred_std)

        
        if (prefix=='DA_A' or prefix=='CDA_A'):
            pred_std = df['Total_Uncertainty'] = recalibrated_std
        elif (prefix=='CARD' or prefix=='mve'):
            pred_std = df['Aleatoric_Uncertainty'] = recalibrated_std
        elif (prefix=='DA' or prefix=='CDA'):
            pred_std = df['Epistemic_Uncertainty'] = recalibrated_std  
        
        new_csv_path = os.path.join(model_path, base_name + 'recalibrated_.csv')
        df.to_csv(new_csv_path, index=False)
        
if __name__ == '__main__':
    main()        
        
    