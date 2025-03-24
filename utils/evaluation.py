"""
To prepare this script we used uncertainty tool-box which can be find in:
    https://uncertainty-toolbox.github.io/about/
For QICE metric we used the source code from:
    https://github.com/XzwHan/CARD
"""

import os
import pandas as pd
import numpy as np
import uncertainty_toolbox as uct
import pickle
from utils.utils import get_mean_std_truth

 
# A method to conduct evaluation on prediction dataframe
def uq_eval(csv_file, prefix, report=False, verbose=False,
            calibration_mode=False, calibration_type=None, recal_model=None,
            mixture_mode=False, mixture_info=None):
    df = pd.read_csv(csv_file)    
    # get ground truth, posterior mean and standard deviation (uncertainty)
    pred_mean, pred_std, y_true = get_mean_std_truth(df=df, uq_method=prefix) 
    # Get all uncertainty quantification metrics
    pred_std = np.maximum(pred_std, 1e-6) # for numerical stability
    uq_metrics = uct.metrics.get_all_metrics(pred_mean, pred_std, y_true)    
    if report:        
        # set the path to report UQ analysis in a .txt file
        base_name, _ = os.path.splitext(os.path.basename(csv_file))
        result_path = os.path.dirname(csv_file)        
        report_name = base_name + 'uq_metrics' + '.txt'
        uq_dict_name = base_name + 'uq_metrics' + '.pkl'
        report_path = os.path.join(result_path, report_name)
        uq_dict_path = os.path.join(result_path, uq_dict_name)            
        # write uq_metrics dictionary into a .txt file
        with open(report_path, 'w') as file:
            # Iterate over the dictionary items and write them to the file
            for key, value in uq_metrics.items():
                file.write(f"{key}: {value}\n") 
        with open(uq_dict_path, 'wb') as file:
            pickle.dump(uq_metrics, file)        
    return uq_metrics 