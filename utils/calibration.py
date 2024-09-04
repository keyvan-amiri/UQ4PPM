"""
To prepare this script we used uncertainty tool-box which can be find in:
    https://uncertainty-toolbox.github.io/about/
"""
import os
from datetime import datetime
import pandas as pd
import uncertainty_toolbox as uct
from utils.utils import get_mean_std_truth, add_suffix_to_csv


def calibrated_regression(calibration_df_path=None, test_df_path=None,
                          uq_method=None,
                          confidence_level=0.95, report_path=None,
                          recalibration_path=None):
    """
    recalibration is done based on two approaches:
    1) Isotonic regression remaps quantiles of the original distribution. The
    recalibrated distribution is unikely to be Gaussian. But, it still can
    provide confidence intervals (confidence_level).
    2) Scaling factor for the standard deviation is computed based on three
    different metrics: constrains the recalibrated distribution to be Gaussian.
    """
    print('Now: start recalibration:')
    calibration_df = pd.read_csv(calibration_df_path) 
    test_df = pd.read_csv(test_df_path)
    start=datetime.now()
    # get prediction means, standard deviations, and ground truths for val set
    (pred_mean, pred_std, y_true
     ) = get_mean_std_truth(df=calibration_df, uq_method=uq_method)
    
    # Gaussian calibration on validation set
    # Compute scaling factor for the standard deviation
    miscal_std_scaling = uct.recalibration.optimize_recalibration_ratio(
      pred_mean, pred_std, y_true, criterion="miscal")
    rms_cal_std_scaling = uct.recalibration.optimize_recalibration_ratio(
      pred_mean, pred_std, y_true, criterion="rms_cal")
    ma_cal_std_scaling = uct.recalibration.optimize_recalibration_ratio(
      pred_mean, pred_std, y_true, criterion="ma_cal")
    # get prediction means, standard deviations, and ground truths for test set
    (test_pred_mean, test_pred_std, test_y_true
     ) = get_mean_std_truth(df=test_df, uq_method=uq_method)
    # Apply the scaling factors to get recalibrated standard deviations
    miscal_test_pred_std = miscal_std_scaling * test_pred_std
    rms_cal_test_pred_std = rms_cal_std_scaling * test_pred_std
    ma_cal_test_pred_std = ma_cal_std_scaling * test_pred_std
    test_df['calibrated_std_miscal'] = miscal_test_pred_std 
    test_df['calibrated_std_rms_cal'] = rms_cal_test_pred_std
    test_df['calibrated_std_ma_cal'] = ma_cal_test_pred_std
    
    # Isotonic regression calibration on validation set
    # Get the expected proportions and observed proportions on calibration set
    exp_props, obs_props = uct.metrics_calibration.get_proportion_lists_vectorized(
        pred_mean, pred_std, y_true)
    # Train a recalibration model.
    recal_model = uct.recalibration.iso_recal(exp_props, obs_props) 
    # Get prediction intervals
    recal_bounds = uct.metrics_calibration.get_prediction_interval(
        test_pred_mean, test_pred_std, confidence_level, recal_model)
    test_df['confidence_lower'] = recal_bounds.lower
    test_df['confidence_upper'] = recal_bounds.upper
    
    recal_name = add_suffix_to_csv(test_df_path, added_suffix='recalibrated_')
    recalibrated_test_path = os.path.join(recalibration_path, recal_name)
    test_df.to_csv(recalibrated_test_path, index=False)
    calibration_time = (datetime.now()-start).total_seconds()
    with open(report_path, 'a') as file:
        file.write('Calibration took  {} seconds. \n'.format(calibration_time))        
    return (recalibrated_test_path, recal_model)