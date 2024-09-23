import os
import re
import yaml
import pandas as pd
import uncertainty_toolbox as uct
from utils.utils import get_mean_std_truth
from utils.evaluation import get_sparsification


def main():
    class IgnoreConstructor(yaml.SafeLoader):
        pass
    def ignore_python_objects(loader, tag_suffix, node):
        return loader.construct_mapping(node, deep=True)
    
    root_path = os.getcwd()    
    dir_path = os.path.join(root_path, 'results', 'HelpDesk', 'dalstm')
    root_card_path = os.path.join(dir_path, 'card')
    file_list = []
    for filename in os.listdir(dir_path):
        if (filename.startswith('CARD_holdout') and 
            filename.endswith('inference_result_validation_.csv')):
            full_path = os.path.join(dir_path, filename)
            if os.path.isfile(full_path):
                file_list.append(full_path)
   
    hpo_results = {'exp_id': [], 'cat_x': [], 'window_cat_x': [],
                   'feature_dim': [], 'beta_start': [], 'joint_train': [],
                   'n_epochs': [], }
    additional_keys = ['mae', 'rmse', 'nll', 'crps', 'sharp', 'ause', 'aurg',
                       'miscal_area', 'check', 'interval']
    for key in additional_keys:
        hpo_results[key] = [] 
    
    pattern = r"Exp_(\d+)"
    for csv_path in file_list:
        match = re.search(pattern, csv_path)
        if match:
            exp_id = match.group(1)
            exp_str = 'Exp_' + exp_id
            cfg_path = os.path.join(root_card_path, exp_str, 'logs',
                                   'dalstm_HelpDesk_card', 'split_0', 
                                   'config.yml')
            IgnoreConstructor.add_multi_constructor('tag:yaml.org,2002:python/object', 
                                                    ignore_python_objects)
            with open(cfg_path, 'r') as file:
                config = yaml.load(file, Loader=IgnoreConstructor)
            cat_x_value = config['model']['cat_x']
            window_cat_x_value = config['model']['window_cat_x']
            feature_dim_value = config['model']['feature_dim']
            beta_start_value = config['diffusion']['beta_start']
            joint_train_value = config['diffusion']['nonlinear_guidance']['joint_train']
            n_epochs_value = config['training']['n_epochs']
            df = pd.read_csv(csv_path) 
            pred_mean, pred_std, y_true = get_mean_std_truth(df=df,
                                                             uq_method='CARD')
            uq_metrics = uct.metrics.get_all_metrics(pred_mean, pred_std, y_true)
            (uq_metrics['Area Under Sparsification Error curve (AUSE)'],
             uq_metrics['Area Under Random Gain curve (AURG)']
             ) = get_sparsification(
                 pred_mean=pred_mean, y_true=y_true, pred_std=pred_std)
            
            hpo_results['exp_id'].append(exp_id)
            hpo_results['cat_x'].append(cat_x_value)
            hpo_results['window_cat_x'].append(window_cat_x_value)
            hpo_results['feature_dim'].append(feature_dim_value)
            hpo_results['beta_start'].append(beta_start_value)
            hpo_results['joint_train'].append(joint_train_value)
            hpo_results['n_epochs'].append(n_epochs_value)            
            hpo_results['mae'].append(uq_metrics.get('accuracy').get('mae'))
            hpo_results['rmse'].append(uq_metrics.get('accuracy').get('rmse'))
            hpo_results['nll'].append(uq_metrics.get('scoring_rule').get('nll'))
            hpo_results['crps'].append(uq_metrics.get('scoring_rule').get('crps'))
            hpo_results['sharp'].append(uq_metrics.get('sharpness').get('sharp'))   
            hpo_results['ause'].append(uq_metrics.get('Area Under Sparsification Error curve (AUSE)'))
            hpo_results['aurg'].append(uq_metrics.get('Area Under Random Gain curve (AURG)'))           
            hpo_results['miscal_area'].append(uq_metrics.get('avg_calibration').get('miscal_area'))                
            hpo_results['check'].append(uq_metrics.get('scoring_rule').get('check'))
            hpo_results['interval'].append(uq_metrics.get('scoring_rule').get('interval'))
    
    hpo_df = pd.DataFrame(hpo_results)
    hpo_name = 'CARD_holdout_seed_42_hpo_result_.csv'
    csv_filename = os.path.join(dir_path, hpo_name) 
    hpo_df.to_csv(csv_filename, index=False)

if __name__ == '__main__':
    main()