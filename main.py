"""
To prepare this script we used the following source codes:
    https://github.com/XzwHan/CARD
We adjusted the source codes to efficiently integrate them into our framework.
"""

import argparse
import os
import yaml
import torch
from utils.DALSTM_train_eval import DALSTM_train_evaluate


def main():    
    # Parse arguments 
    parser = argparse.ArgumentParser(
        description='Probabilistic remaining time prediction')
    parser.add_argument('--dataset', default='HelpDesk',
                        help='Datasets used by model')    
    parser.add_argument('--model', default='pgtnet',
                        help='Type of the predictive model')
    parser.add_argument('--UQ', default='deterministic',
                        help='Uncertainty quantification method to be used')
    parser.add_argument('--split', default='holdout',
                        help='The data split that is used')
    parser.add_argument('--split_num', type=int, default=5,
                        help='Number of splits that is used')
    parser.add_argument('--seed', nargs='+', help='Random seed to use',
                        required=True) #can handle multiple seeds
    parser.add_argument('--device', type=int, default=0, help='GPU device id') 
    args = parser.parse_args()
    # define device name
    device_name = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    # read the relevant configuration file, and append
    root_path = os.getcwd()
    cfg_file = os.path.join(root_path, 'cfg',
                            args.model+'_'+args.dataset+'.yaml')
    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)     
    # append the configuration file
    cfg['net'] = args.model
    cfg['uq_method'] = args.UQ
    cfg['dataset'] = args.dataset
    cfg['split'] = args.split
    cfg['n_splits'] = args.split_num
    cfg['device'] = device_name
    cfg['seed'] = args.seed
    if args.model == 'dalstm':
        DALSTM_train_evaluate(cfg=cfg)
    elif args.model == 'cnn':
        pass
    elif args.model == 'pt':
        pass
    elif args.model == 'pgtnet':
        pass
    else:
        print(f'The backebone model {args.model} is not supported') 
    
if __name__ == '__main__':
    main()