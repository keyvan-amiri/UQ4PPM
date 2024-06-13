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
from utils.utils import parse_temp_config


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
    parser.add_argument('--thread', type=int, default=4,
                        help='number of threads')
    
    #The following arguments are only applicable for CARD model
    #TODO: check MODEL_VERSION_DIR in args to see how to save the result in a
    # more meaningful path! do wee need: ${RUN_NAME}_${SERVER_NAME}? or even: ${N_STEPS}steps
    parser.add_argument('--exp', type=str, default='exp',
                        help='Path for saving running related data.')
    parser.add_argument('--doc', type=str, 
                        help='Name of the log folder- for documentation purpose')
    parser.add_argument('--config', type=str, help='Path to the config file')
    parser.add_argument('--comment', type=str, default='',
                        help='A string for experiment comment')
    # can be used for cross-validation set up.
    parser.add_argument('--run_all', action='store_true', 
                        help='Whether to run all train test splits')
    parser.add_argument('--rmse_timestep', type=int, default=0,
                        help='selected timestep to report metric y RMSE')
    parser.add_argument('--qice_timestep', type=int, default=0,
                        help='selected timestep to report metric y QICE')
    parser.add_argument('--picp_timestep', type=int, default=0,
                        help='selected timestep to report metric y PICP')
    parser.add_argument('--nll_timestep', type=int, default=0,
                        help='selected timestep to report metric y NLL')
    #TODO: important! check how much is important the following arg.
    parser.add_argument('--sample_type', type=str, default='generalized',
                        help='sampling approach (generalized or ddpm_noisy)')
    # loss option for CARD model
    parser.add_argument('--loss', type=str, default='ddpm',
                        help='Which loss to use')
    #TODO: run experiments without knowledge from pre-trained deterministic model.
    parser.add_argument('--noise_prior', action='store_true', 
                        help='Whether to apply a noise prior distribution at timestep T')
    # if the following is not specified explicitly: f_phi is used in concatanation
    parser.add_argument('--no_cat_f_phi', action='store_true',
                        help='Whether to not concatenate f_phi as part of eps_theta input')
    parser.add_argument('--fid', action='store_true')
    parser.add_argument('--interpolation', action='store_true')
    #TODO: check how much is important the following arg.
    parser.add_argument('--skip_type', type=str, default='uniform',
                        help='skip according to (uniform or quadratic)')
    parser.add_argument('--timesteps', type=int, default=None,
                        help='number of steps involved')
    #TODO: important! check how much is important the following arg.
    parser.add_argument('--eta', type=float, default=0.0, 
                        help='eta used to control the variances of sigma')
    #TODO: check what does the follwing control?
    parser.add_argument('--sequence', action='store_true')
    #TODO: check the functionality of following arg
    parser.add_argument('--nll_global_var', action='store_true',
                        help='Apply global variance for NLL computation')
    #TODO: important! check how much is important the following arg.
    parser.add_argument('--nll_test_var', action='store_true',
                        help='Apply sample variance of the test set for NLL computation')
    # Conditional transport options
    #TODO: important! check how much is important the following 2-3 args.
    parser.add_argument('--use_d', action='store_true',
                        help="Whether to take an adversarially trained feature encoder")
    parser.add_argument('--full_joint', action='store_true',
                        help='Whether to take fully joint matching')
    parser.add_argument('--num_sample', type=int, default=1,
                        help='number of samples used in forward and reverse') 
    args = parser.parse_args()
    
    if args.UQ == 'CARD':
        # set scientific number prints to False
        torch.set_printoptions(sci_mode=False)
        temp_config = parse_temp_config(args.doc)
        if args.run_all:
            # TODO: check whether mae is better or keep working with rmse
            if temp_config.data.dataset == "ppm":
                y_mae_all_splits_all_steps_list, \
                    y_qice_all_splits_all_steps_list, \
                        y_picp_all_splits_all_steps_list, \
                            y_nll_all_splits_all_steps_list = [], [], [], []
            else:
                y_rmse_all_splits_all_steps_list, \
                    y_qice_all_splits_all_steps_list, \
                        y_picp_all_splits_all_steps_list,\
                            y_nll_all_splits_all_steps_list = [], [], [], []
            original_doc = args.doc
            original_config = args.config
            
    else:    
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