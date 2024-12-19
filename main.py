"""
To prepare this script we used the following source codes:
    https://github.com/XzwHan/CARD
    https://github.com/rampasek/GraphGPS
We adjusted the source codes to efficiently integrate them into our framework.
"""

import argparse
import os
import yaml
import sys
import logging
import pickle
import traceback
import time
import torch
from models.card_regression import Diffusion
from utils.DALSTM_train_eval import DALSTM_train_evaluate
from utils.diffusion_utils import (parse_temp_config, parse_config,
                                   adjust_arguments, set_experiments,
                                   get_stat, card_hpo, card_calibration) 


# The main method for exceution of CARD models.
def main_card(arg_set=None, key=None):
    config, logger = parse_config(args=arg_set, key=key)

    logging.info('Writing log file to {}'.format(arg_set.log_path))
    logging.info('Exp instance id = {}'.format(os.getpid()))
    logging.info('Exp comment = {}'.format(arg_set.comment))
    try:
        runner = Diffusion(arg_set, config, key=key, device=config.device)
        start_time = time.time()
        procedure = None
        if arg_set.test:
            if arg_set.validation:
                procedure = 'Inference_validation'
                _ = runner.test()
            else:
                if arg_set.loss_guidance == 'L2':
                    (y_rmse_all_steps_list, y_qice_all_steps_list, 
                     y_picp_all_steps_list, y_nll_all_steps_list) = runner.test()
                else:
                    (y_mae_all_steps_list, y_qice_all_steps_list,
                     y_picp_all_steps_list, y_nll_all_steps_list) = runner.test()
                procedure = 'Testing'
        else:
            runner.train()
            procedure = 'Training'
        end_time = time.time()
        logging.info('\n{} procedure finished. It took {:.4f} minutes.\n\n\n'.format(
            procedure, (end_time - start_time) / 60))
        # remove logging handlers
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
        # return test metric lists
        if arg_set.test:
            if not arg_set.validation:
                if arg_set.loss_guidance == 'L2':
                    return (y_rmse_all_steps_list, y_qice_all_steps_list, 
                            y_picp_all_steps_list, y_nll_all_steps_list, config)
                else:
                    return (y_mae_all_steps_list, y_qice_all_steps_list,
                            y_picp_all_steps_list, y_nll_all_steps_list, config)
            else:
                return None
    except Exception:
        logging.error(traceback.format_exc())


# The main method for execution of all UQ techniques
def main():
   
    # Parse arguments     
    parser = argparse.ArgumentParser(
        description='Probabilistic remaining time prediction') 
    
    ##########################################################################
    #######################    general arguments    ##########################
    ##########################################################################
    
    # Argument to select the dataset (i.e., an event log)
    parser.add_argument('--dataset', default='HelpDesk',
                        help='Dataset used by model')    
    # Argument to select backbone architecture.
    parser.add_argument('--model', default='dalstm',
                        help='Type of the predictive model')
    # select the uncertainty quantification approach
    parser.add_argument('--UQ', default='deterministic',
                        help='Uncertainty quantification method to be used')
    parser.add_argument('--cfg', help='configuration for training & inference')
    # five splits are used for cross-fold valiation by default
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of splits that is used')
    # CARD: n_splits == 1 implies holdout split, otherwise cross-fold validation
    # Other UQ techniques: split_mode should be explicitly defined
    parser.add_argument('--split_mode', default='holdout',
                        help='The data split that is used')
    # CARD: each seed should be executed separately from command line.
    # Other UQ techniques: handle multiple seeds e.g., seed=[42,56,79]
    parser.add_argument('--seed', nargs='+', help='Random seed to use',
                        required=True)
    # device and thread arguments.
    parser.add_argument('--device', type=int, default=0, help='GPU device id')
    parser.add_argument('--thread', type=int, default=4,
                        help='number of threads') 
    
    ##########################################################################
    #########################    CARD arguments    ###########################
    ##########################################################################
    """
        --timesteps: number of timesteps can be set using command line.
        If not spcified, it should be included in the configuration file.
    """
    parser.add_argument('--timesteps', type=int, default=None,
                            help='number of steps involved')    

    args = parser.parse_args()    
    root_path = os.getcwd()
       
    # A separate execution route for CARD model    
    if args.UQ == 'CARD':
        # set important torch settings
        torch.set_printoptions(sci_mode=False)
        main_cfg_file = os.path.join(root_path, 'cfg', args.cfg)
        with open(main_cfg_file, 'r') as f:
            main_cfg = yaml.safe_load(f)
        # load dimensions of the model and add them to args
        if args.model == 'dalstm':
            dalstm_class = 'DALSTM_' + args.dataset
            x_dim_path = os.path.join(
                root_path, 'datasets', 
                dalstm_class,f'DALSTM_input_size_{args.dataset}.pkl')            
            with open(x_dim_path, 'rb') as file:
                args.x_dim = pickle.load(file)            
            max_len_path = os.path.join(
                root_path, 'datasets', dalstm_class,
                f'DALSTM_max_len_{args.dataset}.pkl')
            with open(max_len_path, 'rb') as file:
                args.max_len = pickle.load(file) 
        elif args.model == 'pgtnet':
            print('CARD models for PGTNet architecture is not implemented.')
        test_controls = [False, True]
        # attach experments to args
        args = set_experiments(args, main_cfg)
        # conduct train and test for each experiment
        for key in args.exp_dict:
            # one iteration for training, and one for inference            
            for control in test_controls:
                if not control:
                    args = adjust_arguments(args, main_cfg, root_path=root_path)
                    print(f'Training is conducted for {key} from \
                          {len(args.exp_dict.keys())} experiments')
                    # set a path for saving running related data.
                    args.exp = os.path.join(args.instance_path, 'card', key)
                    # set the log folder- for documentation purpose
                    args.doc =  args.model + '_' + args.dataset + '_card' 
                    # configuration file is located in cfg folder
                    args.config = os.path.join(
                        root_path, 'cfg', 
                        args.model+'_'+args.dataset+'_card'+'.yml')
                else:
                    args = adjust_arguments(args, main_cfg, test=True,
                                            validation=True,
                                            root_path=root_path)
                    print(f'Test is conducted for {key} from \
                          {len(args.exp_dict.keys())} experiments')
                    # set a path for saving running related data.
                    args.exp = os.path.join(args.instance_path, 'card', key)
                    # set the log folder- for documentation purpose
                    args.doc =  args.model + '_' + args.dataset + '_card' 
                    # configuration file is the one created during training
                    args.config = os.path.join(args.exp, 'logs/')

                temp_config = parse_temp_config(args.doc)
                
                if args.run_all:
                    original_doc = args.doc
                    original_config = args.config
                    for split in range(args.init_split, args.n_splits):
                        args.split = split
                        args.doc = original_doc + '/split_' + str(args.split)
                        if args.test:
                            args.config = (original_config + args.doc + 
                                           '/config.yml')
                            _ = main_card(arg_set= args, key=key)
                        else:
                            main_card(arg_set= args, key=key)
                else:
                    args.doc = args.doc + "/split_" + str(args.split)
                    if args.test:
                        args.config = args.config + args.doc + "/config.yml"
                    sys.exit(main())
        
        # select best hyper-parameter combination
        best_key = card_hpo(args)
        args = adjust_arguments(args, main_cfg, test=True,
                                validation=False,
                                root_path=root_path)
        print(f'Test is conducted for {best_key} as the best experiment.')
        # set a path for saving running related data.
        args.exp = os.path.join(args.instance_path, 'card', best_key)
        # set the log folder- for documentation purpose
        args.doc =  args.model + '_' + args.dataset + '_card' 
        # configuration file is the one created during training
        args.config = os.path.join(args.exp, 'logs/')
        temp_config = parse_temp_config(args.doc)

        if args.run_all:
            y_qice_all_splits_all_steps_list = []
            y_picp_all_splits_all_steps_list = []
            y_nll_all_splits_all_steps_list = []
            if args.loss_guidance == 'L2':
                y_rmse_all_splits_all_steps_list = []
            else:
                y_mae_all_splits_all_steps_list = []
            original_doc = args.doc
            original_config = args.config
            for split in range(args.init_split, args.n_splits):
                args.split = split
                args.doc = original_doc + '/split_' + str(args.split)
                args.config = (original_config + args.doc + '/config.yml')
                if args.loss_guidance == 'L2':
                    (y_rmse_all_steps_list, y_qice_all_steps_list,
                     y_picp_all_steps_list, y_nll_all_steps_list,
                     config) = main_card(arg_set= args, key=best_key)
                    y_rmse_all_splits_all_steps_list.append(y_rmse_all_steps_list)    
                else:
                    (y_mae_all_steps_list, y_qice_all_steps_list,
                     y_picp_all_steps_list, y_nll_all_steps_list,
                     config) = main_card(arg_set= args, key=best_key)
                    y_mae_all_splits_all_steps_list.append(y_mae_all_steps_list)            
                y_qice_all_splits_all_steps_list.append(y_qice_all_steps_list)
                y_picp_all_splits_all_steps_list.append(y_picp_all_steps_list)
                y_nll_all_splits_all_steps_list.append(y_nll_all_steps_list)                    
            if args.loss_guidance == 'L2':
                get_stat(
                    args, config, temp_config, original_doc, 
                    y_rmse_all_splits_all_steps_list=y_rmse_all_splits_all_steps_list,
                    y_qice_all_splits_all_steps_list=y_qice_all_splits_all_steps_list,
                    y_picp_all_splits_all_steps_list=y_picp_all_splits_all_steps_list,
                    y_nll_all_splits_all_steps_list=y_nll_all_splits_all_steps_list)
            else:
                get_stat(
                    args, config, temp_config, original_doc,
                    y_mae_all_splits_all_steps_list=y_mae_all_splits_all_steps_list,
                    y_qice_all_splits_all_steps_list=y_qice_all_splits_all_steps_list,
                    y_picp_all_splits_all_steps_list=y_picp_all_splits_all_steps_list,
                    y_nll_all_splits_all_steps_list=y_nll_all_splits_all_steps_list)

        else:
            args.doc = args.doc + "/split_" + str(args.split)
            if args.test:
                args.config = args.config + args.doc + "/config.yml"
            sys.exit(main())
        
        card_calibration(args, best_key)

    # execution path for all methods except CARD
    else:    
        # define device name
        device_name = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        # read the relevant configuration file, and append
        cfg_file = os.path.join(root_path, 'cfg', args.cfg)
        with open(cfg_file, 'r') as f:
            cfg = yaml.safe_load(f)     
        # append the configuration file
        cfg['net'] = args.model
        cfg['uq_method'] = args.UQ
        cfg['dataset'] = args.dataset
        cfg['split'] = args.split_mode
        cfg['n_splits'] = args.n_splits
        cfg['device'] = device_name
        cfg['seed'] = args.seed
               
        if args.model == 'dalstm':
            DALSTM_train_evaluate(cfg=cfg)
            
    
if __name__ == '__main__':
    main()