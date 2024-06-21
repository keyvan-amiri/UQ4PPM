"""
To prepare this script we used the following source codes:
    https://github.com/XzwHan/CARD
We adjusted the source codes to efficiently integrate them into our framework.
"""

import argparse
import os
import yaml
import json
import sys
import logging
import pickle
import traceback
import time
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt
import numpy as np
import torch
from models.card_regression import Diffusion
from utils.DALSTM_train_eval import DALSTM_train_evaluate
from utils.utils import parse_temp_config, parse_config, str2bool
#TODO: uncomment this import after adjusting the method
#from utils.utils import delete_preprocessd_tensors


#TODO: understand the role of log_path and use it to control the addresses
def main_card(arg_set=None):
    config, logger = parse_config(args=arg_set)

    logging.info('Writing log file to {}'.format(arg_set.log_path))
    logging.info('Exp instance id = {}'.format(os.getpid()))
    logging.info('Exp comment = {}'.format(arg_set.comment))
    try:
        runner = Diffusion(arg_set, config, device=config.device)
        start_time = time.time()
        procedure = None
        if arg_set.test:
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
            if arg_set.loss_guidance == 'L2':
                return (y_rmse_all_steps_list, y_qice_all_steps_list, 
                        y_picp_all_steps_list, y_nll_all_steps_list, config)
            else:
                return (y_mae_all_steps_list, y_qice_all_steps_list, 
                        y_picp_all_steps_list, y_nll_all_steps_list, config)
    except Exception:
        logging.error(traceback.format_exc())


def main():
   
    # Parse arguments     
    parser = argparse.ArgumentParser(
        description='Probabilistic remaining time prediction') 
    
    ##########################################################################
    #######################    general arguments    ##########################
    ##########################################################################
    
    # Argument to select the dataset (i.e., an event log)
    parser.add_argument('--dataset', default='HelpDesk',
                        help='Datasets used by model')    
    """
    Argument to select architecture:
    For CARD: is the architecture of pre-trained point estimator.
    For dropout approximation, and heterosedastic loss: backbone architecture
    
    """
    parser.add_argument('--model', default='pgtnet',
                        help='Type of the predictive model')
    # select the uncertainty quantification approach
    parser.add_argument('--UQ', default='deterministic',
                        help='Uncertainty quantification method to be used')
    """
    Split arguments: --n_splits and --split_mode:
    dropout and heterosedastic loss: split_mode should be explicitly defined
    CARD: if --n_splits == 1, holdout split is used otherwise cv split
    The default n_splits == 5, is used for preprocessing as well.
    """
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of splits that is used')
    parser.add_argument('--split_mode', default='holdout',
                        help='The data split that is used')
    """
    seed argument:
    CARD: each seed should be executed separately from command line.
    Other approches can handle multiple seeds e.g., seed=[42,56,79]
    """
    parser.add_argument('--seed', nargs='+', help='Random seed to use',
                        required=True)
    # device and thread arguments.
    parser.add_argument('--device', type=int, default=0, help='GPU device id')
    parser.add_argument('--thread', type=int, default=4,
                        help='number of threads') 
    
    ##########################################################################
    #########################    CARD arguments    ###########################
    ##########################################################################
    # if --test is provided evaluation is done, otherwise: training
    parser.add_argument('--test', action='store_true',
                        help='Whether to test the model')
    # if set to true, only deterministic model is trained.
    parser.add_argument('--train_guidance_only', action='store_true', 
                        help='Whether to only pre-train the guidance model f_phi')
    """
    CARD model arguments:
        --loss_guidance: loss function for guidance model that predicts y_0_hat.
        If L2 all CARD, results are reorted in RMSE otherwise it will be
        reported in MAE.  
        --noise_prior: if explicitly added in the command line:
            a noise prior is applied instead of pre-trained guidance model.
            noise_prior_approach (zero, mean, or median) is specified in config.
        --no_cat_f_phi: if explicitly added in the command line:
            new_config.model.cat_y_pred is set to False. This means that f_phi
            (point estimate) is not concatanated in noise estimation network.
        --timesteps: number of timesteps can be set using command line.
        If not spcified, it should be included in the configuration file.
    """
    parser.add_argument('--loss_guidance', type=str, default='L2',
                        help='Which loss to use for guidance model: L1/L2')
    parser.add_argument('--noise_prior', action='store_true', 
                        help='Whether to apply a noise prior distribution at \
                            timestep T')
    parser.add_argument('--no_cat_f_phi', action='store_true',
                        help='Whether to not concatenate f_phi as part of \
                            eps_theta input')  
    parser.add_argument('--timesteps', type=int, default=None,
                            help='number of steps involved')
    
    """
    Execution control arguments:
        --run_all: train or test all available splits together (default: True)
        --split: to train or test different splits seperately
        --init_split: can be used to resume training
        --resume_training: to resume training
        --ni:         
    """
    parser.add_argument('--run_all', type=str2bool, nargs='?',
                        const=True, default=True,
                        help='Whether to run all train test splits.')
    parser.add_argument('--split', type=int, default=0,
                        help='which split to use for regression data')
    parser.add_argument('--init_split', type=int, default=0,
                        help='initial split to train for regression data')
    parser.add_argument('--resume_training', action='store_true',
                        help='Whether to resume training')
    #TODO: important! use the following in large-scale experiments.
    parser.add_argument( '--ni', action='store_true', 
                        help='No interaction. Suitable for Slurm Job launcher')
    
    # Evaluation metrics arguments:
    parser.add_argument('--rmse_timestep', type=int, default=0,
                        help='selected timestep to report metric y RMSE')
    parser.add_argument('--qice_timestep', type=int, default=0,
                        help='selected timestep to report metric y QICE')
    parser.add_argument('--picp_timestep', type=int, default=0,
                        help='selected timestep to report metric y PICP')
    parser.add_argument('--nll_timestep', type=int, default=0,
                        help='selected timestep to report metric y NLL')
    # two following args are related to NLL computation
    parser.add_argument('--nll_global_var', action='store_true',
                        help='Apply global variance for NLL computation')
    parser.add_argument('--nll_test_var', action='store_true',
                        help='Apply sample variance of the test set for NLL \
                            computation')
    
    
    # documentation arguments:
    parser.add_argument('--comment', type=str, default='',
                            help='A string for experiment comment')
    parser.add_argument('--verbose', type=str, default='info', 
                            help='Verbose level: info | debug | warning | critical')
    parser.add_argument('-i', '--image_folder', type=str, default='images',
                        help='The folder name of samples')    
    
    args = parser.parse_args()    
    root_path = os.getcwd()
    
    # A separate execution route for CARD model    
    if args.UQ == 'CARD':
        # set scientific number prints to False
        torch.set_printoptions(sci_mode=False)
        # set a path for instance-level predictions
        instance_path = os.path.join(root_path, 'results', args.dataset,
                                     args.model)
        args.instance_path = instance_path
        # set a path for saving running related data.
        exp_path = os.path.join(instance_path, 'card')
        args.exp = exp_path       
        # set the log folder- for documentation purpose
        args.doc =  args.model + '_' + args.dataset + '_card'                              
        # set the configuration file based on train/test task        
        if args.test:
            # in test: configuration file is the one created during training
            config_path = os.path.join(exp_path, 'logs/')
        else: 
            # in training: there is a seprate configuration file for CARD
            config_path = os.path.join(root_path, 'cfg',
                                args.model+'_'+args.dataset+'_card'+'.yml')
        args.config = config_path
        # load dimensions of the model and add them to args
        if args.model == 'dalstm':
            dalstm_class = 'DALSTM_' + args.dataset
            x_dim_path = os.path.join(root_path, 'datasets', dalstm_class,
                                    'DALSTM_input_size_HelpDesk.pkl')
            with open(x_dim_path, 'rb') as file:
                args.x_dim = pickle.load(file)            
            max_len_path = os.path.join(root_path, 'datasets', dalstm_class,
                                    'DALSTM_max_len_HelpDesk.pkl')
            with open(max_len_path, 'rb') as file:
                args.max_len = pickle.load(file)   

        temp_config = parse_temp_config(args.doc)
        
        if args.run_all:
            if args.loss_guidance == 'L2':
                y_rmse_all_splits_all_steps_list, \
                    y_qice_all_splits_all_steps_list, \
                        y_picp_all_splits_all_steps_list,\
                            y_nll_all_splits_all_steps_list = [], [], [], []
            else:
                y_mae_all_splits_all_steps_list, \
                    y_qice_all_splits_all_steps_list, \
                        y_picp_all_splits_all_steps_list, \
                            y_nll_all_splits_all_steps_list = [], [], [], []
            original_doc = args.doc
            original_config = args.config
            for split in range(args.init_split, args.n_splits):
                args.split = split
                args.doc = original_doc + '/split_' + str(args.split)
                if args.test:
                    args.config = original_config + args.doc + '/config.yml'
                    if args.loss_guidance == 'L2':
                        (y_rmse_all_steps_list, y_qice_all_steps_list,
                         y_picp_all_steps_list, y_nll_all_steps_list, config
                         ) = main_card(arg_set= args)
                        y_rmse_all_splits_all_steps_list.append(
                            y_rmse_all_steps_list)    
                    else:
                        (y_mae_all_steps_list, y_qice_all_steps_list, 
                         y_picp_all_steps_list, y_nll_all_steps_list, config
                         ) = main_card(arg_set= args)
                        y_mae_all_splits_all_steps_list.append(
                            y_mae_all_steps_list)            
                    y_qice_all_splits_all_steps_list.append(y_qice_all_steps_list)
                    y_picp_all_splits_all_steps_list.append(y_picp_all_steps_list)
                    y_nll_all_splits_all_steps_list.append(y_nll_all_steps_list)
                else:
                    main_card(arg_set= args)
            # summary statistics across all splits
            if args.run_all and args.test:
                n_timesteps = config.diffusion.timesteps
                rmse_idx = n_timesteps - args.rmse_timestep
                qice_idx = n_timesteps - args.qice_timestep
                picp_idx = n_timesteps - args.picp_timestep
                nll_idx = n_timesteps - args.nll_timestep
                if not temp_config.testing.compute_metric_all_steps:
                    n_timesteps=rmse_idx=qice_idx=picp_idx=nll_idx=0
                #print(len(y_mae_all_splits_all_steps_list))
                #print(y_mae_all_splits_all_steps_list[0])
                if args.loss_guidance == 'L2':
                    y_rmse_all_splits_list = [metric_list[rmse_idx] 
                                              for metric_list in 
                                              y_rmse_all_splits_all_steps_list]
                else:
                    y_mae_all_splits_list = [metric_list[rmse_idx] 
                                             for metric_list in 
                                             y_mae_all_splits_all_steps_list]                   
                y_qice_all_splits_list = [metric_list[qice_idx]
                                          for metric_list in
                                          y_qice_all_splits_all_steps_list]
                y_picp_all_splits_list = [metric_list[picp_idx]
                                          for metric_list in
                                          y_picp_all_splits_all_steps_list]
                y_nll_all_splits_list = [metric_list[nll_idx]
                                         for metric_list in
                                         y_nll_all_splits_all_steps_list]

                print("\n\n================ Results Across Splits ================")
                if args.loss_guidance == 'L2':
                    print(f"y_RMSE mean: {np.mean(y_rmse_all_splits_list)}\
                          y_RMSE std: {np.std(y_rmse_all_splits_list)}")
                else:
                    print(f"y_MAE mean: {np.mean(y_mae_all_splits_list)} \
                          y_MAE std: {np.std(y_mae_all_splits_list)}")
                print(f"QICE mean: {np.mean(y_qice_all_splits_list)}\
                      QICE std: {np.std(y_qice_all_splits_list)}")
                print(f"PICP mean: {np.mean(y_picp_all_splits_list)}\
                      PICP std: {np.std(y_picp_all_splits_list)}")
                print(f"NLL mean: {np.mean(y_nll_all_splits_list)}\
                      NLL std: {np.std(y_nll_all_splits_list)}")

                """
                plot mean of all metric across all splits at all time steps
                during reverse diffusion
                """
                if args.loss_guidance == 'L2':
                    y_rmse_all_splits_all_steps_array = np.array(
                        y_rmse_all_splits_all_steps_list)
                    y_rmse_mean_all_splits_list = [np.mean(
                            y_rmse_all_splits_all_steps_array[:, idx]) 
                        for idx in range(n_timesteps + 1)]                 
                else:
                    y_mae_all_splits_all_steps_array = np.array(
                        y_mae_all_splits_all_steps_list)        
                    y_mae_mean_all_splits_list = [np.mean(
                            y_mae_all_splits_all_steps_array[:, idx])
                        for idx in range(n_timesteps + 1)]  
                y_qice_all_splits_all_steps_array = np.array(
                    y_qice_all_splits_all_steps_list)
                y_picp_all_splits_all_steps_array = np.array(
                    y_picp_all_splits_all_steps_list)
                y_nll_all_splits_all_steps_array = np.array(
                    y_nll_all_splits_all_steps_list)         
                y_qice_mean_all_splits_list = [np.mean(
                        y_qice_all_splits_all_steps_array[:, idx]) 
                    for idx in range(n_timesteps + 1)]
                y_picp_mean_all_splits_list = [np.mean(
                        y_picp_all_splits_all_steps_array[:, idx]) 
                    for idx in range(n_timesteps + 1)]
                y_nll_mean_all_splits_list = [np.mean(
                        y_nll_all_splits_all_steps_array[:, idx])
                    for idx in range(n_timesteps + 1)]
                n_metric = 4
                fig, axs = plt.subplots(n_metric, 1,
                                        figsize=(8.5, n_metric * 3))  # W x H
                plt.subplots_adjust(hspace=0.5)
                xticks = np.arange(0, n_timesteps + 1,
                                   config.diffusion.vis_step)
                # MAE/RMSE
                if args.loss_guidance == 'L2':
                    axs[0].plot(y_rmse_mean_all_splits_list)
                else:   
                    axs[0].plot(y_mae_mean_all_splits_list)
                # axs[0].set_title('mean y RMSE of All Splits across All Timesteps', fontsize=14)
                axs[0].set_xlabel('timestep', fontsize=12)
                axs[0].set_xticks(xticks)
                axs[0].set_xticklabels(xticks[::-1])
                if args.loss_guidance == 'L2':
                    axs[0].set_ylabel('y RMSE', fontsize=12)
                else:
                    axs[0].set_ylabel('y MAE', fontsize=12)                    
                # NLL
                axs[1].plot(y_nll_mean_all_splits_list)
                # axs[3].set_title('mean y NLL of All Splits across All Timesteps', fontsize=14)
                axs[1].set_xlabel('timestep', fontsize=12)
                axs[1].set_xticks(xticks)
                axs[1].set_xticklabels(xticks[::-1])
                axs[1].set_ylabel('y NLL', fontsize=12)
                # QICE
                axs[2].plot(y_qice_mean_all_splits_list)
                # axs[1].set_title('mean y QICE of All Splits across All Timesteps', fontsize=14)
                axs[2].set_xlabel('timestep', fontsize=12)
                axs[2].set_xticks(xticks)
                axs[2].set_xticklabels(xticks[::-1])
                axs[2].set_ylabel('y QICE', fontsize=12)
                # PICP
                picp_ideal = (
                    config.testing.PICP_range[1]-config.testing.PICP_range[0])/100
                axs[3].plot(y_picp_mean_all_splits_list)
                axs[3].axhline(y=picp_ideal, c='b', label='95 % coverage')
                # axs[2].set_title('mean y PICP of All Splits across All Timesteps', fontsize=14)
                axs[3].set_xlabel('timestep', fontsize=12)
                axs[3].set_xticks(xticks)
                axs[3].set_xticklabels(xticks[::-1])
                axs[3].set_ylabel('y PICP', fontsize=12)
                axs[3].legend()

                # fig.suptitle('Mean y Metrics of All Splits across All Timesteps')
                im_path = os.path.join(args.exp, config.testing.image_folder,
                                       original_doc)
                if not os.path.exists(im_path):
                    os.makedirs(im_path)
                fig.savefig(
                    os.path.join(im_path,
                                 'mean_metrics_all_splits_all_timesteps.pdf'))

                timestr = datetime.now(
                    timezone(timedelta(hours=+2))).strftime(
                        "%Y%m%d-%H%M%S-%f")  # Europe Central time
                res_file_path = os.path.join(
                    args.exp, "logs", original_doc, "metrics_all_splits")
                if args.loss_guidance == 'L2':
                    res_dict = {'split_num': args.n_splits, 'task': original_doc,
                                'y_RMSE mean': float(np.mean(y_rmse_all_splits_list)),
                                'y_RMSE std': float(np.std(y_rmse_all_splits_list)),
                                'QICE mean': float(np.mean(y_qice_all_splits_list)),
                                'QICE std': float(np.std(y_qice_all_splits_list)),
                                'PICP mean': float(np.mean(y_picp_all_splits_list)),
                                'PICP std': float(np.std(y_picp_all_splits_list)),
                                'NLL mean': float(np.mean(y_nll_all_splits_list)),
                                'NLL std': float(np.std(y_nll_all_splits_list))
                                }  
                else:
                    res_dict = {'split_num': args.n_splits, 'task': original_doc,
                                'y_MAE mean': float(np.mean(y_mae_all_splits_list)),
                                'y_MAE std': float(np.std(y_mae_all_splits_list)),
                                'QICE mean': float(np.mean(y_qice_all_splits_list)),
                                'QICE std': float(np.std(y_qice_all_splits_list)),
                                'PICP mean': float(np.mean(y_picp_all_splits_list)),
                                'PICP std': float(np.std(y_picp_all_splits_list)),
                                'NLL mean': float(np.mean(y_nll_all_splits_list)),
                                'NLL std': float(np.std(y_nll_all_splits_list))
                                }              
                args_dict = {'task': config.data.dataset,
                             'loss': args.loss,
                             'guidance': config.model.type,
                             'n_timesteps': n_timesteps,
                             'split_num': args.n_splits
                             }
                # save metrics and model hyperparameters to a json file
                if not os.path.exists(res_file_path):
                    os.makedirs(res_file_path)
                with open(res_file_path + f"/metrics_{timestr}.json", "w") as outfile:
                    json.dump(res_dict, outfile)
                    outfile.write('\n\nExperiment arguments:\n')
                    json.dump(args_dict, outfile)
                print("\nTest metrics saved in .json file.")
                # TODO: uncomment the following for large sclae experiments
                # this should be applied outside calling CARD because
                # in other UQ methods, we also need this functionality
                # delete_preprocessd_tensors(temp_config) # delete preprocessed dataset tensors
        else:
            args.doc = args.doc + "/split_" + str(args.split)
            if args.test:
                args.config = args.config + args.doc + "/config.yml"
            sys.exit(main())
            
    else:    
        # define device name
        device_name = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        # read the relevant configuration file, and append
        cfg_file = os.path.join(root_path, 'cfg',
                                args.model+'_'+args.dataset+'.yaml')
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