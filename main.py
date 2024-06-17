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
import traceback
import time
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt
import numpy as np
import torch
from models.card_regression import Diffusion
from utils.DALSTM_train_eval import DALSTM_train_evaluate
from utils.utils import parse_temp_config, parse_config
#TODO: uncomment this import after adjusting the method
#from utils.utils import delete_preprocessd_tensors



def main_card(arg_set=None):
    config, logger = parse_config(args=arg_set)

    logging.info('Writing log file to {}'.format(arg_set.log_path))
    logging.info('Exp instance id = {}'.format(os.getpid()))
    logging.info('Exp comment = {}'.format(arg_set.comment))
    if arg_set.loss != 'card_conditional':
        raise NotImplementedError('Invalid loss option')
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
    parser.add_argument('--dataset', default='HelpDesk',
                        help='Datasets used by model')    
    # some general configurations
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
    # The remaining arguments are only used for CARD model
    # For CARD: config file should be addressed instead of using dataset name.
    parser.add_argument('--config', type=str, help='Path to the config file')   
    # for CARD model, it is possible to train one split or all of them together
    # can be used for cross-validation set up (run all splits)
    parser.add_argument('--run_all', action='store_true', 
                        help='Whether to run all train test splits')    
    # using --split it is possible to train for different splits seperately.
    parser.add_argument('--split', type=int, default=0,
                        help='which split to use for regression data')
    #TODO: check MODEL_VERSION_DIR in args to see how to save the result in a
    # more meaningful path! do wee need: ${RUN_NAME}_${SERVER_NAME}? or even: ${N_STEPS}steps
    parser.add_argument('--exp', type=str, default='exp',
                        help='Path for saving running related data.')
    parser.add_argument('--doc', type=str, 
                        help='Name of the log folder- for \
                            documentation purpose')
    parser.add_argument('--comment', type=str, default='',
                        help='A string for experiment comment')
    parser.add_argument('--verbose', type=str, default='info', 
                        help='Verbose level: info | debug | warning | critical')    
    # to handle test for CARD model
    parser.add_argument('--test', action='store_true',
                        help='Whether to test the model')
    # for CARD model: if set to true, only deterministic model is trained.
    parser.add_argument('--train_guidance_only', action='store_true', 
                        help='Whether to only pre-train the guidance model f_phi')
    parser.add_argument('--use_pretrained', action='store_true')
    parser.add_argument('--resume_training', action='store_true',
                        help='Whether to resume training')
    # --init_split can be used to resume training
    parser.add_argument('--init_split', type=int, default=0,
                        help='initial split to train for regression data')
    #TODO: run experiments without knowledge from pre-trained deterministic model.
    parser.add_argument('--noise_prior', action='store_true', 
                        help='Whether to apply a noise prior distribution at \
                            timestep T')
    # if the following is not specified explicitly: f_phi is used in concatanation
    parser.add_argument('--no_cat_f_phi', action='store_true',
                        help='Whether to not concatenate f_phi as part of \
                            eps_theta input')
    parser.add_argument('--interpolation', action='store_true')                       
    parser.add_argument('--fid', action='store_true')    
    parser.add_argument('-i', '--image_folder', type=str, default='images',
                        help='The folder name of samples')
    parser.add_argument('--rmse_timestep', type=int, default=0,
                        help='selected timestep to report metric y RMSE')
    parser.add_argument('--qice_timestep', type=int, default=0,
                        help='selected timestep to report metric y QICE')
    parser.add_argument('--picp_timestep', type=int, default=0,
                        help='selected timestep to report metric y PICP')
    parser.add_argument('--nll_timestep', type=int, default=0,
                        help='selected timestep to report metric y NLL')
    #TODO: important! use the following in large-scale experiments.
    parser.add_argument( '--ni', action='store_true', 
                        help='No interaction. Suitable for Slurm Job launcher')
    #TODO: important! check how much is important the following arg.
    parser.add_argument('--sample_type', type=str, default='generalized',
                        help='sampling approach (generalized or ddpm_noisy)')
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
    # loss option for CARD model
    parser.add_argument('--loss', type=str, default='card_conditional',
                        help='Which loss to use')
    # loss option for guidance model
    parser.add_argument('--loss_guidance', type=str, default='L1',
                        help='Which loss to use for guidance model: L1/L2')
    #TODO: check the functionality of following arg
    parser.add_argument('--nll_global_var', action='store_true',
                        help='Apply global variance for NLL computation')
    #TODO: important! check how much is important the following arg.
    parser.add_argument('--nll_test_var', action='store_true',
                        help='Apply sample variance of the test set for NLL \
                            computation')
    # Conditional transport options
    #TODO: important! check how much is important the following 2-3 args.
    parser.add_argument('--use_d', action='store_true',
                        help="Whether to take an adversarially trained feature encoder")
    parser.add_argument('--full_joint', action='store_true',
                        help='Whether to take fully joint matching')
    parser.add_argument('--num_sample', type=int, default=1,
                        help='number of samples used in forward and reverse') 
    args = parser.parse_args()
    
    # A separate execution route for CARD model    
    if args.UQ == 'CARD':
        # set scientific number prints to False
        torch.set_printoptions(sci_mode=False)
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
            for split in range(args.init_split, args.split_num):
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

                # plot mean of all metric across all splits at all time steps during reverse diffusion
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
                    res_dict = {'split_num': args.split_num, 'task': original_doc,
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
                    res_dict = {'split_num': args.split_num, 'task': original_doc,
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
                             'guidance': config.diffusion.conditioning_signal,
                             'n_timesteps': n_timesteps,
                             'split_num': args.split_num
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
                # TODO: this should be applied outside calling CARD because
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