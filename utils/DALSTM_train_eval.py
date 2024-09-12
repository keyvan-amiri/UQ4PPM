import os
import pickle
import shutil
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from models.Laplace_approximation import post_hoc_laplace, inference_laplace
from models.Random_Forest import fit_rf, predict_rf
from models.Train import train_model
from models.Test import test_model
from loss.loss_handler import set_loss
from loss.QuantileLoss import QuantileLoss
from utils.utils import (get_model, get_optimizer_scheduler, set_random_seed,
                         get_exp, add_suffix_to_csv, adjust_model_name)
from utils.evaluation import uq_eval
from utils.calibration import calibrated_regression


# A generic class for training and evaluation of DALSTM model
class DALSTM_train_evaluate ():
    def __init__ (self, cfg=None, dalstm_dir=None): 
        """
        A generic class for handling training, inference, and hyperparameter
        optimization for dropout approximaton, heteroscedastic regression,
        deep ensembles, Laplace approxmitation, embedding-based approaches 
        (i.e., Random Forest), and simultaneous quantile regression with LSTM
        point estimates.
        Parameters:
        cfg : configuration that is used for training, and inference.
        dalstm_dir : user specified folder for dataset which contain all
        information about feature vectors represinting event prefixes, and 
        used by DALSTM model.
        """
        ######################################################################
        #####  Initial operations for effective training, and inference  #####
        ######################################################################
        self.cfg = cfg
        seeds = cfg.get('seed')
        device_name = cfg.get('device')
        # set device    
        self.device = torch.device(device_name)
        # define dataset, and set the path to access pre-processed dataset
        self.dataset = cfg.get('dataset')
        root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if dalstm_dir is None:
            self.dalstm_dir = os.path.join(root_path, 'datasets')
        else:
            self.dalstm_dir = dalstm_dir            
        self.dalstm_class = 'DALSTM_' + self.dataset
        self.dataset_path =  os.path.join(self.dalstm_dir, self.dalstm_class)
        # set the address for all outputs: training and inference
        self.result_path = os.path.join(root_path,
                                        'results', self.dataset, 'dalstm')
        # create the output address (i.e., 'results') if not in root directory.
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        # set the address for calibration on final results
        self.recalibration_path = os.path.join(self.result_path, 'recalibration')
        if not os.path.exists(self.recalibration_path):
            os.makedirs(self.recalibration_path)            
        # set normalization status
        self.normalization = cfg.get('data').get('normalization')
        # get the specific uncertainty quanitification defined in command line
        self.uq_method = cfg.get('uq_method')
        # metric that is used for hyper-parameter optimization
        self.HPO_metric = cfg.get('HPO_metric')
        # type of calibration used (scaling or isotonic regression)
        self.calibration_type = cfg.get('calibration_type')
        # get the type of data split, and possibly number of splits for CV        
        self.split = cfg.get('split')
        self.n_splits = cfg.get('n_splits')     
        # Define important size and dimensions for DALSTM model
        (self.input_size, self.max_len, self.max_train_val
         ) = self.load_dimensions()
        self.hidden_size = cfg.get('model').get('lstm').get('hidden_size')
        self.n_layers = cfg.get('model').get('lstm').get('n_layers')
        # whether to use drop-out or not
        self.dropout = cfg.get('model').get('lstm').get('dropout')
        # the probabibility that is used for dropout
        self.dropout_prob = cfg.get('model').get('lstm').get('dropout_prob')
        # define training hyperparameters
        if self.uq_method != 'SQR':
            # in case of SQR we allow for different epoch sizes 
            self.max_epochs = cfg.get('train').get('max_epochs')
        self.early_stop_patience = cfg.get('train').get('early_stop.patience')
        self.early_stop_min_delta = cfg.get('train').get('early_stop.min_delta')        
        if not (self.uq_method == 'DA' or self.uq_method == 'CDA' or
                self.uq_method == 'DA_A' or self.uq_method == 'CDA_A' or 
                self.uq_method == 'mve'):
            # in case of of these methods is chosen, we get these values
            # as parameters for HPO. it means we only test True, and False
            # for early stopping for dropout methods. For mve, we only have
            # one experiment.
            self.num_mcmc = None      
            self.early_stop = cfg.get('train').get('early_stop')
        ######################################################################
        ##########   experiments for hyperparameter optimization   ###########
        ######################################################################
        HPO_radom_ratio = cfg.get('HPO_radom_ratio')
        if (self.uq_method == 'en_b' or self.uq_method == 'en_b_mve' or
            self.uq_method == 'en_t' or self.uq_method == 'en_t_mve'):
            # set the experiments considering grid search over parameters
            self.experiments, self.max_model_num = get_exp(
                uq_method=self.uq_method, cfg=cfg)
        else:
            # set the experiments considering random search over parameters
            self.experiments = get_exp(
                uq_method=self.uq_method, cfg=cfg, is_random=True,
                random_ratio=HPO_radom_ratio)
            # set the experiments considering grid search over parameters
            #self.experiments = get_exp(uq_method=self.uq_method, cfg=cfg)            
        ######################################################################
        #################   Laplace Approximation setting   ##################
        ######################################################################
        # set necessary parameters for Laplace approximation
        if self.uq_method == 'LA':
            self.laplace = True
            self.last_layer_name = cfg.get('uncertainty').get('laplace').get(
                'last_layer_name')           
        else:
            self.laplace = False        
        ######################################################################
        ########################   Ensemble setting   ########################
        ######################################################################
        if (self.uq_method == 'en_b' or self.uq_method == 'en_b_mve' or
            self.uq_method == 'en_t' or self.uq_method == 'en_t_mve'):
            self.ensemble_mode = True
            if (self.uq_method == 'en_b' or self.uq_method == 'en_b_mve'):
                self.bootstrapping = True
            else:
                self.bootstrapping = False
        else:
            self.ensemble_mode = False
            self.bootstrapping = False
        ######################################################################
        ########################   Dropout setting   ########################
        ######################################################################
        # define parameters required for dropout approximation
        if (self.uq_method == 'DA' or self.uq_method == 'CDA' or
              self.uq_method == 'DA_A' or self.uq_method == 'CDA_A'):
            # hard-coded parameters since we have separate deterministic model
            self.dropout = True
            self.Bayes = True
            # define regularizers for dropout approximation
            self.weight_regularizer = cfg.get('uncertainty').get(
                'dropout_approximation').get('weight_regularizer')
            self.dropout_regularizer = cfg.get('uncertainty').get(
                'dropout_approximation').get('dropout_regularizer')
            # Set the parameter for concrete dropout
            if (self.uq_method == 'DA' or self.uq_method == 'DA_A'):
                self.concrete_dropout = False
            else:
                self.concrete_dropout = True
        else:
            # necessary to use the same execution path
            self.concrete_dropout = None
            self.weight_regularizer = None
            self.dropout_regularizer = None
            self.Bayes = None             
        ######################################################################
        #####################   Random Forest setting   ######################
        ######################################################################
        if self.uq_method == 'RF':
            self.union_mode = True
        else:
            self.union_mode = False
        ######################################################################
        ###############   Simultaneous Quantile Regression   #################
        ######################################################################
        # NOTE: to use the same execution path for other methods as well
        if self.uq_method != 'SQR':
            self.sqr_factor = 12
            self.sqr_q = 'all'       
        ######################################################################
        #########   Loss function  (heteroscedastic/homoscedastic)  ##########
        ######################################################################
        if (self.uq_method == 'deterministic' or self.uq_method == 'DA'
            or self.uq_method == 'CDA' or self.uq_method == 'en_t' or
            self.uq_method == 'en_b'):
            self.criterion = set_loss(
                loss_func=cfg.get('train').get('loss_function'))            
        elif (self.uq_method == 'DA_A' or self.uq_method == 'CDA_A' or
            self.uq_method == 'mve' or self.uq_method == 'en_t_mve' or
            self.uq_method == 'en_b_mve'):
            self.criterion = set_loss(
                loss_func=cfg.get('train').get('loss_function'),
                heteroscedastic=True) 
        elif self.uq_method == 'SQR':
            self.criterion = QuantileLoss()        
        ######################################################################
        ################  model, scheduler, and optimizer  ###################
        ######################################################################
        if not self.ensemble_mode:
            self.model = get_model(
                uq_method=self.uq_method, input_size=self.input_size,
                hidden_size=self.hidden_size, n_layers=self.n_layers,
                max_len=self.max_len, dropout=self.dropout,
                dropout_prob=self.dropout_prob,
                concrete_dropout=self.concrete_dropout,
                weight_regularizer=self.weight_regularizer,
                dropout_regularizer=self.dropout_regularizer,
                Bayes=self.Bayes, device=self.device)
            if ((not self.union_mode) and (not self.laplace)):
                self.optimizer, self.scheduler = get_optimizer_scheduler(
                    model=self.model, cfg=self.cfg)
                # otherwise, learning auxiliary models including the Laplace 
                # model and random forest will be handled using defualt strategy
                # for back-popagation, or decisions on nodes in trees.
        else:
            self.models = get_model(
                uq_method=self.uq_method, input_size=self.input_size,
                hidden_size=self.hidden_size, n_layers=self.n_layers,
                max_len=self.max_len, dropout=self.dropout,
                dropout_prob=self.dropout_prob, num_models=self.max_model_num,
                concrete_dropout=self.concrete_dropout,
                weight_regularizer=self.weight_regularizer,
                dropout_regularizer=self.dropout_regularizer,
                Bayes=self.Bayes, device=self.device)                
            self.optimizers, self.schedulers = get_optimizer_scheduler(
                models=self.models, cfg=self.cfg, ensemble_mode=True,
                num_models=self.max_model_num)                 
        ######################################################################
        ################## Training and evaluation loop  #####################
        ######################################################################
        # loop over seeds
        for execution_seed in seeds:            
            # set random seed
            self.seed = int(execution_seed) 
            set_random_seed(self.seed)
            # lists for best models checkpoints & reports (all experiments)
            self.all_checkpoints, self.all_reports = [], []
            # lists for inference results on val, test sets (all experiments)
            self.all_val_results, self.all_test_results = [], []
            
            # loop over experiments (HPO)
            for exp_id, experiment in enumerate(self.experiments):
                
                ##############################################################
                ######### train-test pipeline for holdout data split  ########
                ##############################################################
                if self.split == 'holdout':
                    
                    # define the report path
                    self.report_path = os.path.join(
                        self.result_path,
                        '{}_{}_seed_{}_exp_{}_report_.txt'.format(
                            self.uq_method, self.split, self.seed, exp_id+1)) 
                    self.all_reports.append(self.report_path)
                    
                    # load train, validation, and test data loaders
                    (self.X_train_path, self.X_val_path, self.X_test_path,
                     self.y_train_path, self.y_val_path, self.y_test_path,
                     self.test_lengths_path) = self.holdout_paths()                    
                    if not self.bootstrapping:
                        # except for Bootstrapping ensemble
                        (self.train_loader, self.val_loader, self.test_loader,
                         self.train_val_loader, self.y_train_val,
                         self.test_lengths) = self.load_data()
                        
                    # training, and inference on validation set
                    if ((not self.ensemble_mode) and (not self.union_mode) and
                        (not self.laplace)): 
                        # For:
                            # 1) Deterministic point estimate.
                            # 2) Dropout approximation
                            # 3) Heteroscedastic regression
                            # combinations of 2,3
                            # 4) Simultaneous Quantile Regression
                        #TODO: check the piepline for SQR!!!
                        self.baseline_approach(exp_id, experiment)
                    elif self.ensemble_mode:
                        # For:
                            # Ensembles (Traditional/Bootstrapping)
                            # Ensembles combined with heteroscedastic regression
                        self.ensemble(exp_id, experiment)                   
                    elif self.union_mode: 
                        # union (embedding) based: for now only Random Forests
                        self.aux_forest(exp_id, experiment)
                    elif self.laplace:
                        # For: Laplace Approximation
                        self.laplace_pipeline(exp_id, experiment)
                        

                # train-test pipeline for cross=validation data split          
                else:
                    for split_key in range(self.n_splits):
                        # define the report path
                        self.report_path = os.path.join(
                            self.result_path,
                            '{}_{}_fold{}_seed_{}_exp_{}_report_.txt'.format(
                                self.uq_method, self.split, split_key+1,
                                self.seed, exp_id+1))

                        # load train, validation, and test data loaders
                        (self.X_train_path, self.X_val_path, self.X_test_path,
                         self.y_train_path, self.y_val_path, self.y_test_path,
                         self.test_lengths_path) = self.cv_paths(
                             split_key=split_key)
                        if not self.bootstrapping:
                            # except for Bootstrapping ensemble
                            (self.train_loader, self.val_loader, self.test_loader,
                             self.train_val_loader, self.y_train_val,
                             self.test_lengths) = self.load_data()  
                        # for train and test:
                            # data_split='cv'
                            # fold = split_key+1
                        # TODO: implement train and test cv split
                            
            if self.uq_method == 'deterministic':
                self.experiments=self.expanded_experiments
            self.final_result, self.final_val = self.select_best() 
            # get uq_metrics for predictions of the best model
            self.uq_metric = uq_eval(self.final_result, self.uq_method,
                                     report=True, verbose=True)  
            self.calibrated_result, self.recal_model = calibrated_regression(
                calibration_df_path=self.final_val,
                test_df_path=self.final_result,
                uq_method=self.uq_method,
                confidence_level=0.95,
                report_path=self.all_reports[0],
                recalibration_path=self.recalibration_path)
            uq_eval(self.calibrated_result, self.uq_method, report=True, 
                    verbose=True, calibration_mode=True, 
                    calibration_type=self.calibration_type,
                    recal_model=self.recal_model)
            if self.uq_method == 'deterministic':
                adjust_model_name(all_checkpoints=self.all_checkpoints)
                
    
    # A method to select best experiment, and get its associated predictions
    def select_best(self):
        if self.split == 'holdout':
            # initilize a dictionary to collect results for all experiments
            hpo_results = {'exp_id': []}
            first_experiment = self.experiments[0]
            extracted_keys = first_experiment.keys()
            for key in extracted_keys:
                hpo_results[key] = []
            additional_keys = ['mae', 'rmse', 'nll', 'crps', 'sharp']
            for key in additional_keys:
                hpo_results[key] = []
            for exp_id, experiment in enumerate(self.experiments):
                hpo_results['exp_id'].append(exp_id+1)
                for key in extracted_keys:
                    hpo_results[key].append(experiment.get(key))
                # call UQ evaluation without report option.
                uq_metrics = uq_eval(self.all_val_results[exp_id],
                                     self.uq_method)                
                hpo_results['mae'].append(
                    uq_metrics.get('accuracy').get('mae'))
                hpo_results['rmse'].append(
                    uq_metrics.get('accuracy').get('rmse'))
                hpo_results['nll'].append(
                    uq_metrics.get('scoring_rule').get('nll'))
                hpo_results['crps'].append(
                    uq_metrics.get('scoring_rule').get('crps'))
                hpo_results['sharp'].append(
                    uq_metrics.get('sharpness').get('sharp'))
            hpo_df = pd.DataFrame(hpo_results)
            csv_filename = os.path.join(
                self.result_path,
                '{}_{}_seed_{}_hpo_result_.csv'.format(
                    self.uq_method, self.split, self.seed))            
            hpo_df.to_csv(csv_filename, index=False)
            self.min_exp_id = hpo_df[self.HPO_metric].idxmin()
            best_exp_str = f'_exp_{self.min_exp_id+1}_'
            print('Best hyper-parameter configuration is selected.')
            for i, file_path in enumerate(self.all_reports):
                if i != self.min_exp_id:
                    os.remove(file_path) 
            # update list of remaining checkpoints
            self.all_reports = [
                file_path for file_path in self.all_reports 
                if best_exp_str in file_path]
            for file_path in self.all_checkpoints:
                if best_exp_str not in file_path:
                    os.remove(file_path)
            # update list of remaining checkpoints
            self.all_checkpoints = [
                file_path for file_path in self.all_checkpoints 
                if best_exp_str in file_path]
            for i, file_path in enumerate(self.all_val_results):
                if i != self.min_exp_id:
                    os.remove(file_path)
            if self.bootstrapping:
                # remove results for test sets on inferior experiments
                for i, file_path in enumerate(self.all_test_results):
                    if i != self.min_exp_id:
                        os.remove(file_path)
                final_result = self.all_test_results[self.min_exp_id]
            else:
                # inferece on test for best hyper-parameter configuration
                report_path = os.path.join(
                    self.result_path,
                    '{}_{}_seed_{}_exp_{}_report_.txt'.format(
                    self.uq_method, self.split, self.seed, self.min_exp_id+1))   
                if self.ensemble_mode:                  
                    final_result = test_model(
                        models=self.models,
                        uq_method=self.uq_method,           
                        test_loader=self.test_loader,
                        test_original_lengths=self.test_lengths,
                        y_scaler=self.max_train_val,
                        processed_data_path= self.result_path,                        
                        report_path=report_path, seed=self.seed,
                        device=self.device, normalization=self.normalization,
                        ensemble_mode=True, ensemble_size=self.experiments[
                            self.min_exp_id].get('num_models'),
                        exp_id=self.min_exp_id+1)
                elif (self.uq_method == 'DA' or self.uq_method == 'DA_A' or 
                      self.uq_method == 'CDA' or self.uq_method == 'CDA_A' or 
                      self.uq_method == 'mve'):
                    final_result = test_model(
                        model=self.model, uq_method=self.uq_method, 
                        num_mc_samples=self.num_mcmc, 
                        test_loader=self.test_loader,
                        test_original_lengths=self.test_lengths,
                        y_scaler=self.max_train_val,
                        processed_data_path= self.result_path,
                        report_path=report_path, seed=self.seed, 
                        device=self.device, normalization=self.normalization,
                        exp_id=self.min_exp_id+1) 
                elif self.uq_method == 'SQR':
                    final_result = test_model(
                        model=self.model, uq_method=self.uq_method, 
                        num_mc_samples=self.num_mcmc, 
                        test_loader=self.test_loader,
                        test_original_lengths=self.test_lengths,
                        y_scaler=self.max_train_val,
                        processed_data_path= self.result_path,
                        report_path=report_path, seed=self.seed, 
                        device=self.device, normalization=self.normalization,
                        sqr_q=self.experiments[self.min_exp_id].get('tau'),
                        sqr_factor=self.experiments[self.min_exp_id].get(
                            'sqr_factor'), exp_id=self.min_exp_id+1)  
                elif self.uq_method == 'deterministic':
                    final_result =  test_model(
                        model=self.model, uq_method=self.uq_method, 
                        num_mc_samples=self.num_mcmc, 
                        test_loader=self.test_loader,
                        test_original_lengths=self.test_lengths,
                        y_scaler=self.max_train_val,
                        processed_data_path= self.result_path,
                        report_path=report_path, seed=self.seed,
                        device=self.device, normalization=self.normalization,
                        exp_id=self.min_exp_id+1, deterministic=True, 
                        std_mean_ratio=self.experiments[
                            self.min_exp_id].get('std_mean_ratio'))
                elif self.uq_method == 'RF':
                    final_result = predict_rf(
                        model_arch=self.model, val_loader=self.val_loader,
                        test_loader=self.test_loader, 
                        test_original_lengths=self.test_lengths, 
                        y_scaler=self.max_train_val,
                        normalization=self.normalization,
                        report_path=report_path, 
                        result_path=self.result_path, seed=self.seed, 
                        device=self.device, exp_id=self.min_exp_id+1,
                        experiment=self.experiments[self.min_exp_id],
                        cfg=self.cfg, dataset_path=self.dataset_path,
                        y_val_path=self.y_val_path)
                elif self.uq_method == 'LA':
                    final_result = inference_laplace(
                        model=self.model, test_loader=self.test_loader, 
                        test_original_lengths=self.test_lengths,
                        y_train_val=self.y_train_val,
                        y_scaler=self.max_train_val, 
                        normalization=self.normalization,
                        last_layer_name=self.last_layer_name,
                        hessian_structure= self.experiments[
                            self.min_exp_id].get('hessian_structure'),
                        sigma_noise= self.experiments[
                            self.min_exp_id].get('sigma_noise'),
                        stat_noise=self.experiments[
                            self.min_exp_id].get('stat_noise'),
                        prior_precision=self.experiments[
                            self.min_exp_id].get('prior_precision'),
                        temperature=self.experiments[
                            self.min_exp_id].get('temperature'),
                        pred_type='glm',                      
                        report_path=self.report_path,
                        result_path=self.result_path, seed=self.seed,
                        device=self.device, exp_id=self.min_exp_id+1)
 
        else:
            #TODO: implement best parameter selection for cross-fold validation
            print('not implemented!')
            
        final_name = os.path.basename(final_result)
        final_val_name = add_suffix_to_csv(final_name, added_suffix='validation_')
        final_val_path = os.path.join(self.result_path, final_val_name)
        return final_result, final_val_path
    
    
    def laplace_pipeline(self, exp_id, experiment):
        # get hyperparameters     
        self.hessian_structure = experiment.get('hessian_structure')
        self.empirical_bayes = experiment.get('empirical_bayes')
        self.method = experiment.get('method')
        self.grid_size = experiment.get('grid_size')
        self.la_epochs = experiment.get('la_epochs')
        self.la_lr = experiment.get('la_lr')
        self.sigma_noise = experiment.get('sigma_noise')
        self.stat_noise = experiment.get('stat_noise')
        self.prior_precision = experiment.get('prior_precision')
        self.temperature = experiment.get('temperature')
        self.n_samples = experiment.get('n_samples')        
        
        la_model, la_path = post_hoc_laplace(
            model=self.model, cfg=self.cfg, y_train_val=self.y_train_val, 
            train_loader=self.train_loader, val_loader=self.val_loader,
            train_val_loader=self.train_val_loader, 
            last_layer_name=self.last_layer_name, 
            hessian_structure=self.hessian_structure,
            empirical_bayes=self.empirical_bayes,
            la_epochs=self.la_epochs, la_lr=self.la_lr, 
            sigma_noise=self.sigma_noise, stat_noise=self.stat_noise, 
            prior_precision=self.prior_precision, temperature=self.temperature,                      
            n_samples=self.n_samples, report_path=self.report_path,
            result_path=self.result_path, seed=self.seed,
            device=self.device, exp_id=exp_id+1)
        self.all_checkpoints.append(la_path)
        
        res_df = inference_laplace(
            la=la_model, val_mode=True, test_loader=self.val_loader, 
            y_scaler=self.max_train_val, normalization=self.normalization,
            report_path=self.report_path, result_path=self.result_path,
            seed=self.seed, device=self.device, exp_id=exp_id+1)
        self.all_val_results.append(res_df)

    
    def aux_forest(self, exp_id, experiment):
        # get hyperparameters     
        self.criterion = experiment.get('criterion')
        self.n_estimators = experiment.get('n_estimators')
        self.depth_control = experiment.get('depth_control') 
        self.max_depth= experiment.get('max_depth')
        # fit the random forest auxiliary model
        res_model, aux_model, aux_model_path = fit_rf(
            model=self.model, cfg=self.cfg, val_loader=self.val_loader,
            criterion=self.criterion, n_estimators=self.n_estimators,
            depth_control=self.depth_control, max_depth=self.max_depth, 
            dataset_path=self.dataset_path, result_path=self.result_path,
            y_val_path=self.y_val_path, report_path=self.report_path, 
            seed=self.seed, device=self.device, exp_id=exp_id+1)
        self.all_checkpoints.append(aux_model_path) 
        # get the results on validation set, to later be used in HPO
        res_df = predict_rf(
            model=res_model, aux_model=aux_model, val_mode=True, 
            test_loader=self.val_loader, y_scaler=self.max_train_val,
            normalization=self.normalization, report_path=self.report_path,
            result_path=self.result_path, seed=self.seed, device=self.device,
            exp_id=exp_id+1, cfg=self.cfg)
        self.all_val_results.append(res_df)
        
    
    def baseline_approach(self, exp_id, experiment):
        if (self.uq_method == 'DA' or self.uq_method == 'DA_A' or 
            self.uq_method == 'CDA' or self.uq_method == 'CDA_A' or 
            self.uq_method == 'mve'):
            # get early-stopping condition, and number of Monte Carlo samples
            self.num_mcmc = experiment.get('num_mcmc') 
            self.early_stop = experiment.get('early_stop')
        elif self.uq_method == 'deterministic':
            self.deterministic = experiment.get('deterministic')
            self.early_stop = experiment.get('early_stop')
        elif self.uq_method == 'SQR':
            self.sqr_q = experiment.get('tau')
            self.max_epochs = experiment.get('max_epochs')
            self.sqr_factor = experiment.get('sqr_factor')
        res_model = train_model(
            model=self.model, uq_method=self.uq_method,
            train_loader=self.train_loader, val_loader=self.val_loader,
            train_val_loader=self.train_val_loader,
            criterion=self.criterion, optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device, num_epochs=self.max_epochs,
            early_patience=self.early_stop_patience, 
            min_delta=self.early_stop_min_delta, early_stop=self.early_stop,
            processed_data_path=self.result_path, report_path=self.report_path,
            data_split='holdout', cfg=self.cfg, seed=self.seed,
            ensemble_mode=self.ensemble_mode, sqr_q=self.sqr_q,
            sqr_factor=self.sqr_factor, exp_id=exp_id+1)
        self.all_checkpoints.append(res_model)
        if self.uq_method != 'deterministic':
            # inference on validation set
            res_df = test_model(
                model=self.model, uq_method=self.uq_method, 
                num_mc_samples=self.num_mcmc, test_loader=self.val_loader,
                y_scaler=self.max_train_val,
                processed_data_path= self.result_path,
                report_path=self.report_path, val_mode=True,
                seed=self.seed, device=self.device, sqr_q=self.sqr_q,
                sqr_factor=self.sqr_factor, normalization=self.normalization,
                exp_id=exp_id+1)                                  
            self.all_val_results.append(res_df)
        else:
            ratio_exp = self.cfg.get('std_mean_ratio')
            num_ratio_exp = len(self.cfg.get('std_mean_ratio'))
            # create multiple copies from the only report, and checkpoint
            copy_lists = [self.all_reports, self.all_checkpoints]
            lists_to_update = [self.all_reports, self.all_checkpoints]
            for list_id in range (2):
                original_file = lists_to_update[list_id][0]
                directory = os.path.dirname(original_file)
                base_filename = os.path.basename(original_file)
                base_name, extension = base_filename.split('exp_1')  
                for j in range(2, num_ratio_exp + 1):
                    new_filename = f"{base_name}exp_{j}{extension}"
                    new_path = os.path.join(directory, new_filename)
                    shutil.copy(original_file, new_path)
                    copy_lists[list_id].append(new_path)
            self.all_reports = copy_lists[0]
            self.all_checkpoints = copy_lists[1]
                    
            # update self.experiment dictionary
            self.expanded_experiments = []
            for current_exp in self.experiments:
                for current_ratio in ratio_exp:
                    # Create a new experiment with 'std_mean_ratio' added
                    new_exp = current_exp.copy()
                    new_exp['std_mean_ratio'] = current_ratio
                    self.expanded_experiments.append(new_exp)
            for experiment_id, new_experiment in enumerate(
                    self.expanded_experiments):
                res_df = test_model(
                    model=self.model, uq_method=self.uq_method, 
                    num_mc_samples=self.num_mcmc,
                    test_loader=self.val_loader,
                    y_scaler=self.max_train_val,
                    processed_data_path= self.result_path,
                    report_path=self.report_path, val_mode=True,
                    seed=self.seed, device=self.device,
                    normalization=self.normalization, exp_id=experiment_id+1,
                    deterministic=True, 
                    std_mean_ratio=new_experiment.get('std_mean_ratio'))
                self.all_val_results.append(res_df)
    
    # A method to conduct train and inference with ensembles
    def ensemble(self, exp_id, experiment):
        # get number of ensemble models, and bootstrpping ratio
        self.num_models = experiment.get('num_models')
        self.Bootstrapping_ratio = experiment.get(
            'Bootstrapping_ratio')                        
        # get random state (before subset selction for Bootstrapping)
        original_rng_state = torch.get_rng_state()                       
        for i in range(1, self.num_models+1):
            # load relevant data for Bootstrapping ensemble
            if self.bootstrapping:
                # Set a unique seed to select a subset of data
                unique_seed = i + 100  
                torch.manual_seed(unique_seed)
                (self.train_loader, self.val_loader, self.test_loader,
                 self.train_val_loader, self.y_train_val, self.test_lengths
                 ) = self.load_data() 
                res_model = train_model(
                    model=self.models[i-1],
                    uq_method=self.uq_method,
                    train_loader=self.train_loader,
                    val_loader=self.val_loader,
                    criterion=self.criterion,
                    optimizer=self.optimizers[i-1],
                    scheduler=self.schedulers[i-1],
                    device=self.device,
                    num_epochs=self.max_epochs,
                    early_patience=self.early_stop_patience,
                    min_delta=self.early_stop_min_delta, 
                    early_stop=self.early_stop,
                    processed_data_path=self.result_path,
                    report_path=self.report_path,
                    cfg=self.cfg,
                    seed=self.seed,
                    model_idx=i,
                    ensemble_mode=self.ensemble_mode,
                    sqr_q=self.sqr_q,
                    sqr_factor=self.sqr_factor,
                    exp_id=exp_id+1)                     
            else:
                # train a member of traditional ensemble (with retraining) 
                res_model = train_model(
                    model=self.models[i-1],
                    uq_method=self.uq_method,
                    train_loader=self.train_loader,
                    val_loader=self.val_loader,
                    train_val_loader=self.train_val_loader,
                    criterion=self.criterion,
                    optimizer=self.optimizers[i-1],
                    scheduler=self.schedulers[i-1],
                    device=self.device,
                    num_epochs=self.max_epochs,
                    early_patience=self.early_stop_patience,
                    min_delta=self.early_stop_min_delta, 
                    early_stop=self.early_stop,
                    processed_data_path=self.result_path,
                    report_path=self.report_path,
                    cfg=self.cfg,
                    seed=self.seed,
                    model_idx=i,
                    ensemble_mode=self.ensemble_mode,
                    sqr_q=self.sqr_q,
                    sqr_factor=self.sqr_factor,
                    exp_id=exp_id+1)
            self.all_checkpoints.append(res_model)
        # Restore the original random state
        torch.set_rng_state(original_rng_state)
        # inference on validation set with all ensemble members
        res_df = test_model(
            models=self.models,
            uq_method=self.uq_method,           
            test_loader=self.val_loader,
            y_scaler=self.max_train_val,
            processed_data_path= self.result_path,
            report_path=self.report_path,
            val_mode=True,
            seed=self.seed,
            device=self.device,
            normalization=self.normalization,
            ensemble_mode=True,
            ensemble_size=self.num_models,
            exp_id=exp_id+1) 
        self.all_val_results.append(res_df)
        # in case of bootstrapping inference should be executed
        # on test set right here, as loaders change in the loop.
        if self.bootstrapping:
            res_df = test_model(
                models=self.models,
                uq_method=self.uq_method,           
                test_loader=self.test_loader,
                test_original_lengths=self.test_lengths,
                y_scaler=self.max_train_val,
                processed_data_path= self.result_path,
                report_path=self.report_path,
                seed=self.seed,
                device=self.device,
                normalization=self.normalization,
                ensemble_mode=True,
                ensemble_size=self.num_models,
                exp_id=exp_id+1)
            self.all_test_results.append(res_df)
    
    # A method to load important dimensions
    def load_dimensions(self):        
        scaler_path = os.path.join(
            self.dataset_path,
            "DALSTM_max_train_val_"+self.dataset+".pkl")
        input_size_path = os.path.join(
            self.dataset_path,
            "DALSTM_input_size_"+self.dataset+".pkl")
        max_len_path = os.path.join(
            self.dataset_path,
            "DALSTM_max_len_"+self.dataset+".pkl")
        # input_size corresponds to vocab_size
        with open(input_size_path, 'rb') as f:
            input_size =  pickle.load(f)
        with open(max_len_path, 'rb') as f:
            max_len =  pickle.load(f) 
        with open(scaler_path, 'rb') as f:
            max_train_val =  pickle.load(f)            
        return (input_size, max_len, max_train_val)
        
    # A method to create list of path for holdout data split
    def holdout_paths(self):
        X_train_path = os.path.join(
            self.dataset_path, "DALSTM_X_train_"+self.dataset+".pt")
        X_val_path = os.path.join(
            self.dataset_path, "DALSTM_X_val_"+self.dataset+".pt")
        X_test_path = os.path.join(
            self.dataset_path, "DALSTM_X_test_"+self.dataset+".pt")
        y_train_path = os.path.join(
            self.dataset_path, "DALSTM_y_train_"+self.dataset+".pt")
        y_val_path = os.path.join(
            self.dataset_path, "DALSTM_y_val_"+self.dataset+".pt")
        y_test_path = os.path.join(
            self.dataset_path, "DALSTM_y_test_"+self.dataset+".pt")
        test_length_path = os.path.join(
            self.dataset_path, "DALSTM_test_length_list_"+self.dataset+".pkl")            
        return (X_train_path, X_val_path, X_test_path, y_train_path,
                y_val_path, y_test_path, test_length_path)
    
        
    # A method to create list of path for holdout data split
    def cv_paths(self, split_key=None):
        X_train_path = os.path.join(
            self.dataset_path,
            "DALSTM_X_train_fold_"+str(split_key+1)+self.dataset+".pt")
        X_val_path = os.path.join(
            self.dataset_path,
            "DALSTM_X_val_fold_"+str(split_key+1)+self.dataset+".pt")
        X_test_path = os.path.join(
            self.dataset_path,
            "DALSTM_X_test_fold_"+str(split_key+1)+self.dataset+".pt")
        y_train_path = os.path.join(
            self.dataset_path,
            "DALSTM_y_train_fold_"+str(split_key+1)+self.dataset+".pt")
        y_val_path = os.path.join(
            self.dataset_path,
            "DALSTM_y_val_fold_"+str(split_key+1)+self.dataset+".pt")
        y_test_path = os.path.join(
            self.dataset_path,
            "DALSTM_y_test_fold_"+str(split_key+1)+self.dataset+".pt") 
        test_length_path = os.path.join(
            self.dataset_path,
            "DALSTM_test_length_list_fold_"+str(split_key+1)+self.dataset+".pkl")
        return (X_train_path, X_val_path, X_test_path, y_train_path,
                y_val_path, y_test_path,test_length_path)
        
    # A method to load training and evaluation 
    def load_data(self):
        X_train = torch.load(self.X_train_path, weights_only=True)
        X_val = torch.load(self.X_val_path, weights_only=True)
        X_test = torch.load(self.X_test_path, weights_only=True)
        y_train = torch.load(self.y_train_path, weights_only=True)
        y_val = torch.load(self.y_val_path, weights_only=True)
        y_test = torch.load(self.y_test_path, weights_only=True)
        if self.laplace:
            y_train = y_train.unsqueeze(dim=1)
            y_val = y_val.unsqueeze(dim=1)
            y_test = y_test.unsqueeze(dim=1)
        # define training dataset
        if self.bootstrapping:
            subset_size = int(X_train.size(0) * self.Bootstrapping_ratio)
            subset_indices = torch.randint(0, X_train.size(0), (subset_size,))
            #print(subset_indices)
            X_sample = X_train[subset_indices]
            y_sample = y_train[subset_indices]
            train_dataset = TensorDataset(X_sample, y_sample)
        else:
            train_dataset = TensorDataset(X_train, y_train)            
        # define validation and test datasets              
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        """
        Define the batch size: check cfg, if there is an explicit size
        there, we use it. Otherwise, we use max_len as the batch size
        similar to the original paper (Navarin et al.)
        """
        try:
            batch_size = self.cfg.get('train').get('batch_size')
        except:
            batch_size = self.max_len  
        try:
            evaluation_batch_size = self.cfg.get('evaluation').get('batch_size')
        except:
            evaluation_batch_size = self.max_len                    
        # define training, validation, test data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=evaluation_batch_size,
                                 shuffle=False)
        with open(self.test_lengths_path, 'rb') as f:
            test_lengths =  pickle.load(f)
        train_val_dataset = ConcatDataset([train_dataset, val_dataset])
        train_val_loader = DataLoader(train_val_dataset, batch_size=batch_size,
                                      shuffle=True)
        if not self.laplace:
            y_list = [y.unsqueeze(0) for _, y in train_val_dataset]
            y_train_val = torch.cat(y_list, dim=0)
        else:
            y_train_val = torch.cat([y for _, y in train_val_dataset], dim=0)
        y_train_val = y_train_val.numpy()
        return (train_loader, val_loader, test_loader, train_val_loader,
                y_train_val, test_lengths)    