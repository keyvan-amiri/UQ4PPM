import torch
from torch.utils.data import TensorDataset, DataLoader
import os
import pickle
from models.Laplace_approximation import post_hoc_laplace
from models.Random_Forest import fit_rf, predict_rf
from models.Train import train_model
from models.Test import test_model
from loss.loss_handler import set_loss
from loss.QuantileLoss import QuantileLoss
from utils.utils import get_model, get_optimizer_scheduler, set_random_seed


# A generic class for training and evaluation of DALSTM model
class DALSTM_train_evaluate ():
    def __init__ (self, cfg=None, dalstm_dir=None):  
        """
        Parameters:
        cfg : configuration that is used for training, and inference.
        dalstm_dir : user specified for dataset folder which contain all
        information about feature vectors represinting event prefixes, and 
        used by DALSTM model.
        """
        #  Initial operations for effective training, and inference
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
        # set normalization status
        self.normalization = cfg.get('data').get('normalization')
        # get the specific uncertainty quanitification defined in command line
        self.uq_method = cfg.get('uq_method')
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
        self.max_epochs = cfg.get('train').get('max_epochs')
        self.early_stop = cfg.get('train').get('early_stop')
        self.early_stop_patience = cfg.get('train').get('early_stop.patience')
        self.early_stop_min_delta = cfg.get('train').get('early_stop.min_delta')
        
        ######################################################################
        ########################   ensemble setting   ########################
        ######################################################################
        if (self.uq_method == 'en_b' or self.uq_method == 'en_b_mve' or
            self.uq_method == 'en_t' or self.uq_method == 'en_t_mve'):
            self.ensemble_mode = True
            # get number of ensembles: list for HPO, otherwise an ineger
            self.hpo1 = cfg.get('uncertainty').get('ensemble').get('num_models')
            if isinstance(self.hpo1, list):
                self.exp_count = len(self.hpo1)
                self.exp_dict = []
                for n_model in self.hpo1:
                    self.exp_dict.append({'num_models':n_model})
            elif isinstance(self.hpo1, int):
                self.exp_count = 1
                self.num_models = self.hpo1
            else:
                raise ValueError('number of models should be integer or list')
            # get Bootstrapping ratio which is used by each ensemble member
            self.Bootstrapping_ratio = cfg.get('uncertainty').get(
                'ensemble').get('Bootstrapping_ratio')
            if (self.uq_method == 'en_b' or self.uq_method == 'en_b_mve'):
                self.bootstrapping = True
            else:
                self.bootstrapping = False
        else:
            self.ensemble_mode = False
            self.bootstrapping = False

        
        # define parameters required for dropout approximation
        if (self.uq_method == 'DA' or self.uq_method == 'CDA' or
              self.uq_method == 'DA_A' or self.uq_method == 'CDA_A'):
            self.num_mcmc = cfg.get('uncertainty').get(
                'dropout_approximation').get('num_stochastic_forward_path')
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
            self.num_mcmc = None 
            self.concrete_dropout = None
            self.weight_regularizer = None
            self.dropout_regularizer = None
            self.Bayes = None
        

        # set relevant parameters for embedding-based UQ
        if self.uq_method == 'RF':
            self.union_mode = True
        else:
            self.union_mode = False
            
        # set necessary parameters for Laplace approximation
        if (self.uq_method == 'LA' or self.uq_method == 'LA_H'):
            self.laplace = True
            self.link_approx = 'mc'
            self.pred_type = 'glm'
            if self.uq_method == 'LA':
                # We only use Laplace approximation which can be applied in 
                # an architecture-agnostic fashion.
                self.subset_of_weights= 'last_layer'
                self.last_layer_name = cfg.get('uncertainty').get(
                    'laplace').get('last_layer_name')
                self.module_names=None
                self.heteroscedastic = False
            else:
                # In case of Heteroscedastic regression, we apply Laplace
                # approximation to last linear layers for mean, log variance
                self.subset_of_weights= 'subnetwork' 
                self.last_layer_name = None
                self.module_names = cfg.get('uncertainty').get(
                                    'laplace').get('module_names')
                self.heteroscedastic = True
            # whether to estimate the prior precision and observation noise 
            # using empirical Bayes after training or not
            self.empirical_bayes = cfg.get('uncertainty').get('laplace').get(
                'empirical_bayes')
            self.la_epochs = cfg.get('uncertainty').get('laplace').get('epochs')
            self.la_lr = cfg.get('uncertainty').get('laplace').get('lr')
            self.hessian_structure = cfg.get('uncertainty').get(
                'laplace').get('hessian_structure')
            self.n_samples = cfg.get('uncertainty').get('laplace').get(
                'n_samples')
            self.sigma_noise = cfg.get('uncertainty').get('laplace').get(
                'sigma_noise')
            self.prior_precision = cfg.get('uncertainty').get('laplace').get(
                'prior_precision')
            self.temperature= cfg.get('uncertainty').get('laplace').get(
                'temperature')
            # method for prior precision optimization
            self.method = cfg.get('uncertainty').get('laplace').get('prior_opt')
            self.grid_size = cfg.get('uncertainty').get('laplace').get('grid_size') 
            self.stat_noise= cfg.get('uncertainty').get('laplace').get('stat_noise')
        else:
            self.laplace = False
            
        # set necessary parameters for Simultaneous Quantile Regression
        # NOTE: to use the same execution path for other methods as well
        try:
            self.sqr_factor = cfg.get('uncertainty').get('sqr').get(
                'scaling_factor')
            self.confidence_level = cfg.get('uncertainty').get('sqr').get(
                'confidence_level')
            self.sqr_q = cfg.get('uncertainty').get('sqr').get('tau')
        except:
            self.confidence_level = 0.95
            self.sqr_factor = 12
            self.sqr_q = 'all'        
        

        # define loss function (heteroscedastic/homoscedastic)
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
        elif self.uq_method == 'RF':
            # define loss funciton for fitting the auxiliary model
            self.criterion = cfg.get('uncertainty').get('union').get(
                'loss_function')
        elif self.uq_method == 'SQR':
            self.criterion = QuantileLoss()
        
        # define model, scheduler, and optimizer
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
        else:
            if self.exp_count == 1:
                self.models = get_model(
                    uq_method=self.uq_method, input_size=self.input_size,
                    hidden_size=self.hidden_size, n_layers=self.n_layers,
                    max_len=self.max_len, dropout=self.dropout,
                    dropout_prob=self.dropout_prob, num_models=self.num_models,
                    concrete_dropout=self.concrete_dropout,
                    weight_regularizer=self.weight_regularizer,
                    dropout_regularizer=self.dropout_regularizer,
                    Bayes=self.Bayes, device=self.device)
                self.optimizers, self.schedulers = get_optimizer_scheduler(
                    models=self.models, cfg=self.cfg, ensemble_mode=True,
                    num_models=self.num_models) 
            else:
                self.models = get_model(
                    uq_method=self.uq_method, input_size=self.input_size,
                    hidden_size=self.hidden_size, n_layers=self.n_layers,
                    max_len=self.max_len, dropout=self.dropout,
                    dropout_prob=self.dropout_prob, num_models=max(self.hpo1),
                    concrete_dropout=self.concrete_dropout,
                    weight_regularizer=self.weight_regularizer,
                    dropout_regularizer=self.dropout_regularizer,
                    Bayes=self.Bayes, device=self.device)
                self.optimizers, self.schedulers = get_optimizer_scheduler(
                    models=self.models, cfg=self.cfg, ensemble_mode=True,
                    num_models=max(self.hpo1)) 
                
      
        # execute training and evaluation loop
        for execution_seed in seeds:            
            # set random seed
            self.seed = int(execution_seed) 
            set_random_seed(self.seed)
            self.all_checkpoints = []
            self.all_val_results = []
            self.all_reports = []
            for exp_id in range (1, self.exp_count+1):
                
                # train-test pipeline for holdout data split
                if self.split == 'holdout':
                    # define the report path
                    self.report_path = os.path.join(
                        self.result_path,
                        '{}_{}_seed_{}_exp_{}_report_.txt'.format(
                            self.uq_method, self.split, self.seed, exp_id)) 
                    self.all_reports.append(self.report_path)
                    # get paths for processed data
                    (self.X_train_path, self.X_val_path, self.X_test_path,
                     self.y_train_path, self.y_val_path, self.y_test_path,
                     self.test_lengths_path) = self.holdout_paths()
                    # load train, validation, and test data loaders
                    if not self.bootstrapping:
                        #except for Bootstrapping ensemble
                        (self.train_loader, self.val_loader, self.test_loader,
                         self.test_lengths) = self.load_data()
                    # execution path for dropout and heteroscedastic
                    if ((not self.ensemble_mode) and (not self.union_mode) and
                        (not self.laplace)):         
                        train_model(model=self.model, uq_method=self.uq_method,
                                    train_loader=self.train_loader,
                                    val_loader=self.val_loader,
                                    criterion=self.criterion,
                                    optimizer=self.optimizer,
                                    scheduler=self.scheduler,
                                    device=self.device,
                                    num_epochs=self.max_epochs,
                                    early_patience=self.early_stop_patience,
                                    min_delta=self.early_stop_min_delta, 
                                    early_stop=self.early_stop,
                                    processed_data_path=self.result_path,
                                    report_path=self.report_path,
                                    data_split='holdout',
                                    cfg=self.cfg,
                                    seed=self.seed,
                                    ensemble_mode=self.ensemble_mode,
                                    sqr_q=self.sqr_q,
                                    sqr_factor=self.sqr_factor)   
                        test_model(model=self.model, uq_method=self.uq_method,
                                   num_mc_samples=self.num_mcmc,              
                                   test_loader=self.test_loader,
                                   test_original_lengths=self.test_lengths,
                                   y_scaler=self.max_train_val,
                                   processed_data_path= self.result_path,
                                   report_path=self.report_path,
                                   data_split = 'holdout',
                                   seed=self.seed,
                                   device=self.device,
                                   normalization=self.normalization,
                                   confidence_level=self.confidence_level,
                                   sqr_factor=self.sqr_factor)
                    elif self.ensemble_mode:
                        # if there are ensemble of models to train
                        # get random state (before subset selction for Bootstrapping)
                        original_rng_state = torch.get_rng_state()
                        if self.exp_count > 1:
                            self.num_models = self.hpo1[exp_id-1]
                        for i in range(1, self.num_models+1):
                            # load relevant data for Bootstrapping ensemble
                            if self.bootstrapping:
                                # Set a unique seed for selection of subset data
                                unique_seed = i + 100  
                                torch.manual_seed(unique_seed)
                                (self.train_loader,self.val_loader,
                                 self.test_loader, self.test_lengths
                                 ) = self.load_data()                           
                            # train a member of ensemble    
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
                                data_split='holdout',
                                cfg=self.cfg,
                                seed=self.seed,
                                model_idx=i,
                                ensemble_mode=self.ensemble_mode,
                                sqr_q=self.sqr_q,
                                sqr_factor=self.sqr_factor,
                                exp_id=exp_id)
                            self.all_checkpoints.append(res_model)
                        # Restore the original random state
                        torch.set_rng_state(original_rng_state)
                        # inference with all ensemble members
                        res_df = test_model(
                            models=self.models,
                            uq_method=self.uq_method,
                            num_mc_samples=self.num_mcmc,              
                            test_loader=self.test_loader,
                            test_original_lengths=self.test_lengths,
                            y_scaler=self.max_train_val,
                            processed_data_path= self.result_path,
                            report_path=self.report_path,
                            data_split = 'holdout',
                            seed=self.seed,
                            device=self.device,
                            normalization=self.normalization,
                            ensemble_mode=self.ensemble_mode,
                            ensemble_size=self.num_models,
                            confidence_level=self.confidence_level,
                            sqr_factor=self.sqr_factor,
                            exp_id=exp_id) 
                        self.all_val_results.append(res_df)
                        
                    elif self.union_mode:                    
                        self.model, self.aux_model = fit_rf(
                            model=self.model, cfg=self.cfg,
                            criterion=self.criterion,
                            val_loader=self.val_loader,
                            dataset_path=self.dataset_path,
                            result_path=self.result_path,
                            y_val_path=self.y_val_path,
                            report_path=self.report_path,
                            split=self.split, seed=self.seed, device=self.device)
                        predict_rf(
                            model=self.model, aux_model=self.aux_model,
                            test_loader=self.test_loader,
                            test_original_lengths=self.test_lengths,
                            y_scaler=self.max_train_val,
                            normalization=self.normalization,
                            report_path=self.report_path,
                            result_path=self.result_path,
                            split=self.split, seed=self.seed, device=self.device) 
                    # execution path for post-hoc Laplace approximation
                    elif self.laplace:
                        post_hoc_laplace(
                            model=self.model, cfg=self.cfg, 
                            X_train_path=self.X_train_path, 
                            X_val_path=self.X_val_path,
                            X_test_path=self.X_test_path,
                            y_train_path=self.y_train_path,
                            y_val_path=self.y_val_path, 
                            y_test_path=self.y_test_path,
                            test_original_lengths=self.test_lengths,
                            y_scaler=self.max_train_val,
                            normalization=self.normalization,
                            subset_of_weights=self.subset_of_weights,
                            hessian_structure=self.hessian_structure,
                            empirical_bayes=self.empirical_bayes,
                            method=self.method, grid_size=self.grid_size,
                            last_layer_name=self.last_layer_name,
                            module_names=self.module_names,
                            sigma_noise=self.sigma_noise, 
                            stat_noise=self.stat_noise,
                            prior_precision=self.prior_precision,
                            temperature=self.temperature,
                            n_samples=self.n_samples, link_approx=self.link_approx,
                            pred_type=self.pred_type,
                            la_epochs=self.la_epochs, la_lr=self.la_lr,
                            heteroscedastic=self.heteroscedastic,
                            report_path=self.report_path,
                            result_path=self.result_path,
                            split=self.split, seed=self.seed, device=self.device)
                    
                # train-test pipeline for cross=validation data split          
                else:
                    for split_key in range(self.n_splits):
                        # define the report path
                        self.report_path = os.path.join(
                            self.result_path,
                            '{}_{}_fold{}_seed_{}_exp_{}_report_.txt'.format(
                                self.uq_method, self.split, split_key+1,
                                self.seed, exp_id))
                        # load train, validation, and test data loaders
                        (self.X_train_path, self.X_val_path, self.X_test_path,
                         self.y_train_path, self.y_val_path, self.y_test_path,
                         self.test_lengths_path) = self.cv_paths(
                             split_key=split_key)
                        # except for Bootstrapping ensemble
                        if not self.bootstrapping:
                            (self.train_loader, self.val_loader,
                             self.test_loader, self.test_lengths
                             ) = self.load_data()
                        if ((not self.ensemble_mode) and (not self.union_mode)):                    
                            train_model(model=self.model,
                                        uq_method=self.uq_method,
                                        train_loader=self.train_loader,
                                        val_loader=self.val_loader,
                                        criterion=self.criterion,
                                        optimizer=self.optimizer,
                                        scheduler=self.scheduler,
                                        device=self.device,
                                        num_epochs=self.max_epochs,
                                        early_patience=self.early_stop_patience,
                                        min_delta=self.early_stop_min_delta, 
                                        early_stop=self.early_stop,
                                        processed_data_path=self.result_path,
                                        report_path=self.report_path,
                                        data_split='cv',
                                        fold = split_key+1,
                                        cfg=self.cfg,
                                        seed=self.seed,
                                        ensemble_mode=self.ensemble_mode,
                                        sqr_q=self.sqr_q,
                                        sqr_factor=self.sqr_factor)
                            test_model(model=self.model, 
                                       uq_method=self.uq_method,
                                       num_mc_samples=self.num_mcmc,  
                                       test_loader=self.test_loader,
                                       test_original_lengths=self.test_lengths,
                                       y_scaler=self.max_train_val,
                                       processed_data_path= self.result_path,
                                       report_path=self.report_path,
                                       data_split='cv',
                                       fold = split_key+1,
                                       seed=self.seed,
                                       device=self.device,
                                       normalization=self.normalization,
                                       confidence_level=self.confidence_level,
                                       sqr_factor=self.sqr_factor) 
                        elif self.ensemble_mode:
                            # if there are ensemble of models to train
                            # get random state 
                            # before subset selction Bootstrapping
                            if self.exp_count > 1:
                                self.num_models = self.hpo1[exp_id-1]
                            original_rng_state = torch.get_rng_state()
                            for i in range(1, self.num_models+1):
                                # load relevant data for Bootstrapping ensemble
                                if self.bootstrapping:
                                    # Set a unique seed for selection of subset data
                                    unique_seed = i + 100  
                                    torch.manual_seed(unique_seed)
                                    (self.train_loader, self.val_loader, 
                                     self.test_loader, self.test_lengths
                                     ) = self.load_data() 
                                # train a member of ensemble 
                                train_model(model=self.models[i-1],
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
                                            data_split='cv',
                                            fold = split_key+1,
                                            cfg=self.cfg,
                                            seed=self.seed,
                                            model_idx=i,
                                            ensemble_mode=self.ensemble_mode,
                                            sqr_q=self.sqr_q,
                                            sqr_factor=self.sqr_factor,
                                            exp_id=exp_id)
                            # Restore the original random state
                            torch.set_rng_state(original_rng_state)
                            # inference with all ensemble members
                            test_model(models=self.models,
                                       uq_method=self.uq_method,
                                       num_mc_samples=self.num_mcmc,  
                                       test_loader=self.test_loader,
                                       test_original_lengths=self.test_lengths,
                                       y_scaler=self.max_train_val,
                                       processed_data_path= self.result_path,
                                       report_path=self.report_path,
                                       data_split='cv',
                                       fold = split_key+1,
                                       seed=self.seed,
                                       device=self.device,
                                       normalization=self.normalization,
                                       ensemble_mode=self.ensemble_mode,
                                       ensemble_size=self.num_models,
                                       confidence_level=self.confidence_level,
                                       sqr_factor=self.sqr_factor,
                                       exp_id=exp_id)
                        elif self.union_mode:
                            self.model, self.aux_model = fit_rf(
                                model=self.model, cfg=self.cfg,
                                criterion=self.criterion,
                                val_loader=self.val_loader,
                                dataset_path=self.dataset_path,
                                result_path=self.result_path,
                                y_val_path=self.y_val_path,
                                report_path=self.report_path,
                                split=self.split, fold = split_key+1,
                                seed=self.seed, device=self.device)
                            predict_rf(
                                model=self.model, aux_model=self.aux_model,
                                test_loader=self.test_loader,
                                test_original_lengths=self.test_lengths,
                                y_scaler=self.max_train_val,
                                normalization=self.normalization,
                                report_path=self.report_path,
                                result_path=self.result_path,
                                split=self.split, fold = split_key+1,
                                seed=self.seed, device=self.device) 
                        
                        # execution path for post-hoc Laplace approximation
                        elif self.laplace:
                            post_hoc_laplace(
                                model=self.model, cfg=self.cfg, 
                                X_train_path=self.X_train_path, 
                                X_val_path=self.X_val_path,
                                X_test_path=self.X_test_path,
                                y_train_path=self.y_train_path,
                                y_val_path=self.y_val_path, 
                                y_test_path=self.y_test_path,
                                test_original_lengths=self.test_lengths,
                                y_scaler=self.max_train_val,
                                normalization=self.normalization,
                                subset_of_weights=self.subset_of_weights,
                                hessian_structure=self.hessian_structure,
                                empirical_bayes=self.empirical_bayes,
                                method=self.method, grid_size=self.grid_size,
                                last_layer_name=self.last_layer_name,
                                sigma_noise=self.sigma_noise, 
                                stat_noise=self.stat_noise,
                                prior_precision=self.prior_precision,
                                temperature=self.temperature,
                                n_samples=self.n_samples,
                                link_approx=self.link_approx,
                                pred_type=self.pred_type,
                                la_epochs=self.la_epochs, la_lr=self.la_lr,
                                report_path=self.report_path,
                                result_path=self.result_path,
                                split=self.split, fold = split_key+1,
                                seed=self.seed, device=self.device)                        
                self.select_best()   
    
    # A method to collect all
    def select_best(self):
        if self.split == 'holdout':
            print(self.all_val_results)
            print(self.all_checkpoints)
            print(self.all_reports)
            print(self.exp_dict)
        else:
            #TODO: implement best parameter selection for cross-fold validation
            print('not implemented!')
        return None
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
        X_train = torch.load(self.X_train_path)
        X_val = torch.load(self.X_val_path)
        X_test = torch.load(self.X_test_path)
        y_train = torch.load(self.y_train_path)
        y_val = torch.load(self.y_val_path)
        y_test = torch.load(self.y_test_path)
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
        return (train_loader, val_loader, test_loader, test_lengths) 