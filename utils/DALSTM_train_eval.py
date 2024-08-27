import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from utils.utils import set_random_seed, set_optimizer, train_model, test_model
from loss.loss_handler import set_loss
from models.dalstm import DALSTMModel, DALSTMModelMve, dalstm_init_weights
from models.stochastic_dalstm import StochasticDALSTM

# A generic class for training and evaluation of DALSTM model
class DALSTM_train_evaluate ():    
    def __init__ (self, cfg=None, dalstm_dir=None):
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
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        # set normalization status
        self.normalization = cfg.get('data').get('normalization')
        # get the specific uncertainty quanitification defined in command line
        self.uq_method = cfg.get('uq_method')
        # get the type of data split, and possibly number of splits for CV        
        self.split = cfg.get('split')
        self.n_splits = cfg.get('n_splits') 
        # get the number of ensembles
        self.num_models = cfg.get('num_models')
        # get Bootstrapping ratio which is used by each ensemble member
        self.Bootstrapping_ratio = cfg.get('Bootstrapping_ratio')
        # get type of the probabilistic model to work upon embedding
        self.union = cfg.get('union')
        # Define important size and dimensions
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
        self.early_stop_patience = cfg.get('train').get('early_stop.patience')
        self.early_stop_min_delta = cfg.get('train').get('early_stop.min_delta')
        
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
        
        # set the ensemble mode and bootstrapping parameters
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
        
        # set relevant parameters for embedding-based UQ
        if self.uq_method == 'union':
            self.union_mode = True
            self.n_estimators = (
                cfg.get('uncertainty').get('union').get('n_estimators'))
            depth_control = (
                cfg.get('uncertainty').get('union').get('depth_control'))
            if depth_control:
                self.max_depth = (
                    cfg.get('uncertainty').get('union').get('max_depth'))
            else:
                self.max_depth = None
            self.min_samples_split = (
                cfg.get('uncertainty').get('union').get('min_samples_split'))
            self.min_samples_leaf = (
                cfg.get('uncertainty').get('union').get('min_samples_leaf'))   
        else:
            self.union_mode = False

        ######################################################################
        ######  define loss function (heteroscedastic/homoscedastic)  ########
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
        elif self.uq_method == 'union':
            # define loss funciton for fitting the auxiliary model
            self.criterion = cfg.get('uncertainty').get('union').get('loss_function')
        
        # execute training and evaluation loop
        for execution_seed in seeds:
            
            # set random seed
            self.seed = int(execution_seed) 
            set_random_seed(self.seed) 
            
            ##################################################################
            ###########  define the model (based on the UQ method)  ##########
            ##################################################################
            # deterministic model (point estimate)
            if self.uq_method == 'deterministic':
                self.model = DALSTMModel(
                    input_size=self.input_size,
                    hidden_size=self.hidden_size,
                    n_layers=self.n_layers,
                    max_len=self.max_len,
                    dropout=self.dropout,
                    p_fix=self.dropout_prob).to(self.device) 
            # embedding-based approach
            if self.uq_method == 'union':
                self.model = DALSTMModel(
                    input_size=self.input_size,
                    hidden_size=self.hidden_size,
                    n_layers=self.n_layers,
                    max_len=self.max_len,
                    dropout=self.dropout,
                    p_fix=self.dropout_prob,
                    exclude_last_layer=True).to(self.device) 
            # dropout approximation
            elif (self.uq_method == 'DA' or self.uq_method == 'CDA'):
                self.model = StochasticDALSTM(
                    input_size=self.input_size,
                    hidden_size=self.hidden_size,
                    n_layers=self.n_layers,
                    max_len=self.max_len,
                    dropout=self.dropout,
                    concrete=self.concrete_dropout,
                    p_fix=self.dropout_prob,
                    weight_regularizer=self.weight_regularizer,
                    dropout_regularizer=self.dropout_regularizer,
                    hs=False,
                    Bayes=self.Bayes,
                    device=self.device).to(self.device)
            # dropout approximation with heteroscedastic regression
            elif (self.uq_method == 'DA_A' or self.uq_method == 'CDA_A'):
                self.model = StochasticDALSTM(
                    input_size=self.input_size,
                    hidden_size=self.hidden_size,
                    n_layers=self.n_layers,
                    max_len=self.max_len,
                    dropout=self.dropout,
                    concrete=self.concrete_dropout,
                    p_fix=self.dropout_prob,
                    weight_regularizer=self.weight_regularizer,
                    dropout_regularizer=self.dropout_regularizer,
                    hs=True,
                    Bayes=self.Bayes,
                    device=self.device).to(self.device)
            # heteroscedastic regression also known as mean variance estimation
            elif self.uq_method == 'mve':                
                self.model = DALSTMModelMve(
                    input_size=self.input_size,
                    hidden_size=self.hidden_size,
                    n_layers=self.n_layers,
                    max_len=self.max_len,
                    dropout=self.dropout,
                    p_fix=self.dropout_prob).to(self.device)
            # b: Bootstrapping ensemble: multiple models, same initialization.            
            elif (self.uq_method == 'en_b' or self.uq_method == 'en_b_mve'):
                # empty lists (ensemble of) models, optimizers, schedulers
                self.models, self.optimizers, self.schedulers = [], [], [] 
                for i in range(self.num_models):
                    if self.uq_method == 'en_b':
                        # each ensemble member is a deterministic model
                        model = DALSTMModel(
                            input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            n_layers=self.n_layers,
                            max_len=self.max_len,
                            dropout=self.dropout,
                            p_fix=self.dropout_prob).to(self.device)
                    else:
                        # last layer include log variance estimation
                        model = DALSTMModelMve(
                            input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            n_layers=self.n_layers,
                            max_len=self.max_len,
                            dropout=self.dropout,
                            p_fix=self.dropout_prob).to(self.device)
                    self.models.append(model)        
            # t: Traditional ensemble: multiple models, different initialization.
            elif (self.uq_method == 'en_t' or self.uq_method == 'en_t_mve'):
                # empty lists (ensemble of) models, optimizers, schedulers
                self.models, self.optimizers, self.schedulers = [], [], []      
                # Original random state (before initializing models)
                original_rng_state = torch.get_rng_state()
                for i in range(self.num_models):
                    # Set a unique seed for each model's initialization
                    unique_seed = i + 100  
                    torch.manual_seed(unique_seed)
                    if self.uq_method == 'en_t':
                        # each ensemble member is a deterministic model
                        model = DALSTMModel(
                            input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            n_layers=self.n_layers,
                            max_len=self.max_len,
                            dropout=self.dropout,
                            p_fix=self.dropout_prob).to(self.device)
                    else:
                        # last layer include log variance estimation
                        model = DALSTMModelMve(
                            input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            n_layers=self.n_layers,
                            max_len=self.max_len,
                            dropout=self.dropout,
                            p_fix=self.dropout_prob).to(self.device)                    
                    # Apply weight initialization function
                    model.apply(dalstm_init_weights)
                    self.models.append(model)
                # Restore the original random state
                torch.set_rng_state(original_rng_state)
            
            ##################################################################
            ############   define optimizer(s) & scheduler(s)   ##############
            ##################################################################
            if ((not self.ensemble_mode) and (not self.union_mode)):
                # get number of model parameters
                total_params = sum(p.numel() for p in self.model.parameters()
                                   if p.requires_grad) 
                # define optimizer
                self.optimizer = set_optimizer(
                    self.model, cfg.get('optimizer').get('type'),
                    cfg.get('optimizer').get('base_lr'), 
                    cfg.get('optimizer').get('eps'), 
                    cfg.get('optimizer').get('weight_decay'))
                # define scheduler
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, factor=0.5)  
                print(f'Total model parameters: {total_params}')
            elif self.ensemble_mode:
                # get number of parameters for the first model
                total_params = sum(p.numel() for p in self.models[0].parameters()
                                   if p.requires_grad) 
                # define list of optimizeers and schedulers
                for i in range(self.num_models):
                    current_optimizer = set_optimizer(
                        self.models[i], cfg.get('optimizer').get('type'),
                        cfg.get('optimizer').get('base_lr'),
                        cfg.get('optimizer').get('eps'),
                        cfg.get('optimizer').get('weight_decay'))
                    self.optimizers.append(current_optimizer)
                    current_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        self.optimizers[i], factor=0.5)
                    self.schedulers.append(current_scheduler)                
                print(f'Total model parameters: {total_params}')
        
            # load train, validation, and test data loaders  
            if self.split == 'holdout':
                # define the report path
                self.report_path = os.path.join(
                    self.result_path, '{}_{}_seed_{}_report_.txt'.format(
                        self.uq_method, self.split, self.seed))  
                # get paths for processed data
                (self.X_train_path, self.X_val_path, self.X_test_path,
                 self.y_train_path, self.y_val_path, self.y_test_path,
                 self.test_lengths_path) = self.holdout_paths()
                # except for Bootstrapping ensemble and embedding-based
                if ((not self.bootstrapping) and (not self.union_mode)):
                    (self.train_loader, self.val_loader, self.test_loader,
                     self.test_lengths) = self.load_data()
                # if there is only one model to train (and not embedding-based)
                if ((not self.ensemble_mode) and (not self.union_mode)):         
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
                                processed_data_path=self.result_path,
                                report_path=self.report_path,
                                data_split='holdout',
                                cfg=self.cfg,
                                seed=self.seed,
                                ensemble_mode=self.ensemble_mode)   
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
                               normalization=self.normalization)
                elif self.ensemble_mode:
                    # if there are ensemble of models to train
                    # get random state (before subset selction for Bootstrapping)
                    original_rng_state = torch.get_rng_state()
                    for i in range(1, self.num_models+1):
                        # load relevant data for Bootstrapping ensemble
                        if self.bootstrapping:
                            # Set a unique seed for selection of subset data
                            unique_seed = i + 100  
                            torch.manual_seed(unique_seed)
                            (self.train_loader,self.val_loader,self.test_loader,
                             self.test_lengths) = self.load_data()                           
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
                                    processed_data_path=self.result_path,
                                    report_path=self.report_path,
                                    data_split='holdout',
                                    cfg=self.cfg,
                                    seed=self.seed,
                                    model_idx=i,
                                    ensemble_mode=self.ensemble_mode)
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
                               data_split = 'holdout',
                               seed=self.seed,
                               device=self.device,
                               normalization=self.normalization,
                               ensemble_mode=self.ensemble_mode,
                               ensemble_size=self.num_models)  
                elif self.union_mode:
                    # check deterministic pre-trained model is available                    
                    deterministic_checkpoint_path = os.path.join(
                        self.result_path,
                        'deterministic_{}_seed_{}_best_model.pt'.format(
                            self.split, self.seed)) 
                    if not os.path.isfile(deterministic_checkpoint_path):
                        raise FileNotFoundError(
                            'Deterministic model should be trained in advance')
                    else:
                        # load the checkpoint except the last layer
                        checkpoint = torch.load(deterministic_checkpoint_path)
                        self.model.load_state_dict(
                            checkpoint['model_state_dict'], strict=False)
                    # define a Random Forest Regressor to work on embeddings
                    if self.union_mode == 'RF':
                        self.aux_model = RandomForestRegressor(
                            n_estimators=self.n_estimators,
                            criterion=self.criterion,                            
                            max_depth=self.max_depth,
                            min_samples_split=self.min_samples_split,
                            min_samples_leaf=self.min_samples_leaf,
                            random_state=self.seed) 
            else:
                for split_key in range(self.n_splits):
                    # define the report path
                    self.report_path = os.path.join(
                        self.result_path,
                        '{}_{}_fold{}_seed_{}_report_.txt'.format(
                            self.uq_method, self.split, split_key+1, self.seed))
                    # load train, validation, and test data loaders
                    (self.X_train_path, self.X_val_path, self.X_test_path,
                     self.y_train_path, self.y_val_path, self.y_test_path,
                     self.test_lengths_path) = self.cv_paths(split_key=split_key)
                    # except for Bootstrapping ensemble and embedding-based
                    if ((not self.bootstrapping) and (not self.union_mode)):
                        (self.train_loader, self.val_loader, self.test_loader,
                         self.test_lengths) = self.load_data()
                    # if there is only one model to train (and not embedding-based)
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
                                    processed_data_path=self.result_path,
                                    report_path=self.report_path,
                                    data_split='cv',
                                    fold = split_key+1,
                                    cfg=self.cfg,
                                    seed=self.seed,
                                    ensemble_mode=self.ensemble_mode)
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
                                   normalization=self.normalization) 
                    elif self.ensemble_mode:
                        # if there are ensemble of models to train
                        # get random state (before subset selction for Bootstrapping)
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
                                        processed_data_path=self.result_path,
                                        report_path=self.report_path,
                                        data_split='cv',
                                        fold = split_key+1,
                                        cfg=self.cfg,
                                        seed=self.seed,
                                        model_idx=i,
                                        ensemble_mode=self.ensemble_mode)
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
                                   ensemble_size=self.num_models)
                    elif self.union_mode:
                        # check deterministic pre-trained model is available                    
                        deterministic_checkpoint_path = os.path.join(
                            self.result_path,
                            'deterministic_{}_fold{}_seed_{}_best_model.pt'.format(
                                self.split, split_key+1, self.seed))                       
                        if not os.path.isfile(deterministic_checkpoint_path):
                            raise FileNotFoundError(
                                'Deterministic model should be trained in advance')    
                        else:
                            # load the checkpoint except the last layer
                            checkpoint = torch.load(deterministic_checkpoint_path)
                            self.model.load_state_dict(
                                checkpoint['model_state_dict'], strict=False) 
                        # define a Random Forest Regressor to work on embeddings
                        if self.union_mode == 'RF':
                            self.aux_model = RandomForestRegressor(
                                n_estimators=self.n_estimators,
                                criterion=self.criterion,                            
                                max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                min_samples_leaf=self.min_samples_leaf,
                                random_state=self.seed) 
                        
                    
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