import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import os
import pickle
from utils.utils import set_random_seed, set_optimizer, train_model, test_model
from loss.loss_handler import set_loss
from models.dalstm import DALSTMModel

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
        self.dalstm_class = 'DALSTM_' + self.dataset_name
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
        # Define important size and dimensions
        (self.test_lengths, self.input_size, self.max_len,
         self.max_train_val) = self.load_dimensions()
        self.hidden_size = cfg.get('model').get('lstm').get('hidden_size')
        self.n_layers = cfg.get('model').get('lstm').get('n_layers')
        # whether to use drop-out or not
        self.dropout = cfg.get('model').get('lstm').get('dropout')
        # the probabibility that is used for dropout
        self.dropout_prob = cfg.get('model').get('lstm').get('dropout_prob')        
        # define loss function
        self.criterion = set_loss(
            loss_func=cfg.get('train').get('loss_function'))
        # define the model (depends to the UQ method decided)
        if self.uq_method == 'deterministic':
            self.model = DALSTMModel(input_size=self.input_size,
                                     hidden_size=self.hidden_size,
                                     n_layers=self.n_layers,
                                     max_len=self.max_len,
                                     dropout=self.dropout,
                                     p_fix=self.dropout_prob).to(self.device)
        else:
            #TODO: define UQ method models here or add them to previous one
            pass
        total_params = sum(p.numel() for p in self.model.parameters()
                           if p.requires_grad)
        print(f'Total model parameters: {total_params}')
        # define optimizer
        self.optimizer = set_optimizer(self.model,
                                  cfg.get('optimizer').get('type'),
                                  cfg.get('optimizer').get('base_lr'),
                                  cfg.get('optimizer').get('eps'),
                                  cfg.get('optimizer').get('weight_decay'))
        # define scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5)
        # define other training hyperparameters
        self.max_epochs = cfg.get('train').get('max_epochs')
        self.early_stop_patience = cfg.get('train').get('early_stop.patience')
        self.early_stop_min_delta = cfg.get('train').get('early_stop.min_delta') 
        # execute training and evaluation loop
        for execution_seed in seeds:
            self.seed = int(execution_seed) 
            # set random seed 
            set_random_seed(self.seed)
            # load train, validation, and test data loaders
            if self.split == 'holdout':
                # define the report path
                self.report_path = os.path.join(
                    self.result_path, '{}_{}_seed_{}_report_.txt'.format(
                        self.uq_method, self.split, self.seed))                
                (self.X_train_path, self.X_val_path, self.X_test_path,
                 self.y_train_path, self.y_val_path, self.y_test_path
                 ) = self.holdout_paths() 
                (self.train_loader, self.val_loader, self.test_loader
                 ) = self.load_data()
                if self.uq_method == 'deterministic':
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
                                data_split='holdout',
                                cfg=self.cfg,
                                seed=self.seed)   
                    test_model(model=self.model,
                               uq_method=self.uq_method,
                               test_loader=self.test_loader,
                               test_original_lengths=self.test_lengths,
                               y_scaler=self.max_train_val,
                               processed_data_path= self.result_path,
                               report_path=self.report_path,
                               data_split = 'holdout',
                               seed=self.seed,
                               device=self.device,
                               normalization=self.normalization)                    
                else:
                    pass
                    #TODO: Add new training method or extend the exisiting one
                    # to implement dropout appriximation paper!                
            else:
                for split_key in range(self.n_splits):
                    # define the report path
                    self.report_path = os.path.join(
                        self.result_path,
                        '{}_{}_fold{}_seed_{}_report_.txt'.format(
                            self.uq_method, self.split, split_key+1, self.seed))
                    # load train, validation, and test data loaders
                    (self.X_train_path, self.X_val_path, self.X_test_path,
                     self.y_train_path, self.y_val_path, self.y_test_path
                     ) = self.cv_paths(split_key=split_key)
                    (self.train_loader, self.val_loader, self.test_loader
                     ) = self.load_data()
                    if self.uq_method == 'deterministic':
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
                                    seed=self.seed)
                        test_model(model=self.model,
                                   uq_method=self.uq_method,
                                   test_loader=self.test_loader,
                                   test_original_lengths=self.test_lengths,
                                   y_scaler=self.max_train_val,
                                   processed_data_path= self.result_path,
                                   report_path=self.report_path,
                                   data_split = 'holdout',
                                   seed=self.seed,
                                   device=self.device,
                                   normalization=self.normalization)    
                    else:                        
                        #TODO: Add new training method or extend the exisiting one
                        # to implement dropout appriximation paper!
                        pass
                    

    
    # A method to load important dimensions
    def load_dimensions(self):
        test_length_path = os.path.join(
            self.dataset_path, "DALSTM_test_length_list_"+self.dataset+".pkl")    
        scaler_path = os.path.join(
            self.dataset_path,
            "DALSTM_max_train_val_"+self.dataset+".pkl")
        input_size_path = os.path.join(
            self.dataset_path,
            "DALSTM_input_size_"+self.dataset+".pkl")
        max_len_path = os.path.join(
            self.dataset_path,
            "DALSTM_max_len_"+self.dataset+".pkl")
        with open(test_length_path, 'rb') as f:
            test_lengths =  pickle.load(f)
        # input_size corresponds to vocab_size
        with open(input_size_path, 'rb') as f:
            input_size =  pickle.load(f)
        with open(max_len_path, 'rb') as f:
            max_len =  pickle.load(f) 
        with open(scaler_path, 'rb') as f:
            max_train_val =  pickle.load(f)            
        return (test_lengths, input_size, max_len, max_train_val)
        
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
        return (X_train_path, X_val_path, X_test_path, y_train_path,
                y_val_path, y_test_path)
        
    # A method to create list of path for holdout data split
    def cv_paths(self, split_key=None):
        X_train_path = os.path.join(
            self.dataset_path,
            "DALSTM_X_train_fold_"+str(split_key)+self.dataset_name+".pt")
        X_val_path = os.path.join(
            self.dataset_path,
            "DALSTM_X_val_fold_"+str(split_key)+self.dataset_name+".pt")
        X_test_path = os.path.join(
            self.dataset_path,
            "DALSTM_X_test_fold_"+str(split_key)+self.dataset_name+".pt")
        y_train_path = os.path.join(
            self.dataset_path,
            "DALSTM_y_train_fold_"+str(split_key)+self.dataset_name+".pt")
        y_val_path = os.path.join(
            self.dataset_path,
            "DALSTM_y_val_fold_"+str(split_key)+self.dataset_name+".pt")
        y_test_path = os.path.join(
            self.dataset_path,
            "DALSTM_y_test_fold_"+str(split_key)+self.dataset_name+".pt")        
        return (X_train_path, X_val_path, X_test_path, y_train_path,
                y_val_path, y_test_path)
        
    # A method to load training and evaluation 
    def load_data(self):
        X_train = torch.load(self.X_train_path)
        X_val = torch.load(self.X_val_path)
        X_test = torch.load(self.X_test_path)
        y_train = torch.load(self.y_train_path)
        y_val = torch.load(self.y_val_path)
        y_test = torch.load(self.y_test_path)        
        # define training, validation, test datasets                    
        train_dataset = TensorDataset(X_train, y_train)
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
        return (train_loader, val_loader, test_loader)
    