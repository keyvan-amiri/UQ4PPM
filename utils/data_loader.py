import logging
import os
import pickle
import torch
from torch.utils.data import TensorDataset

# A calss for customized exception related to normalization error.
class NormalizationError(Exception):
    def __init__(self, message):
        super().__init__(message)

    
def get_dataset(args, config, test_set=False, validation=False):
    data_object = None
    split_method = 'cv' if args.n_splits > 1 else 'holdout'
    if config.model.type == 'dalstm':
        data_object = DALSTM_Dataset(config, args.split, validation, 
                                     split_method) 
    #TODO implement separate objects for PGTNet architecture
    elif config.model.type == 'pgtnet':
        print(f'The backebone model {args.model} is not yet supported') 
    else: 
        print(f'The backebone model {args.model} is not recognized')     
    data_type = 'test' if test_set else 'train'
    logging.info(data_object.summary_dataset(split=data_type))
    data = data_object.return_dataset(split=data_type)   
    
    return data_object, data

# A class to create datasets to be used by DALSTM model
# In general a separate class is required for each architecture. 
class DALSTM_Dataset(object):
    def __init__(self, config, split, validation=False,
                 split_method='holdout'):
        # get dataset name, and the path to pre-processed data
        self.dataset = config.data.dir
        self.dalstm_class = 'DALSTM_' + self.dataset
        root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))        
        self.dataset_path =  os.path.join(
            os.path.join(root_path, 'datasets'), self.dalstm_class)
        # get relevant paths to import all required data
        if split_method == 'holdout':
            (self.X_train_path, self.X_val_path, self.X_test_path,
             self.y_train_path, self.y_val_path, self.y_test_path,
             self.test_lengths_path, self.max_train_val_path,
             self.mean_train_val_path, self.median_train_val_path
             ) = self.holdout_paths() 
        else:
            (self.X_train_path, self.X_val_path, self.X_test_path,
             self.y_train_path, self.y_val_path, self.y_test_path,
             self.test_lengths_path, self.max_train_val_path,
             self.mean_train_val_path, self.median_train_val_path
             ) = self.cv_paths(split_key=split)
        # Load data
        X_train = torch.load(self.X_train_path, weights_only=True)
        X_val = torch.load(self.X_val_path, weights_only=True)
        X_test = torch.load(self.X_test_path, weights_only=True)
        y_train = torch.load(self.y_train_path, weights_only=True)
        y_val = torch.load(self.y_val_path, weights_only=True)
        y_test = torch.load(self.y_test_path, weights_only=True) 
        # define training and test set bsed on arguments
        if validation:
            # train set = train set, while test set = validation set 
            self.x_train = X_train
            self.x_test = X_val
            self.y_train = y_train
            self.y_test = y_val
        else:
            # train set = train + validation sets, while test set = test set 
            self.x_train = torch.cat((X_train, X_val), 0)
            self.x_test = X_test
            self.y_train = torch.cat((y_train, y_val), 0)
            self.y_test = y_test
        # load length of prefixes in the test set
        with open(self.test_lengths_path, 'rb') as file:
            test_original_lengths = pickle.load(file)  
        self.test_original_lengths = test_original_lengths            
        # Load statistics of remanining time
        # Load maximum remaining time in training, and validation sets
        with open(self.max_train_val_path, 'rb') as file:
            max_target_value = pickle.load(file)
        # Load mean remaining time in training, and validation sets
        with open(self.mean_train_val_path, 'rb') as file:
            mean_target_value = pickle.load(file)
        # Load median remaining time in training, and validation sets
        with open(self.median_train_val_path, 'rb') as file:
            median_target_value = pickle.load(file)
        # check normalization in pre-processing
        if max_target_value == None:
            if config.model.target_norm:
                raise NormalizationError(
                    'Repeat pre-processing step with target normalization')
            else:
                # Although normalization is not applied we still need maximum,
                # mean, and median of remaining time (in case we want to use 
                # a noise_prior instead of pre-trained deterministic model.
                self.max_target_value = torch.max(
                    torch.cat((y_train, y_val), 0)).numpy()
                self.mean_target_value = torch.mean(
                    torch.cat((y_train, y_val), 0)).numpy()
                self.median_target_value = torch.median(
                    torch.cat((y_train, y_val), 0)).numpy()
        else:
            self.max_target_value = max_target_value
            self.mean_target_value = mean_target_value
            self.median_target_value = median_target_value
        # compute important dimensions
        self.train_n_samples = self.x_train.shape[0]
        self.test_n_samples = self.x_test.shape[0] 
        # dimension of training data input (features)
        self.train_dim_x = self.x_train.shape[2] 
        # dimension of testing data input (features)
        self.test_dim_x = self.x_test.shape[2]  
        # dimension of training data input (sequence dimension)
        self.train_sequence_dim =  self.x_train.shape[1] 
        # dimension of testing data input (sequence dimension)  
        self.test_sequence_dim =  self.x_test.shape[1]       
            
    def return_dataset(self, split="train"):
        if split == "train":
            train_dataset = TensorDataset(self.x_train, self.y_train)
            return train_dataset
        else:
            test_dataset = TensorDataset(self.x_test, self.y_test)
            return test_dataset
    
    def summary_dataset(self, split="train"):
        if split == "train":
            return {'n_samples': self.train_n_samples,
                    'dim_x': self.train_dim_x,
                    'sequence_dim': self.train_sequence_dim,
                    'dim_y': 1}
        else:
            return {'n_samples': self.test_n_samples,
                    'dim_x': self.test_dim_x,
                    'sequence_dim': self.test_sequence_dim,
                    'dim_y': 1}
        
    def return_mean_target_arrtibute(self):
        return self.mean_target_value
    
    def return_median_target_arrtibute(self):
        return self.median_target_value

    def return_max_target_arrtibute(self):
        return self.max_target_value
    
    def return_prefix_lengths(self):
        return self.test_original_lengths         
           
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
        max_train_val_path = os.path.join(
            self.dataset_path, "DALSTM_max_train_val_"+self.dataset+".pkl") 
        mean_train_val_path = os.path.join(
            self.dataset_path, "DALSTM_mean_train_val_"+self.dataset+".pkl")    
        median_train_val_path = os.path.join(
            self.dataset_path, "DALSTM_median_train_val_"+self.dataset+".pkl")               
        return (X_train_path, X_val_path, X_test_path, y_train_path,
                y_val_path, y_test_path, test_length_path, max_train_val_path,
                mean_train_val_path, median_train_val_path)
    
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
        max_train_val_path = os.path.join(
            self.dataset_path, "DALSTM_max_train_val_"+self.dataset+".pkl")
        mean_train_val_path = os.path.join(
            self.dataset_path, "DALSTM_mean_train_val_"+self.dataset+".pkl")  
        median_train_val_path = os.path.join(
            self.dataset_path, "DALSTM_median_train_val_"+self.dataset+".pkl")  
        return (X_train_path, X_val_path, X_test_path, y_train_path,
                y_val_path, y_test_path,test_length_path, max_train_val_path,
                mean_train_val_path, median_train_val_path)