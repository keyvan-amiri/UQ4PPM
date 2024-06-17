import logging
import os
import pickle
from pathlib import Path
import torch
from torch.utils.data import TensorDataset
   
def get_dataset(args, config, test_set=False, validation=False, kfold_handler=None):
    data_object = None
    split_method = 'cv' if args.n_splits > 1 else 'holdout'
    data_object = DALSTM_Dataset(config, args.split, validation, split_method,
                              kfold_handler)        
    data_type = "test" if test_set else "train"
    logging.info(data_object.summary_dataset(split=data_type))
    data = data_object.return_dataset(split=data_type)   
    
    return data_object, data

# A class to create datasets to be used by DALSTM model
class DALSTM_Dataset(object):
    def __init__(self, config, split, validation=False, split_method='holdout',
                 kfold_handler=None):
        _DATA_DIRECTORY_PATH = os.path.join(config.data.data_root,
                                            config.data.dir, "data")      
        _XES_FILE_PATH = os.path.join(_DATA_DIRECTORY_PATH,
                                      os.path.basename(config.data.dir) + ".xes")
        _DATA_TRAIN_FILE_PATH = os.path.join(_DATA_DIRECTORY_PATH,
                                             "x_train.pt")
        _TARGET_TRAIN_FILE_PATH = os.path.join(_DATA_DIRECTORY_PATH,
                                               "y_train.pt")
        _DATA_TEST_FILE_PATH = os.path.join(_DATA_DIRECTORY_PATH,
                                            "x_test.pt")
        _TARGET_TEST_FILE_PATH = os.path.join(_DATA_DIRECTORY_PATH,
                                              "y_test.pt")
        _LENGTH_TEST_FILE_PATH = os.path.join(_DATA_DIRECTORY_PATH,
                                              "test_length_list.pkl")
        _MEAN_REM_PATH = os.path.join(_DATA_DIRECTORY_PATH,
                                      "mean_rem_time.pkl")
        _MEDIAN_REM_PATH = os.path.join(_DATA_DIRECTORY_PATH,
                                        "median_rem_time.pkl")
        _STD_REM_PATH = os.path.join(_DATA_DIRECTORY_PATH, "std_rem_time.pkl")
        pre_processing_paths = [_DATA_TRAIN_FILE_PATH, _TARGET_TRAIN_FILE_PATH,
                                _DATA_TEST_FILE_PATH, _TARGET_TEST_FILE_PATH,
                                _LENGTH_TEST_FILE_PATH, _MEAN_REM_PATH,
                                _MEDIAN_REM_PATH, _STD_REM_PATH]
        all_paths_exist = all(os.path.exists(path) for path in pre_processing_paths)
        if config.diffusion.conditioning_signal == 'DALSTM':
            if all_paths_exist:
                print('Preprocessing is already done.')
                x_train = torch.load(_DATA_TRAIN_FILE_PATH)
                x_test = torch.load(_DATA_TEST_FILE_PATH)
                y_train = torch.load(_TARGET_TRAIN_FILE_PATH)
                y_test = torch.load(_TARGET_TEST_FILE_PATH)
                with open(_LENGTH_TEST_FILE_PATH, 'rb') as file:
                    test_original_lengths = pickle.load(file)  
                with open(_MEAN_REM_PATH, 'rb') as file:
                    mean_target_value = pickle.load(file)
                with open(_MEDIAN_REM_PATH, 'rb') as file:
                    median_target_value = pickle.load(file) 
                with open(_STD_REM_PATH, 'rb') as file:
                    std_target_value = pickle.load(file)  
            else:
                pd_event_log = convert_xes_to_csv(_XES_FILE_PATH) 
                attributes = load_attributes_from_file(config.data.preprocessing_config,
                                                       Path(_XES_FILE_PATH).name) 
                input_columns = [XES_Fields.CASE_COLUMN,
                                 XES_Fields.ACTIVITY_COLUMN,
                                 XES_Fields.TIMESTAMP_COLUMN] + attributes
                event_log = select_columns(pd_event_log,
                                           input_columns=input_columns,
                                           timestamp_format="%Y-%m-%d %H:%M:%S")
                ordered_columns = [XES_Fields.CASE_COLUMN,
                                   XES_Fields.ACTIVITY_COLUMN,
                                   XES_Fields.TIMESTAMP_COLUMN]
                event_log = reorder_columns(event_log,
                                            ordered_columns=ordered_columns) 
                train_log, test_log = split_csv(event_log, config,
                                                XES_Fields.CASE_COLUMN,
                                                split_method, split,
                                                kfold_handler)
                # preprocessing to get Pytoch tensors
                # + get length of prefixes, target attribute: mean, median, std 
                x_train, x_test, y_train, y_test, test_original_lengths, \
                    mean_target_value, median_target_value, \
                        std_target_value = dalstm_preprocessing(event_log,
                                                                train_log,
                                                                test_log,
                                                                config)
                torch.save(x_train, _DATA_TRAIN_FILE_PATH)                  
                torch.save(x_test, _DATA_TEST_FILE_PATH)                      
                torch.save(y_train, _TARGET_TRAIN_FILE_PATH)
                torch.save(y_test, _TARGET_TEST_FILE_PATH)
                # save test prefix lengths to be used later in earliness analysis
                with open(_LENGTH_TEST_FILE_PATH, 'wb') as file:
                    pickle.dump(test_original_lengths, file)
                with open(_MEAN_REM_PATH, 'wb') as file:
                    pickle.dump(mean_target_value, file)
                with open(_MEDIAN_REM_PATH, 'wb') as file:
                    pickle.dump(median_target_value, file)
                with open(_STD_REM_PATH, 'wb') as file:
                    pickle.dump(std_target_value, file)
        
        # split train set further into train and validation set for hyperparameter tuning
        if validation:
            num_training_examples = int(
                config.diffusion.nonlinear_guidance.train_ratio * x_train.shape[0])
            x_test = x_train[num_training_examples:, :, :]
            y_test = y_train[num_training_examples:]
            x_train = x_train[:num_training_examples, :, :]
            y_train = y_train[:num_training_examples]
        
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.test_original_lengths = test_original_lengths
        self.mean_target_value = mean_target_value
        self.median_target_value = median_target_value
        self.std_target_value = std_target_value
        
        self.train_n_samples = x_train.shape[0]
        self.test_n_samples = x_test.shape[0] 
        # dimension of training data input (featurs)
        self.train_dim_x = self.x_train.shape[2] 
        # dimension of testing data input (featurs)
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
    
    def return_std_target_arrtibute(self):
        return self.std_target_value
      
    # TODO: add nevessary code to obtain it!
    def return_max_target_arrtibute(self):
        return self.max_target_value
    
    def return_prefix_lengths(self):
        return self.test_original_lengths