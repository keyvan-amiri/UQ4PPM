"""
To prepare thi script we used the following source codes:
    https://gitlab.citius.usc.es/efren.rama/pmdlcompararator
We adjusted the source codes to efficiently integrate them into our framework.
"""

import os
import yaml
import numpy as np
import pm4py
import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_import_factory
from pathlib import Path
from datetime import datetime
import torch
import pickle
import time
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing import sequence


class XES_Fields:
    CASE_COLUMN = 'case:concept:name'
    ACTIVITY_COLUMN = 'concept:name'
    TIMESTAMP_COLUMN = 'time:timestamp'
    LIFECYCLE_COLUMN = 'lifecycle:transition'
    
class Timestamp_Formats:
    TIME_FORMAT_DALSTM = "%Y-%m-%d %H:%M:%S"
    #TIME_FORMAT_DALSTM2 = '%Y-%m-%d %H:%M:%S.%f%z' 
    TIME_FORMAT_DALSTM2 = '%Y-%m-%d %H:%M:%S%z' # all BPIC 2012 logs
    TIME_FORMAT_DALSTM_list = [TIME_FORMAT_DALSTM, TIME_FORMAT_DALSTM2]
    
# DALSTM pre-processing class
class DALSTM_preprocessing ():   
    
    def __init__ (self, xes_dir=None, dalstm_dir=None, conversion_cfg=None,
                  dataset_name=None, split_ratio=None, n_splits=None,
                  normalization=None, normalization_type=None, overwrite=None, 
                  perform_lifecycle_trick=None, fill_na=None, seed=None):
        # set random seed for cross-validation
        if seed is None:
            self.seed = 42
        else:
            self.seed = seed
        self.dataset_name = dataset_name
        root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if xes_dir is None:
            xes_dir = os.path.join(root_path, 'datasets', 'event_logs')
        self.xes_file = os.path.join(xes_dir, self.dataset_name+'.xes')
        # read the conversion config file
        if conversion_cfg is None:
            conversion_cfg = os.path.join(root_path,'datasets',
                                          'conversion_cfg.yaml') 
        with open(conversion_cfg, 'r') as f:
            guide_dict = yaml.safe_load(f)
        data_guide_dict = guide_dict[self.dataset_name]
        # set data attributes to be included
        self.event_attributes = data_guide_dict.get('event_categorical', [])
        # set the output folder
        if dalstm_dir is None:
            self.dalstm_dir = os.path.join(root_path, 'datasets')
        else:
            self.dalstm_dir = dalstm_dir            
        self.dalstm_class = 'DALSTM_' + self.dataset_name
        self.dataset_path =  os.path.join(self.dalstm_dir, self.dalstm_class)
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
        # Check if overwrite is False and holdout data availble: stop!
        self.save_addresses = ["DALSTM_X_train_"+self.dataset_name+".pt",
                               "DALSTM_X_val_"+self.dataset_name+".pt",
                               "DALSTM_X_test_"+self.dataset_name+".pt",
                               "DALSTM_y_train_"+self.dataset_name+".pt",
                               "DALSTM_y_val_"+self.dataset_name+".pt",
                               "DALSTM_y_test_"+self.dataset_name+".pt"] 
        self.out_add0 = os.path.join(self.dataset_path, self.save_addresses[0])
        self.out_add1 = os.path.join(self.dataset_path, self.save_addresses[1])
        self.out_add2 = os.path.join(self.dataset_path, self.save_addresses[2])
        self.out_add3 = os.path.join(self.dataset_path, self.save_addresses[3])
        self.out_add4 = os.path.join(self.dataset_path, self.save_addresses[4])
        self.out_add5 = os.path.join(self.dataset_path, self.save_addresses[5])
        self.overwrite = overwrite if overwrite is not None else False 
        if (not self.overwrite and os.path.exists(self.out_add0) and
            os.path.exists(self.out_add1) and os.path.exists(self.out_add2)
            and os.path.exists(self.out_add3) and os.path.exists(self.out_add4)
            and os.path.exists(self.out_add5)):
            print(f"For '{dataset_name}' DALST preprocessing is already done.")
            # to not continue with transformation
            return
        self.split_ratio = split_ratio if split_ratio is not None\
            else [0.64,0.16,0.20]   
        self.normalization = normalization if normalization is not None\
            else False 
        self.normalization_type = normalization_type\
            if normalization_type is not None else 'max_norm'
        self.perform_lifecycle_trick = perform_lifecycle_trick\
            if perform_lifecycle_trick is not None else True
        self.fill_na = fill_na
        self.n_splits = n_splits if n_splits is not None else 5
                
        # execute preprocessing in two steps:
        # create training, validation, test csv files.
        self.data_handling()
        # create train, validation, test tensors for both holdout and CV
        self.dalstm_process()        
        
    # A method to tranform XES to CSV and execute some preprocessing steps
    def xes_to_csv(self):        
        xes_path = self.xes_file
        csv_file = Path(self.xes_file).stem.split('.')[0] + '.csv'
        csv_path = os.path.join(self.dataset_path, csv_file)
        log = xes_import_factory.apply(xes_path,
                                       parameters={'timestamp_sort': True})
        equivalent_dataframe = pm4py.convert_to_dataframe(log)
        equivalent_dataframe.to_csv(csv_path)
        pd_log = pd.read_csv(csv_path)   
        if self.fill_na is not None:
            pd_log.fillna(self.fill_na, inplace=True)
            pd_log.replace("-", self.fill_na, inplace=True)
            pd_log.replace(np.nan, self.fill_na)
        if 'BPI_2012' in self.dataset_name:
            counter_list = []
            for counter in range (len(pd_log)):
                for format_str in Timestamp_Formats.TIME_FORMAT_DALSTM_list:
                    try:
                        incr_timestamp = datetime.strptime(
                            str(pd_log.iloc[counter][
                                XES_Fields.TIMESTAMP_COLUMN]), format_str)  
                        if format_str == '%Y-%m-%d %H:%M:%S%z':
                            counter_list.append(counter)
                        break
                    except ValueError:
                        continue
            pd_log = pd_log.drop(index=counter_list)
        # Use integers always for case identifiers.
        # We need this to make a split that is equal for every dataset
        pd_log[XES_Fields.CASE_COLUMN] = pd.Categorical(
            pd_log[XES_Fields.CASE_COLUMN])
        pd_log[XES_Fields.CASE_COLUMN] = pd_log[XES_Fields.CASE_COLUMN].cat.codes    
        # lifecycle_trick: ACTIVITY NAME + LIFECYCLE-TRANSITION
        unique_lifecycle = pd_log[XES_Fields.LIFECYCLE_COLUMN].unique()
        if len(unique_lifecycle) > 1 and self.perform_lifecycle_trick:
            pd_log[XES_Fields.ACTIVITY_COLUMN] = pd_log[
                XES_Fields.ACTIVITY_COLUMN].astype(str) + "+" + pd_log[
                    XES_Fields.LIFECYCLE_COLUMN]       
        pd_log.to_csv(csv_path, encoding="utf-8")

        return csv_file, csv_path
    
    def select_columns(self, file=None, input_columns=None,
                       category_columns=None, timestamp_format=None,
                       output_columns=None, categorize=False, fill_na=None,
                       save_category_assignment=None):

        dataset = pd.read_csv(file)
        if fill_na is not None:
            dataset = dataset.fillna(fill_na)
        if input_columns is not None:
            dataset = dataset[input_columns]
        timestamp_column = XES_Fields.TIMESTAMP_COLUMN
        dataset[timestamp_column] = pd.to_datetime(
            dataset[timestamp_column], utc=True)
        dataset[timestamp_column] = dataset[
            timestamp_column].dt.strftime(timestamp_format)
        if categorize:
            for category_column in category_columns:
                if category_column == XES_Fields.ACTIVITY_COLUMN:
                    category_list = dataset[
                        category_column].astype("category").cat.categories.tolist()
                    category_dict = {c : i for i, c in enumerate(category_list)}
                    if save_category_assignment is None:
                        print("Activity assignment: ", category_dict)
                    else:
                        file_name = Path(file).name
                        with open(os.path.join(
                                save_category_assignment, file_name), "w") as fw:
                            fw.write(str(category_dict))
                dataset[category_column] = dataset[
                    category_column].astype("category").cat.codes
        if output_columns is not None:
            dataset.rename(
                output_columns,
                axis="columns",
                inplace=True)
        dataset.to_csv(file, sep=",", index=False)
    
    # a metohd to change the order of columns
    def reorder_columns(self, file=None, ordered_columns=None):
        df = pd.read_csv(file)
        df = df.reindex(columns=(ordered_columns + list(
            [a for a in df.columns if a not in ordered_columns])))
        df.to_csv(file, sep=",", index=False)
        
    # A method to split the cases into training, validation, and test sets
    def split_data(self, file=None, case_column=None):
        # split data for cv
        pandas_init = pd.read_csv(file)
        pd.set_option('display.expand_frame_repr', False)
        groups = [pandas_df for _, pandas_df in \
                  pandas_init.groupby(case_column, sort=False)]
        train_size = round(len(groups) * self.split_ratio[0])
        val_size = round(len(groups) * (self.split_ratio[0]+ self.split_ratio[1]))
        train_groups = groups[:train_size]
        val_groups = groups[train_size:val_size]
        test_groups = groups[val_size:]
        # Disable the sorting to not mess with the order of the timestamps.
        train = pd.concat(train_groups, sort=False).reset_index(drop=True)
        val = pd.concat(val_groups, sort=False).reset_index(drop=True)
        test = pd.concat(test_groups, sort=False).reset_index(drop=True)
        train_hold_path = os.path.join(self.dataset_path,
                                       "train_" + Path(file).stem + ".csv")
        val_hold_path = os.path.join(self.dataset_path,
                                     "val_" + Path(file).stem + ".csv")
        test_hold_path = os.path.join(self.dataset_path,
                                      "test_" + Path(file).stem + ".csv")
        train.to_csv(train_hold_path, index=False)
        val.to_csv(val_hold_path, index=False)
        test.to_csv(test_hold_path, index=False)
    
    # method to handle initial steps of preprocessing for DALSTM
    def data_handling(self):        
        # create equivalent csv file
        csv_file, csv_path = self.xes_to_csv() 
        # Define relevant attributes
        attributes = self.event_attributes
        # Handling special cases for input event logs
        if self.dataset_name == "Traffic_Fine":
            attributes.remove('dismissal')     
        if (self.dataset_name == "BPI_2012" or self.dataset_name == "BPI_2012W" 
            or self.dataset_name == "BPI_2013_I"):
            attributes.append(XES_Fields.LIFECYCLE_COLUMN)       
        if 'BPI_2012' in self.dataset_name:
            selected_timestamp_format =Timestamp_Formats.TIME_FORMAT_DALSTM
        else:
            selected_timestamp_format =Timestamp_Formats.TIME_FORMAT_DALSTM
        # select related columns  
        self.select_columns(file=csv_path,
                            input_columns=[XES_Fields.CASE_COLUMN,
                                           XES_Fields.ACTIVITY_COLUMN,
                                           XES_Fields.TIMESTAMP_COLUMN
                                           ] + attributes, 
                            category_columns=None, 
                            timestamp_format=selected_timestamp_format,
                            output_columns=None, categorize=False) 
            
        # Reorder columns   
        self.reorder_columns(file=csv_path,
                             ordered_columns=[XES_Fields.CASE_COLUMN,
                                              XES_Fields.ACTIVITY_COLUMN,
                                              XES_Fields.TIMESTAMP_COLUMN])     
        
        # execute data split
        self.split_data(file=csv_path, case_column=XES_Fields.CASE_COLUMN)
    
    # Auxiliary method for preprocessing   
    def buildOHE(self, index=None, n=None):
        L = [0] * n
        L[index] = 1
        return L
    
    # A method for DALSTM preprocessing (output: Pytorch tensors for training)
    def dalstm_load_dataset(self, filename=None, prev_values=None):
        dataframe = pd.read_csv(filename, header=0)
        dataframe = dataframe.replace(r's+', 'empty', regex=True)
        dataframe = dataframe.replace("-", "UNK")
        dataframe = dataframe.fillna(0)

        dataset = dataframe.values
        if prev_values is None:
            values = []
            for i in range(dataset.shape[1]):
                try:
                    values.append(len(np.unique(dataset[:, i])))  # +1
                except:
                    dataset[:, i] = dataset[:, i].astype(str)       
                    values.append(len(np.unique(dataset[:, i])))  # +1

            # output is changed to handle prefix lengths
            #print(values)
            return (None, None, None), values 
        else:
            values = prev_values

        #print("Dataset: ", dataset)
        #print("Values: ", values)

        datasetTR = dataset

        def generate_set(dataset):

            data = []
            # To collect prefix lengths (required for earliness analysis)
            original_lengths = []  
            newdataset = []
            temptarget = []
            
            # analyze first dataset line
            caseID = dataset[0][0]
            starttime = datetime.fromtimestamp(
                time.mktime(time.strptime(dataset[0][2], "%Y-%m-%d %H:%M:%S")))
            lastevtime = datetime.fromtimestamp(
                time.mktime(time.strptime(dataset[0][2], "%Y-%m-%d %H:%M:%S")))
            t = time.strptime(dataset[0][2], "%Y-%m-%d %H:%M:%S")
            midnight = datetime.fromtimestamp(
                time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
            timesincemidnight = (
                datetime.fromtimestamp(time.mktime(t)) - midnight).total_seconds()
            n = 1
            temptarget.append(
                datetime.fromtimestamp(time.mktime(
                    time.strptime(dataset[0][2], "%Y-%m-%d %H:%M:%S"))))
            a = [(datetime.fromtimestamp(
                time.mktime(time.strptime(
                    dataset[0][2], "%Y-%m-%d %H:%M:%S"))) - starttime).total_seconds()]
            a.append((datetime.fromtimestamp(
                time.mktime(time.strptime(
                    dataset[0][2], "%Y-%m-%d %H:%M:%S"))) - lastevtime).total_seconds())
            a.append(timesincemidnight)
            a.append(datetime.fromtimestamp(time.mktime(t)).weekday() + 1)
            a.extend(
                self.buildOHE(
                    index=one_hot(dataset[0][1], values[1], split="|")[0],
                    n=values[1]))
            field = 3
            for i in dataset[0][3:]:
                if not np.issubdtype(dataframe.dtypes[field], np.number):
                    a.extend(
                        self.buildOHE(
                            index=one_hot(str(i), values[field], split="|")[0],
                            n=values[field]))
                    #print(field, values[field])
                else:
                    #print('numerical', field)
                    a.append(i)
                field += 1
            newdataset.append(a)
            #line_counter = 1
            for line in dataset[1:, :]:
                #print(line_counter)
                case = line[0]
                if case == caseID:
                    # continues the current case
                    t = time.strptime(line[2], "%Y-%m-%d %H:%M:%S")
                    midnight = datetime.fromtimestamp(time.mktime(t)).replace(
                        hour=0, minute=0, second=0, microsecond=0)
                    timesincemidnight = (datetime.fromtimestamp(
                        time.mktime(t)) - midnight).total_seconds()
                    temptarget.append(datetime.fromtimestamp(
                        time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S"))))
                    a = [(datetime.fromtimestamp(
                        time.mktime(time.strptime(
                            line[2], "%Y-%m-%d %H:%M:%S"))) - starttime).total_seconds()]
                    a.append((datetime.fromtimestamp(
                        time.mktime(time.strptime(
                            line[2], "%Y-%m-%d %H:%M:%S"))) - lastevtime).total_seconds())
                    a.append(timesincemidnight)
                    a.append(datetime.fromtimestamp(time.mktime(t)).weekday() + 1)

                    lastevtime = datetime.fromtimestamp(
                        time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S")))

                    a.extend(
                        self.buildOHE(
                            index=one_hot(line[1], values[1], filters=[],
                                          split="|")[0], n=values[1]))

                    field = 3
                    for i in line[3:]:
                        if not np.issubdtype(
                                dataframe.dtypes[field], np.number):
                            a.extend(
                                self.buildOHE(
                                    index= one_hot(str(i), values[field],
                                                   filters=[],split="|")[0],
                                    n=values[field]))
                        else:
                            a.append(i)
                        field += 1
                    newdataset.append(a)
                    n += 1
                    finishtime = datetime.fromtimestamp(
                        time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S")))
                else:
                    caseID = case
                    # Exclude prefix of length one: the loop range is changed.
                    # +1 not adding last case. target is 0, not interesting. era 1
                    for i in range(2, len(newdataset)): 
                        data.append(newdataset[:i])
                        # Keep track of prefix lengths (earliness analysis)
                        original_lengths.append(i) 
                        # print newdataset[:i]
                    newdataset = []
                    starttime = datetime.fromtimestamp(
                        time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S")))
                    lastevtime = datetime.fromtimestamp(
                        time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S")))

                    t = time.strptime(line[2], "%Y-%m-%d %H:%M:%S")
                    midnight = datetime.fromtimestamp(
                        time.mktime(t)).replace(
                            hour=0, minute=0, second=0, microsecond=0)
                    timesincemidnight = (
                        datetime.fromtimestamp(
                            time.mktime(t)) - midnight).total_seconds()

                    a = [(datetime.fromtimestamp(
                        time.mktime(time.strptime(
                            line[2], "%Y-%m-%d %H:%M:%S"))) - starttime).total_seconds()]
                    a.append((datetime.fromtimestamp(
                        time.mktime(time.strptime(
                            line[2], "%Y-%m-%d %H:%M:%S"))) - lastevtime).total_seconds())
                    a.append(timesincemidnight)
                    a.append(datetime.fromtimestamp(time.mktime(t)).weekday() + 1)

                    a.extend(
                        self.buildOHE(
                            index=one_hot(line[1], values[1], split="|")[0],
                            n=values[1]))

                    field = 3
                    for i in line[3:]:
                        if not np.issubdtype(dataframe.dtypes[field], np.number):
                            a.extend(
                                self.buildOHE(
                                    index=one_hot(str(i), values[field],
                                                  split="|")[0], n=values[field]))
                        else:
                            a.append(i)
                        field += 1
                    newdataset.append(a)
                    for i in range(n):  
                        # try-except: error handling of the original implementation.
                        try:
                            temptarget[-(i + 1)] = (
                                finishtime - temptarget[-(i + 1)]).total_seconds()
                        except UnboundLocalError:
                            # Set target value to zero if finishtime is not defined
                            # The effect is negligible as only for one dataset,
                            # this exception is for one time executed
                            print('one error in loading dataset is observed', i, n)
                            temptarget[-(i + 1)] = 0
                    # Remove the target attribute for the prefix of length one
                    if n > 1:
                        temptarget.pop(0-n)
                    temptarget.pop()  # remove last element with zero target
                    temptarget.append(
                        datetime.fromtimestamp(
                            time.mktime(time.strptime(
                                line[2], "%Y-%m-%d %H:%M:%S"))))
                    finishtime = datetime.fromtimestamp(
                        time.mktime(time.strptime(line[2], "%Y-%m-%d %H:%M:%S")))

                    n = 1
                #line_counter += 1
            # last case
            # To exclude prefix of length 1: the loop range is adjusted.
            # + 1 not adding last event, target is 0 in that case. era 1
            for i in range(2, len(newdataset)):  
                data.append(newdataset[:i])
                original_lengths.append(i) # Keep track of prefix lengths
                # print newdataset[:i]
            for i in range(n):  # era n.
                temptarget[-(i + 1)] = (
                    finishtime - temptarget[-(i + 1)]).total_seconds()
                # print temptarget[-(i + 1)]
            # Remove the target attribute for the prefix of length one
            if n > 1:
                temptarget.pop(0-n)
            temptarget.pop()  # remove last element with zero target

            # print temptarget
            print("Generated dataset with n_samples:", len(temptarget))
            assert (len(temptarget) == len(data))
            #Achtung! original_lengths is added to output
            return data, temptarget, original_lengths 

        return generate_set(datasetTR), values
    
    # a method to delete csv files after being used for preproessing
    def delete_files(self, substring=None, extension=None):
        files = os.listdir(self.dataset_path)    
        for file in files:
            if (substring!= None) and (substring in file):
                file_path = os.path.join(self.dataset_path, file)
                os.remove(file_path)
            if (extension!= None) and (file.endswith(extension)):
                file_path = os.path.join(self.dataset_path, file)
                os.remove(file_path)
    
    # A method for DALSTM preprocessing (prepoessing on pytorch tensors)
    def dalstm_process(self):
        # define important file names and paths
        full_dataset_name = self.dataset_name + '.csv'
        full_dataset_path = os.path.join(self.dataset_path, full_dataset_name)
        train_dataset_name = 'train_' + self.dataset_name + '.csv'
        train_dataset_path = os.path.join(self.dataset_path, train_dataset_name)
        val_dataset_name = 'val_' + self.dataset_name + '.csv'
        val_dataset_path = os.path.join(self.dataset_path, val_dataset_name)
        test_dataset_name = 'test_' + self.dataset_name + '.csv'
        test_dataset_path = os.path.join(self.dataset_path, test_dataset_name) 
        X_train_path = os.path.join(
            self.dataset_path, "DALSTM_X_train_"+self.dataset_name+".pt")
        X_val_path = os.path.join(
            self.dataset_path, "DALSTM_X_val_"+self.dataset_name+".pt")
        X_test_path = os.path.join(
            self.dataset_path, "DALSTM_X_test_"+self.dataset_name+".pt")
        y_train_path = os.path.join(
            self.dataset_path, "DALSTM_y_train_"+self.dataset_name+".pt")
        y_val_path = os.path.join(
            self.dataset_path, "DALSTM_y_val_"+self.dataset_name+".pt")
        y_test_path = os.path.join(
            self.dataset_path, "DALSTM_y_test_"+self.dataset_name+".pt") 
        test_length_path = os.path.join(
            self.dataset_path,
            "DALSTM_test_length_list_"+self.dataset_name+".pkl")    
        scaler_path = os.path.join(
            self.dataset_path, "DALSTM_max_train_val_"+self.dataset_name+".pkl")
        target_mean_path = os.path.join(
            self.dataset_path, "DALSTM_mean_train_val_"+self.dataset_name+".pkl")
        target_median_path = os.path.join(
            self.dataset_path, "DALSTM_median_train_val_"+self.dataset_name+".pkl")
        input_size_path = os.path.join(
            self.dataset_path, "DALSTM_input_size_"+self.dataset_name+".pkl")
        max_len_path = os.path.join(
            self.dataset_path, "DALSTM_max_len_"+self.dataset_name+".pkl")    
        
        # call dalstm_load_dataset for the whole dataset
        (_, _, _), values = self.dalstm_load_dataset(filename=full_dataset_path)
        # call dalstm_load_dataset for training, validation, and test sets
        (X_train,y_train, train_lengths
         ), _ =  self.dalstm_load_dataset(filename=train_dataset_path,
                                     prev_values=values)
        (X_val, y_val, valid_lengths), _ = self.dalstm_load_dataset(
            filename=val_dataset_path, prev_values=values)
        (X_test, y_test, test_lengths), _ = self.dalstm_load_dataset(
            filename=test_dataset_path, prev_values=values)
            
        # normalize input data
        # compute the normalization values only on training set
        max = [0] * len(X_train[0][0])
        for a1 in X_train:
            for s in a1:
                for i in range(len(s)):
                    if s[i] > max[i]:
                        max[i] = s[i]
        # normalization for train, validation, and test sets
        for a1 in X_train:
            for s in a1:
                for i in range(len(s)):
                    if (max[i] > 0):
                        s[i] = s[i] / max[i]
        for a1 in X_val:
            for s in a1:
                for i in range(len(s)):
                    if (max[i] > 0):
                        s[i] = s[i] / max[i]
        for a1 in X_test:
            for s in a1:
                for i in range(len(s)):
                    if (max[i] > 0):
                        s[i] = s[i] / max[i]
        
        # convert the results to numpy arrays
        X_train = np.asarray(X_train, dtype='object')
        X_val = np.asarray(X_val, dtype='object')
        X_test = np.asarray(X_test, dtype='object')
        y_train = np.asarray(y_train)
        y_val = np.asarray(y_val)
        y_test = np.asarray(y_test)
        # execute padding, and error handling for BPIC13I
        if self.dataset_name == 'BPI_2013_I':
            X_train = sequence.pad_sequences(X_train, dtype="int16")
            X_test = sequence.pad_sequences(X_test, maxlen=X_train.shape[1], 
                                            dtype="int16")
            X_val = sequence.pad_sequences(X_val, maxlen=X_train.shape[1],
                                           dtype="int16")
        else:
            X_train = sequence.pad_sequences(X_train)
            X_test = sequence.pad_sequences(X_test, maxlen=X_train.shape[1])
            X_val = sequence.pad_sequences(X_val, maxlen=X_train.shape[1])
        # Convert target attribute to days
        y_train /= (24*3600) 
        y_val /= (24*3600) 
        y_test /= (24*3600) 
        # Target attribute normalization
        if self.normalization:
            # get the maximum value for remaining time in train and val sets.
            max_y_train = np.max(y_train)
            max_y_val = np.max(y_val)
            max_train_val = np.max([max_y_train, max_y_val])         
            #print(max_train_val)
            # concatenate remaining times for training and validations sets
            y_train_val = np.concatenate([y_train, y_val])
            # get mean and median of remaining time for CARD model
            mean_target_value = np.mean(y_train_val)
            median_target_value = np.median(y_train_val)            
            # min-max normalization (we assume min equals zero)
            y_train /= max_train_val
            y_val /= max_train_val
            y_test /= max_train_val
        else:
            max_train_val = None
            mean_target_value = None
            median_target_value = None
            
            
        # convert numpy arrays to tensors
        # manage disk space for huge event logs
        if (('BPIC15' in self.dataset_name) or 
            (self.dataset_name== 'Traffic_Fine') or
            (self.dataset_name== 'Hospital')):
            X_train = torch.tensor(X_train).type(torch.bfloat16)
            X_val = torch.tensor(X_val).type(torch.bfloat16)
            X_test = torch.tensor(X_test).type(torch.bfloat16)
        else:
            X_train = torch.tensor(X_train).type(torch.float)
            X_val = torch.tensor(X_val).type(torch.float)
            X_test = torch.tensor(X_test).type(torch.float)
        y_train = torch.tensor(y_train).type(torch.float)
        y_val = torch.tensor(y_val).type(torch.float)
        y_test = torch.tensor(y_test).type(torch.float)
        input_size = X_train.size(2)
        max_len = X_train.size(1) 
        # save training, validation, test tensors
        torch.save(X_train, X_train_path)                  
        torch.save(X_val, X_val_path)
        torch.save(X_test, X_test_path)                      
        torch.save(y_train, y_train_path)
        torch.save(y_val, y_val_path)
        torch.save(y_test, y_test_path)
        # save test prefix lengths
        with open(test_length_path, 'wb') as file:
            pickle.dump(test_lengths, file)
        # save normalization constant (maximum remaining time in training data)
        with open(scaler_path, 'wb') as file:
            pickle.dump(max_train_val, file)
        # Save mean and median for remaining time
        with open(target_mean_path, 'wb') as file:
            pickle.dump(mean_target_value, file)
        with open(target_median_path, 'wb') as file:
            pickle.dump(median_target_value, file)
        # save max_len, input_size to be used in the definition of model
        with open(input_size_path, 'wb') as file:
            pickle.dump(input_size, file)
        with open(max_len_path, 'wb') as file:
            pickle.dump(max_len, file)
        # Delete csv files as they are not require anymore
        self.delete_files(extension='.csv')
        # Now, we create train, valid, test splits for cross-validation
        # Put all prefixes in one dataset
        X_total = torch.cat((X_train, X_val, X_test), dim=0)
        y_total = torch.cat((y_train, y_val, y_test), dim=0)
        total_lengths = train_lengths + valid_lengths + test_lengths
        # get indices for train, validation, and test
        n_samples = X_total.shape[0]
        splits={}
        kf = KFold(n_splits=self.n_splits, shuffle=True,
                   random_state=self.seed)
        kf_split = kf.split(np.zeros(n_samples)) 
        for i, (_, ids) in enumerate(kf_split):
            splits[i] = ids.tolist()
        for split_key in range(self.n_splits):
            test_ids = splits[split_key]
            val_ids = splits[((split_key + 1) % self.n_splits)]      
            train_ids = []
            for fold in range(self.n_splits):
                if fold != split_key and fold != (split_key + 1) % self.n_splits: 
                    train_ids.extend(splits[fold]) 
            # now get training, validation, and test prefixes
            X_train = X_total[train_ids]
            y_train = y_total[train_ids]
            X_val = X_total[val_ids]
            y_val = y_total[val_ids]
            X_test = X_total[test_ids]
            y_test = y_total[test_ids]
            test_lengths = [total_lengths[i] for i in test_ids]          
            # define file names, and paths 
            X_train_path = os.path.join(
                self.dataset_path,
                "DALSTM_X_train_fold_"+str(split_key+1)+self.dataset_name+".pt")
            X_val_path = os.path.join(
                self.dataset_path,
                "DALSTM_X_val_fold_"+str(split_key+1)+self.dataset_name+".pt")
            X_test_path = os.path.join(
                self.dataset_path,
                "DALSTM_X_test_fold_"+str(split_key+1)+self.dataset_name+".pt")
            y_train_path = os.path.join(
                self.dataset_path,
                "DALSTM_y_train_fold_"+str(split_key+1)+self.dataset_name+".pt")
            y_val_path = os.path.join(
                self.dataset_path,
                "DALSTM_y_val_fold_"+str(split_key+1)+self.dataset_name+".pt")
            y_test_path = os.path.join(
                self.dataset_path,
                "DALSTM_y_test_fold_"+str(split_key+1)+self.dataset_name+".pt")        
            test_length_path = os.path.join(
                self.dataset_path,
                "DALSTM_test_length_list_fold_"+str(
                    split_key+1)+self.dataset_name+".pkl")
            # save training, validation, test tensors   
            torch.save(X_train, X_train_path) 
            torch.save(X_val, X_val_path)
            torch.save(X_test, X_test_path)
            torch.save(y_train, y_train_path)
            torch.save(y_val, y_val_path)
            torch.save(y_test, y_test_path)
            # save lengths
            with open(test_length_path, 'wb') as file:
                pickle.dump(test_lengths, file)  
        print('Preprocessing is done for both holdout and CV data split.')      