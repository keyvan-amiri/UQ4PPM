import yaml
import os
import os.path as osp
import numpy as np
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
import bisect
import random
from sklearn.preprocessing import OneHotEncoder
import torch
from torch_geometric.data import Data
import pickle

class XES_Fields:
    CASE_COLUMN = 'case:concept:name'
    ACTIVITY_COLUMN = 'concept:name'
    TIMESTAMP_COLUMN = 'time:timestamp'
    LIFECYCLE_COLUMN = 'lifecycle:transition'
    RESOURCE_COLUMN = 'org:resource'    

# PGTNet convertor class for case-centric event logs
class PGTNet_convertor_case_centric ():   
    
    def __init__ (self, xes_dir=None, graph_dir=None, conversion_cfg=None,
                  dataset_name=None, split_ratio=None, normalization=None,
                  normalization_type=None, overwrite=None):
        
        root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
        if xes_dir is None:
            xes_dir = os.path.join(root_path, 'datasets', 'event_logs')
        event_log_path = os.path.join(xes_dir, dataset_name+'.xes')
        # read the conversion config file
        if conversion_cfg is None:
            conversion_cfg = os.path.join(root_path,'datasets',
                                          'conversion_cfg.yaml')        
        with open(conversion_cfg, 'r') as f:
            guide_dict = yaml.safe_load(f)
        data_guide_dict = guide_dict[dataset_name]
        # set default columns
        if not('case_column' in data_guide_dict):
            self.case_column = XES_Fields.CASE_COLUMN           
        else:
            self.case_column = data_guide_dict.case_column
        if not('activity_column' in data_guide_dict):
            self.activity_column = XES_Fields.ACTIVITY_COLUMN            
        else:
            self.activity_column = data_guide_dict.activity_column
        if not('timestamp_column' in data_guide_dict):
            self.timestamp_column = XES_Fields.TIMESTAMP_COLUMN            
        else:
            self.timestamp_column = data_guide_dict.timestamp_column
        if not('lifecycle_column' in data_guide_dict):
            self.lifecycle_column = XES_Fields.LIFECYCLE_COLUMN            
        else:
            self.lifecycle_column = data_guide_dict.lifecycle_column
        if not('resource_column' in data_guide_dict):
            self.resource_column = XES_Fields.RESOURCE_COLUMN            
        else:
            self.resource_column = data_guide_dict.resource_column
        # set data attributes to be included
        self.event_attributes = data_guide_dict.get('event_categorical', [])
        self.event_num_att = data_guide_dict.get('event_numerical', [])
        case_attributes = data_guide_dict.get('case_categorical', [])
        case_num_att = data_guide_dict.get('case_numerical', [])
        self.case_attributes , self.case_num_att = [], []
        for att in case_attributes:
            self.case_attributes.append('case:' + att)
        for att in case_num_att:
            self.case_num_att.append('case:' + att)        
        # required variables for transformation
        if graph_dir is None:
            self.graph_dir = os.path.join(root_path, 'datasets')
        else:
            self.graph_dir = graph_dir
        self.graph_class = 'PGTNet_' + dataset_name
        self.dataset_path =  os.path.join(self.graph_dir, self.graph_class)
        self.dataset_raw =  os.path.join(self.dataset_path, 'raw')
        self.dataset_processed =  os.path.join(self.dataset_path, 'processed')
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
        if not os.path.exists(self.dataset_raw):
            os.makedirs(self.dataset_raw)
        if not os.path.exists(self.dataset_processed):
            os.makedirs(self.dataset_processed)        
        self.save_addresses = ['train.pickle', 'val.pickle', 'test.pickle']        
        self.out_add0 = os.path.join(self.dataset_raw, self.save_addresses[0])
        self.out_add1 = os.path.join(self.dataset_raw, self.save_addresses[1])
        self.out_add2 = os.path.join(self.dataset_raw, self.save_addresses[2])
        self.overwrite = overwrite if overwrite is not None else False 
        if (not self.overwrite and os.path.exists(self.out_add0) and
            os.path.exists(self.out_add1) and os.path.exists(self.out_add2)):
            print(f"For '{dataset_name}' graph dataset is already available.")
            # to not continue with transformation
            return           
        self.split_ratio = split_ratio if split_ratio is not None\
            else [0.64,0.16,0.20]   
        self.normalization = normalization if normalization is not None\
            else True 
        self.normalization_type = normalization_type\
            if normalization_type is not None else 'max_norm'
        
        # import the event log
        self.log = xes_importer.apply(event_log_path)
        self.event_log = pm4py.read_xes(event_log_path)     
        # split the dataset (Holdout split w.r.t time horizon)
        (self.train_df, self.val_df, self.test_df, self.train_val_df,
         train_log, val_log, self.test_log, self.train_val_log,
         self.max_time_norm, self.sorted_start_dates,
         self.sorted_end_dates) = self.log_split()
        # get global variables
        (self.node_class_dict, self.max_case_df, self.max_active_cases,
         self.min_num_list, self.avg_num_list, self.max_num_list,
         self.event_min_num_list, self.event_avg_num_list,
         self.event_max_num_list, self.attribute_encoder_list,
         self.case_encoder_list, self.node_dim,
         self.edge_dim, self.activity_dict) = self.global_stat()
        # Convert prefixes into directed attributed graphs
        self.removed_cases = [] # (list of cases with length less than 3)
        self.idx = 0 # index for graphs
        self.data_list = [] # a list of all Pytorch geometric data objects.
        # create training set  
        self.conversion(train_log)
        self.last_train_idx = self.idx
        # create validation set
        self.conversion(val_log)
        self.last_val_idx = self.idx
        # create test set
        self.conversion(self.test_log)
        # create indices for dataset        
        indices = list(range(len(self.data_list)))
        # data split based on the global graph list
        self.train_indices = indices[:self.last_train_idx]
        self.val_indices = indices[self.last_train_idx:self.last_val_idx]
        self.test_indices = indices[self.last_val_idx:] 
        self.data_train = [self.data_list[i] for i in self.train_indices]
        self.data_val = [self.data_list[i] for i in self.val_indices]
        self.data_test = [self.data_list[i] for i in self.test_indices]
        # shuffle all splits, to avoid order affect training process
        random.shuffle(self.data_train)
        random.shuffle(self.data_val)
        random.shuffle(self.data_test)
        # Save the training, validation, and test datasets
        dataset_parts = [self.data_train, self.data_val, self.data_test] 
        for address_counter in range(len(self.save_addresses)):
            save_address = osp.join(self.dataset_raw,
                                    self.save_addresses[address_counter])
            save_flie = open(save_address, "wb")
            pickle.dump(dataset_parts[address_counter], save_flie)
            save_flie.close() 
    
    # Get the number of active cases (given a timestamp)
    def ActiveCase(self, L1, L2, T):
        num_cases = bisect.bisect_right(L1, T) - bisect.bisect_right(L2, T)
        return num_cases
    
    def log_split(self):
        # Split the event log into training, validation, and test sets
        start_dates, end_dates, durations, case_ids = [], [], [], []
        train_validation_index = int(len(self.log) * (
            self.split_ratio[0]+self.split_ratio[1]))
        train_index = int(len(self.log) * self.split_ratio[0])
        for i in range (len(self.log)):
            current_case = self.log[i]
            current_length = len(current_case)
            start_dates.append(current_case[0].get(self.timestamp_column))
            end_dates.append(current_case[current_length-1].get(
                self.timestamp_column))
            durations.append((current_case[current_length-1].get(
                self.timestamp_column) - current_case[0].get(
                    self.timestamp_column)).total_seconds()/3600/24)
            short_case_column = self.case_column.replace('case:', '', 1)
            case_ids.append(current_case.attributes.get(short_case_column))
        combined_data = list(zip(start_dates, end_dates, durations, case_ids))
        sorted_data = sorted(combined_data, key=lambda x: x[0])
        (sorted_start_dates, sorted_end_dates, sorted_durations,
         sorted_case_ids) = zip(*sorted_data)
        train_case_ids = sorted_case_ids[:train_index]
        validation_case_ids = sorted_case_ids[
            train_index:train_validation_index]
        train_validation_case_ids = sorted_case_ids[:train_validation_index]
        test_case_ids = sorted_case_ids[train_validation_index:]
        train_validation_durations = sorted_durations[:train_validation_index]
        max_case_duration = max(train_validation_durations)
        training_dataframe = pm4py.filter_trace_attribute_values(
            self.event_log, self.case_column, train_case_ids, 
            case_id_key=self.case_column)
        validation_dataframe = pm4py.filter_trace_attribute_values(
            self.event_log, self.case_column, validation_case_ids,
            case_id_key=self.case_column)
        test_dataframe = pm4py.filter_trace_attribute_values(
            self.event_log, self.case_column, test_case_ids,
            case_id_key=self.case_column)
        train_validation_dataframe = pm4py.filter_trace_attribute_values(
            self.event_log, self.case_column, train_validation_case_ids,
            case_id_key=self.case_column)
        training_event_log = pm4py.convert_to_event_log(training_dataframe)
        validation_event_log = pm4py.convert_to_event_log(validation_dataframe)
        test_event_log = pm4py.convert_to_event_log(test_dataframe)
        train_validation_event_log = pm4py.convert_to_event_log(
            train_validation_dataframe)
        return (training_dataframe, validation_dataframe, test_dataframe,
                train_validation_dataframe, training_event_log,
                validation_event_log, test_event_log,
                train_validation_event_log, max_case_duration,
                sorted_start_dates, sorted_end_dates)   

    def global_stat(self):
        """
        Get global information required for transformation
        only use training and validation sets!
        use all observed values for categorical attributes (one-hot encoding)
        The outputs are:
        node_class_dict (event classes):
            keys: tuple of activity/lifecycle, value: integer representation
        max_case_df: maximum number of same df relationship in one case
        max_active_cases: maximum concurrent cases in training-validation sets
        min_num_list: minimum values for numerical case attributes
        avg_num_list: average values for numerical case attributes
        max_num_list: maximum values for numerical case attributes
        event_min_num_list: minimum values for numerical event attributes
        event_avg_num_list: average values for numerical event attributes
        event_max_num_list: maximum values for numerical event attributes
        attribute_encoder_list: encoders to handel event categorical attributes
        case_encoder_list: encoders to handel case categorical attributes
        node_dim: dimension for node attributes
        edge_dim: dimension for edge attributes
        """
        max_case_df, node_class_rep = 0, 0
        node_class_dict = {}
        start_dates, end_dates = [], []
        # Get node_class_dict, max_case_df
        for case_counter in range (len(self.train_val_log)):
            # Dict: all activity-class df relationships and their frequencies
            df_dict = {}
            current_case = self.train_val_log[case_counter]
            case_length = len(current_case)
            if case_length > 1:
                # iterate over all events of the case, collect df information
                for event_counter in range(case_length-1): 
                    source_class = (
                        current_case[event_counter].get(self.activity_column),
                        current_case[event_counter].get(
                            self.lifecycle_column))
                    target_class = (
                        current_case[event_counter+1].get(
                            self.activity_column),
                        current_case[event_counter+1].get(
                            self.lifecycle_column))
                    df_class = (source_class, target_class)
                    if df_class in df_dict:
                        df_dict[df_class] += 1
                    else:
                        df_dict[df_class] = 1
                    if not (source_class in node_class_dict):
                        node_class_dict[source_class] = node_class_rep
                        node_class_rep += 1                                 
                if max((df_dict).values()) > max_case_df:
                    max_case_df = max((df_dict).values())
        # repeat for test set but without updating max_case_df
        for case_counter in range (len(self.test_log)):
            df_dict = {}
            current_case = self.test_log[case_counter]
            case_length = len(current_case)
            if case_length > 1:
                for event_counter in range(case_length-1): 
                    source_class = (
                        current_case[event_counter].get(self.activity_column),
                        current_case[event_counter].get(self.lifecycle_column))
                    target_class = (
                        current_case[event_counter+1].get(
                            self.activity_column),
                        current_case[event_counter+1].get(
                            self.lifecycle_column))
                    df_class = (source_class, target_class)
                    if df_class in df_dict:
                        df_dict[df_class] += 1
                    else:
                        df_dict[df_class] = 1
                    if not (source_class in node_class_dict):
                        node_class_dict[source_class] = node_class_rep
                        node_class_rep += 1 

        # Iterate over train-val data, get list of start and end dates,
        # then, use them to get max_active_cases
        for case_counter in range (len(self.train_val_log)):
            current_case = self.train_val_log[case_counter]
            case_length = len(current_case)
            start_dates.append(current_case[0].get(self.timestamp_column))
            end_dates.append(current_case[case_length-1].get(
                self.timestamp_column))          
        sorted_start_dates = sorted(start_dates)
        sorted_end_dates = sorted(end_dates)
        max_active_cases = 0
        unique_timestamps = list(self.train_val_df[
            self.timestamp_column].unique())
        for any_time in unique_timestamps:
            cases_in_system = self.ActiveCase(sorted_start_dates,
                                              sorted_end_dates, any_time)
            if cases_in_system > max_active_cases:
                max_active_cases = cases_in_system

        # handle numerical case and event attributes
        # get 3 lists for min/max/avg values for each attribute
        min_num_list, max_num_list, avg_num_list = [], [], []
        case_card = len(self.case_num_att)
        for num_att in self.case_num_att:
            unique_values = self.train_val_df[num_att].dropna().tolist()
            unique_values_float = [float(val) for val in unique_values]
            min_num_list.append(float(min(unique_values_float)))
            max_num_list.append(float(max(unique_values_float)))
            avg_num_list.append(self.train_val_df[num_att].mean())
        # get 3 lists for min/max/avg values for each attribute
        event_min_num_list, event_max_num_list, event_avg_num_list = [], [], []
        event_card = len(self.event_num_att)
        for num_att in self.event_num_att:
            unique_values = self.train_val_df[num_att].dropna().tolist()
            unique_values_float = [float(val) for val in unique_values]
            event_min_num_list.append(float(min(unique_values_float)))
            event_max_num_list.append(float(max(unique_values_float)))
            event_avg_num_list.append(self.train_val_df[num_att].mean())
        
        # handle categorical case and event attributes
        # List of one-hot encoders: categorical event attributes of intrest
        attribute_encoder_list = []
        event_cat_card = 0
        for event_attribute in self.event_attributes:
            unique_values = list(self.event_log[event_attribute].unique())
            att_array = np.array(unique_values)
            att_enc = OneHotEncoder(handle_unknown='ignore')
            att_enc.fit(att_array.reshape(-1, 1))
            attribute_encoder_list.append(att_enc)
            event_cat_card += len(unique_values)
        # List of one-hot encoders (for case attributes of intrest)
        case_encoder_list = []
        case_cat_dim = 0
        for case_attribute in self.case_attributes:
            unique_values = list(self.event_log[case_attribute].unique())
            att_array = np.array(unique_values)
            att_enc = OneHotEncoder(handle_unknown='ignore')
            att_enc.fit(att_array.reshape(-1, 1))
            case_encoder_list.append(att_enc)
            case_cat_dim += len(unique_values)
            
        # Get node and edge dimensions
        node_dim = len(node_class_dict.keys()) # size for node featuers
        edge_dim  = event_cat_card + case_cat_dim + case_card + event_card + 7
        
        #create a dictionary for next activity prediction target attribute
        activity_dict = {}
        unique_acitivities = self.event_log[self.activity_column].unique()        
        for index, value in enumerate(unique_acitivities):
            activity_dict[value] = index 
        
        return node_class_dict, max_case_df, max_active_cases, min_num_list,\
            avg_num_list, max_num_list, event_min_num_list, event_avg_num_list,\
                event_max_num_list, attribute_encoder_list, case_encoder_list,\
                    node_dim, edge_dim, activity_dict
                    
    # method for converting an event log into graph dataset
    def conversion (self, split_log):        
        # iterate over cases, transform them if they have at least three events
        for case_counter in range(len(split_log)):
            current_case = split_log[case_counter]
            short_case_column = self.case_column.replace('case:', '', 1)
            case_id = split_log[case_counter].attributes.get(short_case_column)   
            case_length = len(current_case)
            if case_length < 3:
                self.removed_cases.append(case_id)
            else:    
                # collect all case-level information
                case_level_feat = np.empty((0,))                
                # first categorical attributes
                for att_index in range(len(self.case_attributes)):
                    case_att = split_log[case_counter].attributes.get(
                        self.case_attributes[att_index])
                    case_att = str(case_att)
                    case_att_enc = self.case_encoder_list[att_index].transform(
                        [[case_att]]).toarray()
                    case_att_enc = case_att_enc.reshape(-1)
                    case_level_feat = np.append(case_level_feat, case_att_enc)                
                # now, numerical attributes
                for att_index in range(len(self.case_num_att)):
                    case_att = float(split_log[case_counter].attributes.get(
                        self.case_num_att[att_index]))
                    # impute NaN values with average value
                    if np.isnan(case_att):
                        case_att_normalized = (
                            self.avg_num_list[att_index] - self.min_num_list[
                                att_index])/(self.max_num_list[
                                    att_index]- self.min_num_list[att_index])
                    else:
                        case_att_normalized = (
                            case_att - self.min_num_list[att_index])/(
                                self.max_num_list[
                                    att_index]- self.min_num_list[att_index])
                    case_level_feat = np.append(
                        case_level_feat, np.array(case_att_normalized)) 
                
                # collect all events of the case, compute case start, end time
                case_events = split_log[case_counter][:]  
                case_start = split_log[case_counter][0].get(
                    self.timestamp_column) 
                case_end = split_log[case_counter][case_length-1].get(
                    self.timestamp_column)            
                # for each event collect the follwoing info in a list of lists:
                # event class, timestamp, activity, other attributes of intrest
                collection_lists = [[] for _ in range(
                    len(self.event_attributes)+len(self.event_num_att)+3)]
                for event_index in range(case_length):            
                    current_event = case_events[event_index]
                    collection_lists[0].append((current_event.get(
                        self.activity_column),current_event.get(
                            self.lifecycle_column)))
                    collection_lists[1].append(
                        current_event.get(self.timestamp_column))
                    collection_lists[2].append(
                        current_event.get(self.activity_column))
                    for attribute_counter in range (3,len(
                            self.event_attributes)+3):
                        collection_lists[attribute_counter].append(
                            current_event.get(
                                self.event_attributes[attribute_counter-3]))
                    for attribute_counter in range (
                            len(self.event_attributes)+3,
                            len(self.event_attributes)+len(
                                self.event_num_att)+3):
                        collection_lists[attribute_counter].append(
                            current_event.get(self.event_num_att[
                                attribute_counter-len(
                                    self.event_attributes)-3]))
            
                # create a graph for each prefix (lengths: 2, case length-1)
                for prefix_length in range (2, case_length):
                    prefix_event_classes = collection_lists[0][:prefix_length]
                    # only includes unique classes
                    prefix_classes = list(set(prefix_event_classes)) 
                    prefix_times = collection_lists[1][:prefix_length]
                    # Define nodes and their features
                    # define zero array to collect node features 
                    node_feature = np.zeros((len(prefix_classes), 1),
                                            dtype=np.int64)
                    # collect node type: iterate over all nodes in the graph.
                    for prefix_class in prefix_classes:
                        # get index of the relevant prefix class
                        # then, update its row in node feature matirx         
                        node_feature[prefix_classes.index(
                            prefix_class)] = self.node_class_dict[prefix_class]
                    x = torch.from_numpy(node_feature).long()
                    # Compute edge index list.
                    # Each item in pair_result:
                    # tuple of tuples representing df between two event classes 
                    pair_result = list(zip(prefix_event_classes,
                                           prefix_event_classes [1:]))
                    pair_freq = {}            
                    for item in pair_result:
                        source_index = prefix_classes.index(item[0])
                        target_index = prefix_classes.index(item[1])
                        if ((source_index, target_index) in pair_freq):
                            pair_freq[(source_index, target_index)] += 1
                        else:
                            pair_freq[(source_index, target_index)] = 1
                    edges_list = list(pair_freq.keys())
                    edge_index = torch.tensor(edges_list, dtype=torch.long)
                    # Compute edge attributes
                    # initialize edge feature matrix
                    edge_feature = np.zeros((len(edge_index), self.edge_dim),
                                            dtype=np.float64) 
                    edge_counter = 0
                    for edge in edge_index:
                        source_indices = [i for i, x in enumerate(
                            prefix_event_classes) if x == prefix_classes[edge[0]]]
                        target_indices = [i for i, x in enumerate(
                            prefix_event_classes) if x == prefix_classes[edge[1]]]
                        acceptable_indices = [(x, y) for x in source_indices \
                                              for y in target_indices if x + 1 == y]
                        # collect all special features
                        special_feat = np.empty((0,)) 
                        # Add edge weights to the special feature vector
                        num_occ = len(acceptable_indices)/self.max_case_df
                        special_feat = np.append(special_feat,
                                                 np.array(num_occ))
                        # Add temporal features to the special feature vector
                        sum_dur = 0
                        for acceptable_index in acceptable_indices:
                            #TODO: implement log normalization for this feature
                            last_dur = (prefix_times[
                                acceptable_index[1]]- prefix_times[
                                    acceptable_index[0]]
                                    ).total_seconds(
                                        )/3600/24/self.max_time_norm
                            sum_dur += last_dur
                        special_feat = np.append(special_feat,
                                                 np.array(last_dur))
                        special_feat = np.append(special_feat,
                                                 np.array(sum_dur))
                        # only meaningful for the latest event in prefix
                        if acceptable_indices[-1][1] == prefix_length-1:
                            #TODO: implement log normalization for this feature
                            temp_feat1 = (prefix_times[
                                acceptable_indices[-1][1]]-case_start
                                ).total_seconds()/3600/24/self.max_time_norm
                            temp_feat2 = prefix_times[
                                acceptable_indices[-1][1]
                                ].hour/24 + prefix_times[
                                    acceptable_indices[-1][1]
                                    ].minute/60/24 + prefix_times[
                                        acceptable_indices[-1][1]
                                        ].second/3600/24
                            temp_feat3 = (prefix_times[
                                acceptable_indices[-1][1]].weekday(
                                    ) + temp_feat2)/7
                        else:
                            temp_feat1 = temp_feat2 = temp_feat3 = 0
                        special_feat = np.append(special_feat,
                                                 np.array(temp_feat1))
                        special_feat = np.append(special_feat,
                                                 np.array(temp_feat2))
                        special_feat = np.append(special_feat,
                                                 np.array(temp_feat3))
                        # add workload features to special features
                        num_cases = self.ActiveCase(self.sorted_start_dates,
                                                    self.sorted_end_dates,
                                                    prefix_times[
                                                        acceptable_indices[-1][1]]
                                                    )/self.max_active_cases
                        special_feat = np.append(
                            special_feat, np.array(num_cases))
                        # add case-level features
                        partial_edge_feature = np.append(
                            special_feat, case_level_feat)                        
                        # For last occurence of target node:
                        # One-hot encoding + numerical event attributes
                        for attribute_counter in range (
                                3,len(self.event_attributes)+3):
                            attribute_value = np.array(collection_lists[
                                attribute_counter][acceptable_indices[-1][1]]
                                ).reshape(-1, 1)
                            if str(attribute_value[0][0]) == 'nan':
                                num_zeros = len(
                                    self.attribute_encoder_list[
                                        attribute_counter - 3].categories_[0])
                                onehot_att = np.zeros((len(attribute_value),
                                                       num_zeros))
                            else:
                                onehot_att = self.attribute_encoder_list[
                                    attribute_counter-3].transform(
                                        attribute_value).toarray()
                            # add categorical event attributes to edge features
                            partial_edge_feature = np.append(
                                partial_edge_feature, onehot_att)                        
                        # Numerical event attributes
                        for attribute_counter in range (
                                len(self.event_attributes)+3,len(
                                    self.event_attributes)+len(
                                        self.event_num_att)+3):
                            attribute_value = np.array(
                                collection_lists[attribute_counter][
                                    acceptable_indices[-1][1]])
                            if np.isnan(attribute_value):
                                norm_att_val = (self.event_avg_num_list[
                                    attribute_counter-len(self.event_attributes)-3
                                    ] - self.event_min_num_list[
                                        attribute_counter-len(
                                            self.event_attributes)-3])/(
                                                self.event_max_num_list[
                                                    attribute_counter-len(
                                                        self.event_attributes)-3
                                                    ]- self.event_min_num_list[
                                                        attribute_counter-len(
                                                            self.event_attributes)-3])
                            else:
                                norm_att_val = (
                                    attribute_value - self.event_min_num_list[
                                        attribute_counter-len(
                                            self.event_attributes)-3])/(
                                                self.event_max_num_list[
                                                    attribute_counter-len(
                                                        self.event_attributes)-3
                                                    ]- self.event_min_num_list[
                                                        attribute_counter-len(
                                                            self.event_attributes)-3])
                            # add numerical event attributes to edge features
                            partial_edge_feature = np.append(
                                partial_edge_feature, norm_att_val)
                        edge_feature[edge_counter, :] = partial_edge_feature
                        edge_counter += 1
                    edge_attr = torch.from_numpy(edge_feature).float() 
                    # Include next activity as a target attribute
                    next_act =  torch.tensor(
                        self.activity_dict[collection_lists[2][prefix_length]])                    
                    # Include next timestamp as a target attribute
                    # TODO: implement log normalization for next timestamp
                    if self.normalization:
                        next_t = np.array(
                            (collection_lists[1][prefix_length] - collection_lists[1][
                                prefix_length-1]).total_seconds()/3600/24/self.max_time_norm)
                    else:
                        next_t = np.array(
                            (collection_lists[1][prefix_length] - collection_lists[1][
                                prefix_length-1]).total_seconds()/3600/24)
                    next_timestamp = torch.from_numpy(next_t).float()                    
                    # TODO: implement next resource target attribute 
                    # Include remaining time as a target attribute
                    # TODO: implement log normalization for remaining time                   
                    if self.normalization:
                        target_cycle = np.array((
                            case_end - collection_lists[1][prefix_length-1]
                            ).total_seconds()/3600/24/self.max_time_norm)
                    else:    
                        target_cycle = np.array(
                            (case_end - collection_lists[1][prefix_length-1]
                             ).total_seconds()/3600/24)
                    remaining_time = torch.from_numpy(target_cycle).float()
                    # by default: y is equivalent to remaining time
                    y = remaining_time
                    # put everything together and create a graph
                    graph = Data(x=x, edge_index=edge_index.t().contiguous(),
                                 edge_attr=edge_attr, y=y,
                                 rem_time=remaining_time,
                                 next_time=next_timestamp, 
                                 next_act=next_act, cid=case_id,
                                 pl = prefix_length)
                    #print(graph)
                    self.data_list.append(graph)
                    self.idx += 1