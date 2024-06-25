import os
import os.path as osp
import shutil
import pickle
import torch
from tqdm import tqdm
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_zip)

# TODO: update the uploaded datasets which include more target attributes
# A set of url addresses for downloading graph datasets
helpdesk_url = 'https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/helpdesk_graph_raw.zip'
bpic15_1_url = 'https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/bpic15m1_graph_raw.zip'
bpic15_2_url = 'https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/bpic15m2_graph_raw.zip'
bpic15_3_url = 'https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/bpic15m3_graph_raw.zip' 
bpic15_4_url = 'https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/bpic15m4_graph_raw.zip'
bpic15_5_url = 'https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/bpic15m5_graph_raw.zip' 
bpic12_url = 'https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/bpi12_graph_raw.zip'
bpic12_a_url ='https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/bpi12a_graph_raw.zip'  
bpic12_c_url = 'https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/bpi12c_graph_raw.zip' 
bpic12_cw_url = 'https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/bpi12cw_graph_raw.zip'
bpic12_o_url = 'https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/bpi12o_graph_raw.zip'
bpic12_w_url = 'https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/bpi12w_graph_raw.zip'
bpic13_c_url = 'https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/bpic13c_graph_raw.zip'
bpic13_i_url = 'https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/bpic13i_graph_raw.zip'
bpic20_d_url = 'https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/bpi20d_graph_raw.zip' 
bpic20_i_url = 'https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/bpi20i_graph_raw.zip'
#evnpermit_url = 'https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/envpermit_graph_raw.zip'
evnpermit_url = 'https://github.com/keyvan-amiri/PGTNet/raw/main/transfordmation/envpermit_graph_raw.zip'
hospital_url ='https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/hospital_graph_raw.zip'
sepsis_url = 'https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/sepsis_graph_raw.zip'
trafficfine_url = 'https://github.com/keyvan-amiri/PGTNet/raw/main/transformation/trafficfines_graph_raw.zip' 


##############################################################################
#################### A base class for PGTNet datasets ########################
##############################################################################

class BasePGTNetDataset(InMemoryDataset):
    def __init__(self, root, url, name, folder_name, split='train',
                 transform=None, pre_transform=None, pre_filter=None):
        self.url = url
        self.name = name
        self.folder_name = folder_name
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, self.folder_name), self.raw_dir)
        os.unlink(path)
    
    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')
            
            data_list = []
            for idx in indices:
                graph = graphs[idx]
                x = graph.x
                edge_attr = graph.edge_attr
                edge_index = graph.edge_index
                y = graph.y
                rem_time = graph.rem_time
                next_time = graph.next_time
                next_act = graph.next_act                
                cid = graph.cid
                pl = graph.pl

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                            y=y, rem_time=rem_time, next_time=next_time,
                            next_act=next_act, cid=cid, pl=pl)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(self.collate(data_list), osp.join(self.processed_dir, 
                                                         f'{split}.pt'))  

##############################################################################
################# separate classes for different datasets ####################
##############################################################################

class PGTNet_HelpDesk(BasePGTNetDataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        super().__init__(root, helpdesk_url, 'EVENTBPIC15M1', 'helpdesk_graph_raw',
                         split, transform, pre_transform, pre_filter)            

class PGTNet_BPIC15M1(BasePGTNetDataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        super().__init__(root, bpic15_1_url, 'PGTNet_BPIC15M1', 'bpic15m1_graph_raw',
                         split, transform, pre_transform, pre_filter)     
    

class PGTNet_BPIC15M2(BasePGTNetDataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        super().__init__(root, bpic15_2_url, 'PGTNet_BPIC15M2', 'bpic15m2_graph_raw',
                         split, transform, pre_transform, pre_filter)  

class PGTNet_BPIC15M3(BasePGTNetDataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        super().__init__(root, bpic15_3_url, 'PGTNet_BPIC15M3', 'bpic15m3_graph_raw',
                         split, transform, pre_transform, pre_filter)    

class PGTNet_BPIC15M4(BasePGTNetDataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        super().__init__(root, bpic15_4_url, 'PGTNet_BPIC15M4', 'bpic15m4_graph_raw',
                         split, transform, pre_transform, pre_filter)  

class PGTNet_BPIC15M5(BasePGTNetDataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        super().__init__(root, bpic15_5_url, 'PGTNet_BPIC15M5', 'bpic15m5_graph_raw',
                         split, transform, pre_transform, pre_filter)  

class PGTNet_BPIC12(BasePGTNetDataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        super().__init__(root, bpic12_url, 'PGTNet_BPIC12', 'bpi12_graph_raw',
                         split, transform, pre_transform, pre_filter)
        
class PGTNet_BPIC12A(BasePGTNetDataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        super().__init__(root, bpic12_a_url, 'PGTNet_BPIC12A', 'bpi12a_graph_raw',
                         split, transform, pre_transform, pre_filter) 

class PGTNet_BPIC12C(BasePGTNetDataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        super().__init__(root, bpic12_c_url, 'PGTNet_BPIC12C', 'bpi12c_graph_raw',
                         split, transform, pre_transform, pre_filter)  
        
class PGTNet_BPIC12CW(BasePGTNetDataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        super().__init__(root, bpic12_cw_url, 'PGTNet_BPIC12CW', 'bpi12cw_graph_raw',
                         split, transform, pre_transform, pre_filter) 
        
class PGTNet_BPIC12O(BasePGTNetDataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        super().__init__(root, bpic12_o_url, 'PGTNet_BPIC12O', 'bpi12o_graph_raw',
                         split, transform, pre_transform, pre_filter) 
        
class PGTNet_BPIC12W(BasePGTNetDataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        super().__init__(root, bpic12_w_url, 'PGTNet_BPIC12W', 'bpi12w_graph_raw',
                         split, transform, pre_transform, pre_filter) 
        
class PGTNet_BPIC13C(BasePGTNetDataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        super().__init__(root, bpic13_c_url, 'PGTNet_BPIC13C', 'bpic13c_graph_raw',
                         split, transform, pre_transform, pre_filter) 
        
class PGTNet_BPIC13I(BasePGTNetDataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        super().__init__(root, bpic13_i_url, 'PGTNet_BPIC13I', 'bpic13i_graph_raw',
                         split, transform, pre_transform, pre_filter) 
        
class PGTNet_BPIC20D(BasePGTNetDataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        super().__init__(root, bpic20_d_url, 'PGTNet_BPIC20D', 'bpi20d_graph_raw',
                         split, transform, pre_transform, pre_filter) 
        
class PGTNet_BPIC20I(BasePGTNetDataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        super().__init__(root, bpic20_i_url, 'PGTNet_BPIC20I', 'bpi20i_graph_raw' ,
                         split, transform, pre_transform, pre_filter) 
        
class PGTNet_env_permit(BasePGTNetDataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        super().__init__(root, evnpermit_url, 'PGTNet_env_permit', 'envpermit_graph_raw',
                         split, transform, pre_transform, pre_filter) 
        
class PGTNet_Hospital(BasePGTNetDataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        super().__init__(root, hospital_url, 'PGTNet_Hospital', 'hospital_graph_raw',
                         split, transform, pre_transform, pre_filter) 

class PGTNet_Sepsis(BasePGTNetDataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        super().__init__(root, sepsis_url, 'PGTNet_Sepsis', 'sepsis_graph_raw',
                         split, transform, pre_transform, pre_filter) 
        
class PGTNet_Trafficfines(BasePGTNetDataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        super().__init__(root, trafficfine_url, 'PGTNet_Trafficfines',
                         'trafficfines_graph_raw', split, transform,
                         pre_transform, pre_filter) 