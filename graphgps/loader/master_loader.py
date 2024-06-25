"""
To prepare this script we used the following source codes:
    https://github.com/rampasek/GraphGPS
We adjusted the source codes to efficiently integrate them into our framework.
"""

import logging
import os.path as osp
import time
from functools import partial
import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loader import load_pyg
from torch_geometric.graphgym.register import register_loader
from graphgps.loader.dataset.GTeventlogHandler import (
    PGTNet_HelpDesk, PGTNet_BPIC15M1, PGTNet_BPIC15M2, PGTNet_BPIC15M3,
    PGTNet_BPIC15M4, PGTNet_BPIC15M5, PGTNet_BPIC12, PGTNet_BPIC12A,
    PGTNet_BPIC12C, PGTNet_BPIC12CW, PGTNet_BPIC12O, PGTNet_BPIC12W,
    PGTNet_BPIC13C, PGTNet_BPIC13I, PGTNet_BPIC20D, PGTNet_BPIC20I,
    PGTNet_env_permit, PGTNet_Hospital, PGTNet_Sepsis, PGTNet_Trafficfines)
from graphgps.loader.split_generator import (prepare_splits,
                                             set_dataset_splits)
from graphgps.transform.posenc_stats import compute_posenc_stats
from graphgps.transform.task_preprocessing import task_specific_preprocessing
from graphgps.transform.transforms import pre_transform_in_memory


def log_loaded_dataset(dataset, format, name):
    logging.info(f"[*] Loaded dataset '{name}' from '{format}':")
    logging.info(f"  {dataset.data}")
    logging.info(f"  undirected: {dataset[0].is_undirected()}")
    logging.info(f"  num graphs: {len(dataset)}")

    total_num_nodes = 0
    if hasattr(dataset.data, 'num_nodes'):
        total_num_nodes = dataset.data.num_nodes
    elif hasattr(dataset.data, 'x'):
        total_num_nodes = dataset.data.x.size(0)
    logging.info(f"  avg num_nodes/graph: "
                 f"{total_num_nodes // len(dataset)}")
    logging.info(f"  num node features: {dataset.num_node_features}")
    logging.info(f"  num edge features: {dataset.num_edge_features}")
    if hasattr(dataset, 'num_tasks'):
        logging.info(f"  num tasks: {dataset.num_tasks}")

    if hasattr(dataset.data, 'y') and dataset.data.y is not None:
        if isinstance(dataset.data.y, list):
            # A special case for ogbg-code2 dataset.
            logging.info("num classes: n/a")
        elif dataset.data.y.numel() == dataset.data.y.size(0) and \
                torch.is_floating_point(dataset.data.y):
            logging.info("num classes: (appears to be a regression task)")
        else:
            logging.info(f"  num classes: {dataset.num_classes}")
    elif hasattr(dataset.data, 'train_edge_label') or hasattr(dataset.data,
                                                              'edge_label'):
        # Edge/link prediction task.
        if hasattr(dataset.data, 'train_edge_label'):
            labels = dataset.data.train_edge_label  # Transductive link task
        else:
            labels = dataset.data.edge_label  # Inductive link task
        if labels.numel() == labels.size(0) and \
                torch.is_floating_point(labels):
            logging.info("num edge classes: (probably a regression task)")
        else:
            logging.info(f"  num edge classes: {len(torch.unique(labels))}")

    ## Show distribution of graph sizes.
    # graph_sizes = [d.num_nodes if hasattr(d, 'num_nodes') else d.x.shape[0]
    #                for d in dataset]
    # hist, bin_edges = np.histogram(np.array(graph_sizes), bins=10)
    # logging.info(f'   Graph size distribution:')
    # logging.info(f'     mean: {np.mean(graph_sizes)}')
    # for i, (start, end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
    #     logging.info(
    #         f'     bin {i}: [{start:.2f}, {end:.2f}]: '
    #         f'{hist[i]} ({hist[i] / hist.sum() * 100:.2f}%)'
    #     )


@register_loader('custom_master_loader')
def load_dataset_master(format, name, dataset_dir):
    """
    Master loader that controls loading of all datasets, overshadowing execution
    of any default GraphGym dataset loader. Default GraphGym dataset loader are
    instead called from this function, the format keywords `PyG` and `OGB` are
    reserved for these default GraphGym loaders.
    Custom transforms and dataset splitting is applied to each loaded dataset.
    Args:
        format: dataset format name that identifies Dataset class
        name: dataset name to select from the class identified by `format`
        dataset_dir: path where to store the processed dataset
    Returns:
        PyG dataset object with applied perturbation transforms and data splits
    """
    if format.startswith('PyG-'):
        pyg_dataset_id = format.split('-', 1)[1]
        dataset_dir = osp.join(dataset_dir, pyg_dataset_id)
        
        # which dataset is at hand!
        if pyg_dataset_id == 'PGTNet_HelpDesk':
            dataset = preformat_pgtnet_dataset(PGTNet_HelpDesk, dataset_dir)
        elif pyg_dataset_id == 'PGTNet_BPIC12':
            dataset = preformat_pgtnet_dataset(PGTNet_BPIC12, dataset_dir)     
        elif pyg_dataset_id == 'PGTNet_BPIC12W':
            dataset = preformat_pgtnet_dataset(PGTNet_BPIC12W, dataset_dir)         
        elif pyg_dataset_id == 'PGTNet_BPIC12CW':
            dataset = preformat_pgtnet_dataset(PGTNet_BPIC12CW, dataset_dir)             
        elif pyg_dataset_id == 'PGTNet_BPIC12A':
            dataset = preformat_pgtnet_dataset(PGTNet_BPIC12A, dataset_dir)         
        elif pyg_dataset_id == 'PGTNet_BPIC12O':
            dataset = preformat_pgtnet_dataset(PGTNet_BPIC12O, dataset_dir)         
        elif pyg_dataset_id == 'PGTNet_BPIC12C':
            dataset = preformat_pgtnet_dataset(PGTNet_BPIC12C, dataset_dir)     
        elif pyg_dataset_id == 'PGTNet_BPIC13I':
            dataset = preformat_pgtnet_dataset(PGTNet_BPIC13I, dataset_dir)             
        elif pyg_dataset_id == 'PGTNet_BPIC13C':
            dataset = preformat_pgtnet_dataset(PGTNet_BPIC13C, dataset_dir)             
        elif pyg_dataset_id == 'PGTNet_Sepsis':
            dataset = preformat_pgtnet_dataset(PGTNet_Sepsis, dataset_dir)         
        elif pyg_dataset_id == 'PGTNet_env_permit':
            dataset = preformat_pgtnet_dataset(PGTNet_env_permit, dataset_dir)         
        elif pyg_dataset_id == 'PGTNet_Hospital':
            dataset = preformat_pgtnet_dataset(PGTNet_Hospital, dataset_dir)            
        elif pyg_dataset_id == 'PGTNet_BPIC20D':
            dataset = preformat_pgtnet_dataset(PGTNet_BPIC20D, dataset_dir)         
        elif pyg_dataset_id == 'PGTNet_BPIC20I':
            dataset = preformat_pgtnet_dataset(PGTNet_BPIC20I, dataset_dir)       
        elif pyg_dataset_id == 'PGTNet_Trafficfines':
            dataset = preformat_pgtnet_dataset(PGTNet_Trafficfines, dataset_dir)          
        elif pyg_dataset_id == 'PGTNet_BPIC15M1':
            dataset = preformat_pgtnet_dataset(PGTNet_BPIC15M1, dataset_dir)              
        elif pyg_dataset_id == 'PGTNet_BPIC15M2':
            dataset = preformat_pgtnet_dataset(PGTNet_BPIC15M2, dataset_dir)          
        elif pyg_dataset_id == 'PGTNet_BPIC15M3':
            dataset = preformat_pgtnet_dataset(PGTNet_BPIC15M3, dataset_dir)          
        elif pyg_dataset_id == 'PGTNet_BPIC15M4':
            dataset = preformat_pgtnet_dataset(PGTNet_BPIC15M4, dataset_dir)             
        elif pyg_dataset_id == 'PGTNet_BPIC15M5':
            dataset = preformat_pgtnet_dataset(PGTNet_BPIC15M5, dataset_dir)  
        else:
            raise ValueError(f"Unexpected PyG Dataset identifier: {format}")

    # GraphGym default loader for Pytorch Geometric datasets
    elif format == 'PyG':
        dataset = load_pyg(name, dataset_dir)
    else:
        raise ValueError(f"Unknown data format: {format}")

    pre_transform_in_memory(dataset,
                            partial(task_specific_preprocessing, cfg=cfg))

    log_loaded_dataset(dataset, format, name)

    # Precompute necessary statistics for positional encodings.
    pe_enabled_list = []
    for key, pecfg in cfg.items():
        if key.startswith('posenc_') and pecfg.enable:
            pe_name = key.split('_', 1)[1]
            pe_enabled_list.append(pe_name)
            if hasattr(pecfg, 'kernel'):
                # Generate kernel times if functional snippet is set.
                if pecfg.kernel.times_func:
                    pecfg.kernel.times = list(eval(pecfg.kernel.times_func))
                logging.info(f"Parsed {pe_name} PE kernel times / steps: "
                             f"{pecfg.kernel.times}")
    if pe_enabled_list:
        start = time.perf_counter()
        logging.info(f"Precomputing Positional Encoding statistics: "
                     f"{pe_enabled_list} for all graphs...")
        # Estimate directedness based on 10 graphs to save time.
        is_undirected = all(d.is_undirected() for d in dataset[:10])
        logging.info(f"  ...estimated to be undirected: {is_undirected}")
        pre_transform_in_memory(dataset,
                                partial(compute_posenc_stats,
                                        pe_types=pe_enabled_list,
                                        is_undirected=is_undirected,
                                        cfg=cfg),
                                show_progress=True
                                )
        elapsed = time.perf_counter() - start
        timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
                  + f'{elapsed:.2f}'[-3:]
        logging.info(f"Done! Took {timestr}")

    # Set standard dataset train/val/test splits
    if hasattr(dataset, 'split_idxs'):
        set_dataset_splits(dataset, dataset.split_idxs)
        delattr(dataset, 'split_idxs')

    # Verify or generate dataset train/val/test splits
    prepare_splits(dataset)

    # Precompute in-degree histogram if needed for PNAConv.
    if cfg.gt.layer_type.startswith('PNA') and len(cfg.gt.pna_degrees) == 0:
        cfg.gt.pna_degrees = compute_indegree_histogram(
            dataset[dataset.data['train_graph_index']])
        # print(f"Indegrees: {cfg.gt.pna_degrees}")
        # print(f"Avg:{np.mean(cfg.gt.pna_degrees)}")

    return dataset


def compute_indegree_histogram(dataset):
    """Compute histogram of in-degree of nodes needed for PNAConv.
    Args:
        dataset: PyG Dataset object
    Returns:
        List where i-th value is the number of nodes with in-degree equal to `i`
    """
    from torch_geometric.utils import degree

    deg = torch.zeros(1000, dtype=torch.long)
    max_degree = 0
    for data in dataset:
        d = degree(data.edge_index[1],
                   num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, d.max().item())
        deg += torch.bincount(d, minlength=deg.numel())
    return deg.numpy().tolist()[:max_degree + 1]


def preformat_pgtnet_dataset(dataset_class, dataset_dir):
    """Load and preformat datasets.
    Args:
        dataset_class: the class of the dataset to load
        dataset_dir: path where to store the cached dataset
    Returns:
        PyG dataset object
    """
    dataset = join_dataset_splits(
        [dataset_class(root=dataset_dir, split=split)
         for split in ['train', 'val', 'test']]
    )
    return dataset



def join_dataset_splits(datasets):
    """Join train, val, test datasets into one dataset object.
    Args:
        datasets: list of 3 PyG datasets to merge
    Returns:
        joint dataset with `split_idxs` property storing the split indices
    """
    assert len(datasets) == 3, "Expecting train, val, test datasets"

    n1, n2, n3 = len(datasets[0]), len(datasets[1]), len(datasets[2])
    data_list = [datasets[0].get(i) for i in range(n1)] + \
                [datasets[1].get(i) for i in range(n2)] + \
                [datasets[2].get(i) for i in range(n3)]

    datasets[0]._indices = None
    datasets[0]._data_list = data_list
    datasets[0].data, datasets[0].slices = datasets[0].collate(data_list)
    split_idxs = [list(range(n1)),
                  list(range(n1, n1 + n2)),
                  list(range(n1 + n2, n1 + n2 + n3))]
    datasets[0].split_idxs = split_idxs

    return datasets[0]