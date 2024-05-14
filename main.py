import argparse
import os
import yaml
import torch
from utils.utils import set_random_seed

def main():    
    # Parse arguments 
    parser = argparse.ArgumentParser(
        description='Probabilistic remaining time prediction')
    parser.add_argument('--dataset', help='Datasets used by model')
    parser.add_argument('--model', help='Type of the predictive model')
    parser.add_argument('--UQ',
                        help='Uncertainty quantification method to be used')
    parser.add_argument('--seed', help='Random seed to use')
    parser.add_argument('--device', type=int, default=0, help='GPU device id') 
    args = parser.parse_args()
    dataset_name = args.dataset
    net = args.model
    uq_method = args.UQ
    seed = int(args.seed)
    device_name = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    # set device    
    device = torch.device(device_name)
    # set random seed    
    set_random_seed(seed)
    root_path = os.getcwd()  
    # read the relevant configuration file
    cfg_file = os.path.join(root_path, 'cfg', net+'_'+dataset_name+'.yaml')
    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)
    print(cfg)

    

    
if __name__ == '__main__':
    main()