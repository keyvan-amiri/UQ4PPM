import argparse
import os
import pickle
import torch

def main():
    parser = argparse.ArgumentParser(description='get prefix lengths') 
    parser.add_argument('--dataset') 
    args = parser.parse_args()
    root_path = os.getcwd()
    dalstm_name = 'DALSTM_' + args.dataset
    data_path = os.path.join(root_path, 'datasets', dalstm_name)
    X_path = os.path.join(data_path, 'DALSTM_X_val_'+args.dataset+'.pt')
    length_path = os.path.join(data_path, 'DALSTM_val_length_list_'+args.dataset+'.pkl')   
    X = torch.load(X_path)
    padding_value = 0.0    
    lengths = (X.abs().sum(dim=2) != padding_value).sum(dim=1)
    lengths = lengths.numpy()
    with open(length_path, 'wb') as file:
        pickle.dump(lengths, file)
    

if __name__ == '__main__':
    main()
    