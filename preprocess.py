import warnings
import argparse
from utils.PGTNet_convertor import PGTNet_convertor_case_centric
from utils.DALSTM_processing import DALSTM_preprocessing

def main():    
    warnings.filterwarnings('ignore')
    # Parse arguments for data preprocessing
    parser = argparse.ArgumentParser(description='Data pre-processing')
    # can handle multiple event logs at the same time
    parser.add_argument('--datasets', nargs='+',
                        help='Raw datasets to be pre-processed', required=True)
    args = parser.parse_args()
    dataset_names = args.datasets
    print(dataset_names)
    for dataset in dataset_names:
        # create the graph dataset representation of the event log for PGTNet
        PGTNet_convertor_case_centric(dataset_name=dataset)
        # handle data preprocessing for DALSTM approach
        DALSTM_preprocessing(dataset_name=dataset)
    
if __name__ == '__main__':
    main()