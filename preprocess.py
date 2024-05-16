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
    # Note that by default each model performs as follows:
    # DALSTM do'nt apply any normalization to remaining time (target attribute)
    # PGTNet apply min-max normalization to remaining time (target attribute)
    parser.add_argument('--normalization',
                        action='store_true',
                        help='Boolean for normalization (default: False)')    
    parser.add_argument('--', default= False,
                        help='The data split that is used')
    args = parser.parse_args()
    dataset_names = args.datasets
    #print(dataset_names)
    for dataset in dataset_names:
        # create the graph dataset representation of the event log for PGTNet
        PGTNet_convertor_case_centric(dataset_name=dataset)
        # handle data preprocessing for DALSTM approach
        if args.normalization:
            # if normalization is included in command line arguments:
            DALSTM_preprocessing(dataset_name=dataset,
                                 normalization=args.normalization)
        else:
            #otherwise
            DALSTM_preprocessing(dataset_name=dataset)
    
if __name__ == '__main__':
    main()