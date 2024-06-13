import warnings
import argparse
#TODO: uncomment the following line after working on PGTNet
#from utils.PGTNet_convertor import PGTNet_convertor_case_centric
from utils.DALSTM_processing import DALSTM_preprocessing

def main():    
    # supress warnings for PM4PY
    warnings.filterwarnings('ignore')
    # Parse arguments for data preprocessing
    parser = argparse.ArgumentParser(description='Data pre-processing')
    # can handle multiple event logs at the same time
    parser.add_argument('--datasets', nargs='+',
                        help='Raw datasets to be pre-processed', required=True)
    parser.add_argument('--seed', type=int, default=42,
                        help='Number of splits that is used')
    # Note that by default each model performs as follows:
    # DALSTM do'nt apply any normalization to remaining time (target attribute)
    # PGTNet apply min-max normalization to remaining time (target attribute)
    parser.add_argument('--normalization',
                        action='store_true',
                        help='Boolean for normalization (default: False)')    
    args = parser.parse_args()
    dataset_names = args.datasets
    #print(dataset_names)
    for dataset in dataset_names:
        # create the graph dataset representation of the event log for PGTNet
        #TODO: uncomment the following line after working on PGTNet
        #PGTNet_convertor_case_centric(dataset_name=dataset)
        # handle data preprocessing for DALSTM approach
        if args.normalization:
            # if normalization is included in command line arguments:
            DALSTM_preprocessing(dataset_name=dataset, seed=args.seed,
                                 normalization=args.normalization)
        else:
            #otherwise
            DALSTM_preprocessing(dataset_name=dataset, seed=args.seed)
    
if __name__ == '__main__':
    main()