import warnings
import argparse
from utils.utils import str2bool
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
    parser.add_argument('--model', default='pgtnet',
                        help='Type of the predictive model')
    parser.add_argument('--seed', type=int, default=42,
                        help='Number of splits that is used')
    # DALSTM don't apply any normalization to remaining time (target attribute)
    parser.add_argument('--normalization_lstm', type=str2bool, nargs='?',
                        const=True, default=False,
                        help='Whether to apply normalization for dalstm.')
    # PGTNet apply min-max normalization to remaining time (target attribute)  
    parser.add_argument('--normalization_pgtnet', type=str2bool, nargs='?',
                        const=True, default=True,
                        help='Whether to apply normalization for PGTNet.')
    
    args = parser.parse_args()
    dataset_names = args.datasets
    #print(dataset_names)
    for dataset in dataset_names:
        print(f'Pre-processing is conducted for {dataset} dataset, \
                  and {args.model} model')
        # data preprocessing for DALSTM approach
        if args.model == 'dalstm':            
            if args.normalization_lstm:
                DALSTM_preprocessing(dataset_name=dataset, seed=args.seed,
                                     normalization=args.normalization_lstm)
            else:
                DALSTM_preprocessing(dataset_name=dataset, seed=args.seed)
        """
        # create graph dataset representation of the event log for PGTNet
        elif args.model == 'pgtnet':      
            # if normalization is included in command line arguments:
            if args.normalization_pgtnet:                   
                PGTNet_convertor_case_centric(dataset_name=dataset)
            else:           
                PGTNet_convertor_case_centric(
                    dataset_name=dataset,
                    normalization=args.normalization_pgtnet)
        else:
            print(f'{args.model} architecture is currently not supported.')
        """
    
if __name__ == '__main__':
    main()