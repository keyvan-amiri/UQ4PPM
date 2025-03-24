import argparse
import os
import pandas as pd
from scipy.stats import wilcoxon
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)


def compare_models_wilcoxon(model1_scores, model2_scores, higher_is_better=True):
    """
    Compare two models using the Wilcoxon signed-rank test at three confidence levels.
    
    Parameters:
    - model1_scores (list): Performance scores of model 1 across datasets.
    - model2_scores (list): Performance scores of model 2 across datasets.
    - higher_is_better (bool): If True, higher values indicate better performance;
                               if False, lower values indicate better performance.
    """
    if not higher_is_better:
        # Invert the values so that higher values always mean better performance
        model1_scores = [-x for x in model1_scores]
        model2_scores = [-x for x in model2_scores]
    
    # Perform Wilcoxon signed-rank test
    stat, p_value = wilcoxon(model1_scores, model2_scores)
    
    # Print results for different confidence levels
    print("Wilcoxon Signed-Rank Test Results:")
    print(f"Statistic: {stat:.4f}, p-value: {p_value:.4f}")
    
    for alpha in [0.05, 0.10, 0.20]:
        confidence = 1 - alpha
        if p_value < alpha:
            print(f"At {confidence*100:.0f}% confidence, there is a significant difference between the models.")
        else:
            print(f"At {confidence*100:.0f}% confidence, there is NO significant difference between the models.")
            
def main():
    # Parse arguments     
    parser = argparse.ArgumentParser(
        description='Process_Aware Calibrated Regression') 
    #['DA+H', 'CDA+H', 'LA', 'BE+H', 'H', 'CARD', 'LA+I', 'LA+S']
    parser.add_argument('--model_1', help='First model in comparison')
    parser.add_argument('--model_2', help='Second model in comparison')
    # 'AURG' 'MAE' 'MA' 'Sharp'
    parser.add_argument('--metric', help='Metric used for comparison')
    args = parser.parse_args()  
    
    datasets = ['BPIC20DD', 'BPIC20ID', 'BPIC20RFP', 'BPIC20PTC', 'BPIC20TPD',
                'BPIC15_1', 'BPIC12', 'HelpDesk', 'Sepsis', 'BPIC13I']
    root_path = os.getcwd()
    
    list_1 = []
    list_2 = []
    if args.metric == 'AURG':
        higher_is_better=True
    else:
        higher_is_better=False
    for dataset in datasets:
        folder = os.path.join(root_path, 'results', dataset)      
        csv_path =  os.path.join(folder, dataset, dataset + '_overal_results.csv')
        eval_df = pd.read_csv(csv_path)
        df1 = eval_df[eval_df['Model'] == args.model_1]
        df2 = eval_df[eval_df['Model'] == args.model_2]
        list_1.append(float(df1[args.metric]))
        list_2.append(float(df2[args.metric]))
    compare_models_wilcoxon(list_1, list_2, higher_is_better=higher_is_better)

    
if __name__ == '__main__':
    main()  