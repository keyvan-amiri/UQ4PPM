import glob
import os
import re


# A mothod to retrieve all results for a combination of dataset-model
def get_csv_files(folder_path):
    # Get all csv files in the folder
    all_csv_files = glob.glob(os.path.join(folder_path, '*.csv'))    
    # Filter out files containing 'deterministic' in their names
    filtered_csv_files = [f for f in all_csv_files if 
                          'deterministic' not in os.path.basename(f).lower()]
    # Collect name of uncertainty quantification approaches
    prefixes = []
    pattern = re.compile(r'(.*?)(_holdout_|_cv_)')
    for file_path in filtered_csv_files:
        file_name = os.path.basename(file_path)
        match = pattern.match(file_name)
        if match:
            prefixes.append(match.group(1))   
    return filtered_csv_files, prefixes


