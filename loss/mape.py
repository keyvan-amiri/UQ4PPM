import torch

"""
# Custom function for Mean Absolute Percentage Error (MAPE)
def mape(outputs, targets, epsilon=1e-8):
    # Add epsilon to the targets to avoid division by zero
    return torch.mean(torch.abs((targets - outputs) / (targets + epsilon))) * 100
"""

def mape(outputs, targets, epsilon=1e-8, threshold=1e-6):
    # Create a mask for non-zero targets
    non_zero_mask = torch.abs(targets) > threshold
    
    # Compute the absolute percentage error only for non-zero targets
    absolute_percentage_error = torch.abs((targets - outputs) / (targets + epsilon))
    
    # Apply the mask to ignore errors where targets are zero or close to zero
    mape_value = torch.mean(absolute_percentage_error[non_zero_mask])
    
    return mape_value * 100