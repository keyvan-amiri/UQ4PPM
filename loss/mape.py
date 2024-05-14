import torch

# Custom function for Mean Absolute Percentage Error (MAPE)
def mape(outputs, targets):
    return torch.mean(torch.abs((targets - outputs) / targets)) * 100 