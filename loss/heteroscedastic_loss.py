import torch

# Custom function for heteroscedastic loss.
def heteroscedastic_loss(true, mean, log_var, metric='mse'): 
    '''
    ARGUMENTS:
    true: true values with shape of: batch_size x num outputs
    mean: predictions with shape of: batch_size x num outputs
    log_var: Logaritms of uncertainty estimates. shape: batch_size x num outputs
    metric: "mae" or "rmse" (default is "rmse")

    OUTPUTS:
    loss: Tensor (0)
    '''
    precision = torch.exp(-log_var)
    if metric == 'mae':
        # based on L1-loss and its relation to Laplace distribution
        loss = torch.mean(torch.sum(
            precision** 0.5 * torch.abs(true - mean) + log_var, dim=1), dim=0)
    elif metric == 'mse':
        loss = torch.mean(torch.sum(
            0.5 * precision * (true - mean) ** 2 + 0.5 * log_var, dim=1), dim=0)
    else:
        raise ValueError("Metric has to be 'mse' or 'mae'")
        
    return loss



