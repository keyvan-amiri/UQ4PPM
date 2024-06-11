import torch.nn as nn
from loss.LogCosh import LogCoshLoss
from loss.rmse import RMSELoss
from loss.heteroscedastic_loss import heteroscedastic_loss

def set_loss(loss_func=None, heteroscedastic=False): 
    if heteroscedastic:
        criterion = heteroscedastic_loss(metric=loss_func)
    else:
        if loss_func == 'mae':
            criterion = nn.L1Loss()
        elif loss_func == 'LogCoshLoss':
            criterion = LogCoshLoss()
        elif loss_func == 'mse':
            criterion = nn.MSELoss()
        elif loss_func == 'Huber':
            criterion = nn.HuberLoss()
        elif loss_func == 'smooth_mae':
            criterion = nn.SmoothL1Loss()
        elif loss_func == 'rmse':
            criterion = RMSELoss()        
    return criterion