import torch.nn as nn
from loss.LogCosh import LogCoshLoss

def set_loss(loss_func=None): 
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
    # TODO: implement rmse as the default loss function for dropout approximation
    elif loss_func == 'rmse':
        pass
    return criterion