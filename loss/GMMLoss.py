import torch
   
class GMM_Loss(torch.nn.Module):
    def __init__(self):
        super(GMM_Loss, self).__init__()

    def forward(self, yhat, yhat_std, y, alpha):
        acc_term = torch.abs(yhat-y)*alpha
        nll_term = (0.5*((torch.square(yhat-y)/torch.square(yhat_std))
                         +(torch.log(torch.square(yhat_std))))
                    +0.5*(torch.log(torch.tensor(2 * torch.pi))))*(1-alpha)
        loss = torch.mean(acc_term + nll_term)
        return loss