"""
This python script is adapted from the origninal script:
    https://github.com/XzwHan/CARD
    CARD: Classification and Regression Diffusion Models by Xizewen Han,
    Huangjie Zheng, and Mingyuan Zhou.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        out = self.lin(x)
        gamma = self.embed(t)
        out = gamma.view(-1, self.num_out) * out
        return out


class ConditionalGuidedModel(nn.Module):
    def __init__(self, config):
        super(ConditionalGuidedModel, self).__init__()
        n_steps = config.diffusion.timesteps + 1
        data_dim = config.model.y_dim
        feature_dim = config.model.feature_dim
        hidden_size = config.diffusion.nonlinear_guidance.hidden_size
        dropout_rate = config.diffusion.nonlinear_guidance.dropout_rate
        max_len = config.model.max_len
        self.arch = config.model.arch
        self.cat_x = config.model.cat_x
        self.cat_y_pred = config.model.cat_y_pred
        self.dalstm = True if config.diffusion.conditioning_signal == "DALSTM" else False
        self.n_layers = config.diffusion.nonlinear_guidance.n_layers
        self.dropout = config.diffusion.nonlinear_guidance.dropout       
        if self.cat_x:
            if self.dalstm:
                data_dim += hidden_size
            else:
                # TODO: for further implementation of ProcessTransformer and PGTNet
                print('to be implemented')
        if self.cat_y_pred:
            data_dim += config.model.y_dim
        self.lstm1 = nn.LSTM(config.model.x_dim, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)        
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.batch_norm1 = nn.BatchNorm1d(max_len) 
        self.lin1 = ConditionalLinear(data_dim, feature_dim, n_steps)
        self.lin2 = ConditionalLinear(feature_dim, feature_dim, n_steps)
        self.lin3 = ConditionalLinear(feature_dim, feature_dim, n_steps)
        self.lin4 = nn.Linear(feature_dim, 1)

    def forward(self, x, y_t, y_0_hat, t):
        
        # specialized computations for DALSTM model
        if self.dalstm:
            x, (hidden_state,cell_state) = self.lstm1(x)
            if self.dropout:
                x = self.dropout_layer(x)
            x = self.batch_norm1(x)
            if self.arch == 'LSTM' and self.n_layers > 1:
                for i in range(self.n_layers - 1):
                    x, (hidden_state,cell_state) = self.lstm2(
                        x, (hidden_state,cell_state))
                    if self.dropout:
                        x = self.dropout_layer(x)
                    x = self.batch_norm1(x)            
            x = x[:, -1, :] # only the last one in the sequence 
        # add one dimension for later concatanation
        y_t = y_t.unsqueeze(1) 
        # add one dimension for later concatanation
        y_0_hat = y_0_hat.unsqueeze(1) 
        
        # define proper concatanation based on the config file
        if self.cat_x:
            if self.cat_y_pred:
                eps_pred = torch.cat((y_t, y_0_hat, x), dim=1)
            else:
                eps_pred = torch.cat((y_t, x), dim=1)
        else:
            if self.cat_y_pred:
                eps_pred = torch.cat((y_t, y_0_hat), dim=1)
            else:
                eps_pred = y_t
        
        """
        Hadamard product between concatanation result and the corresponding
        timestep embedding + Softplus non-linearity
        """ 
        eps_pred = F.softplus(self.lin1(eps_pred, t))
        eps_pred = F.softplus(self.lin2(eps_pred, t))
        eps_pred = F.softplus(self.lin3(eps_pred, t))
        return self.lin4(eps_pred)


