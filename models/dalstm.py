import torch.nn as nn
import torch.nn.functional as F

##############################################################################
# Backbone Data-aware LSTM model for remaining time prediction
##############################################################################
class DALSTMModel(nn.Module):
    def __init__(self, input_size=None, hidden_size=None, n_layers=None,
                 max_len=None, dropout=True, p_fix=0.2):
        '''
        ARGUMENTS:
        input_size: number of features
        hidden_size: number of neurons in LSTM layers
        n_layers: number of LSTM layers
        max_len: maximum length for prefixes in the dataset
        dropout: apply dropout if "True", otherwise no dropout
        p_fix: dropout probability
        '''
        super(DALSTMModel, self).__init__()
        
        self.n_layers = n_layers 
        self.dropout = dropout
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout_layer = nn.Dropout(p=p_fix)
        self.batch_norm1 = nn.BatchNorm1d(max_len)
        self.linear1 = nn.Linear(hidden_size, 1) 
        
    def forward(self, x):
        x = x.float() # if tensors are saved in a different format
        x, (hidden_state,cell_state) = self.lstm1(x)
        if self.dropout:
            x = self.dropout_layer(x)
        x = self.batch_norm1(x)
        if self.n_layers > 1:
            for i in range(self.n_layers - 1):
                x, (hidden_state,cell_state) = self.lstm2(
                    x, (hidden_state,cell_state))
                if self.dropout:
                    x = self.dropout_layer(x)
                x = self.batch_norm1(x)
        yhat = self.linear1(x[:, -1, :]) # only the last one in the sequence 
        return yhat.squeeze(dim=1) 
    
##############################################################################
# Stochastic Data-aware LSTM modelL remaining time prediction (mean & variance)
##############################################################################

class DALSTMModelMve(nn.Module):
    def __init__(self, input_size=None, hidden_size=None, n_layers=None,
                 max_len=None, dropout=True, p_fix=0.2):
        '''
        ARGUMENTS:
        input_size: number of features
        hidden_size: number of neurons in LSTM layers
        n_layers: number of LSTM layers
        max_len: maximum length for prefixes in the dataset
        dropout: apply dropout if "True", otherwise no dropout
        p_fix: dropout probability
        '''
        super(DALSTMModelMve, self).__init__()
        
        self.n_layers = n_layers 
        self.dropout = dropout
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout_layer = nn.Dropout(p=p_fix)
        self.batch_norm1 = nn.BatchNorm1d(max_len)
        
        # Linear layers for mean and variance
        self.linear_mu = nn.Linear(hidden_size, 1)
        self.linear_logvar = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = x.float() # if tensors are saved in a different format
        x, (hidden_state, cell_state) = self.lstm1(x)
        if self.dropout:
            x = self.dropout_layer(x)
        x = self.batch_norm1(x)
        if self.n_layers > 1:
            for i in range(self.n_layers - 1):
                x, (hidden_state, cell_state) = self.lstm2(
                    x, (hidden_state, cell_state))
                if self.dropout:
                    x = self.dropout_layer(x)
                x = self.batch_norm1(x)
        
        # Predict mean and variance
        mu = self.linear_mu(x[:, -1, :]) # mean prediction       
        logvar = self.linear_logvar(x[:, -1, :]) # log-variance prediction
        
        return mu.squeeze(dim=1), logvar.squeeze(dim=1)