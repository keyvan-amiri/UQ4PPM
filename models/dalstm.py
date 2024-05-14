"""
To prepare thi script we used the following source codes:
    https://gitlab.citius.usc.es/efren.rama/pmdlcompararator
We adjusted the source codes to efficiently integrate them into our framework.
"""

import torch.nn as nn

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