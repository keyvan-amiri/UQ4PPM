"""
To prepare thi script we used the following source codes:
    https://github.com/hansweytjens/uncertainty
    https://github.com/nlhkh/dropout-in-rnn
    https://gitlab.citius.usc.es/efren.rama/pmdlcompararator
We adjusted the source codes to efficiently integrate them into our framework.
"""
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from typing import Tuple
from models.Concrete_dropout import ConcreteDropout


# a class for models based on stochastic DALSTM
class StochasticDALSTM(nn.Module):
    """DALSTM equipped with dropout and MCMC"""
    def __init__(self, input_size=None, hidden_size=None, n_layers=None,
                 max_len=None, dropout=True, concrete=True, p_fix=0.2,
                 weight_regularizer=0.1, dropout_regularizer=0.1,
                 hs=False, Bayes=True, device=None):
        '''
        ARGUMENTS:
        input_size: number of features
        hidden_size: number of neurons in LSTM layers
        n_layers: number of LSTM layers
        max_len: maximum length for prefixes in the dataset
        concrete: 'True': concrete dropout, otherwise dropout probability fixed
        p_fix: dropout probability
        weight_regularizer: param for weight regularization in reformulated ELBO
        dropout_regularizer: param for dropout regularization in reformulated ELBO
        hs: "True" if heteroscedastic, "False" if homoscedastic
        Bayes: is always True since we have a separate model deterministic 
        '''
        super(StochasticDALSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers 
        self.max_len = max_len
        self.dropout = dropout
        self.concrete = concrete
        self.p_fix = p_fix        
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.heteroscedastic = hs
        self.Bayes = Bayes
        self.device = device
        self.first_layer = StochasticLSTMCell(
            self.input_size, self.hidden_size, p_fix=self.p_fix,
            concrete=self.concrete, weight_regularizer=self.weight_regularizer,
            dropout_regularizer=self.dropout_regularizer, Bayes=self.Bayes,
            device=self.device)
        self.hidden_layers = nn.ModuleList(
            [StochasticLSTMCell(self.hidden_size, self.hidden_size, self.p_fix,
                                concrete=self.concrete,
                                weight_regularizer=self.weight_regularizer,
                                dropout_regularizer=self.dropout_regularizer,
                                Bayes=self.Bayes, device=self.device)
                                            for i in range(self.n_layers - 1)])
        self.linear2_mu = nn.Linear(self.hidden_size, 1)
        if self.heteroscedastic:
            self.linear2_logvar = nn.Linear(self.hidden_size, 1)
        self.conc_drop2_mu = ConcreteDropout(
            dropout=self.dropout, concrete=self.concrete, p_fix=self.p_fix,
            weight_regularizer=self.weight_regularizer,
            dropout_regularizer=self.dropout_regularizer, conv="lin",
            Bayes=self.Bayes, device=self.device)
        if self.heteroscedastic:
            self.conc_drop2_logvar = ConcreteDropout(
                dropout=self.dropout, concrete=self.concrete, p_fix=self.p_fix,
                weight_regularizer=self.weight_regularizer,
                dropout_regularizer=self.dropout_regularizer, conv="lin",
                Bayes=self.Bayes, device=self.device)
        self.batch_norm1 = nn.BatchNorm1d(max_len)
        self.relu = nn.ReLU()
    
    def regularizer(self):
        total_weight_reg = self.first_layer.regularizer()
        for l in self.hidden_layers:
            total_weight_reg += l.regularizer()
        return total_weight_reg
    
    # TODO: remove stop_dropout since for deterministic version we have a separate model
    # TODO: to do so you should check numerous stop dropouts in this script!
    def forward(self, x, stop_dropout=False):
        '''
        ARGUMENTS:
        stop_dropout: if "True" prevents dropout in inference (deterministic)
        OUTPUTS:
        mean: outputs (point estimates). shape: batch size x number of outputs
        log_var: log of uncertainty estimates. shape: batch size x number of outputs
        regularization.sum(): sum of KL regularizers over all model layers
        '''
        regularization = torch.empty(2, device=x.device)
        batch_size = x.shape[1]
        h_n = torch.zeros(self.n_layers, batch_size,
                          self.first_layer.hidden_size)
        c_n = torch.zeros(self.n_layers, batch_size,
                          self.first_layer.hidden_size)
        x, (h, c) = self.first_layer(x)
        h_n[0] = h
        c_n[0] = c
        x = self.batch_norm1(x)
        for i, layer in enumerate(self.hidden_layers):
            x, (h, c) = layer(x, (h, c))
            h_n[i+1] = h
            c_n[i+1] = c
            x = self.batch_norm1(x)
            
        mean, regularization[0] = self.conc_drop2_mu(
            x[:, -1, :], nn.Sequential(self.linear2_mu, self.relu),
            stop_dropout)        
        if self.heteroscedastic:
            log_var, regularization[1] = \
                self.conc_drop2_logvar(x[:, -1, :],
                                       self.linear2_logvar, stop_dropout)
        else:
            regularization[1] = 0
            log_var = torch.empty(mean.size())
        return mean.squeeze(dim=1), log_var.squeeze(dim=1), regularization.sum()

# Stochastic LSTM cell that can handle fix dropout as well as concrete dropout
class StochasticLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, p_fix=0.01,
                 concrete=True, weight_regularizer=.1, dropout_regularizer=.1,
                 Bayes=True, device=None):
        '''
        ARGUMENTS:
        input_size: number of features
        hidden_size: number of neurons in LSTM layers
        concrete: 'True': concrete dropout, otherwise dropout probability fixed
        p_fix: dropout probability
        weight_regularizer: param for weight regularization in reformulated ELBO
        dropout_regularizer: param for dropout regularization in reformulated ELBO
        hs: "True" if heteroscedastic, "False" if homoscedastic
        Bayes: is always True since we have a separate model deterministic 
        '''

        super(StochasticLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.concrete = concrete
        self.wr = weight_regularizer
        self.dr = dropout_regularizer
        self.Bayes = Bayes
        self.device = device

        if concrete:
            self.p_logit = nn.Parameter(torch.empty(1).normal_())
        else:
            if np.isnan(p_fix):
                p_fix = .5
            self.p_logit = torch.full([1], p_fix)

        self.Wi = nn.Linear(self.input_size, self.hidden_size)
        self.Wf = nn.Linear(self.input_size, self.hidden_size)
        self.Wo = nn.Linear(self.input_size, self.hidden_size)
        self.Wg = nn.Linear(self.input_size, self.hidden_size)

        self.Ui = nn.Linear(self.hidden_size, self.hidden_size)
        self.Uf = nn.Linear(self.hidden_size, self.hidden_size)
        self.Uo = nn.Linear(self.hidden_size, self.hidden_size)
        self.Ug = nn.Linear(self.hidden_size, self.hidden_size)

        self.init_weights()

    def init_weights(self):
        k = torch.tensor(
            self.hidden_size, dtype=torch.float32).reciprocal().sqrt()

        self.Wi.weight.data.uniform_(-k, k).to(self.device)
        self.Wi.bias.data.uniform_(-k, k).to(self.device)

        self.Wf.weight.data.uniform_(-k, k).to(self.device)
        self.Wf.bias.data.uniform_(-k, k).to(self.device)

        self.Wo.weight.data.uniform_(-k, k).to(self.device)
        self.Wo.bias.data.uniform_(-k, k).to(self.device)

        self.Wg.weight.data.uniform_(-k, k).to(self.device)
        self.Wg.bias.data.uniform_(-k, k).to(self.device)

        self.Ui.weight.data.uniform_(-k, k).to(self.device)
        self.Ui.bias.data.uniform_(-k, k).to(self.device)

        self.Uf.weight.data.uniform_(-k, k).to(self.device)
        self.Uf.bias.data.uniform_(-k, k).to(self.device)

        self.Uo.weight.data.uniform_(-k, k).to(self.device)
        self.Uo.bias.data.uniform_(-k, k).to(self.device)

        self.Ug.weight.data.uniform_(-k, k).to(self.device)
        self.Ug.bias.data.uniform_(-k, k).to(self.device)

    def _sample_mask(self, batch_size, stop_dropout):
        '''
        ARGUMENTS:
        batch_size: batch size
        stop_dropout: if "True" prevents dropout in inference (deterministic)

        OUTPUTS:
        zx: dropout masks for inputs. Tensor (GATES x batch_size x input size (after embedding))
        zh: dropout masks for hiddens states. Tensor (GATES x batch_size x number hidden states)
        '''

        if not self.concrete:
            p = self.p_logit.to(self.device)
        else:
            p = torch.sigmoid(self.p_logit).to(self.device)
        GATES = 4
        eps = torch.tensor(1e-7)
        t = 1e-1

        if not stop_dropout:
            ux = torch.rand(GATES, batch_size, self.input_size).to(self.device)
            uh = torch.rand(GATES, batch_size, self.hidden_size).to(self.device)

            if self.input_size == 1:
                zx = (1 - torch.sigmoid(
                    (torch.log(eps) - torch.log(1 + eps) + torch.log(
                        ux + eps) - torch.log(1 - ux + eps))/ t))
            else:
                zx = (1 - torch.sigmoid(
                    (torch.log(p + eps) - torch.log(1 - p + eps) + torch.log(
                        ux + eps) - torch.log(1 - ux + eps))/ t)) / (1 - p)
            zh = (1 - torch.sigmoid(
                (torch.log(p + eps) - torch.log(1 - p + eps) + torch.log(
                    uh + eps) - torch.log(1 - uh + eps))/ t)) / (1 - p)
        else:
            zx = torch.ones(GATES, batch_size, self.input_size).to(self.device)
            zh = torch.ones(GATES, batch_size, self.input_size).to(self.device)

        return zx, zh

    
    def regularizer(self):
        '''
        OUTPUTS:
        self.wr * weight_sum: weight regularization in reformulated ELBO
        self.wr * bias_sum: bias regularization in reformulated ELBO
        self.dr * dropout_reg: dropout regularization in reformulated ELBO
        '''

        if not self.concrete:
            p = self.p_logit.to(self.device)
        else:
            p = torch.sigmoid(self.p_logit)

        if self.Bayes:
            weight_sum = torch.tensor([torch.sum(params ** 2) for name, params \
                                       in self.named_parameters() if \
                                           name.endswith("weight")]
                                      ).sum() / (1. - p)

            bias_sum = torch.tensor([torch.sum(params ** 2) for name, params \
                                     in self.named_parameters() if \
                                         name.endswith("bias")]).sum()

            if not self.concrete:
                dropout_reg = torch.zeros(1)
            else:
                dropout_reg = self.input_size * (
                    p * torch.log(p) + (1 - p) * torch.log(1 - p))
            return self.wr * weight_sum, self.wr * bias_sum, self.dr * dropout_reg
        else:
            return torch.zeros(1)


    def forward(self, input: Tensor, stop_dropout=False) -> Tuple[
        Tensor, Tuple[Tensor, Tensor]]:
        '''
        ARGUMENTS:
        input: sequence length x batch size x input size(after embedding)
        stop_dropout: if "True" prevents dropout in inference (deterministic)

        OUTPUTS:
        hn: tensor of hidden states h_t. shape: sequence_length x batch_size x hidden size
        h_t: hidden states at time t. shape: batch size x hidden size (nodes in LSTM layer)
        c_t: cell states. shape: batch size x hidden size (nodes in LSTM layer)
        '''

        seq_len, batch_size = input.shape[0:2]

        h_t = torch.zeros(
            batch_size, self.hidden_size, dtype=input.dtype).to(self.device)
        c_t = torch.zeros(
            batch_size, self.hidden_size, dtype=input.dtype).to(self.device)

        hn = torch.empty(
            seq_len, batch_size, self.hidden_size, dtype=input.dtype)

        zx, zh = self._sample_mask(batch_size, stop_dropout)

        for t in range(seq_len):
            x_i, x_f, x_o, x_g = (input[t] * zx_ for zx_ in zx)
            h_i, h_f, h_o, h_g = (h_t * zh_ for zh_ in zh)

            i = torch.sigmoid(self.Ui(h_i) + self.Wi(x_i))
            f = torch.sigmoid(self.Uf(h_f) + self.Wf(x_f))
            o = torch.sigmoid(self.Uo(h_o) + self.Wo(x_o))
            g = torch.tanh(self.Ug(h_g) + self.Wg(x_g))

            c_t = f * c_t + i * g
            h_t = o * torch.tanh(c_t)
            hn[t] = h_t
            hn = hn.to(self.device)

        return hn, (h_t, c_t)  
