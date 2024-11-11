import torch
import torch.nn as nn
from utils.misc import weights_init_


class Policy(nn.Module):
    def __init__(self, state_size, layer_size, action_size):
        super(Policy, self).__init__()
        self.input_shape = state_size
        self.output_size = action_size
        self.layer_size = layer_size
        self.seq = nn.Sequential(nn.Linear(state_size, layer_size),
                                 nn.Dropout(p=0.6),
                                 nn.ReLU(),
                                 nn.Linear(layer_size, action_size)
                                 )
        self.apply(weights_init_)

    def forward(self, x):
        x = self.seq(x)
        # x += ((mask - 1) * x.max().item() * 100)
        # return F.softmax(x, dim=1)
        return x


class Probability(nn.Module):
    def __init__(self, state_size, layer_size):
        super(Probability, self).__init__()
        self.input_shape = state_size
        self.layer_size = layer_size
        self.seq = nn.Sequential(nn.Linear(state_size, layer_size),
                                 nn.Dropout(p=0.6),
                                 nn.ReLU(),
                                 nn.Linear(layer_size, 1)
                                 )
        self.apply(weights_init_)

    def forward(self, x):
        x = self.seq(x)
        return torch.sigmoid(x)


class Critic(nn.Module):
    def __init__(self, state_size, layer_size):
        super(Critic, self).__init__()
        self.input_shape = state_size
        self.layer_size = layer_size
        self.seq_mu = nn.Sequential(nn.Linear(state_size, layer_size),
                                 nn.Dropout(p=0.6),
                                 nn.ReLU(),
                                 nn.Linear(layer_size, 1)
                                 )
        self.seq_log_var = nn.Sequential(nn.Linear(state_size, layer_size),
                                nn.Dropout(p=0.6),
                                nn.ReLU(),
                                nn.Linear(layer_size, 1)
                                )
        self.apply(weights_init_)

    def forward(self, x):
        mu = self.seq_mu(x)
        log_var = self.seq_log_var(x)
        return mu, log_var

