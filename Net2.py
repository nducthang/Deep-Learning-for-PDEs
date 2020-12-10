from multiprocessing import Semaphore
import torch
import torch.nn as nn
from torch.nn.modules import activation
import torch.nn.functional as F
class Net2(nn.Module):
    def __init__(self, hidden_size, dim_input):
        super().__init__()
        self.hidden_size = hidden_size
        self.dim_input = dim_input

        self.input_layer = nn.Linear(dim_input, hidden_size)
        self.h1 = nn.Linear(hidden_size, hidden_size)
        self.h2 = nn.Linear(hidden_size, hidden_size)
        self.h3 = nn.Linear(hidden_size, hidden_size)
        self.h4 = nn.Linear(hidden_size, hidden_size)
        self.h5 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.view(-1, x.shape[0])
        x = F.tanh(self.input_layer(x))
        x = F.tanh(self.h1(x))
        x = F.tanh(self.h2(x))
        x = F.tanh(self.h3(x))
        x = F.tanh(self.h4(x))
        x = F.tanh(self.h5(x))
        x = F.tanh(self.output_layer(x))
        return x