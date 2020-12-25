import torch
import torch.nn as nn
from torch.types import Number
import numpy as np


class NeuralNetwork(nn.Module):
    def __init__(self, dim_input, dim_output, hidden_size, num_hidden_layer):
        super().__init__()

        self.dim_input = dim_input
        self.dim_output = dim_output
        self.hidden_size = hidden_size
        self.num_hidden_layer = num_hidden_layer
        self.W = {}
        self.b = {}

        # hidden layer
        for i in range(num_hidden_layer):
            if i == 0:
                self.W[i] = nn.Parameter(torch.Tensor(hidden_size, dim_input))
            else:
                self.W[i] = nn.Parameter(
                    torch.Tensor(hidden_size, hidden_size))
            self.b[i] = nn.Parameter(torch.Tensor(hidden_size, 1))

        # Layer output
        self.W[num_hidden_layer] = nn.Parameter(
            torch.Tensor(dim_output, hidden_size))
        self.b[num_hidden_layer] = nn.Parameter(torch.Tensor(dim_output, 1))

        self.myparameters = nn.ParameterList(
            [w for w in self.W.values()] + [b for b in self.b.values()])

    def init_weights(self):
        for p in self.parameters():
            nn.init.xavier_uniform_(p.data)

    def forward(self, x):
        layer = None
        for i in range(self.num_hidden_layer):
            if i == 0:
                layer = torch.mm(self.W[i], x) + self.b[i]
            else:
                layer = torch.mm(self.W[i], layer) + self.b[i]
            layer = torch.tanh(layer)

        out = torch.mm(self.W[self.num_hidden_layer],
                       layer) + self.b[self.num_hidden_layer]

        return out


def count_parameters(model):
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            if param.dim() > 1:
                print(name, ':', 'x'.join(str(x)
                                          for x in list(param.size())), '=', num_param)
            else:
                print(name, ':', num_param)
            total_param += num_param
    print(total_param)
