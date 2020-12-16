import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, hidden_size, dim_input):
        super().__init__()

        self.hidden_size = hidden_size

        # Layer input
        self.W1 = nn.Parameter(torch.Tensor(hidden_size, dim_input))
        self.b1 = nn.Parameter(torch.Tensor(hidden_size, 1))

        # Layer output
        self.W = nn.Parameter(torch.Tensor(1, hidden_size))
        self.b = nn.Parameter(torch.Tensor(1,1))

        

    def init_weights(self):
        for p in self.parameters():
            nn.init.xavier_uniform_(p.data)

    def forward(self, x):

        S1 = torch.sigmoid(torch.mm(self.W1, x) + self.b1)
        # S1 = torch.tanh(torch.mm(self.W1, x)+ self.b1)
        out = torch.mm(self.W, S1) + self.b

        return out
        