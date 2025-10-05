import torch
import torch.nn as nn

def make_mlp(sizes, activation="relu", dropout=0.0):
    acts = dict(relu=nn.ReLU, tanh=nn.Tanh, gelu=nn.GELU)
    layers = []
    for i in range(len(sizes)-1):
        layers += [nn.Linear(sizes[i], sizes[i+1])]
        if i < len(sizes)-2:
            layers += [acts[activation]()]
            if dropout > 0: layers += [nn.Dropout(dropout)]
    return nn.Sequential(*layers)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes=(128,128), num_classes=10, activation="relu", dropout=0.0):
        super().__init__()
        sizes = [input_dim, *hidden_sizes, num_classes]
        self.net = make_mlp(sizes, activation, dropout)

    def forward(self, x):
        return self.net(x)
