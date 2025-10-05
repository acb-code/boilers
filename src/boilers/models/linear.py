import torch
import torch.nn as nn

class LinearRegressor(nn.Module):
    def __init__(self, in_features: int, out_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)
