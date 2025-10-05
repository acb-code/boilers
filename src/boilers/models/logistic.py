import torch.nn as nn


class LogisticClassifier(nn.Module):
    def __init__(self, in_features: int, num_classes: int = 2):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.linear(x)  # use with CrossEntropyLoss
