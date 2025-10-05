import torch

from boilers.models.logistic import LogisticClassifier
from boilers.models.mlp import MLP


def test_mlp_forward_shape():
    model = MLP(input_dim=2, hidden_sizes=(16, 8), num_classes=3)
    x = torch.randn(5, 2)
    y = model(x)
    assert y.shape == (5, 3)


def test_logistic_forward_shape():
    model = LogisticClassifier(in_features=2, num_classes=2)
    x = torch.randn(7, 2)
    y = model(x)
    assert y.shape == (7, 2)
