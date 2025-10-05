import torch.nn as nn

from boilers.data.datasets import moons
from boilers.models.mlp import MLP
from boilers.training.loop import TrainCfg, fit


def test_one_epoch_smoke():
    # small dataset for speed
    train_dl, val_dl, meta = moons(batch_size=64, n_samples=512, noise=0.2)
    model = MLP(
        input_dim=meta["input_dim"], hidden_sizes=(32,), num_classes=meta["num_classes"]
    )
    cfg = TrainCfg(epochs=1, lr=1e-2, optimizer="sgd", device="cpu")
    hist = fit(model, train_dl, val_dl, cfg, loss_fn=nn.CrossEntropyLoss())
    assert "train_loss" in hist and len(hist["train_loss"]) == 1
    assert "val_loss" in hist and len(hist["val_loss"]) == 1
