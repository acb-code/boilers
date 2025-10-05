from dataclasses import dataclass
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader


@dataclass
class TrainCfg:
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adam"  # "sgd" or "adam"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def make_optimizer(model, cfg: TrainCfg):
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=0.9
        )
    return torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )


def fit(
    model,
    train_dl: DataLoader,
    val_dl: DataLoader,
    cfg: TrainCfg,
    loss_fn,
    metrics: Dict[str, Any] = None,
):
    device = cfg.device
    model.to(device)
    opt = make_optimizer(model, cfg)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        train_loss = running / len(train_dl.dataset)

        model.eval()
        running = 0.0
        correct = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                running += loss.item() * xb.size(0)
                if logits.ndim == 2 and logits.size(1) > 1:
                    pred = logits.argmax(dim=1)
                else:
                    pred = (logits.squeeze() > 0).long()
                correct += (pred == yb).sum().item()
        val_loss = running / len(val_dl.dataset)
        val_acc = correct / len(val_dl.dataset)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(
            f"epoch {epoch+1:02d} | train {train_loss:.4f} | val {val_loss:.4f} | acc {val_acc:.3f}"
        )

    return history
