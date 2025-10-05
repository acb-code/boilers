# src/boilers/nb/nbtools.py
import matplotlib.pyplot as plt
import numpy as np
import torch

__all__ = ["plot_decision_boundary", "plot_learning_curves"]


def plot_decision_boundary(
    model, dataloader, title="Decision Boundary", device=None, resolution=200
):
    """Plot 2D decision boundary for binary or multi-class classifiers."""
    model.eval()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Collect dataset samples (assumes 2D inputs)
    X, y = [], []
    for xb, yb in dataloader:
        X.append(xb)
        y.append(yb)
    X = torch.cat(X).to(device)
    y = torch.cat(y).cpu().numpy()

    # Mesh grid over input space
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )

    grid = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float().to(device)
    with torch.no_grad():
        logits = model(grid)
        if logits.ndim == 2 and logits.shape[1] > 1:
            preds = logits.argmax(dim=1).cpu().numpy()
        else:
            preds = (logits.squeeze() > 0).long().cpu().numpy()
    preds = preds.reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, preds, alpha=0.3, cmap="coolwarm")
    plt.scatter(X[:, 0].cpu(), X[:, 1].cpu(), c=y, s=15, cmap="coolwarm", edgecolor="k")
    plt.title(title)
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.tight_layout()
    plt.show()


def plot_learning_curves(
    history, metrics=("train_loss", "val_loss"), title="Learning Curves"
):
    """Plot training and validation loss/accuracy from fit() history dict."""
    plt.figure(figsize=(6, 4))
    for key in metrics:
        if key in history:
            plt.plot(history[key], label=key.replace("_", " ").title())
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
