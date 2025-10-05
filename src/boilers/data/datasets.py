from typing import Tuple, Dict
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def moons(batch_size=128, n_samples=4000, noise=0.2):
    X, y = make_moons(n_samples=n_samples, noise=noise)
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X).astype("float32")
    y = y.astype("int64")
    Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.2, random_state=1337)
    tr = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    val = TensorDataset(torch.from_numpy(Xval), torch.from_numpy(yval))
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=True),
        DataLoader(val, batch_size=batch_size),
        {"input_dim": 2, "num_classes": 2}
    )
