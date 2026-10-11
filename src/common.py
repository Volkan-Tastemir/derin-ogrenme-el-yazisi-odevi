"""Ortak yardımcılar: veri yükleme (train/val/test), MLP modeli, eğitim ve değerlendirme."""
import copy
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
DATA_DIR = os.path.join(ROOT, "data")

# MNIST eğitim verisinin ortalama / standart sapması (normalizasyon için)
MEAN, STD = 0.1307, 0.3081


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize(x):
    """x: [0,1] aralığında tensor -> (x - mean) / std"""
    return (x - MEAN) / STD


def load_mnist(val_size=6000, seed=42):
    """
    MNIST'i indirir ve şu şekilde böler:
      - Eğitim (train):   60000 - val_size  (varsayılan 54000)
      - Doğrulama (val):  val_size          (varsayılan 6000)  -> eğitim setinden ayrılır
      - Test:             10000             (resmi MNIST test seti, model geliştirirken KULLANILMAZ)
    """
    train_full = datasets.MNIST(DATA_DIR, train=True, download=True)
    test = datasets.MNIST(DATA_DIR, train=False, download=True)

    x_all = normalize(train_full.data.float() / 255.0)
    y_all = train_full.targets
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(x_all), generator=g)
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    data = {
        "x_train": x_all[train_idx], "y_train": y_all[train_idx],
        "x_val": x_all[val_idx], "y_val": y_all[val_idx],
        "x_test": normalize(test.data.float() / 255.0), "y_test": test.targets,
    }
    print(f"Train: {len(data['x_train'])} | Validation: {len(data['x_val'])} | Test: {len(data['x_test'])}")
    return data


class MLP(nn.Module):
    """784 -> hidden[0] -> ReLU -> hidden[1] -> ReLU -> ... -> 10 (logits)"""

    def __init__(self, hidden=(256, 128), dropout=0.0):
        super().__init__()
        layers = [nn.Flatten()]
        in_dim = 28 * 28
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_model(cfg):
    return MLP(hidden=tuple(cfg["hidden"]), dropout=cfg.get("dropout", 0.0))


def build_optimizer(model, cfg):
    wd = cfg.get("weight_decay", 0.0)
    if cfg["optimizer"].lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=wd)
    if cfg["optimizer"].lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg.get("momentum", 0.0),
                               weight_decay=wd)
    raise ValueError("optimizer 'adam' veya 'sgd' olmalı")


@torch.no_grad()
def evaluate(model, x, y, device, batch_size=2048):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct = 0.0, 0
    for i in range(0, len(x), batch_size):
        xb, yb = x[i:i + batch_size].to(device), y[i:i + batch_size].to(device)
        out = model(xb)
        total_loss += criterion(out, yb).item()
        correct += (out.argmax(1) == yb).sum().item()
    return total_loss / len(x), correct / len(x)


def train_model(model, data, cfg, device, verbose=True):
    """
    Modeli eğitir. Sadece train ve validation verisi kullanılır (test verisine dokunulmaz).
    En yüksek validation accuracy'ye sahip epoch'un ağırlıkları geri yüklenir.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)
    loader = DataLoader(TensorDataset(data["x_train"], data["y_train"]),
                        batch_size=cfg["batch_size"], shuffle=True)

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc, best_state, best_epoch = -1.0, None, 0
    start = time.perf_counter()

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        loss_sum, correct, n = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(xb)
            correct += (out.argmax(1) == yb).sum().item()
            n += len(xb)
        train_loss, train_acc = loss_sum / n, correct / n
        val_loss, val_acc = evaluate(model, data["x_val"], data["y_val"], device)

        hist["train_loss"].append(train_loss); hist["train_acc"].append(train_acc)
        hist["val_loss"].append(val_loss); hist["val_acc"].append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            best_state = copy.deepcopy(model.state_dict())
        if verbose:
            print(f"Epoch {epoch:02d}/{cfg['epochs']} | train loss {train_loss:.4f} acc {train_acc:.4f} "
                  f"| val loss {val_loss:.4f} acc {val_acc:.4f}")

    train_time = time.perf_counter() - start
    model.load_state_dict(best_state)
    hist["best_epoch"] = best_epoch
    return hist, train_time


def plot_history(hist, path, title=""):
    epochs = range(1, len(hist["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(epochs, hist["train_loss"], label="Train loss")
    ax[0].plot(epochs, hist["val_loss"], label="Validation loss")
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss"); ax[0].set_title("Loss"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(epochs, hist["train_acc"], label="Train accuracy")
    ax[1].plot(epochs, hist["val_acc"], label="Validation accuracy")
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Accuracy"); ax[1].set_title("Accuracy"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def save_model(model, cfg, path):
    torch.save({"state_dict": model.state_dict(), "cfg": cfg}, path)


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["cfg"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["cfg"]
