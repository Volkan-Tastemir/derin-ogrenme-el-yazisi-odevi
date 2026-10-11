"""
Deney 2 - Hiperparametre / mimari karşılaştırması.
Konfigürasyonlar yalnızca VALIDATION başarımına göre karşılaştırılır ve seçilir.
Test seti sadece seçilen en iyi konfigürasyon için, en sonda bir kez kullanılır.
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (RESULTS_DIR, build_model, evaluate, get_device, load_mnist,
                    save_model, set_seed, train_model)

BASE = {"hidden": [256, 128], "dropout": 0.0, "optimizer": "adam", "lr": 1e-3,
        "batch_size": 128, "epochs": 15, "seed": 42}

EXPERIMENTS = {
    "baseline (256-128, Adam 1e-3)": {},
    "dar (64-32)": {"hidden": [64, 32]},
    "geniş (512-256)": {"hidden": [512, 256]},
    "derin (256-128-64)": {"hidden": [256, 128, 64]},
    "dropout 0.3": {"dropout": 0.3},
    "Adam lr=1e-2 (yüksek)": {"lr": 1e-2},
    "Adam lr=1e-4 (düşük)": {"lr": 1e-4},
    "SGD lr=0.01 momentum=0.9": {"optimizer": "sgd", "lr": 0.01, "momentum": 0.9},
    "batch 32": {"batch_size": 32},
    "batch 512": {"batch_size": 512},
}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = get_device()
    print("Cihaz:", device)
    data = load_mnist(seed=BASE["seed"])

    rows, histories, models = [], {}, {}
    for name, override in EXPERIMENTS.items():
        cfg = {**BASE, **override}
        print(f"\n>>> {name}")
        set_seed(cfg["seed"])
        model = build_model(cfg)
        hist, t = train_model(model, data, cfg, device, verbose=False)
        b = hist["best_epoch"] - 1
        row = {"experiment": name, "best_epoch": hist["best_epoch"],
               "train_loss": hist["train_loss"][b], "val_loss": hist["val_loss"][b],
               "train_acc": hist["train_acc"][b], "val_acc": hist["val_acc"][b],
               "train_time_sec": t}
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
        rows.append(row); histories[name] = hist; models[name] = (model, cfg)

    # Tablo
    with open(os.path.join(RESULTS_DIR, "exp2_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(RESULTS_DIR, "exp2_results.md"), "w") as f:
        f.write("| Deney | En iyi epoch | Train loss | Val loss | Train acc | Val acc | Süre (sn) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['experiment']} | {r['best_epoch']} | {r['train_loss']:.4f} | {r['val_loss']:.4f} | "
                    f"{r['train_acc']:.4f} | {r['val_acc']:.4f} | {r['train_time_sec']:.1f} |\n")

    # Validation accuracy / loss karşılaştırma grafiği
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for name, h in histories.items():
        ax[0].plot(range(1, len(h["val_acc"]) + 1), h["val_acc"], label=name)
        ax[1].plot(range(1, len(h["val_loss"]) + 1), h["val_loss"], label=name)
    ax[0].set_title("Validation accuracy"); ax[1].set_title("Validation loss")
    for a in ax:
        a.set_xlabel("Epoch"); a.grid(alpha=.3)
    ax[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp2_comparison.png"), dpi=150)
    plt.close(fig)

    # En iyi konfigürasyon (validation accuracy'ye göre) -> test seti bir kez kullanılır
    best_row = max(rows, key=lambda r: r["val_acc"])
    model, cfg = models[best_row["experiment"]]
    test_loss, test_acc = evaluate(model, data["x_test"], data["y_test"], device)
    save_model(model, cfg, os.path.join(RESULTS_DIR, "best_model.pt"))
    with open(os.path.join(RESULTS_DIR, "exp2_best.json"), "w") as f:
        json.dump({"best_experiment": best_row["experiment"], "config": cfg,
                   "val_acc": best_row["val_acc"], "test_loss": test_loss, "test_acc": test_acc}, f, indent=2)
    print(f"\nEn iyi (validation'a göre): {best_row['experiment']} | val acc {best_row['val_acc']:.4f} "
          f"| test acc {test_acc:.4f}")


if __name__ == "__main__":
    main()
