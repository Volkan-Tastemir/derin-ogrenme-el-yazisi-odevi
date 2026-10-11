"""Deney 1 - Temel MLP modeli (784-256-128-10, ReLU, CrossEntropy, Adam)."""
import json
import os

from common import (RESULTS_DIR, build_model, evaluate, get_device, load_mnist,
                    plot_history, save_model, set_seed, train_model)

CFG = {
    "hidden": [256, 128],      # 2 hidden layer
    "dropout": 0.0,
    "optimizer": "adam",
    "lr": 1e-3,
    "batch_size": 128,
    "epochs": 20,
    "seed": 42,
}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_seed(CFG["seed"])
    device = get_device()
    print("Cihaz:", device)
    data = load_mnist(seed=CFG["seed"])

    model = build_model(CFG)
    print(model)
    hist, train_time = train_model(model, data, CFG, device)

    # Test seti SADECE bu noktada, eğitim bittikten sonra bir kez kullanılır.
    test_loss, test_acc = evaluate(model, data["x_test"], data["y_test"], device)
    best = hist["best_epoch"] - 1
    results = {
        "config": CFG,
        "best_epoch": hist["best_epoch"],
        "train_loss": hist["train_loss"][best], "val_loss": hist["val_loss"][best],
        "train_acc": hist["train_acc"][best], "val_acc": hist["val_acc"][best],
        "final_epoch_train_loss": hist["train_loss"][-1], "final_epoch_val_loss": hist["val_loss"][-1],
        "test_loss": test_loss, "test_acc": test_acc,
        "train_time_sec": train_time,
        "history": {k: v for k, v in hist.items() if k != "best_epoch"},
    }
    with open(os.path.join(RESULTS_DIR, "exp1_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    plot_history(hist, os.path.join(RESULTS_DIR, "exp1_curves.png"), "Deney 1 - Temel MLP")
    save_model(model, CFG, os.path.join(RESULTS_DIR, "exp1_model.pt"))

    print("\n===== DENEY 1 SONUÇLARI =====")
    print(f"En iyi epoch        : {hist['best_epoch']}")
    print(f"Training loss       : {results['train_loss']:.4f}")
    print(f"Validation loss     : {results['val_loss']:.4f}")
    print(f"Training accuracy   : {results['train_acc']:.4f}")
    print(f"Validation accuracy : {results['val_acc']:.4f}")
    print(f"Test accuracy       : {test_acc:.4f}")
    print(f"Eğitim süresi       : {train_time:.1f} sn")


if __name__ == "__main__":
    main()
