"""Eğitilmiş modeli kendi el yazısı öğrenci numarası rakamları üzerinde değerlendirir."""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import RESULTS_DIR, ROOT, get_device, load_model, normalize
from prepare_handwriting import STUDENT_NUMBER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="model dosyası (varsayılan: best_model.pt, yoksa exp1_model.pt)")
    args = ap.parse_args()

    path = args.model
    if path is None:
        best = os.path.join(RESULTS_DIR, "best_model.pt")
        path = best if os.path.exists(best) else os.path.join(RESULTS_DIR, "exp1_model.pt")
    device = get_device()
    model, cfg = load_model(path, device)
    print("Model:", path, "| config:", cfg)

    digits = np.load(os.path.join(ROOT, "my_handwriting", "digits", "all_digits.npy"))  # (N,28,28) [0,1]
    x = normalize(torch.tensor(digits, dtype=torch.float32))   # MNIST ile aynı normalizasyon
    with torch.no_grad():
        probs = torch.softmax(model(x.to(device)), dim=1).cpu().numpy()
    preds, conf = probs.argmax(1), probs.max(1)
    labels = np.array([int(c) for c in STUDENT_NUMBER])
    correct = preds == labels
    acc = correct.mean()

    print(f"\nGerçek numara : {STUDENT_NUMBER}")
    print(f"Model tahmini : {''.join(map(str, preds))}")
    print(f"Doğruluk      : {correct.sum()}/{len(labels)} = {acc:.2%}")
    for i, (l, p, c) in enumerate(zip(labels, preds, conf)):
        print(f"  rakam {i + 1:2d}: gerçek={l} tahmin={p} güven={c:.2f} {'OK' if l == p else 'YANLIŞ'}")

    fig, ax = plt.subplots(1, len(digits), figsize=(1.5 * len(digits), 2.2))
    for i, a in enumerate(ax):
        a.imshow(digits[i], cmap="gray"); a.axis("off")
        a.set_title(f"gerçek {labels[i]}\ntahmin {preds[i]}\n({conf[i]:.2f})", fontsize=8,
                    color="green" if correct[i] else "red")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "handwriting_predictions.png"), dpi=150)

    with open(os.path.join(RESULTS_DIR, "handwriting_results.json"), "w") as f:
        json.dump({"model": os.path.basename(path), "true": STUDENT_NUMBER,
                   "pred": "".join(map(str, preds)), "accuracy": float(acc),
                   "confidence": [float(c) for c in conf]}, f, indent=2)


if __name__ == "__main__":
    main()
