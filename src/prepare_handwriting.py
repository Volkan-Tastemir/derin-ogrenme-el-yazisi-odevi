"""
El yazısı öğrenci numarası fotoğrafını tek tek rakamlara ayırır ve
MNIST formatına (28x28, siyah arka plan üzerine beyaz rakam, [0,1] aralığı) çevirir.

Kullanım:
    python src/prepare_handwriting.py
Çıktılar:
    my_handwriting/digits/digit_XX.png   (28x28 gri tonlamalı)
    my_handwriting/digits/all_digits.npy (N,28,28 float32, [0,1])
    my_handwriting/preview.png           (kontrol görseli)
"""
import os
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_PATH = os.path.join(ROOT, "my_handwriting", "student_number_original.jpeg")
OUT_DIR = os.path.join(ROOT, "my_handwriting", "digits")
STUDENT_NUMBER = "24120205046"  # etiket (gerçek değerler)


def ink_mask(rgb):
    """Mavi mürekkebi gri kareli kağıttan ayırır (mavi - kırmızı farkı)."""
    rgb = rgb.astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    score = b - r                       # mavi mürekkepte yüksek, gri çizgide ~0
    thr = max(25.0, 0.45 * np.percentile(score, 99.9))
    mask = score > thr
    mask = ndi.binary_opening(mask, iterations=1)
    mask = ndi.binary_dilation(mask, iterations=2)  # kopuk vuruşları birleştir
    return mask, np.clip(score, 0, None)


def split_digits(mask, expected=None):
    lab, n = ndi.label(mask)
    objs = ndi.find_objects(lab)
    boxes = []
    for i, sl in enumerate(objs, start=1):
        area = (lab[sl] == i).sum()
        if area < 150:                  # gürültü
            continue
        boxes.append([sl[1].start, sl[1].stop, sl[0].start, sl[0].stop])
    boxes.sort()
    # x aralığı belirgin çakışan kutuları birleştir (ör. iki parçalı '4')
    merged = []
    for b in boxes:
        if merged and b[0] < merged[-1][1] - 0.3 * min(b[1] - b[0], merged[-1][1] - merged[-1][0]):
            m = merged[-1]
            merged[-1] = [min(m[0], b[0]), max(m[1], b[1]), min(m[2], b[2]), max(m[3], b[3])]
        else:
            merged.append(b)
    # Beklenenden fazla parça varsa (çok vuruşlu rakamlar, ör. iki parçalı '4'),
    # birleşince makul bir rakam oranına (genişlik/yükseklik <= 1) sahip olan komşu
    # parçalardan, aradaki boşluğu en küçük olanı birleştir.
    while expected and len(merged) > expected:
        best, best_gap = None, 1e9
        for i in range(len(merged) - 1):
            a, c = merged[i], merged[i + 1]
            w = max(a[1], c[1]) - min(a[0], c[0])
            h = max(a[3], c[3]) - min(a[2], c[2])
            gap = c[0] - a[1]
            if w / h <= 1.0 and gap < best_gap:
                best, best_gap = i, gap
        if best is None:
            break
        a, c = merged[best], merged[best + 1]
        merged[best:best + 2] = [[min(a[0], c[0]), max(a[1], c[1]), min(a[2], c[2]), max(a[3], c[3])]]
    return merged


def to_mnist(score, box, pad=6):
    x0, x1, y0, y1 = box
    crop = score[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad].copy()
    crop = crop / (crop.max() + 1e-8)               # [0,1], beyaz = mürekkep
    crop[crop < 0.15] = 0
    ys, xs = np.nonzero(crop > 0)
    crop = crop[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    scale = 20.0 / max(h, w)                        # MNIST: 20x20 kutuya sığdır
    im = Image.fromarray((crop * 255).astype(np.uint8)).resize(
        (max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    canvas = np.zeros((28, 28), np.float32)
    top, left = (28 - arr.shape[0]) // 2, (28 - arr.shape[1]) // 2
    canvas[top:top + arr.shape[0], left:left + arr.shape[1]] = arr
    cy, cx = ndi.center_of_mass(canvas)             # ağırlık merkezini ortala (MNIST gibi)
    canvas = ndi.shift(canvas, (13.5 - cy, 13.5 - cx), order=1)
    return np.clip(canvas, 0, 1)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rgb = np.asarray(Image.open(IMG_PATH).convert("RGB"))
    mask, score = ink_mask(rgb)
    boxes = split_digits(mask, expected=len(STUDENT_NUMBER))
    print(f"Bulunan rakam sayısı: {len(boxes)} (beklenen {len(STUDENT_NUMBER)})")
    if len(boxes) != len(STUDENT_NUMBER):
        print("UYARI: rakam sayısı eşleşmiyor, preview.png dosyasını kontrol edin.")
    digits = np.stack([to_mnist(score, b) for b in boxes])
    for i, d in enumerate(digits):
        Image.fromarray((d * 255).astype(np.uint8)).save(os.path.join(OUT_DIR, f"digit_{i:02d}.png"))
    np.save(os.path.join(OUT_DIR, "all_digits.npy"), digits)

    fig, ax = plt.subplots(1, len(digits), figsize=(1.4 * len(digits), 1.8))
    for i, a in enumerate(np.atleast_1d(ax)):
        a.imshow(digits[i], cmap="gray"); a.axis("off")
        if i < len(STUDENT_NUMBER):
            a.set_title(STUDENT_NUMBER[i])
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "my_handwriting", "preview.png"), dpi=120)


if __name__ == "__main__":
    main()
