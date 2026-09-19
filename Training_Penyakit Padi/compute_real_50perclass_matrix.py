"""
Compute the REAL confusion matrix for a genuine 500-image evaluation subset
(50 images per class, the first 50 in sorted filename order within each
val/<class>/ folder -- deterministic, reproducible, disclosed here), matching
the scope described in the manuscript ("500 images, 50 per class").

This replaces the hand-typed FIG8_AFTER matrix in generate_confusion_matrix.py
with an actually-measured result on an actually-defined subset.
"""
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
N_PER_CLASS = 50


def load_image(path, img_size=224):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size))
    return tf.cast(img, tf.float32).numpy()


def main():
    here = Path(__file__).resolve().parent
    class_names = json.loads((here / "class_names.json").read_text(encoding="utf-8"))
    K = len(class_names)
    model = keras.models.load_model(here / "efficientnetb0_model.h5")

    val_dir = here / "val"
    cm = np.zeros((K, K), dtype=np.int64)
    used_files = {}

    for ci, cname in enumerate(class_names):
        cdir = val_dir / cname
        images = sorted(p for p in cdir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        subset = images[:N_PER_CLASS]
        used_files[cname] = [p.name for p in subset]
        print(f"[scan] {cname}: using first {len(subset)} of {len(images)} images (sorted filename order)")
        batch = [load_image(str(p)) for p in subset]
        probs = model.predict(np.stack(batch), verbose=0)
        for p in probs:
            pj = int(np.argmax(p))
            cm[ci, pj] += 1

    total = cm.sum()
    correct = np.trace(cm)
    acc = correct / total * 100
    print(f"\n=== REAL 50-per-class subset ({total} images): accuracy = {acc:.2f}% ===\n")
    print(f"{'class':<26}{'n':>4}{'correct':>9}{'acc%':>8}")
    for ci, cname in enumerate(class_names):
        n = cm[ci].sum()
        print(f"{cname:<26}{n:>4}{cm[ci, ci]:>9}{(cm[ci, ci]/n*100 if n else 0):>7.1f}%")

    bi, bri = class_names.index("blast"), class_names.index("brown_spot")
    print(f"\nblast -> brown_spot : {cm[bi, bri]} / {cm[bi].sum()} ({cm[bi, bri]/cm[bi].sum()*100:.1f}%)")
    print(f"brown_spot -> blast : {cm[bri, bi]} / {cm[bri].sum()} ({cm[bri, bi]/cm[bri].sum()*100:.1f}%)")

    out = {
        "n_total": int(total),
        "n_per_class": N_PER_CLASS,
        "accuracy_percent": float(acc),
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),
        "sampling_rule": "first N_PER_CLASS images per class in sorted filename order, from val/<class>/",
        "files_used": used_files,
    }
    (here / "real_50perclass_confusion_matrix.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved: real_50perclass_confusion_matrix.json")


if __name__ == "__main__":
    main()
