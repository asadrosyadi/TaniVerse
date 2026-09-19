"""
CNN backbone comparison for rice-leaf pest classification -- manuscript Table 4.

Reproduces "Table 4. CNN Performance Comparison for Rice Leaf Pest Classification":
ResNet50, MobileNetV2, VGG16, EfficientNet-B0 (proposed) are trained under **one
identical deployment-oriented Keras transfer-learning protocol** (this is the
sec. 4.1 setup, *not* the sec. 4.4 / Table 7 timm+YOLO benchmark which lives in
backend/python/detect_penyakit/benchmark_modern_models.py).

Shared protocol (manuscript sec. 3.3 - 3.7):
  * input 224 x 224 x 3, per-backbone preprocess_input applied inside the graph
  * head  : GAP -> Dropout(0.2) -> Dense(128, ReLU) -> Dropout(0.2) -> Dense(K, softmax)
  * aug   : rotation +/-25 deg, H+V flip, crop scale 0.8-1.0, Gaussian noise (var 0.01)
  * optim : Adam 1e-3 (head warm-up, 40 ep) then 1e-5 (fine-tune top 30 layers, 10 ep)
  * loss  : class-weighted categorical cross-entropy, w_c = N / (K * n_c)
  * eval  : stratified 5-fold CV, metrics reported as mean +/- std
  * seed  : 42

Outputs: table4_cnn_comparison.csv / .md / .json in the working directory.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# reuse the dataset / split / augmentation helpers from the main trainer
import train as cnn_train


BACKBONES = {
    "ResNet50": (
        keras.applications.ResNet50,
        keras.applications.resnet.preprocess_input,
    ),
    "MobileNetV2": (
        keras.applications.MobileNetV2,
        keras.applications.mobilenet_v2.preprocess_input,
    ),
    "VGG16": (
        keras.applications.VGG16,
        keras.applications.vgg16.preprocess_input,
    ),
    "EfficientNet-B0": (
        keras.applications.EfficientNetB0,
        keras.applications.efficientnet.preprocess_input,
    ),
}


def parse_args():
    p = argparse.ArgumentParser(description="Bandingkan backbone CNN (Table 4 manuskrip)")
    p.add_argument("--data-dir", type=str, default=".")
    p.add_argument("--train-dir", type=str, default=None)
    p.add_argument("--val-dir", type=str, default=None)
    p.add_argument("--test-dir", type=str, default=None)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=40, help="epoch head warm-up")
    p.add_argument("--fine-tune-epochs", type=int, default=10)
    p.add_argument("--fine-tune-layers", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--fine-tune-lr", type=float, default=1e-5)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache-images", dest="cache_images", action="store_true", default=True,
                   help="decode semua citra sekali ke RAM (default: aktif)")
    p.add_argument("--no-cache-images", dest="cache_images", action="store_false")
    p.add_argument("--models", type=str, default=",".join(BACKBONES),
                   help="daftar backbone dipisah koma")
    return p.parse_args()


def build_backbone_model(name, img_size, n_classes, dropout, fine_tune_layers, seed):
    ctor, preprocess = BACKBONES[name]
    base = ctor(include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3))
    base.trainable = False

    aug = cnn_train.build_augmentation(seed)
    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = aug(inputs)
    x = layers.Lambda(preprocess, name="preprocess")(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name=name.replace("-", "_")), base


def decode_all_images(filepaths, img_size):
    """Decode and resize every image once into a uint8 array held in RAM.

    Identical images, identical resize; only the I/O path changes. Re-decoding
    ~6,800 JPEGs per epoch makes the pipeline CPU-bound and starves the GPU, so
    with 5 folds x 50 epochs the decode work would otherwise dominate the run.
    Storage is len(filepaths) x img_size x img_size x 3 bytes (~1.6 GB for the
    full 10,407-image dataset at 224 x 224).
    """
    out = np.zeros((len(filepaths), img_size, img_size, 3), dtype=np.uint8)
    for i, p in enumerate(filepaths):
        raw = tf.io.read_file(p)
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, (img_size, img_size))
        out[i] = tf.cast(tf.clip_by_value(img, 0, 255), tf.uint8).numpy()
        if (i + 1) % 1000 == 0:
            print(f"    decoded {i + 1}/{len(filepaths)}", flush=True)
    return out


def make_raw_dataset(filepaths, labels, n_classes, img_size, batch_size, shuffle, seed,
                     images=None, indices=None):
    """0-255 float images; per-backbone preprocess happens inside the model graph.

    When `images` (the pre-decoded uint8 array) and `indices` are supplied the
    JPEG decode is skipped and the cached pixels are used instead.
    """
    if images is not None and indices is not None:
        # Keep the ~1 GB cache in host memory: on a 4 GB GPU, materialising it as a
        # device constant starves the model of VRAM.
        with tf.device("/CPU:0"):
            ds = tf.data.Dataset.from_tensor_slices((images[indices], labels))
            if shuffle:
                ds = ds.shuffle(min(len(indices), 4096), seed=seed, reshuffle_each_iteration=True)
            ds = ds.map(lambda im, lb: (tf.cast(im, tf.float32), tf.one_hot(lb, n_classes)),
                        num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
        return ds.prefetch(tf.data.AUTOTUNE)

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if shuffle:
        ds = ds.shuffle(min(len(filepaths), 4096), seed=seed, reshuffle_each_iteration=True)

    def _load(path, label):
        raw = tf.io.read_file(path)
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, (img_size, img_size))
        return tf.cast(img, tf.float32), tf.one_hot(label, n_classes)

    return ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train_eval_fold(name, tr_idx, val_idx, te_idx, filepaths, labels, class_names, args, images=None):
    n_classes = len(class_names)
    cw = cnn_train.compute_class_weights(labels[tr_idx], n_classes)
    train_ds = make_raw_dataset(filepaths[tr_idx], labels[tr_idx], n_classes, args.img_size,
                                args.batch_size, True, args.seed, images, tr_idx)
    val_ds = make_raw_dataset(filepaths[val_idx], labels[val_idx], n_classes, args.img_size,
                              args.batch_size, False, args.seed, images, val_idx)

    model, base = build_backbone_model(name, args.img_size, n_classes, args.dropout, args.fine_tune_layers, args.seed)
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=0),
    ]

    model.compile(optimizer=keras.optimizers.Adam(args.lr), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, class_weight=cw, callbacks=cbs, verbose=2)

    if args.fine_tune_epochs > 0:
        base.trainable = True
        for layer in base.layers[:-args.fine_tune_layers]:
            layer.trainable = False
        model.compile(optimizer=keras.optimizers.Adam(args.fine_tune_lr), loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs, class_weight=cw, callbacks=cbs, verbose=2)

    test_ds = make_raw_dataset(filepaths[te_idx], labels[te_idx], n_classes, args.img_size,
                               args.batch_size, False, args.seed, images, te_idx)
    probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    y_true = labels[te_idx]

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, pr in zip(y_true, y_pred):
        cm[t, pr] += 1
    acc = float(np.trace(cm) / max(cm.sum(), 1))
    prec, rec, f1 = [], [], []
    for c in range(n_classes):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        pcs = tp / (tp + fp) if (tp + fp) else 0.0
        rcs = tp / (tp + fn) if (tp + fn) else 0.0
        prec.append(pcs); rec.append(rcs)
        f1.append(2 * pcs * rcs / (pcs + rcs) if (pcs + rcs) else 0.0)
    return {"accuracy": acc, "precision": float(np.mean(prec)),
            "recall": float(np.mean(rec)), "f1": float(np.mean(f1))}


def main():
    args = parse_args()
    keras.utils.set_random_seed(args.seed)

    data_dir = Path(args.data_dir)
    train_dir = Path(args.train_dir) if args.train_dir else data_dir / "train"
    val_dir = Path(args.val_dir) if args.val_dir else data_dir / "val"
    test_dir = Path(args.test_dir) if args.test_dir else (data_dir / "test" if (data_dir / "test").is_dir() else None)

    filepaths, labels, class_names = cnn_train.collect_all_images(train_dir, val_dir, test_dir)
    print(f"Total gambar: {len(filepaths)} | {len(class_names)} kelas")

    images = None
    if args.cache_images:
        gb = len(filepaths) * args.img_size * args.img_size * 3 / 1e9
        print(f"Men-decode seluruh {len(filepaths)} citra sekali ke RAM (~{gb:.1f} GB uint8)...", flush=True)
        images = decode_all_images(filepaths, args.img_size)
        print("    selesai; JPEG tidak akan di-decode ulang tiap epoch", flush=True)

    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    rows = []
    for name in wanted:
        if name not in BACKBONES:
            print(f"[skip] backbone tidak dikenal: {name}")
            continue
        print(f"\n############ {name} : stratified {args.cv_folds}-fold CV ############")
        fold_scores = []
        for fold, (tr_idx, te_idx) in enumerate(cnn_train.stratified_kfold(labels, args.cv_folds, args.seed), start=1):
            inner_val_frac = args.val_fraction / (1.0 - 1.0 / args.cv_folds)
            tr_sub, val_sub = cnn_train.stratified_split(labels[tr_idx], [1.0 - inner_val_frac, inner_val_frac], args.seed + fold)
            keras.backend.clear_session()
            keras.utils.set_random_seed(args.seed + fold)
            print(f"  -- {name} fold {fold}/{args.cv_folds}", flush=True)
            fold_scores.append(
                train_eval_fold(name, tr_idx[tr_sub], tr_idx[val_sub], te_idx,
                                filepaths, labels, class_names, args, images)
            )
        summary = {}
        for metric in ("accuracy", "precision", "recall", "f1"):
            vals = [s[metric] for s in fold_scores]
            std = float(np.std(vals, ddof=0))
            summary[metric] = {
                "mean": float(np.mean(vals)),
                "std": std,
                "ci95": float(1.96 * std / np.sqrt(max(len(vals), 1))),   # manuscript Eq. 9
            }
        rows.append({"model": name, "folds": fold_scores, "summary": summary})
        s = summary
        print(f"  => acc {s['accuracy']['mean']*100:.1f} +/- {s['accuracy']['std']*100:.1f} | "
              f"P {s['precision']['mean']:.2f} R {s['recall']['mean']:.2f} F1 {s['f1']['mean']:.2f}", flush=True)

        # Persist per-backbone results immediately: a multi-day CPU run must not
        # lose finished backbones if a later one is interrupted.
        Path(f"table4_fold_results_{name}.json").write_text(
            json.dumps({"model": name, "folds": fold_scores, "summary": summary,
                        "config": vars(args)}, indent=2), encoding="utf-8")
        print(f"  [saved] table4_fold_results_{name}.json", flush=True)

    # ---- emit Table 4 ----
    Path("table4_cnn_comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    hdr = "| Model | Accuracy (%) | Precision | Recall | F1-Score |"
    sep = "| --- | --- | --- | --- | --- |"
    lines = [hdr, sep]
    csv_lines = ["model,accuracy_mean,accuracy_std,precision_mean,precision_std,recall_mean,recall_std,f1_mean,f1_std"]
    for r in rows:
        s = r["summary"]
        tag = r["model"] + (" (Proposed)" if r["model"] == "EfficientNet-B0" else "")
        lines.append(
            f"| {tag} | {s['accuracy']['mean']*100:.1f} +/- {s['accuracy']['std']*100:.1f} "
            f"| {s['precision']['mean']:.2f} +/- {s['precision']['std']:.2f} "
            f"| {s['recall']['mean']:.2f} +/- {s['recall']['std']:.2f} "
            f"| {s['f1']['mean']:.2f} +/- {s['f1']['std']:.2f} |"
        )
        csv_lines.append(
            f"{r['model']},{s['accuracy']['mean']:.4f},{s['accuracy']['std']:.4f},"
            f"{s['precision']['mean']:.4f},{s['precision']['std']:.4f},"
            f"{s['recall']['mean']:.4f},{s['recall']['std']:.4f},"
            f"{s['f1']['mean']:.4f},{s['f1']['std']:.4f}"
        )
    Path("table4_cnn_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path("table4_cnn_comparison.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    print("\n=== Table 4. CNN Performance Comparison for Rice Leaf Pest Classification ===")
    print("\n".join(lines))
    print("\nDisimpan: table4_cnn_comparison.{md,csv,json}")


if __name__ == "__main__":
    main()
