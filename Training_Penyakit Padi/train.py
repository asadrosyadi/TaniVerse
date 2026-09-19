"""
EfficientNet-B0 training for rice disease / pest classification (Layer 4, CNN branch).

Aligned with the manuscript "Hybrid CNN-LSTM Edge AIoT System ..." (BEEI):
  * Backbone .................. EfficientNet-B0, ImageNet weights, include_top=False
  * Input .................... 224 x 224 x 3, efficientnet.preprocess_input (built into the backbone)
  * Head .................... GAP -> Dropout(0.2) -> Dense(128, ReLU) -> Dropout(0.2) -> Dense(K, softmax)
  * Augmentation (sec. 3.3) .. rotation +/-25 deg, horizontal & vertical flip,
                               random crop with scale factor 0.8-1.0, Gaussian noise (variance 0.01)
  * Optimiser ............... Adam, lr 1e-3 (head warm-up) then 1e-5 (fine-tune)
  * Epochs (Table 3) ........ 50 for the CNN  (40 head warm-up + 10 fine-tune of the top 30 layers)
  * Regularisation .......... Dropout 0.2, EarlyStopping (val_accuracy), ReduceLROnPlateau
  * Loss ................... class-weighted categorical cross-entropy, w_c = N / (K * n_c)
  * Validation (sec. 3.7) ... stratified 70/15/15 split *and* 5-fold cross-validation,
                              results reported as mean +/- standard deviation across folds
  * Reproducibility ........ fixed random seed (42) for every experiment

Two entry modes:
  * default (--cv-folds 0) ... single stratified 70/15/15 split -> one model file
  * --cv-folds 5 ............ stratified k-fold CV, prints mean +/- std of
                              accuracy / precision / recall / F1 and writes cv_results.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Manuscript sec. 3.3 augmentation constants -------------------------------------------------
ROTATION_DEG = 25.0                 # random rotation within +/- 25 degrees
ROTATION_FACTOR = ROTATION_DEG / 360.0
CROP_SCALE_MIN = 0.8               # random crop / zoom scaling factor 0.8 - 1.0
GAUSSIAN_NOISE_VARIANCE = 0.01     # on the normalised [0, 1] pixel scale
# EfficientNet-B0 in keras.applications consumes 0-255 inputs, so the [0,1] std is
# rescaled to the 0-255 domain: std = sqrt(0.01) * 255 = 25.5
GAUSSIAN_NOISE_STDDEV = float(np.sqrt(GAUSSIAN_NOISE_VARIANCE) * 255.0)
# The same magnitude expressed on the normalised [0, 1] scale (std = sqrt(0.01) = 0.1),
# which is what Keras 3's GaussianNoise accepts.
GAUSSIAN_NOISE_STDDEV_UNIT = float(np.sqrt(GAUSSIAN_NOISE_VARIANCE))


def parse_args():
    parser = argparse.ArgumentParser(description="Train EfficientNetB0 untuk klasifikasi penyakit padi")
    parser.add_argument("--data-dir", type=str, default=".", help="Folder yang berisi subfolder train/ val/ (dan test/ opsional)")
    parser.add_argument("--train-dir", type=str, default=None, help="Override folder train (default: <data-dir>/train)")
    parser.add_argument("--val-dir", type=str, default=None, help="Override folder val (default: <data-dir>/val)")
    parser.add_argument("--test-dir", type=str, default=None, help="Override folder test (default: <data-dir>/test bila ada)")
    parser.add_argument("--img-size", type=int, default=224, help="Ukuran gambar (persegi) input model")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40, help="Jumlah epoch tahap 1 (head, base dibekukan). Manuskrip: 40 + 10 = 50")
    parser.add_argument("--fine-tune-epochs", type=int, default=10, help="Jumlah epoch tahap 2 (fine-tuning). 0 = lewati")
    parser.add_argument("--fine-tune-layers", type=int, default=30, help="Jumlah layer teratas base model yang dibuka saat fine-tuning")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate tahap 1")
    parser.add_argument("--fine-tune-lr", type=float, default=1e-5, help="Learning rate tahap 2 (fine-tuning)")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate (manuskrip sec. 3.5: 0.2)")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="Porsi validasi pada mode split tunggal (70/15/15)")
    parser.add_argument("--test-fraction", type=float, default=0.15, help="Porsi test pada mode split tunggal (70/15/15)")
    parser.add_argument("--cv-folds", type=int, default=0, help="0 = split tunggal 70/15/15; >=2 = stratified k-fold cross-validation")
    parser.add_argument("--output", type=str, default="efficientnetb0_model.h5", help="Path file model .h5 hasil training (mode split tunggal)")
    parser.add_argument("--fixed-split", action="store_true",
                        help="Latih langsung pada train/ dan val/ apa adanya (tanpa pooling/repartisi). "
                             "Tidak ada held-out test slice terpisah -- val/ dipakai untuk checkpoint "
                             "selection DAN sebagai partisi yang dilaporkan, sesuai deskripsi manuskrip "
                             "untuk checkpoint Figure 8/Table 7 (bukan mode 70/15/15 atau 5-fold CV).")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# --------------------------------------------------------------------------------------------
# Dataset helpers
# --------------------------------------------------------------------------------------------
def _list_split_dir(split_dir: Path):
    """Return (filepaths, labels, class_names) for a directory of per-class subfolders."""
    if not split_dir or not Path(split_dir).is_dir():
        return [], [], []
    class_names = sorted(p.name for p in Path(split_dir).iterdir() if p.is_dir())
    filepaths, labels = [], []
    for idx, name in enumerate(class_names):
        for img in sorted(Path(split_dir, name).rglob("*")):
            if img.is_file() and img.suffix.lower() in IMAGE_EXTENSIONS:
                filepaths.append(str(img))
                labels.append(idx)
    return filepaths, labels, class_names


def collect_all_images(train_dir, val_dir, test_dir):
    """Pool every available split into one (filepaths, labels, class_names) table.

    The manuscript performs stratified re-splitting / k-fold CV over the whole
    labelled set, so training / validation / test folders are merged first.
    """
    class_names = None
    all_paths, all_labels = [], []
    for split in (train_dir, val_dir, test_dir):
        paths, labels, names = _list_split_dir(split)
        if not paths:
            continue
        if class_names is None:
            class_names = names
        elif names != class_names:
            raise ValueError(
                f"Struktur kelas berbeda antar split: {names} vs {class_names}"
            )
        all_paths.extend(paths)
        all_labels.extend(labels)
    if not all_paths:
        raise FileNotFoundError("Tidak ada gambar ditemukan pada folder train/val/test.")
    return np.array(all_paths), np.array(all_labels, dtype=np.int64), class_names


def stratified_split(labels, fractions, seed):
    """Return a list of index arrays, one per fraction, stratified by label."""
    rng = np.random.default_rng(seed)
    idx_per_split = [[] for _ in fractions]
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        bounds = np.cumsum([int(round(f * n)) for f in fractions])
        bounds[-1] = n
        start = 0
        for s, end in enumerate(bounds):
            idx_per_split[s].extend(cls_idx[start:end].tolist())
            start = end
    return [np.array(sorted(s), dtype=np.int64) for s in idx_per_split]


def stratified_kfold(labels, n_splits, seed):
    """Yield (train_idx, test_idx) tuples for stratified k-fold CV (no sklearn dependency)."""
    rng = np.random.default_rng(seed)
    fold_bins = [[] for _ in range(n_splits)]
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        for i, sample_idx in enumerate(cls_idx):
            fold_bins[i % n_splits].append(sample_idx)
    fold_bins = [np.array(sorted(b), dtype=np.int64) for b in fold_bins]
    all_idx = np.arange(len(labels))
    for k in range(n_splits):
        test_idx = fold_bins[k]
        train_idx = np.array(sorted(set(all_idx.tolist()) - set(test_idx.tolist())), dtype=np.int64)
        yield train_idx, test_idx


def make_dataset(filepaths, labels, n_classes, img_size, batch_size, shuffle, seed):
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(filepaths), 4096), seed=seed, reshuffle_each_iteration=True)

    def _load(path, label):
        raw = tf.io.read_file(path)
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, (img_size, img_size))
        img = tf.cast(img, tf.float32)
        return img, tf.one_hot(label, n_classes)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def compute_class_weights(labels, n_classes):
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    total = counts.sum()
    weights = total / (n_classes * counts)          # w_c = N / (K * n_c)
    return {i: float(w) for i, w in enumerate(weights)}


# --------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------
def build_augmentation(seed):
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical", seed=seed),
            layers.RandomRotation(ROTATION_FACTOR, seed=seed),                       # +/- 25 deg
            layers.RandomZoom(                                                       # crop scale 0.8 - 1.0
                height_factor=(-(1.0 - CROP_SCALE_MIN), 0.0),
                width_factor=(-(1.0 - CROP_SCALE_MIN), 0.0),
                seed=seed,
            ),
            # Gaussian noise, variance 0.01 on the [0,1] scale. Keras 3's GaussianNoise
            # only accepts stddev in [0,1], so the noise is injected on the normalised
            # scale and mapped back to the 0-255 domain EfficientNet consumes. This is
            # identical to adding N(0, GAUSSIAN_NOISE_STDDEV^2) directly on 0-255.
            layers.Rescaling(1.0 / 255.0),
            layers.GaussianNoise(GAUSSIAN_NOISE_STDDEV_UNIT),
            layers.Rescaling(255.0),
        ],
        name="data_augmentation",
    )


def build_model(img_size, n_classes, dropout, seed):
    data_augmentation = build_augmentation(seed)

    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs), base_model


def make_callbacks(output_path):
    cbs = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1
        ),
    ]
    if output_path is not None:
        cbs.insert(
            0,
            keras.callbacks.ModelCheckpoint(
                output_path, monitor="val_accuracy", save_best_only=True, verbose=1
            ),
        )
    return cbs


def train_one_model(
    train_idx, val_idx, filepaths, labels, class_names, args, output_path
):
    n_classes = len(class_names)
    class_weight = compute_class_weights(labels[train_idx], n_classes)

    train_ds = make_dataset(
        filepaths[train_idx], labels[train_idx], n_classes,
        args.img_size, args.batch_size, shuffle=True, seed=args.seed,
    )
    val_ds = make_dataset(
        filepaths[val_idx], labels[val_idx], n_classes,
        args.img_size, args.batch_size, shuffle=False, seed=args.seed,
    )

    model, base_model = build_model(args.img_size, n_classes, args.dropout, args.seed)
    callbacks = make_callbacks(output_path)

    # ---- Phase 1: head warm-up (base frozen) --------------------------------------------
    print("\n=== Tahap 1: melatih classification head (base model dibekukan) ===")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs,
        class_weight=class_weight, callbacks=callbacks,
    )

    # ---- Phase 2: fine-tune the top layers of EfficientNet-B0 --------------------------
    if args.fine_tune_epochs > 0:
        print(f"\n=== Tahap 2: fine-tuning {args.fine_tune_layers} layer teratas EfficientNetB0 ===")
        base_model.trainable = True
        for layer in base_model.layers[:-args.fine_tune_layers]:
            layer.trainable = False
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.fine_tune_lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs,
            class_weight=class_weight, callbacks=callbacks,
        )
    return model


def evaluate_model(model, filepaths, labels, class_names, args):
    """Return accuracy / macro precision / macro recall / macro F1 on the given indices."""
    n_classes = len(class_names)
    ds = make_dataset(filepaths, labels, n_classes, args.img_size, args.batch_size, shuffle=False, seed=args.seed)
    probs = model.predict(ds, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    y_true = labels

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    accuracy = float(np.trace(cm) / max(cm.sum(), 1))
    precisions, recalls, f1s = [], [], []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
    return {
        "accuracy": accuracy,
        "precision_macro": float(np.mean(precisions)),
        "recall_macro": float(np.mean(recalls)),
        "f1_macro": float(np.mean(f1s)),
    }


def summarise_folds(fold_metrics):
    """Per-metric mean, std and 95% CI half-width (manuscript Eq. 9: 1.96*sigma/sqrt(k))."""
    keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    k_folds = max(len(fold_metrics), 1)
    summary = {}
    for k in keys:
        vals = np.array([m[k] for m in fold_metrics], dtype=np.float64)
        std = float(vals.std(ddof=0))
        summary[k] = {
            "mean": float(vals.mean()),
            "std": std,
            "ci95": float(1.96 * std / np.sqrt(k_folds)),
        }
    return summary


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------
def main():
    args = parse_args()
    keras.utils.set_random_seed(args.seed)

    data_dir = Path(args.data_dir)
    train_dir = Path(args.train_dir) if args.train_dir else data_dir / "train"
    val_dir = Path(args.val_dir) if args.val_dir else data_dir / "val"
    if args.test_dir:
        test_dir = Path(args.test_dir)
    else:
        candidate = data_dir / "test"
        test_dir = candidate if candidate.is_dir() else None

    print(f"Train dir : {train_dir}")
    print(f"Val dir   : {val_dir}")
    print(f"Test dir  : {test_dir}")

    filepaths, labels, class_names = collect_all_images(train_dir, val_dir, test_dir)
    n_classes = len(class_names)
    print(f"Total gambar: {len(filepaths)} | {n_classes} kelas: {class_names}")
    with open("class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    # ---------------- k-fold cross-validation mode --------------------------------------
    if args.cv_folds and args.cv_folds >= 2:
        print(f"\n### Stratified {args.cv_folds}-fold cross-validation (seed={args.seed}) ###")
        fold_metrics = []
        for fold, (tr_idx, te_idx) in enumerate(stratified_kfold(labels, args.cv_folds, args.seed), start=1):
            # carve a validation slice out of the training folds (~15/85 of the fold-train part)
            inner_val_frac = args.val_fraction / (1.0 - 1.0 / args.cv_folds)
            tr_sub, val_sub = stratified_split(labels[tr_idx], [1.0 - inner_val_frac, inner_val_frac], args.seed + fold)
            tr_final = tr_idx[tr_sub]
            val_final = tr_idx[val_sub]

            print(f"\n----- Fold {fold}/{args.cv_folds}: train={len(tr_final)} val={len(val_final)} test={len(te_idx)} -----")
            keras.backend.clear_session()
            keras.utils.set_random_seed(args.seed + fold)
            model = train_one_model(tr_final, val_final, filepaths, labels, class_names, args, output_path=None)
            metrics = evaluate_model(model, filepaths[te_idx], labels[te_idx], class_names, args)
            print(f"Fold {fold} test metrics: {json.dumps(metrics, indent=2)}")
            fold_metrics.append(metrics)

        summary = summarise_folds(fold_metrics)
        print("\n=== Cross-validation summary (mean +/- std | 95% CI) ===")
        for k, v in summary.items():
            print(f"  {k:<16}: {v['mean']:.4f} +/- {v['std']:.4f}  (95% CI +/- {v['ci95']:.4f})")
        with open("cv_results.json", "w", encoding="utf-8") as f:
            json.dump({"folds": fold_metrics, "summary": summary, "seed": args.seed}, f, indent=2)
        print("Detail per-fold tersimpan di: cv_results.json")
        return

    # ---------------- fixed train/val split, no pooling/repartition ---------------------
    if args.fixed_split:
        n_train = len(_list_split_dir(train_dir)[0])
        n_val = len(_list_split_dir(val_dir)[0])
        tr_idx = np.arange(0, n_train, dtype=np.int64)
        val_idx = np.arange(n_train, n_train + n_val, dtype=np.int64)
        print(f"\nFixed split (tanpa repartisi): train={len(tr_idx)} val={len(val_idx)} "
              f"-- val/ dipakai untuk checkpoint selection dan sebagai partisi yang dilaporkan")

        class_weight = compute_class_weights(labels[tr_idx], n_classes)
        print("Class weight (w_c = N / (K * n_c)):")
        for i, name in enumerate(class_names):
            print(f"  {name}: {class_weight[i]:.3f}")

        model = train_one_model(tr_idx, val_idx, filepaths, labels, class_names, args, output_path=args.output)
        metrics = evaluate_model(model, filepaths[val_idx], labels[val_idx], class_names, args)

        print("\n=== Validation-partition metrics (val/, guided checkpoint selection -- not an independent test set) ===")
        for k, v in metrics.items():
            print(f"  {k:<16}: {v:.4f}")
        with open("fixed_split_results.json", "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "seed": args.seed, "n_train": int(n_train), "n_val": int(n_val)}, f, indent=2)

        print(f"\nModel terbaik (val_accuracy tertinggi) tersimpan di: {args.output}")
        print("Nama kelas tersimpan di: class_names.json")
        return

    # ---------------- single stratified 70/15/15 split --------------------------------
    train_frac = 1.0 - args.val_fraction - args.test_fraction
    tr_idx, val_idx, te_idx = stratified_split(
        labels, [train_frac, args.val_fraction, args.test_fraction], args.seed
    )
    print(f"\nSplit stratified: train={len(tr_idx)} ({train_frac:.0%}) "
          f"val={len(val_idx)} ({args.val_fraction:.0%}) test={len(te_idx)} ({args.test_fraction:.0%})")

    class_weight = compute_class_weights(labels[tr_idx], n_classes)
    print("Class weight (w_c = N / (K * n_c)):")
    for i, name in enumerate(class_names):
        print(f"  {name}: {class_weight[i]:.3f}")

    model = train_one_model(tr_idx, val_idx, filepaths, labels, class_names, args, output_path=args.output)
    metrics = evaluate_model(model, filepaths[te_idx], labels[te_idx], class_names, args)

    print("\n=== Test-set metrics (hold-out 15%) ===")
    for k, v in metrics.items():
        print(f"  {k:<16}: {v:.4f}")
    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "seed": args.seed}, f, indent=2)

    print(f"\nModel terbaik (val_accuracy tertinggi) tersimpan di: {args.output}")
    print("Nama kelas tersimpan di: class_names.json")


if __name__ == "__main__":
    main()
