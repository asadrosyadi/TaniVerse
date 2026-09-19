"""
Shared data loading / splitting / tf.data pipeline for the hybrid model scripts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from . import MICROCLIMATE_TARGETS, SENSOR_FEATURES, PEST_CLASSES, HEALTH_CLASSES

STEP_HOURS = 4                       # 4 h grid, T=6 -> 24 h window


def engineer_sensor_window(raw_window: np.ndarray, start_hour: int,
                           step_hours: int = STEP_HOURS) -> np.ndarray:
    """Algorithm 1 step 5 -- turn the raw (T x 5) micro-climate window into the
    (T x 8) LSTM feature window: append cyclical hour encoding (sin/cos) and the
    temp x humidity interaction term. Applied identically at train and inference.
    """
    raw_window = np.asarray(raw_window, dtype=np.float32)
    T = raw_window.shape[0]
    hours = (start_hour + step_hours * np.arange(T)) % 24
    hour_sin = np.sin(2 * np.pi * hours / 24.0)
    hour_cos = np.cos(2 * np.pi * hours / 24.0)
    temp = raw_window[:, MICROCLIMATE_TARGETS.index("temperature")]
    rh = raw_window[:, MICROCLIMATE_TARGETS.index("humidity")]
    temp_humidity = temp * rh / 100.0
    extra = np.stack([hour_sin, hour_cos, temp_humidity], axis=1).astype(np.float32)
    return np.concatenate([raw_window, extra], axis=1)          # (T, 8)


def build_feature_windows(raw_windows: np.ndarray, start_hours: np.ndarray) -> np.ndarray:
    """Vectorised engineer_sensor_window over a whole (N x T x 5) array -> (N x T x 8)."""
    return np.stack([
        engineer_sensor_window(raw_windows[i], int(start_hours[i]))
        for i in range(len(raw_windows))
    ]).astype(np.float32)


def ci95(std: float, n_folds: int) -> float:
    """95% confidence interval half-width (manuscript Eq. 9): 1.96 * sigma / sqrt(k)."""
    return float(1.96 * std / np.sqrt(max(n_folds, 1)))


def load_paired(out_dir: str | Path, engineer: bool = True):
    """Load the artefacts written by make_paired_dataset.py.

    With engineer=True (default) the raw (N x T x 5) windows are expanded to the
    (N x T x 8) LSTM feature windows via engineer_sensor_window (using the
    per-pair window_start_hour column). Pass engineer=False to get raw windows.
    """
    out_dir = Path(out_dir)
    manifest = pd.read_csv(out_dir / "paired_manifest.csv")
    windows = np.load(out_dir / "paired_sensor_windows.npy").astype(np.float32)
    micro_targets = np.load(out_dir / "paired_micro_targets.npy").astype(np.float32)
    if not (len(manifest) == len(windows) == len(micro_targets)):
        raise ValueError("manifest / windows / targets length mismatch -- rebuild the dataset")
    if engineer and windows.shape[-1] == len(MICROCLIMATE_TARGETS):
        start_hours = manifest.get("window_start_hour", pd.Series(np.zeros(len(manifest)))).to_numpy()
        windows = build_feature_windows(windows, start_hours)
    return manifest, windows, micro_targets


def stratified_kfold(labels: np.ndarray, n_splits: int, seed: int):
    """Yield (train_idx, test_idx) for stratified k-fold CV."""
    rng = np.random.default_rng(seed)
    bins = [[] for _ in range(n_splits)]
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        for i, s in enumerate(cls_idx):
            bins[i % n_splits].append(s)
    bins = [np.array(sorted(b), dtype=np.int64) for b in bins]
    everything = set(range(len(labels)))
    for k in range(n_splits):
        test_idx = bins[k]
        train_idx = np.array(sorted(everything - set(test_idx.tolist())), dtype=np.int64)
        yield train_idx, test_idx


def stratified_holdout(labels: np.ndarray, val_frac: float, seed: int):
    """Return (train_idx, val_idx) stratified by label."""
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        cut = int(round((1.0 - val_frac) * len(cls_idx)))
        tr.extend(cls_idx[:cut].tolist())
        va.extend(cls_idx[cut:].tolist())
    return np.array(sorted(tr), dtype=np.int64), np.array(sorted(va), dtype=np.int64)


class MinMax:
    """Tiny per-feature min-max scaler (fit on training rows only -> no leakage)."""

    def __init__(self):
        self.min_ = None
        self.range_ = None

    def fit(self, x):                                   # x: (..., F)
        flat = x.reshape(-1, x.shape[-1])
        self.min_ = flat.min(axis=0)
        rng = flat.max(axis=0) - self.min_
        rng[rng == 0] = 1.0
        self.range_ = rng
        return self

    def transform(self, x):
        return ((x - self.min_) / self.range_).astype(np.float32)

    def inverse_transform(self, x):
        return (x * self.range_ + self.min_).astype(np.float32)


def _decode_image(path, img_size):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size))
    return tf.cast(img, tf.float32)                     # 0-255 (EfficientNet preprocess is built in)


def make_dataset(
    idx: np.ndarray,
    manifest: pd.DataFrame,
    windows_scaled: np.ndarray,
    micro_scaled: np.ndarray,
    img_size: int = 224,
    batch_size: int = 32,
    shuffle: bool = False,
    seed: int = 42,
    outputs: str = "hybrid",           # "hybrid" | "cnn" | "lstm"
):
    paths = np.asarray(manifest["image_path"].to_numpy()[idx], dtype=str)
    pest = tf.one_hot(manifest["pest_idx"].to_numpy()[idx], len(PEST_CLASSES))
    health = tf.one_hot(manifest["health_idx"].to_numpy()[idx], len(HEALTH_CLASSES))
    sensors = windows_scaled[idx]
    micro = micro_scaled[idx]

    ds = tf.data.Dataset.from_tensor_slices({
        "path": paths, "sensors": sensors, "pest": pest, "health": health, "micro": micro,
    })
    if shuffle:
        ds = ds.shuffle(min(len(idx), 4096), seed=seed, reshuffle_each_iteration=True)

    def _map(row):
        img = _decode_image(row["path"], img_size)
        if outputs == "cnn":
            return img, row["pest"]
        if outputs == "lstm":
            return row["sensors"], {"microclimate": row["micro"], "health": row["health"]}
        return (
            {"image": img, "sensors": row["sensors"]},
            {"pest": row["pest"], "health": row["health"], "microclimate": row["micro"]},
        )

    return ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def macro_prf(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int):
    """Return (accuracy, precision_macro, recall_macro, f1_macro)."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    acc = float(np.trace(cm) / max(cm.sum(), 1))
    ps, rs, fs = [], [], []
    for c in range(n_classes):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        ps.append(p); rs.append(r)
        fs.append(2 * p * r / (p + r) if (p + r) else 0.0)
    return acc, float(np.mean(ps)), float(np.mean(rs)), float(np.mean(fs))
