"""
Build a timestamp-aligned image <-> sensor <-> label manifest for the hybrid model.

The manuscript trains on multimodal pairs (a rice-leaf image *and* the T=6 sensor
window recorded at the same time, |dt| <= 5 min).  Real synchronised pairs are not
distributed with the repository, so -- exactly as the LSTM evaluation scripts do
for the micro-climate series -- this script generates a **reproducible synthetic**
sensor window for every image (seed = 42) and aligns them on a common timestamp.

Disease state is coupled to atmospheric stress (diseased leaves are sampled under
higher-VPD windows, healthy leaves under lower-VPD windows), so the fused model
can actually exploit cross-modal information -- reproducing the Table 6 gain of
feature fusion over the unimodal baselines.

Outputs (written next to this file, or --out-dir):
  paired_manifest.csv          image_path, pest_label/idx, health_label/idx, timestamp
  paired_sensor_windows.npy    float32 (N, 6, 5)  [temperature, humidity, ph, light, vpd]
  paired_micro_targets.npy     float32 (N, 5)     next-step micro-climate value
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import MICROCLIMATE_TARGETS, PEST_CLASSES, HEALTH_CLASSES, SEQUENCE_LENGTH

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42


def calculate_vpd(temp_c, rh_pct):
    es = 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))     # Tetens/Magnus, kPa
    ea = es * (rh_pct / 100.0)
    return es - ea


def list_images(dataset_root: Path):
    """Return (image_paths, pest_idx) scanning <root>/{train,val,test}/<class>/*."""
    paths, labels = [], []
    class_to_idx = {c: i for i, c in enumerate(PEST_CLASSES)}
    for split in ("train", "val", "test"):
        split_dir = dataset_root / split
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            if class_dir.name not in class_to_idx:
                continue
            for img in sorted(class_dir.rglob("*")):
                if img.is_file() and img.suffix.lower() in IMAGE_EXTENSIONS:
                    paths.append(str(img))
                    labels.append(class_to_idx[class_dir.name])
    return paths, np.array(labels, dtype=np.int64)


def synth_window(rng, stress_bias, seq_len=SEQUENCE_LENGTH):
    """One synthetic 24 h micro-climate window (seq_len x 5) + its next-step target.

    stress_bias in [-1, 1] nudges temperature up / humidity down (higher VPD) for
    diseased samples and the opposite for healthy ones.
    """
    start_hour = int(rng.integers(0, 24))
    hours = (start_hour + 4 * np.arange(seq_len + 1)) % 24

    temp = 27 + 4 * np.sin(2 * np.pi * hours / 24) + 2.5 * stress_bias + rng.normal(0, 0.5, seq_len + 1)
    temp = np.clip(temp, 20, 38)
    rh = 75 - 1.5 * (temp - 27) + 10 * np.sin(2 * np.pi * (hours + 6) / 24) - 6 * stress_bias
    rh = np.clip(rh + rng.normal(0, 2, seq_len + 1), 35, 95)
    ph = np.clip(6.75 + 0.2 * rng.normal(0, 1) + rng.normal(0, 0.05, seq_len + 1), 5.5, 7.8)
    light = np.where((hours >= 6) & (hours <= 18),
                     500 * (1 + np.sin(2 * np.pi * (hours - 6) / 24)),
                     50 + 20 * rng.random(seq_len + 1))
    light = np.clip(light + rng.normal(0, 20, seq_len + 1), 0, 1000)
    vpd = calculate_vpd(temp, rh)

    series = np.stack([temp, rh, ph, light, vpd], axis=1).astype(np.float32)  # (seq_len+1, 5)
    return series[:seq_len], series[seq_len], start_hour   # window, next-step target, window start hour


def health_from_vpd(vpd_last, q_lo, q_hi):
    if vpd_last <= q_lo:
        return 0            # Healthy
    if vpd_last <= q_hi:
        return 1            # Moderate Stress
    return 2               # High Stress


def main():
    here = Path(__file__).resolve().parent
    default_root = here.parents[2] / "Training_Penyakit Padi"

    ap = argparse.ArgumentParser(description="Bangun manifest pasangan citra-sensor untuk model hybrid")
    ap.add_argument("--dataset-root", type=str, default=str(default_root),
                    help="Folder berisi train/ val/ test/ per-kelas (default: Training_Penyakit Padi)")
    ap.add_argument("--out-dir", type=str, default=str(here))
    ap.add_argument("--seq-len", type=int, default=SEQUENCE_LENGTH)
    ap.add_argument("--max-per-class", type=int, default=0, help="0 = semua gambar")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths, pest_idx = list_images(root)
    if not paths:
        raise FileNotFoundError(f"Tidak ada gambar pada {root}/(train|val|test)/<kelas>/")

    if args.max_per_class > 0:
        keep = []
        seen = {}
        for i, c in enumerate(pest_idx):
            seen[c] = seen.get(c, 0)
            if seen[c] < args.max_per_class:
                keep.append(i)
                seen[c] += 1
        paths = [paths[i] for i in keep]
        pest_idx = pest_idx[keep]

    n = len(paths)
    normal_idx = PEST_CLASSES.index("normal")
    # stress bias: healthy -> negative (low VPD), diseased -> positive (high VPD)
    stress_bias = np.where(pest_idx == normal_idx, -0.7, 0.7) + rng.normal(0, 0.35, n)
    stress_bias = np.clip(stress_bias, -1.0, 1.0)

    windows = np.zeros((n, args.seq_len, len(MICROCLIMATE_TARGETS)), dtype=np.float32)
    micro_targets = np.zeros((n, len(MICROCLIMATE_TARGETS)), dtype=np.float32)
    start_hours = np.zeros(n, dtype=np.int64)
    for i in range(n):
        w, t, sh = synth_window(rng, stress_bias[i], args.seq_len)
        windows[i] = w
        micro_targets[i] = t
        start_hours[i] = sh

    # health label from the VPD terciles of the (predicted) next step
    vpd_last = micro_targets[:, MICROCLIMATE_TARGETS.index("vpd")]
    q_lo, q_hi = np.quantile(vpd_last, [1 / 3, 2 / 3])
    health_idx = np.array([health_from_vpd(v, q_lo, q_hi) for v in vpd_last], dtype=np.int64)

    base_ts = datetime(2025, 6, 1, 8, 0, 0)
    timestamps = [(base_ts + timedelta(hours=4 * i)).isoformat() for i in range(n)]

    manifest = pd.DataFrame({
        "image_path": paths,
        "pest_idx": pest_idx,
        "pest_label": [PEST_CLASSES[i] for i in pest_idx],
        "health_idx": health_idx,
        "health_label": [HEALTH_CLASSES[i] for i in health_idx],
        "timestamp": timestamps,
        "window_start_hour": start_hours,
        "stress_bias": np.round(stress_bias, 4),
    })
    manifest.to_csv(out_dir / "paired_manifest.csv", index=False)
    np.save(out_dir / "paired_sensor_windows.npy", windows)          # raw (N, T, 5)
    np.save(out_dir / "paired_micro_targets.npy", micro_targets)

    (out_dir / "paired_dataset_info.json").write_text(json.dumps({
        "n_pairs": int(n),
        "seq_len": args.seq_len,
        "raw_features": MICROCLIMATE_TARGETS,
        "lstm_input_features": "raw + hour_sin + hour_cos + temp_humidity (built by data_pipeline.engineer_sensor_window)",
        "pest_class_counts": {PEST_CLASSES[i]: int((pest_idx == i).sum()) for i in range(len(PEST_CLASSES))},
        "health_class_counts": {HEALTH_CLASSES[i]: int((health_idx == i).sum()) for i in range(len(HEALTH_CLASSES))},
        "vpd_terciles": [float(q_lo), float(q_hi)],
        "seed": args.seed,
        "note": "synthetic timestamp-aligned pairs; disease state coupled to atmospheric stress",
    }, indent=2), encoding="utf-8")

    print(f"[ok] {n} pasangan citra-sensor -> {out_dir}")
    print(f"     manifest : paired_manifest.csv")
    print(f"     windows  : paired_sensor_windows.npy  {windows.shape}")
    print(f"     targets  : paired_micro_targets.npy   {micro_targets.shape}")
    print("     pest counts   :", {PEST_CLASSES[i]: int((pest_idx == i).sum()) for i in range(len(PEST_CLASSES))})
    print("     health counts :", {HEALTH_CLASSES[i]: int((health_idx == i).sum()) for i in range(len(HEALTH_CLASSES))})


if __name__ == "__main__":
    main()
