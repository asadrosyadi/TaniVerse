"""
Train the hybrid CNN-LSTM feature-fusion model.

Config (manuscript Table 3 / sec. 3.5):
  optimiser Adam (lr 1e-3), batch 32, dropout 0.2, seed 42,
  stratified 5-fold cross-validation (mean +/- std reported),
  fusion weights read from fusion_weights.json (default w1=0.6, w2=0.4).

Artefacts written to --out-dir:
  hybrid_cnn_lstm_model.h5     final model (trained on the full 85% train split)
  hybrid_scaler_stats.json     min-max stats for the sensor window + micro targets
  hybrid_cv_results.json       per-fold + summary metrics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from . import DEFAULT_W1, DEFAULT_W2, PEST_CLASSES, HEALTH_CLASSES, MICROCLIMATE_TARGETS
from .data_pipeline import (
    load_paired, stratified_kfold, stratified_holdout, MinMax, make_dataset, macro_prf, ci95,
)
from .fusion_model import build_hybrid_model


def parse_args():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Latih model hybrid CNN-LSTM (feature fusion)")
    ap.add_argument("--data-dir", type=str, default=str(here))
    ap.add_argument("--out-dir", type=str, default=str(here))
    ap.add_argument("--weights-json", type=str, default=str(here / "fusion_weights.json"))
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--fusion", type=str, default="adaptive", choices=["adaptive", "concat"])
    ap.add_argument("--trainable-backbone", action="store_true",
                    help="buka EfficientNet-B0 (fine-tune penuh); default: backbone beku")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def load_w1(weights_json: str) -> float:
    p = Path(weights_json)
    if p.is_file():
        try:
            w1 = float(json.loads(p.read_text())["w1"])
            print(f"[fusion] w1={w1} (dari {p.name})")
            return w1
        except Exception as e:
            print(f"[fusion] gagal baca {p.name} ({e}); pakai default w1={DEFAULT_W1}")
    else:
        print(f"[fusion] {p.name} tidak ada; pakai default w1={DEFAULT_W1} (jalankan tune_fusion_weights.py dulu)")
    return DEFAULT_W1


def callbacks():
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ]


def main():
    args = parse_args()
    keras.utils.set_random_seed(args.seed)

    manifest, windows, micro = load_paired(args.data_dir)
    w1 = load_w1(args.weights_json) if args.fusion == "adaptive" else DEFAULT_W1
    pest_labels = manifest["pest_idx"].to_numpy()
    vpd_col = MICROCLIMATE_TARGETS.index("vpd")

    # ---------------- 5-fold cross-validation ----------------
    print(f"\n### Hybrid ({args.fusion}) -- stratified {args.cv_folds}-fold CV ###")
    rows = []
    for fold, (tr_idx, te_idx) in enumerate(stratified_kfold(pest_labels, args.cv_folds, args.seed), start=1):
        tr_sub, val_sub = stratified_holdout(pest_labels[tr_idx], val_frac=0.2, seed=args.seed + fold)
        tr_final, val_final = tr_idx[tr_sub], tr_idx[val_sub]

        sx = MinMax().fit(windows[tr_final])
        sy = MinMax().fit(micro[tr_final])
        w_s, m_s = sx.transform(windows), sy.transform(micro)

        keras.backend.clear_session()
        keras.utils.set_random_seed(args.seed + fold)
        model = build_hybrid_model(
            img_size=args.img_size, w1=w1, fusion=args.fusion,
            n_sensor_features=windows.shape[-1], n_micro=micro.shape[-1],
            trainable_backbone=args.trainable_backbone,
        )
        tr_ds = make_dataset(tr_final, manifest, w_s, m_s, args.img_size, args.batch_size, shuffle=True, seed=args.seed)
        val_ds = make_dataset(val_final, manifest, w_s, m_s, args.img_size, args.batch_size)
        te_ds = make_dataset(te_idx, manifest, w_s, m_s, args.img_size, args.batch_size)
        model.fit(tr_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks(), verbose=2)

        preds = model.predict(te_ds, verbose=0)
        acc, p, r, f1 = macro_prf(pest_labels[te_idx], np.argmax(preds["pest"], axis=1), len(PEST_CLASSES))
        h_acc, _, _, h_f1 = macro_prf(
            manifest["health_idx"].to_numpy()[te_idx], np.argmax(preds["health"], axis=1), len(HEALTH_CLASSES)
        )
        micro_pred = sy.inverse_transform(preds["microclimate"])
        micro_true = micro[te_idx]
        vpd_rmse = float(np.sqrt(np.mean((micro_pred[:, vpd_col] - micro_true[:, vpd_col]) ** 2)))

        row = {"fold": fold, "pest_accuracy": acc, "pest_precision": p, "pest_recall": r,
               "pest_f1": f1, "health_accuracy": h_acc, "health_f1": h_f1, "vpd_rmse": vpd_rmse}
        rows.append(row)
        print(f"  fold {fold}: acc={acc:.4f} F1={f1:.4f} health_acc={h_acc:.4f} VPD_RMSE={vpd_rmse:.4f}")

    keys = ["pest_accuracy", "pest_precision", "pest_recall", "pest_f1",
            "health_accuracy", "health_f1", "vpd_rmse"]
    n_folds = len(rows)
    summary = {}
    for k in keys:
        vals = [r[k] for r in rows]
        std = float(np.std(vals, ddof=0))
        summary[k] = {"mean": float(np.mean(vals)), "std": std, "ci95": ci95(std, n_folds)}
    print("\n=== Hybrid CV summary (mean +/- std | 95% CI) ===")
    for k in keys:
        s = summary[k]
        print(f"  {k:<16}: {s['mean']:.4f} +/- {s['std']:.4f}  (95% CI +/- {s['ci95']:.4f})")

    Path(args.out_dir, "hybrid_cv_results.json").write_text(
        json.dumps({"fusion": args.fusion, "w1": w1, "w2": round(1 - w1, 3),
                    "folds": rows, "summary": summary, "seed": args.seed}, indent=2),
        encoding="utf-8",
    )

    # ---------------- final model on the full 85/15 split ----------------
    print("\n### Training final model (85% train / 15% val) ###")
    tr_idx, val_idx = stratified_holdout(pest_labels, val_frac=0.15, seed=args.seed)
    sx = MinMax().fit(windows[tr_idx])
    sy = MinMax().fit(micro[tr_idx])
    w_s, m_s = sx.transform(windows), sy.transform(micro)

    keras.backend.clear_session()
    keras.utils.set_random_seed(args.seed)
    model = build_hybrid_model(
        img_size=args.img_size, w1=w1, fusion=args.fusion,
        n_sensor_features=windows.shape[-1], n_micro=micro.shape[-1],
        trainable_backbone=args.trainable_backbone,
    )
    tr_ds = make_dataset(tr_idx, manifest, w_s, m_s, args.img_size, args.batch_size, shuffle=True, seed=args.seed)
    val_ds = make_dataset(val_idx, manifest, w_s, m_s, args.img_size, args.batch_size)
    model.fit(tr_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks(), verbose=2)

    model_path = Path(args.out_dir, "hybrid_cnn_lstm_model.h5")
    model.save(model_path)
    Path(args.out_dir, "hybrid_scaler_stats.json").write_text(json.dumps({
        "features": MICROCLIMATE_TARGETS,
        "sensor_min": sx.min_.tolist(), "sensor_range": sx.range_.tolist(),
        "micro_min": sy.min_.tolist(), "micro_range": sy.range_.tolist(),
        "w1": w1, "w2": round(1 - w1, 3), "fusion": args.fusion,
    }, indent=2), encoding="utf-8")
    print(f"\n[ok] model  -> {model_path}")
    print(f"[ok] scaler -> {Path(args.out_dir, 'hybrid_scaler_stats.json')}")


if __name__ == "__main__":
    main()
