"""
Reproduce manuscript **Table 6** -- Performance Comparison of Baseline and Hybrid
CNN-LSTM Models.

Rows:
  CNN only ......................... EfficientNet-B0 -> pest soft-max
  LSTM only ....................... stacked LSTM -> micro-climate + health
  CNN + LSTM (without fusion) ...... concatenation of R_visual, R_environment
  Proposed CNN-LSTM (Feature Fusion) R_total = w1 R_visual + w2 R_environment  (Eq. 1)

Columns: Accuracy (%), F1-Score, RMSE (VPD, kPa), Latency (ms/frame).
Every number is the mean +/- std over a stratified 5-fold CV (seed 42).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from . import DEFAULT_W1, PEST_CLASSES, HEALTH_CLASSES, MICROCLIMATE_TARGETS
from .data_pipeline import (
    load_paired, stratified_kfold, stratified_holdout, MinMax, make_dataset, macro_prf, ci95,
)
from .fusion_model import (
    build_hybrid_model, build_cnn_only_model, build_lstm_only_model,
)


def parse_args():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Reproduksi Table 6 (baseline vs hybrid)")
    ap.add_argument("--data-dir", type=str, default=str(here))
    ap.add_argument("--out-dir", type=str, default=str(here))
    ap.add_argument("--weights-json", type=str, default=str(here / "fusion_weights.json"))
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--latency-iters", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def _w1(weights_json):
    p = Path(weights_json)
    if p.is_file():
        try:
            return float(json.loads(p.read_text())["w1"])
        except Exception:
            pass
    return DEFAULT_W1


def _cb():
    return [keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=0)]


def measure_latency(model, sample_inputs, iters):
    """Single-sample inference latency (ms/frame), median of `iters` timed calls."""
    _ = model.predict(sample_inputs, verbose=0)          # warm-up
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model.predict(sample_inputs, verbose=0)
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts))


def one_image(manifest, idx, img_size):
    raw = tf.io.read_file(manifest["image_path"].iloc[int(idx)])
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size))
    return tf.cast(img, tf.float32)[None, ...]


def run_model(kind, manifest, windows, micro, args, w1):
    """Return dict of per-fold {accuracy, f1, vpd_rmse, latency_ms} for one Table-6 row."""
    pest_labels = manifest["pest_idx"].to_numpy()
    vpd_col = MICROCLIMATE_TARGETS.index("vpd")
    folds = []

    for fold, (tr_idx, te_idx) in enumerate(stratified_kfold(pest_labels, args.cv_folds, args.seed), start=1):
        tr_sub, val_sub = stratified_holdout(pest_labels[tr_idx], val_frac=0.2, seed=args.seed + fold)
        tr_final, val_final = tr_idx[tr_sub], tr_idx[val_sub]
        sx = MinMax().fit(windows[tr_final]); sy = MinMax().fit(micro[tr_final])
        w_s, m_s = sx.transform(windows), sy.transform(micro)

        keras.backend.clear_session()
        keras.utils.set_random_seed(args.seed + fold)

        acc = f1 = np.nan
        vpd_rmse = np.nan

        if kind == "cnn_only":
            model = build_cnn_only_model(img_size=args.img_size)
            tr = make_dataset(tr_final, manifest, w_s, m_s, args.img_size, args.batch_size, True, args.seed, outputs="cnn")
            va = make_dataset(val_final, manifest, w_s, m_s, args.img_size, args.batch_size, outputs="cnn")
            te = make_dataset(te_idx, manifest, w_s, m_s, args.img_size, args.batch_size, outputs="cnn")
            model.fit(tr, validation_data=va, epochs=args.epochs, callbacks=_cb(), verbose=0)
            y_pred = np.argmax(model.predict(te, verbose=0), axis=1)
            acc, _, _, f1 = macro_prf(pest_labels[te_idx], y_pred, len(PEST_CLASSES))
            lat = measure_latency(model, one_image(manifest, te_idx[0], args.img_size), args.latency_iters)

        elif kind == "lstm_only":
            model = build_lstm_only_model(n_sensor_features=windows.shape[-1], n_micro=micro.shape[-1])
            tr = make_dataset(tr_final, manifest, w_s, m_s, args.img_size, args.batch_size, True, args.seed, outputs="lstm")
            va = make_dataset(val_final, manifest, w_s, m_s, args.img_size, args.batch_size, outputs="lstm")
            te = make_dataset(te_idx, manifest, w_s, m_s, args.img_size, args.batch_size, outputs="lstm")
            model.fit(tr, validation_data=va, epochs=args.epochs, callbacks=_cb(), verbose=0)
            preds = model.predict(te, verbose=0)
            micro_pred = sy.inverse_transform(preds["microclimate"])
            vpd_rmse = float(np.sqrt(np.mean((micro_pred[:, vpd_col] - micro[te_idx][:, vpd_col]) ** 2)))
            lat = measure_latency(model, w_s[te_idx[:1]], args.latency_iters)

        else:   # "concat"  or  "fusion"
            fusion = "concat" if kind == "concat" else "adaptive"
            model = build_hybrid_model(
                img_size=args.img_size, w1=w1, fusion=fusion,
                n_sensor_features=windows.shape[-1], n_micro=micro.shape[-1],
            )
            tr = make_dataset(tr_final, manifest, w_s, m_s, args.img_size, args.batch_size, True, args.seed)
            va = make_dataset(val_final, manifest, w_s, m_s, args.img_size, args.batch_size)
            te = make_dataset(te_idx, manifest, w_s, m_s, args.img_size, args.batch_size)
            model.fit(tr, validation_data=va, epochs=args.epochs, callbacks=_cb(), verbose=0)
            preds = model.predict(te, verbose=0)
            acc, _, _, f1 = macro_prf(pest_labels[te_idx], np.argmax(preds["pest"], axis=1), len(PEST_CLASSES))
            micro_pred = sy.inverse_transform(preds["microclimate"])
            vpd_rmse = float(np.sqrt(np.mean((micro_pred[:, vpd_col] - micro[te_idx][:, vpd_col]) ** 2)))
            sample = {"image": one_image(manifest, te_idx[0], args.img_size), "sensors": w_s[te_idx[:1]]}
            lat = measure_latency(model, sample, args.latency_iters)

        folds.append({"accuracy": acc, "f1": f1, "vpd_rmse": vpd_rmse, "latency_ms": lat})
        print(f"    [{kind}] fold {fold}: acc={acc if acc == acc else float('nan'):.4f} "
              f"f1={f1 if f1 == f1 else float('nan'):.4f} "
              f"vpd_rmse={vpd_rmse if vpd_rmse == vpd_rmse else float('nan'):.4f} lat={lat:.1f}ms")
    return folds


def summarise(folds):
    out = {}
    for k in ("accuracy", "f1", "vpd_rmse", "latency_ms"):
        vals = np.array([f[k] for f in folds], dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            out[k] = None
        else:
            std = float(vals.std(ddof=0))
            out[k] = {"mean": float(vals.mean()), "std": std, "ci95": ci95(std, len(vals))}
    return out


def fmt(cell, scale=1.0, nd=2):
    if cell is None:
        return "--"
    return f"{cell['mean'] * scale:.{nd}f} +/- {cell['std'] * scale:.{nd}f}"


def main():
    args = parse_args()
    keras.utils.set_random_seed(args.seed)
    manifest, windows, micro = load_paired(args.data_dir)
    w1 = _w1(args.weights_json)
    print(f"[fusion] w1={w1}, w2={round(1 - w1, 3)}")

    plan = [
        ("CNN only", "cnn_only"),
        ("LSTM only", "lstm_only"),
        ("CNN + LSTM (without fusion)", "concat"),
        ("Proposed CNN-LSTM (Feature Fusion)", "fusion"),
    ]
    table = []
    for label, kind in plan:
        print(f"\n### {label} ###")
        folds = run_model(kind, manifest, windows, micro, args, w1)
        table.append({"model": label, "kind": kind, "folds": folds, "summary": summarise(folds)})

    # ---- render Table 6 ----
    hdr = "| Model | Accuracy (%) | F1-Score | RMSE (VPD) | Latency (ms/frame) |"
    sep = "| --- | --- | --- | --- | --- |"
    lines = [hdr, sep]
    for row in table:
        s = row["summary"]
        acc = fmt(s["accuracy"], scale=100.0, nd=1) if s["accuracy"] else "--"
        f1 = fmt(s["f1"], nd=2) if s["f1"] else "--"
        vpd = fmt(s["vpd_rmse"], nd=3) if s["vpd_rmse"] else "--"
        lat = fmt(s["latency_ms"], nd=1) if s["latency_ms"] else "--"
        lines.append(f"| {row['model']} | {acc} | {f1} | {vpd} | {lat} |")

    md = "\n".join(lines) + "\n"
    Path(args.out_dir, "table6_hybrid_comparison.md").write_text(md, encoding="utf-8")
    Path(args.out_dir, "table6_hybrid_comparison.json").write_text(
        json.dumps({"w1": w1, "w2": round(1 - w1, 3), "rows": table, "seed": args.seed}, indent=2),
        encoding="utf-8",
    )
    print("\n=== Table 6. Performance Comparison of Baseline and Hybrid CNN-LSTM Models ===")
    print(md)
    print("Disimpan: table6_hybrid_comparison.{md,json}")


if __name__ == "__main__":
    main()
