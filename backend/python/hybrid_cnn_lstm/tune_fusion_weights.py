"""
Grid search for the adaptive feature-fusion weights  (manuscript Eq. 1, sec. 3.3).

    R_total = w1 * R_visual + w2 * R_environment ,   w1 + w2 = 1

w1 is swept over {0.0, 0.1, ..., 1.0}.  For each value a hybrid model is trained
under stratified 5-fold cross-validation; the configuration that maximises the
mean validation pest-classification F1 (ties broken by the lower mean validation
loss) is selected.  The manuscript reports the optimum at w1 = 0.6, w2 = 0.4.

Writes fusion_weights.json  ->  consumed by train_hybrid.py / evaluate_hybrid.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from . import DEFAULT_W1, PEST_CLASSES, MICROCLIMATE_TARGETS
from .data_pipeline import (
    load_paired, stratified_kfold, stratified_holdout, MinMax, make_dataset, macro_prf,
)
from .fusion_model import build_hybrid_model


def parse_args():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Grid-search bobot fusi w1/w2 (Eq. 1)")
    ap.add_argument("--data-dir", type=str, default=str(here))
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=12, help="epoch per fold (grid search cepat)")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--grid", type=str, default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=str(here / "fusion_weights.json"))
    return ap.parse_args()


def evaluate_w1(w1, manifest, windows, micro, args):
    pest_labels = manifest["pest_idx"].to_numpy()
    fold_f1, fold_loss = [], []

    for fold, (tr_idx, te_idx) in enumerate(stratified_kfold(pest_labels, args.cv_folds, args.seed), start=1):
        tr_sub, val_sub = stratified_holdout(pest_labels[tr_idx], val_frac=0.2, seed=args.seed + fold)
        tr_final, val_final = tr_idx[tr_sub], tr_idx[val_sub]

        sx = MinMax().fit(windows[tr_final])
        sy = MinMax().fit(micro[tr_final])
        w_s, m_s = sx.transform(windows), sy.transform(micro)

        keras.backend.clear_session()
        keras.utils.set_random_seed(args.seed + fold)
        model = build_hybrid_model(
            img_size=args.img_size, w1=w1, fusion="adaptive",
            n_sensor_features=windows.shape[-1], n_micro=micro.shape[-1],
        )
        tr_ds = make_dataset(tr_final, manifest, w_s, m_s, args.img_size, args.batch_size, shuffle=True, seed=args.seed)
        val_ds = make_dataset(val_final, manifest, w_s, m_s, args.img_size, args.batch_size)
        model.fit(tr_ds, validation_data=val_ds, epochs=args.epochs, verbose=0)

        te_ds = make_dataset(te_idx, manifest, w_s, m_s, args.img_size, args.batch_size)
        ev = model.evaluate(te_ds, verbose=0, return_dict=True)
        preds = model.predict(te_ds, verbose=0)
        y_pred = np.argmax(preds["pest"], axis=1)
        _, _, _, f1 = macro_prf(pest_labels[te_idx], y_pred, len(PEST_CLASSES))
        fold_f1.append(f1)
        fold_loss.append(float(ev.get("loss", np.nan)))
        print(f"    w1={w1:.1f} fold {fold}/{args.cv_folds}: F1={f1:.4f} loss={fold_loss[-1]:.4f}")

    return {
        "w1": round(float(w1), 3), "w2": round(1.0 - float(w1), 3),
        "f1_mean": float(np.mean(fold_f1)), "f1_std": float(np.std(fold_f1, ddof=0)),
        "loss_mean": float(np.nanmean(fold_loss)),
        "folds_f1": [float(x) for x in fold_f1],
    }


def main():
    args = parse_args()
    keras.utils.set_random_seed(args.seed)
    manifest, windows, micro = load_paired(args.data_dir)
    grid = [float(x) for x in args.grid.split(",") if x.strip() != ""]
    print(f"Grid search w1 in {grid}  ({args.cv_folds}-fold CV, {args.epochs} ep/fold)")

    results = []
    for w1 in grid:
        print(f"\n=== w1 = {w1:.1f} (w2 = {1 - w1:.1f}) ===")
        results.append(evaluate_w1(w1, manifest, windows, micro, args))

    # rank: highest mean F1, tie-break lowest mean loss
    results.sort(key=lambda r: (-r["f1_mean"], r["loss_mean"]))
    best = results[0]

    payload = {
        "w1": best["w1"],
        "w2": best["w2"],
        "selection": "max mean 5-fold validation F1 (tie-break: min mean loss)",
        "manuscript_optimum": {"w1": DEFAULT_W1, "w2": round(1 - DEFAULT_W1, 3)},
        "grid_results": results,
        "seed": args.seed,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== Grid-search ranking (best first) ===")
    for r in results:
        print(f"  w1={r['w1']:.1f} w2={r['w2']:.1f}  F1={r['f1_mean']:.4f} +/- {r['f1_std']:.4f}  loss={r['loss_mean']:.4f}")
    print(f"\nSelected: w1={best['w1']}, w2={best['w2']}  ->  {args.out}")
    print(f"(manuscript reports the optimum at w1={DEFAULT_W1}, w2={round(1 - DEFAULT_W1, 3)})")


if __name__ == "__main__":
    main()
