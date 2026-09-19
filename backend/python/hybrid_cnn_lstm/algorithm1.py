"""
Algorithm 1 -- Hybrid CNN-LSTM Edge-AIoT Framework for Multimodal Pest Detection
and Microclimate Prediction  (manuscript Table 2).

Runnable, step-numbered implementation of the 17-step per-timestep loop:

   1  for each timestep t in T
   2  capture image F_t + sensor data S_t (camera / IoT modules)
   3  if S_t has missing values -> linear (spline) interpolation
   4  image preprocessing: resize 224x224, scale [0,1], (augment: train only)
   5  sensor preprocessing: Min-Max scaling, cyclical time encoding, temp x humidity
   6  R_visual      = CNN(F_t)
   7  R_environment = LSTM(S_t)
   8  synchronise F_t / S_t on timestamp, tolerance |dt| <= 5 min
   9  R_total = w1 * R_visual + w2 * R_environment            (Eq. 1)
  10  P = softmax(R_total)                                    (Eq. 3)
  11  pest_class, pest_conf   = argmax(P_pest)
  12  if pest_conf   >= tau_p       -> pest_label = pest_class      else "No pest detected"
  13  health_class, health_conf = argmax(P_health)
  14  if health_conf >= tau_health  -> status = health_class        else "Monitor"
  15  actuation:
        pest present & class == Hispa -> ultrasonic repeller + UV lamp
        pest present (other)          -> selective insecticide spraying
        status == High Stress         -> irrigation notification
        otherwise                     -> continue monitoring
  16  transmit -> Laravel REST API + MQTT broker
  17  end for

Inputs: the paired manifest from make_paired_dataset.py, OR --image-dir + --sensor-csv.
Transmit is off by default (--dry-run); pass --laravel-url / --mqtt-host to enable.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from . import (
    DEFAULT_W1, TAU_PEST, TAU_HEALTH,
    PEST_CLASSES, HEALTH_CLASSES, MICROCLIMATE_TARGETS, SENSOR_FEATURES,
)
from .data_pipeline import engineer_sensor_window
from .fusion_model import AdaptiveFeatureFusion, build_hybrid_model

SYNC_TOLERANCE_S = 5 * 60          # |dt| <= 5 minutes  (step 8)


# ---------------------------------------------------------------------------------------
# Steps 3-5 : pre-processing
# ---------------------------------------------------------------------------------------
def interpolate_missing(window: np.ndarray) -> np.ndarray:
    """Step 3: linear interpolation of NaNs along the time axis, per feature."""
    w = window.astype(np.float32).copy()
    for j in range(w.shape[1]):
        col = w[:, j]
        nan = np.isnan(col)
        if nan.any() and (~nan).sum() >= 1:
            idx = np.arange(len(col))
            col[nan] = np.interp(idx[nan], idx[~nan], col[~nan])
            w[:, j] = col
    return np.nan_to_num(w)


def preprocess_image(path: str, img_size: int = 224) -> np.ndarray:
    """Step 4: resize to 224x224 (EfficientNet preprocess_input is inside the model)."""
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size))
    return tf.cast(img, tf.float32).numpy()[None, ...]


def preprocess_sensors(raw_window: np.ndarray, start_hour: int, stats: dict) -> np.ndarray:
    """Step 5: feature engineering + Min-Max scaling.

    raw_window : (T, 5) micro-climate window [temperature, humidity, ph, light, vpd].
    Returns the scaled (1, T, 8) LSTM feature tensor -- the 5 raw channels plus the
    cyclical hour encoding (sin/cos) and the temp x humidity interaction term
    (data_pipeline.engineer_sensor_window), consumed directly by the LSTM branch.
    """
    raw_window = interpolate_missing(raw_window)                       # step 3
    feat = engineer_sensor_window(raw_window, int(start_hour))         # step 5: (T, 8)

    mn = np.asarray(stats["sensor_min"], dtype=np.float32)
    rng = np.asarray(stats["sensor_range"], dtype=np.float32)
    if mn.shape[0] != feat.shape[1]:                                  # older 5-feature stats
        mn = np.concatenate([mn, np.zeros(feat.shape[1] - mn.shape[0], np.float32)])
        rng = np.concatenate([rng, np.ones(feat.shape[1] - rng.shape[0], np.float32)])
    scaled = (feat - mn) / rng
    return scaled.astype(np.float32)[None, ...]


# ---------------------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------------------
def load_or_build_model(model_path: str, w1: float, n_features: int):
    p = Path(model_path)
    if p.is_file():
        model = keras.models.load_model(
            p, custom_objects={"AdaptiveFeatureFusion": AdaptiveFeatureFusion}
        )
        print(f"[model] loaded {p.name}")
    else:
        print(f"[model] {p.name} not found -> building an UNTRAINED hybrid model "
              f"(flow demo only; run train_hybrid.py for real weights)")
        model = build_hybrid_model(
            w1=w1, n_sensor_features=n_features, n_micro=len(MICROCLIMATE_TARGETS)
        )
    try:
        fusion_layer = model.get_layer("feature_fusion")
        r_total_model = keras.Model(model.inputs, fusion_layer.output, name="R_total")
    except (ValueError, KeyError):
        r_total_model = None            # e.g. the concat baseline has no fusion layer
    return model, r_total_model


# ---------------------------------------------------------------------------------------
# Step 15 : actuation decision
# ---------------------------------------------------------------------------------------
def decide_actuation(pest_label: str, pest_class: str, status: str) -> dict:
    if pest_label != "No pest detected":
        if pest_class.lower() == "hispa":
            return {"action": "activate_ultrasonic_repeller_and_uv_lamp",
                    "actuators": ["ultrasonic_repeller", "uv_lamp"],
                    "reason": f"pest '{pest_class}' detected"}
        return {"action": "recommend_selective_insecticide_spraying",
                "actuators": [], "reason": f"pest '{pest_class}' detected"}
    if status == "High Stress":
        return {"action": "irrigation_notification", "actuators": ["irrigation_valve"],
                "reason": "plant health = High Stress"}
    return {"action": "continue_monitoring", "actuators": [], "reason": "no pest, health OK"}


# ---------------------------------------------------------------------------------------
# Step 16 : transmit
# ---------------------------------------------------------------------------------------
def transmit(payload: dict, laravel_url: str | None, mqtt_host: str | None,
             mqtt_topic: str, dry_run: bool):
    if dry_run or (not laravel_url and not mqtt_host):
        print(f"   [transmit:dry-run] {json.dumps(payload, default=str)}")
        return

    if laravel_url:
        try:
            import requests
            r = requests.post(laravel_url, json=payload, timeout=5)
            print(f"   [transmit:laravel] {laravel_url} -> HTTP {r.status_code}")
        except Exception as e:
            print(f"   [transmit:laravel] FAILED ({e})")

    if mqtt_host:
        try:
            import paho.mqtt.publish as publish
            publish.single(mqtt_topic, json.dumps(payload, default=str), hostname=mqtt_host)
            print(f"   [transmit:mqtt] {mqtt_host}/{mqtt_topic} published")
        except Exception as e:
            print(f"   [transmit:mqtt] FAILED ({e})")


# ---------------------------------------------------------------------------------------
# Main loop  (steps 1-17)
# ---------------------------------------------------------------------------------------
def parse_args():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Jalankan Algorithm 1 (hybrid CNN-LSTM edge-AIoT)")
    ap.add_argument("--data-dir", type=str, default=str(here),
                    help="folder berisi paired_manifest.csv + *.npy + hybrid_scaler_stats.json")
    ap.add_argument("--model", type=str, default=str(here / "hybrid_cnn_lstm_model.h5"))
    ap.add_argument("--weights-json", type=str, default=str(here / "fusion_weights.json"))
    ap.add_argument("--iot-id", type=str, default="jTZids5M")
    ap.add_argument("--iot-token", type=str, default="",
                    help="shared secret checked by POST /api/inference (wajib jika --laravel-url dipakai)")
    ap.add_argument("--limit", type=int, default=10, help="jumlah timestep yang diproses (0 = semua)")
    ap.add_argument("--tau-pest", type=float, default=TAU_PEST)
    ap.add_argument("--tau-health", type=float, default=TAU_HEALTH)
    ap.add_argument("--laravel-url", type=str, default=None,
                    help="mis. https://petaniasik.my.id/api/inference (step 16)")
    ap.add_argument("--mqtt-host", type=str, default=None)
    ap.add_argument("--mqtt-topic", type=str, default=None)
    ap.add_argument("--dry-run", action="store_true", help="jangan kirim ke jaringan (default aktif bila URL kosong)")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    keras.utils.set_random_seed(args.seed)
    data_dir = Path(args.data_dir)

    manifest = pd.read_csv(data_dir / "paired_manifest.csv")
    windows = np.load(data_dir / "paired_sensor_windows.npy").astype(np.float32)

    n_feat = len(SENSOR_FEATURES)                       # engineered LSTM input width (8)
    stats_path = data_dir / "hybrid_scaler_stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text())
    else:                                   # fall back to identity scaling
        stats = {"sensor_min": [0.0] * n_feat, "sensor_range": [1.0] * n_feat}
        print("[warn] hybrid_scaler_stats.json missing -> using identity sensor scaling")

    w1 = DEFAULT_W1
    if Path(args.weights_json).is_file():
        try:
            w1 = float(json.loads(Path(args.weights_json).read_text())["w1"])
        except Exception:
            pass
    w2 = round(1.0 - w1, 3)

    model, r_total_model = load_or_build_model(args.model, w1, n_feat)
    vpd_col = MICROCLIMATE_TARGETS.index("vpd")
    mqtt_topic = args.mqtt_topic or f"taniverse/{args.iot_id}/inference"

    n = len(manifest) if args.limit in (0, None) else min(args.limit, len(manifest))
    print(f"\n=== Algorithm 1: processing {n} timesteps (w1={w1}, w2={w2}, "
          f"tau_p={args.tau_pest}, tau_health={args.tau_health}) ===\n")

    log = []
    for t in range(n):                                                        # step 1
        row = manifest.iloc[t]
        img_path = row["image_path"]
        ts = datetime.fromisoformat(row["timestamp"])
        start_hour = int(row["window_start_hour"]) if "window_start_hour" in manifest.columns else ts.hour
        S_t = windows[t].copy()                                               # step 2 (raw T x 5)

        # step 3: inject a missing block on ~every 7th sample to exercise interpolation
        if t % 7 == 3:
            S_t[2, :] = np.nan

        img = preprocess_image(img_path)                                      # step 4
        sensors = preprocess_sensors(S_t, start_hour, stats)                  # steps 3+5 -> (1, T, 8)

        # steps 6-9 : branch encoders + timestamp sync + adaptive fusion
        r_total = None
        if r_total_model is not None:
            r_total = r_total_model.predict({"image": img, "sensors": sensors}, verbose=0)
        dt_ok = True   # single paired record -> image and sensor share one timestamp (|dt| = 0 <= 5 min)

        # step 10 : soft-max decision heads  P = softmax(W . R_total)
        preds = model.predict({"image": img, "sensors": sensors}, verbose=0)
        p_pest = preds["pest"][0]
        p_health = preds["health"][0]
        micro = preds["microclimate"][0]

        # steps 11-12 : pest class + threshold tau_p
        pest_class = PEST_CLASSES[int(np.argmax(p_pest))]
        pest_conf = float(np.max(p_pest))
        pest_label = pest_class if pest_conf >= args.tau_pest else "No pest detected"

        # steps 13-14 : health status + threshold tau_health
        health_class = HEALTH_CLASSES[int(np.argmax(p_health))]
        health_conf = float(np.max(p_health))
        status = health_class if health_conf >= args.tau_health else "Monitor"

        # step 15 : actuation
        act = decide_actuation(pest_label, pest_class, status)

        # step 16 : transmit
        payload = {
            "iot_id": args.iot_id,
            "iot_token": args.iot_token,           # shared secret for POST /api/inference
            "timestamp": ts.isoformat(),
            "image": Path(img_path).name,
            "sync_within_5min": bool(dt_ok),
            "fusion": {"w1": w1, "w2": w2,
                       "R_total_dim": None if r_total is None else int(r_total.shape[-1])},
            "pest": {"label": pest_label, "class": pest_class, "confidence": round(pest_conf, 4)},
            "health": {"status": status, "class": health_class, "confidence": round(health_conf, 4)},
            "microclimate": {k: round(float(v), 3) for k, v in zip(MICROCLIMATE_TARGETS, micro)},
            "vpd_kpa": round(float(micro[vpd_col]), 3),
            "actuation": act,
        }
        print(f"[t={t:03d}] {ts:%Y-%m-%d %H:%M}  img={Path(img_path).name}")
        print(f"         pest   : {pest_label} (conf {pest_conf:.2f})")
        print(f"         health : {status} (conf {health_conf:.2f}) | VPD {micro[vpd_col]:.3f} kPa")
        print(f"         action : {act['action']}  {act['actuators'] or ''}")
        transmit(payload, args.laravel_url, args.mqtt_host, mqtt_topic, args.dry_run)
        log.append(payload)

    out = data_dir / "algorithm1_run_log.json"
    out.write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
    print(f"\n=== done. {n} timesteps -> {out} ===")


if __name__ == "__main__":
    main()
