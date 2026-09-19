"""
Generate a standalone SYNTHETIC micro-climate dataset (CSV).

This does NOT touch the real labelled file (plant_health_data.csv). It writes a
new, clearly-named synthetic file that mirrors the data produced inside
evaluate_lstm_microclimate_10k.py:

    - 10,000 rows, 4-hour interval, seed = 42 (fully reproducible)
    - columns: timestamp, temperature, humidity, ph, light_intensity, vpd, stress_level
    - stress_level is derived from VPD quantile thresholds calibrated so the class
      balance matches the 1,200-sample reference distribution
      (Healthy 24.92%, Moderate Stress 33.42%, High Stress 41.67%
       -> 2492 / 3341 / 4167 for n = 10,000)

Paper wording:
    "The micro-climate series is synthetic. Stress classes were defined by VPD
     quantile thresholds calibrated to match the reference class distribution
     (Healthy 24.9%, Moderate 33.4%, High 41.7%)."
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

N_SAMPLES = 10000
SEED = 42
OUTPUT_CSV = "synthetic_microclimate_stress_10k.csv"

STRESS_CLASSES = ["Healthy", "Moderate Stress", "High Stress"]
REFERENCE_CLASS_COUNTS = {"Healthy": 299, "Moderate Stress": 401, "High Stress": 500}


def calculate_vpd(temperature, humidity):
    """Vapor Pressure Deficit (kPa) from temperature (°C) and RH (%), Magnus formula."""
    es = 0.6108 * np.exp((17.27 * temperature) / (temperature + 237.3))
    ea = es * (humidity / 100)
    return es - ea


def generate_synthetic_microclimate_data(n_samples=N_SAMPLES):
    """Same generator used in evaluate_lstm_microclimate_10k.py."""
    np.random.seed(SEED)

    start_date = datetime(2025, 1, 1)
    timestamps = [start_date + timedelta(hours=4 * i) for i in range(n_samples)]

    hours = np.array([ts.hour for ts in timestamps])
    days = np.array([ts.timetuple().tm_yday for ts in timestamps])

    # Temperature (20-35 °C, daily + seasonal pattern)
    temperature = 27 + 4 * np.sin(2 * np.pi * hours / 24) + 2 * np.sin(2 * np.pi * days / 365)
    temperature = temperature + np.random.normal(0, 0.5, n_samples)
    temperature = np.clip(temperature, 20, 35)

    # Humidity (40-90 %, inversely related to temperature)
    humidity = 75 - 1.5 * (temperature - 27) + 10 * np.sin(2 * np.pi * (hours + 6) / 24)
    humidity = humidity + np.random.normal(0, 2, n_samples)
    humidity = np.clip(humidity, 40, 90)

    # Soil pH (6.0-7.5, small seasonal drift)
    ph = 6.75 + 0.3 * np.sin(2 * np.pi * days / 365) + np.random.normal(0, 0.05, n_samples)
    ph = np.clip(ph, 6.0, 7.5)

    # Light intensity (0-1000 lux, strong daily pattern)
    light_base = 500 * (1 + np.sin(2 * np.pi * (hours - 6) / 24))
    light_base = np.where((hours >= 6) & (hours <= 18), light_base,
                          50 + 20 * np.random.random(n_samples))
    light_intensity = np.clip(light_base + np.random.normal(0, 20, n_samples), 0, 1000)

    vpd = calculate_vpd(temperature, humidity)

    return pd.DataFrame({
        "timestamp": timestamps,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "light_intensity": light_intensity,
        "vpd": vpd,
    })


def scale_class_counts(reference_counts, n_total):
    """Scale a reference class distribution to n_total (largest-remainder method)."""
    ref_total = sum(reference_counts.values())
    raw = {k: v / ref_total * n_total for k, v in reference_counts.items()}
    floored = {k: int(np.floor(v)) for k, v in raw.items()}
    remainder = n_total - sum(floored.values())
    for k in sorted(raw, key=lambda k: raw[k] - floored[k], reverse=True)[:remainder]:
        floored[k] += 1
    return floored


def assign_stress_labels(data, reference_counts=REFERENCE_CLASS_COUNTS):
    """Rank rows by VPD, cut into 3 blocks sized by the scaled reference distribution."""
    n = len(data)
    target = scale_class_counts(reference_counts, n)
    order = np.argsort(data["vpd"].values, kind="mergesort")  # ascending VPD

    labels = np.empty(n, dtype=object)
    start = 0
    for cls in STRESS_CLASSES:  # Healthy -> Moderate -> High
        end = start + target[cls]
        labels[order[start:end]] = cls
        start = end

    data = data.copy()
    data["stress_level"] = labels
    return data, target


def main():
    data = generate_synthetic_microclimate_data(N_SAMPLES)
    data, target = assign_stress_labels(data)

    # Record the VPD cut points actually used (for the paper's Methods section)
    vpd_sorted = np.sort(data["vpd"].values)
    cut1 = vpd_sorted[target["Healthy"] - 1]
    cut2 = vpd_sorted[target["Healthy"] + target["Moderate Stress"] - 1]

    data.to_csv(OUTPUT_CSV, index=False)

    counts = data["stress_level"].value_counts().reindex(STRESS_CLASSES)
    print(f"Written: {OUTPUT_CSV}  ({len(data)} rows)")
    print("\nClass distribution")
    print("-" * 48)
    for cls in STRESS_CLASSES:
        c = int(counts[cls])
        print(f"{cls:<18}{c:>8}{c / len(data) * 100:>10.2f}%")
    print("-" * 48)
    print(f"{'Total':<18}{len(data):>8}{100.0:>10.2f}%")
    print(f"\nVPD thresholds used:")
    print(f"  Healthy         : VPD <= {cut1:.4f} kPa")
    print(f"  Moderate Stress : {cut1:.4f} < VPD <= {cut2:.4f} kPa")
    print(f"  High Stress     : VPD >  {cut2:.4f} kPa")


if __name__ == "__main__":
    main()
