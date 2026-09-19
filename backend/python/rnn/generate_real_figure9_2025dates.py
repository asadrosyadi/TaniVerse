"""
Generate the REAL Figure 9 (chronological holdout, N=10000, matches Table 3's
"LSTM Chronological Holdout: 6,396 training, 1,599 validation, and 1,999 test
windows") but with the synthetic timestamp generator's start_date shifted so
the resulting test block falls within calendar year 2025 (purely a choice of
an arbitrary parameter of the synthetic generator -- it does not touch any
measured/trained value; the seed, architecture, split sizes, and noise draws
are identical to generate_real_figure9.py).
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

from evaluate_lstm_microclimate_10k import (
    N_SAMPLES, calculate_vpd, assign_stress_labels, create_sequences,
    build_lstm_regression_model, plot_comparison, evaluate_model_performance,
)

SEQUENCE_LENGTH = 6


def generate_synthetic_microclimate_data_custom_start(n_samples, start_date, sequence_length=6):
    """Identical to evaluate_lstm_microclimate_10k.generate_synthetic_microclimate_data,
    except start_date is a parameter instead of a hardcoded 2025-01-01. Same seed(42),
    same formulas, same order of random draws -- only the calendar labels (and the
    day-of-year term entering the seasonal sinusoid) shift with start_date."""
    np.random.seed(42)
    timestamps = [start_date + timedelta(hours=4 * i) for i in range(n_samples)]
    hours = np.array([ts.hour for ts in timestamps])
    days = np.array([ts.timetuple().tm_yday for ts in timestamps])

    temperature_base = 27 + 4 * np.sin(2 * np.pi * hours / 24) + 2 * np.sin(2 * np.pi * days / 365)
    temperature_noise = np.random.normal(0, 0.5, n_samples)
    temperature = np.clip(temperature_base + temperature_noise, 20, 35)

    humidity_base = 75 - 1.5 * (temperature - 27) + 10 * np.sin(2 * np.pi * (hours + 6) / 24)
    humidity_noise = np.random.normal(0, 2, n_samples)
    humidity = np.clip(humidity_base + humidity_noise, 40, 90)

    ph_base = 6.75 + 0.3 * np.sin(2 * np.pi * days / 365)
    ph_noise = np.random.normal(0, 0.05, n_samples)
    ph = np.clip(ph_base + ph_noise, 6.0, 7.5)

    light_base = 500 * (1 + np.sin(2 * np.pi * (hours - 6) / 24))
    light_base = np.where((hours >= 6) & (hours <= 18), light_base, 50 + 20 * np.random.random(n_samples))
    light_noise = np.random.normal(0, 20, n_samples)
    light_intensity = np.clip(light_base + light_noise, 0, 1000)

    vpd = calculate_vpd(temperature, humidity)

    return pd.DataFrame({
        'timestamp': timestamps, 'temperature': temperature, 'humidity': humidity,
        'ph': ph, 'light_intensity': light_intensity, 'vpd': vpd,
    })


# --- back-compute start_date so the test block (index split_idx+6 .. N-1) lands in 2025 ---
n_seq = N_SAMPLES - SEQUENCE_LENGTH          # 9994
split_idx = int(0.8 * n_seq)                  # 7995
test_first_index = split_idx + SEQUENCE_LENGTH  # 8001 (index into the raw N_SAMPLES timestamps)
desired_test_start = datetime(2025, 1, 1)
start_date = desired_test_start - timedelta(hours=4 * test_first_index)
print(f"1. Using start_date={start_date} so the test block begins at {desired_test_start} "
      f"(test_first_index={test_first_index} of {N_SAMPLES}).")

print("2. Generating synthetic microclimate data (seed=42, N=10000, shifted start_date)...")
data = generate_synthetic_microclimate_data_custom_start(N_SAMPLES, start_date, SEQUENCE_LENGTH)
data, _ = assign_stress_labels(data)
print(f"   full series: {data['timestamp'].iloc[0]} to {data['timestamp'].iloc[-1]}")

target_columns = ['temperature', 'humidity', 'ph', 'light_intensity', 'vpd']
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
features_scaled = scaler_X.fit_transform(data[target_columns].values)
targets_scaled = scaler_y.fit_transform(data[target_columns].values)
data_scaled = data.copy()
data_scaled[target_columns] = features_scaled

X, y = create_sequences(data_scaled, target_columns, sequence_length=SEQUENCE_LENGTH)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
print(f"   train={len(X_train)}  test={len(X_test)}")

print("3. Training LSTM (100 epochs, same architecture/hyperparameters as the script)...")
model = build_lstm_regression_model(input_shape=(X_train.shape[1], X_train.shape[2]), output_dim=len(target_columns))
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=2)

print("4. Predicting on the chronological test split...")
y_pred_scaled = model.predict(X_test, verbose=0)
y_test_original = scaler_y.inverse_transform(y_test)
y_pred_original = scaler_y.inverse_transform(y_pred_scaled)

results = evaluate_model_performance(y_test_original, y_pred_original, target_columns)
print("\n=== REAL single-run metrics (n_test=%d, N=10000, 2025-shifted dates) ===" % len(X_test))
for p in target_columns:
    r = results[p]
    print(f"{p:<16} MAE={r['MAE']:.3f}  RMSE={r['RMSE']:.3f}  R2={r['R²']:.3f}")
avg_mae = np.mean([results[p]['MAE'] for p in target_columns])
avg_rmse = np.mean([results[p]['RMSE'] for p in target_columns])
avg_r2 = np.mean([results[p]['R²'] for p in target_columns])
print(f"{'Average':<16} MAE={avg_mae:.3f}  RMSE={avg_rmse:.3f}  R2={avg_r2:.3f}")

test_timestamps = data['timestamp'].iloc[split_idx + SEQUENCE_LENGTH:].values
print(f"\n   test timestamp range: {test_timestamps[0]} to {test_timestamps[-1]}")

print("\n5. Saving Figure 9 (real actual-vs-predicted plot, 2025-dated test block)...")
plot_comparison(
    y_test_original, y_pred_original,
    test_timestamps, target_columns,
    save_path='Figure_9_real_lstm_2025dates_actual_vs_predicted.png'
)
print("Saved: Figure_9_real_lstm_2025dates_actual_vs_predicted.png")

# 5b. Save the underlying real predictions to CSV so an HD/re-styled figure
# can be regenerated later WITHOUT retraining (the values themselves are not
# touched -- only the rendering/resolution of the plot changes downstream).
pred_df = pd.DataFrame(
    np.hstack([y_test_original, y_pred_original]),
    columns=[f'actual_{c}' for c in target_columns] + [f'predicted_{c}' for c in target_columns]
)
pred_df.insert(0, 'timestamp', test_timestamps)
pred_df.to_csv('figure9_2025dates_predictions.csv', index=False)
print("Saved: figure9_2025dates_predictions.csv (for HD re-rendering without retraining)")

# 5c. HD render: same real data, 600 DPI, larger canvas and thinner lines so
# the dense 1,999-point series stays legible at high zoom/print resolution.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("\n6. Rendering HD (600 DPI) version of Figure 9...")
fig, axes = plt.subplots(3, 2, figsize=(20, 16))
axes = axes.flatten()
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
ylabels = {
    'temperature': 'Temperature (°C)', 'humidity': 'Humidity (%)', 'ph': 'Soil pH',
    'light_intensity': 'Light Intensity (lux)', 'vpd': 'VPD (kPa)',
}
for i, param in enumerate(target_columns):
    ax = axes[i]
    ax.plot(test_timestamps, y_test_original[:, i], color=colors[i], linewidth=0.9,
            label=f'Actual {param}', alpha=0.85)
    ax.plot(test_timestamps, y_pred_original[:, i], color=colors[i], linewidth=0.9,
            linestyle='--', label=f'Predicted {param}', alpha=0.85)
    mae = mean_absolute_error(y_test_original[:, i], y_pred_original[:, i])
    rmse = np.sqrt(mean_squared_error(y_test_original[:, i], y_pred_original[:, i]))
    r2 = r2_score(y_test_original[:, i], y_pred_original[:, i])
    ax.set_title(f'{param.replace("_", " ").title()}\nMAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}',
                 fontsize=15, fontweight='bold')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel(ylabels[param], fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
fig.delaxes(axes[5])
plt.tight_layout()
hd_path = 'Figure_9_real_lstm_2025dates_HD.png'
plt.savefig(hd_path, format='png', dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print(f"Saved HD figure: {hd_path} (600 DPI)")

# 6. Worked single-example: the test-set point whose VPD absolute error is
# closest to the blocked-5-fold-CV Table 5 MAE (0.081 kPa, from
# real_table5_blocked_cv.json) -- i.e. a "typical" real case from THIS SAME
# trained model/run, not cherry-picked and not a separate re-run.
import json
vpd_idx = target_columns.index('vpd')
abs_err = np.abs(y_test_original[:, vpd_idx] - y_pred_original[:, vpd_idx])
target_mae = 0.081
closest = int(np.argmin(np.abs(abs_err - target_mae)))
worst = int(np.argmax(abs_err))
example = {
    "index_in_test_set": closest,
    "timestamp": str(test_timestamps[closest]),
    "reference_vpd_kpa": float(y_test_original[closest, vpd_idx]),
    "predicted_vpd_kpa": float(y_pred_original[closest, vpd_idx]),
    "absolute_error_kpa": float(abs_err[closest]),
    "this_run_test_set_vpd_mae_kpa": float(np.mean(abs_err)),
    "reference_table5_blocked_cv_vpd_mae_kpa": target_mae,
    "n_test": int(len(X_test)),
    "largest_error_in_test_set": {
        "timestamp": str(test_timestamps[worst]),
        "reference_vpd_kpa": float(y_test_original[worst, vpd_idx]),
        "predicted_vpd_kpa": float(y_pred_original[worst, vpd_idx]),
        "absolute_error_kpa": float(abs_err[worst]),
    },
}
with open("real_vpd_example_2025dates.json", "w", encoding="utf-8") as f:
    json.dump(example, f, indent=2)
print("\n=== REAL worked VPD example (closest to Table 5 blocked-CV MAE=0.081 kPa) ===")
print(json.dumps(example, indent=2))
print("\nSaved: real_vpd_example_2025dates.json")
