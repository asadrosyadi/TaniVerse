"""
Produce ONE real VPD prediction example (timestamp, reference vs predicted,
absolute error) for the manuscript paragraph, using the ACTUAL functions from
evaluate_lstm_microclimate_10k.py (same synthetic-data generator, same seed=42,
same LSTM architecture/hyperparameters, same chronological 80/20 split) -- just
skipping the 5-fold cross-validation loop, which this single-example paragraph
does not need, to keep the run short. Everything else is identical to running
`python evaluate_lstm_microclimate_10k.py` directly.
"""
import json
import numpy as np

from evaluate_lstm_microclimate_10k import (
    N_SAMPLES, generate_synthetic_microclimate_data, assign_stress_labels,
    create_sequences, build_lstm_regression_model, calculate_vpd,
)
from sklearn.preprocessing import MinMaxScaler

print("1. Generating synthetic microclimate data (seed=42, N=10000, same as evaluate_lstm_microclimate_10k.py)...")
data = generate_synthetic_microclimate_data(n_samples=N_SAMPLES, sequence_length=6)
data, _ = assign_stress_labels(data)

target_columns = ['temperature', 'humidity', 'ph', 'light_intensity', 'vpd']
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
features_scaled = scaler_X.fit_transform(data[target_columns].values)
targets_scaled = scaler_y.fit_transform(data[target_columns].values)
data_scaled = data.copy()
data_scaled[target_columns] = features_scaled

X, y = create_sequences(data_scaled, target_columns, sequence_length=6)
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
print(f"   train={len(X_train)}  test={len(X_test)}")

print("2. Training LSTM (100 epochs, same architecture/hyperparameters as the script)...")
model = build_lstm_regression_model(input_shape=(X_train.shape[1], X_train.shape[2]), output_dim=len(target_columns))
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=2)

print("3. Predicting on the chronological test split...")
y_pred_scaled = model.predict(X_test, verbose=0)
y_test_original = scaler_y.inverse_transform(y_test)
y_pred_original = scaler_y.inverse_transform(y_pred_scaled)

test_timestamps = data['timestamp'].iloc[split_idx + 6:].values
vpd_idx = target_columns.index('vpd')
abs_err = np.abs(y_test_original[:, vpd_idx] - y_pred_original[:, vpd_idx])

# aggregate metrics (for sanity-check against Table 5)
mae = float(np.mean(abs_err))
rmse = float(np.sqrt(np.mean((y_test_original[:, vpd_idx] - y_pred_original[:, vpd_idx]) ** 2)))
print(f"\n   VPD test-set MAE={mae:.4f} kPa  RMSE={rmse:.4f} kPa  (manuscript Table 5: MAE=0.038, RMSE=0.061)")

# pick ONE real example: the test-set point whose absolute error is closest to
# the reported MAE (0.038 kPa) -- i.e. a "typical" real case, not cherry-picked
# for being unusually good or bad.
target_mae = 0.038
closest = int(np.argmin(np.abs(abs_err - target_mae)))

record = {
    "index_in_test_set": closest,
    "timestamp": str(test_timestamps[closest]),
    "reference_vpd_kpa": float(y_test_original[closest, vpd_idx]),
    "predicted_vpd_kpa": float(y_pred_original[closest, vpd_idx]),
    "absolute_error_kpa": float(abs_err[closest]),
    "test_set_mae_kpa": mae,
    "test_set_rmse_kpa": rmse,
    "n_test": int(len(X_test)),
}
with open("real_vpd_example.json", "w", encoding="utf-8") as f:
    json.dump(record, f, indent=2)

print("\n=== REAL example (closest to reported MAE) ===")
print(json.dumps(record, indent=2))

# also show the single largest error in the test set, for context
worst = int(np.argmax(abs_err))
print("\n=== For context: largest absolute VPD error in this test set ===")
print(json.dumps({
    "timestamp": str(test_timestamps[worst]),
    "reference_vpd_kpa": float(y_test_original[worst, vpd_idx]),
    "predicted_vpd_kpa": float(y_pred_original[worst, vpd_idx]),
    "absolute_error_kpa": float(abs_err[worst]),
}, indent=2))
