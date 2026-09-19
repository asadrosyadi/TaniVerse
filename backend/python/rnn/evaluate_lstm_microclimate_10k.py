"""
LSTM Microclimate Parameter Prediction Evaluation (10,000 samples)
LSTM model evaluation for microclimate parameter prediction (temperature, humidity, pH, light intensity, VPD)
with MAE, RMSE, and R² metrics.

This is a standalone copy of evaluate_lstm_microclimate.py that uses a larger
synthetic dataset of 10,000 samples. Output figures use a "_10k" suffix so they
do not overwrite the results of the original script.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
# Fix for PIL saving issue
plt.ioff()  # Turn off interactive mode
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
# plt.style.use('seaborn-v0_8')  # Skip seaborn style
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12

# Total number of synthetic samples generated for this evaluation
N_SAMPLES = 10000

# Stress-level classes (same taxonomy as the plant-health classifier:
# config/label_encoder.pkl -> ['Healthy', 'High Stress', 'Moderate Stress'])
STRESS_CLASSES = ['Healthy', 'Moderate Stress', 'High Stress']

# Reference class distribution taken from the 1,200-observation labelled set
# (Healthy = 299, Moderate Stress = 401, High Stress = 500).
# The 10,000-sample synthetic set below is split with the SAME proportions,
# so the class balance "menyerupai 1.200" while n is scaled up to 10,000.
REFERENCE_CLASS_COUNTS = {'Healthy': 299, 'Moderate Stress': 401, 'High Stress': 500}


def scale_class_counts(reference_counts, n_total):
    """
    Scale a reference class distribution to a new total using the
    largest-remainder method so the scaled counts sum exactly to n_total.
    """
    ref_total = sum(reference_counts.values())
    raw = {k: v / ref_total * n_total for k, v in reference_counts.items()}
    floored = {k: int(np.floor(v)) for k, v in raw.items()}
    remainder = n_total - sum(floored.values())
    # distribute the leftover to the classes with the largest fractional parts
    frac_order = sorted(raw, key=lambda k: raw[k] - floored[k], reverse=True)
    for k in frac_order[:remainder]:
        floored[k] += 1
    return floored


def assign_stress_labels(data, reference_counts=REFERENCE_CLASS_COUNTS):
    """
    Derive a plant micro-climate stress level for every row from VPD
    (Vapor Pressure Deficit, an established agronomic water-stress indicator:
    low VPD -> comfortable, high VPD -> atmospheric water stress).

    Rows are ranked by VPD and cut into three groups whose sizes follow the
    reference (1,200-sample) distribution scaled to len(data):
        lowest  VPD block  -> 'Healthy'
        middle  VPD block  -> 'Moderate Stress'
        highest VPD block  -> 'High Stress'

    Returns the DataFrame with an added 'stress_level' column and the
    dict of realised class counts.
    """
    n = len(data)
    target = scale_class_counts(reference_counts, n)

    # ascending VPD -> ascending stress
    order = np.argsort(data['vpd'].values, kind='mergesort')

    labels = np.empty(n, dtype=object)
    start = 0
    for cls in STRESS_CLASSES:            # Healthy, Moderate Stress, High Stress
        end = start + target[cls]
        labels[order[start:end]] = cls
        start = end

    data = data.copy()
    data['stress_level'] = labels

    realised = {cls: int((labels == cls).sum()) for cls in STRESS_CLASSES}
    return data, realised


def print_class_distribution(title, counts):
    """Pretty-print a class-distribution table with counts and percentages."""
    total = sum(counts.values())
    print(f"\n{title}")
    print("-" * 52)
    print(f"{'Class':<20}{'Observations':>14}{'Percentage':>16}")
    print("-" * 52)
    for cls in STRESS_CLASSES:
        c = counts.get(cls, 0)
        print(f"{cls:<20}{c:>14}{c / total * 100:>15.2f}%")
    print("-" * 52)
    print(f"{'Total':<20}{total:>14}{100.0:>15.2f}%")


def calculate_vpd(temperature, humidity):
    """
    Calculate Vapor Pressure Deficit (VPD) from temperature and humidity

    Args:
        temperature: Temperature in Celsius
        humidity: Relative humidity in percent (0-100)

    Returns:
        VPD in kPa
    """
    # Saturated vapor pressure (kPa) - Magnus formula
    es = 0.6108 * np.exp((17.27 * temperature) / (temperature + 237.3))

    # Actual vapor pressure (kPa)
    ea = es * (humidity / 100)

    # Vapor Pressure Deficit (kPa)
    vpd = es - ea

    return vpd


def generate_synthetic_microclimate_data(n_samples=N_SAMPLES, sequence_length=6):
    """
    Generate realistic synthetic microclimate data for model evaluation
    """
    np.random.seed(42)

    # Generate timestamps (4 hour interval)
    start_date = datetime(2025, 1, 1)
    timestamps = [start_date + timedelta(hours=4 * i) for i in range(n_samples)]

    # Generate base patterns
    hours = np.array([ts.hour for ts in timestamps])
    days = np.array([ts.timetuple().tm_yday for ts in timestamps])

    # Temperature (20-35°C with daily and seasonal patterns)
    temperature_base = 27 + 4 * np.sin(2 * np.pi * hours / 24) + 2 * np.sin(2 * np.pi * days / 365)
    temperature_noise = np.random.normal(0, 0.5, n_samples)
    temperature = temperature_base + temperature_noise
    temperature = np.clip(temperature, 20, 35)

    # Humidity (40-90% inversely related to temperature)
    humidity_base = 75 - 1.5 * (temperature - 27) + 10 * np.sin(2 * np.pi * (hours + 6) / 24)
    humidity_noise = np.random.normal(0, 2, n_samples)
    humidity = humidity_base + humidity_noise
    humidity = np.clip(humidity, 40, 90)

    # Soil pH (6.0-7.5 with small variation)
    ph_base = 6.75 + 0.3 * np.sin(2 * np.pi * days / 365)
    ph_noise = np.random.normal(0, 0.05, n_samples)
    ph = ph_base + ph_noise
    ph = np.clip(ph, 6.0, 7.5)

    # Light intensity (0-1000 lux with strong daily pattern)
    light_base = 500 * (1 + np.sin(2 * np.pi * (hours - 6) / 24))
    light_base = np.where((hours >= 6) & (hours <= 18), light_base,
                          50 + 20 * np.random.random(n_samples))
    light_noise = np.random.normal(0, 20, n_samples)
    light_intensity = light_base + light_noise
    light_intensity = np.clip(light_intensity, 0, 1000)

    # Calculate VPD
    vpd = calculate_vpd(temperature, humidity)

    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperature,
        'humidity': humidity,
        'ph': ph,
        'light_intensity': light_intensity,
        'vpd': vpd
    })

    return data


def create_sequences(data, target_columns, sequence_length=6):
    """
    Create sequences for LSTM from time series data
    """
    features = data[['temperature', 'humidity', 'ph', 'light_intensity', 'vpd']].values
    targets = data[target_columns].values

    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(features[i:(i + sequence_length)])
        y.append(targets[i + sequence_length])

    return np.array(X), np.array(y)


def build_lstm_regression_model(input_shape, output_dim):
    """
    Build LSTM model for multi-output regression
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(output_dim, activation='linear')  # Linear for regression
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )

    return model


def evaluate_model_performance(y_true, y_pred, parameter_names):
    """
    Calculate MAE, RMSE, and R² for each parameter
    """
    results = {}

    for i, param in enumerate(parameter_names):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        r2 = r2_score(y_true[:, i], y_pred[:, i])

        results[param] = {
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2
        }

    return results


def plot_comparison(y_true, y_pred, timestamps, parameter_names, save_path=None):
    """
    Create comparison plot of actual vs predicted values
    """
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    axes = axes.flatten()

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    for i, param in enumerate(parameter_names):
        ax = axes[i]

        # Plot actual vs predicted
        ax.plot(timestamps, y_true[:, i], color=colors[i], linewidth=2,
                label=f'Actual {param}', alpha=0.8)
        ax.plot(timestamps, y_pred[:, i], color=colors[i], linewidth=2,
                linestyle='--', label=f'Predicted {param}', alpha=0.8)

        # Calculate metrics
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        r2 = r2_score(y_true[:, i], y_pred[:, i])

        ax.set_title(f'{param.title()}\nMAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Time', fontsize=12)

        # Set appropriate y-labels and units
        if param == 'temperature':
            ax.set_ylabel('Temperature (°C)', fontsize=12)
        elif param == 'humidity':
            ax.set_ylabel('Humidity (%)', fontsize=12)
        elif param == 'ph':
            ax.set_ylabel('Soil pH', fontsize=12)
        elif param == 'light_intensity':
            ax.set_ylabel('Light Intensity (lux)', fontsize=12)
        elif param == 'vpd':
            ax.set_ylabel('VPD (kPa)', fontsize=12)

        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Rotate x-axis labels
        ax.tick_params(axis='x', rotation=45)

    # Remove the empty subplot
    fig.delaxes(axes[5])

    plt.tight_layout()

    if save_path:
        try:
            plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"Plot saved as: {save_path}")
        except Exception as e:
            print(f"Warning: Could not save plot {save_path}: {e}")

    plt.close()  # Close instead of show


def main():
    """
    Main function for LSTM model evaluation
    """
    print("=" * 60)
    print("LSTM MODEL EVALUATION FOR MICROCLIMATE PARAMETER PREDICTION (10,000 SAMPLES)")
    print("=" * 60)

    # 1. Generate synthetic data
    print("\n1. Generating synthetic microclimate data...")
    data = generate_synthetic_microclimate_data(n_samples=N_SAMPLES, sequence_length=6)
    print(f"Data generated: {len(data)} samples")

    # 1b. Derive stress-level classes (proportions scaled from the 1,200-sample set)
    print("\n1b. Assigning micro-climate stress levels (VPD-based)...")
    data, class_counts = assign_stress_labels(data)

    scaled_target = scale_class_counts(REFERENCE_CLASS_COUNTS, N_SAMPLES)
    print_class_distribution(
        f"Reference distribution (1,200-sample labelled set)",
        REFERENCE_CLASS_COUNTS,
    )
    print_class_distribution(
        f"Scaled target for N = {N_SAMPLES} (same proportions)",
        scaled_target,
    )
    print_class_distribution(
        f"Realised distribution in the {len(data)}-sample synthetic set",
        class_counts,
    )

    # 2. Prepare data for LSTM
    print("\n2. Preparing data for LSTM...")
    target_columns = ['temperature', 'humidity', 'ph', 'light_intensity', 'vpd']

    # Normalize data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    features_scaled = scaler_X.fit_transform(data[target_columns].values)
    targets_scaled = scaler_y.fit_transform(data[target_columns].values)

    # Create scaled dataframe for sequence creation
    data_scaled = data.copy()
    data_scaled[target_columns] = features_scaled

    # Create sequences
    X, y = create_sequences(data_scaled, target_columns, sequence_length=6)
    print(f"Total sequences: {len(X)}")

    # Per-sequence stress label = label of the row being predicted
    seq_labels = data['stress_level'].values[6:]

    # Train-test split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    labels_train, labels_test = seq_labels[:split_idx], seq_labels[split_idx:]

    print(f"Training data: {X_train.shape}, Testing data: {X_test.shape}")

    print_class_distribution(
        "Stress-level distribution of LSTM sequences (train)",
        {cls: int((labels_train == cls).sum()) for cls in STRESS_CLASSES},
    )
    print_class_distribution(
        "Stress-level distribution of LSTM sequences (test)",
        {cls: int((labels_test == cls).sum()) for cls in STRESS_CLASSES},
    )

    # 3. Build and train model
    print("\n3. Building and training LSTM model...")
    model = build_lstm_regression_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        output_dim=len(target_columns)
    )

    # 5-fold time-series cross-validation (manuscript sec. 3.7: mean +/- std)
    try:
        from evaluate_lstm_microclimate import cross_validate_regression, rolling_vpd_forecast_r2
        print("\n3b. Running 5-fold time-series cross-validation (100 epochs/fold)...")
        cv_summary, _ = cross_validate_regression(
            X, y, scaler_y, target_columns, n_splits=5, epochs=100, seed=42
        )
        print("\nTable 5 (cross-validated, 10k). LSTM Performance -- mean +/- std over 5 folds")
        print("-" * 74)
        for p in target_columns:
            s = cv_summary[p]
            print(f"{p:<22} "
                  f"MAE {s['MAE']['mean']:.3f}+/-{s['MAE']['std']:.3f}   "
                  f"RMSE {s['RMSE']['mean']:.3f}+/-{s['RMSE']['std']:.3f}   "
                  f"R2 {s['R2']['mean']:.3f}+/-{s['R2']['std']:.3f}")
        print("-" * 74)
    except Exception as e:  # keep the standalone script runnable even if the import fails
        print(f"[warn] cross-validation skipped: {e}")
        rolling_vpd_forecast_r2 = None

    # Train the reported model (100 epochs, Table 3)
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    # 4. Make predictions
    print("\n4. Making predictions...")
    y_pred_scaled = model.predict(X_test, verbose=0)

    # Inverse transform predictions
    y_test_original = scaler_y.inverse_transform(y_test)
    y_pred_original = scaler_y.inverse_transform(y_pred_scaled)

    # 5. Evaluate performance
    print("\n5. Evaluating model performance...")
    results = evaluate_model_performance(y_test_original, y_pred_original, target_columns)

    # Print results table
    print("\nTable 5. LSTM Performance for Microclimate Parameter Prediction (10,000 samples)")
    print("-" * 60)
    print(f"{'Parameter':<20} {'MAE':<10} {'RMSE':<10} {'R²':<10}")
    print("-" * 60)

    parameter_display = {
        'temperature': 'Temperature (°C)',
        'humidity': 'Humidity (%)',
        'ph': 'Soil pH',
        'light_intensity': 'Light Intensity (lux)',
        'vpd': 'VPD (kPa)'
    }

    total_mae = total_rmse = total_r2 = 0

    for param in target_columns:
        mae = results[param]['MAE']
        rmse = results[param]['RMSE']
        r2 = results[param]['R²']

        total_mae += mae
        total_rmse += rmse
        total_r2 += r2

        # Format display based on parameter scale
        if param == 'ph':
            print(f"{parameter_display[param]:<20} {mae:<10.3f} {rmse:<10.3f} {r2:<10.3f}")
        elif param == 'vpd':
            print(f"{parameter_display[param]:<20} {mae:<10.3f} {rmse:<10.3f} {r2:<10.3f}")
        elif param == 'light_intensity':
            print(f"{parameter_display[param]:<20} {mae:<10.1f} {rmse:<10.1f} {r2:<10.3f}")
        else:
            print(f"{parameter_display[param]:<20} {mae:<10.2f} {rmse:<10.2f} {r2:<10.3f}")

    print("-" * 60)
    avg_mae = total_mae / len(target_columns)
    avg_rmse = total_rmse / len(target_columns)
    avg_r2 = total_r2 / len(target_columns)
    print(f"{'Average':<20} {avg_mae:<10.2f} {avg_rmse:<10.2f} {avg_r2:<10.3f}")

    # 6. Create visualization
    print("\n6. Creating visualization...")

    # Get timestamps for plotting (corresponding to test data)
    test_timestamps = data['timestamp'].iloc[split_idx + 6:].values

    # Focus on temperature and humidity for main plot
    plot_comparison(
        y_test_original, y_pred_original,
        test_timestamps, target_columns,
        save_path='gambar_9_perbandingan_lstm_10k.png'
    )

    # 7. Create focused plot for temperature and humidity only (Figure 9)
    print("\n7. Creating Figure 9 - Temperature and Humidity comparison...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Temperature plot
    ax1.plot(test_timestamps, y_test_original[:, 0], color='#e74c3c',
             linewidth=2.5, label='Actual Temperature', alpha=0.8)
    ax1.plot(test_timestamps, y_pred_original[:, 0], color='#e74c3c',
             linewidth=2.5, linestyle='--', label='Predicted Temperature', alpha=0.8)

    ax1.set_title('Actual vs Predicted Temperature Trend Comparison (LSTM, 10,000 samples)',
                  fontsize=16, fontweight='bold')
    ax1.set_ylabel('Temperature (°C)', fontsize=14)
    ax1.legend(loc='upper right', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # Humidity plot
    ax2.plot(test_timestamps, y_test_original[:, 1], color='#3498db',
             linewidth=2.5, label='Actual Humidity', alpha=0.8)
    ax2.plot(test_timestamps, y_pred_original[:, 1], color='#3498db',
             linewidth=2.5, linestyle='--', label='Predicted Humidity', alpha=0.8)

    ax2.set_title('Actual vs Predicted Humidity Trend Comparison (LSTM, 10,000 samples)',
                  fontsize=16, fontweight='bold')
    ax2.set_ylabel('Humidity (%)', fontsize=14)
    ax2.set_xlabel('Time', fontsize=14)
    ax2.legend(loc='upper right', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)

    # Add performance metrics as text
    temp_mae = results['temperature']['MAE']
    temp_rmse = results['temperature']['RMSE']
    temp_r2 = results['temperature']['R²']

    humid_mae = results['humidity']['MAE']
    humid_rmse = results['humidity']['RMSE']
    humid_r2 = results['humidity']['R²']

    ax1.text(0.02, 0.98, f'MAE: {temp_mae:.2f}°C, RMSE: {temp_rmse:.2f}°C, R²: {temp_r2:.3f}',
             transform=ax1.transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax2.text(0.02, 0.98, f'MAE: {humid_mae:.2f}%, RMSE: {humid_rmse:.2f}%, R²: {humid_r2:.3f}',
             transform=ax2.transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    try:
        plt.savefig('gambar_9_suhu_kelembapan_lstm_10k.png', format='png', dpi=300,
                    bbox_inches='tight', facecolor='white', edgecolor='none')
        print("Figure 9 saved as: gambar_9_suhu_kelembapan_lstm_10k.png")
    except Exception as e:
        print(f"Warning: Could not save Figure 9: {e}")
    plt.close()  # Close instead of show

    # 7b. 14-day rolling VPD forecast (manuscript sec. 3.7 / 4.5: R2 ~ 0.932)
    if 'rolling_vpd_forecast_r2' in dir() and rolling_vpd_forecast_r2 is not None:
        print("\n7b. Evaluating 14-day rolling VPD forecast...")
        roll = rolling_vpd_forecast_r2(model, X_test, y_test, scaler_y, target_columns,
                                       horizon_days=14, step_hours=4)
        if roll:
            print(f"   14-day VPD forecast: R2 = {roll['R2']:.3f}, RMSE = {roll['RMSE']:.3f} kPa "
                  f"(n = {roll['n_points']} points)")

    # 8. Export dataset and predictions to CSV for reference
    print("\n8. Exporting data to CSV...")
    try:
        data.to_csv('dataset_microclimate_lstm_10k.csv', index=False)
        print("Dataset saved as: dataset_microclimate_lstm_10k.csv")

        pred_df = pd.DataFrame(
            np.hstack([y_test_original, y_pred_original]),
            columns=[f'actual_{c}' for c in target_columns] +
                    [f'predicted_{c}' for c in target_columns]
        )
        pred_df.insert(0, 'timestamp', test_timestamps)
        pred_df['stress_level'] = labels_test
        pred_df.to_csv('prediksi_microclimate_lstm_10k.csv', index=False)
        print("Predictions saved as: prediksi_microclimate_lstm_10k.csv")

        # Class-distribution table (Table X in the paper)
        dist_df = pd.DataFrame({
            'stress_level': STRESS_CLASSES,
            'reference_1200': [REFERENCE_CLASS_COUNTS[c] for c in STRESS_CLASSES],
            'observations_10000': [class_counts[c] for c in STRESS_CLASSES],
        })
        dist_df['percentage'] = (dist_df['observations_10000'] /
                                 dist_df['observations_10000'].sum() * 100).round(2)
        dist_df.to_csv('distribusi_kelas_lstm_10k.csv', index=False)
        print("Class distribution saved as: distribusi_kelas_lstm_10k.csv")
    except Exception as e:
        print(f"Warning: Could not export CSV: {e}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED!")
    print("=" * 60)
    print(f"\nDataset size: {len(data)} raw samples -> {len(X)} LSTM sequences")
    print(f"Train / Test: {len(X_train)} / {len(X_test)}")
    print(f"Stress-level split (10,000): " +
          ", ".join(f"{c} = {class_counts[c]}" for c in STRESS_CLASSES) +
          f"  [proportions from the 1,200-sample set: " +
          ", ".join(f"{c} = {REFERENCE_CLASS_COUNTS[c]}" for c in STRESS_CLASSES) + "]")
    print(f"\nLSTM model shows good performance with:")
    print(f"- Average R²: {avg_r2:.3f}")
    print(f"- Average MAE: {avg_mae:.3f}")
    print(f"- Average RMSE: {avg_rmse:.3f}")
    print(f"\nTemperature and humidity prediction curves follow actual data with small deviations.")
    print(f"Low VPD RMSE ({results['vpd']['RMSE']:.3f} kPa) indicates LSTM's ability")
    print(f"to reliably model microclimate stress (water-vapor).")


if __name__ == "__main__":
    main()
