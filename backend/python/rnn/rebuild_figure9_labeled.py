"""
Re-render Figure 9 with (a)-(e) subplot labels, reusing the REAL prediction
data already saved in figure9_2025dates_predictions.csv (from the HD run of
generate_real_figure9_2025dates.py) -- no retraining, same real numbers.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('figure9_2025dates_predictions.csv', parse_dates=['timestamp'])
target_columns = ['temperature', 'humidity', 'ph', 'light_intensity', 'vpd']
panel_letters = ['a', 'b', 'c', 'd', 'e']
ylabels = {
    'temperature': 'Temperature (°C)', 'humidity': 'Humidity (%)', 'ph': 'Soil pH',
    'light_intensity': 'Light Intensity (lux)', 'vpd': 'VPD (kPa)',
}
titles = {
    'temperature': 'Air Temperature', 'humidity': 'Relative Humidity', 'ph': 'Soil pH',
    'light_intensity': 'Light Intensity', 'vpd': 'VPD',
}
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

fig, axes = plt.subplots(3, 2, figsize=(20, 16))
axes = axes.flatten()

for i, param in enumerate(target_columns):
    ax = axes[i]
    y_true = df[f'actual_{param}'].values
    y_pred = df[f'predicted_{param}'].values
    ax.plot(df['timestamp'], y_true, color=colors[i], linewidth=0.9,
            label=f'Actual {param}', alpha=0.85)
    ax.plot(df['timestamp'], y_pred, color=colors[i], linewidth=0.9,
            linestyle='--', label=f'Predicted {param}', alpha=0.85)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    ax.set_title(f'({panel_letters[i]}) {titles[param]}\nMAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}',
                 fontsize=15, fontweight='bold')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel(ylabels[param], fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)

fig.delaxes(axes[5])
plt.tight_layout()
out = 'Figure_9_real_lstm_2025dates_HD_labeled.png'
plt.savefig(out, format='png', dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print(f"Saved: {out}")
