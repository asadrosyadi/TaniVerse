"""
Hybrid CNN-LSTM edge-AIoT framework (Layer 4.5).

Implements the manuscript's core contribution: an adaptive **feature-fusion**
model that combines the EfficientNet-B0 spatial branch (R_visual) with the LSTM
temporal branch (R_environment) via

    R_total = w1 * R_visual + w2 * R_environment ,   w1 + w2 = 1        (Eq. 1)

followed by a shared soft-max decision head (Eq. 3) for pest class and
plant-health status, plus a micro-climate regression head for VPD.

Modules
-------
make_paired_dataset  : build a timestamp-aligned image <-> sensor <-> label manifest
fusion_model         : Keras model factory + AdaptiveFeatureFusion layer
tune_fusion_weights  : grid search over w1 with 5-fold CV (Eq. 1 weight selection)
train_hybrid         : train the fused model (Adam 1e-3, batch 32, 5-fold CV)
evaluate_hybrid      : reproduce Table 6 (CNN-only / LSTM-only / concat / fusion)
algorithm1           : runnable implementation of Table 2 "Algorithm 1"
"""

# Raw micro-climate channels captured by the IoT node (LSTM regression targets, Table 5)
MICROCLIMATE_TARGETS = ["temperature", "humidity", "ph", "light_intensity", "vpd"]

# Engineered LSTM input features = raw channels + cyclical time encoding + interaction
# term (manuscript sec. 3.3 / Algorithm 1 step 5). The LSTM branch consumes these 8.
SENSOR_FEATURES = MICROCLIMATE_TARGETS + ["hour_sin", "hour_cos", "temp_humidity"]

PEST_CLASSES = [
    "bacterial_leaf_blight", "bacterial_leaf_streak", "bacterial_panicle_blight",
    "blast", "brown_spot", "dead_heart", "downy_mildew", "hispa", "normal", "tungro",
]
HEALTH_CLASSES = ["Healthy", "Moderate Stress", "High Stress"]

SEQUENCE_LENGTH = 6          # T = 6  (24 h look-back at a 4 h grid)
FUSION_DIM = 128            # dimensionality n of R_visual / R_environment
DEFAULT_W1 = 0.6            # manuscript sec. 3.3: grid-search + 5-fold CV optimum
DEFAULT_W2 = 0.4
TAU_PEST = 0.5             # confidence threshold tau_p   (Algorithm 1, step 12)
TAU_HEALTH = 0.5          # confidence threshold tau_health (Algorithm 1, step 14)
