"""
Keras model factory for the hybrid CNN-LSTM feature-fusion network.

    image (224x224x3) --> EfficientNet-B0 --> GAP --> Dense(n, ReLU) = R_visual
    sensors (T x d)   --> LSTM64 -> BN -> LSTM32 -> BN -> Drop(.2)
                          -> Dense(n, ReLU)                = R_environment
    R_total = w1 * R_visual + w2 * R_environment           (Eq. 1, AdaptiveFeatureFusion)
    R_total --> Drop(.2) --> Dense(K, softmax)             pest class      (Eq. 3)
            --> Drop(.2) --> Dense(3, softmax)             plant health
            --> Drop(.2) --> Dense(M, linear)              micro-climate / VPD regression

The fusion weights w1, w2 are **fixed hyper-parameters** (not learned): they are
chosen by grid search + 5-fold cross-validation in tune_fusion_weights.py, which
matches the manuscript (optimum w1 = 0.6, w2 = 0.4).
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from . import (
    FUSION_DIM, SEQUENCE_LENGTH, DEFAULT_W1,
    PEST_CLASSES, HEALTH_CLASSES, MICROCLIMATE_TARGETS, SENSOR_FEATURES,
)


@keras.utils.register_keras_serializable(package="hybrid_cnn_lstm")
class AdaptiveFeatureFusion(layers.Layer):
    """R_total = w1 * a + (1 - w1) * b  with w1 in [0, 1] a fixed hyper-parameter."""

    def __init__(self, w1: float = DEFAULT_W1, **kwargs):
        super().__init__(**kwargs)
        if not 0.0 <= w1 <= 1.0:
            raise ValueError(f"w1 must be in [0, 1], got {w1}")
        self.w1 = float(w1)
        self.w2 = 1.0 - self.w1

    def call(self, inputs):
        r_visual, r_environment = inputs
        return self.w1 * r_visual + self.w2 * r_environment

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"w1": self.w1})
        return cfg


def build_cnn_branch(img_size: int, fusion_dim: int, trainable_backbone: bool = False):
    """EfficientNet-B0 spatial encoder -> R_visual in R^fusion_dim."""
    base = keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3)
    )
    base.trainable = trainable_backbone
    inp = keras.Input(shape=(img_size, img_size, 3), name="image")
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D(name="cnn_gap")(x)
    x = layers.Dropout(0.2)(x)
    r_visual = layers.Dense(fusion_dim, activation="relu", name="R_visual")(x)
    return inp, r_visual, base


def build_lstm_branch(seq_len: int, n_features: int, fusion_dim: int):
    """Stacked LSTM temporal encoder -> R_environment in R^fusion_dim  (Eq. 4)."""
    inp = keras.Input(shape=(seq_len, n_features), name="sensors")
    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.LSTM(32, return_sequences=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    r_env = layers.Dense(fusion_dim, activation="relu", name="R_environment")(x)
    return inp, r_env


def build_hybrid_model(
    img_size: int = 224,
    n_sensor_features: int = len(SENSOR_FEATURES),
    seq_len: int = SEQUENCE_LENGTH,
    fusion_dim: int = FUSION_DIM,
    w1: float = DEFAULT_W1,
    n_pest: int = len(PEST_CLASSES),
    n_health: int = len(HEALTH_CLASSES),
    n_micro: int = len(MICROCLIMATE_TARGETS),
    fusion: str = "adaptive",          # "adaptive" (Eq. 1) | "concat" (no-fusion baseline)
    trainable_backbone: bool = False,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Build & compile the multimodal model.

    fusion="concat" reproduces the "CNN + LSTM (without fusion)" row of Table 6:
    the two representations are concatenated and projected, i.e. no weighted sum.
    """
    img_in, r_visual, _base = build_cnn_branch(img_size, fusion_dim, trainable_backbone)
    sens_in, r_env = build_lstm_branch(seq_len, n_sensor_features, fusion_dim)

    if fusion == "adaptive":
        r_total = AdaptiveFeatureFusion(w1=w1, name="feature_fusion")([r_visual, r_env])
    elif fusion == "concat":
        r_total = layers.Concatenate(name="feature_concat")([r_visual, r_env])
        r_total = layers.Dense(fusion_dim, activation="relu", name="concat_proj")(r_total)
    else:
        raise ValueError("fusion must be 'adaptive' or 'concat'")

    pest_out = layers.Dense(n_pest, activation="softmax", name="pest")(layers.Dropout(0.2)(r_total))
    health_out = layers.Dense(n_health, activation="softmax", name="health")(layers.Dropout(0.2)(r_total))
    vpd_out = layers.Dense(n_micro, activation="linear", name="microclimate")(layers.Dropout(0.2)(r_total))

    model = keras.Model(
        inputs={"image": img_in, "sensors": sens_in},
        outputs={"pest": pest_out, "health": health_out, "microclimate": vpd_out},
        name=f"hybrid_cnn_lstm_{fusion}",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "pest": "categorical_crossentropy",
            "health": "categorical_crossentropy",
            "microclimate": "mse",
        },
        loss_weights={"pest": 1.0, "health": 0.5, "microclimate": 0.5},
        metrics={"pest": "accuracy", "health": "accuracy", "microclimate": "mae"},
    )
    return model


def build_cnn_only_model(img_size: int = 224, fusion_dim: int = FUSION_DIM,
                         n_pest: int = len(PEST_CLASSES), learning_rate: float = 1e-3):
    """Unimodal CNN baseline (Table 6, row 'CNN only')."""
    inp, r_visual, _ = build_cnn_branch(img_size, fusion_dim)
    out = layers.Dense(n_pest, activation="softmax", name="pest")(layers.Dropout(0.2)(r_visual))
    model = keras.Model(inp, out, name="cnn_only")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_lstm_only_model(seq_len: int = SEQUENCE_LENGTH,
                          n_sensor_features: int = len(SENSOR_FEATURES),
                          fusion_dim: int = FUSION_DIM,
                          n_micro: int = len(MICROCLIMATE_TARGETS),
                          n_health: int = len(HEALTH_CLASSES),
                          learning_rate: float = 1e-3):
    """Unimodal LSTM baseline (Table 6, row 'LSTM only')."""
    inp, r_env = build_lstm_branch(seq_len, n_sensor_features, fusion_dim)
    micro = layers.Dense(n_micro, activation="linear", name="microclimate")(layers.Dropout(0.2)(r_env))
    health = layers.Dense(n_health, activation="softmax", name="health")(layers.Dropout(0.2)(r_env))
    model = keras.Model(inp, {"microclimate": micro, "health": health}, name="lstm_only")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss={"microclimate": "mse", "health": "categorical_crossentropy"},
        loss_weights={"microclimate": 1.0, "health": 0.5},
        metrics={"microclimate": "mae", "health": "accuracy"},
    )
    return model
