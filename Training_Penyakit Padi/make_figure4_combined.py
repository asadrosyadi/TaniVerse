"""
Build the complete Figure 4 for the manuscript:

  (a) one real training image plus augmented variants generated from that same
      image by the actual training augmentation pipeline (train.py), and
  (b) the system's JSON reporting output, produced by actually running this
      project's ForecastService/ModelService (backend/python/rnn) -- rendered
      verbatim from figure4b_real_forecast_entry.json.

Both panels come from real code and real data; no generative imagery and no
hand-typed values.

Run make_figure4b_json_output.py (in backend/python/rnn) first to refresh (b).
"""
import json
import textwrap
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from train import (
    ROTATION_DEG, ROTATION_FACTOR, CROP_SCALE_MIN,
    GAUSSIAN_NOISE_VARIANCE, GAUSSIAN_NOISE_STDDEV, GAUSSIAN_NOISE_STDDEV_UNIT,
    build_augmentation,
)

HERE = Path(__file__).resolve().parent
JSON_PATH = HERE.parent / "backend" / "python" / "rnn" / "figure4b_real_forecast_entry.json"
SOURCE_IMAGE = "train/brown_spot/100001.jpg"
# Augmentation is applied at the model input resolution actually used in training.
# The panels are only enlarged when the figure is composed, so every panel is the
# exact tensor the training pipeline would produce.
IMG_SIZE = 224
SEED = 42


def load_image(path, img_size=IMG_SIZE):
    raw = tf.io.read_file(str(path))
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    return tf.cast(tf.image.resize(img, (img_size, img_size)), tf.float32)


def to_uint8(x):
    return np.clip(np.asarray(x), 0, 255).astype("uint8")


def apply_layer(layer, img):
    return layer(tf.expand_dims(img, 0), training=True)[0]


def flip_seed(mode, img, reference, max_seed=200):
    for s in range(max_seed):
        out = np.asarray(apply_layer(layers.RandomFlip(mode, seed=s), img))
        if np.allclose(out, reference, atol=1e-4):
            return s, out
    raise RuntimeError(f"no seed triggered the {mode} flip")


def build_panels(img):
    """Labels use plain text that matches the wording of Section 3.3 of the
    manuscript (rotation +/-25 deg, horizontal and vertical flipping, random crop
    with scale 0.8-1.0, Gaussian noise of variance 0.01). No mathtext markup, so
    nothing can be mangled at export time."""
    base = np.asarray(img)
    panels = [("Original", to_uint8(img), "no augmentation")]

    rot = apply_layer(layers.RandomRotation((ROTATION_FACTOR, ROTATION_FACTOR), seed=SEED), img)
    panels.append(("Rotation", to_uint8(rot),
                   f"rotation +{ROTATION_DEG:.0f}° (range ±{ROTATION_DEG:.0f}°)"))

    hs, hflip = flip_seed("horizontal", img, base[:, ::-1, :])
    panels.append(("Horizontal flip", to_uint8(hflip), f"horizontal flip (seed {hs})"))

    vs, vflip = flip_seed("vertical", img, base[::-1, :, :])
    panels.append(("Vertical flip", to_uint8(vflip), f"vertical flip (seed {vs})"))

    z = -(1.0 - CROP_SCALE_MIN)
    zoom = apply_layer(layers.RandomZoom(height_factor=(z, z), width_factor=(z, z), seed=SEED), img)
    panels.append(("Random crop", to_uint8(zoom),
                   f"random crop, scale {CROP_SCALE_MIN:.1f} (range {CROP_SCALE_MIN:.1f}–1.0)"))

    tf.keras.utils.set_random_seed(SEED)
    noise_block = tf.keras.Sequential([
        layers.Rescaling(1.0 / 255.0),
        layers.GaussianNoise(GAUSSIAN_NOISE_STDDEV_UNIT),
        layers.Rescaling(255.0),
    ])
    # train.py injects the noise after Rescaling(1/255) and rescales back to 0-255,
    # so the stated variance is on the normalised [0, 1] pixel scale.
    panels.append(("Gaussian noise", to_uint8(apply_layer(noise_block, img)),
                   f"Gaussian noise, variance {GAUSSIAN_NOISE_VARIANCE} on the [0, 1] pixel scale"))

    aug = build_augmentation(SEED)
    for k in range(2):
        tf.keras.utils.set_random_seed(SEED + k)
        out = aug(tf.expand_dims(img, 0), training=True)[0]
        panels.append((f"Full pipeline #{k + 1}", to_uint8(out),
                       f"full pipeline, random draw (seed {SEED + k})"))
    return panels


def main():
    src = HERE / SOURCE_IMAGE
    if not src.exists():
        raise SystemExit(f"Source image not found: {src}")
    if not JSON_PATH.exists():
        raise SystemExit(f"Run make_figure4b_json_output.py first: {JSON_PATH} missing")

    img = load_image(src)
    panels = build_panels(img)
    entry = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    json_text = json.dumps(entry, indent=4, sort_keys=True)

    fig = plt.figure(figsize=(19, 10.4))
    gs = GridSpec(1, 2, width_ratios=[1.62, 1.0], wspace=0.10,
                  left=0.015, right=0.985, top=0.875, bottom=0.05)

    # ---------------- panel (a) ----------------
    gs_a = gs[0, 0].subgridspec(2, 4, hspace=0.62, wspace=0.09)
    for i, (title, im, sub) in enumerate(panels):
        ax = fig.add_subplot(gs_a[i // 4, i % 4])
        ax.imshow(im, interpolation="lanczos")
        ax.set_title(title, fontsize=15, fontweight="bold", pad=7)
        ax.set_xlabel(textwrap.fill(sub, 26), fontsize=11.5, labelpad=8)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#9aa5b1")

    ax_a = fig.add_subplot(gs[0, 0], frameon=False)
    ax_a.set_xticks([]); ax_a.set_yticks([])
    ax_a.set_title("(a)  Augmented variants from one training image\n"
                   f"source: {SOURCE_IMAGE}",
                   fontsize=15, fontweight="bold", pad=40)
    ax_a.text(0.5, 1.012,
              f"augmentation applied at the {IMG_SIZE} × {IMG_SIZE} px model input "
              f"resolution used in training; panels enlarged for display",
              fontsize=11.5, style="italic", color="#444444",
              ha="center", va="bottom", transform=ax_a.transAxes)
    ax_a.patch.set_alpha(0)

    # ---------------- panel (b) ----------------
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.axis("off")
    ax_b.set_title("(b)  Encoded JSON output\n"
                   "produced by the system's forecast service",
                   fontsize=14, fontweight="bold", pad=26)
    ax_b.text(0.02, 0.995, json_text, family="DejaVu Sans Mono", fontsize=12.2,
              va="top", ha="left", linespacing=1.5, transform=ax_b.transAxes,
              bbox=dict(boxstyle="round,pad=0.7", facecolor="#f5f8fb",
                        edgecolor="#8fa8bf", linewidth=1.3))

    out = HERE / "Figure_4_real_combined.png"
    plt.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
    # Vector copy: embed this in the manuscript so the text is never re-rasterised.
    out_pdf = HERE / "Figure_4_real_combined.pdf"
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out}  (600 DPI raster)")
    print(f"Saved: {out_pdf}  (vector)")
    print(f"  (a) source image : {SOURCE_IMAGE}")
    print(f"  (b) JSON source  : {JSON_PATH.relative_to(HERE.parent)}")
    print("\nPanel (a) labels exported (verify these against the embedded figure):")
    for title, _, sub in panels:
        print(f"  {title:<18} | {sub}")


if __name__ == "__main__":
    main()
