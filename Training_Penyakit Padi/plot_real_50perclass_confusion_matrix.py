"""
Plot Figure 8 from the REAL, actually-measured 500-image (50-per-class) subset
confusion matrix in real_50perclass_confusion_matrix.json (produced by
compute_real_50perclass_matrix.py running efficientnetb0_model.h5 inference
on the first 50 sorted-filename images per class in val/). No fabricated or
hand-typed numbers -- this is the measured result (43.4% accuracy).
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

here = Path(__file__).resolve().parent
d = json.loads((here / "real_50perclass_confusion_matrix.json").read_text(encoding="utf-8"))
cm = np.array(d["confusion_matrix"])
names = d["class_names"]
K = len(names)
n_per_class = d["n_per_class"]

annot = np.empty_like(cm, dtype=object)
for i in range(K):
    for j in range(K):
        pct = cm[i, j] / n_per_class * 100
        annot[i, j] = f"{cm[i, j]}\n({pct:.1f}%)"

plt.figure(figsize=(11, 9))
ax = sns.heatmap(cm, annot=annot, fmt="", cmap="Blues",
                  xticklabels=names, yticklabels=names,
                  cbar_kws={"label": "Number of Predictions"}, square=True,
                  linewidths=0.5, linecolor="white", vmin=0)

bi, bri = names.index("blast"), names.index("brown_spot")
for (i, j) in [(bi, bri), (bri, bi)]:
    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="red", lw=3))

acc = d["accuracy_percent"]
n_total = d["n_total"]
plt.title(
    f"EfficientNet-B0 predictions on a {n_total}-image validation subset "
    f"({n_per_class} images per class)\n"
    f"First {n_per_class} images per class in sorted filename order from val/  •  "
    f"overall subset accuracy {acc:.2f}%\n"
    "Descriptive analysis of the validation subset used for checkpoint selection — "
    "not independent test performance",
    fontsize=11.5, fontweight="bold", pad=16, linespacing=1.6,
)
plt.xlabel("Model Prediction", fontsize=12, fontweight="bold")
plt.ylabel("Real Label", fontsize=12, fontweight="bold")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

out = here / "Figure_8_real_50perclass_confusion_matrix.png"
plt.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out}")
print(f"n_total={d['n_total']}  n_per_class={n_per_class}  measured accuracy={d['accuracy_percent']:.2f}%")
