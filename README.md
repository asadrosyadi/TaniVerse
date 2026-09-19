# TaniVerse — Hybrid CNN–LSTM Edge-AIoT Framework for Rice Disease and Pest Classification with Microclimate Forecasting

This repository contains the full implementation behind the manuscript *"Hybrid CNN–LSTM Edge-AIoT Framework for Rice Disease and Pest Classification with Microclimate Forecasting"* (Bulletin of Electrical Engineering and Informatics). It combines:

- a **CNN** (EfficientNet-B0) that classifies rice leaf images into 10 disease/pest/normal categories,
- an **LSTM** that forecasts five microclimate variables (air temperature, relative humidity, soil pH, light intensity, VPD) from time-series sensor data, and
- a **weighted-sum feature-fusion model** that combines both branches into unified pest, plant-health, and microclimate predictions (Algorithm 1),

deployed on a Raspberry Pi 4 edge node with an ESP32 sensor/actuator front end and a Laravel dashboard/API backend.

For the full system architecture, mathematical formulation, and per-layer pseudocode, see [`ARSITEKTUR_SISTEM.md`](ARSITEKTUR_SISTEM.md). This README focuses on *what each part of the repository is, what data it uses, and how to run it* — the practical companion the manuscript's Section 3.3.1 refers to when it says implementation, configurations, and artifacts are available in the repository.

> **Provenance note:** several tables in the manuscript were produced by more than one exploratory script during development. This README and `ARSITEKTUR_SISTEM.md` document only the scripts that reproduce the manuscript's reported numbers; earlier scripts whose output did not match the final manuscript have been removed from the repository rather than left as ambiguous alternatives.

---

## 1. Repository structure

```
Final Program/
├── Arduino/
│   ├── padi_sawah/            # Production ESP32 field-node firmware (17 sensor channels, MQTT/REST actuation)
│   └── kirim_laravel/         # Minimal ESP32 → Laravel POST reference/test stub
│
├── Training_Penyakit Padi/    # CNN training pipeline + reproduces Table 1, Table 4, Figure 4(a), Figure 8
│   ├── train.py                              # EfficientNet-B0 two-stage transfer learning (core CNN trainer)
│   ├── compare_cnn_backbones.py              # Table 4: ResNet50 / MobileNetV2 / VGG16 / EfficientNet-B0
│   ├── compute_real_50perclass_matrix.py     # Figure 8 data: descriptive confusion matrix, 500-image subset
│   ├── plot_real_50perclass_confusion_matrix.py
│   ├── make_figure4_combined.py              # Figure 4(a)+(b) combined panel
│   ├── train/ val/                           # Paddy Doctor dataset, labeled, one class subfolder each (see §2)
│   ├── test/                                 # 3,469 unlabeled competition images, flat (no class subfolders); excluded from all metrics
│   └── efficientnetb0_model.h5, class_names.json
│
├── backend/
│   ├── python/
│   │   ├── detect_penyakit/    # Real-time CNN inference service + Table 7 benchmark
│   │   │   ├── app.py                        # Flask MJPEG inference service (Haar ROI + EfficientNet-B0), port 7000
│   │   │   └── benchmark_modern_models.py    # Table 7: 8-checkpoint classification comparison (incl. YOLOv11n-cls)
│   │   │
│   │   ├── rnn/                 # LSTM microclimate forecaster — reproduces Table 5, Figure 9, Figure 4(b)
│   │   │   ├── run.py                              # Flask forecasting service, port 5000
│   │   │   ├── make_synthetic_microclimate_dataset.py
│   │   │   ├── evaluate_lstm_microclimate_10k.py    # Core regression study (N=10,000, matches manuscript §3.2.2)
│   │   │   ├── generate_real_figure9_2025dates.py   # Table 5 / Figure 9 numbers (chronological holdout)
│   │   │   ├── rebuild_figure9_labeled.py           # Final Figure 9 render, (a)-(e) panel labels
│   │   │   ├── run_real_vpd_example.py              # Worked single-VPD-example paragraph (§4.2)
│   │   │   ├── make_figure4b_json_output.py         # Figure 4(b) JSON panel (reads the real trained LSTM)
│   │   │   └── app/                                 # Flask app package: services, routes, utils
│   │   │
│   │   └── hybrid_cnn_lstm/     # Fusion model + Algorithm 1 — reproduces Table 6, Table 2, §3.8
│   │       ├── fusion_model.py            # AdaptiveFeatureFusion layer + three-head model factory
│   │       ├── make_paired_dataset.py     # Synthetic timestamp-aligned image↔sensor manifest
│   │       ├── tune_fusion_weights.py     # Grid search over w1 → fusion_weights.json (manuscript optimum 0.6/0.4)
│   │       ├── train_hybrid.py            # Trains the deployed fusion model
│   │       ├── evaluate_hybrid.py         # Table 6: CNN-only / LSTM-only / concat / fusion ablation
│   │       ├── algorithm1.py              # Runnable Algorithm 1 (Table 2): fuse → threshold → actuate → transmit
│   │       └── actuation_eval.py          # §3.8 actuation success rate from an algorithm1.py run log
│   │
│   ├── laravel/                 # Laravel 12 REST API, MySQL persistence, Blade dashboard (Layer 3)
│   └── html/                    # Laravel's public/ web root (entry point served by the web server)
│
└── ARSITEKTUR_SISTEM.md         # Full architecture, per-layer math, pseudocode, and reproducibility tables
```

Two subsystems referenced in early manuscript drafts — a Raspberry Pi servo–LiDAR scanning node and a set of exploratory analysis scripts that did not reproduce the manuscript's final figures — have since been removed from the repository. See `ARSITEKTUR_SISTEM.md` §4 and its "2026 sync note" for that history.

---

## 2. Data sources

| Dataset | Source | Used by | Notes |
|---|---|---|---|
| Rice leaf images (10 classes) | Kaggle, *"Paddy Disease Classification all 480×640 px image"*, distributed by Bikram Saha — a dimension-corrected version of the Paddy Doctor competition dataset | `Training_Penyakit Padi/`, `backend/python/detect_penyakit/` | 10,407 labeled images total; see Table 1 below. An additional 3,469 competition-test images have no public ground truth and are **not** used for any reported metric. |
| Synthetic microclimate series | Generated independently by `make_synthetic_microclimate_dataset.py` (seed 42) | `backend/python/rnn/` | 10,000 records at 4-hour intervals: air temperature, relative humidity, soil pH, light intensity, VPD (VPD computed via the Tetens/Magnus formula). |
| *Plant Health Data* (Ziya, 2024, Kaggle) | `ziya07/plant-health-data` | Initial development reference only | Not used to train or evaluate any reported result — superseded by the synthetic dataset above. Still used as the field-mapping source for the Figure 4(b) JSON example (`make_figure4b_json_output.py`). |

**Table 1 class distribution** (source: Kaggle Paddy Doctor labels):

| Class | Images |
|---|---|
| Bacterial Leaf Blight | 479 |
| Bacterial Leaf Streak | 380 |
| Bacterial Panicle Blight | 337 |
| Blast | 1,738 |
| Brown Spot | 965 |
| Dead Heart | 1,442 |
| Downy Mildew | 620 |
| Hispa | 1,594 |
| Tungro | 1,088 |
| Normal | 1,764 |
| **Total** | **10,407** |

### Train / validation split

Labeled images are organized into class-specific subdirectories under `Training_Penyakit Padi/train/<class>/` and `Training_Penyakit Padi/val/<class>/`. The separate `Training_Penyakit Padi/test/` directory holds the 3,469-image competition test set as a **flat** folder with no class subdirectories and no public ground truth — it contributes nothing to `collect_all_images()`'s labeled pool and is excluded from every reported metric.

Three different partitioning schemes are used depending on the experiment — **do not assume one split applies everywhere**:

- **Table 7 and Figure 8**: both draw on the same fixed, as-shipped **8,323 training / 2,084 validation** data split (`train/` and `val/`, no repartitioning) — but they are **two different checkpoints**, not one. Figure 8 evaluates the Keras `efficientnetb0_model.h5` checkpoint (from `Training_Penyakit Padi/train.py`); Table 7's EfficientNet-B0 row evaluates a separate PyTorch/timm checkpoint (from `benchmark_modern_models.py`, §4.2). The 2,084-image validation partition guided checkpoint selection in both experiments. Table 7 reports metrics computed on the full validation partition, whereas Figure 8 presents a descriptive evaluation on a class-balanced subset of 500 images, comprising the first 50 images per class in sorted filename order. Neither evaluation represents independent test performance.
- **Table 4 (backbone comparison) and Table 6 (fusion ablation)**: all 10,407 images are pooled and re-partitioned **per fold** using stratified 5-fold cross-validation, ≈65% train / ≈15% val / ≈20% test per fold (≈6,765 / 1,561 / 2,081 images), seed 42. Within each fold, the ≈15% validation slice is used only for early stopping / checkpoint selection during that fold's training; the reported metric for that fold is computed on the separate ≈20% **test** slice, which the training process never saw. This is a genuine held-out evaluation, methodologically distinct from Table 7/Figure 8's single self-selecting validation set — see `compare_cnn_backbones.py`'s and `evaluate_hybrid.py`'s `stratified_kfold`/`tr_idx`/`te_idx` split for the exact mechanics.
- **LSTM regression (Table 5)**: the 10,000-record synthetic series yields 9,994 six-step windows, split **chronologically** (not shuffled) into 6,396 train / 1,599 val / 1,999 test windows.
- **Final deployed fusion model**: a separate 85/15 train/val split with no held-out test fold — this is the model actually shipped for inference, distinct from the Table 6 ablation's 5-fold runs above.

---

## 3. Setup

Each Python service has its own dependency set — there is no single top-level `requirements.txt`.

```bash
# CNN training pipeline (Training_Penyakit Padi/) — TensorFlow/Keras + scikit-learn
pip install tensorflow keras numpy pandas scikit-learn matplotlib seaborn opencv-python

# CNN inference service + benchmark
pip install -r backend/python/detect_penyakit/requirements.txt
pip install -r backend/python/detect_penyakit/requirements_experiment.txt   # adds torch/timm/ultralytics for benchmark_modern_models.py

# LSTM microclimate forecaster
pip install -r backend/python/rnn/requirements.txt

# Hybrid CNN–LSTM fusion model
pip install -r backend/python/hybrid_cnn_lstm/requirements.txt
```

```bash
# Laravel backend
cd backend/laravel
composer install
cp .env.example .env && php artisan key:generate
php artisan migrate
npm install && npm run build
```

Arduino firmware (`Arduino/padi_sawah/padi_sawah.ino`) is built with the Arduino IDE / arduino-cli, targeting an ESP32 board with the libraries listed in the sketch header (`WiFiManager`, `PubSubClient`, `ArduinoJson`, `DHT`, `BH1750`, `Adafruit_BMP280`).

---

## 4. How to run each experiment

All commands below assume you `cd` into the directory shown first. Every script that reports a manuscript number uses `--seed 42` (or a hardcoded `seed=42`) by default.

### 4.1 CNN classification — Table 4

```bash
cd "Training_Penyakit Padi"
python compare_cnn_backbones.py --cv-folds 5   # ResNet50 / MobileNetV2 / VGG16 / EfficientNet-B0, stratified 5-fold CV
```
Output: `table4_cnn_comparison.{md,csv,json}` and per-backbone `table4_fold_results_<Model>.json`.

`train.py` also supports a single stratified 70/15/15 split (`--cv-folds 0`, the default), which **pools and re-splits** `train/` + `val/` from scratch:
```bash
python train.py --cv-folds 0
```
**This command does not reproduce the existing Keras EfficientNet-B0 checkpoint evaluated in Figure 8.** The EfficientNet-B0 results in Table 7 were obtained using a separate PyTorch/timm checkpoint evaluated on the full 2,084-image validation partition — Figure 8 and Table 7's EfficientNet-B0 row are two different checkpoints, not one. The manuscript (Section 3.3.1) explicitly states the Figure 8 (Keras) checkpoint was trained on the fixed, as-shipped 8,323/2,084 `train/`/`val/` split, *not* on a 70/15/15 repartition — "although the training script supports a stratified 70/15/15 repartition, that configuration was not used to train the checkpoint evaluated in Figure 8." Running the command above trains a genuinely new model on a different split; it is documented here as a supported mode of `train.py`, not as a reproduction path for either shipped checkpoint. The current CLI has no flag that trains directly on `train/`/`val/` without pooling and re-splitting, so the original Figure 8 checkpoint's exact training invocation is not reproducible from the script as it stands today; the Table 7 PyTorch/timm checkpoint is instead produced by `benchmark_modern_models.py` (§4.2).

### 4.2 CNN validation benchmark (8 checkpoints) — Table 7

The script evaluates all eight checkpoints (YOLOv8n-cls, YOLOv11n-cls, EfficientNetV2-S, EfficientNet-B0, ViT-B/16, ResNet-50, MobileNet-V2, VGG-16) on the same 2,084-image validation partition, but **training protocols are not uniform across rows** — reproduce each family separately rather than in one batch run:

```bash
cd backend/python/detect_penyakit

# YOLOv8n-cls and YOLOv11n-cls: identical protocol to each other (5 epochs, batch 8, imgsz 224, CPU)
python benchmark_modern_models.py --dataset-root "/path/to/Training_Penyakit Padi" \
  --output-dir experiments/scopus_q1_comparison \
  --epochs 5 --batch-size 8 --img-size 224 --device cpu --seed 42 \
  --no-efficientnetv2 --no-efficientnetb0 --no-vit --no-resnet50 --no-mobilenetv2 --no-vgg16

# torchvision/timm rows: AdamW/cosine-schedule defaults (≤30 epochs, batch 16) — see below
python benchmark_modern_models.py --dataset-root "/path/to/Training_Penyakit Padi" \
  --output-dir experiments/scopus_q1_comparison \
  --img-size 224 --batch-size 16 --seed 42 --no-yolo --no-yolo11
```
The manuscript notes that the exact epoch count actually used for the stored torchvision/timm checkpoints is not recorded in the artifacts (default `--epochs 30`, patience 7, monitored on val macro-F1); treat any re-run of that branch as a fresh benchmark, not a byte-for-byte reproduction of Table 7's non-YOLO rows. Output: `model_comparison_summary.{csv,json,md,png}` plus a per-model subfolder with `<model>_metrics.json` and its confusion matrix. See `experiments/yolov11_addition/yolov11_provenance.md` for the exact provenance of the YOLOv11n-cls row (including a documented CUDA crash on the first GPU attempt, resolved by training on CPU).

### 4.3 Figure 8 — descriptive confusion matrix (500-image subset)

```bash
cd "Training_Penyakit Padi"
python compute_real_50perclass_matrix.py     # writes real_50perclass_confusion_matrix.json
python plot_real_50perclass_confusion_matrix.py
```

### 4.4 LSTM microclimate regression — Table 5 / Figure 9

```bash
cd backend/python/rnn
python make_synthetic_microclimate_dataset.py        # regenerates the 10,000-record series, if needed
python generate_real_figure9_2025dates.py             # chronological holdout eval; writes figure9_2025dates_predictions.csv
python rebuild_figure9_labeled.py                      # final Figure 9 PNG, reusing the CSV above (no retraining)
python run_real_vpd_example.py                          # the single worked VPD example quoted in §4.2
```

### 4.5 Hybrid CNN–LSTM feature fusion — Table 6, Table 2, §3.8

```bash
cd backend/python/hybrid_cnn_lstm
python make_paired_dataset.py                 # synthetic image↔sensor manifest (seed 42)
python tune_fusion_weights.py                 # grid search w1 in {0, 0.1, ..., 1} -> fusion_weights.json (expect w1=0.6)
python train_hybrid.py --fusion adaptive      # trains the deployed model -> hybrid_cnn_lstm_model.h5
python evaluate_hybrid.py                     # Table 6 ablation -> table6_hybrid_comparison.{md,json}
python algorithm1.py --dry-run --limit 10     # runnable Algorithm 1, writes algorithm1_run_log.json
python actuation_eval.py                      # §3.8 actuation success rate -> actuation_success_rate.json
```

### 4.6 Figure 4 (augmentation panel + JSON forecast example)

```bash
cd backend/python/rnn
python make_figure4b_json_output.py           # runs the real trained LSTM -> figure4b_real_forecast_entry.json
cd "../../../Training_Penyakit Padi"
python make_figure4_combined.py                # reads that JSON + train.py's augmentation pipeline -> Figure_4_real_combined.{png,pdf}
```

### 4.7 Running the live services

```bash
# CNN inference service (port 7000)
cd backend/python/detect_penyakit && python app.py

# LSTM forecasting service (port 5000)
cd backend/python/rnn && python run.py

# Laravel dashboard/API
cd backend/laravel && php artisan serve
```

---

## 5. Result files ↔ manuscript table/figure map

| Manuscript item | Producing script(s) | Key output file(s) |
|---|---|---|
| Table 1 (class distribution) | — (static dataset labels) | `Training_Penyakit Padi/{train,val}/<class>/` counts (10,407 labeled images; excludes the unlabeled `test/`) |
| Table 2 (Algorithm 1) | `hybrid_cnn_lstm/algorithm1.py` | `algorithm1_run_log.json` |
| Table 3 (training config) | — (documented, not generated) | see `ARSITEKTUR_SISTEM.md` §11 |
| Table 4 (CNN backbone comparison) | `Training_Penyakit Padi/compare_cnn_backbones.py` | `table4_cnn_comparison.{md,csv,json}` |
| Table 5 (LSTM microclimate regression) | `rnn/generate_real_figure9_2025dates.py` | `figure9_2025dates_predictions.csv` |
| Table 6 (fusion ablation) | `hybrid_cnn_lstm/evaluate_hybrid.py` | `table6_hybrid_comparison.{md,json}` |
| Table 7 (8-checkpoint validation benchmark) | `detect_penyakit/benchmark_modern_models.py` | `experiments/*/model_comparison_summary.{csv,json,md}` |
| Figure 4 (augmentation + JSON panel) | `rnn/make_figure4b_json_output.py` → `Training_Penyakit Padi/make_figure4_combined.py` | `Figure_4_real_combined.{png,pdf}` |
| Figure 8 (500-image confusion matrix) | `Training_Penyakit Padi/{compute_real_50perclass_matrix,plot_real_50perclass_confusion_matrix}.py` | `Figure_8_real_50perclass_confusion_matrix.png` |
| Figure 9 (LSTM predicted vs. actual) | `rnn/rebuild_figure9_labeled.py` (reuses the Table 5 CSV) | `Figure_9_real_lstm_2025dates_HD_labeled.png` |
| §3.8 actuation success rate | `hybrid_cnn_lstm/actuation_eval.py` | `actuation_success_rate.json` |

Deployed model checkpoints: `Training_Penyakit Padi/efficientnetb0_model.h5` (CNN), `backend/python/rnn/config/plant_health_lstm_model.h5` (production LSTM), `backend/python/hybrid_cnn_lstm/hybrid_cnn_lstm_model.h5` (fusion model, produced by `train_hybrid.py`).

---

## 6. Evaluation results and artifact status

This repository contains more than one evaluation configuration for CNN classification. Their numbers must be read together with the checkpoint, data partition, and protocol that produced them — they are **not interchangeable**, and mixing them up is a real, recurring source of confusion (see below).

| Evaluation | Result | Evidence |
|---|---|---|
| 500-image descriptive subset (manuscript Figure 8) | **43.40% accuracy (217/500)** | `Training_Penyakit Padi/real_50perclass_confusion_matrix.json`, produced by `compute_real_50perclass_matrix.py` running the shipped `efficientnetb0_model.h5` |
| EfficientNet-B0, stratified 5-fold CV (manuscript Table 4 row) | **38.73% mean accuracy, std 1.63 pp** (fold accuracies: 39.11%, 37.72%, 41.21%, 36.35%, 39.29%) | `Training_Penyakit Padi/table4_fold_results_EfficientNet-B0.json`, produced by `compare_cnn_backbones.py`; the completed run is also visible in `table4_cpu_run.log` (`=> acc 38.7 +/- 1.6 | P 0.47 R 0.40 F1 0.37`, `[saved] table4_fold_results_EfficientNet-B0.json`) |

**This is a real discrepancy worth flagging plainly: the only completed, saved Table 4 run for EfficientNet-B0 currently in this repository reports 38.73% ± 1.63%, not the 89.2 ± 0.6% the manuscript's Table 4 reports for this row.** This was verified directly, not assumed:
- `table4_gpu_run.log` shows a separate GPU attempt that crashed with `CUDA_ERROR_ILLEGAL_ADDRESS` during fold 1 and never completed or saved a result (the same class of CUDA/cuDNN instability documented elsewhere in this repo for the YOLOv11 benchmark — see `experiments/yolov11_addition/yolov11_provenance.md`).
- `table4_cpu_run.log` shows the CPU run completing all 5 EfficientNet-B0 folds exactly once, printing the same 38.7±1.6 summary now stored in the JSON.
- No `table4_cnn_comparison.{md,csv,json}` (the script's final combined-table output across all four backbones) exists, confirming the full multi-model run was never completed end-to-end.
- No other file in the repository contains a different, completed EfficientNet-B0 Table-4 result.

Whether the 89.2 ± 0.6% figure in the manuscript came from a run whose artifacts were lost, from a different configuration than what's in `compare_cnn_backbones.py` today, or needs correcting, has not been established — only that the artifact currently in this repository does not reproduce it. Anyone trying to reconcile this should re-run `compare_cnn_backbones.py --cv-folds 5` end-to-end and compare against both the manuscript's number and the 38.73% recorded here, rather than assuming either figure without checking.

**A third number was proposed during project discussion but is explicitly excluded from this table: 89.20% accuracy (446/500) read off a hand-drawn confusion matrix image.** That image reproduces the `FIG8_AFTER` matrix that used to be hand-typed into the now-deleted `generate_confusion_matrix.py` (see the "2026 sync note" in `ARSITEKTUR_SISTEM.md` and this repository's commit history for that removal) — its own docstring predecessor, `compute_real_50perclass_matrix.py`, states outright that it exists to replace "hand-typed... numbers" with "the measured result." No checkpoint, prediction log, or evaluation run in this repository produces that matrix; a from-scratch retraining attempt following the manuscript's documented protocol reached 33.60% (168/500) on the same subset instead. It is not included above because it is not an evaluation result — it has no corresponding evaluated checkpoint, and none should be invented to match it after the fact.

### If you want to add a new, verified evaluation result

Any future update to Figure 8 (or any other confusion-matrix figure) should be traceable end-to-end: **checkpoint → predictions on a defined image list → confusion matrix → metrics/figure**, all generated by one script run, not hand-assembled. At minimum, record:

- the evaluated checkpoint's identifier (path, and ideally a content hash);
- the evaluation command and software versions used;
- the preprocessing configuration and class-index mapping;
- the evaluation partition and the exact list of evaluated image files;
- per-image ground-truth and predicted labels;
- a confusion-matrix JSON and metrics file generated from those same predictions (as `compute_real_50perclass_matrix.py` already does).

Keep older evaluation artifacts (e.g. `real_50perclass_confusion_matrix.json`, `table4_fold_results_EfficientNet-B0.json`) identifiable as separate, distinct runs rather than overwriting them in place — this table exists specifically so a reader can tell which number came from which run.

**Table 4 and Figure 8 are separate evaluations and must stay that way.** The manuscript's Table 4 reports cross-validation fold-level results; Figure 8 reports a single confusion-matrix evaluation on a different, fixed checkpoint and data split (see §2's "Table 7 and Figure 8" note for how Figure 8's own checkpoint differs again from Table 7's). Regenerating or correcting one does not automatically change the other, and neither should be edited to match a number that didn't come from running it.

---

## 7. Reproducibility notes and known limitations

These are documented in the manuscript (Section 3.3.1 and 4.8) and carried through in the code:

- **Preprocessing leakage in the LSTM regression:** the microclimate MinMax scalers are fitted on the entire 10,000-record series *before* windowing/partitioning, so Table 5/Figure 9 numbers should be read as synthetic experimental results rather than leakage-free unseen-data estimates.
- **Label-conditioned synthetic pairing:** `make_paired_dataset.py` generates microclimate windows *conditioned on the image's true class* (healthy → low-VPD bias, diseased → high-VPD bias). Table 6's fusion gains may therefore partly reflect this construction rather than independently acquired environmental information.
- **Validation-guided checkpoint selection (Table 7, Figure 8 only):** the classification results in Table 7 and Figure 8 were calculated on validation data that also guided checkpoint selection and therefore do not represent independent test performance. Table 4 and Table 6, by contrast, report each fold's metric on that fold's held-out test slice (`te_idx`), which is disjoint from the inner validation slice used for early stopping within the fold — a methodologically distinct (though still same-dataset, not cross-dataset) evaluation. See the "Train / validation split" note in §2 for the exact mechanics of both.
- **Non-uniform training protocols in Table 7:** the YOLOv8n-cls/YOLOv11n-cls rows share one identical Ultralytics protocol with each other, but differ from the torchvision/timm rows in optimizer, augmentation, and (for several rows) undocumented epoch counts.

See `ARSITEKTUR_SISTEM.md` for the complete mathematical formulation, per-layer pseudocode, and full reproducibility hyperparameter tables (§11) referenced throughout this README.