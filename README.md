# TaniVerse: CNN–LSTM Edge-AIoT for Rice Monitoring

TaniVerse contains software associated with the manuscript **Hybrid CNN–LSTM Edge-AIoT Framework for Rice Disease and Pest Classification with Microclimate Forecasting**, prepared for the *Bulletin of Electrical Engineering and Informatics*. The project integrates artificial intelligence of things (AIoT), image classification, environmental time-series modeling, monitoring, and control interfaces.

The principal components are:

- An EfficientNet-B0 convolutional neural network (CNN) for ten rice image classes covering disease, pest-related damage, and normal conditions.
- A long short-term memory (LSTM) regression experiment predicting air temperature, relative humidity, soil pH, light intensity, and vapor pressure deficit (VPD).
- A weighted-sum CNN–LSTM fusion implementation with image-class, health-status, and microclimate outputs.
- ESP32 sensing/relay firmware, Python inference services, and a Laravel application programming interface (API) and dashboard.

The manuscript describes a Raspberry Pi 4 deployment. A component's presence in the repository does not establish that it was deployed or evaluated in the field. The standalone regression experiment, online health-status service, and fusion model are distinct implementations.

**Documentation scope.** This revision uses the uploaded README and selected source/artifact checks against repository commit `89151d592b388795345e9bac0dd4aa5d57058f5f`. Evidence statements below refer to that reviewed snapshot; later runs require their own records. No training, field trial, or device benchmark was executed as part of this documentation revision.

**Evidence policy.** Describe supported code paths separately from completed experiments. Preserve results even when they differ from manuscript values. A reproducible result requires a defined data partition, configuration, checkpoint, and evaluation output. A matching number in a figure or script comment is insufficient.

For additional architecture details, see [ARSITEKTUR_SISTEM.md](ARSITEKTUR_SISTEM.md). That companion document requires separate reconciliation where its claims conflict with source code or evaluation artifacts; it is not a substitute for those records.

## Contents

1. [Repository components](#1-repository-components)
2. [Data and evaluation partitions](#2-data-and-evaluation-partitions)
3. [Environment setup](#3-environment-setup)
4. [Running experiments and services](#4-running-experiments-and-services)
5. [Manuscript-to-artifact map](#5-manuscript-to-artifact-map)
6. [Recorded results and unresolved discrepancies](#6-recorded-results-and-unresolved-discrepancies)
7. [Methodological and operational limitations](#7-methodological-and-operational-limitations)
8. [Recording a new evaluation](#8-recording-a-new-evaluation)

## 1. Repository components

Paths are relative to the repository root. Dataset files, pretrained weights, and generated outputs may need to be obtained or produced separately.

| Path | Purpose |
|---|---|
| `Arduino/padi_sawah/padi_sawah.ino` | ESP32 telemetry and relay-control firmware; documented payload includes 17 channels/variables, not necessarily 17 separate sensors. |
| `Arduino/kirim_laravel/kirim_laravel.ino` | Minimal telemetry submission example. |
| `Training_Penyakit Padi/train.py` | Keras EfficientNet-B0 training: fixed train/validation, repartitioned holdout, or cross-validation modes. |
| `Training_Penyakit Padi/compare_cnn_backbones.py` | Keras backbone comparison workflow associated with manuscript Table 4. |
| `Training_Penyakit Padi/compute_real_50perclass_matrix.py` | Evaluation of a selected checkpoint on 50 validation images per class. |
| `Training_Penyakit Padi/plot_real_50perclass_confusion_matrix.py` | Plotting of the corresponding confusion-matrix data. |
| `Training_Penyakit Padi/make_figure4_combined.py` | Image-augmentation panel and system JSON reporting example. |
| `backend/python/detect_penyakit/app.py` | Camera/inference service, including Haar-based region proposals and image classification. |
| `backend/python/detect_penyakit/benchmark_modern_models.py` | PyTorch/timm and Ultralytics classification benchmark. |
| `backend/python/rnn/` | Offline synthetic regression experiments and a separate online forecasting/health-status service. |
| `backend/python/hybrid_cnn_lstm/` | Synthetic pairing, fusion model, weight selection, training, ablation evaluation, and inference/control logic. |
| `backend/laravel/` | Laravel API, persistence, dashboard, and recommendation services. |
| `backend/html/` | Web assets and entry-point files; configure the served directory according to the deployment. |
| `ARSITEKTUR_SISTEM.md` | Companion technical description, subject to source/artifact reconciliation. |

The historical servo–LiDAR scanning node is not an active component in the reviewed snapshot. Legacy routes or polling code do not establish an operational LiDAR data source. Do not infer depth-map input to the CNN from historical LiDAR references.

## 2. Data and evaluation partitions

### 2.1 Data sources

| Dataset | Source | Role |
|---|---|---|
| Rice images | [Paddy Doctor image distribution on Kaggle](https://www.kaggle.com/datasets/imbikramsaha/paddy-doctor) | 10,407 labeled images across ten classes. The separate 3,469-image competition test directory lacks public ground-truth labels. |
| Synthetic microclimate series | `backend/python/rnn/make_synthetic_microclimate_dataset.py` | 10,000 records at four-hour intervals, with five variables including VPD derived from temperature and humidity. This is synthetic data, not a field sensor record. |
| Plant Health Data | [Kaggle development reference](https://www.kaggle.com/datasets/ziya07/plant-health-data) | Development/field-mapping reference described in the original documentation. Its role must be distinguished from the synthetic regression data and the training provenance of the separate online model. |
| Classification notebook | [Kaggle implementation reference](https://www.kaggle.com/code/mhassaan1122/rice-disease-classification-99-accuracy) | Development reference; its title or reported accuracy does not establish this repository's performance. |

Class counts reported for the labeled rice dataset:

| Class | Images |
|---|---:|
| Bacterial leaf blight | 479 |
| Bacterial leaf streak | 380 |
| Bacterial panicle blight | 337 |
| Blast | 1,738 |
| Brown spot | 965 |
| Dead heart | 1,442 |
| Downy mildew | 620 |
| Hispa | 1,594 |
| Normal | 1,764 |
| Tungro | 1,088 |
| **Total** | **10,407** |

The labeled layout is `Training_Penyakit Padi/train/<class>/` and `Training_Penyakit Padi/val/<class>/`, with 8,323 and 2,084 images, respectively. The competition `test/` directory is flat and unlabeled; it cannot supply a supervised confusion matrix. Save the actual class-index mapping and partition indices for each run.

### 2.2 Separate evaluation configurations

| Experiment | Partition and interpretation |
|---|---|
| Table 7 classification benchmark | Fixed 8,323-image training and 2,084-image validation partitions. Validation also guides checkpoint selection; reported scores are not independent-test performance. |
| Existing Figure 8 evaluation | Keras checkpoint evaluated on the first 50 images per class in sorted filename order from `val/`, totaling 500 images. This is a descriptive validation subset. It uses a different checkpoint/implementation from Table 7's PyTorch/timm EfficientNet-B0 row. |
| CNN repartitioned holdout | `train.py --cv-folds 0` pools labeled images and creates a new stratified 70/15/15 partition. It is not the fixed-split Figure 8 protocol. |
| CNN cross-validation | Five outer folds with approximately 20% test data each. In the reviewed `train.py`, an inner validation fraction of `0.15 / 0.8` leaves approximately 65% training, 15% validation, and 20% test overall. Preserve exact indices and verify the backbone-comparison script's corresponding split. |
| Fusion ablation | In the reviewed `evaluate_hybrid.py`, each outer training pool reserves `val_frac=0.2` for validation: approximately 64% training, 16% validation, and 20% test overall. This differs from the CNN split. Synthetic label conditioning remains a limitation even with disjoint indices. |
| Standalone LSTM regression | 9,994 six-step windows from 10,000 records; chronological split of 6,396 training, 1,599 validation, and 1,999 test windows. The recorded full-series scaling introduces preprocessing leakage. |
| Final fusion training | A separate 85/15 train/validation configuration is described in project documentation. Deployment identity and execution must be supported by the corresponding run records; this is not an ablation test result. |

Six observations spaced four hours apart span **20 hours between the first and last input observations**. The next-step target is four hours after the last observation. Do not label this as a 24-hour input timestamp span.

## 3. Environment setup

Use separate Python virtual environments for TensorFlow/Keras and PyTorch/Ultralytics services where their dependencies differ. Dependency installation is not a record of the versions used by historical checkpoints; retain an environment export for each new run.

From the repository root, install the requirements for the component being used:

```bash
# Standalone Keras training dependencies; pin compatible versions for a recorded run.
python -m pip install tensorflow keras numpy pandas scikit-learn matplotlib seaborn opencv-python

# CNN inference and benchmark environments, as appropriate.
python -m pip install -r backend/python/detect_penyakit/requirements.txt
python -m pip install -r backend/python/detect_penyakit/requirements_experiment.txt

# Online LSTM service.
python -m pip install -r backend/python/rnn/requirements.txt

# Fusion experiments.
python -m pip install -r backend/python/hybrid_cnn_lstm/requirements.txt
```

Laravel setup, from the repository root:

```bash
cd backend/laravel
composer install
cp .env.example .env
php artisan key:generate
# Set database and service configuration in .env before migrating.
php artisan migrate
npm install
npm run build
```

Configure device identifiers, credentials, service addresses, and MQTT settings for the target installation. Keep secrets outside version control. Compile the ESP32 sketch using the board configuration and libraries required by its source, including WiFiManager, PubSubClient, ArduinoJson, DHT, BH1750, and Adafruit_BMP280.

## 4. Running experiments and services

Start **each command block from the repository root** unless otherwise stated. Commands below describe supported workflows, not guaranteed reproduction of historical scores. Check `--help` before a new run. Record the resolved configuration, including framework-specific defaults; a top-level seed does not establish the effective seed of every training backend.

Some scripts write fixed filenames. Use an isolated working copy or archive existing checkpoints and outputs before retraining or regenerating results. Existing evaluation artifacts must not be overwritten to make them agree with manuscript values.

### 4.1 CNN training and backbone comparison

```bash
cd "Training_Penyakit Padi"
python compare_cnn_backbones.py --cv-folds 5
```

The comparison workflow is intended to produce `table4_cnn_comparison.{md,csv,json}` and per-backbone `table4_fold_results_<Model>.json`. The existence of the script does not establish a completed comparison for all backbones. Section 6 identifies the recorded EfficientNet-B0 result.

To train a new Keras model on the fixed directories:

```bash
cd "Training_Penyakit Padi"
python train.py --fixed-split --cv-folds 0 --seed 42 --output efficientnetb0_fixed_split_new.h5
```

The reviewed trainer supports `--fixed-split` and writes `fixed_split_results.json`. Validation is used for both checkpoint selection and reported evaluation in this mode. A newly trained checkpoint is not automatically the checkpoint used by an earlier Figure 8.

For a new stratified 70/15/15 holdout experiment:

```bash
cd "Training_Penyakit Padi"
python train.py --cv-folds 0 --seed 42 --output efficientnetb0_holdout_new.h5
```

Never evaluate an existing checkpoint on a newly repartitioned test set without establishing that those images were excluded from its training and selection history.

### 4.2 Classification benchmark associated with Table 7

Eight model families are represented: YOLOv8n-cls, YOLO11n-cls (called YOLOv11n-cls in the manuscript), EfficientNetV2, EfficientNet-B0, ViT, ResNet50, MobileNetV2, and VGG16. Their training protocols are not uniform.

Example fresh YOLO-family benchmark:

```bash
cd backend/python/detect_penyakit
python benchmark_modern_models.py --dataset-root "/absolute/path/to/Training_Penyakit Padi" \
  --output-dir "/absolute/path/to/new_yolo_run" \
  --epochs 5 --batch-size 8 --img-size 224 --device cpu --num-workers 0 --seed 42 \
  --no-efficientnetv2 --no-efficientnetb0 --no-vit --no-resnet50 --no-mobilenetv2 --no-vgg16
```

Use actual absolute paths. The historical YOLO runs are documented as using five epochs, batch size eight, 224-pixel inputs, and CPU execution. Capture Ultralytics' resolved `args.yaml`, selected optimizer, augmentations, seed, version, and evaluated checkpoint; identical headline settings do not prove every effective setting is identical.

Example fresh torchvision/timm benchmark:

```bash
cd backend/python/detect_penyakit
python benchmark_modern_models.py --dataset-root "/absolute/path/to/Training_Penyakit Padi" \
  --output-dir "/absolute/path/to/new_timm_run" \
  --epochs 30 --img-size 224 --batch-size 16 --seed 42 --no-yolo --no-yolo11
```

The actual historical epoch counts for several stored non-YOLO checkpoints were not recorded in the supplied documentation. These commands therefore initiate new comparisons rather than guarantee reproduction of existing numbers. Keep YOLO and timm preprocessing descriptions separate.

Historical outputs are under `experiments/scopus_q1_comparison/` and `experiments/yolov11_addition/`. See their per-model metrics and provenance documents. A function selecting the newest checkpoint is not sufficient evidence that the same checkpoint produced an older metrics file.

### 4.3 Figure 8: descriptive validation-subset evaluation

```bash
cd "Training_Penyakit Padi"
python compute_real_50perclass_matrix.py
python plot_real_50perclass_confusion_matrix.py
```

Before running, inspect the checkpoint path, class mapping, and output paths in these scripts. The recorded historical result is 217 correct predictions out of 500 images (43.40%). Regenerate the JSON and plot from the same predictions when evaluating a different checkpoint. Do not assume a fresh run will reproduce either 43.40% or 89.20%.

### 4.4 Synthetic LSTM regression and Figure 9

```bash
cd backend/python/rnn
# Only regenerate the dataset in a separate run when needed.
python make_synthetic_microclimate_dataset.py
python generate_real_figure9_2025dates.py
python rebuild_figure9_labeled.py
python run_real_vpd_example.py
```

Archive the source dataset and existing prediction CSV first. `rebuild_figure9_labeled.py` plots the stored `figure9_2025dates_predictions.csv`; plotting is distinct from retraining. The figure compares synthetic reference values with predictions, not measured field values.

The historical regression results use full-series scaling before partitioning. Correcting this leakage requires retraining and reevaluation; changing this README cannot make existing metrics leakage-free. Any 14-day rolling evaluation must have its own predictions and metrics and must not reuse the full-holdout metrics as period-specific results.

### 4.5 Fusion experiments and decision-logic checks

```bash
cd backend/python/hybrid_cnn_lstm
python make_paired_dataset.py
python tune_fusion_weights.py
python train_hybrid.py --fusion adaptive
python evaluate_hybrid.py
```

The CLI value `adaptive` and class name `AdaptiveFeatureFusion` are implementation identifiers. The described fusion operation uses **fixed weighted-sum coefficients during inference**. Record the coefficients selected by the actual tuning run; do not require the search to return 0.6/0.4 merely because those values appear in the manuscript. Concatenation is an alternative fusion baseline, not a no-fusion model.

The paired windows are generated conditionally on image labels. Their assigned timestamps do not establish real field synchronization or independent environmental evidence.

To inspect inference/decision logic without a field trial:

```bash
cd backend/python/hybrid_cnn_lstm
python algorithm1.py --dry-run --limit 10
python actuation_eval.py --run-log algorithm1_run_log.json --out decision_rule_agreement.json
```

The reviewed `actuation_eval.py` compares logged action names with rule-derived expected actions using information in the same log. It measures **decision-rule agreement**, despite its output key `success_rate_percent`. It does not verify relay switching, water delivery, pest suppression, uptime, or the manuscript's 30-day field-trial success rate. Physical actuation success requires independent observations linked to individual field events.

### 4.6 Figure 4: augmentation and reporting example

```bash
cd backend/python/rnn
python make_figure4b_json_output.py
cd "../../../Training_Penyakit Padi"
python make_figure4_combined.py
```

Use one original training image and transformations produced by the actual augmentation pipeline. Keep the image identifier, transform parameters, and random seeds. The JSON panel is a system reporting example; its values are not normalized model-input tensors, and additional fields are not necessarily regression outputs. Preserve the provenance of the JSON example separately from field measurements and model evaluation.

### 4.7 Live services

Run each service in a separate terminal from the repository root:

```bash
# Terminal 1: CNN service, documented port 7000.
cd backend/python/detect_penyakit
python app.py
```

```bash
# Terminal 2: online forecasting service, documented port 5000.
cd backend/python/rnn
python run.py
```

```bash
# Terminal 3: Laravel local development server.
cd backend/laravel
php artisan serve
```

These are service startup instructions, not a complete production deployment. Verify configured endpoints, hardware connections, and model files. Identify which inference services actually execute on the Raspberry Pi and which execute on a server before describing a deployment as entirely edge-based.

## 5. Manuscript-to-artifact map

Table and figure numbers refer to the manuscript version discussed during this review. Expected filenames indicate workflow outputs, not a certification that all files exist or reproduce the manuscript.

| Manuscript item | Workflow or artifact | Evidence boundary |
|---|---|---|
| Table 1 | Labeled dataset counts | Exclude the unlabeled competition directory. |
| Table 2 / Algorithm 1 | `backend/python/hybrid_cnn_lstm/algorithm1.py` | Source and execution logs establish implemented logic; dry-runs do not establish field outcomes. |
| Table 3 | Run configurations and environment exports | Distinguish defaults, resolved run settings, and deployment settings. |
| Table 4 | `table4_fold_results_<Model>.json`; optional combined comparison outputs | Reviewed EfficientNet-B0 result differs from manuscript values; other rows require their own fold records. |
| Table 5 / Figure 9 | `backend/python/rnn/figure9_2025dates_predictions.csv` | Recalculate metrics from the same stored targets/predictions; retain synthetic-data and scaling limitations. |
| Table 6 | `backend/python/hybrid_cnn_lstm/evaluate_hybrid.py`; expected `table6_hybrid_comparison.{md,json}` | Script availability alone does not verify reported ablation scores or latency. |
| Table 7 | Per-model metrics under `experiments/scopus_q1_comparison/` and `experiments/yolov11_addition/` | Link each metric to the exact evaluated checkpoint and full validation partition. |
| Figure 4 | `make_figure4_combined.py` and `figure4b_real_forecast_entry.json` | Experimental augmentation and system reporting example; no generative reconstruction of data. |
| Figure 8 | `real_50perclass_confusion_matrix.json` and its plotted figure | Historical 500-image validation result is 43.40%; supplied 89.20% figure requires reconciliation. |
| Section 3.8 field actuation | Independent event-level field records | `actuation_eval.py` alone does not substantiate 116 successful physical events out of 120. |

Checkpoint names referenced by the project include `Training_Penyakit Padi/efficientnetb0_model.h5`, `backend/python/rnn/config/plant_health_lstm_model.h5`, and the fusion training output `hybrid_cnn_lstm_model.h5`. File naming alone does not establish deployment or linkage to published metrics.

## 6. Recorded results and unresolved discrepancies

### 6.1 CNN evidence available in the reviewed snapshot

| Evaluation | Recorded or displayed value | Status |
|---|---|---|
| Historical Figure 8 validation subset | 43.40% accuracy: 217/500 | Recorded in `Training_Penyakit Padi/real_50perclass_confusion_matrix.json`. |
| EfficientNet-B0 five-fold run | Mean accuracy 38.73%; stored SD 1.63 percentage points | Recorded in `Training_Penyakit Padi/table4_fold_results_EfficientNet-B0.json`. |
| Separately supplied 500-image figure | 89.20% accuracy: 446/500 | Arithmetic derived from the displayed matrix; no matching evaluated-checkpoint/prediction linkage was established in the reviewed artifacts. |
| Manuscript Table 4 EfficientNet-B0 claim | 89.2 ± 0.6% | Not supported by the reviewed five-fold file. |

The five recorded fold accuracies are 39.11%, 37.72%, 41.21%, 36.35%, and 39.29%. The reviewed training summarizer uses `ddof=0` for SD. Specify this convention; a population SD across the five observed scores differs from the sample SD obtained with `ddof=1`. Approximate confidence intervals generated from fold scores should not be described as independent-test guarantees because training sets overlap across folds.

No complete four-backbone result set was established in the prior artifact review. Absence of a file in that snapshot does not prove an experiment never ran elsewhere. Supply the missing run records or conduct a new documented comparison before retaining unsupported manuscript numbers.

The uploaded documentation attributes the 89.20% figure to an earlier manually specified matrix and mentions another 33.60% run. Those provenance claims were not independently established in this revision. Neither claim is treated here as a verified new evaluation. Resolve figure provenance through source history and prediction records, without constructing predictions to match a desired score.

### 6.2 Fusion and regression results

The manuscript's reported fusion accuracy of 93.6%, F1-score of 0.93, and associated improvements remain unverified against complete ablation run artifacts in this review. Label-conditioned pairing disclosure does not replace evidence that a reported experiment produced those values.

The documented standalone regression values are synthetic chronological-holdout results. Their interpretation remains limited by preprocessing leakage. Neither those metrics nor the image-subset metrics establish field predictive accuracy.

### 6.3 Benchmark and latency interpretation

Table 7 compares validation checkpoints, including a PyTorch/timm EfficientNet-B0 distinct from Figure 8's Keras checkpoint. Its scores must not replace cross-validation or fusion-baseline results. Low checkpoint scores alone do not establish training divergence or an architectural performance limit; inspect predictions and training records before assigning causes.

The supplied benchmark values give YOLO11n-cls 90.93% accuracy and 89.90% macro F1, versus EfficientNetV2's 86.90% and 84.91%. The differences are **4.03 and 4.99 percentage points**, respectively. These validation metrics alone do not establish deployment efficiency.

The manuscript's 92 ms CNN and 110 ms fusion figures concern different model configurations. Retain them as measured device results only with hardware, runtime, precision, batch size, warm-up, repetitions, and timing boundaries recorded. Do not infer Raspberry Pi latency from a laptop run or infer TensorFlow Lite latency from Keras `model.predict()` timing.

In the reviewed fusion evaluator, latency is measured within each fold and the summarizer aggregates fold-level latency values with a mean and SD. Identify both aggregation levels before comparing that output with a manuscript table reporting a single median. Prediction-call timing excludes acquisition, preprocessing, communication, and physical actuation unless explicitly included.

## 7. Methodological and operational limitations

- **Regression preprocessing leakage:** historical input/target scalers were fitted on the entire synthetic series before partitioning. A corrected experiment must fit them on training data only, then reevaluate.
- **Label-conditioned pairing:** synthetic environmental windows encode information derived from the image target labels. This limits claims about independently measured environmental benefits. Health-category thresholds must also be learned from training data only in a corrected evaluation.
- **Validation reuse:** Table 7 and historical Figure 8 use validation data involved in model selection. Distinct outer test folds are a different protocol, but their independence still depends on actual training, tuning, preprocessing, and duplicate/group handling.
- **Tuning versus evaluation:** fusion-weight selection must not use outer test scores. Save tuning/evaluation indices and use an inner selection procedure when reporting outer-fold results.
- **Online versus offline models:** the five-target regression experiment and the online health classifier are different. Rule-generated environmental trajectories with repeated predictions on an unchanged historical sequence must not be presented as recursively updated LSTM forecasts.
- **Fallback values:** zero-filled response templates indicate unavailable output, not measured zero conditions or valid predictions. Downstream reporting/control should distinguish missing data from valid observations.
- **Operational claims:** local code, MQTT/REST messages, and rule agreement do not demonstrate continuous operation or successful physical intervention. Field event logs and downtime records are separate evidence.
- **Historical components:** retained LiDAR routes or removed scripts must be documented through version history, not represented as active or removed merely because their scores differ from manuscript values.

## 8. Recording a new evaluation

For each run, preserve:

1. Git commit, run identifier, date, hardware, and exported software environment.
2. Dataset source/version or checksum, class-index mapping, and exact partition/image lists.
3. Training and preprocessing configuration, seeds, augmentation settings, and checkpoint-selection criterion.
4. Checkpoint path and SHA-256 hash, including any conversion or optimization history.
5. Ground-truth and predicted labels, or timestamped regression targets and predictions.
6. Metrics calculated from those records, with averaging, units, and SD conventions stated.
7. Figures generated from the same records and the command used to produce them.
8. For physical actuation, independent observations of the actual response and the event success criteria.

Use separate run directories, for example:

```text
# Proposed organization for future runs; not a claim that these folders already exist.
evaluation_results/
    historical_validation_500/
    efficientnet_b0_cv_<run_id>/
    updated_validation_500_<run_id>/
    fusion_ablation_<run_id>/
```

Update manuscript tables, narrative claims, figure captions, and this README from the same verified records. An updated Figure 8 does not automatically update Tables 4, 6, or 7. Keep historical results traceable and document corrections explicitly.
