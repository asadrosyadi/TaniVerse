# YOLOv11 addition — provenance for Section 4.4 / Table 6 (reviewer comment 4)

Added in response to reviewer comment 4 ("newer benchmark model not added"). This
document has the same structure as `experiments/scopus_q1_comparison/yolov8_baseline_provenance.md`
so the two YOLO rows in Table 6 can be checked against each other line by line.

## Variant and checkpoint

| Item | Value | Evidence |
|---|---|---|
| Task | `classify` (image classification, **not** detection) | `yolo_runs/yolo11_cls/args.yaml` -> `task: classify` |
| Variant | **nano** (`n`) | `model: yolo11n-cls.pt` in `args.yaml`; matches the nano variant used for the YOLOv8 baseline |
| Starting checkpoint | `yolo11n-cls.pt` (Ultralytics ImageNet-pretrained classification weights), `pretrained: true` | `args.yaml` |
| Fine-tuned checkpoint used for the reported metrics | `yolo_runs/yolo11_cls/weights/best.pt` | trained fresh in this run; `train_args.epochs == 5` and `names` (10 classes) verified by loading the checkpoint directly |
| Classes in the checkpoint | 10 (the ten study labels, same label set/order as the other checkpoints) | class names embedded in `best.pt`: bacterial_leaf_blight, bacterial_leaf_streak, bacterial_panicle_blight, blast, brown_spot, dead_heart, downy_mildew, hispa, normal, tungro |

## Software versions

| Package | Version | Evidence |
|---|---|---|
| ultralytics | **8.4.24** | identical to the YOLOv8 baseline run; verified both from the installed package and from the checkpoint's embedded `version` field |
| torch / torchvision / timm | 2.10.0+cu128 / 0.25.0+cu128 / 1.0.25 | installed environment (`.venv`), same environment used for the other Table 6 checkpoints |

## Training configuration

```
task: classify        model: yolo11n-cls.pt   pretrained: true
epochs: 5              batch: 8                imgsz: 224
device: cpu            workers: 0              amp: false
optimizer: auto        lr0: 0.01               lrf: 0.01
momentum: 0.937        weight_decay: 0.0005    warmup_epochs: 3.0
seed: 0                deterministic: true     dropout: 0.0
fraction: 1.0          val: true               split: val
```

This is **identical** to the recorded YOLOv8n-cls training configuration
(`yolov8_baseline_provenance.md`) in every field: same epochs, batch size,
image size, device, worker count, and Ultralytics classification augmentation
defaults (`fliplr: 0.5`, `flipud: 0.0`, `scale: 0.5`, `translate: 0.1`,
`degrees: 0.0`, `hsv_h/s/v`, `auto_augment: randaugment`, `erasing: 0.4`,
`mixup/cutmix: 0.0`). The two YOLO rows in Table 6 therefore differ only in
architecture (YOLOv8n-cls vs YOLOv11n-cls), not in training protocol.

**Deviation from the first attempt:** an initial run with `device=auto` (which
resolved to CUDA on this machine) crashed with a CUDA `misaligned address`
kernel error during the first epoch (loss became `nan` immediately before the
crash). This mirrors the instability already documented for the YOLOv8
baseline, which is why that baseline was also trained on CPU. The run was
restarted with `device: cpu`, `workers: 0`, which completed cleanly; no metric
from the crashed GPU attempt was used.

Per-epoch training curve (`yolo_runs/yolo11_cls/results.csv`):

| epoch | train/loss | val top-1 acc | val top-5 acc | val/loss |
|---|---|---|---|---|
| 1 | 1.718 | 0.643 | 0.967 | 0.993 |
| 2 | 1.131 | 0.798 | 0.983 | 0.597 |
| 3 | 0.916 | 0.844 | 0.992 | 0.491 |
| 4 | 0.774 | 0.881 | 0.993 | 0.383 |
| 5 | 0.653 | 0.909 | 0.992 | 0.324 |

## Inference / preprocessing and evaluation set

Evaluation reused the same `run_yolo_experiment()` code path as the YOLOv8
baseline (generalized in this change to accept a model key / weights /
run-name so both YOLO variants share one implementation): per-image
`yolo_model.predict(source=img_path, imgsz=224, device="cpu", verbose=False)`,
prediction taken from `result.probs.top1`. `validate_dataset_structure()`
falls back from the unlabelled `test/` folder to `val/`, so — exactly as for
every other row in Table 6 — the reported metrics are computed on the
**2,084-image validation set**, not an independent test set. Metrics
(accuracy, macro/weighted precision/recall/F1) are computed with the same
`compute_metrics()` (scikit-learn) used for every other checkpoint.

## Reported metrics (`yolov11/yolov11_metrics.json`)

```
accuracy 0.9093090211   precision_macro 0.9014017930   recall_macro 0.9003449816
f1_macro 0.8989896459   precision_weighted 0.9114913558
recall_weighted 0.9093090211   f1_weighted 0.9088923209
```

## Known discrepancy to state alongside this addition

As already noted for the rest of Table 6, "identical training configuration"
across architectures does not hold globally: the two YOLO branches (v8, v11)
now share one identical protocol with each other, but still differ from the
torchvision/timm branches (EfficientNetV2, EfficientNet-B0, ResNet50,
MobileNetV2, VGG16, ViT), which use a separate pipeline, optimizer, and
augmentation set, and whose actual training epoch count is not recorded in
the stored artefacts. Table 6's existing caveat sentence already covers this
and needs no change beyond adding the YOLOv11 row.
