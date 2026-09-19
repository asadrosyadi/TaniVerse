# YOLOv8 baseline — provenance for Section 4.4 / Table 7

Every value below was read from the code or from the stored run artefacts in this
folder; none of it is inferred from the fact that the system targets an edge device.

## Variant and checkpoint

| Item | Value | Evidence |
|---|---|---|
| Task | `classify` (image classification, **not** detection) | `yolo_runs/yolov8_cls6/args.yaml` → `task: classify` |
| Variant | **nano** (`n`) | `model: yolov8n-cls.pt` in every `args.yaml`; also the CLI default `--yolo-weights yolov8n-cls.pt` in `benchmark_modern_models.py` |
| Starting checkpoint | `yolov8n-cls.pt` (Ultralytics ImageNet-pretrained classification weights), `pretrained: true` | `args.yaml` |
| Fine-tuned checkpoint used for the reported metrics | `yolo_runs/yolov8_cls6/weights/best.pt` | `find_latest_yolo_best()` selects the newest `**/weights/best.pt`; `cls6/best.pt` mtime 2026-03-23 08:51:46 is the newest, and `yolov8/yolov8_metrics.json` was written later the same day (14:16:17) |
| Classes in the checkpoint | 10 (the ten study labels) | class names embedded in `cls6/weights/best.pt`; `run_yolo_experiment()` raises if the checkpoint class count ≠ dataset class count |

## Software versions

| Package | Version | Evidence |
|---|---|---|
| ultralytics | **8.4.24** | version string embedded in every trained checkpoint (`best.pt` metadata), training date 2026-03-23 |
| ultralytics (declared constraint) | `>=8.2.0` | `requirements_experiment.txt` |
| torch / torchvision / timm (declared) | `>=2.2.0` / `>=0.17.0` / `>=1.0.0` | `requirements_experiment.txt` |

## Training configuration (final run `yolov8_cls6`)

```
task: classify        model: yolov8n-cls.pt   pretrained: true
epochs: 5             batch: 8                imgsz: 224
device: cpu           workers: 0              amp: false
optimizer: auto       lr0: 0.01               lrf: 0.01
momentum: 0.937       weight_decay: 0.0005    warmup_epochs: 3.0
seed: 0               deterministic: true     dropout: 0.0
fraction: 1.0         val: true               split: val
```

Training augmentation = Ultralytics classification defaults:
`fliplr: 0.5`, `flipud: 0.0`, `scale: 0.5`, `translate: 0.1`, `degrees: 0.0`,
`hsv_h: 0.015`, `hsv_s: 0.7`, `hsv_v: 0.4`, `auto_augment: randaugment`,
`erasing: 0.4`, `mixup/cutmix: 0.0`.

Final epoch of `cls6/results.csv`: `metrics/accuracy_top1 = 0.86612`, `val/loss = 0.41309`.

## Inference / preprocessing at evaluation time

`run_yolo_experiment()` calls, per image:

```python
result = yolo_model.predict(source=str(img_path), imgsz=cfg.img_size, device=yolo_device, verbose=False)[0]
```

so preprocessing is Ultralytics' internal classification transform at `imgsz=224`
(no external torchvision transform is applied to the YOLO branch). The prediction is
taken from `result.probs.top1`, and the code explicitly raises if `result.probs is None`
("Pastikan memakai model YOLOv8 classification"), which rules out a detection model.

## Evaluation metrics and evaluation set

* Metrics are computed with scikit-learn over image-level predictions
  (`compute_metrics`): accuracy, precision/recall/F1 in macro and weighted form.
  **No bounding-box metrics (mAP, IoU) are computed anywhere.**
* `validate_dataset_structure()` sets `test_dir = dataset_root/"test"`, but falls back to
  `val_dir` when `test/` is missing **or has no class subfolders**. In this dataset
  `test/` holds 3,469 unlabelled competition images in a flat folder, so the fallback
  applies and the reported metrics are computed on the **2,084-image validation set**.
  This matches the manuscript's own statement in Section 3.3.1 that Table 7 reports
  validation performance.

## Reported metrics (yolov8/yolov8_metrics.json) — match Table 7 exactly

```
accuracy 0.8651631478  precision_macro 0.8533325303  recall_macro 0.8451506668
f1_macro 0.8482119273  precision_weighted 0.8652891544
recall_weighted 0.8651631478  f1_weighted 0.8641488787
```

## Two discrepancies worth resolving in the manuscript

1. **MobileNetV2 is missing from Table 7.** `model_comparison_summary.json` contains a
   seventh model, MobileNetV2, with accuracy 0.0461 and F1-macro 0.0088. Table 7 lists
   only six models. Either include it or state why it was excluded.

   **Diagnosis of the two collapsed checkpoints (verified arithmetically).**
   `mobilenetv2/mobilenetv2_metrics.json` records `test_loss: NaN`, and both MobileNetV2
   and ViT report `recall_macro` of exactly 0.1 — the signature of assigning every image
   to a single class in a ten-class problem (recall 1.0 for one class, 0 for the other
   nine). The accuracies confirm which class:

   | Checkpoint | Accuracy | × 2,084 val images | val class with exactly that count |
   |---|---|---|---|
   | MobileNetV2 | 0.046065259117 | 96.0000 | `bacterial_leaf_blight` (96) |
   | ViT | 0.138675623800 | 289.0000 | `dead_heart` (289) |

   So MobileNetV2 predicted `bacterial_leaf_blight` for all 2,084 validation images and
   ViT predicted `dead_heart` for all of them. For MobileNetV2 the `NaN` test loss
   indicates numerical divergence during training. These are properties of the stored
   checkpoints under this evaluation configuration, not of the architectures.
2. **"Consistent training configuration" needs qualification.** The YOLOv8 branch is
   trained by Ultralytics with its own optimizer schedule and augmentation defaults
   (listed above), whereas the torchvision/timm branches use a separate pipeline
   (`paper_figure_summary.md`: resize 224, horizontal flip p=0.5, rotation 10°,
   colour jitter 0.15). The YOLO run also used 5 epochs (`args.yaml`), while the CLI
   default for the other models is 30 and the actual value used for them is not
   recorded in the stored artefacts. The configurations are therefore comparable in
   input size and evaluation protocol, but not identical in optimizer, augmentation,
   or training budget.
