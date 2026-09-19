import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

try:
    import timm
except ImportError as exc:
    raise ImportError("Package 'timm' belum terpasang. Jalankan: pip install timm") from exc

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise ImportError("Package 'ultralytics' belum terpasang. Jalankan: pip install ultralytics") from exc


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def has_class_folders(directory: Path) -> bool:
    if not directory.exists() or not directory.is_dir():
        return False
    return any(child.is_dir() for child in directory.iterdir())


@dataclass
class ExperimentConfig:
    dataset_root: Path
    output_dir: Path
    epochs: int
    batch_size: int
    img_size: int
    lr: float
    weight_decay: float
    num_workers: int
    device: str
    disable_cudnn: bool
    seed: int
    patience: int
    max_test_samples: int
    run_yolo: bool
    run_yolo11: bool
    run_efficientnetv2: bool
    run_efficientnetb0: bool
    run_vit: bool
    run_resnet50: bool
    run_mobilenetv2: bool
    run_vgg16: bool
    skip_train: bool
    yolo_weights: str
    yolo11_weights: str
    efficientnetv2_ckpt: str
    efficientnetb0_ckpt: str
    vit_ckpt: str
    resnet50_ckpt: str
    mobilenetv2_ckpt: str
    vgg16_ckpt: str
    efficientnetv2_model: str
    efficientnetb0_model: str
    vit_model: str
    resnet50_model: str
    mobilenetv2_model: str
    vgg16_model: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def build_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, eval_tf


def validate_dataset_structure(dataset_root: Path) -> Tuple[Path, Path, Path]:
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    test_dir = dataset_root / "test"

    if not train_dir.exists():
        raise FileNotFoundError(f"Folder train tidak ditemukan: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Folder val tidak ditemukan: {val_dir}")
    if not test_dir.exists():
        print("[INFO] Folder test tidak ditemukan, evaluasi akan menggunakan folder val.")
        test_dir = val_dir
    elif not has_class_folders(test_dir):
        print("[INFO] Folder test kosong/tidak punya subfolder kelas, evaluasi akan menggunakan folder val.")
        test_dir = val_dir

    return train_dir, val_dir, test_dir


def create_dataloaders(
    train_dir: Path,
    val_dir: Path,
    test_dir: Path,
    img_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], List[str]]:
    train_tf, eval_tf = build_transforms(img_size)

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_tf)
    test_ds = datasets.ImageFolder(test_dir, transform=eval_tf)

    class_to_idx = train_ds.class_to_idx
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, class_to_idx, class_names


def train_one_epoch(model, loader, criterion, optimizer, device: str) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    y_true, y_pred = [], []

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / max(len(loader.dataset), 1)
    epoch_acc = accuracy_score(y_true, y_pred)
    return epoch_loss, epoch_acc


def evaluate_torch_model(
    model,
    loader,
    criterion,
    device: str,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Eval", leave=False):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels.detach().cpu().numpy().tolist())
            y_pred.extend(preds.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / max(len(loader.dataset), 1)
    return epoch_loss, np.array(y_true), np.array(y_pred)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int | None = None) -> Dict[str, float]:
    labels = list(range(num_classes)) if num_classes is not None else None

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
    }


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    title: str,
    out_path: Path,
) -> None:
    num_classes = len(class_names)
    labels_idx = list(range(num_classes))

    invalid_pred_mask = (y_pred < 0) | (y_pred >= num_classes)
    invalid_count = int(np.sum(invalid_pred_mask))
    if invalid_count > 0:
        print(
            f"[WARNING] Ditemukan {invalid_count} prediksi di luar rentang kelas dataset "
            f"(0..{num_classes - 1}). Prediksi di luar rentang diabaikan pada confusion matrix."
        )

    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
    cm_percent = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100

    labels = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            labels[i, j] = f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)"

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Jumlah Prediksi"},
    )
    if invalid_count > 0:
        title = f"{title} (invalid_pred_ignored={invalid_count})"

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Prediksi Model")
    plt.ylabel("Label Aktual")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_metrics_files(metrics: Dict[str, float], out_dir: Path, model_tag: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{model_tag}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def build_model(model_name: str, num_classes: int) -> nn.Module:
    return timm.create_model(model_name, pretrained=True, num_classes=num_classes)


def run_torch_experiment(
    model_key: str,
    model_name: str,
    cfg: ExperimentConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    class_to_idx: Dict[str, int],
    class_names: List[str],
    ckpt_path_override: str,
) -> Dict[str, float]:
    model_out = cfg.output_dir / model_key
    model_out.mkdir(parents=True, exist_ok=True)

    device = cfg.device
    num_classes = len(class_names)

    model = build_model(model_name, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1))

    ckpt_path = Path(ckpt_path_override) if ckpt_path_override else model_out / "best_checkpoint.pt"

    if not cfg.skip_train:
        best_f1 = -1.0
        patience_counter = 0

        for epoch in range(1, cfg.epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, y_val_true, y_val_pred = evaluate_torch_model(model, val_loader, criterion, device)
            val_metrics = compute_metrics(y_val_true, y_val_pred)
            scheduler.step()

            print(
                f"[{model_key}] Epoch {epoch:03d}/{cfg.epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_f1={val_metrics['f1_macro']:.4f}"
            )

            if val_metrics["f1_macro"] > best_f1:
                best_f1 = val_metrics["f1_macro"]
                patience_counter = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "model_name": model_name,
                        "class_to_idx": class_to_idx,
                        "epoch": epoch,
                        "val_metrics": val_metrics,
                    },
                    ckpt_path,
                )
            else:
                patience_counter += 1

            if patience_counter >= cfg.patience:
                print(f"[{model_key}] Early stopping aktif pada epoch {epoch}.")
                break

    if cfg.skip_train and not ckpt_path.exists():
        print(
            f"[{model_key}] Checkpoint tidak ditemukan pada mode --skip-train. "
            "Model pretrained akan dievaluasi langsung tanpa fine-tuning."
        )
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_name": model_name,
                "class_to_idx": class_to_idx,
                "epoch": 0,
                "val_metrics": {},
            },
            ckpt_path,
        )

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint {model_key} tidak ditemukan: {ckpt_path}. "
            "Gunakan mode training atau berikan path checkpoint yang benar."
        )

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    test_loss, y_true, y_pred = evaluate_torch_model(model, test_loader, criterion, device)
    test_metrics = compute_metrics(y_true, y_pred, num_classes=len(class_names))
    test_metrics["test_loss"] = float(test_loss)
    test_metrics["model"] = model_key

    save_metrics_files(test_metrics, model_out, model_key)
    save_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        f"Confusion Matrix - {model_key}",
        model_out / f"{model_key}_confusion_matrix.png",
    )

    return test_metrics


def collect_test_images(
    test_dir: Path,
    class_to_idx: Dict[str, int],
    max_test_samples: int,
) -> Tuple[List[Path], List[int]]:
    image_paths: List[Path] = []
    labels: List[int] = []

    for class_name, class_idx in sorted(class_to_idx.items(), key=lambda item: item[1]):
        class_dir = test_dir / class_name
        if not class_dir.exists():
            continue

        class_images = [
            p
            for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        class_images = sorted(class_images)

        if max_test_samples > 0:
            class_images = class_images[:max_test_samples]

        image_paths.extend(class_images)
        labels.extend([class_idx] * len(class_images))

    return image_paths, labels


def find_latest_yolo_best(run_dir: Path, name_prefix: str) -> Path:
    """Newest best.pt belonging to this YOLO variant's own run-name prefix.

    `yolo_runs/` accumulates runs from every YOLO variant benchmarked over
    time (e.g. `yolov8_cls*` and `yolo11_cls*`); scoping the glob by
    `name_prefix` prevents a later YOLOv11 run from picking up an older
    YOLOv8 checkpoint (or vice versa) just because it has a newer mtime.
    """
    candidates = sorted(
        run_dir.glob(f"{name_prefix}*/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not candidates:
        raise FileNotFoundError(f"Tidak ada YOLO best.pt untuk '{name_prefix}*' di {run_dir}")
    return candidates[0]


def run_yolo_experiment(
    cfg: ExperimentConfig,
    test_dir: Path,
    class_to_idx: Dict[str, int],
    class_names: List[str],
    model_key: str = "yolov8",
    default_weights: str = "yolov8n-cls.pt",
    weights_override: str = "",
    run_name_prefix: str = "yolov8_cls",
    display_name: str = "YOLOv8",
) -> Dict[str, float]:
    model_out = cfg.output_dir / model_key
    model_out.mkdir(parents=True, exist_ok=True)

    # Must be absolute: Ultralytics silently re-roots a *relative* `project`
    # path under its own "runs/<task>/" cwd-relative directory, which caused
    # the trained checkpoint to land outside `cfg.output_dir` entirely.
    yolo_run_dir = (cfg.output_dir / "yolo_runs").resolve()
    yolo_run_dir.mkdir(parents=True, exist_ok=True)

    weights_arg = weights_override or default_weights
    weights_path = Path(weights_arg)
    yolo_device = cfg.device
    if cfg.disable_cudnn and cfg.device.startswith("cuda"):
        # Saat mode stabil aktif, YOLO lebih aman dijalankan di CPU pada beberapa GPU laptop.
        yolo_device = "cpu"

    if not cfg.skip_train:
        yolo_model = YOLO(weights_arg)
        train_kwargs = {
            "data": str(cfg.dataset_root),
            "epochs": cfg.epochs,
            "imgsz": cfg.img_size,
            "batch": cfg.batch_size,
            "project": str(yolo_run_dir),
            "name": run_name_prefix,
            "device": yolo_device,
            "verbose": True,
            "workers": cfg.num_workers,
            "amp": False,
        }

        try:
            yolo_model.train(**train_kwargs)
        except RuntimeError as exc:
            err_msg = str(exc).lower()
            # Fallback ini penting untuk GPU consumer yang kadang tidak stabil di cuDNN.
            if "cudnn" in err_msg or "cudnn_status_execution_failed" in err_msg:
                print(f"[{display_name}] CUDA/cuDNN error terdeteksi, fallback ke CPU agar eksperimen tetap jalan.")
                yolo_device = "cpu"
                yolo_model = YOLO(weights_arg)
                train_kwargs["device"] = yolo_device
                train_kwargs["workers"] = 0
                yolo_model.train(**train_kwargs)
            else:
                raise

        weights_path = find_latest_yolo_best(yolo_run_dir, run_name_prefix)
    elif weights_arg == default_weights:
        # Pada mode skip-train, bobot default 1000 kelas membuat prediksi tidak cocok untuk dataset 10 kelas.
        # Jika ada hasil training sebelumnya, gunakan otomatis checkpoint terbaru.
        try:
            weights_path = find_latest_yolo_best(yolo_run_dir, run_name_prefix)
            print(f"[{display_name}] Mode --skip-train: menggunakan checkpoint terbaru {weights_path}")
        except FileNotFoundError:
            pass

    if not weights_path.exists():
        raise FileNotFoundError(
            f"File bobot {display_name} tidak ditemukan: {weights_path}. "
            "Jalankan training dulu atau isi argumen weights dengan path best.pt."
        )

    yolo_model = YOLO(str(weights_path))

    yolo_num_classes = len(yolo_model.names) if getattr(yolo_model, "names", None) is not None else None
    expected_num_classes = len(class_names)
    if yolo_num_classes is not None and yolo_num_classes != expected_num_classes:
        raise RuntimeError(
            f"Jumlah kelas bobot {display_name} tidak cocok dengan dataset. "
            f"Bobot saat ini memiliki {yolo_num_classes} kelas, sedangkan dataset memiliki {expected_num_classes} kelas. "
            "Gunakan checkpoint hasil training dataset padi (best.pt), bukan bobot default ImageNet."
        )

    test_images, y_true = collect_test_images(test_dir, class_to_idx, cfg.max_test_samples)
    if not test_images:
        raise RuntimeError(f"Tidak ada gambar test ditemukan di {test_dir}")

    y_pred: List[int] = []
    for img_path in tqdm(test_images, desc=f"{display_name} Predict", leave=False):
        try:
            result = yolo_model.predict(
                source=str(img_path),
                imgsz=cfg.img_size,
                device=yolo_device,
                verbose=False,
            )[0]
        except RuntimeError as exc:
            err_msg = str(exc).lower()
            if yolo_device != "cpu" and ("cudnn" in err_msg or "cudnn_status_execution_failed" in err_msg):
                print(f"[{display_name}] Prediksi GPU gagal, fallback ke CPU untuk evaluasi.")
                yolo_device = "cpu"
                result = yolo_model.predict(
                    source=str(img_path),
                    imgsz=cfg.img_size,
                    device=yolo_device,
                    verbose=False,
                )[0]
            else:
                raise

        if result.probs is None:
            raise RuntimeError(f"Output {display_name} tidak memiliki field probs. Pastikan memakai model klasifikasi.")

        if hasattr(result.probs, "top1"):
            pred_idx = int(result.probs.top1)
        else:
            pred_idx = int(torch.argmax(result.probs.data).item())

        y_pred.append(pred_idx)

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    metrics = compute_metrics(y_true_arr, y_pred_arr, num_classes=len(class_names))
    metrics["model"] = model_key
    save_metrics_files(metrics, model_out, model_key)

    save_confusion_matrix(
        y_true_arr,
        y_pred_arr,
        class_names,
        f"Confusion Matrix - {display_name}",
        model_out / f"{model_key}_confusion_matrix.png",
    )

    return metrics


def save_markdown_table(df: pd.DataFrame, out_path: Path) -> None:
    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, row in df.iterrows():
        values = []
        for col in headers:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def plot_summary_bars(df: pd.DataFrame, out_path: Path) -> None:
    plot_df = df.copy()
    metric_cols = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    melted = plot_df.melt(id_vars=["model"], value_vars=metric_cols, var_name="metric", value_name="score")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted, x="metric", y="score", hue="model")
    plt.ylim(0, 1)
    plt.title("Perbandingan Model Modern")
    plt.ylabel("Skor")
    plt.xlabel("Metrik")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_experiments(cfg: ExperimentConfig) -> None:
    set_seed(cfg.seed)
    if cfg.device.startswith("cuda") and cfg.disable_cudnn:
        # Mode stabil: beberapa GPU laptop mengalami CUDNN_STATUS_EXECUTION_FAILED.
        torch.backends.cudnn.enabled = False
        print("[INFO] cuDNN dinonaktifkan untuk stabilitas training di CUDA.")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    train_dir, val_dir, test_dir = validate_dataset_structure(cfg.dataset_root)

    train_loader, val_loader, test_loader, class_to_idx, class_names = create_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        img_size=cfg.img_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device.startswith("cuda"),
    )

    summary_rows: List[Dict[str, float]] = []

    if cfg.run_yolo:
        print("\n=== Menjalankan eksperimen YOLOv8 ===")
        yolo_metrics = run_yolo_experiment(
            cfg, test_dir, class_to_idx, class_names,
            model_key="yolov8", default_weights="yolov8n-cls.pt",
            weights_override=cfg.yolo_weights, run_name_prefix="yolov8_cls",
            display_name="YOLOv8",
        )
        summary_rows.append(yolo_metrics)

    if cfg.run_yolo11:
        print("\n=== Menjalankan eksperimen YOLOv11 ===")
        yolo11_metrics = run_yolo_experiment(
            cfg, test_dir, class_to_idx, class_names,
            model_key="yolov11", default_weights="yolo11n-cls.pt",
            weights_override=cfg.yolo11_weights, run_name_prefix="yolo11_cls",
            display_name="YOLOv11",
        )
        summary_rows.append(yolo11_metrics)

    if cfg.run_efficientnetv2:
        print("\n=== Menjalankan eksperimen EfficientNetV2 ===")
        eff_metrics = run_torch_experiment(
            model_key="efficientnetv2",
            model_name=cfg.efficientnetv2_model,
            cfg=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            class_to_idx=class_to_idx,
            class_names=class_names,
            ckpt_path_override=cfg.efficientnetv2_ckpt,
        )
        summary_rows.append(eff_metrics)

    if cfg.run_efficientnetb0:
        print("\n=== Menjalankan eksperimen EfficientNet-B0 (usulan) ===")
        effb0_metrics = run_torch_experiment(
            model_key="efficientnet_b0",
            model_name=cfg.efficientnetb0_model,
            cfg=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            class_to_idx=class_to_idx,
            class_names=class_names,
            ckpt_path_override=cfg.efficientnetb0_ckpt,
        )
        summary_rows.append(effb0_metrics)

    if cfg.run_vit:
        print("\n=== Menjalankan eksperimen ViT ===")
        vit_metrics = run_torch_experiment(
            model_key="vit",
            model_name=cfg.vit_model,
            cfg=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            class_to_idx=class_to_idx,
            class_names=class_names,
            ckpt_path_override=cfg.vit_ckpt,
        )
        summary_rows.append(vit_metrics)

    if cfg.run_resnet50:
        print("\n=== Menjalankan eksperimen ResNet50 ===")
        resnet_metrics = run_torch_experiment(
            model_key="resnet50",
            model_name=cfg.resnet50_model,
            cfg=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            class_to_idx=class_to_idx,
            class_names=class_names,
            ckpt_path_override=cfg.resnet50_ckpt,
        )
        summary_rows.append(resnet_metrics)

    if cfg.run_mobilenetv2:
        print("\n=== Menjalankan eksperimen MobileNetV2 ===")
        mobilenet_metrics = run_torch_experiment(
            model_key="mobilenetv2",
            model_name=cfg.mobilenetv2_model,
            cfg=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            class_to_idx=class_to_idx,
            class_names=class_names,
            ckpt_path_override=cfg.mobilenetv2_ckpt,
        )
        summary_rows.append(mobilenet_metrics)

    if cfg.run_vgg16:
        print("\n=== Menjalankan eksperimen VGG16 ===")
        vgg_metrics = run_torch_experiment(
            model_key="vgg16",
            model_name=cfg.vgg16_model,
            cfg=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            class_to_idx=class_to_idx,
            class_names=class_names,
            ckpt_path_override=cfg.vgg16_ckpt,
        )
        summary_rows.append(vgg_metrics)

    if not summary_rows:
        raise RuntimeError("Tidak ada model yang dipilih. Aktifkan minimal satu model.")

    summary_df = pd.DataFrame(summary_rows)
    metric_order = [
        "model",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    ]
    summary_df = summary_df[[c for c in metric_order if c in summary_df.columns]]
    summary_df = summary_df.sort_values(by="f1_macro", ascending=False).reset_index(drop=True)

    summary_csv = cfg.output_dir / "model_comparison_summary.csv"
    summary_json = cfg.output_dir / "model_comparison_summary.json"
    summary_md = cfg.output_dir / "model_comparison_summary.md"
    summary_plot = cfg.output_dir / "model_comparison_summary.png"

    summary_df.to_csv(summary_csv, index=False)
    summary_df.to_json(summary_json, orient="records", indent=2)
    save_markdown_table(summary_df, summary_md)
    plot_summary_bars(summary_df, summary_plot)

    print("\n=== Eksperimen selesai ===")
    print(f"Ringkasan CSV   : {summary_csv}")
    print(f"Ringkasan JSON  : {summary_json}")
    print(f"Ringkasan MD    : {summary_md}")
    print(f"Ringkasan Plot  : {summary_plot}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark model modern untuk klasifikasi penyakit padi: "
            "YOLOv8, EfficientNetV2, ViT, ResNet50, MobileNetV2, VGG16, dan EfficientNet-B0"
        )
    )
    parser.add_argument("--dataset-root", type=str, required=True, help="Path dataset dengan struktur train/val/test")
    parser.add_argument("--output-dir", type=str, default="./experiments/model_comparison")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda")
    parser.add_argument(
        "--disable-cudnn",
        action="store_true",
        help="Nonaktifkan cuDNN saat menggunakan CUDA (lebih stabil, bisa lebih lambat)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=0,
        help="Batas jumlah sampel test per kelas (0 = semua)",
    )

    parser.add_argument("--skip-train", action="store_true", help="Lewati training dan langsung evaluasi bobot")
    parser.add_argument("--yolo-weights", type=str, default="yolov8n-cls.pt")
    parser.add_argument("--yolo11-weights", type=str, default="yolo11n-cls.pt")
    parser.add_argument("--efficientnetv2-ckpt", type=str, default="")
    parser.add_argument("--efficientnetb0-ckpt", type=str, default="")
    parser.add_argument("--vit-ckpt", type=str, default="")
    parser.add_argument("--resnet50-ckpt", type=str, default="")
    parser.add_argument("--mobilenetv2-ckpt", type=str, default="")
    parser.add_argument("--vgg16-ckpt", type=str, default="")
    parser.add_argument("--efficientnetv2-model", type=str, default="efficientnetv2_rw_s")
    parser.add_argument("--efficientnetb0-model", type=str, default="efficientnet_b0")
    parser.add_argument("--vit-model", type=str, default="vit_base_patch16_224")
    parser.add_argument("--resnet50-model", type=str, default="resnet50")
    parser.add_argument("--mobilenetv2-model", type=str, default="mobilenetv2_100")
    parser.add_argument("--vgg16-model", type=str, default="vgg16")

    parser.add_argument("--no-yolo", action="store_true", help="Nonaktifkan YOLOv8")
    parser.add_argument("--no-yolo11", action="store_true", help="Nonaktifkan YOLOv11")
    parser.add_argument("--no-efficientnetv2", action="store_true", help="Nonaktifkan EfficientNetV2")
    parser.add_argument("--no-efficientnetb0", action="store_true", help="Nonaktifkan EfficientNet-B0")
    parser.add_argument("--no-vit", action="store_true", help="Nonaktifkan ViT")
    parser.add_argument("--no-resnet50", action="store_true", help="Nonaktifkan ResNet50")
    parser.add_argument("--no-mobilenetv2", action="store_true", help="Nonaktifkan MobileNetV2")
    parser.add_argument("--no-vgg16", action="store_true", help="Nonaktifkan VGG16")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    cfg = ExperimentConfig(
        dataset_root=Path(args.dataset_root),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        device=device,
        disable_cudnn=args.disable_cudnn,
        seed=args.seed,
        patience=args.patience,
        max_test_samples=args.max_test_samples,
        run_yolo=not args.no_yolo,
        run_yolo11=not args.no_yolo11,
        run_efficientnetv2=not args.no_efficientnetv2,
        run_efficientnetb0=not args.no_efficientnetb0,
        run_vit=not args.no_vit,
        run_resnet50=not args.no_resnet50,
        run_mobilenetv2=not args.no_mobilenetv2,
        run_vgg16=not args.no_vgg16,
        skip_train=args.skip_train,
        yolo_weights=args.yolo_weights,
        yolo11_weights=args.yolo11_weights,
        efficientnetv2_ckpt=args.efficientnetv2_ckpt,
        efficientnetb0_ckpt=args.efficientnetb0_ckpt,
        vit_ckpt=args.vit_ckpt,
        resnet50_ckpt=args.resnet50_ckpt,
        mobilenetv2_ckpt=args.mobilenetv2_ckpt,
        vgg16_ckpt=args.vgg16_ckpt,
        efficientnetv2_model=args.efficientnetv2_model,
        efficientnetb0_model=args.efficientnetb0_model,
        vit_model=args.vit_model,
        resnet50_model=args.resnet50_model,
        mobilenetv2_model=args.mobilenetv2_model,
        vgg16_model=args.vgg16_model,
    )

    print("Konfigurasi eksperimen:")
    print(json.dumps({k: str(v) for k, v in cfg.__dict__.items()}, indent=2))

    run_experiments(cfg)


if __name__ == "__main__":
    main()
