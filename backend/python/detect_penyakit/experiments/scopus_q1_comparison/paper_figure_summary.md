# Figure Summary for Scopus Q1

## Generated Figures

- Confusion matrix comparison: all_models_confusion_matrix_comparison.png
- After data augmentation examples: after_data_augmentation_examples.png

## Model Ranking by F1-macro

1. EfficientNetV2: accuracy=0.8690, f1_macro=0.8491
2. YOLOv8: accuracy=0.8652, f1_macro=0.8482
3. EfficientNet-B0: accuracy=0.8589, f1_macro=0.8474
4. ResNet50: accuracy=0.4290, f1_macro=0.2560
5. VGG16: accuracy=0.1108, f1_macro=0.0642
6. ViT: accuracy=0.1387, f1_macro=0.0244
7. MobileNetV2: accuracy=0.0461, f1_macro=0.0088

## Augmentation Pipeline

- Resize ke 224x224
- Random horizontal flip (p=0.5)
- Random rotation 10 derajat
- Color jitter untuk brightness, contrast, dan saturation sebesar 0.15