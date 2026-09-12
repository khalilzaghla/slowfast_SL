# Shoplifting Detection Pipeline (Showcase)

> **Disclaimer:** This repository is a sanitized, public-facing showcase portfolio. All proprietary model checkpoints, `.pth` / `.pt` weights, surveillance datasets, and full server infrastructures have been explicitly removed to comply with confidentiality and data protection policies. The reference examples provided illustrate mathematical implementations and architecture concepts exclusively.

## Architecture Pipeline

```text
[Video Stream / File]
        │
        ▼
[YOLOv8 Object Tracking] ──(Centroid Check)──> Stationary Cashier Filter
        │
        ▼ (ema_boxes)
[Dynamic Crop & Square-Pad]
        │
        ▼ (3, 64, 224, 224)
[Temporal Tubelet Buffer]
        │
        ├──> Fast Pathway [1, 3, 32, 224, 224] (Stride 2)
        └──> Slow Pathway [1, 3, 8, 224, 224] (Stride 4)
                 │
                 ▼
     [SlowFast 3D ResNet-50]
                 │
                 ▼
     [Theft Probability Latch] ──> HUD Visualization (Red/Amber/Green)
```

## Internal Benchmarks

| Metric | Score |
| :--- | :--- |
| **Recall** | 81.88% |
| **Accuracy** | 80.97% |
| **F1-Score** | 65.95% |
