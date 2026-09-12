# Spatio-Temporal Shoplifting Detection & Action Recognition Pipeline

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-00FFFF?logo=yolo)](https://docs.ultralytics.com/)
[![PyTorchVideo](https://img.shields.io/badge/Meta%20AI-PyTorchVideo-blue)](https://github.com/facebookresearch/pytorchvideo)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An industrial-grade action recognition pipeline engineered to detect subtle shoplifting gestures (pocketing, bagging, and waistband concealment) in high-angle CCTV footage.

By decoupling continuous object detection from action classification through **isolated 3D tubelets**, this system eliminates background retail clutter, bypasses stationary store clerks via **centroid variance gating**, and operates with strict VRAM efficiency on edge-class hardware.

---

## 📽️ Live Demonstration

![Pipeline Demo](assets/demo.gif)
*Figure 1: Real-time inference showing YOLO person tracking, motion-gated stationary cashier suppression (`STATIONARY_DESK`), and SlowFast dual-pathway temporal classification with a 25-frame alert latch.*

---

## 📑 Architectural Design & Dataflow

Static 2D detectors evaluate isolated frames and inevitably fail on temporal concealment actions where a hand hovering near a coat mimics reaching into a pocket. This pipeline constructs localized temporal video cubes (tubelets) around individual tracked persons across time.

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        Surveillance Video Stream                       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
               ┌───────────────────────────────────────────┐
               │  YOLOv8 + ByteTrack (imgsz=960, conf=0.15)│
               └─────────────────────┬─────────────────────┘
                                     │
               ┌─────────────────────┴───────────────┐
               ▼                                     ▼
   [Centroid Variance Gating]              [EMA Box Smoother]
   (Δ Centroid < 15px over 64f)            (α = 0.75 Context)
               │                                     │
               ▼                                     ▼
       [STATIONARY_DESK]                  [Square Padding & Crop]
     (Suppress Inference)                 (15% Margin, 224x224)
                                                     │
                                                     ▼
                                         [64-Frame Tubelet Buffer]
                                                     │
                               ┌─────────────────────┴─────────────────────┐
                               ▼                                           ▼
                        [Fast Pathway]                              [Slow Pathway]
                     32 frames @ stride 2                        8 frames @ stride 8
                     [1, 3, 32, 224, 224]                        [1, 3, 8, 224, 224]
                               └─────────────────────┬─────────────────────┘
                                                     ▼
                                          [SlowFast R50 Backbone]
                                         (Frozen Kinetics Weights)
                                                     │
                                                     ▼
                                          [Linear Projection Head]
                                              (2304 -> 2 dims)
                                                     │
                                                     ▼
                                              [Softmax Output]
                                                     │
                                                     ▼
                                         [25-Frame Temporal Latch]
                                      THEFT ≥ 0.70 | SUSPICIOUS ≥ 0.35
```

### 1. Spatio-Temporal Motion Gating
Cashiers and stationary retail staff frequently interact with checkout desks, POS registers, and conveyor belts, creating local hand motion that triggers false positives inside tight crops. 

The pipeline tracks the coordinate history $\mathcal{H}_i = \{(x_t, y_t)\}_{t=1}^{T}$ of track $i$ over a 64-frame horizon ($T \approx 2.1\text{s}$). If maximum displacement satisfies:

$$\max_{t} |x_t - \bar{x}| < \tau_{\text{disp}} \quad \text{and} \quad \max_{t} |y_t - \bar{y}| < \tau_{\text{disp}} \quad (\tau_{\text{disp}} = 15\text{ px})$$

the track is designated as `STATIONARY_DESK`, bypassing the 3D convolutional network entirely and preserving GPU compute.

### 2. Dual-Pathway Temporal Decomposition
Human actions consist of slow-moving spatial trajectories (body posture, limb stance) and high-frequency motions (hand insertions, fabric manipulation).
- **Slow Pathway ($\tau = 8$):** Operates on $T = 8$ frames with high spatial channel capacity to capture visual posture semantics.
- **Fast Pathway ($\alpha = 4, \tau = 2$):** Operates on $\alpha T = 32$ frames with lightweight channel dimensionality ($\beta = 1/8$) to resolve high-frequency motion boundaries.

Lateral connections fuse Fast pathway feature maps into the Slow pathway across residual stages using 3D cross-pathway convolutions.

---

## 📊 Benchmarks, Complexity & Memory Profile

### Classification Performance
Evaluated on held-out retail surveillance validation clips (662 action segments: 513 normal, 149 theft gestures) trained with dynamic cross-entropy balancing ($w_{\text{theft}} = 3.8\times$):

| Metric | Validation Score | Operational Impact |
| :--- | :---: | :--- |
| **Recall** | **81.88%** | Critical metric; captures over 8 out of 10 rapid concealments |
| **Accuracy** | **80.97%** | High baseline discrimination across general retail activities |
| **F1-Score** | **65.95%** | Calibrated against high recall to minimize false security alerts |

### Computational Complexity & Parameter Budget

| Component | Parameters | Compute (FLOPs / GFLOPs) | Input Tensor Spec |
| :--- | :---: | :---: | :--- |
| **YOLOv8 Object Tracker** | ~3.2M (Nano) / ~25.9M (Medium) | 8.7 GFLOPs / 78.9 GFLOPs | $1 \times 3 \times 960 \times 960$ |
| **SlowFast R50 (Backbone)** | 33,644,488 (Frozen) | ~65.7 GFLOPs | Slow: $[1, 3, 8, 224, 224]$<br>Fast: $[1, 3, 32, 224, 224]$ |
| **Linear Projection Head** | **4,610** (Trainable) | $< 0.01$ GFLOPs | $1 \times 2304 \rightarrow 1 \times 2$ |
| **Total Pipeline Footprint** | **~33.65 M** | **Variable (Gated)** | Dynamic batch based on active tracks |

### Edge Memory Profiling (NVIDIA RTX 3050 Laptop GPU - 4GB VRAM)

| Runtime Phase | Precision | Dedicated VRAM Allocation | Average Execution Speed |
| :--- | :---: | :---: | :---: |
| **Detection & Tracking (imgsz=960)** | FP16 | ~780 MB | 38–45 FPS |
| **SlowFast Forward Pass (per active track)** | FP16 (AMP) | ~1,250 MB | 22–28 ms / tubelet |
| **Full Production Inference Pipeline** | Mixed | **~2,150 MB (Peak)** | **Real-Time (~24–30 FPS overall)** |

---

## 🚀 Getting Started

### Prerequisites
- Linux or Windows 10/11
- NVIDIA GPU with CUDA 11.8+ support (Tested on RTX 3050, RTX 3060, T4)
- Python 3.10 or 3.11
- FFmpeg installed and present in system `PATH`

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/khalilzaghla/slowfast_SL.git
cd slowfast_SL

# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
.\venv\Scripts\activate
# Linux:
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configuration (`configs/inference_config.yaml`)
Customize detection sensitivities, stationary gating horizons, and alert thresholds:
```yaml
tracker:
  conf_floor: 0.15          # Uncaps tracker to catch truncated bodies at frame borders
  imgsz: 960               # High-res inference for small objects on CCTV
  stationary_threshold_px: 15.0 # Max displacement pixel bound
  stationary_window: 64    # Temporal evaluation frame buffer

tubelet:
  crop_margin: 0.15        # Context expansion around bounding box
  square_padding: true     # Prevents aspect ratio distortion
  input_size: [224, 224]   # Standard 3D ResNet input dimension

classification:
  threshold_theft: 0.70      # RED alert bounding box floor
  threshold_suspicious: 0.35 # AMBER alert bounding box floor
  latch_horizon_frames: 25   # Persistence hold duration for concealment events
```

### 3. Running Reference Modules
Test the standalone mathematical reference scripts:
```bash
# Run stationary variance calculation and soft recovery test
python examples/tubelet_gating_concept.py

# Run square-padding and dynamic context crop normalization
python examples/crop_normalizer.py
```

## 🔬 Methodological Innovations

### Soft-Threshold & Cumulative Anomaly Density Recovery
A persistent issue in surveillance dataset annotation is the false-negative classification of subtle concealments. When auditing candidate clips, single-frame confidence may peak at only $0.15$–$0.25$, dropping under standard classification filters ($0.50$).

We recover these true-positive concealments across $K = 16$ sampled frames via dual-criteria density:

$$\text{IsTheft} = \left( \max_{k \in K} c_k \ge \tau_{\text{peak}} \right) \;\lor\; \left( \sum_{k=1}^{K} c_k \ge \tau_{\text{cum}} \right)$$

where $\tau_{\text{peak}} = 0.18$ and $\tau_{\text{cum}} = 0.45$. This rescued 139 subtle theft sequences from false negative classifications during dataset preparation.

## 📚 References
- Feichtenhofer, C., Fan, H., Malik, J., & He, K. (2019). SlowFast Networks for Video Recognition. IEEE/CVF International Conference on Computer Vision (ICCV), 6202-6211. arXiv:1812.03982
- Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., Luo, P., Liu, W., & Wang, X. (2022). ByteTrack: Multi-Object Tracking by Associating Every Detection Box. European Conference on Computer Vision (ECCV). arXiv:2110.06864
- Fan, H., Xiong, B., Mangalam, K., Li, Y., Yan, Z., Malik, J., & Feichtenhofer, C. (2021). PyTorchVideo: A Deep Learning Library for Video Understanding. ACM International Conference on Multimedia (MM '21).
- Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLOv8. GitHub Repository

## 🛡️ Confidentiality & Legal Notice
This repository is published as a technical showcase. Production checkpoints (`.pth`, `.pt`), real surveillance footage, proprietary customer CCTV recordings, and live Flask backend deployment routes are strictly withheld under confidentiality policies. The reference implementations in `examples/` are released under the MIT License.
