import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

def normalize_person_crop(
    frame: np.ndarray, 
    box: np.ndarray, 
    padding_ratio: float = 0.15,
    crop_size: int = 224,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Illustrates dynamic margin expansion, centered square-padding, 
    and Kinetics tensor normalization.
    """
    h, w, _ = frame.shape
    x1, y1, x2, y2 = box

    # 1. Dynamic Margin Expansion
    bw = x2 - x1
    bh = y2 - y1
    w_pad = bw * (1.0 + padding_ratio)
    h_pad = bh * (1.0 + padding_ratio)
    
    # 2. Centered Square-Padding (S = max(w, h))
    S = max(w_pad, h_pad)
    S_int = max(1, int(round(S)))

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    
    half_s = S / 2.0
    bx1 = int(round(cx - half_s))
    by1 = int(round(cy - half_s))
    bx2 = bx1 + S_int
    by2 = by1 + S_int

    # 3. Boundary Clamping
    cx1 = max(0, bx1)
    cy1 = max(0, by1)
    cx2 = min(w, bx2)
    cy2 = min(h, by2)

    crop = frame[cy1:cy2, cx1:cx2]
    canvas = np.zeros((S_int, S_int, 3), dtype=np.uint8)

    if crop.size > 0:
        dx1 = cx1 - bx1
        dy1 = cy1 - by1
        dx2 = dx1 + (cx2 - cx1)
        dy2 = dy1 + (cy2 - cy1)
        
        dx1, dy1 = max(0, dx1), max(0, dy1)
        dx2, dy2 = min(S_int, dx2), min(S_int, dy2)
        
        ch, cw = dy2 - dy1, dx2 - dx1
        if ch > 0 and cw > 0:
            canvas[dy1:dy2, dx1:dx2] = crop[:ch, :cw]

    # 4. Kinetics Tensor Normalization (ImageNet / Kinetics stats)
    mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1).to(device)
    std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1).to(device)

    # BGR -> RGB -> Tensor [3, 224, 224]
    rgb = torch.from_numpy(canvas[:, :, ::-1].copy()).permute(2, 0, 1).float().to(device) / 255.0
    resized = TF.resize(rgb, [crop_size, crop_size], antialias=True)
    normalized = (resized - mean) / std
    
    return normalized
