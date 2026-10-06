"""U-Net lane-slot segmentation model and its inference wrapper."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch

from ..config import NUM_CLASSES, SLOTS
from ..measure import add_measurements
from ..types import LaneResult
from .dataset import to_tensor


def pick_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(encoder: str = "resnet34", pretrained: bool = True) -> torch.nn.Module:
    """U-Net with an ImageNet-pretrained encoder; one output channel per lane slot + bg."""
    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                    in_channels=3, classes=NUM_CLASSES)


def extract_lanes(probs: np.ndarray, out_size: tuple[int, int], thresh: float = 0.4,
                  min_rows: int = 8, window: int = 6, degree: int = 2
                  ) -> dict[str, np.ndarray]:
    """Turn per-class probability maps (C, h, w) into image-space lane polylines.

    For each lane class and each row, take the probability-weighted x around the row's
    peak. Rows whose peak is below ``thresh`` are skipped. A low-order polynomial
    x = f(y) is then fitted to remove pixel noise and resampled over the detected span.
    """
    _, h, w = probs.shape
    sx, sy = out_size[0] / w, out_size[1] / h
    cols = np.arange(w)
    lanes = {}
    for cls, slot in enumerate(SLOTS, start=1):
        p = probs[cls]
        peaks = p.argmax(axis=1)
        pts = []
        for y in range(h):
            if p[y, peaks[y]] < thresh:
                continue
            lo, hi = max(peaks[y] - window, 0), min(peaks[y] + window + 1, w)
            weights = p[y, lo:hi]
            pts.append((float((cols[lo:hi] * weights).sum() / weights.sum()), float(y)))
        if len(pts) < min_rows:
            continue
        pts = np.array(pts)
        coef = np.polyfit(pts[:, 1], pts[:, 0], min(degree, len(pts) - 1))
        ys = np.linspace(pts[:, 1].min(), pts[:, 1].max(), 72)
        lanes[slot] = np.stack([np.polyval(coef, ys) * sx, ys * sy], axis=1)
    return lanes


class DeepLaneDetector:
    name = "deep"

    def __init__(self, checkpoint: str | Path, device: str | None = None,
                 thresh: float = 0.4, classical_cfg=None):
        self.device = pick_device(device)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.size = tuple(ckpt["input_size"])
        self.model = build_model(ckpt["encoder"], pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        self.thresh = thresh
        self.classical_cfg = classical_cfg
        self.last_inference_ms = 0.0

    @torch.no_grad()
    def probabilities(self, bgr: np.ndarray) -> np.ndarray:
        x = to_tensor(bgr, self.size).unsqueeze(0).to(self.device)
        start = time.perf_counter()
        probs = torch.softmax(self.model(x), dim=1)[0].float().cpu().numpy()
        self.last_inference_ms = (time.perf_counter() - start) * 1000
        return probs

    def __call__(self, bgr: np.ndarray) -> LaneResult:
        h, w = bgr.shape[:2]
        lanes = extract_lanes(self.probabilities(bgr), (w, h), self.thresh)
        result = LaneResult(size=(w, h), lanes=lanes)
        if self.classical_cfg is not None:
            add_measurements(result, self.classical_cfg)
        return result
