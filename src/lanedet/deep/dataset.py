"""PyTorch dataset: TuSimple frames -> (image tensor, lane-slot mask)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import TUSIMPLE_SIZE
from ..tusimple import LaneRecord, assign_slots, render_mask

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_tensor(bgr: np.ndarray, size: tuple[int, int]) -> torch.Tensor:
    """BGR uint8 image -> normalized CHW float tensor at ``size`` (w, h)."""
    rgb = cv2.cvtColor(cv2.resize(bgr, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    x = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class TuSimpleDataset(Dataset):
    """Lane-slot segmentation samples.

    Augmentations (train only) are applied to the image *and* the lane points:
    horizontal flip, small random scale/shift, and brightness/contrast jitter. Slots are
    assigned *after* augmenting, so a flipped "left" lane correctly becomes "right".
    """

    def __init__(self, records: list[LaneRecord], root: str | Path, size=(512, 288),
                 augment: bool = False, thickness: int = 5, seed: int | None = None):
        self.records = records
        self.root = Path(root)
        self.size = size
        self.augment = augment
        self.thickness = thickness
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.records)

    def _augment(self, img: np.ndarray, lanes: list[np.ndarray]):
        w, h = TUSIMPLE_SIZE
        if self.rng.random() < 0.5:
            img = img[:, ::-1].copy()
            lanes = [np.column_stack([w - 1 - p[:, 0], p[:, 1]]) for p in lanes]
        scale = self.rng.uniform(0.9, 1.1)
        tx, ty = self.rng.uniform(-0.05, 0.05) * w, self.rng.uniform(-0.05, 0.05) * h
        m = np.array([[scale, 0, (1 - scale) * w / 2 + tx],
                      [0, scale, (1 - scale) * h / 2 + ty]], dtype=np.float32)
        img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)
        lanes = [p @ m[:, :2].T + m[:, 2] for p in lanes]
        alpha, beta = self.rng.uniform(0.7, 1.3), self.rng.uniform(-30, 30)
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return img, lanes

    def __getitem__(self, i: int):
        rec = self.records[i]
        img = cv2.imread(str(self.root / rec.raw_file))
        if img is None:
            raise FileNotFoundError(self.root / rec.raw_file)
        lanes = [rec.lane_points(k) for k in range(len(rec.lanes))]
        if self.augment:
            img, lanes = self._augment(img, lanes)
        mask = render_mask(lanes, assign_slots(lanes), self.size, thickness=self.thickness)
        return to_tensor(img, self.size), torch.from_numpy(mask.astype(np.int64))
