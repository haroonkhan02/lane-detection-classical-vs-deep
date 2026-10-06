"""TuSimple annotations: parsing, lane-slot assignment and mask rendering.

Each annotation line is JSON with ``lanes`` (list of x-coordinate lists, -2 = no point),
``h_samples`` (the shared y-coordinates) and ``raw_file`` (image path).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import SLOTS, TUSIMPLE_SIZE


@dataclass
class LaneRecord:
    raw_file: str
    lanes: list[list[float]]  # x per h_sample, -2 where the lane is absent
    h_samples: list[float]

    def lane_points(self, i: int) -> np.ndarray:
        """(N, 2) array of valid (x, y) points of lane ``i``."""
        xs = np.asarray(self.lanes[i], dtype=float)
        ys = np.asarray(self.h_samples, dtype=float)
        valid = xs >= 0
        return np.stack([xs[valid], ys[valid]], axis=1)


def load_records(paths: Iterable[str | Path]) -> list[LaneRecord]:
    records = []
    for path in paths:
        with Path(path).open() as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    records.append(LaneRecord(d["raw_file"], d["lanes"], d["h_samples"]))
    return records


def x_at_bottom(points: np.ndarray, height: int) -> float | None:
    """Extrapolate a lane to the bottom image row with a line fit on its lower half."""
    if len(points) < 2:
        return None
    pts = points[np.argsort(points[:, 1])]
    lower = pts[len(pts) // 2:] if len(pts) >= 4 else pts
    slope, intercept = np.polyfit(lower[:, 1], lower[:, 0], 1)
    return float(slope * (height - 1) + intercept)


def assign_slots(lanes_points: list[np.ndarray], width: int = TUSIMPLE_SIZE[0],
                 height: int = TUSIMPLE_SIZE[1]) -> dict[str, int]:
    """Map lane slots (left_left, left, right, right_right) to lane indices.

    Lanes are placed left/right of the image centre (the camera is mounted centrally)
    by where they hit the bottom row; the closest one on each side is the ego lane.
    Extra lanes beyond two per side are ignored.
    """
    centre = width / 2
    bottoms = [(x_at_bottom(p, height), i) for i, p in enumerate(lanes_points)]
    bottoms = [(x, i) for x, i in bottoms if x is not None]
    left = sorted([b for b in bottoms if b[0] < centre], reverse=True)
    right = sorted([b for b in bottoms if b[0] >= centre])
    slots: dict[str, int] = {}
    for name, entry in zip(("left", "left_left"), left, strict=False):
        slots[name] = entry[1]
    for name, entry in zip(("right", "right_right"), right, strict=False):
        slots[name] = entry[1]
    return slots


def render_mask(lanes_points: list[np.ndarray], slots: dict[str, int], out_size: tuple[int, int],
                src_size: tuple[int, int] = TUSIMPLE_SIZE, thickness: int = 5) -> np.ndarray:
    """Draw lanes into a class mask of size ``out_size`` (w, h); 0 = background."""
    w, h = out_size
    sx, sy = w / src_size[0], h / src_size[1]
    mask = np.zeros((h, w), dtype=np.uint8)
    for cls, name in enumerate(SLOTS, start=1):
        if name not in slots:
            continue
        pts = lanes_points[slots[name]]
        if len(pts) < 2:
            continue
        scaled = np.round(pts * [sx, sy]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(mask, [scaled], False, cls, thickness, cv2.LINE_8)
    return mask


def slot_lanes_from_record(record: LaneRecord) -> dict[str, list[float]]:
    """Ground-truth lanes keyed by slot, as x-lists over h_samples."""
    pts = [record.lane_points(i) for i in range(len(record.lanes))]
    return {name: record.lanes[idx] for name, idx in assign_slots(pts).items()}


def available_records(json_names: Iterable[str], root: str | Path) -> list[LaneRecord]:
    """Records from the given annotation files whose image exists under ``root``."""
    root = Path(root)
    paths = [root / n for n in json_names if (root / n).exists()]
    return [r for r in load_records(paths) if (root / r.raw_file).exists()]
