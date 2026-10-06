"""Common output type shared by both detectors."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LaneResult:
    """Detected lanes as image-space polylines keyed by slot name.

    Slot names: ``left_left``, ``left``, ``right``, ``right_right`` (``left``/``right``
    are the ego lane's boundaries).
    """

    size: tuple[int, int]  # (width, height) of the image the lanes live in
    lanes: dict[str, np.ndarray] = field(default_factory=dict)  # slot -> (N, 2) x, y points
    curvature_m: float | None = None
    offset_m: float | None = None  # + = car is right of lane centre
    departure: bool = False

    def x_at(self, slot: str, ys) -> list[float]:
        """Lane x at each requested row (TuSimple format: -2 where undefined)."""
        ys = np.asarray(ys, dtype=float)
        pts = self.lanes.get(slot)
        if pts is None or len(pts) < 2:
            return [-2.0] * len(ys)
        pts = pts[np.argsort(pts[:, 1])]
        y_min, y_max = pts[0, 1], pts[-1, 1]
        xs = np.interp(ys, pts[:, 1], pts[:, 0])
        valid = (ys >= y_min) & (ys <= y_max) & (xs >= 0) & (xs < self.size[0])
        return [float(x) if v else -2.0 for x, v in zip(xs, valid, strict=True)]

    def tusimple_lanes(self, ys, slots=None) -> list[list[float]]:
        """Predicted lanes over ``ys``, dropping lanes with no valid point."""
        out = []
        for slot in slots or self.lanes:
            xs = self.x_at(slot, ys)
            if any(x >= 0 for x in xs):
                out.append(xs)
        return out
