"""Curvature / offset / departure for any detector's ego lanes.

The deep model outputs image-space lane points; warping them into the same bird's-eye
view as the classical pipeline lets both report metric curvature and offset.
"""

from __future__ import annotations

import cv2
import numpy as np

from .classical.config import ClassicalConfig
from .classical.geometry import PerspectiveTransform
from .classical.lane_fit import curvature_radius_m, vehicle_offset_m
from .types import LaneResult


def add_measurements(result: LaneResult, cfg: ClassicalConfig) -> LaneResult:
    if "left" not in result.lanes or "right" not in result.lanes:
        return result
    w, h = result.size
    tf = PerspectiveTransform((w, h), cfg.src, cfg.dst)
    fits = []
    for slot in ("left", "right"):
        pts = cv2.perspectiveTransform(
            result.lanes[slot].astype(np.float32).reshape(-1, 1, 2), tf.M).reshape(-1, 2)
        pts = pts[(pts[:, 1] >= 0) & (pts[:, 1] < h)]
        if len(pts) < 5:
            return result
        fits.append(np.polyfit(pts[:, 1], pts[:, 0], 2))
    result.curvature_m = float(np.mean(
        [curvature_radius_m(f, h - 1, cfg.xm_per_pix, cfg.ym_per_pix) for f in fits]))
    result.offset_m = vehicle_offset_m(fits[0], fits[1], w, h, cfg.xm_per_pix)
    result.departure = abs(result.offset_m) > cfg.departure_warning_m
    return result
