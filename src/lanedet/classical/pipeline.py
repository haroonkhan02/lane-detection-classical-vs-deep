"""End-to-end classical lane detector (single images or video with tracking)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import LaneResult
from .calibration import Calibration
from .config import ClassicalConfig
from .geometry import PerspectiveTransform, threshold_binary
from .lane_fit import (
    LaneSearch,
    curvature_radius_m,
    fit_polynomial,
    lanes_plausible,
    search_around,
    sliding_window,
    vehicle_offset_m,
)


@dataclass
class ClassicalDebug:
    warped: np.ndarray
    binary: np.ndarray
    search: LaneSearch


class ClassicalLaneDetector:
    """Undistort -> bird's-eye warp -> threshold -> sliding windows -> polynomial fit.

    With ``video=True`` the detector searches around the previous fit, smooths fits with
    an exponential moving average and keeps the last good fit for a few frames when a
    detection fails its sanity checks.
    """

    name = "classical"

    def __init__(self, cfg: ClassicalConfig | None = None, calibration: Calibration | None = None,
                 video: bool = False):
        self.cfg = cfg or ClassicalConfig()
        self.calibration = calibration
        self.video = video
        self._transform: PerspectiveTransform | None = None
        self.reset()

    def reset(self) -> None:
        self._fits: tuple[np.ndarray, np.ndarray] | None = None
        self._lost = 0

    def transform(self, size: tuple[int, int]) -> PerspectiveTransform:
        if self._transform is None or self._transform.size != size:
            self._transform = PerspectiveTransform(size, self.cfg.src, self.cfg.dst)
        return self._transform

    def detect(self, bgr: np.ndarray) -> tuple[LaneResult, ClassicalDebug]:
        cfg = self.cfg
        if self.calibration is not None:
            bgr = self.calibration.undistort(bgr)
        h, w = bgr.shape[:2]
        tf = self.transform((w, h))
        warped = tf.warp(bgr)
        binary = threshold_binary(warped, cfg)

        if self.video and self._fits is not None:
            search = search_around(binary, *self._fits, cfg)
        else:
            search = sliding_window(binary, cfg)
        left = fit_polynomial(*search.left_xy, cfg.min_fit_pixels)
        right = fit_polynomial(*search.right_xy, cfg.min_fit_pixels)
        ok = left is not None and right is not None and lanes_plausible(left, right, w, h, cfg)

        if self.video:
            if ok and self._fits is not None:
                a = cfg.smoothing
                left = a * left + (1 - a) * self._fits[0]
                right = a * right + (1 - a) * self._fits[1]
            if ok:
                self._fits, self._lost = (left, right), 0
            else:
                self._lost += 1
                if self._fits is not None and self._lost <= cfg.max_lost_frames:
                    left, right = self._fits
                    ok = True
                else:
                    self._fits = None  # force a fresh sliding-window search
        debug = ClassicalDebug(warped, binary, search)
        if not ok:
            return LaneResult(size=(w, h)), debug

        lanes = {"left": self._to_image(tf, left, h), "right": self._to_image(tf, right, h)}
        curvature = np.mean([curvature_radius_m(f, h - 1, cfg.xm_per_pix, cfg.ym_per_pix)
                             for f in (left, right)])
        offset = vehicle_offset_m(left, right, w, h, cfg.xm_per_pix)
        return LaneResult(size=(w, h), lanes=lanes, curvature_m=float(curvature),
                          offset_m=offset,
                          departure=abs(offset) > cfg.departure_warning_m), debug

    def __call__(self, bgr: np.ndarray) -> LaneResult:
        return self.detect(bgr)[0]

    @staticmethod
    def _to_image(tf: PerspectiveTransform, fit: np.ndarray, height: int) -> np.ndarray:
        """Sample the bird's-eye polynomial and map it back to image coordinates."""
        ys = np.linspace(0, height - 1, 72)
        pts = np.stack([np.polyval(fit, ys), ys], axis=1)
        return tf.unwarp_points(pts)
