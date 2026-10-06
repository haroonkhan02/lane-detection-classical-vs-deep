"""Bird's-eye perspective transform and colour/gradient thresholding."""

from __future__ import annotations

import cv2
import numpy as np

from .config import ClassicalConfig


class PerspectiveTransform:
    """Homography between the road trapezoid (src) and a top-down rectangle (dst).

    A flat road seen through a pinhole camera is a plane, so one 3x3 homography maps
    it to a bird's-eye view where lane lines become (nearly) parallel.
    """

    def __init__(self, size: tuple[int, int], src_frac, dst_frac):
        w, h = size
        scale = np.array([w, h], dtype=np.float32)
        self.size = size
        self.src = (np.asarray(src_frac, dtype=np.float32) * scale).astype(np.float32)
        self.dst = (np.asarray(dst_frac, dtype=np.float32) * scale).astype(np.float32)
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.M_inv = cv2.getPerspectiveTransform(self.dst, self.src)

    def warp(self, img: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(img, self.M, self.size, flags=cv2.INTER_LINEAR)

    def unwarp(self, img: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(img, self.M_inv, self.size, flags=cv2.INTER_LINEAR)

    def unwarp_points(self, pts: np.ndarray) -> np.ndarray:
        """Map (N, 2) bird's-eye points back to image coordinates."""
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.M_inv).reshape(-1, 2)


def threshold_binary(bgr: np.ndarray, cfg: ClassicalConfig) -> np.ndarray:
    """Combine white, yellow and horizontal-gradient masks into a binary lane image.

    * White: HLS lightness above an *adaptive* percentile, so the threshold follows
      scene brightness (sunny vs. overcast) instead of being a fixed number.
    * Yellow: high values in the LAB b channel (blue-yellow axis).
    * Gradient: Sobel-x on lightness picks up lane edges in the top-down view.
    """
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lightness = hls[:, :, 1]

    road = lightness[lightness > 0]  # ignore black borders introduced by warping
    level = np.percentile(road, cfg.white_percentile) if road.size else 255
    white = lightness >= max(level, cfg.white_min)
    yellow = lab[:, :, 2] >= cfg.yellow_b_min

    sobel = np.abs(cv2.Sobel(lightness, cv2.CV_64F, 1, 0, ksize=cfg.sobel_kernel))
    sobel = np.uint8(255 * sobel / max(sobel.max(), 1e-6))
    grad = (sobel >= cfg.sobel_thresh[0]) & (sobel <= cfg.sobel_thresh[1])

    return ((white | yellow | grad) & (lightness > 0)).astype(np.uint8)
