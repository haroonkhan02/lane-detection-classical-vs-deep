"""Find lane pixels in a bird's-eye binary image and fit 2nd-order polynomials.

Polynomials are x = A*y^2 + B*y + C (x as a function of y, because lanes are near-
vertical in the top-down view and would not be functions of x).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ClassicalConfig


@dataclass
class LaneSearch:
    left_xy: tuple[np.ndarray, np.ndarray]  # (xs, ys) of pixels assigned to the left lane
    right_xy: tuple[np.ndarray, np.ndarray]
    windows: list[tuple[int, int, int, int]]  # (x0, y0, x1, y1) boxes, for visualization


def lane_bases(binary: np.ndarray, search: float = 0.35) -> tuple[int, int]:
    """Histogram of the bottom half; strongest column each side of the centre."""
    h, w = binary.shape
    hist = binary[h // 2:].sum(axis=0)
    mid = w // 2
    lo, hi = int(mid - search * w), int(mid + search * w)
    left = lo + int(np.argmax(hist[lo:mid]))
    right = mid + int(np.argmax(hist[mid:hi]))
    return left, right


def sliding_window(binary: np.ndarray, cfg: ClassicalConfig) -> LaneSearch:
    """Walk windows up from each lane base, re-centring on the pixels found."""
    h, w = binary.shape
    ys, xs = binary.nonzero()
    margin = int(cfg.margin * w)
    win_h = h // cfg.n_windows
    current = list(lane_bases(binary, cfg.base_search))
    keep: list[list[np.ndarray]] = [[], []]
    windows = []
    for k in range(cfg.n_windows):
        y1, y0 = h - k * win_h, h - (k + 1) * win_h
        for side in (0, 1):
            x0, x1 = current[side] - margin, current[side] + margin
            windows.append((x0, y0, x1, y1))
            inside = ((ys >= y0) & (ys < y1) & (xs >= x0) & (xs < x1)).nonzero()[0]
            keep[side].append(inside)
            if len(inside) > cfg.min_pixels:
                current[side] = int(xs[inside].mean())
    li, ri = np.concatenate(keep[0]), np.concatenate(keep[1])
    return LaneSearch((xs[li], ys[li]), (xs[ri], ys[ri]), windows)


def search_around(binary: np.ndarray, left_fit: np.ndarray, right_fit: np.ndarray,
                  cfg: ClassicalConfig) -> LaneSearch:
    """Video shortcut: look only near last frame's polynomials."""
    ys, xs = binary.nonzero()
    margin = cfg.margin * binary.shape[1]
    lm = np.abs(xs - np.polyval(left_fit, ys)) < margin
    rm = np.abs(xs - np.polyval(right_fit, ys)) < margin
    return LaneSearch((xs[lm], ys[lm]), (xs[rm], ys[rm]), [])


def fit_polynomial(xs: np.ndarray, ys: np.ndarray, min_pixels: int = 200) -> np.ndarray | None:
    if len(xs) < min_pixels or np.ptp(ys) < 10:
        return None
    return np.polyfit(ys, xs, 2)


def curvature_radius_m(fit: np.ndarray, y_eval: float, xm: float, ym: float) -> float:
    """Radius of curvature in metres at pixel row ``y_eval``.

    Rescale the pixel polynomial to metres (x_m = xm * x, y_m = ym * y), then
    R = (1 + (2Ay + B)^2)^(3/2) / |2A|.
    """
    a = fit[0] * xm / ym**2
    b = fit[1] * xm / ym
    y = y_eval * ym
    if abs(a) < 1e-9:
        return float("inf")
    return float((1 + (2 * a * y + b) ** 2) ** 1.5 / abs(2 * a))


def vehicle_offset_m(left_fit: np.ndarray, right_fit: np.ndarray, width: int, height: int,
                     xm: float) -> float:
    """Signed distance of the camera (image centre) from the lane centre; + = right."""
    y = height - 1
    lane_centre = (np.polyval(left_fit, y) + np.polyval(right_fit, y)) / 2
    return float((width / 2 - lane_centre) * xm)


def lanes_plausible(left_fit: np.ndarray, right_fit: np.ndarray, width: int, height: int,
                    cfg: ClassicalConfig) -> bool:
    """Reject pairs that are too narrow/wide or far from parallel."""
    ys = np.linspace(0, height - 1, 5)
    widths = np.polyval(right_fit, ys) - np.polyval(left_fit, ys)
    if widths.min() < cfg.min_lane_width * width or widths.max() > cfg.max_lane_width * width:
        return False
    return (widths.max() - widths.min()) / widths.mean() <= cfg.max_width_change
