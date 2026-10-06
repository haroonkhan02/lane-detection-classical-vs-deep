"""Classical-pipeline parameters, loaded from YAML (see configs/)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class ClassicalConfig:
    # Perspective transform, as fractions of (width, height); order: TL, TR, BR, BL
    src: list[list[float]] = field(default_factory=lambda: [
        [0.457, 0.632], [0.551, 0.632], [0.883, 1.0], [0.148, 1.0]])
    dst: list[list[float]] = field(default_factory=lambda: [
        [0.25, 0.0], [0.75, 0.0], [0.75, 1.0], [0.25, 1.0]])
    # Thresholds (applied in the bird's-eye view)
    white_percentile: float = 97.0  # adaptive brightness threshold on the L channel
    white_min: int = 160
    yellow_b_min: int = 150  # LAB b channel
    sobel_kernel: int = 5
    sobel_thresh: tuple[int, int] = (25, 255)
    # Sliding-window search
    n_windows: int = 9
    margin: float = 0.08  # window half-width as a fraction of the warped width
    min_pixels: int = 50
    min_fit_pixels: int = 200
    base_search: float = 0.35  # search lane bases within +-this fraction from the centre
    # Sanity checks on the lane pair (fractions of warped width)
    min_lane_width: float = 0.25
    max_lane_width: float = 0.75
    max_width_change: float = 0.35
    # Real-world scale for curvature / offset
    xm_per_pix: float = 3.7 / 640
    ym_per_pix: float = 30 / 720
    departure_warning_m: float = 0.5
    # Video smoothing
    smoothing: float = 0.3  # EMA weight of the new fit
    max_lost_frames: int = 5

    @classmethod
    def from_yaml(cls, path: str | Path) -> ClassicalConfig:
        data = yaml.safe_load(Path(path).read_text()) or {}
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
        if "sobel_thresh" in data:
            data["sobel_thresh"] = tuple(data["sobel_thresh"])
        return cls(**data)
