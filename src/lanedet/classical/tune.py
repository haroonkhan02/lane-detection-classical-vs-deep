"""Tune the classical pipeline on the TuSimple *train* split (never on test).

1. Estimate the median ego-lane lines from the training labels; the bird's-eye
   trapezoid is placed on them so straight lanes become parallel after warping.
2. Grid-search the trapezoid's far edge and the threshold parameters, scoring the
   official TuSimple accuracy on ego lanes.
3. Write the best setting to configs/tusimple.yaml.

Example:
    python -m lanedet.classical.tune
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from ..config import CONFIGS_DIR, TUSIMPLE_DIR, TUSIMPLE_SIZE, TUSIMPLE_TRAIN_JSONS
from ..tusimple import LaneRecord, available_records, slot_lanes_from_record
from .config import ClassicalConfig
from .pipeline import ClassicalLaneDetector


def median_ego_lines(records: list[LaneRecord]) -> dict[str, tuple[float, float]]:
    """Median (slope, intercept) of x = slope * y + intercept for each ego lane."""
    lines: dict[str, list[tuple[float, float]]] = {"left": [], "right": []}
    for rec in records:
        slots = slot_lanes_from_record(rec)
        ys = np.asarray(rec.h_samples, dtype=float)
        for side in lines:
            if side not in slots:
                continue
            xs = np.asarray(slots[side], dtype=float)
            valid = (xs >= 0) & (ys >= 400)  # near field is ~straight
            if valid.sum() >= 5:
                lines[side].append(tuple(np.polyfit(ys[valid], xs[valid], 1)))
    return {k: tuple(np.median(np.array(v), axis=0)) for k, v in lines.items()}


def trapezoid(lines: dict[str, tuple[float, float]], top_y: float,
              size: tuple[int, int] = TUSIMPLE_SIZE) -> list[list[float]]:
    w, h = size
    pts = []
    for y, side in ((top_y, "left"), (top_y, "right"), (h - 1, "right"), (h - 1, "left")):
        slope, icpt = lines[side]
        pts.append([round(float((slope * y + icpt) / w), 4), round(float(y / h), 4)])
    return pts


def score(cfg: ClassicalConfig, records: list[LaneRecord], root: Path) -> float:
    from ..evaluate import evaluate_detector

    summary, _ = evaluate_detector(ClassicalLaneDetector(cfg), records, root, progress=False)
    return summary["ego"]["accuracy"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=TUSIMPLE_DIR)
    parser.add_argument("--output", type=Path, default=CONFIGS_DIR / "tusimple.yaml")
    parser.add_argument("--limit", type=int, help="use only N training images")
    args = parser.parse_args()

    records = available_records(TUSIMPLE_TRAIN_JSONS, args.root)[:args.limit]
    lines = median_ego_lines(records)
    base = ClassicalConfig(xm_per_pix=3.7 / 640, ym_per_pix=40 / 720)

    grid = {"top_y": (300, 320, 340, 380), "white_percentile": (97.0, 98.5, 99.3),
            "sobel_low": (15, 25, 40)}
    best = (-1.0, None, None)
    for top_y, pct, sob in itertools.product(*grid.values()):
        cfg = replace(base, src=trapezoid(lines, top_y), white_percentile=pct,
                      sobel_thresh=(sob, 255))
        acc = score(cfg, records, args.root)
        print(f"top_y={top_y} white_pct={pct} sobel_low={sob}: ego acc {acc:.3f}")
        if acc > best[0]:
            best = (acc, cfg, (top_y, pct, sob))

    acc, cfg, params = best
    out = {"src": cfg.src, "dst": cfg.dst, "white_percentile": cfg.white_percentile,
           "sobel_thresh": list(cfg.sobel_thresh), "xm_per_pix": cfg.xm_per_pix,
           "ym_per_pix": cfg.ym_per_pix}
    header = (f"# Tuned by lanedet.classical.tune on {len(records)} TuSimple train images\n"
              f"# (top_y, white_percentile, sobel_low) = {params}; train ego accuracy {acc:.3f}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(header + yaml.safe_dump(out, sort_keys=False))
    print(f"Best {params} -> ego acc {acc:.3f}; saved {args.output}")


if __name__ == "__main__":
    main()
