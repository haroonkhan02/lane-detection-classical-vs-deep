"""Benchmark the classical and deep detectors on TuSimple with the official metric.

Two protocols are reported:
  * ego lanes  - only the two lanes bounding the car's lane (what the classical pipeline
                 is designed for, and what lane keeping needs);
  * all lanes  - the standard TuSimple protocol (up to 4 lanes per image).

Example:
    lanes-evaluate --split test --methods classical deep
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .classical.config import ClassicalConfig
from .classical.pipeline import ClassicalLaneDetector
from .config import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_ROOT,
    EGO_SLOTS,
    REPORTS_DIR,
    SLOTS,
    TUSIMPLE_TEST_JSON,
    TUSIMPLE_TRAIN_JSONS,
    TUSIMPLE_VAL_JSONS,
)
from .metrics import LaneEvalAccumulator
from .tusimple import LaneRecord, available_records, slot_lanes_from_record
from .types import LaneResult
from .viz import draw_lanes, mosaic

SPLITS = {"train": TUSIMPLE_TRAIN_JSONS, "val": TUSIMPLE_VAL_JSONS, "test": (TUSIMPLE_TEST_JSON,)}


def evaluate_detector(detector, records: list[LaneRecord], root: Path, progress: bool = True
                      ) -> tuple[dict, list[tuple[float, LaneRecord, LaneResult]]]:
    ego, full = LaneEvalAccumulator(), LaneEvalAccumulator()
    times, per_image = [], []
    for rec in tqdm(records, desc=getattr(detector, "name", "eval"), disable=not progress):
        img = cv2.imread(str(root / rec.raw_file))
        start = time.perf_counter()
        result = detector(img)
        times.append((time.perf_counter() - start) * 1000)

        gt_slots = slot_lanes_from_record(rec)
        gt_ego = [gt_slots[s] for s in EGO_SLOTS if s in gt_slots]
        acc, _, _ = ego.add(result.tusimple_lanes(rec.h_samples, EGO_SLOTS), gt_ego,
                            rec.h_samples)
        gt_all = [x for x in rec.lanes if any(v >= 0 for v in x)]
        full.add(result.tusimple_lanes(rec.h_samples, SLOTS), gt_all, rec.h_samples)
        per_image.append((acc, rec, result))
    summary = {"ego": ego.summary(), "all": full.summary(),
               "ms_per_frame": float(np.mean(times)) if times else 0.0}
    return summary, per_image


def gallery(rows: list[tuple[str, list]], root: Path, out: Path, n: int = 4) -> None:
    """Side-by-side images: one column per method, worst ego-accuracy cases first."""
    first = rows[0][1]
    order = np.argsort([acc for acc, _, _ in first])
    picks = list(order[:n // 2]) + list(order[len(order) // 2:len(order) // 2 + n - n // 2])
    tiles = []
    for idx in picks:
        rec = first[idx][1]
        img = cv2.imread(str(root / rec.raw_file))
        for name, per_image in rows:
            acc, _, result = per_image[idx]
            vis = draw_lanes(img, result)
            cv2.putText(vis, f"{name}: ego acc {acc:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.3, (255, 255, 255), 3, cv2.LINE_AA)
            tiles.append(vis)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), mosaic(tiles, cols=len(rows), width=560))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=tuple(SPLITS), default="test")
    parser.add_argument("--methods", nargs="+", choices=("classical", "deep"),
                        default=["classical", "deep"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    records = available_records(SPLITS[args.split], args.root)[:args.limit]
    if not records:
        raise SystemExit(f"No {args.split} images under {args.root}. Run `lanes-download`.")
    cfg = ClassicalConfig.from_yaml(args.config)

    results, rows = {}, []
    for method in args.methods:
        if method == "classical":
            det = ClassicalLaneDetector(cfg)
        else:
            from .deep.model import DeepLaneDetector

            det = DeepLaneDetector(args.checkpoint, classical_cfg=cfg)
        summary, per_image = evaluate_detector(det, records, args.root)
        results[method] = summary
        rows.append((method, per_image))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {"split": args.split, "images": len(records), "results": results}
    (REPORTS_DIR / f"results_{args.split}.json").write_text(json.dumps(out, indent=2))
    gallery(rows, args.root, REPORTS_DIR / f"gallery_{args.split}.jpg")

    print(f"TuSimple {args.split}: {len(records)} images")
    print(f"{'method':10s} {'ego acc':>8s} {'ego FP':>7s} {'ego FN':>7s} "
          f"{'all acc':>8s} {'all FP':>7s} {'all FN':>7s} {'ms/frame':>9s}")
    for m, r in results.items():
        e, a = r["ego"], r["all"]
        print(f"{m:10s} {e['accuracy']:8.3f} {e['fp_rate']:7.3f} {e['fn_rate']:7.3f} "
              f"{a['accuracy']:8.3f} {a['fp_rate']:7.3f} {a['fn_rate']:7.3f} "
              f"{r['ms_per_frame']:9.1f}")


if __name__ == "__main__":
    main()
