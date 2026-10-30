"""Export the U-Net to ONNX (FP32 / FP16 / INT8) and CoreML, check parity, re-score on TuSimple.

For every variant the report gives the file size, the largest probability difference to the
PyTorch model on ``--parity`` test images, how far the extracted lanes move (pixels at the
TuSimple sample rows), and the full TuSimple metric on the test split.

Example:
    lanes-export                       # all variants, full test set
    lanes-export --limit 200 --no-coreml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .config import (
    DEFAULT_CHECKPOINT,
    DEFAULT_ROOT,
    MODELS_DIR,
    REPORTS_DIR,
    SLOTS,
    TUSIMPLE_TEST_JSON,
    TUSIMPLE_TRAIN_JSONS,
)
from .evaluate import evaluate_detector
from .tusimple import LaneRecord, available_records

EXPORT_DIR = MODELS_DIR / "onnx"


def file_mb(path: Path) -> float:
    if path.is_dir():  # .mlpackage
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6
    return path.stat().st_size / 1e6


def lane_shift_px(a, b, ys) -> float | None:
    """Largest x difference (px) between two LaneResults over slots both of them found."""
    diffs = []
    for slot in SLOTS:
        if slot in a.lanes and slot in b.lanes:
            xa, xb = np.array(a.x_at(slot, ys)), np.array(b.x_at(slot, ys))
            ok = (xa >= 0) & (xb >= 0)
            if ok.any():
                diffs.append(float(np.abs(xa[ok] - xb[ok]).max()))
    return max(diffs) if diffs else None


def parity(reference, detector, records: list[LaneRecord], root: Path) -> dict:
    """Probability and lane-geometry agreement with the PyTorch reference."""
    prob_diff, shifts, same_slots = 0.0, [], 0
    for rec in records:
        img = cv2.imread(str(root / rec.raw_file))
        p_ref, p_det = reference.probabilities(img), detector.probabilities(img)
        prob_diff = max(prob_diff, float(np.abs(p_ref - p_det).max()))
        r_ref, r_det = reference(img), detector(img)
        same_slots += set(r_ref.lanes) == set(r_det.lanes)
        shift = lane_shift_px(r_ref, r_det, rec.h_samples)
        if shift is not None:
            shifts.append(shift)
    return {"images": len(records), "max_prob_diff": prob_diff,
            "same_lane_slots": same_slots,
            "lane_shift_px_median": float(np.median(shifts)) if shifts else 0.0,
            "lane_shift_px_max": float(max(shifts)) if shifts else 0.0}


def calibration_batches(root: Path, size: tuple[int, int], n: int, seed: int = 0):
    from .deep.export import image_batch

    records = available_records(TUSIMPLE_TRAIN_JSONS, root)
    picks = np.random.default_rng(seed).choice(len(records), size=min(n, len(records)),
                                               replace=False)
    for i in picks:
        yield image_batch(cv2.imread(str(root / records[i].raw_file)), size)


def build_variants(checkpoint: Path, root: Path, out_dir: Path, coreml: bool,
                   calib: int) -> dict[str, Path]:
    from .deep import export

    wrapper, size = export.load_wrapped(checkpoint)
    paths = {"onnx_fp32": export.export_onnx(wrapper, size, out_dir / "lane_unet_fp32.onnx")}
    paths["onnx_fp16"] = export.to_fp16(paths["onnx_fp32"], out_dir / "lane_unet_fp16.onnx")
    paths["onnx_int8"] = export.to_int8(paths["onnx_fp32"], out_dir / "lane_unet_int8.onnx",
                                        calibration_batches(root, size, calib))
    if coreml:
        try:
            for bits in (16, 32):
                paths[f"coreml_fp{bits}"] = export.export_coreml(
                    wrapper, size, out_dir / f"lane_unet_fp{bits}.mlpackage", fp16=bits == 16)
        except Exception as err:  # coremltools lags behind new torch releases
            print(f"CoreML export failed: {err}")
    return paths


def load_detector(kind: str, path: Path, size: tuple[int, int]):
    from .deep.export import CoreMLLaneDetector, OnnxLaneDetector

    det = OnnxLaneDetector(path) if kind.startswith("onnx") else CoreMLLaneDetector(path, size)
    det.name = kind
    return det


def markdown(report: dict) -> str:
    lines = ["# Exported models: parity and TuSimple test accuracy\n",
             f"{report['test_images']} test images; parity on {report['parity_images']} of them "
             "against the PyTorch model on CPU.\n",
             "| Variant | Size (MB) | Max prob diff | Same lane slots | Lane shift median / max "
             "(px) | Ego acc | Ego FP | Ego FN | All-lanes acc |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, r in report["variants"].items():
        e, a, p = r["metrics"]["ego"], r["metrics"]["all"], r.get("parity")
        par = (f"{p['max_prob_diff']:.2g} | {p['same_lane_slots']}/{p['images']} | "
               f"{p['lane_shift_px_median']:.2f} / {p['lane_shift_px_max']:.1f}"
               if p else "reference | | ")
        lines.append(f"| {name} | {r['size_mb']:.1f} | {par} | {e['accuracy']:.4f} | "
                     f"{e['fp_rate']:.4f} | {e['fn_rate']:.4f} | {a['accuracy']:.4f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=EXPORT_DIR)
    parser.add_argument("--limit", type=int, help="test images to score (default: all)")
    parser.add_argument("--parity", type=int, default=100, help="images for the parity check")
    parser.add_argument("--calib", type=int, default=128, help="INT8 calibration images (train)")
    parser.add_argument("--no-coreml", action="store_true")
    parser.add_argument("--skip-export", action="store_true", help="reuse files in --out-dir")
    args = parser.parse_args()

    from .deep.model import DeepLaneDetector

    records = available_records((TUSIMPLE_TEST_JSON,), args.root)[:args.limit]
    if not records:
        raise SystemExit(f"No test images under {args.root}. Run `lanes-download`.")
    if args.skip_export:
        names = {"onnx_fp32": "lane_unet_fp32.onnx", "onnx_fp16": "lane_unet_fp16.onnx",
                 "onnx_int8": "lane_unet_int8.onnx", "coreml_fp16": "lane_unet_fp16.mlpackage",
                 "coreml_fp32": "lane_unet_fp32.mlpackage"}
        paths = {k: args.out_dir / v for k, v in names.items() if (args.out_dir / v).exists()}
    else:
        paths = build_variants(args.checkpoint, args.root, args.out_dir, not args.no_coreml,
                               args.calib)

    reference = DeepLaneDetector(args.checkpoint, device="cpu")
    reference.name = "pytorch_fp32"
    parity_records = records[::max(1, len(records) // args.parity)][:args.parity]
    report = {"checkpoint": str(args.checkpoint), "test_images": len(records),
              "parity_images": len(parity_records), "variants": {}}
    summary, _ = evaluate_detector(reference, records, args.root)
    report["variants"]["pytorch_fp32"] = {"size_mb": file_mb(args.checkpoint), "metrics": summary}
    for kind, path in paths.items():
        det = load_detector(kind, path, reference.size)
        summary, _ = evaluate_detector(det, records, args.root)
        report["variants"][kind] = {"file": path.name, "size_mb": file_mb(path),
                                    "parity": parity(reference, det, parity_records, args.root),
                                    "metrics": summary}
        print(kind, json.dumps(report["variants"][kind]["parity"]),
              f"ego acc {summary['ego']['accuracy']:.4f}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "export.json").write_text(json.dumps(report, indent=2))
    (REPORTS_DIR / "export.md").write_text(markdown(report))
    print(markdown(report))


if __name__ == "__main__":
    main()
