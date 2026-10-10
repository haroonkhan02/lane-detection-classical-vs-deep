"""Run a detector over a video and write an annotated copy.

Examples:
    lanes-video data/udacity/project_video.mp4 --method classical --config configs/udacity.yaml \
        --calibration models/udacity_calibration.npz -o outputs/project_classical.mp4
    lanes-video clip.mp4 --method both -o outputs/side_by_side.mp4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from .classical.calibration import Calibration
from .classical.config import ClassicalConfig
from .classical.pipeline import ClassicalLaneDetector
from .config import DEFAULT_CHECKPOINT, DEFAULT_CONFIG
from .viz import draw_hud, draw_lanes


def build_detectors(method: str, cfg: ClassicalConfig, calibration: Calibration | None,
                    checkpoint: Path) -> list:
    dets = []
    if method in ("classical", "both"):
        dets.append(ClassicalLaneDetector(cfg, calibration, video=True))
    if method in ("deep", "both"):
        from .deep.model import DeepLaneDetector

        dets.append(DeepLaneDetector(checkpoint, classical_cfg=cfg))
    return dets


def annotate(frame: np.ndarray, detector, calibration: Calibration | None) -> np.ndarray:
    start = time.perf_counter()
    result = detector(frame)
    fps = 1.0 / max(time.perf_counter() - start, 1e-6)
    base = calibration.undistort(frame) if calibration is not None else frame
    return draw_hud(draw_lanes(base, result), result, detector.name, fps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--method", choices=("classical", "deep", "both"), default="classical")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()

    cfg = ClassicalConfig.from_yaml(args.config)
    calib = Calibration.load(args.calibration) if args.calibration else None
    detectors = build_detectors(args.method, cfg, calib, args.checkpoint)

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    writer = None
    n = 0
    while args.max_frames is None or n < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        panels = [annotate(frame, d, calib) for d in detectors]
        out = np.hstack(panels) if len(panels) > 1 else panels[0]
        if writer is None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (out.shape[1], out.shape[0]))
        writer.write(out)
        n += 1
    cap.release()
    if writer is not None:
        writer.release()
    print(f"Wrote {n} frames to {args.output}")


if __name__ == "__main__":
    main()
