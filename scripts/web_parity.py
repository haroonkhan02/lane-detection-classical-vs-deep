"""Check the browser demo's JavaScript pipeline against the Python one on TuSimple test frames.

Dumps decoded frames, runs ``scripts/web_parity/parity.mjs`` (web/lanes.js + onnxruntime-web
WASM, the same code the static Space runs) and compares its lanes with ``OnnxLaneDetector`` on
the same ONNX file: lane shift in pixels and the TuSimple metric on both outputs.

Example (after `npm install` in scripts/web_parity):
    python scripts/web_parity.py --model models/onnx/lane_unet_int8.onnx --images 200
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from lanedet.config import DEFAULT_ROOT, EGO_SLOTS, OUTPUTS_DIR, REPORTS_DIR, SLOTS
from lanedet.deep.export import OnnxLaneDetector
from lanedet.export_models import lane_shift_px
from lanedet.metrics import LaneEvalAccumulator
from lanedet.tusimple import available_records, slot_lanes_from_record
from lanedet.types import LaneResult

ROOT = Path(__file__).resolve().parents[1]


def dump(records, root: Path, out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    images = []
    for i, rec in enumerate(records):
        rgb = cv2.cvtColor(cv2.imread(str(root / rec.raw_file)), cv2.COLOR_BGR2RGB)
        rgb.tofile(out / f"{i}.rgb")
        images.append({"file": f"{i}.rgb", "width": rgb.shape[1], "height": rgb.shape[0]})
    (out / "meta.json").write_text(json.dumps({"images": images}))


def score(acc_ego, acc_all, result: LaneResult, rec) -> None:
    gt = slot_lanes_from_record(rec)
    acc_ego.add(result.tusimple_lanes(rec.h_samples, EGO_SLOTS),
                [gt[s] for s in EGO_SLOTS if s in gt], rec.h_samples)
    acc_all.add(result.tusimple_lanes(rec.h_samples, SLOTS),
                [x for x in rec.lanes if any(v >= 0 for v in x)], rec.h_samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=ROOT / "models/onnx/lane_unet_int8.onnx")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--images", type=int, default=200)
    parser.add_argument("--work", type=Path, default=OUTPUTS_DIR / "web_parity")
    args = parser.parse_args()

    records = available_records(("test_label.json",), args.root)
    records = records[::max(1, len(records) // args.images)][:args.images]
    dump(records, args.root, args.work)
    subprocess.run(["node", str(ROOT / "scripts/web_parity/parity.mjs"), str(args.work),
                    str(args.model)], check=True)
    js = json.loads((args.work / "js_lanes.json").read_text())

    det = OnnxLaneDetector(args.model)
    py_ego, py_all, js_ego, js_all = (LaneEvalAccumulator() for _ in range(4))
    shifts, same = [], 0
    for rec, item in zip(records, js["results"], strict=True):
        img = cv2.imread(str(args.root / rec.raw_file))
        r_py = det(img)
        r_js = LaneResult(r_py.size, {k: np.array(v) for k, v in item["lanes"].items()})
        same += set(r_py.lanes) == set(r_js.lanes)
        shift = lane_shift_px(r_py, r_js, rec.h_samples)
        if shift is not None:
            shifts.append(shift)
        score(py_ego, py_all, r_py, rec)
        score(js_ego, js_all, r_js, rec)

    report = {"model": args.model.name, "images": len(records),
              "onnxruntime_web": js["ort_version"], "js_ms_per_frame_node_wasm":
              js["ms_per_frame"], "same_lane_slots": same,
              "lane_shift_px_median": float(np.median(shifts)),
              "lane_shift_px_max": float(np.max(shifts)),
              "python": {"ego": py_ego.summary(), "all": py_all.summary()},
              "javascript": {"ego": js_ego.summary(), "all": js_all.summary()}}
    (REPORTS_DIR / "web_parity.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    shutil.rmtree(args.work)


if __name__ == "__main__":
    main()
