"""TuSimple lane evaluation (re-implementation of the official LaneEval).

For every ground-truth lane, a predicted point counts as correct when it is within
20 px (divided by cos of the lane's angle) of the ground truth at the same row. A lane
is *matched* when >= 85 % of its points are correct.

* Accuracy = mean over GT lanes of the best per-point accuracy of any prediction
* FP rate  = unmatched predictions / predictions
* FN rate  = unmatched GT lanes / GT lanes
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PIXEL_THRESH = 20.0
PT_THRESH = 0.85


def lane_angle(xs: np.ndarray, ys: np.ndarray) -> float:
    valid = xs >= 0
    if valid.sum() < 2:
        return 0.0
    slope = np.polyfit(ys[valid], xs[valid], 1)[0]
    return float(np.arctan(slope))


def line_accuracy(pred: np.ndarray, gt: np.ndarray, thresh: float) -> float:
    pred = np.where(pred >= 0, pred, -100.0)
    gt = np.where(gt >= 0, gt, -100.0)
    return float(np.mean(np.abs(pred - gt) < thresh))


def bench(pred: list[list[float]], gt: list[list[float]], y_samples: list[float]
          ) -> tuple[float, float, float]:
    """Accuracy, FP rate and FN rate for one image (same rules as the official script)."""
    ys = np.asarray(y_samples, dtype=float)
    if any(len(p) != len(ys) for p in pred):
        raise ValueError("every predicted lane must have one x per h_sample")
    if len(gt) + 2 < len(pred):
        return 0.0, 1.0, 1.0
    if not gt:
        return 1.0, (1.0 if pred else 0.0), 0.0

    preds = [np.asarray(p, dtype=float) for p in pred]
    line_accs, fn, matched = [], 0.0, 0.0
    for g in gt:
        g = np.asarray(g, dtype=float)
        thresh = PIXEL_THRESH / np.cos(lane_angle(g, ys))
        best = max((line_accuracy(p, g, thresh) for p in preds), default=0.0)
        if best < PT_THRESH:
            fn += 1
        else:
            matched += 1
        line_accs.append(best)

    fp = len(pred) - matched
    if len(gt) > 4 and fn > 0:
        fn -= 1
    total = sum(line_accs)
    if len(gt) > 4:
        total -= min(line_accs)
    acc = total / max(min(4.0, len(gt)), 1.0)
    return acc, (fp / len(pred) if pred else 0.0), fn / max(min(len(gt), 4.0), 1.0)


@dataclass
class LaneEvalAccumulator:
    accs: list[float] = field(default_factory=list)
    fps: list[float] = field(default_factory=list)
    fns: list[float] = field(default_factory=list)

    def add(self, pred, gt, y_samples) -> tuple[float, float, float]:
        acc, fp, fn = bench(pred, gt, y_samples)
        self.accs.append(acc)
        self.fps.append(fp)
        self.fns.append(fn)
        return acc, fp, fn

    def summary(self) -> dict[str, float]:
        n = max(len(self.accs), 1)
        return {"accuracy": sum(self.accs) / n, "fp_rate": sum(self.fps) / n,
                "fn_rate": sum(self.fns) / n, "images": len(self.accs)}
