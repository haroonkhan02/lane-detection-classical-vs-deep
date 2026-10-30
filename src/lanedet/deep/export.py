"""Export the lane U-Net to ONNX (FP32, FP16, INT8) and CoreML, and run the exported files.

The exported graph takes an RGB image in [0, 1] (1, 3, H, W) and returns per-class
probabilities (1, C, H, W): ImageNet normalisation and the softmax live inside the graph, so a
browser or mobile client only has to resize the image and divide by 255.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch

from ..measure import add_measurements
from ..types import LaneResult
from .dataset import IMAGENET_MEAN, IMAGENET_STD, to_rgb01
from .model import build_model, extract_lanes

INPUT_NAME, OUTPUT_NAME = "image", "probs"


class ExportWrapper(torch.nn.Module):
    """Normalise -> U-Net -> softmax, so the exported graph is self-contained."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.model((x - self.mean) / self.std), dim=1)


def load_wrapped(checkpoint: str | Path) -> tuple[ExportWrapper, tuple[int, int]]:
    """Checkpoint -> (eval-mode wrapper on CPU, input size (w, h))."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(ckpt["encoder"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return ExportWrapper(model).eval(), tuple(ckpt["input_size"])


def image_batch(bgr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """BGR image -> (1, 3, h, w) float32 RGB in [0, 1], the exported graph's input."""
    return np.ascontiguousarray(to_rgb01(bgr, size).transpose(2, 0, 1)[None])


def export_onnx(wrapper: ExportWrapper, size: tuple[int, int], out: Path,
                opset: int = 17) -> Path:
    """Trace the wrapper to a single-file ONNX model with a fixed 1x3xHxW input."""
    import onnx

    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.rand(1, 3, size[1], size[0])
    torch.onnx.export(wrapper, (dummy,), str(out), input_names=[INPUT_NAME],
                      output_names=[OUTPUT_NAME], opset_version=opset, dynamo=False)
    model = onnx.load(str(out))  # re-save so weights are embedded, not external data
    onnx.checker.check_model(model)
    onnx.save(model, str(out))
    return out


def to_fp16(src: Path, out: Path) -> Path:
    """FP16 weights and activations; input and output stay float32 for the callers."""
    import onnx
    from onnxconverter_common import float16

    model = float16.convert_float_to_float16(onnx.load(str(src)), keep_io_types=True)
    onnx.save(model, str(out))
    return out


class _Calibration:
    """onnxruntime CalibrationDataReader over a list of preprocessed batches."""

    def __init__(self, batches: Iterable[np.ndarray]):
        self._it = iter(batches)

    def get_next(self):
        batch = next(self._it, None)
        return None if batch is None else {INPUT_NAME: batch}


def to_int8(src: Path, out: Path, calibration: Iterable[np.ndarray]) -> Path:
    """Static INT8 (QDQ, per-channel weights) calibrated on real images.

    The softmax stays in float: quantising probabilities to 8 bits would move the 0.4 lane
    threshold onto a coarse grid for no speed gain.
    """
    import onnx
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    prep = out.with_name(out.stem + "_prep.onnx")
    quant_pre_process(str(src), str(prep), skip_symbolic_shape=True)
    softmax = [n.name for n in onnx.load(str(prep)).graph.node if n.op_type == "Softmax"]
    quantize_static(str(prep), str(out), _Calibration(calibration),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax, nodes_to_exclude=softmax)
    prep.unlink()
    return out


def export_coreml(wrapper: ExportWrapper, size: tuple[int, int], out: Path,
                  fp16: bool = True) -> Path:
    """CoreML ML Program (Apple Neural Engine / GPU / CPU chosen by the OS at run time)."""
    import coremltools as ct

    traced = torch.jit.trace(wrapper, torch.rand(1, 3, size[1], size[0]))
    precision = ct.precision.FLOAT16 if fp16 else ct.precision.FLOAT32
    mlmodel = ct.convert(traced, inputs=[ct.TensorType(name=INPUT_NAME,
                                                       shape=(1, 3, size[1], size[0]))],
                         outputs=[ct.TensorType(name=OUTPUT_NAME)],
                         convert_to="mlprogram", compute_precision=precision,
                         minimum_deployment_target=ct.target.macOS13)
    mlmodel.save(str(out))
    return out


class _ExportedDetector:
    """Shared wrapper: preprocess -> backend probabilities -> the same lane extraction."""

    name = "exported"

    def __init__(self, size: tuple[int, int], thresh: float = 0.4, classical_cfg=None):
        self.size = size
        self.thresh = thresh
        self.classical_cfg = classical_cfg
        self.last_inference_ms = 0.0

    def run(self, batch: np.ndarray) -> np.ndarray:  # (1, C, h, w) probabilities
        raise NotImplementedError

    def probabilities(self, bgr: np.ndarray) -> np.ndarray:
        batch = image_batch(bgr, self.size)
        start = time.perf_counter()
        probs = self.run(batch)[0]
        self.last_inference_ms = (time.perf_counter() - start) * 1000
        return probs.astype(np.float32)

    def __call__(self, bgr: np.ndarray) -> LaneResult:
        h, w = bgr.shape[:2]
        result = LaneResult(size=(w, h),
                            lanes=extract_lanes(self.probabilities(bgr), (w, h), self.thresh))
        if self.classical_cfg is not None:
            add_measurements(result, self.classical_cfg)
        return result


class OnnxLaneDetector(_ExportedDetector):
    def __init__(self, path: str | Path, threads: int = 0, **kwargs):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        shape = self.session.get_inputs()[0].shape
        super().__init__((int(shape[3]), int(shape[2])), **kwargs)
        self.name = f"onnx:{Path(path).stem}"

    def run(self, batch: np.ndarray) -> np.ndarray:
        return self.session.run([OUTPUT_NAME], {INPUT_NAME: batch})[0]


class CoreMLLaneDetector(_ExportedDetector):
    def __init__(self, path: str | Path, size: tuple[int, int], compute_units: str = "ALL",
                 **kwargs):
        import coremltools as ct

        self.model = ct.models.MLModel(str(path),
                                       compute_units=getattr(ct.ComputeUnit, compute_units))
        super().__init__(size, **kwargs)
        self.name = f"coreml:{compute_units.lower()}"

    def run(self, batch: np.ndarray) -> np.ndarray:
        return self.model.predict({INPUT_NAME: batch})[OUTPUT_NAME]
