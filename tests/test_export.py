import numpy as np
import pytest
import torch

pytest.importorskip("onnxruntime")
pytest.importorskip("onnxconverter_common")

from lanedet.config import NUM_CLASSES  # noqa: E402
from lanedet.deep import export  # noqa: E402
from lanedet.deep.model import DeepLaneDetector, build_model  # noqa: E402
from lanedet.deep.train import save_checkpoint  # noqa: E402
from lanedet.export_models import lane_shift_px  # noqa: E402
from lanedet.types import LaneResult  # noqa: E402

SIZE = (160, 96)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("ckpt") / "m.pt"
    save_checkpoint(build_model("resnet18", pretrained=False).eval(), path, "resnet18", SIZE, {})
    return path


@pytest.fixture(scope="module")
def onnx_fp32(checkpoint, tmp_path_factory):
    wrapper, size = export.load_wrapped(checkpoint)
    return export.export_onnx(wrapper, size, tmp_path_factory.mktemp("onnx") / "m.onnx")


def _road(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (720, 1280, 3), dtype=np.uint8)


def test_wrapper_matches_detector_probabilities(checkpoint):
    wrapper, size = export.load_wrapped(checkpoint)
    img = _road()
    with torch.no_grad():
        probs = wrapper(torch.from_numpy(export.image_batch(img, size)))[0].numpy()
    ref = DeepLaneDetector(checkpoint, device="cpu").probabilities(img)
    np.testing.assert_allclose(probs, ref, atol=1e-5)
    np.testing.assert_allclose(probs.sum(axis=0), 1.0, atol=1e-5)


def test_onnx_fp32_parity_and_io(onnx_fp32, checkpoint):
    det = export.OnnxLaneDetector(onnx_fp32)
    assert det.size == SIZE
    img = _road(1)
    ref = DeepLaneDetector(checkpoint, device="cpu").probabilities(img)
    probs = det.probabilities(img)
    assert probs.shape == (NUM_CLASSES, SIZE[1], SIZE[0])
    assert np.abs(probs - ref).max() < 1e-4
    result = det(img)
    assert isinstance(result, LaneResult) and result.size == (1280, 720)


def test_fp16_halves_the_file_and_stays_close(onnx_fp32, tmp_path):
    fp16 = export.to_fp16(onnx_fp32, tmp_path / "m16.onnx")
    assert fp16.stat().st_size < 0.6 * onnx_fp32.stat().st_size
    img = _road(2)
    a = export.OnnxLaneDetector(onnx_fp32).probabilities(img)
    b = export.OnnxLaneDetector(fp16).probabilities(img)
    assert b.dtype == np.float32 and np.abs(a - b).max() < 0.05


def test_int8_quantises_with_calibration(onnx_fp32, tmp_path):
    batches = [export.image_batch(_road(s), SIZE) for s in range(4)]
    int8 = export.to_int8(onnx_fp32, tmp_path / "m8.onnx", batches)
    assert int8.stat().st_size < 0.4 * onnx_fp32.stat().st_size
    assert not (tmp_path / "m8_prep.onnx").exists()
    probs = export.OnnxLaneDetector(int8).probabilities(_road(5))
    np.testing.assert_allclose(probs.sum(axis=0), 1.0, atol=1e-3)  # softmax kept in float


def test_lane_shift_only_compares_shared_slots():
    ys = np.arange(300, 700, 10)
    line = np.stack([np.full(40, 100.0), np.arange(300, 700, 10.0)], axis=1)
    a = LaneResult((1280, 720), {"left": line, "right": line + [500, 0]})
    b = LaneResult((1280, 720), {"left": line + [3, 0]})
    assert lane_shift_px(a, b, ys) == pytest.approx(3.0)
    assert lane_shift_px(a, LaneResult((1280, 720)), ys) is None
