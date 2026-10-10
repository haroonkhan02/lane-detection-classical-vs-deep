import json

import cv2
import numpy as np
import torch

from lanedet.config import NUM_CLASSES
from lanedet.deep.dataset import TuSimpleDataset
from lanedet.deep.model import DeepLaneDetector, build_model, extract_lanes
from lanedet.deep.train import save_checkpoint
from lanedet.tusimple import load_records

YS = list(range(260, 720, 10))


def test_extract_lanes_from_probability_maps():
    h, w = 72, 128
    probs = np.zeros((NUM_CLASSES, h, w), np.float32)
    probs[0] = 1.0
    probs[2, 20:, 30] = 0.9  # "left" lane: vertical at x=30 for rows 20..71
    lanes = extract_lanes(probs, (1280, 720))
    assert set(lanes) == {"left"}
    pts = lanes["left"]
    np.testing.assert_allclose(pts[:, 0], 300.0, atol=1e-3)  # scaled x10
    assert pts[:, 1].min() == 200.0 and pts[:, 1].max() == 710.0


def test_extract_lanes_ignores_weak_or_short_lanes():
    probs = np.zeros((NUM_CLASSES, 72, 128), np.float32)
    probs[3, :, 60] = 0.2  # below threshold
    probs[4, :5, 90] = 0.9  # too few rows
    assert extract_lanes(probs, (1280, 720)) == {}


def _fake_dataset(tmp_path):
    raw = "clips/x/1/20.jpg"
    (tmp_path / raw).parent.mkdir(parents=True)
    cv2.imwrite(str(tmp_path / raw), np.full((720, 1280, 3), 100, np.uint8))
    lanes = [[400 - 0.6 * (y - 719) for y in YS], [900 + 0.6 * (y - 719) for y in YS]]
    (tmp_path / "labels.json").write_text(
        json.dumps({"lanes": lanes, "h_samples": YS, "raw_file": raw}) + "\n")
    return load_records([tmp_path / "labels.json"])


def test_dataset_item_shapes(tmp_path):
    recs = _fake_dataset(tmp_path)
    for augment in (False, True):
        x, y = TuSimpleDataset(recs, tmp_path, (256, 144), augment=augment, seed=0)[0]
        assert x.shape == (3, 144, 256) and y.shape == (144, 256)
        assert set(torch.unique(y).tolist()) == {0, 2, 3}  # ego left/right only


def test_flip_keeps_left_on_the_left(tmp_path):
    recs = _fake_dataset(tmp_path)
    ds = TuSimpleDataset(recs, tmp_path, (256, 144), augment=True, seed=0)
    for _ in range(6):
        _, y = ds[0]
        left_x = torch.nonzero(y == 2)[:, 1].float().mean()
        right_x = torch.nonzero(y == 3)[:, 1].float().mean()
        assert left_x < right_x


def test_model_forward_and_checkpoint_roundtrip(tmp_path):
    model = build_model("resnet18", pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 96, 160))
    assert out.shape == (1, NUM_CLASSES, 96, 160)
    ckpt = tmp_path / "m.pt"
    save_checkpoint(model, ckpt, "resnet18", (160, 96), {})
    det = DeepLaneDetector(ckpt, device="cpu")
    result = det(np.zeros((720, 1280, 3), np.uint8))
    assert result.size == (1280, 720)


def test_seed_worker_gives_each_worker_its_own_rng(monkeypatch):
    from types import SimpleNamespace

    import lanedet.deep.dataset as dataset

    draws = []
    for worker_id in range(2):
        ds = dataset.TuSimpleDataset([], ".", augment=True, seed=0)
        info = SimpleNamespace(seed=1234 + worker_id, dataset=ds)
        monkeypatch.setattr(dataset.torch.utils.data, "get_worker_info", lambda info=info: info)
        dataset.seed_worker(worker_id)
        draws.append(ds.rng.random(5).tolist())
    assert draws[0] != draws[1]
