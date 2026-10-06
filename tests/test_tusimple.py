import json

import numpy as np

from lanedet.tusimple import (
    LaneRecord,
    assign_slots,
    available_records,
    load_records,
    render_mask,
    slot_lanes_from_record,
    x_at_bottom,
)
from lanedet.types import LaneResult

YS = list(range(260, 720, 10))


def line(x_bottom, slope):
    """Lane hitting x_bottom at y=719 with dx/dy = slope."""
    return [x_bottom + slope * (y - 719) for y in YS]


def make_record():
    lanes = [line(100, -1.2), line(400, -0.6), line(900, 0.6), line(1250, 1.2)]
    return LaneRecord("clips/a/1/20.jpg", lanes, YS)


def test_x_at_bottom():
    pts = np.array([[500 + 0.5 * (y - 719), y] for y in YS], dtype=float)
    assert abs(x_at_bottom(pts, 720) - 500) < 1e-6


def test_assign_slots_orders_by_distance_from_centre():
    rec = make_record()
    pts = [rec.lane_points(i) for i in range(4)]
    assert assign_slots(pts) == {"left": 1, "left_left": 0, "right": 2, "right_right": 3}


def test_assign_slots_ignores_extra_lanes():
    pts = [np.array([[x + 0.0, y] for y in YS]) for x in (50, 200, 400, 900, 1100, 1250)]
    slots = assign_slots(pts)
    assert len(slots) == 4 and slots["left"] == 2 and slots["right"] == 3


def test_render_mask_classes():
    rec = make_record()
    pts = [rec.lane_points(i) for i in range(4)]
    mask = render_mask(pts, assign_slots(pts), (512, 288))
    assert mask.shape == (288, 512)
    assert set(np.unique(mask)) == {0, 1, 2, 3, 4}


def test_slot_lanes_from_record():
    rec = make_record()
    assert slot_lanes_from_record(rec)["left"] == rec.lanes[1]


def test_load_and_filter_records(tmp_path):
    rec = make_record()
    (tmp_path / "labels.json").write_text(json.dumps(
        {"lanes": rec.lanes, "h_samples": YS, "raw_file": rec.raw_file}) + "\n")
    assert len(load_records([tmp_path / "labels.json"])) == 1
    assert available_records(["labels.json"], tmp_path) == []  # image missing
    img = tmp_path / rec.raw_file
    img.parent.mkdir(parents=True)
    img.write_bytes(b"x")
    assert len(available_records(["labels.json"], tmp_path)) == 1


def test_lane_result_x_at():
    pts = np.array([[100.0, 300.0], [200.0, 500.0]])
    r = LaneResult(size=(1280, 720), lanes={"left": pts})
    xs = r.x_at("left", [200, 300, 400, 500, 600])
    assert xs == [-2.0, 100.0, 150.0, 200.0, -2.0]
    assert r.x_at("right", [300]) == [-2.0]
    assert r.tusimple_lanes([300, 400], ["left", "right"]) == [[100.0, 150.0]]
