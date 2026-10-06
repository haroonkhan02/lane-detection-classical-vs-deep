import numpy as np
import pytest

from lanedet.classical.calibration import calibrate
from lanedet.classical.config import ClassicalConfig
from lanedet.classical.geometry import PerspectiveTransform, threshold_binary
from lanedet.classical.lane_fit import (
    curvature_radius_m,
    fit_polynomial,
    lane_bases,
    lanes_plausible,
    vehicle_offset_m,
)
from lanedet.classical.pipeline import ClassicalLaneDetector
from lanedet.measure import add_measurements

W, H = 1280, 720


def test_perspective_roundtrip(cfg):
    tf = PerspectiveTransform((W, H), cfg.src, cfg.dst)
    pts = np.array([[300.0, 700.0], [640.0, 500.0]])
    back = tf.unwarp_points(np.array(
        [tf.M @ np.append(p, 1) for p in pts])[:, :2] / np.array(
        [tf.M @ np.append(p, 1) for p in pts])[:, 2:])
    np.testing.assert_allclose(back, pts, atol=1e-3)


def test_threshold_finds_white_lines(synthetic_road, cfg):
    image, tf = synthetic_road
    binary = threshold_binary(tf.warp(image), cfg)
    left, right = lane_bases(binary)
    assert abs(left - 0.25 * W) < 15 and abs(right - 0.75 * W) < 15


def test_detector_recovers_synthetic_lanes(synthetic_road, cfg):
    image, _ = synthetic_road
    result = ClassicalLaneDetector(cfg)(image)
    assert set(result.lanes) == {"left", "right"}
    assert abs(result.offset_m) < 0.1
    assert result.curvature_m > 2000  # straight road
    assert not result.departure


def test_detector_reports_nothing_on_blank_image(cfg):
    blank = np.full((H, W, 3), 100, np.uint8)
    assert ClassicalLaneDetector(cfg)(blank).lanes == {}


def test_video_mode_keeps_last_fit_briefly(synthetic_road, cfg):
    image, _ = synthetic_road
    det = ClassicalLaneDetector(cfg, video=True)
    assert det(image).lanes
    blank = np.full_like(image, 100)
    assert det(blank).lanes  # coasts on the previous fit
    for _ in range(cfg.max_lost_frames):
        det(blank)
    assert det(blank).lanes == {}


def test_curvature_of_known_circle():
    # x = y^2 / (2R) has curvature radius R at y = 0 (pixel units, scale 1)
    fit = np.array([1 / (2 * 500.0), 0.0, 0.0])
    assert curvature_radius_m(fit, 0.0, 1.0, 1.0) == pytest.approx(500.0)
    assert curvature_radius_m(np.array([0.0, 0.0, 10.0]), 0, 1, 1) == float("inf")


def test_vehicle_offset_sign():
    left, right = np.array([0, 0, 300.0]), np.array([0, 0, 900.0])
    assert vehicle_offset_m(left, right, 1280, 720, 0.01) == pytest.approx(0.4)  # car right


def test_fit_polynomial_needs_pixels():
    assert fit_polynomial(np.arange(10), np.arange(10)) is None
    ys = np.arange(0, 500, 1.0)
    np.testing.assert_allclose(fit_polynomial(2 * ys + 3, ys), [0, 2, 3], atol=1e-6)


def test_lanes_plausible(cfg):
    left, right = np.array([0, 0, 320.0]), np.array([0, 0, 960.0])
    assert lanes_plausible(left, right, W, H, cfg)
    assert not lanes_plausible(left, np.array([0, 0, 400.0]), W, H, cfg)  # too narrow
    assert not lanes_plausible(left, np.array([0.002, 0, 700.0]), W, H, cfg)  # diverging


def test_measurements_for_image_space_lanes(synthetic_road, cfg):
    image, _ = synthetic_road
    result = ClassicalLaneDetector(cfg)(image)
    result.curvature_m = result.offset_m = None
    add_measurements(result, cfg)
    assert abs(result.offset_m) < 0.1


def test_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("src: [[0,0],[1,0],[1,1],[0,1]]\nbogus: 1\n")
    with pytest.raises(ValueError):
        ClassicalConfig.from_yaml(p)


def test_calibration_needs_boards():
    with pytest.raises(ValueError):
        calibrate([np.zeros((100, 100, 3), np.uint8)] * 3)
