"""Synthetic road scenes for camera-free, dataset-free tests."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from lanedet.classical.config import ClassicalConfig
from lanedet.classical.geometry import PerspectiveTransform

W, H = 1280, 720


@pytest.fixture
def cfg() -> ClassicalConfig:
    return ClassicalConfig()


@pytest.fixture
def synthetic_road(cfg):
    """Camera image of a straight road whose lanes sit exactly at dst x = 0.25w / 0.75w.

    Built in the bird's-eye view (two vertical dashed white lines on asphalt) and then
    unwarped, so the classical detector should recover the lanes exactly.
    """
    tf = PerspectiveTransform((W, H), cfg.src, cfg.dst)
    rng = np.random.default_rng(0)
    bird = np.full((H, W, 3), 90, np.uint8) + rng.integers(0, 12, (H, W, 3), dtype=np.uint8)
    for x in (int(0.25 * W), int(0.75 * W)):
        for y0 in range(0, H, 80):
            cv2.rectangle(bird, (x - 8, y0), (x + 8, y0 + 50), (235, 235, 235), -1)
    camera = tf.unwarp(bird)
    camera[camera.sum(axis=2) == 0] = (120, 140, 160)  # sky/sides outside the warped area
    return camera, tf
