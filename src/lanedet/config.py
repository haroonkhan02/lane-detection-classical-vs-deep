"""Paths and dataset constants."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
TUSIMPLE_DIR = DATA_DIR / "tusimple"
UDACITY_DIR = DATA_DIR / "udacity"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
CONFIGS_DIR = ROOT / "configs"
OUTPUTS_DIR = ROOT / "outputs"
TUSIMPLE_FULL_DIR = DATA_DIR / "tusimple_full"

# The model trained on the full TuSimple set is the default once it exists; otherwise the CLIs
# fall back to the subset model so a fresh clone still works after `lanes-download tusimple`.
FULL_CHECKPOINT = MODELS_DIR / "unet_resnet34_full.pt"
_HAS_FULL = FULL_CHECKPOINT.exists()
DEFAULT_CONFIG = CONFIGS_DIR / ("tusimple_full.yaml" if _HAS_FULL else "tusimple.yaml")
DEFAULT_CHECKPOINT = FULL_CHECKPOINT if _HAS_FULL else MODELS_DIR / "unet_resnet34.pt"
DEFAULT_ROOT = TUSIMPLE_FULL_DIR if TUSIMPLE_FULL_DIR.exists() else DATA_DIR / "tusimple"

# TuSimple benchmark
TUSIMPLE_REPO = "dhbloo/TuSimple"  # Hugging Face mirror of github.com/TuSimple/tusimple-benchmark
TUSIMPLE_TRAIN_JSONS = ("label_data_0313.json", "label_data_0601.json")
TUSIMPLE_VAL_JSONS = ("label_data_0531.json",)
TUSIMPLE_TEST_JSON = "test_label.json"
TUSIMPLE_SIZE = (1280, 720)  # (width, height)

# Lane "slots": each lane is assigned a fixed class by its position relative to the car,
# so the network predicts ego-left / ego-right etc. directly (SCNN-style).
SLOTS = ("left_left", "left", "right", "right_right")
NUM_CLASSES = len(SLOTS) + 1  # + background
EGO_SLOTS = ("left", "right")

# Udacity CarND Advanced Lane Lines (MIT licence): chessboards + test videos
UDACITY_RAW = "https://raw.githubusercontent.com/udacity/CarND-Advanced-Lane-Lines/master"
UDACITY_FILES = (
    *(f"camera_cal/calibration{i}.jpg" for i in range(1, 21)),
    *(f"test_images/{n}.jpg" for n in ("straight_lines1", "straight_lines2", "test1", "test2",
                                       "test3", "test4", "test5", "test6")),
    "project_video.mp4",
    "challenge_video.mp4",
)
