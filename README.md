<div align="center">

# 🛣️ Lane Detection: Classical CV vs. Deep Learning

**The same problem solved two ways — a hand-engineered OpenCV pipeline (camera calibration, bird's-eye warp, sliding windows, curvature) and a U-Net lane segmentation network — benchmarked head-to-head with the official TuSimple metric.**

[![CI](https://github.com/haroonkhan02/lane-detection-classical-vs-deep/actions/workflows/ci.yml/badge.svg)](https://github.com/haroonkhan02/lane-detection-classical-vs-deep/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-U--Net-ee4c2c)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

<img src="assets/classical_project_video.gif" width="640" alt="Classical pipeline on the Udacity project video">

<sub>Classical pipeline on the Udacity project video — lane area, curvature radius and offset from lane centre.</sub>

</div>

---

## Highlights

- **Two complete pipelines behind one interface** — both return lanes in the same format, so they share evaluation, visualization, video processing and the demo.
- **Fair benchmark** — classical parameters are tuned on the *train* split, the network checkpoint is chosen on *val*, and both are scored once on *test* with a re-implementation of the official TuSimple metric.
- **Real geometry** — camera calibration from chessboards, a homography to a bird's-eye view, and metric curvature / vehicle offset / lane-departure warning. The deep model's lanes reuse the same geometry for measurements.
- **Lane-slot segmentation** — the network predicts *which* lane each pixel belongs to (ego-left, ego-right, outer lanes), so no clustering step is needed.
- **Interactive demo** — a Gradio app shows both methods side by side plus the classical bird's-eye debug view.

## Results

Benchmarked on the TuSimple **test** images available in the public mirror (233 images). Official TuSimple metric; time per frame includes pre/post-processing on an Apple M5 (classical on CPU, deep on the GPU via MPS).

| Method | Ego acc ↑ | Ego FP ↓ | Ego FN ↓ | All-lanes acc ↑ | All FP ↓ | All FN ↓ | ms / frame |
|---|---|---|---|---|---|---|---|
| Classical (tuned on train) | 0.783 | 0.358 | 0.401 | 0.536 | 0.356 | 0.648 | **10.5** |
| U-Net ResNet-34 (60 epochs, best on val) | **0.914** | **0.148** | **0.148** | **0.922** | **0.223** | **0.138** | 20.7 |

*Ego* = the two lanes bounding the car's lane (what lane keeping needs, and all the classical pipeline looks for); *all lanes* = standard TuSimple protocol (classical scores low there by design).

<p align="center"><img src="reports/gallery_test.jpg" width="720" alt="Test gallery: classical vs deep"></p>
<p align="center"><sub>Hardest and median test cases. Classical (left) misses faint dashed markings entirely; the network (right) finds them — and the outer lanes.</sub></p>

**What the numbers say**

- On its home domain the network is clearly better: faint dashes and Botts' dots defeat brightness/gradient thresholds, while the U-Net has *learned* what a lane looks like. It also finds outer lanes the classical design never looks for.
- The classical pipeline is twice as fast, needs **no training data**, and every failure is explainable from its debug views.
- **Neither is robust off-domain.** On Udacity's challenge video (different camera, shadows, asphalt seams) the classical pipeline locks onto the dark seam beside the barrier, and the network — trained on only 233 TuSimple frames from another camera — produces unstable curves and a false departure warning:

<p align="center"><img src="assets/challenge_domain_shift.jpg" width="640" alt="Domain shift on the Udacity challenge video"></p>

> **Data caveat.** The public TuSimple mirror has only ~500 of ~6,400 labelled clips (233 train / 34 val / 233 test images here), so these numbers are a *subset* benchmark and not comparable to the published TuSimple leaderboard. Training on the full dataset uses the same commands.

Training history (loss and val accuracy per epoch) is in [`reports/training_history.json`](reports/training_history.json); raw metrics in [`reports/results_test.json`](reports/results_test.json).

## How it works

```mermaid
flowchart LR
    A[Road image] --> B1
    A --> C1
    subgraph Classical["Classical pipeline"]
        B1[Undistort<br/>camera calibration] --> B2[Bird's-eye warp<br/>homography]
        B2 --> B3[Threshold<br/>white · yellow · Sobel-x]
        B3 --> B4[Sliding windows<br/>+ 2nd-order fit]
        B4 --> B5[Unwarp]
    end
    subgraph Deep["Deep pipeline"]
        C1[U-Net · ResNet-34<br/>5-class lane slots] --> C2[Row-wise peak<br/>extraction + fit]
    end
    B5 --> D[Lanes per slot]
    C2 --> D
    D --> E[TuSimple metric]
    D --> F[Curvature · offset<br/>departure warning]
    D --> G[Overlay / video / demo]
```

**Classical.** Undistort with the calibrated camera model, warp the road trapezoid to a top-down view (lanes become parallel), threshold white (adaptive lightness percentile), yellow (LAB *b*) and horizontal gradients, find lane bases with a histogram, follow them upward with sliding windows and fit `x = Ay² + By + C`. In video mode it searches around the previous fit, smooths with an EMA and coasts through a few bad frames.

**Deep.** A U-Net with an ImageNet-pretrained ResNet-34 encoder segments the image into background + four *lane slots* (`left_left`, `left`, `right`, `right_right`), trained with weighted cross-entropy + Dice (lanes are ~2 % of pixels). Lanes are recovered by taking each row's probability-weighted peak per slot and fitting a low-order polynomial.

## Repository structure

```text
lane-detection-classical-vs-deep/
├── src/lanedet/
│   ├── classical/
│   │   ├── calibration.py   # Chessboard camera calibration
│   │   ├── geometry.py      # Homography (bird's-eye) + colour/gradient thresholds
│   │   ├── lane_fit.py      # Sliding windows, polyfit, curvature, offset, sanity checks
│   │   ├── pipeline.py      # ClassicalLaneDetector (image + video tracking)
│   │   ├── tune.py          # Grid-search parameters on the train split
│   │   └── config.py        # YAML-backed parameters
│   ├── deep/
│   │   ├── dataset.py       # TuSimple -> lane-slot masks, geometry-aware augmentation
│   │   ├── model.py         # U-Net, lane extraction, DeepLaneDetector
│   │   └── train.py         # Training loop (MPS / CUDA / CPU)
│   ├── tusimple.py          # Annotation parsing, slot assignment, mask rendering
│   ├── metrics.py           # Official TuSimple accuracy / FP / FN
│   ├── evaluate.py          # Benchmark both methods, write report + gallery
│   ├── measure.py           # Curvature/offset for any detector's lanes
│   ├── video.py             # Annotate videos (single or side-by-side)
│   ├── demo.py              # Gradio web demo
│   ├── viz.py               # Overlays, HUD, mosaics
│   ├── download.py          # TuSimple (HF mirror) + Udacity assets
│   └── types.py             # LaneResult shared by both detectors
├── configs/                 # Classical parameters (udacity.yaml, tusimple.yaml)
├── tests/                   # Unit tests on synthetic roads (no data / GPU needed)
├── reports/                 # Benchmark results, training history, galleries
├── assets/                  # README media
├── data/  models/           # Downloaded data & weights (git-ignored)
└── .github/workflows/       # CI: lint + tests on Python 3.10–3.12
```

## Getting started

```bash
git clone https://github.com/haroonkhan02/lane-detection-classical-vs-deep.git
cd lane-detection-classical-vs-deep
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[demo,dev]"

# Data
lanes-download udacity                       # chessboards + test images + videos (~30 MB)
lanes-download tusimple --split train        # labelled frames available in the mirror
lanes-download tusimple --split val
lanes-download tusimple --split test

# Classical pipeline
lanes-calibrate data/udacity/camera_cal -o models/udacity_calibration.npz
python -m lanedet.classical.tune             # writes configs/tusimple.yaml
lanes-video data/udacity/project_video.mp4 --method classical \
    --config configs/udacity.yaml --calibration models/udacity_calibration.npz \
    -o outputs/project_classical.mp4

# Deep pipeline
lanes-train --epochs 60                      # ~15 min on an Apple M5 (MPS)
lanes-evaluate --split test                  # both methods, writes reports/
lanes-demo                                   # http://127.0.0.1:7860
```

> **Using the full TuSimple dataset:** the public Hugging Face mirror only contains ~500 of the ~6,400 labelled clips. If you have the full dataset (e.g. from Kaggle), extract it into `data/tusimple/` with the standard layout (`clips/`, `label_data_*.json`, `test_label.json`) — every command above works unchanged.

## Design decisions

| Decision | Why |
|---|---|
| Lane *slots* instead of binary lane segmentation | The network directly tells ego-left from ego-right; no clustering or heuristics afterwards. |
| Slots assigned **after** augmentation | A horizontally flipped "left" lane is a "right" lane — assigning first would teach the model wrong labels (unit-tested). |
| Adaptive (percentile) white threshold | A fixed brightness threshold breaks between sunny and overcast frames. |
| Trapezoid fitted to median training lanes | Data-driven bird's-eye calibration for a camera with no published intrinsics. |
| Checkpoint selected on val TuSimple accuracy, not loss | Select on the metric you report. |
| Shared `LaneResult` type | Both detectors plug into the same metric, measurements, video tool and demo. |

## Testing

```bash
pytest     # synthetic roads and lanes — no dataset, camera or GPU needed
ruff check .
```

## Future work

- Train on the full TuSimple / CULane and add night & rain scenes (where the classical pipeline is expected to break down)
- Row-anchor classification (UFLD-style) for >200 FPS inference
- Temporal smoothing of the network output for video
- Deploy the Gradio demo to Hugging Face Spaces

## Acknowledgements

[TuSimple lane benchmark](https://github.com/TuSimple/tusimple-benchmark) (via the [Hugging Face mirror](https://huggingface.co/datasets/dhbloo/TuSimple)) · [Udacity CarND Advanced Lane Lines](https://github.com/udacity/CarND-Advanced-Lane-Lines) (calibration images and videos, MIT) · [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)

## License

[MIT](LICENSE) © Muhammad Haroon Khan
