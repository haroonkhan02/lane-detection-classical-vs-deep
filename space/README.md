---
title: Lane Detection Classical vs Deep
emoji: 🛣️
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: 6.29.1
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: OpenCV lane pipeline vs a U-Net, side by side
models:
  - haroonx02/lane-detection-unet-resnet34
---

Upload a front-camera road image and compare a classical OpenCV lane pipeline (bird's-eye
warp, thresholds, sliding windows) with a U-Net lane-slot segmentation model trained on
TuSimple (0.951 ego-lane accuracy on the TuSimple test set).

Code, training and full results: [github.com/haroonkhan02/lane-detection-classical-vs-deep](https://github.com/haroonkhan02/lane-detection-classical-vs-deep).
Runs on a free CPU Space, so the deep model takes a little longer here than on a GPU.
