---
title: Lane Detection in the Browser
emoji: 🛣️
colorFrom: gray
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: TuSimple lane U-Net, INT8 ONNX, runs in your browser
models:
  - haroonx02/lane-detection-unet-resnet34
custom_headers:
  cross-origin-embedder-policy: credentialless
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---

A U-Net lane-slot segmentation model (ResNet-34 encoder, trained on TuSimple) exported to
ONNX, quantised to INT8 and run client-side with onnxruntime-web (WebAssembly). Pre- and
post-processing are ported to JavaScript and checked against the Python pipeline.

Code, training, evaluation and the classical OpenCV pipeline it is compared with:
[github.com/haroonkhan02/lane-detection-classical-vs-deep](https://github.com/haroonkhan02/lane-detection-classical-vs-deep).
