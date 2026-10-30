# Exported models: parity and TuSimple test accuracy

2782 test images; parity on 100 of them against the PyTorch model on CPU.

| Variant | Size (MB) | Max prob diff | Same lane slots | Lane shift median / max (px) | Ego acc | Ego FP | Ego FN | All-lanes acc |
|---|---|---|---|---|---|---|---|---|
| pytorch_fp32 | 97.9 | reference | |  | 0.9507 | 0.0527 | 0.0514 | 0.9499 |
| onnx_fp32 | 97.7 | 1.5e-05 | 100/100 | 0.00 / 0.2 | 0.9507 | 0.0527 | 0.0514 | 0.9499 |
| onnx_fp16 | 48.9 | 0.0094 | 100/100 | 0.01 / 1.8 | 0.9507 | 0.0527 | 0.0514 | 0.9499 |
| onnx_int8 | 24.7 | 0.73 | 98/100 | 2.84 / 298.3 | 0.9498 | 0.0555 | 0.0543 | 0.9493 |
| coreml_fp16 | 48.9 | 0.025 | 100/100 | 8.24 / 17.3 | 0.9506 | 0.0530 | 0.0518 | 0.9498 |
| coreml_fp32 | 97.8 | 1.7e-05 | 100/100 | 0.00 / 0.2 | 0.9507 | 0.0527 | 0.0514 | 0.9499 |
