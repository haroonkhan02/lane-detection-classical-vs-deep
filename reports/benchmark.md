# Model latency per backend

arm64 macOS-27.0.1-arm64-arm-64bit; input 1x3x288x512; median and 90th percentile of 100 runs after 10 warm-up runs, network only.

| Backend | Size (MB) | Median (ms) | p90 (ms) |
|---|---|---|---|
| pytorch_fp32 (cpu) | 97.9 | 71.5 | 72.5 |
| pytorch_fp32 (mps) | 97.9 | 14.9 | 15.7 |
| onnx_fp32 (onnxruntime CPU) | 97.7 | 39.0 | 47.3 |
| onnx_fp16 (onnxruntime CPU) | 48.9 | 37.2 | 38.0 |
| onnx_int8 (onnxruntime CPU) | 24.7 | 34.7 | 37.6 |
| coreml_fp16 (all) | 48.9 | 3.7 | 5.7 |
| coreml_fp16 (cpu_and_gpu) | 48.9 | 5.5 | 5.7 |
| coreml_fp16 (cpu_only) | 48.9 | 13.6 | 14.0 |
| coreml_fp32 (all) | 97.8 | 9.6 | 9.8 |
| coreml_fp32 (cpu_and_gpu) | 97.8 | 9.5 | 10.0 |
| coreml_fp32 (cpu_only) | 97.8 | 26.1 | 27.0 |
