# Benchmark NanoDet + Landmark on Raspberry Pi 5

Start local monitor with performance logging:

```bash
python cli.py monitor --skip-auth --backend opencv --camera 0 --width 1280 --height 720 --profile --profile-every 30
```

For SSH without OpenCV GUI, add `--headless`. Press Ctrl+C to exit headless.

## What is measured

- `camera_read_ms`: time waiting for `camera.read()` (camera FPS or USB transfer can dominate this).
- `nanodet_pre_ms`, `nanodet_onnx_ms`, `nanodet_post_ms`: zoom + resize + blob; **ONNXRuntime session.run**; decoding, NMS, box mapping.
- `landmark_pre_ms`, `landmark_onnx_ms`, `landmark_post_ms`: face crop + conversion; **ONNXRuntime session.run**; landmark/pose decoding.
- `eye_geometry_ms`: eye EAR evaluation and calibration.
- `drowsiness_ms`: pose, mouth, temporal engine.
- `overlay_ms`: drawing + alert/event/eye CSV handling (not model inference).
- `process_total_ms`: complete monitor processing including NanoDet, Landmark and overlays.
- `display_ms`: OpenCV `imshow`/`waitKey` (0 when headless).
- `frame_total_ms`: one iteration, camera read through GUI; the CSV write is not included.

Printed metrics are **mean and p95** of the latest `--profile-every` iterations; `n=` indicates the number of iterations executing that stage. Landmark metrics are blank when no face is detected. CSV is written per frame to `logs/performance.csv`; change using `--profile-csv logs/my_run.csv`.

Take 30-60 seconds of readings after warm-up; ONNXRuntime often has a slow first inference. Compare `*_onnx_ms` to `*_pre_ms` and `*_post_ms` to distinguish actual neural-network cost from image and bounding-box operations. If `camera_read_ms` dominates, investigate camera capture FPS/format instead of changing models. For true throughput measurements turn off `EYE_LOG`, avoid running a debugger, and compare with/without `--headless`.

The profiling flags are opt-in. Without `--profile`, original workflow remains unchanged.
