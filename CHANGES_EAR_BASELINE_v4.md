> Tài liệu lịch sử của V4. Logic V5 thay thế các ngưỡng gây UNKNOWN; xem `CHANGES_PERSONAL_EAR_v5.md`.

# V4 - Session EAR baseline + distance-aware quality

## Behavior

- Immediately after a successful API session start, or at `--skip-auth` monitor startup, require a valid 2.5-second frontal, eyes-open calibration interval **before enabling eye-derived warnings**. The server session API remains unchanged; this is pre-monitor calibration, not a pre-HTTP request.
- Calibrate each eye from raw WFLW-98 eye geometry (median three upper-to-lower eyelid distances divided by eye-corner distance), without requiring EAR >= 0.20.
- Store per-eye open EAR, per-eye pixel width, and baseline NanoDet face-box height in memory for the session only. Use upper 60% EAR observations to reduce blink contamination; small invalid samples are skipped.
- Closed if EAR <= 0.58 * personal open reference; open if EAR >= 0.78 * personal reference; in between UNKNOWN. Thresholds are initial experiment parameters, not clinically validated values.
- Quality gate uses `face_scale = current_detection_height / calibrated_detection_height`, `expected_eye_width = baseline_eye_width * face_scale`; dynamic minimum is `max(EAR_MIN_WIDTH_PX, 0.55 * expected_eye_width)` and, for near-frontal faces, require measured/expected eye width in [0.55,1.75]. Require 3 stable consecutive eye-width observations. Frame resolution remains a hard limit.
- Eye-derived drowsiness cannot be inferred while calibration is incomplete or tracking is unreliable. Pose/yawn/away logic is retained and API payloads unchanged. No eye boxes or `EYE_UNCERTAIN` warnings.
- Pose/mouth calibration gathers measurements in parallel when open-eye calibration provides a valid raw sample, instead of adding another 2 seconds.
- Press `R` to restart all calibration (eyes, head, mouth) without restarting the backend session. New driver sessions always discard the previous baseline.

## Settings

```env
EAR_CALIBRATION_SECONDS=2.5
EAR_CALIBRATION_MIN_SAMPLES=12
EAR_CALIBRATION_MAX_POSE=15
EAR_PERSONAL_CLOSED_RATIO=0.58
EAR_PERSONAL_OPEN_RATIO=0.78
EAR_EXPECTED_WIDTH_MIN_RATIO=0.55
EAR_STABILITY_FRAMES=3
EAR_MIN_WIDTH_PX=8
SHOW_EYE_DEBUG=1
EYE_LOG=1
```

## Limitations

- The baseline is inferred under an explicit eyes-open instruction; if the user deliberately keeps eyes shut for calibration, this method cannot independently confirm that the baseline represents open eyes.
- A smaller detection box is only a proxy for distance, also affected by head angle, detector jitter, camera optics and zoom. It does not improve true optical resolution.
- Notebook/model weights for Landmark V4 are not embedded; put the exported ONNX in `models/` and set `LANDMARK_MODEL` accordingly.
- Synthetic logic tests pass; end-to-end ONNX, camera throughput, threshold accuracy and real driving conditions are not validated in this environment.
