"""Opt-in per-frame performance logs for Raspberry Pi profiling.

Times are milliseconds measured via perf_counter. A missing Landmark value means
there was no valid face / no inference; it is NOT recorded as a 0ms inference.
"""

import csv
import math
import time
from pathlib import Path


FIELDS = (
    "camera_read_ms",
    "nanodet_pre_ms", "nanodet_onnx_ms", "nanodet_post_ms", "nanodet_total_ms",
    "landmark_pre_ms", "landmark_onnx_ms", "landmark_post_ms", "landmark_total_ms",
    "eye_geometry_ms", "drowsiness_ms", "overlay_ms",
    "process_total_ms", "display_ms", "frame_total_ms",
)


class PerformanceLogger:
    """CSV per frame, rolling mean and p95 every N frames."""

    def __init__(self, every=30, csv_path="logs/performance.csv"):
        self.every = int(every)
        if self.every < 1:
            raise ValueError("--profile-every must be >= 1")
        self.path = Path(csv_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", encoding="utf-8", newline="")
        self.csv = csv.DictWriter(
            self.file,
            fieldnames=("time", "frame", "faces", "face_detected", "error") + FIELDS,
            extrasaction="ignore",
        )
        self.csv.writeheader()
        self.window = []
        self.n_frames = 0
        print(f"[PERF] Enabled | every={self.every} frames | CSV={self.path}", flush=True)
        print("[PERF] ONNX = session.run only; PRE = crop/resize/blob; POST = decode/NMS/coordinates.", flush=True)

    def record(self, values):
        self.n_frames += 1
        row = {"time": f"{time.time():.3f}", "frame": self.n_frames}
        for key in ("faces", "face_detected", "error") + FIELDS:
            value = values.get(key)
            if value is None:
                row[key] = ""
            elif isinstance(value, float):
                row[key] = f"{value:.3f}" if math.isfinite(value) else ""
            else:
                row[key] = value
        self.csv.writerow(row)
        self.window.append(dict(values))
        if len(self.window) >= self.every:
            self.report()

    def report(self):
        if not self.window:
            return
        win = self.window
        n = len(win)

        def stats(name):
            vals = sorted(float(x[name]) for x in win
                          if isinstance(x.get(name), (int, float)) and math.isfinite(x[name]))
            if not vals:
                return None
            rank = max(0, math.ceil(0.95 * len(vals)) - 1)
            return (sum(vals) / len(vals), vals[rank], vals[-1], len(vals))

        def fmt(name):
            data = stats(name)
            if data is None:
                return "skip"
            return f"{data[0]:.1f}ms (p95 {data[1]:.1f}, n={data[3]})"

        total = stats("frame_total_ms")
        fps = 1000.0 / total[0] if total and total[0] > 0 else 0.0
        nfaces = sum(1 for x in win if x.get("face_detected"))
        print(f"[PERF] {n} frames | FPS~{fps:.1f} | face found {nfaces}/{n}", flush=True)
        print(f"[PERF] Camera read {fmt('camera_read_ms')} | Full frame {fmt('frame_total_ms')}", flush=True)
        print(f"[PERF] NanoDet total {fmt('nanodet_total_ms')} | PRE {fmt('nanodet_pre_ms')} | ONNX {fmt('nanodet_onnx_ms')} | POST {fmt('nanodet_post_ms')}", flush=True)
        print(f"[PERF] Landmark total {fmt('landmark_total_ms')} | PRE {fmt('landmark_pre_ms')} | ONNX {fmt('landmark_onnx_ms')} | POST {fmt('landmark_post_ms')}", flush=True)
        print(f"[PERF] EAR geometry {fmt('eye_geometry_ms')} | State engine {fmt('drowsiness_ms')} | Overlay/log {fmt('overlay_ms')} | GUI {fmt('display_ms')}", flush=True)
        stages = (
            ("Camera read", "camera_read_ms"),
            ("NanoDet", "nanodet_total_ms"),
            ("Landmark", "landmark_total_ms"),
            ("EAR", "eye_geometry_ms"),
            ("State engine", "drowsiness_ms"),
            ("Overlay/log", "overlay_ms"),
            ("GUI", "display_ms"),
        )
        valid = [(label, stats(key)) for label, key in stages if stats(key)]
        if valid:
            name, data = max(valid, key=lambda item: item[1][0])
            print(f"[PERF] Largest average stage (among executed): {name} {data[0]:.1f}ms | compare ONNX vs PRE/POST before changing models.", flush=True)
        self.file.flush()
        self.window.clear()

    def close(self):
        if not self.file.closed:
            if self.window:
                self.report()
            self.file.close()
