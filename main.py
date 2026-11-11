import signal
import time

import cv2

from app.api_client import FaceClient
from app.camera import Camera
from app.config import (
    LANDMARK_MODEL,
    NANODET_MODEL,
)
from app.monitor import FullMonitor
from app.profiling import PerformanceLogger


WINDOW_NAME = (
    "PBL6 - Driver Monitor"
)


def build_camera(
    backend,
    camera_index,
    width,
    height,
):
    return Camera(
        backend=backend,
        camera_index=camera_index,
        width=width,
        height=height,
    )


def run_camera(
    backend,
    camera_index,
    width,
    height,
):
    camera = build_camera(
        backend,
        camera_index,
        width,
        height,
    )

    try:
        camera.open()

        print(
            "Camera đã mở. Q để thoát."
        )

        while True:
            frame = camera.read()

            if frame is None:
                time.sleep(0.05)
                continue

            cv2.putText(
                frame,
                "CAMERA TEST | Q: Quit",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                WINDOW_NAME,
                frame,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):
                break

    finally:
        camera.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # OpenCV headless: no HighGUI backend


def run_context():
    client = FaceClient()

    try:
        status_code, result = (
            client.context()
        )

        print(
            "HTTP:",
            status_code,
        )
        print(result)

    finally:
        client.close()


def run_monitor(
    backend,
    camera_index,
    width,
    height,
    skip_auth=False,
    headless=False,
    profile=False,
    profile_every=30,
    profile_csv="logs/performance.csv",
):
    camera = build_camera(
        backend,
        camera_index,
        width,
        height,
    )

    monitor = None
    profiler = PerformanceLogger(every=profile_every, csv_path=profile_csv) if profile else None

    # systemd stop / đóng terminal gửi SIGTERM/SIGHUP: đổi thành KeyboardInterrupt
    # để khối finally vẫn chạy và kịp end session.
    def _stop(signum, _frame):
        raise KeyboardInterrupt

    for name in ("SIGTERM", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _stop)

    try:
        camera.open()

        monitor = FullMonitor(
            require_auth=not skip_auth,
            profile=profile,
        )

        print()
        print(
            "Full monitor started."
        )
        print(
            "Q = quit"
        )
        print(
            "R = recalibrate eyes, pose, mouth (look straight, eyes open)"
        )

        if skip_auth:
            print(
                "WARNING: --skip-auth, "
                "local AI test only."
            )
        else:
            print(
                "NanoDet sẽ tự tìm đúng 1 mặt "
                "ổn định rồi gửi full frame "
                "lên server xác thực."
            )

        print()

        while True:
            frame_started = time.perf_counter()
            frame = camera.read()
            camera_ms = (time.perf_counter() - frame_started) * 1000.0

            if frame is None:
                display = monitor.camera_failed_display(width, height)
                time.sleep(0.02)
            else:
                display = monitor.process(frame)

            key = -1
            display_started = time.perf_counter()
            if not headless:
                try:
                    cv2.imshow(WINDOW_NAME, display)
                    key = cv2.waitKey(1) & 0xFF
                except cv2.error as exc:
                    raise RuntimeError(
                        "OpenCV cannot open a window. Use --headless for SSH "
                        "or install GUI-enabled opencv-python."
                    ) from exc

            if profiler is not None:
                profile_row = dict(monitor.last_profile) if frame is not None else {}
                profile_row.update({
                    "camera_read_ms": camera_ms,
                    "display_ms": (time.perf_counter() - display_started) * 1000.0,
                    "frame_total_ms": (time.perf_counter() - frame_started) * 1000.0,
                    "face_detected": bool(frame is not None and profile_row.get("faces", 0)),
                })
                profiler.record(profile_row)

            if headless:
                continue  # No HighGUI calls on SSH or opencv-python-headless.
            if key == ord("q"):
                break
            if key == ord("r"):
                monitor.reset_eye_calibration()


    except KeyboardInterrupt:
        print("\nĐang thoát...")

    finally:
        camera.release()

        if profiler is not None:
            profiler.close()

        if monitor is not None:
            monitor.close()

        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # OpenCV headless: no HighGUI backend