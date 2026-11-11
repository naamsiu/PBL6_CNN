import argparse
import json
import os
import sys

from app.config import (
    CAMERA_BACKEND,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    DEVICE_CODE,
    LANDMARK_MODEL,
    NANODET_MODEL,
    WEB_SERVER_URL,
)
from app.inference.landmark import Landmark98
from app.inference.nanodet import NanoDetFaceDetector
from main import (
    run_camera,
    run_context,
    run_monitor,
)


def ensure_display():
    if not os.environ.get(
        "DISPLAY"
    ):
        os.environ[
            "DISPLAY"
        ] = ":0"


def add_camera_args(parser):
    parser.add_argument(
        "--backend",
        choices=[
            "opencv",
            "picamera2",
        ],
        default=CAMERA_BACKEND,
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=CAMERA_INDEX,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=CAMERA_WIDTH,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=CAMERA_HEIGHT,
    )


def print_info():
    print()
    print("=" * 60)
    print("PBL6 RASPBERRY PI")
    print("=" * 60)
    print(
        "Server       :",
        WEB_SERVER_URL,
    )
    print(
        "Device code  :",
        DEVICE_CODE or "(optional)",
    )
    print(
        "Camera       :",
        CAMERA_BACKEND,
        CAMERA_INDEX,
    )
    print(
        "Resolution   :",
        f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}",
    )
    print(
        "NanoDet      :",
        NANODET_MODEL,
    )
    print(
        "Landmark     :",
        LANDMARK_MODEL,
    )
    print(
        "DISPLAY      :",
        os.environ.get(
            "DISPLAY",
            "(not set)",
        ),
    )
    print("=" * 60)
    print()


def print_model_info():
    print(
        "Loading NanoDet..."
    )
    nanodet = (
        NanoDetFaceDetector(
            NANODET_MODEL
        )
    )

    print(
        json.dumps(
            nanodet.info(),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )

    print()
    print(
        "Loading Landmark..."
    )

    landmark = Landmark98(
        LANDMARK_MODEL
    )

    print(
        json.dumps(
            landmark.info(),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )



def build_parser():
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description=(
            "PBL6 Raspberry Pi CLI"
        ),
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    camera = sub.add_parser(
        "camera",
        help="Test live camera.",
    )
    add_camera_args(camera)

    monitor = sub.add_parser(
        "monitor",
        help=(
            "Full flow: NanoDet -> "
            "server auth -> Landmark -> landmark EAR -> "
            "drowsiness local."
        ),
    )
    add_camera_args(monitor)

    monitor.add_argument(
        "--headless",
        action="store_true",
        help="Do not open a GUI window; useful over SSH/with headless OpenCV.",
    )
    monitor.add_argument(
        "--profile",
        action="store_true",
        help="Print NanoDet/Landmark preprocessing, ONNX and postprocessing timings; write CSV.",
    )
    monitor.add_argument(
        "--profile-every",
        type=int,
        default=30,
        metavar="N",
        help="Print average/p95 timings every N frames (default: 30).",
    )
    monitor.add_argument(
        "--profile-csv",
        default="logs/performance.csv",
        metavar="PATH",
        help="Per-frame timing CSV path (default: logs/performance.csv).",
    )
    monitor.add_argument(
        "--skip-auth",
        action="store_true",
        help=(
            "Bỏ server auth để test "
            "2 model local."
        ),
    )

    sub.add_parser(
        "context",
        help=(
            "Test device token/context API."
        ),
    )

    sub.add_parser(
        "model-info",
        help=(
            "In input/output shape "
            "của 2 ONNX."
        ),
    )

    sub.add_parser(
        "info",
        help="In config.",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command in {
        "camera",
        "monitor",
    }:
        ensure_display()

    try:
        if args.command == "camera":
            run_camera(
                args.backend,
                args.camera,
                args.width,
                args.height,
            )

        elif args.command == "monitor":
            run_monitor(
                args.backend,
                args.camera,
                args.width,
                args.height,
                skip_auth=args.skip_auth,
                headless=args.headless,
                profile=args.profile,
                profile_every=args.profile_every,
                profile_csv=args.profile_csv,
            )

        elif args.command == "context":
            run_context()

        elif args.command == "model-info":
            print_model_info()

        elif args.command == "info":
            print_info()

    except KeyboardInterrupt:
        print(
            "\nStopped by Ctrl+C."
        )
        sys.exit(130)


if __name__ == "__main__":
    main()
