import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ============================================================
# SERVER / DEVICE
# ============================================================

WEB_SERVER_URL = os.getenv(
    "WEB_SERVER_URL",
    "http://10.177.62.189:8000",
).rstrip("/")

DEVICE_API_KEY = os.getenv(
    "DEVICE_API_KEY",
    "",
)

# Local test có thể chạy không token. Nếu backend yêu cầu Bearer token,
# chỉ cần điền DEVICE_API_KEY trong .env.


# Optional compatibility header.
DEVICE_CODE = os.getenv(
    "DEVICE_CODE",
    "",
)

# Một request: nhận diện + kiểm tra điều kiện + mở phiên.
SESSION_START_PATH = "/api/v1/device/sessions/start/"
# Kết thúc phiên lái khi tắt ứng dụng (Bearer token như các API khác).
SESSION_END_PATH = "/api/v1/device/sessions/{session_id}/end/"
CONTEXT_PATH = "/api/v1/device/context/"

# ============================================================
# CAMERA
# ============================================================

CAMERA_BACKEND = os.getenv(
    "CAMERA_BACKEND",
    "opencv",
).strip().lower()

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "1280"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "720"))
# Optional MJPG for UVC cameras that cannot output high resolutions in YUYV.
CAMERA_FOURCC = os.getenv("CAMERA_FOURCC", "").strip().upper()

# ============================================================
# MODEL
# ============================================================

NANODET_MODEL = os.getenv(
    "NANODET_MODEL",
    "models/nanodet_face_512_fp32.onnx",
)

LANDMARK_MODEL = os.getenv(
    "LANDMARK_MODEL",
    "models/landmark_v4_2_pi_256_f3f4_2conv_pose3.onnx",
)

ORT_THREADS = int(os.getenv("ORT_THREADS", "4"))

# NanoDet notebook configuration.
NANODET_INPUT_SIZE = 512
NANODET_REG_MAX = 7
NANODET_SCORE_THRESHOLD = float(
    os.getenv("NANODET_SCORE_THRESHOLD", "0.25")
)
NANODET_NMS_THRESHOLD = float(
    os.getenv("NANODET_NMS_THRESHOLD", "0.50")
)
# Center crop applies to NanoDet only. Landmark still uses the original
# camera frame, so bounding-box coordinates must be mapped back.
NANODET_CENTER_ZOOM = float(os.getenv("NANODET_CENTER_ZOOM", "1.0"))

# Landmark notebook configuration.
LANDMARK_INPUT_SIZE = 256
LANDMARK_CROP_SCALE = float(
    os.getenv("LANDMARK_CROP_SCALE", "1.25")
)
# Shift square crop downward by a fraction of the original detection height.
LANDMARK_CROP_Y_SHIFT = float(os.getenv("LANDMARK_CROP_Y_SHIFT", "0.0"))

# Only affect the displayed bounding box, never detection/landmark/pose/API.
FACE_BOX_PAD_X = float(os.getenv("FACE_BOX_PAD_X", "0.06"))
FACE_BOX_PAD_TOP = float(os.getenv("FACE_BOX_PAD_TOP", "0.05"))
FACE_BOX_PAD_BOTTOM = float(os.getenv("FACE_BOX_PAD_BOTTOM", "0.16"))

# ============================================================
# AUTH BEST-FRAME SELECTION
# ============================================================

AUTH_STABLE_FRAMES = int(
    os.getenv("AUTH_STABLE_FRAMES", "6")
)
AUTH_FACE_CONFIDENCE = float(
    os.getenv("AUTH_FACE_CONFIDENCE", "0.70")
)
AUTH_MIN_IOU = float(
    os.getenv("AUTH_MIN_IOU", "0.40")
)
AUTH_RETRY_SECONDS = float(
    os.getenv("AUTH_RETRY_SECONDS", "2.0")
)
# Nhận diện đúng nhưng DRIVER_NOT_ELIGIBLE (GPLX hết hạn, tài khoản khóa...):
# thử lại thưa hơn để không spam server với cùng một khuôn mặt.
AUTH_INELIGIBLE_RETRY_SECONDS = float(
    os.getenv("AUTH_INELIGIBLE_RETRY_SECONDS", "8.0")
)

# ============================================================
# IMAGE UPLOAD
# ============================================================

JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "85"))
MAX_UPLOAD_WIDTH = int(os.getenv("MAX_UPLOAD_WIDTH", "1280"))
MAX_UPLOAD_HEIGHT = int(os.getenv("MAX_UPLOAD_HEIGHT", "720"))
MAX_JPEG_BYTES = int(
    os.getenv(
        "MAX_JPEG_BYTES",
        str(2 * 1024 * 1024),
    )
)


def _model_path(value):
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else PROJECT_ROOT / path)

NANODET_MODEL = _model_path(NANODET_MODEL)
LANDMARK_MODEL = _model_path(LANDMARK_MODEL)
