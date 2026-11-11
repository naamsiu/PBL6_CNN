import cv2

from app.config import CAMERA_FOURCC


class Camera:
    def __init__(
        self,
        backend="opencv",
        camera_index=0,
        width=1280,
        height=720,
    ):
        self.backend = backend
        self.camera_index = camera_index
        self.width = width
        self.height = height

        self.cap = None
        self.picam2 = None

    def open(self):
        if self.backend == "picamera2":
            self._open_picamera2()
        else:
            self._open_opencv()

    def _open_opencv(self):
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Không mở được camera index {self.camera_index}. "
                "Kiểm tra /dev/video* hoặc thử --camera 1."
            )

        if CAMERA_FOURCC:
            if len(CAMERA_FOURCC) != 4:
                raise ValueError("CAMERA_FOURCC must contain exactly four letters")
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAMERA_FOURCC))

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            self.width,
        )
        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            self.height,
        )
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[CAMERA] Requested {self.width}x{self.height}; actual {actual_w}x{actual_h}")
        if actual_w and actual_h and (actual_w < self.width or actual_h < self.height):
            print("[CAMERA] Camera did not accept the requested resolution; "
                  "try supported camera modes or CAMERA_FOURCC=MJPG for UVC.")

    def _open_picamera2(self):
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError(
                "Không import được picamera2. "
                "Nếu dùng Pi Camera hãy cài python3-picamera2."
            ) from exc

        self.picam2 = Picamera2()

        config = self.picam2.create_preview_configuration(
            main={
                "size": (self.width, self.height),
                "format": "RGB888",
            }
        )

        self.picam2.configure(config)
        self.picam2.start()

    def read(self):
        if self.backend == "picamera2":
            frame = self.picam2.capture_array()

            if frame is None:
                return None

            return cv2.cvtColor(
                frame,
                cv2.COLOR_RGB2BGR,
            )

        ok, frame = self.cap.read()

        if not ok or frame is None:
            return None

        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.picam2 is not None:
            self.picam2.stop()
            self.picam2 = None
