import time
from datetime import datetime, timezone

import cv2


class BestFrameSelector:
    def __init__(
        self,
        stable_frames=6,
        confidence=0.7,
        min_iou=0.4,
        max_window_seconds=2.0,
    ):
        self.stable_frames = stable_frames
        self.confidence = confidence
        self.min_iou = min_iou
        self.max_window_seconds = max_window_seconds
        self.reset()

    def reset(self):
        self.previous_box = None
        self.count = 0
        self.best = None
        self.best_score = -1.0
        self.window_start = None

    @staticmethod
    def iou(a, b):
        overlap_w = max(
            0,
            min(a[2], b[2]) - max(a[0], b[0]),
        )
        overlap_h = max(
            0,
            min(a[3], b[3]) - max(a[1], b[1]),
        )
        overlap = overlap_w * overlap_h

        area_a = max(0, a[2] - a[0]) * max(
            0,
            a[3] - a[1],
        )
        area_b = max(0, b[2] - b[0]) * max(
            0,
            b[3] - b[1],
        )

        return overlap / max(
            1,
            area_a + area_b - overlap,
        )

    def update(
        self,
        frame_bgr,
        face_boxes,
    ):
        """
        face_boxes:
            [(x1,y1,x2,y2,confidence), ...]

        Return:
            (full_frame_bgr, captured_at)
        hoặc None.
        """

        boxes = [
            box
            for box in face_boxes
            if len(box) >= 5
            and box[4] >= self.confidence
        ]

        if len(boxes) != 1:
            self.reset()
            return None

        height, width = frame_bgr.shape[:2]
        raw = boxes[0]

        box = (
            max(0, min(width, int(raw[0]))),
            max(0, min(height, int(raw[1]))),
            max(0, min(width, int(raw[2]))),
            max(0, min(height, int(raw[3]))),
        )

        if (
            box[2] <= box[0]
            or box[3] <= box[1]
        ):
            self.reset()
            return None

        now = time.monotonic()

        moved = (
            self.previous_box is not None
            and self.iou(
                box,
                self.previous_box,
            ) < self.min_iou
        )

        timeout = (
            self.window_start is not None
            and now - self.window_start
            > self.max_window_seconds
        )

        if moved or timeout:
            self.reset()

        if self.window_start is None:
            self.window_start = now

        self.previous_box = box

        roi = frame_bgr[
            box[1]:box[3],
            box[0]:box[2],
        ]

        if roi.size == 0:
            self.reset()
            return None

        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY,
        )

        sharpness = cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()

        if sharpness > self.best_score:
            self.best_score = sharpness
            self.best = (
                frame_bgr.copy(),
                datetime.now(
                    timezone.utc
                ).isoformat(),
            )

        self.count += 1

        if self.count < self.stable_frames:
            return None

        best = self.best
        self.reset()
        return best
