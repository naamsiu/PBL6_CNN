import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from app.api_client import FaceClient, ViolationOutbox
from app.best_frame import BestFrameSelector
from app.config import (
    AUTH_FACE_CONFIDENCE,
    AUTH_INELIGIBLE_RETRY_SECONDS,
    AUTH_MIN_IOU,
    AUTH_RETRY_SECONDS,
    AUTH_STABLE_FRAMES,
    LANDMARK_MODEL,
    NANODET_MODEL,
    FACE_BOX_PAD_X,
    FACE_BOX_PAD_TOP,
    FACE_BOX_PAD_BOTTOM,
)
from app.drowsiness import DrowsinessEngine
from app.inference.eye_landmarks import LandmarkEyes, unknown_eyes
from app.inference.landmark import Landmark98
from app.inference.nanodet import NanoDetFaceDetector


# Lý do server từ chối mở phiên (data.reason_code), chữ không dấu vì cv2.putText
# (font Hershey) không vẽ được dấu tiếng Việt.
REASON_TEXT = {
    "DRIVER_ACCOUNT_INACTIVE": "TAI KHOAN BI KHOA",
    "DRIVER_PROFILE_NOT_APPROVED": "HO SO CHUA DUOC DUYET",
    "DRIVER_PROFILE_INCOMPLETE": "HO SO THIEU THONG TIN",
    "LICENSE_MISSING": "CHUA CO GPLX",
    "LICENSE_EXPIRED": "GPLX HET HAN",
    "LICENSE_NOT_ACTIVE": "GPLX CHUA HOP LE",
    "LICENSE_ISSUE_DATE_INVALID": "NGAY CAP GPLX KHONG HOP LE",
    "FACE_PROFILE_NOT_APPROVED": "HO SO KHUON MAT CHUA DUYET",
    "FACE_PROFILE_CHANGED": "DU LIEU KHUON MAT DA THAY DOI",
    "VEHICLE_SPECIFICATIONS_MISSING": "XE THIEU THONG SO",
    "LICENSE_STANDARD_UNSUPPORTED": "HE HANG GPLX CHUA HO TRO",
    "LICENSE_CLASS_INCOMPATIBLE": "HANG GPLX KHONG DU LAI XE NAY",
    "DRIVER_NOT_ELIGIBLE": "TAI XE KHONG DU DIEU KIEN",
    "NO_ACTIVE_ASSIGNMENT": "CHUA DUOC PHAN CONG XE",
    "FACE_NOT_MATCHED": "KHUON MAT KHONG KHOP TAI XE NAO",
}

ORANGE = (0, 165, 255)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)


def _extract_session_info(result):
    """
    Response của POST /api/v1/device/sessions/start/:
      code, message, action, retry_after_seconds   : ngoài cùng
      data.verified / session_started / session_id : trạng thái
      data.reason_code                             : lý do không đủ điều kiện
      data.driver.{id, full_name}                  : tài xế vừa nhận diện
    """
    if not isinstance(result, dict):
        result = {}

    data = result.get("data")
    if not isinstance(data, dict):
        data = {}

    driver = data.get("driver")
    if not isinstance(driver, dict):
        driver = {}

    return {
        "code": result.get("code"),
        "message": result.get("message"),
        "action": result.get("action"),
        "request_id": result.get("request_id"),
        "retry_after_seconds": result.get("retry_after_seconds"),
        "has_verdict": "verified" in data,
        "verified": bool(data.get("verified", False)),
        "session_started": bool(data.get("session_started", False)),
        "session_id": data.get("session_id"),
        "reason_code": data.get("reason_code"),
        "driver_id": driver.get("id"),
        "driver_name": driver.get("full_name"),
    }


def _largest_face(
    detections,
):
    if not detections:
        return None

    return max(
        detections,
        key=lambda b: (
            max(0.0, b[2] - b[0])
            * max(0.0, b[3] - b[1])
        ),
    )


def _draw_text(
    frame,
    text,
    y,
    color=WHITE,
    scale=0.65,
    x=20,
):
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_box(
    frame,
    box,
    color,
    label=None,
):
    # Presentation ONLY: never feed expanded rectangle into landmark, EAR,
    # head-pose, or auth. Original detection bbox is left untouched.
    x1, y1, x2, y2 = map(float, box[:4])
    bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
    fh, fw = frame.shape[:2]
    x1 = int(round(max(0.0, x1 - FACE_BOX_PAD_X * bw)))
    x2 = int(round(min(fw - 1, x2 + FACE_BOX_PAD_X * bw)))
    y1 = int(round(max(0.0, y1 - FACE_BOX_PAD_TOP * bh)))
    y2 = int(round(min(fh - 1, y2 + FACE_BOX_PAD_BOTTOM * bh)))

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        3,
    )

    if label:
        ty = max(
            25,
            y1 - 10,
        )

        cv2.putText(
            frame,
            label,
            (x1, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )


def _draw_landmarks(
    frame,
    landmarks,
    drowsy=False,
):
    h, w = frame.shape[:2]

    for idx, point in enumerate(
        landmarks
    ):
        x = int(round(point[0]))
        y = int(round(point[1]))

        if (
            x < 0
            or x >= w
            or y < 0
            or y >= h
        ):
            continue

        if 60 <= idx < 76:
            # Eye landmarks.
            color = (
                (0, 0, 255)
                if drowsy
                else (0, 255, 255)
            )
            radius = 2
        elif 76 <= idx < 96:
            # Mouth.
            color = (255, 0, 255)
            radius = 2
        else:
            color = (255, 255, 0)
            radius = 1

        cv2.circle(
            frame,
            (x, y),
            radius,
            color,
            -1,
            lineType=cv2.LINE_AA,
        )


def _draw_alert_banner(
    frame,
    text,
    level,
):
    if not text or level <= 0:
        return

    h, w = frame.shape[:2]

    flash_on = int(time.monotonic() * 4) % 2 == 0

    if level >= 2:
        color = (0, 0, 255) if flash_on else (0, 0, 150)
    else:
        color = (0, 140, 255)

    cv2.rectangle(
        frame,
        (0, h - 70),
        (w, h),
        color,
        -1,
    )

    cv2.putText(
        frame,
        text,
        (20, h - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )


class FullMonitor:
    """
    Luồng:

    WAIT_DRIVER
      NanoDet -> stable face -> chọn best full frame
      -> background POST /api/v1/device/sessions/start/
         (server nhận diện + kiểm tra điều kiện + mở phiên)
         201 SESSION_STARTED            -> sang VERIFIED
         403 DRIVER_NOT_ELIGIBLE        -> hiện reason_code, thử lại thưa hơn
         verified=false / lỗi token,ảnh -> thử lại

    VERIFIED
      NanoDet -> Landmark98 -> EAR geometry -> temporal engine -> overlay local

    DROWSY -> durable outbox -> POST violations in background.
    """

    def __init__(
        self,
        require_auth=True,
        profile=False,
    ):
        print("[AI] Loading NanoDet:", NANODET_MODEL)
        self.face_detector = NanoDetFaceDetector(NANODET_MODEL)

        print("[AI] Loading Landmark:", LANDMARK_MODEL)
        self.landmark = Landmark98(LANDMARK_MODEL)
        self.profile_enabled = bool(profile)
        self.last_profile = {}
        self.face_detector.profile_enabled = self.profile_enabled
        self.landmark.profile_enabled = self.profile_enabled

        self.eye_landmarks = LandmarkEyes()
        print("[AI] Two ONNX models loaded (face detector + landmark).")

        self.require_auth = require_auth

        self.selector = BestFrameSelector(
            stable_frames=AUTH_STABLE_FRAMES,
            confidence=AUTH_FACE_CONFIDENCE,
            min_iou=AUTH_MIN_IOU,
        )

        self.drowsiness = DrowsinessEngine()

        self.client = FaceClient() if require_auth else None

        self.executor = (
            ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="face-auth",
            )
            if require_auth
            else None
        )

        self.violation_outbox = ViolationOutbox(FaceClient()) if require_auth else None
        self._violation_events = {}
        self._no_face_since = None
        self._no_face_last = None

        self.auth_future = None
        self.auth_retry_after = 0.0
        self.auth_status = None   # (text, color, action) hiện trên màn hình chờ

        self.session_id = None    # phiên đang mở trên server, cần end khi tắt app
        self._session_ended = False

        self._last_alert_key = None

        self.verified = not require_auth
        self.driver_info = (
            {
                "driver_id": "LOCAL_TEST",
                "driver_name": "LOCAL TEST",
                "session_id": None,
            }
            if not require_auth
            else None
        )

        # EYE_LOG=1 -> ghi EAR, chất lượng quan sát và bộ đếm thời gian.
        self._eye_log_file = None
        self._eye_log = None
        if os.getenv("EYE_LOG", "0") == "1":
            os.makedirs("logs", exist_ok=True)
            self._eye_log_file = open(
                os.path.join("logs", "eye_log.csv"),
                "w",
                newline="",
                encoding="utf-8",
            )
            self._eye_log = csv.writer(self._eye_log_file)
            self._eye_log.writerow(
                ["t", "state", "eye_status", "ear_l", "ear_r", "closed_seconds", "open_seconds", "recovery_seconds", "eye_reliable", "quality_reason", "eye_width_px", "perclos", "perclos_ready", "perclos_coverage", "geometry_ms", "alert_key", "base_ear_l", "base_ear_r", "face_scale", "expected_width_l", "ear_calibrated", "ear_sigma_l", "ear_sigma_r", "base_eye_width_l", "base_eye_width_r", "base_shape_jitter_l", "base_shape_jitter_r", "ear_filtered_l", "ear_filtered_r", "ear_close_l", "ear_close_r", "ear_open_l", "ear_open_r", "shape_deviation_l", "shape_deviation_r", "status_l", "status_r", "reason_l", "reason_r"]
            )
            print("[EYE_LOG] Ghi vào logs/eye_log.csv")

        self._ear_ready_logged = False
        if self.verified:
            self.drowsiness.reset()
            self.eye_landmarks.start_calibration()
            print('[EAR] V5 calibration: look straight with both eyes open for 3 seconds.')

    def end_session(self):
        """
        Báo server kết thúc phiên lái. Chỉ gọi 1 lần; lỗi mạng không làm app
        treo hay crash lúc tắt.
        """
        if self._session_ended or self.client is None:
            return

        # Request mở phiên còn đang bay dở lúc bấm thoát: server có thể đã mở
        # phiên rồi, đợi kết quả để còn đóng nó lại.
        if self.auth_future is not None:
            future = self.auth_future
            self.auth_future = None
            try:
                status_code, result = future.result(timeout=10)
                info = _extract_session_info(result)
                if 200 <= status_code < 300 and info["session_started"]:
                    self.session_id = info["session_id"]
            except Exception as exc:
                print("[SESSION] Không rõ kết quả request mở phiên dở dang:", exc)

        if self.session_id is None:
            return

        for attempt in (1, 2):
            try:
                status_code, result = self.client.end_session(self.session_id)
            except Exception as exc:
                print(f"[SESSION] End lần {attempt} lỗi:", exc)
                continue

            info = _extract_session_info(result)
            print(
                f"[SESSION] End session {self.session_id}: HTTP={status_code} "
                f"code={info['code']} message={info['message']}"
            )

            # 2xx: xong. 4xx (vd phiên đã kết thúc/không tồn tại): retry không đổi được gì.
            if status_code < 500:
                self._session_ended = True
                return

        print(
            f"[SESSION] CHƯA end được session {self.session_id}. "
            "Cần đóng thủ công hoặc để server tự timeout."
        )

    def close(self):
        if self.violation_outbox is not None:
            self.violation_outbox.close()
        self.end_session()

        if self.client is not None:
            self.client.close()

        if self.executor is not None:
            self.executor.shutdown(
                wait=False,
                cancel_futures=True,
            )

        if self._eye_log_file is not None:
            self._eye_log_file.close()
            self._eye_log_file = None
            self._eye_log = None

    # ------------------------------------------------------------
    # AUTH / SESSION START
    # ------------------------------------------------------------

    def _start_auth(
        self,
        frame,
        captured_at,
    ):
        if self.auth_future is not None:
            return

        print("[AUTH] Stable face -> POST sessions/start ...")

        self.auth_future = self.executor.submit(
            self.client.start_session,
            frame,
            captured_at,
        )

    def _auth_failed(
        self,
        text,
        retry_seconds,
        color=ORANGE,
        action=None,
    ):
        self.auth_status = (text, color, action)
        self.auth_retry_after = time.monotonic() + float(retry_seconds)
        self.selector.reset()

    def _on_session_started(self, info):
        self.verified = True
        self.driver_info = info
        self.session_id = info["session_id"]
        self._session_ended = False
        self.auth_status = None
        self.drowsiness.reset()
        self.eye_landmarks.start_calibration()  # no cross-driver baseline leakage
        self._ear_ready_logged = False
        print('[EAR] Session started: look straight with eyes open for 2-3 seconds.')

        print()
        print("=" * 60)
        print("OK - SESSION STARTED")
        print("=" * 60)
        print("Driver ID   :", info["driver_id"])
        print("Driver name :", info["driver_name"])
        print("Session ID  :", info["session_id"])
        print("=> START LOCAL DROWSINESS MONITORING")
        print("=" * 60)
        print()

    def _poll_auth(self):
        if self.auth_future is None or not self.auth_future.done():
            return

        future = self.auth_future
        self.auth_future = None

        try:
            status_code, result = future.result()
        except Exception as exc:
            print("[AUTH] Request error:", exc)
            self._auth_failed("LOI KET NOI SERVER", AUTH_RETRY_SECONDS)
            return

        info = _extract_session_info(result)

        print(
            f"[AUTH] HTTP={status_code} code={info['code']} "
            f"reason={info['reason_code']} message={info['message']}"
        )

        # 1) Nhận diện đúng + đủ điều kiện + phiên đã mở (HTTP 201 SESSION_STARTED).
        if 200 <= status_code < 300 and info["session_started"]:
            self._on_session_started(info)
            return

        # 2) Nhận diện đúng nhưng không đủ điều kiện (HTTP 403 DRIVER_NOT_ELIGIBLE).
        if info["verified"] and info["driver_id"] is not None:
            reason = info["reason_code"] or info["code"] or "NOT_ELIGIBLE"
            who = info["driver_name"] or f"ID {info['driver_id']}"

            self._auth_failed(
                f"{who}: {REASON_TEXT.get(reason, reason)}",
                AUTH_INELIGIBLE_RETRY_SECONDS,
                RED,
                info["action"],
            )
            return

        # 3a) Server trả verdict nhưng không nhận diện được ai.
        if info["has_verdict"]:
            self._auth_failed(
                "KHONG NHAN DIEN DUOC",
                info["retry_after_seconds"] or AUTH_RETRY_SECONDS,
                ORANGE,
            )
            return

        # 3b) Server từ chối kèm code (vd NO_ACTIVE_ASSIGNMENT) hoặc lỗi token/ảnh/server.
        code = info["code"] or "UNKNOWN"
        retry = info["retry_after_seconds"] or (
            AUTH_INELIGIBLE_RETRY_SECONDS if status_code == 403 else AUTH_RETRY_SECONDS
        )
        self._auth_failed(
            f"{REASON_TEXT.get(code, f'LOI {status_code}: {code}')}",
            retry,
            RED if status_code == 403 else ORANGE,
        )

    # ------------------------------------------------------------
    # FRAME LOOP
    # ------------------------------------------------------------

    def process(
        self,
        frame,
    ):
        started = time.perf_counter()
        if self.profile_enabled:
            self.last_profile = {}

        try:
            detections = self.face_detector.detect(frame)
            if self.profile_enabled:
                self.last_profile.update(self.face_detector.last_timing)
                self.last_profile["faces"] = len(detections)
        except Exception as exc:
            display = frame.copy()
            state = self.drowsiness.observation_failed("FACE_DETECTOR_ERROR")
            self._show_state(display, state)
            _draw_text(display, str(exc)[:85], 430, ORANGE, 0.45)
            if self.profile_enabled:
                self.last_profile["process_total_ms"] = (time.perf_counter() - started) * 1000.0
                self.last_profile["error"] = "nanodet"
            return display

        display = frame.copy()

        if not self.verified:
            self._poll_auth()

        if not self.verified:
            self._draw_auth_state(display, frame, detections)
        else:
            self._draw_monitor_state(display, frame, detections)

        elapsed = max(time.perf_counter() - started, 1e-6)
        w = display.shape[1]

        _draw_text(
            display,
            f"FPS {1.0 / elapsed:.1f}",
            35,
            WHITE,
            0.60,
            x=max(20, w - 150),
        )

        if self.profile_enabled:
            self.last_profile["process_total_ms"] = (time.perf_counter() - started) * 1000.0
        return display

    def _draw_auth_state(
        self,
        display,
        frame,
        detections,
    ):
        _draw_text(display, "STATE: WAIT DRIVER", 35, YELLOW, 0.75)

        if self.auth_future is not None:
            _draw_text(display, "AUTHENTICATING...", 70, YELLOW, 0.75)
        else:
            _draw_text(display, "Keep exactly 1 face stable", 70, WHITE)

        for box in detections:
            color = (
                YELLOW
                if box[4] >= AUTH_FACE_CONFIDENCE
                else (120, 120, 120)
            )

            _draw_box(display, box, color, f"face {box[4]:.2f}")

        now = time.monotonic()

        if self.auth_future is None and now >= self.auth_retry_after:
            selected = self.selector.update(frame, detections)

            if selected is not None:
                best_frame, captured_at = selected
                self._start_auth(best_frame, captured_at)

        y = 105
        _draw_text(display, f"Faces: {len(detections)}", y, WHITE)

        if self.auth_status is not None and self.auth_future is None:
            text, color, action = self.auth_status

            y += 40
            _draw_text(display, text, y, color, 0.70)

            remaining = self.auth_retry_after - now
            if remaining > 0:
                y += 30
                _draw_text(
                    display,
                    f"Thu lai sau {remaining:.0f}s",
                    y,
                    WHITE,
                    0.55,
                )

            if action == "CONTACT_ADMIN":
                y += 30
                _draw_text(display, "LIEN HE QUAN TRI VIEN", y, color, 0.60)

        _draw_text(display, "Q: Quit", y + 35, WHITE)


    def _handle_alert(
        self,
        state,
    ):
        key = state.get("alert_key")

        if key != self._last_alert_key:
            if key:
                print("[ALERT]", state.get("alert_text"))
            elif self._last_alert_key:
                print("[ALERT CLEARED] Driver returned to a normal monitored state")
            self._last_alert_key = key

    def _log_eye(self, state):
        if self._eye_log is None:
            return
        values = [f"{time.monotonic():.3f}"]
        for key in ("state", "eye_status", "ear_left", "ear_right", "closed_seconds",
                    "open_seconds", "recovery_seconds", "eye_reliable", "eye_quality_reason",
                    "eye_width_px", "perclos", "perclos_ready", "perclos_coverage", "geometry_ms", "alert_key"):
            value = (state.get("eye_observation", {}).get("left", {}).get("ear") if key == "ear_left" else
                     state.get("eye_observation", {}).get("right", {}).get("ear") if key == "ear_right" else state.get(key))
            values.append("" if value is None else value)
        cal = state.get('ear_calibration', {})
        left = state.get('eye_observation', {}).get('left', {})
        right = state.get('eye_observation', {}).get('right', {})
        values.extend([cal.get('ear', {}).get('left', ''), cal.get('ear', {}).get('right', ''),
                       left.get('face_scale', ''), left.get('expected_width_px', ''),
                       cal.get('calibrated', False)])
        for stat in ('variability', 'width_px', 'landmark_stability'):
            values.extend([cal.get(stat, {}).get(side, '') for side in ('left', 'right')])
        for field in ('filtered_ear', 'closed_threshold', 'open_threshold',
                      'landmark_deviation', 'status', 'reason'):
            values.extend([obs.get(field, '') for obs in (left, right)])
        self._eye_log.writerow(values)
        self._eye_log_file.flush()

    def _draw_monitor_state(self, display, frame, detections):
        face = _largest_face(detections)
        if face is None:
            self.eye_landmarks.note_unobserved()
            state = self.drowsiness.no_face()
            self._show_state(display, state, evidence_frame=frame)
            return
        try:
            landmarks = self.landmark.predict(frame, face)
            if self.profile_enabled:
                self.last_profile.update(self.landmark.last_timing)
        except Exception as exc:
            self.eye_landmarks.note_unobserved()
            state = self.drowsiness.landmark_failed()
            self._show_state(display, state, face=face)
            _draw_text(display, str(exc)[:85], 430, ORANGE, 0.45)
            return
        started = time.perf_counter()
        now = time.monotonic()
        if getattr(self.landmark, "has_pose", False) and landmarks.get("pose") is None:
            self.eye_landmarks.note_unobserved()
            observation = unknown_eyes("MISSING_POSE")
        else:
            observation = self.eye_landmarks.predict(
                landmarks["frame"], landmarks.get("pose"),
                frame_shape=frame.shape, face_bbox=face, now=now)
        geometry_ms = (time.perf_counter() - started) * 1000.0
        engine_started = time.perf_counter() if self.profile_enabled else None
        state = self.drowsiness.update(
            landmarks["normalized"], face_bbox=face, frame_shape=frame.shape,
            pose=landmarks.get("pose"), landmark_frame=landmarks["frame"],
            eye_observation=observation, now=now,
        )
        if self.profile_enabled:
            self.last_profile["eye_geometry_ms"] = geometry_ms
            self.last_profile["drowsiness_ms"] = (time.perf_counter() - engine_started) * 1000.0
        state['ear_calibration'] = self.eye_landmarks.calibration_info()
        if state['ear_calibration']['calibrated'] and not self._ear_ready_logged:
            c = state['ear_calibration']
            print('[EAR] V5 ready | baseline L/R:', c['ear'],
                  '| sigma L/R:', c['variability'],
                  '| shape stability L/R:', c['landmark_stability'],
                  '| face height:', round(c['face_height_px'], 1))
            self._ear_ready_logged = True
        state["geometry_ms"] = geometry_ms
        self._show_state(display, state, face=face, landmarks=landmarks["frame"], evidence_frame=frame)

    def _queue_violation(self, frame, state):
        """Independent episodes per code; technical failures never fabricate evidence."""
        now = time.monotonic()
        reason = state.get("observation_reason")
        no_face = frame is not None and reason == "NO_FACE"
        if no_face:
            if self._no_face_last is None or now - self._no_face_last > 0.5:
                self._no_face_since = now
            self._no_face_last = now
        else:
            self._no_face_since = self._no_face_last = None
        if self.violation_outbox is None or self.session_id is None:
            return
        # Missing observations pause episode recovery. A latched DROWSY/away
        # flag on a failed sensor frame must not trigger a new behavior event.
        body_ok = frame is not None and not reason and state.get("observation_ok", False)
        active = set()
        if body_ok:
            if (state.get("drowsy") and
                    (state.get("eye_reliable") or
                     (state.get("alert_key") == "DROWSY:YAWN" and state.get("yawning")))):
                active.add("DROWSINESS")
            if state.get("distracted"):
                code = {"YAW": "DISTRACTED_LOOK_AWAY", "LOOK_DOWN": "DISTRACTED_LOOK_DOWN"}.get(state.get("away_reason"))
                if code:
                    active.add(code)
            if state.get("yawning"):
                active.add("YAWNING")
            if state.get("head_nod"):
                active.add("HEAD_NOD")
        if no_face and now - self._no_face_since >= 2.0:
            active.add("FACE_NOT_VISIBLE")

        codes = ("DROWSINESS", "DISTRACTED_LOOK_AWAY", "DISTRACTED_LOOK_DOWN",
                 "YAWNING", "HEAD_NOD", "FACE_NOT_VISIBLE")
        for code in codes:
            event = self._violation_events.setdefault(code, {
                "sent": False, "retry_at": 0.0, "cooldown_until": 0.0,
                "clear_since": None, "clear_last": None,
            })
            if code not in active:
                # DROWSINESS requires the engine's full recovery, other codes
                # require 0.4 s of valid observations without that behavior.
                clear = body_ok and (code != "DROWSINESS" or state.get("state") == "AWAKE")
                if clear:
                    if event["clear_last"] is None or now - event["clear_last"] > 0.5:
                        event["clear_since"] = now
                    event["clear_last"] = now
                    if code == "DROWSINESS" or now - event["clear_since"] >= 0.4:
                        event["sent"] = False
                else:
                    event["clear_since"] = event["clear_last"] = None
                continue
            event["clear_since"] = event["clear_last"] = None
            if event["sent"] or now < max(event["retry_at"], event["cooldown_until"]):
                continue
            # Landmark geometry has no calibrated probability. All codes use 0.0
            # as the documented 'not estimated' convention, not P(no violation).
            confidence = 0.0
            try:
                event_id = self.violation_outbox.enqueue(
                    frame, self.session_id, confidence, violation_code=code)
            except Exception as exc:
                event["retry_at"] = now + 5
                print("[VIOLATION] cannot persist evidence:", code, type(exc).__name__)
                continue
            event["sent"] = True
            event["cooldown_until"] = now + 30.0
            print("[VIOLATION] queued", code, event_id)

    def _show_state(self, display, state, face=None, landmarks=None, evidence_frame=None):
        overlay_started = time.perf_counter() if self.profile_enabled else None
        self._queue_violation(evidence_frame, state)
        self._handle_alert(state)
        self._log_eye(state)
        color = (RED if state["alert_level"] >= 2 else
                 GREEN if state["state"] in ("AWAKE", "MONITORING", "NO_FACE") else ORANGE)
        if face is not None:
            _draw_box(display, face, color, state["state"])
        # No eye rectangles, ever. Landmarks are optional debug-only dots.
        if landmarks is not None and os.getenv("SHOW_LANDMARKS", "0") == "1":
            _draw_landmarks(display, landmarks, drowsy=state["drowsy"])
        info = self.driver_info or {}
        _draw_text(display, f"DRIVER: {info.get('driver_name') or info.get('driver_id') or '-'}", 35, WHITE, 0.65)
        def probability(value):
            return "?" if value is None else f"{value:.3f}"
        lines = [
            f"STATE: {state['state']}",
            f"Pose: yaw={probability(state.get('yaw'))} pitch={probability(state.get('pitch'))}",
            f"Mouth: {state['mouth_ratio']:.2f} | yawns: {state['yawn_count']} | away: {state['away_seconds']:.1f}s",
            "Q: Quit | R: recalibrate eyes, pose and mouth",
        ]
        cal = state.get('ear_calibration', {})
        if cal.get('calibrating'):
            lines.insert(1, f"CALIBRATE EAR: LOOK STRAIGHT, EYES OPEN  {cal['progress'] * 100:.0f}%")
        if os.getenv("SHOW_EYE_DEBUG", "0") == "1":
            obs_l = state['eye_observation']['left']
            obs_r = state['eye_observation']['right']
            lines.append(f"EAR raw L/R: {probability(obs_l.get('ear'))} / {probability(obs_r.get('ear'))} | {state['eye_status']}")
            lines.append(f"EAR median L/R: {probability(obs_l.get('filtered_ear'))} / {probability(obs_r.get('filtered_ear'))}")
            lines.append(f"CLOSED: {state['closed_seconds']:.2f}s | PERCLOS: {state['perclos'] * 100:.0f}%")
            if cal.get('calibrated'):
                baseline = cal.get('ear', {})
                sigma = cal.get('variability', {})
                lines.append(f"BASE EAR: {probability(baseline.get('left'))} / {probability(baseline.get('right'))} | SIGMA: {probability(sigma.get('left'))} / {probability(sigma.get('right'))}")
                lines.append(f"FACE SCALE: {probability(obs_l.get('face_scale'))} | eye width: {obs_l.get('width_px',0):.1f}px")
                lines.append(f"THRESHOLD CLOSE/OPEN L: {probability(obs_l.get('closed_threshold'))} / {probability(obs_l.get('open_threshold'))}")
                lines.append(f"EYE STATUS L/R: {obs_l.get('status')} / {obs_r.get('status')}")
        for index, line in enumerate(lines):
            _draw_text(display, line, 75 + index * 30,
                       color if index == 0 else WHITE,
                       0.65 if index == 0 else 0.48)
        _draw_alert_banner(display, state["alert_text"], state["alert_level"])
        if self.profile_enabled:
            self.last_profile["overlay_ms"] = (time.perf_counter() - overlay_started) * 1000.0

    def camera_failed_display(self, width, height):
        self.eye_landmarks.note_unobserved()
        display = np.zeros((height, width, 3), dtype=np.uint8)
        state = self.drowsiness.camera_failed()
        self._show_state(display, state)
        _draw_text(display, "CAMERA ERROR: no frame", 420, RED, 0.7)
        return display

    def reset_eye_calibration(self):
        self._ear_ready_logged = False
        self.eye_landmarks.start_calibration()
        self.drowsiness.eyes.reset()  # don't carry old closure duration into new calibration
        self.drowsiness.reset_calibration()
        print('[EAR] Recalibrating open-eye EAR, face scale, head pose and mouth.')

    def start_closed_calibration(self):
        print("[DROWSY] Use session open-eye calibration instead; no closed-eye step.")
        return False
