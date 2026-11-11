import os
import base64
import json
import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

import cv2
import requests

from app.config import (
    CONTEXT_PATH,
    DEVICE_API_KEY,
    DEVICE_CODE,
    JPEG_QUALITY,
    MAX_JPEG_BYTES,
    MAX_UPLOAD_HEIGHT,
    MAX_UPLOAD_WIDTH,
    SESSION_END_PATH,
    SESSION_START_PATH,
    WEB_SERVER_URL,
)


class FaceClient:
    def __init__(
        self,
        base_url=None,
        device_code=None,
        api_key=None,
    ):
        self.base_url = (
            base_url
            or WEB_SERVER_URL
            or os.environ["WEB_SERVER_URL"]
        ).rstrip("/")

        if not (
            self.base_url.startswith("http://")
            or self.base_url.startswith("https://")
        ):
            raise ValueError(
                "WEB_SERVER_URL phải bắt đầu bằng http:// hoặc https://."
            )

        token = (
            api_key
            or DEVICE_API_KEY
            or os.environ.get("DEVICE_API_KEY")
        )

        self.session = requests.Session()

        # Local test: token là optional.
        # Nếu backend có auth Bearer thì chỉ cần điền DEVICE_API_KEY trong .env.
        if token:
            self.session.headers.update({
                "Authorization": f"Bearer {token}",
            })

        code = (
            device_code
            or DEVICE_CODE
            or os.environ.get("DEVICE_CODE")
        )

        if code:
            self.session.headers[
                "X-Device-Code"
            ] = code

    @staticmethod
    def _parse_json(response):
        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError(
                "Server không trả JSON hợp lệ. "
                f"HTTP {response.status_code}: "
                f"{response.text[:500]}"
            ) from exc

    def context(self):
        response = self.session.get(
            self.base_url + CONTEXT_PATH,
            timeout=(5, 10),
            allow_redirects=False,
        )
        return (
            response.status_code,
            self._parse_json(response),
        )

    @staticmethod
    def _resize_for_upload(frame_bgr):
        height, width = frame_bgr.shape[:2]

        scale = min(
            1.0,
            MAX_UPLOAD_WIDTH / width,
            MAX_UPLOAD_HEIGHT / height,
        )

        if scale >= 1.0:
            return frame_bgr

        return cv2.resize(
            frame_bgr,
            (
                max(1, int(width * scale)),
                max(1, int(height * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def _encode_jpeg(frame_bgr):
        ok, encoded = cv2.imencode(
            ".jpg",
            frame_bgr,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                JPEG_QUALITY,
            ],
        )

        if not ok:
            raise ValueError(
                "JPEG encoding failed."
            )

        jpeg_bytes = encoded.tobytes()

        if len(jpeg_bytes) > MAX_JPEG_BYTES:
            raise ValueError(
                f"JPEG exceeds "
                f"{MAX_JPEG_BYTES / 1024 / 1024:.1f} MiB."
            )

        return jpeg_bytes

    def start_session(
        self,
        frame_bgr,
        captured_at,
    ):
        """
        POST /api/v1/device/sessions/start/
        Trả (http_status, json). `code` nằm ngoài cùng, `reason_code` nằm trong `data`.
        """
        frame_bgr = self._resize_for_upload(
            frame_bgr
        )

        jpeg_bytes = self._encode_jpeg(
            frame_bgr
        )

        response = self.session.post(
            self.base_url + SESSION_START_PATH,
            headers={
                "X-Request-ID": str(uuid4()),
            },
            data={
                "captured_at": captured_at,
            },
            files={
                "frame": (
                    "frame.jpg",
                    jpeg_bytes,
                    "image/jpeg",
                ),
            },
            timeout=(5, 45),
            allow_redirects=False,
        )

        return (
            response.status_code,
            self._parse_json(response),
        )

    def end_session(self, session_id):
        """
        POST /api/v1/device/sessions/<session_id>/end/  (Authorization: Bearer ...)
        Timeout ngắn vì gọi lúc đang tắt ứng dụng. Trả (http_status, json).
        """
        response = self.session.post(
            self.base_url
            + SESSION_END_PATH.format(
                session_id=quote(str(session_id), safe="")
            ),
            headers={
                "X-Request-ID": str(uuid4()),
            },
            timeout=(3, 8),
            allow_redirects=False,
        )

        return (
            response.status_code,
            self._parse_json(response),
        )

    def post_violation(self, metadata, jpeg_bytes):
        """Send one immutable event. Return (HTTP status, JSON); never regenerate its ID."""
        required = {"event_id", "session_id", "violation_code", "detected_at", "confidence"}
        if set(metadata) != required:
            raise ValueError("Violation requires exactly five metadata fields")
        from uuid import UUID
        UUID(str(metadata["event_id"]))
        sid = metadata["session_id"]
        if isinstance(sid, bool) or not str(sid).isdigit() or int(sid) <= 0:
            raise ValueError("session_id must be a positive integer")
        confidence = float(metadata["confidence"])
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be finite and between 0 and 1")
        stamp = datetime.fromisoformat(str(metadata["detected_at"]).replace("Z", "+00:00"))
        if stamp.utcoffset() is None:
            raise ValueError("detected_at must include a timezone")
        code = metadata["violation_code"]
        if not isinstance(code, str) or not code or code != code.strip():
            raise ValueError("violation_code must match the server code exactly")
        if not self.session.headers.get("Authorization", "").startswith("Bearer "):
            raise ValueError("DEVICE_API_KEY is required for violations")
        import numpy as np
        decoded = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if (not jpeg_bytes.startswith(b"\xff\xd8") or len(jpeg_bytes) > 2 * 1024 * 1024
                or decoded is None or decoded.shape[1] > 1920 or decoded.shape[0] > 1080):
            raise ValueError("Evidence must be JPEG <= 2 MiB and <= 1920x1080")
        request = requests.Request(
            "POST", self.base_url + "/api/v1/device/violations/",
            data=metadata, files={"image": ("evidence.jpg", jpeg_bytes, "image/jpeg")},
            headers={"X-Request-ID": str(uuid4())},
        )
        prepared = self.session.prepare_request(request)
        if int(prepared.headers["Content-Length"]) > 2 * 1024 * 1024 + 16 * 1024:
            raise ValueError("Multipart request exceeds server size limit")
        settings = self.session.merge_environment_settings(prepared.url, {}, None, None, None)
        response = self.session.send(prepared, timeout=(5, 45), allow_redirects=False, **settings)
        # Even HTML error responses must reach the retry policy.
        try:
            result = response.json()
        except ValueError:
            result = {"code": "NON_JSON_RESPONSE"}
        if not isinstance(result, dict):
            result = {"code": "INVALID_JSON_RESPONSE"}
        try:
            retry_after = max(5.0, float(response.headers.get("Retry-After", 5)))
            if not math.isfinite(retry_after):
                retry_after = 5.0
        except (ValueError, TypeError):
            retry_after = 5.0
        result = dict(result, _retry_after_seconds=retry_after)
        return response.status_code, result

    # Tên cũ, giữ để code khác chưa sửa vẫn chạy.
    verify = start_session

    def close(self):
        self.session.close()

class ViolationOutbox:
    """Single-process durable queue. Only acknowledged events are deleted.

    Run one monitor per outbox directory. 4xx/redirect failures are retained
    as blocked entries for manual correction, not retried on every frame.
    """

    def __init__(self, client, directory=None, start_worker=True):
        from app.config import PROJECT_ROOT
        self.client = client
        self.directory = Path(directory or os.getenv(
            "VIOLATION_OUTBOX_DIR", str(PROJECT_ROOT / "violation_outbox")))
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._stop = threading.Event()
        self._thread = None
        if start_worker:
            self._thread = threading.Thread(target=self._run, name="violation-upload", daemon=True)
            self._thread.start()

    def _save(self, path, entry):
        temporary = path.with_suffix(".tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            os.chmod(temporary, 0o600)
            json.dump(entry, handle, ensure_ascii=False, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        # Persist the rename on the Pi's filesystem before reporting queued.
        if hasattr(os, "O_DIRECTORY"):
            fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

    def enqueue(self, frame, session_id, confidence, violation_code="DROWSINESS", detected_at=None):
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Invalid confidence")
        height, width = frame.shape[:2]
        scale = min(1.0, 1920 / width, 1080 / height,
                    MAX_UPLOAD_WIDTH / width, MAX_UPLOAD_HEIGHT / height)
        if scale < 1:
            frame = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))))
        limit = min(MAX_JPEG_BYTES, 2 * 1024 * 1024)
        for quality in range(max(30, min(95, JPEG_QUALITY)), 19, -5):
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok and len(encoded) <= limit:
                break
        else:
            raise ValueError("Cannot encode evidence within JPEG limit")
        event_id = str(uuid4())
        entry = {
            "metadata": {"event_id": event_id, "session_id": session_id,
                         "violation_code": violation_code,
                         "detected_at": detected_at or datetime.now(timezone.utc).isoformat(),
                         "confidence": confidence},
            "jpeg": base64.b64encode(encoded.tobytes()).decode("ascii"),
            "base_url": self.client.base_url,
            "next_attempt": 0, "blocked": False,
        }
        self._save(self.directory / (event_id + ".json"), entry)
        return event_id

    def flush_once(self):
        for path in sorted(self.directory.glob("*.json")):
            if self._stop.is_set():
                break
            try:
                entry = json.loads(path.read_text(encoding="utf-8"))
                if (entry["blocked"] or entry["next_attempt"] > time.time()
                        or entry["base_url"] != self.client.base_url):
                    continue
                try:
                    status, result = self.client.post_violation(
                        entry["metadata"], base64.b64decode(entry["jpeg"], validate=True))
                except requests.RequestException:
                    status, result = 503, {"code": "NETWORK_ERROR"}
                except ValueError:
                    status, result = 400, {"code": "LOCAL_VALIDATION_ERROR"}
                success = ((status == 201 and result.get("code") == "VIOLATION_RECORDED")
                           or (status == 200 and result.get("code") == "EVENT_ALREADY_RECORDED"))
                if success:
                    path.unlink()
                    print("[VIOLATION]", entry["metadata"]["event_id"], result.get("code"))
                    continue
                entry["blocked"] = not (status >= 500 or status in (408, 429) or 200 <= status < 300)
                entry["next_attempt"] = time.time() + max(5, result.get("_retry_after_seconds", 5))
                entry["last_error"] = {"http_status": status, "code": result.get("code")}
                self._save(path, entry)
                print("[VIOLATION] retained", path.name, entry["last_error"], "blocked=", entry["blocked"])
            except Exception as exc:
                # A corrupt file/disk error must not kill the worker or delete evidence.
                print("[VIOLATION] queue error", path.name, type(exc).__name__)

    def _run(self):
        while not self._stop.is_set():
            self.flush_once()
            self._stop.wait(1)

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()  # bounded by current HTTP request; queued files remain
        self.client.close()
