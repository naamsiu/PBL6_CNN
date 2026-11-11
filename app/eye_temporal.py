"""Confidence + observed time; never infer OPEN from missing data."""
from collections import deque
from dataclasses import dataclass
import math
import os


@dataclass
class EyeTimingConfig:
    blink_seconds: float = 0.30
    drowsy_seconds: float = 1.50
    recover_seconds: float = 1.00
    max_gap_seconds: float = 0.50
    perclos_window: float = 30.0
    perclos_min_seconds: float = 10.0
    perclos_min_coverage: float = 0.60
    perclos_threshold: float = 0.20
    perclos_rearm: float = 0.15

    @classmethod
    def from_env(cls):
        names = {
            "blink_seconds": "EYE_MIN_CLOSED_SECONDS", "drowsy_seconds": "DROWSY_CLOSED_SECONDS",
            "recover_seconds": "EYE_RECOVER_SECONDS", "max_gap_seconds": "MAX_OBSERVATION_GAP_SECONDS",
            "perclos_window": "PERCLOS_WINDOW_SECONDS", "perclos_min_seconds": "PERCLOS_MIN_SECONDS",
            "perclos_min_coverage": "PERCLOS_MIN_COVERAGE", "perclos_threshold": "PERCLOS_THRESHOLD",
            "perclos_rearm": "PERCLOS_REARM_THRESHOLD",
        }
        defaults = cls()
        return cls(**{k: float(os.getenv(v, str(getattr(defaults, k)))) for k, v in names.items()})

    def __post_init__(self):
        if not all(math.isfinite(x) for x in vars(self).values()):
            raise ValueError("Eye timing values must be finite")
        if not (0 < self.blink_seconds < self.drowsy_seconds and self.recover_seconds > 0 and self.max_gap_seconds > 0):
            raise ValueError("Eye duration thresholds must be positive and ordered")
        if not (0 < self.perclos_min_seconds <= self.perclos_window and
                0 < self.perclos_min_coverage <= 1 and
                0 <= self.perclos_rearm < self.perclos_threshold <= 1):
            raise ValueError("Invalid PERCLOS configuration")


class EyeTemporal:
    def __init__(self, config=None):
        self.config = config or EyeTimingConfig.from_env()
        self.reset()

    def reset(self):
        self.previous_time = None
        self.previous_status = "UNKNOWN"
        self.first_time = None
        self.closed_seconds = self.open_seconds = 0.0
        self.history = deque()  # mutable [start, end, counted_closed]
        self.pending_closed = []
        self.perclos_armed = True

    def _classify(self, eyes):
        """We need both reliable CLOSED eyes to infer bilateral closure.

        One valid OPEN eye proves *both* eyes are not closed, even if the
        other is blocked or winking. One valid CLOSED eye alone is insufficient
        evidence for a drowsiness alert. We never fill missing data as CLOSED.
        """
        statuses = []
        reasons = []
        for side in ("left", "right"):
            eye = (eyes or {}).get(side, {})
            if eye.get("valid") and eye.get("status") in ("OPEN", "CLOSED"):
                statuses.append(eye["status"])
            else:
                statuses.append("UNKNOWN")
                reasons.append(eye.get("reason") or "NO_EYE_MEASUREMENT")
        if statuses == ["CLOSED", "CLOSED"]:
            return "CLOSED", None
        if "OPEN" in statuses:
            return "OPEN", None if statuses == ["OPEN", "OPEN"] else "PARTIAL_EYE_OPEN"
        return "UNKNOWN", reasons[0] if reasons else "BILATERAL_CLOSURE_UNPROVEN"

    def update(self, eyes, now):
        now = float(now)
        if not math.isfinite(now) or (self.previous_time is not None and now < self.previous_time):
            raise ValueError("EyeTemporal requires a monotonic finite timestamp")
        if self.first_time is None:
            self.first_time = now
        status, reason = self._classify(eyes)
        raw_dt = 0.0 if self.previous_time is None else now - self.previous_time
        contiguous = self.previous_time is not None and 0 < raw_dt <= self.config.max_gap_seconds
        same = contiguous and status == self.previous_status and status != "UNKNOWN"
        if not same:
            self.closed_seconds = self.open_seconds = 0.0
            self.pending_closed = []
        if same:
            segment = [self.previous_time, now, False]
            self.history.append(segment)
            if status == "CLOSED":
                self.closed_seconds += raw_dt
                self.pending_closed.append(segment)
                if self.closed_seconds + 1e-9 >= self.config.blink_seconds:
                    for pending in self.pending_closed:
                        pending[2] = True
                    self.pending_closed = []
            else:
                self.open_seconds += raw_dt
        self.previous_status, self.previous_time = status, now
        cutoff = now - self.config.perclos_window
        while self.history and self.history[0][1] <= cutoff:
            self.history.popleft()
        valid_seconds = closed_time = 0.0
        for start, end, is_closed in self.history:
            duration = max(0.0, end - max(start, cutoff))
            valid_seconds += duration
            if is_closed:
                closed_time += duration
        span = min(self.config.perclos_window, now - self.first_time)
        coverage = min(1.0, valid_seconds / span) if span > 0 else 0.0
        perclos = closed_time / valid_seconds if valid_seconds > 0 else 0.0
        ready = valid_seconds + 1e-9 >= self.config.perclos_min_seconds and coverage >= self.config.perclos_min_coverage
        if ready and perclos <= self.config.perclos_rearm:
            self.perclos_armed = True
        event = bool(ready and status != "UNKNOWN" and perclos >= self.config.perclos_threshold and self.perclos_armed)
        if event:
            self.perclos_armed = False
        return {
            "eye_status": status, "eye_quality_reason": reason,
            "eye_reliable": status != "UNKNOWN", "eyes_closed": status == "CLOSED",
            "closed_seconds": self.closed_seconds, "open_seconds": self.open_seconds,
            "long_eye_closure": status == "CLOSED" and self.closed_seconds + 1e-9 >= self.config.drowsy_seconds,
            "perclos": perclos, "perclos_ready": ready, "perclos_coverage": coverage,
            "perclos_valid_seconds": valid_seconds, "perclos_event": event,
        }
