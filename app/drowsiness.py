"""Landmark eye observations + temporal rules + landmark mouth/head signals.

DROWSY is an observable-risk alert, not a direct physiological sleep diagnosis.
Unobservable eyes are skipped; never fabricate an eye-related alert.
"""
import math
import os
import time
from collections import deque

import numpy as np

from app import config as _config  # Load the project .env before reading thresholds.
from app.eye_temporal import EyeTemporal
from app.inference.eye_landmarks import unknown_eyes


class ContinuousTimer:
    """Accumulate only between adjacent observations that both satisfy a condition."""
    def __init__(self, max_gap):
        self.max_gap = max_gap
        self.reset()

    def reset(self):
        self.last = None
        self.active = False
        self.seconds = 0.0

    def update(self, active, now):
        dt = 0.0 if self.last is None else now - self.last
        if active and self.active and 0 < dt <= self.max_gap:
            self.seconds += dt
        else:
            self.seconds = 0.0
        self.active, self.last = bool(active), now
        return self.seconds


def mouth_ratio_8(points):
    points = np.asarray(points, dtype=np.float32)
    width = np.linalg.norm(points[0] - points[4])
    if width < 1e-6:
        return None
    vertical = sum(np.linalg.norm(points[a] - points[b]) for a, b in ((1, 7), (2, 6), (3, 5))) / 3.0
    ratio = float(vertical / width)
    return ratio if math.isfinite(ratio) and 0 <= ratio <= 2 else None


class DrowsinessEngine:
    def __init__(self, clock=None, eye_config=None):
        self.clock = clock or time.monotonic
        self.eyes = EyeTemporal(eye_config)
        settings = {
            'calibration_seconds': ('POSE_CALIBRATION_SECONDS', 2.0),
            'calibration_samples': ('CALIBRATION_MIN_SAMPLES', 8),
            'yawn_ratio': ('YAWN_BASELINE_RATIO', 2.0),
            'yawn_increase': ('YAWN_MIN_INCREASE', 0.12),
            'yawn_floor': ('YAWN_ABSOLUTE_MIN_RATIO', 0.35),
            'yawn_seconds': ('YAWN_MIN_SECONDS', 0.70),
            'yawn_count_threshold': ('YAWN_COUNT_THRESHOLD', 2),
            'yawn_window': ('YAWN_WINDOW_SECONDS', 60.0),
            'nod_degrees': ('HEAD_PITCH_DOWN_DEG', 20.0),
            'nod_seconds': ('HEAD_NOD_SECONDS', 0.45),
            'drop_threshold': ('HEAD_DROP_THRESHOLD', 0.15),
            'eye_nod_seconds': ('EYE_NOD_COMBO_SECONDS', 0.70),
            'yaw_away': ('YAW_AWAY_DEG', 35.0),
            'pitch_away': ('PITCH_AWAY_DOWN_DEG', 30.0),
            'away_seconds': ('LOOK_AWAY_SECONDS', 2.0),
            'away_critical_seconds': ('LOOK_AWAY_CRITICAL_SECONDS', 5.0),
            'away_recover_seconds': ('AWAY_RECOVER_SECONDS', 0.40),
        }
        for attr, (name, default) in settings.items():
            value = float(os.getenv(name, str(default)))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
            setattr(self, attr, value)
        self.pitch_down_sign = float(os.getenv('PITCH_DOWN_SIGN', '-1'))
        if self.pitch_down_sign not in (-1, 1):
            raise ValueError('PITCH_DOWN_SIGN phải là -1 hoặc 1')
        if not (self.away_critical_seconds >= self.away_seconds and
                self.eyes.config.blink_seconds <= self.eye_nod_seconds <= self.eyes.config.drowsy_seconds):
            raise ValueError('Invalid combined-eye or look-away duration thresholds')
        self.reset()

    def reset(self):
        """Start a new monitoring session."""
        self.eyes.reset()
        self._drowsy_active = False
        self._drowsy_key = self._drowsy_text = None
        self._expects_pose = False
        gap = self.eyes.config.max_gap_seconds
        self._recovery = ContinuousTimer(gap)
        self.reset_calibration()

    def reset_calibration(self):
        """R resets mouth/head reference only; it cannot silence a latched alert."""
        self.pose0 = None
        self.mouth_baseline = None
        self.yawn_threshold = self.yawn_floor
        self.center0 = self.face_height0 = None
        self.calibrated = False
        self._calibration_data = []
        gap = self.eyes.config.max_gap_seconds
        self._calibration = ContinuousTimer(gap)
        self._late_pose_timer = ContinuousTimer(gap)
        self._late_pose_samples = []
        self._mouth = ContinuousTimer(gap)
        self._head = ContinuousTimer(gap)
        self._away = ContinuousTimer(gap)
        self._away_clear = ContinuousTimer(gap)
        self._away_latched = False
        self._away_critical = False
        self._away_reason = None
        self._last_away_seconds = 0.0
        self._yawn_counted = False
        self.yawn_history = deque()
        if hasattr(self, '_recovery'):
            self._recovery.reset()

    @staticmethod
    def _pose(pose):
        if pose is None:
            return None
        try:
            vec = np.array([pose[k] for k in ('yaw', 'pitch', 'roll')], dtype=np.float64)
            return vec if np.isfinite(vec).all() and (np.abs(vec) <= 180).all() else None
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _face_position(bbox, frame_shape):
        if bbox is None or frame_shape is None:
            return None, None
        b = np.asarray(bbox[:4], dtype=float)
        if b.shape != (4,) or not np.isfinite(b).all() or b[3] <= b[1] or b[2] <= b[0] or frame_shape[0] <= 0:
            return None, None
        return float((b[1] + b[3]) / (2 * frame_shape[0])), float((b[3] - b[1]) / frame_shape[0])

    def _update_away(self, active, reason, now, observed=True):
        seconds = self._away.update(active and observed, now)
        clear_seconds = self._away_clear.update(observed and not active, now)
        if active and observed:
            self._away_reason = reason
            self._last_away_seconds = seconds
            if seconds + 1e-9 >= self.away_seconds:
                self._away_latched = True
            if seconds + 1e-9 >= self.away_critical_seconds:
                self._away_critical = True
        elif clear_seconds + 1e-9 >= self.away_recover_seconds:
            self._away_latched = self._away_critical = False
            self._away_reason = None
            self._last_away_seconds = 0.0
        return {
            'away_now': bool(active), 'away_seconds': self._last_away_seconds if self._away_latched else seconds,
            'away_reason': self._away_reason if active or self._away_latched else None,
            'distracted': self._away_latched, 'away_critical': self._away_critical,
        }

    def _body(self, lm, bbox, frame_shape, pose, eye, now, observation=None):
        vec = self._pose(pose)
        if vec is not None:
            self._expects_pose = True
        pose_available = vec is not None
        pose_ok = pose_available or not self._expects_pose
        center, height = self._face_position(bbox, frame_shape)
        mouth = mouth_ratio_8(lm[88:96])
        # During session start, collect neutral pose/mouth alongside OPEN-eye
        # geometry, rather than waiting another two seconds after EAR calibration.
        frontal = pose_ok and (vec is None or np.max(np.abs(vec)) <= 15)
        eye_sample_ok = bool(observation and all(
            observation.get(side, {}).get('calibration_sample_ok', False)
            for side in ('left', 'right')))
        calibration_ok = (eye['eye_status'] == 'OPEN' or eye_sample_ok) and frontal and mouth is not None and mouth <= 0.35
        if not self.calibrated:
            elapsed = self._calibration.update(calibration_ok, now)
            if not calibration_ok or elapsed == 0:
                self._calibration_data = []
            if calibration_ok:
                self._calibration_data.append((vec, mouth, center, height))
            if elapsed + 1e-9 >= self.calibration_seconds and len(self._calibration_data) >= self.calibration_samples:
                samples = self._calibration_data
                poses = [s[0] for s in samples if s[0] is not None]
                self.pose0 = np.median(poses, axis=0) if poses else None
                self.mouth_baseline = float(np.median([s[1] for s in samples]))
                self.yawn_threshold = max(self.yawn_floor, self.mouth_baseline * self.yawn_ratio,
                                          self.mouth_baseline + self.yawn_increase)
                valid_face = [s for s in samples if s[2] is not None and s[3] is not None]
                if valid_face:
                    self.center0 = float(np.median([s[2] for s in valid_face]))
                    self.face_height0 = float(np.median([s[3] for s in valid_face]))
                self.calibrated = True
                self._calibration_data = []
        # Late pose availability: establish it only from a new frontal/open interval.
        if self.calibrated and self.pose0 is None:
            good_pose = bool(calibration_ok and vec is not None)
            elapsed = self._late_pose_timer.update(good_pose, now)
            if not good_pose or elapsed == 0:
                self._late_pose_samples = []
            if good_pose:
                self._late_pose_samples.append(vec)
            if elapsed + 1e-9 >= self.calibration_seconds and len(self._late_pose_samples) >= self.calibration_samples:
                self.pose0 = np.median(self._late_pose_samples, axis=0)
                self._late_pose_samples = []
        pose_valid = vec is not None and self.pose0 is not None
        rel = vec - self.pose0 if pose_valid else None
        pitch_down = self.pitch_down_sign * rel[1] if rel is not None else None
        # Absolute + relative pose gate for mouth visibility.
        mouth_ok = self.calibrated and pose_ok and (vec is None or (abs(vec[0]) <= 40 and abs(vec[1]) <= 35))
        mouth_open = bool(mouth_ok and mouth is not None and mouth > self.yawn_threshold)
        yawn_duration = self._mouth.update(mouth_open, now)
        if not mouth_open:
            self._yawn_counted = False
        new_yawn = mouth_open and yawn_duration + 1e-9 >= self.yawn_seconds and not self._yawn_counted
        if new_yawn:
            self.yawn_history.append(now)
            self._yawn_counted = True
        while self.yawn_history and now - self.yawn_history[0] > self.yawn_window:
            self.yawn_history.popleft()
        yawning = mouth_open and yawn_duration + 1e-9 >= self.yawn_seconds
        head_drop = 0.0
        if center is not None and self.center0 is not None and height > 0:
            head_drop = (center - self.center0) / height
        if pose_valid:
            down = pitch_down > self.nod_degrees
        else:
            scale_ok = (not self._expects_pose and height is not None and self.face_height0 is not None
                        and abs(height / self.face_height0 - 1) <= 0.25)
            down = bool(self.calibrated and scale_ok and head_drop > self.drop_threshold)
        head_seconds = self._head.update(down, now)
        head_nod = down and head_seconds + 1e-9 >= self.nod_seconds
        # Absolute pose also works before calibration; relative pose handles a natural baseline.
        yaw = float(rel[0]) if pose_valid else (float(vec[0]) if pose_available else None)
        down_angle = pitch_down if pose_valid else (self.pitch_down_sign * float(vec[1]) if pose_available else None)
        away_now = False
        reason = None
        if yaw is not None and abs(yaw) > self.yaw_away:
            away_now, reason = True, 'YAW'
        elif down_angle is not None and down_angle > self.pitch_away:
            away_now, reason = True, 'LOOK_DOWN'
        away = self._update_away(away_now, reason, now, observed=pose_ok)
        return dict(away, mouth_ratio=mouth or 0.0, yawn_threshold=self.yawn_threshold,
                    yawning=yawning, yawn_seconds=yawn_duration, yawn_count=len(self.yawn_history),
                    repeated_yawn_event=bool(new_yawn and len(self.yawn_history) >= self.yawn_count_threshold),
                    head_nod=head_nod, head_down_seconds=head_seconds, head_drop=head_drop,
                    pose_valid=pose_valid, pose_available=pose_available,
                    yaw=float(vec[0]) if pose_available else None,
                    pitch=float(vec[1]) if pose_available else None,
                    roll=float(vec[2]) if pose_available else None,
                    observation_ok=pose_ok, head_down=down, mouth_open=mouth_open)

    def _finish(self, eye, observation, body, now, sensor_reason=None):
        combo = eye['eyes_closed'] and body['head_nod'] and eye['closed_seconds'] + 1e-9 >= self.eye_nod_seconds
        if eye['long_eye_closure'] or combo:
            self._drowsy_active = True
            self._drowsy_key = 'DROWSY:EYES_NOD' if combo else 'DROWSY:EYES'
            self._drowsy_text = 'CANH BAO BUON NGU: NHAM MAT + GAT DAU' if combo else 'CANH BAO BUON NGU: NHAM MAT LAU'
        elif eye['perclos_event']:
            self._drowsy_active = True
            self._drowsy_key, self._drowsy_text = 'DROWSY:PERCLOS', 'CANH BAO BUON NGU: NHAM MAT LAP LAI'
        elif body['repeated_yawn_event']:
            self._drowsy_active = True
            self._drowsy_key, self._drowsy_text = 'DROWSY:YAWN', 'CANH BAO BUON NGU: NGAP NHIEU LAN'
        # A broken eye observation interrupts the measured eye-closure
        # episode. Never resurrect the old eye alarm when tracking resumes.
        if not eye['eye_reliable'] and self._drowsy_active and self._drowsy_key != 'DROWSY:YAWN':
            self._drowsy_active = False
            self._drowsy_key = self._drowsy_text = None
        # Only reliable OPEN eyes + no currently open mouth can clear DROWSY.
        # Head turning/nodding is tracked independently; a small movement must
        # not restart the eye recovery timer. UNKNOWN eyes never recover.
        recovery_ok = eye['eye_status'] == 'OPEN' and not body['mouth_open']
        recovery_seconds = self._recovery.update(recovery_ok, now)
        recovered = recovery_seconds + 1e-9 >= self.eyes.config.recover_seconds
        if self._drowsy_active and recovered:
            self._drowsy_active = False
            self._drowsy_key = self._drowsy_text = None
        # Invalid eye frames must NOT be treated as evidence of sleep.
        # Keep latched state internally until valid recovery, but never show or
        # upload a stale eye alert while the eyes cannot be measured.
        eyes_visible = bool(eye['eye_reliable'])
        yawn_alert_visible = bool(self._drowsy_key == 'DROWSY:YAWN' and
                                  body['observation_ok'] and body['yawning'])
        visible_drowsy = bool(self._drowsy_active and (eyes_visible or yawn_alert_visible))
        visible_away = bool(sensor_reason is None and body['observation_ok'])
        if visible_drowsy:
            state = 'DROWSY'
        elif visible_away and (body['distracted'] or body['away_now']):
            state = 'LOOK_AWAY'
        elif sensor_reason == 'NO_FACE':
            state = 'NO_FACE'
        elif body['yawning']:
            state = 'YAWNING'  # independent of eye visibility
        elif body['head_nod']:
            state = 'HEAD_NOD'  # independent of eye visibility
        elif not eyes_visible:
            state = 'MONITORING'  # neutral, no eye uncertainty banner or event
        elif body['mouth_open']:
            state = 'MOUTH_OPEN'
        elif eye['eyes_closed']:
            state = ('EYES_CLOSED' if eye['closed_seconds'] + 1e-9 >=
                     self.eyes.config.blink_seconds else 'BLINK_OR_CLOSING')
        else:
            state = 'AWAKE'
        level, key, text = 0, None, None
        if visible_drowsy:
            level, key, text = 2, self._drowsy_key, self._drowsy_text
        elif visible_away and body['distracted']:
            level = 2 if body['away_critical'] else 1
            key = f"LOOK_AWAY:{body['away_reason']}:{level}"
            text = f"CANH BAO MAT TAP TRUNG: {body['away_reason']}"
        out = dict(eye, **body)
        out.update(state=state, drowsy=visible_drowsy, alert_level=level,
                   alert_key=key, alert_text=text, recovery_seconds=recovery_seconds,
                   calibration_progress=1.0 if self.calibrated else min(1.0, self._calibration.seconds / self.calibration_seconds),
                   calibrated=self.calibrated, observation_reason=sensor_reason,
                   p_closed_left=observation['left'].get('p_closed'),
                   p_closed_right=observation['right'].get('p_closed'),
                   eye_width_px=min(observation[s].get('width_px', 0.0) for s in ('left', 'right')),
                   eye_observation=observation)
        self.last_state = state
        return out

    def update(self, landmarks_norm, face_bbox=None, frame_shape=None, pose=None,
               landmark_frame=None, eye_observation=None, now=None):
        now = self.clock() if now is None else float(now)
        lm = np.asarray(landmarks_norm, dtype=np.float32)
        if lm.size != 196 or not np.isfinite(lm).all():
            return self.observation_failed('LANDMARK_ERROR', now)
        lm = lm.reshape(98, 2)
        observation = eye_observation if eye_observation is not None else unknown_eyes('MISSING_LANDMARK_EYES')
        if (self._expects_pose or pose is not None) and self._pose(pose) is None:
            observation = unknown_eyes('MISSING_POSE' if pose is None else 'BAD_POSE')
        eye = self.eyes.update(observation, now)
        body = self._body(lm, face_bbox, frame_shape, pose, eye, now, observation)
        return self._finish(eye, observation, body, now)

    def observation_failed(self, reason, now=None):
        now = self.clock() if now is None else float(now)
        observation = unknown_eyes(reason)
        eye = self.eyes.update(observation, now)
        self._mouth.update(False, now)
        self._head.update(False, now)
        self._late_pose_timer.update(False, now)
        self._late_pose_samples = []
        self._yawn_counted = False
        if not self.calibrated:
            self._calibration.update(False, now)
            self._calibration_data = []
        # No face is not a measured head turn. FACE_NOT_VISIBLE API stays separate.
        away = self._update_away(False, None, now, observed=False)
        body = dict(away, mouth_ratio=0.0, yawn_threshold=self.yawn_threshold,
                    yawning=False, yawn_seconds=0.0, yawn_count=len(self.yawn_history),
                    repeated_yawn_event=False, head_nod=False, head_down_seconds=0.0,
                    head_drop=0.0, pose_valid=False, pose_available=False,
                    yaw=None, pitch=None, roll=None, observation_ok=False,
                    head_down=False, mouth_open=False)
        return self._finish(eye, observation, body, now, sensor_reason=reason)

    def no_face(self):
        return self.observation_failed('NO_FACE')

    def landmark_failed(self):
        return self.observation_failed('LANDMARK_ERROR')

    def camera_failed(self):
        return self.observation_failed('CAMERA_ERROR')

    def start_closed_calibration(self):
        return False  # Open-eye session calibration is handled by LandmarkEyes.
