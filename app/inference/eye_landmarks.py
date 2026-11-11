"""Per-driver EAR baseline and reliable two-state eye tracking (V5).

A 3-second, frontal, eyes-open calibration learns per-eye EAR distribution,
eye width, relative face size and eyelid-independent landmark jitter.

OPEN/CLOSED are tracked using a short hysteresis band, a median filter, and
consecutive evidence. An EAR in the hysteresis band *keeps the old state*.
UNKNOWN is reserved for genuinely missing/untrustworthy eye measurements.
No eye CNN, no eye rectangles, no synthetic measurements for missing frames.
"""
from collections import deque
import math
import os
import time

import numpy as np

SIDES = ('left', 'right')


def unknown_eyes(reason):
    return {side: dict(valid=False, status='UNKNOWN', p_closed=None,
                       reason=reason, width_px=0.0, ear=None)
            for side in SIDES}


class LandmarkEyes:
    """WFLW landmark loops [60:68] and [68:76], eye corners 0 and 4."""

    def __init__(self):
        self.closed = float(os.getenv('EAR_CLOSED_THRESHOLD', '0.16'))
        self.open = float(os.getenv('EAR_OPEN_THRESHOLD', '0.22'))
        self.min_width = float(os.getenv('EAR_MIN_WIDTH_PX', '8'))
        self.max_yaw = float(os.getenv('EAR_MAX_YAW', '40'))
        self.max_pitch = float(os.getenv('EAR_MAX_PITCH', '35'))
        self.max_roll = float(os.getenv('EAR_MAX_ROLL', '35'))
        self.calibration_seconds = float(os.getenv('EAR_CALIBRATION_SECONDS', '3.0'))
        self.calibration_samples = int(os.getenv('EAR_CALIBRATION_MIN_SAMPLES', '15'))
        self.calibration_pose = float(os.getenv('EAR_CALIBRATION_MAX_POSE', '15'))
        self.calibration_gap = float(os.getenv('MAX_OBSERVATION_GAP_SECONDS', '0.5'))
        self.close_ratio = float(os.getenv('EAR_PERSONAL_CLOSED_RATIO', '0.68'))
        self.open_ratio = float(os.getenv('EAR_PERSONAL_OPEN_RATIO', '0.74'))
        self.width_ratio = float(os.getenv('EAR_EXPECTED_WIDTH_MIN_RATIO', '0.55'))
        self.stability_frames = int(os.getenv('EAR_STABILITY_FRAMES', '3'))
        self.confirm_frames = int(os.getenv('EAR_CONFIRM_FRAMES', '2'))
        self.ear_filter_frames = int(os.getenv('EAR_FILTER_FRAMES', '3'))
        self.shape_min_tolerance = float(os.getenv('EAR_SHAPE_MIN_TOLERANCE', '0.14'))
        if not (0 < self.closed < self.open < 1 and self.min_width >= 3 and
                1 <= self.calibration_seconds <= 15 and self.calibration_samples >= 5 and
                0 < self.calibration_gap <= 2 and 0 < self.calibration_pose <= 30 and
                0 < self.close_ratio < self.open_ratio < 1 and
                0 < self.width_ratio < 1 and 1 <= self.stability_frames <= 10 and
                1 <= self.confirm_frames <= 6 and 1 <= self.ear_filter_frames <= 7 and
                0.03 <= self.shape_min_tolerance <= 0.5):
            raise ValueError('Invalid V5 EAR configuration')
        self.reset()

    def reset(self):
        self.calibration_requested = False
        self.calibrated = False
        self.baseline_ear = {}
        self.baseline_variability = {}
        self.baseline_width_px = {}
        self.baseline_face_height = None
        self.baseline_shape = {}
        self.baseline_landmark_stability = {}
        self._samples = []
        self._first_sample_t = None
        self._last_sample_t = None
        self._width_history = {side: deque(maxlen=self.stability_frames) for side in SIDES}
        self._shape_history = {side: deque(maxlen=self.stability_frames) for side in SIDES}
        self._ear_history = {side: deque(maxlen=self.ear_filter_frames) for side in SIDES}
        self._width_last_t = {side: None for side in SIDES}
        self._state = {side: 'OPEN' for side in SIDES}
        self._last_valid_t = {side: None for side in SIDES}
        self._candidate = {side: None for side in SIDES}
        self._candidate_count = {side: 0 for side in SIDES}

    def start_calibration(self):
        self.reset()
        self.calibration_requested = True

    @property
    def calibration_progress(self):
        if self.calibrated:
            return 1.0
        if not self.calibration_requested or not self._samples:
            return 0.0
        duration = self._samples[-1]['time'] - self._samples[0]['time']
        return min(.99, max(0., duration / self.calibration_seconds))

    def calibration_info(self):
        return dict(calibrating=self.calibration_requested and not self.calibrated,
                    calibrated=self.calibrated, progress=self.calibration_progress,
                    samples=len(self._samples), ear=dict(self.baseline_ear),
                    variability=dict(self.baseline_variability),
                    width_px=dict(self.baseline_width_px),
                    landmark_stability=dict(self.baseline_landmark_stability),
                    face_height_px=self.baseline_face_height)

    def _clear_runtime(self, side):
        self._width_history[side].clear()
        self._shape_history[side].clear()
        self._ear_history[side].clear()
        self._width_last_t[side] = None
        self._last_valid_t[side] = None
        self._candidate[side] = None
        self._candidate_count[side] = 0
        # A fresh observation after a lost track cannot inherit a stale CLOSED.
        self._state[side] = 'OPEN'

    def note_unobserved(self):
        if self.calibration_requested and not self.calibrated:
            self._samples = []
            self._first_sample_t = self._last_sample_t = None
        for side in SIDES:
            self._clear_runtime(side)

    @staticmethod
    def _compute(eye):
        width = float(np.linalg.norm(eye[0] - eye[4]))
        if not math.isfinite(width) or width <= 0:
            return None, width
        gaps = [float(np.linalg.norm(eye[a] - eye[b])) for a, b in
                ((1, 7), (2, 6), (3, 5))]
        return float(np.median(gaps) / width), width

    @staticmethod
    def _signature(eye):
        """X positions in eye-local axes; ignores vertical eyelid motion/blinks.

        Endpoint rotation/translation and isotropic scale are removed. This
        measures landmark *tracking geometry*, not how far the lids are open.
        """
        axis = np.asarray(eye[4] - eye[0], dtype=float)
        width = float(np.linalg.norm(axis))
        if not math.isfinite(width) or width <= 1e-6:
            return None
        unit = axis / width
        return np.asarray((eye[[1, 2, 3, 5, 6, 7]] - eye[0]) @ unit / width,
                          dtype=np.float64)

    @staticmethod
    def _box_height(face_bbox):
        if face_bbox is None:
            return None
        try:
            b = np.asarray(face_bbox[:4], dtype=np.float64)
            if b.shape != (4,) or not np.isfinite(b).all() or b[2] <= b[0] or b[3] <= b[1]:
                return None
            return float(b[3] - b[1])
        except (TypeError, ValueError, IndexError):
            return None

    @staticmethod
    def _open_sample_stats(values):
        """Exclude bottom 40% (blinks), robust median/MAD of the remainder."""
        values = np.asarray(values, dtype=float)
        order = np.argsort(values)
        selected = order[int(.40 * len(order)):]
        top = values[selected]
        if len(top) < 3:
            return None
        baseline = float(np.median(top))
        mad = float(1.4826 * np.median(np.abs(top - baseline)))
        if baseline < .055 or baseline > .55 or float(np.std(top)) > .30 * baseline:
            return None
        return baseline, max(.002, mad), selected

    def _collect(self, raw, face_height, frontal, now):
        if (not frontal or face_height is None or face_height < 40 or
                any(raw[s]['reason'] is not None for s in SIDES)):
            self.note_unobserved()
            return False
        # Blinks may occur during calibration. Drop these frames without resetting.
        if any(not (.055 <= raw[s]['ear'] <= .55) for s in SIDES):
            return False
        if self._last_sample_t is not None and (now <= self._last_sample_t or
                                                now - self._last_sample_t > self.calibration_gap):
            self.note_unobserved()
        record = {'time': now, 'face_height': face_height,
                  'ear': {s: raw[s]['ear'] for s in SIDES},
                  'width': {s: raw[s]['width_px'] for s in SIDES},
                  'shape': {s: raw[s]['signature'].copy() for s in SIDES}}
        self._samples.append(record)
        self._last_sample_t = now
        if self._first_sample_t is None:
            self._first_sample_t = now
        if len(self._samples) > 500:
            self._samples = self._samples[-500:]
        if (now - self._first_sample_t + 1e-9 < self.calibration_seconds or
                len(self._samples) < self.calibration_samples):
            return True
        heights = np.asarray([x['face_height'] for x in self._samples], dtype=float)
        median_height = float(np.median(heights))
        if float(np.std(heights) / max(1., median_height)) > .18:
            self.note_unobserved()
            return False
        stats = {}
        for side in SIDES:
            ear = [x['ear'][side] for x in self._samples]
            stats[side] = self._open_sample_stats(ear)
            if stats[side] is None:
                self.note_unobserved()
                return False
        for side in SIDES:
            base, sigma, selected = stats[side]
            shapes = np.asarray([self._samples[int(i)]['shape'][side]
                                 for i in selected], dtype=float)
            reference = np.median(shapes, axis=0)
            residual = np.linalg.norm(shapes - reference, axis=1) / math.sqrt(shapes.shape[1])
            self.baseline_ear[side] = base
            self.baseline_variability[side] = sigma
            self.baseline_shape[side] = reference
            self.baseline_landmark_stability[side] = max(.005, float(np.median(residual)))
            self.baseline_width_px[side] = float(np.median(
                [self._samples[int(i)]['width'][side] for i in selected]))
        self.baseline_face_height = median_height
        self.calibrated = True
        self._samples = []
        self._first_sample_t = self._last_sample_t = None
        return True

    def _thresholds(self, side):
        if not self.calibrated:
            return self.closed, self.open
        baseline = self.baseline_ear[side]
        variability = self.baseline_variability[side]
        # Variability guards against interpreting normal open-eye jitter as CLOSED.
        low = min(baseline * self.close_ratio, baseline - 2.5 * variability)
        high = min(baseline * self.open_ratio, baseline - 1.5 * variability)
        high = max(high, low + max(.004, baseline * .035))
        return max(.002, low), min(baseline, high)

    def _observe_quality(self, side, width, signature, expected, scale, now, frontal):
        last = self._width_last_t[side]
        if last is not None and (now <= last or now - last > self.calibration_gap):
            self._clear_runtime(side)
        self._width_last_t[side] = now
        relative = width / max(expected, 1e-6)
        self._width_history[side].append(relative)
        self._shape_history[side].append(signature)
        if len(self._width_history[side]) < self.stability_frames:
            return False, 'WAIT_STABLE_EYE', None
        widths = np.asarray(self._width_history[side], dtype=float)
        width_jitter = float((np.max(widths) - np.min(widths)) / max(np.median(widths), 1e-6))
        if width_jitter > .35:
            return False, 'UNSTABLE_EYE_WIDTH', None
        shapes = np.asarray(self._shape_history[side], dtype=float)
        ref = self.baseline_shape[side]
        per_frame = np.linalg.norm(shapes - ref, axis=1) / math.sqrt(shapes.shape[1])
        deviation = float(np.median(per_frame))
        tolerance = max(self.shape_min_tolerance,
                        4 * self.baseline_landmark_stability[side] + .04)
        # Farther eyes are noisier due to pixel quantization. Do not hallucinate detail.
        tolerance += .018 / max(.35, scale)
        if frontal and deviation > tolerance:
            return False, 'UNSTABLE_LANDMARK_SHAPE', deviation
        return True, None, deviation

    def _update_state(self, side, ear):
        self._ear_history[side].append(ear)
        filtered = float(np.median(self._ear_history[side]))
        low, high = self._thresholds(side)
        previous = self._state[side]
        desired = ('CLOSED' if filtered <= low else
                   'OPEN' if filtered >= high else previous)
        if desired == previous:
            self._candidate[side] = None
            self._candidate_count[side] = 0
        else:
            if self._candidate[side] == desired:
                self._candidate_count[side] += 1
            else:
                self._candidate[side] = desired
                self._candidate_count[side] = 1
            if self._candidate_count[side] >= self.confirm_frames:
                self._state[side] = desired
                self._candidate[side] = None
                self._candidate_count[side] = 0
        return self._state[side], filtered, low, high

    def predict(self, points, pose=None, frame_shape=None, face_bbox=None, now=None):
        now = time.monotonic() if now is None else float(now)
        lm = np.asarray(points, dtype=np.float32)
        if lm.shape != (98, 2) or not np.isfinite(lm).all() or not math.isfinite(now):
            self.note_unobserved()
            return unknown_eyes('BAD_LANDMARK')
        frontal = pose is None
        yaw = pitch = roll = 0.
        if pose is not None:
            try:
                yaw, pitch, roll = (float(pose[k]) for k in ('yaw', 'pitch', 'roll'))
                if not np.isfinite([yaw, pitch, roll]).all():
                    raise ValueError('non-finite pose')
                if (abs(yaw) > self.max_yaw or abs(pitch) > self.max_pitch or
                        abs(roll) > self.max_roll):
                    self.note_unobserved()
                    return unknown_eyes('POSE_OUT_OF_RANGE')
                frontal = max(abs(yaw), abs(pitch), abs(roll)) <= self.calibration_pose
            except (KeyError, TypeError, ValueError):
                self.note_unobserved()
                return unknown_eyes('BAD_POSE')
        face_height = self._box_height(face_bbox)
        raw = {}
        for side, start in (('left', 60), ('right', 68)):
            eye = lm[start:start + 8]
            ear, width = self._compute(eye)
            signature = self._signature(eye)
            reason = None
            if (ear is None or signature is None or width < self.min_width or
                    not 0 <= ear <= .65):
                reason = 'BAD_EYE_GEOMETRY'
            elif frame_shape is not None:
                h, w = frame_shape[:2]
                if (h <= 0 or w <= 0 or np.any(eye[:, 0] < 0) or
                        np.any(eye[:, 0] >= w) or np.any(eye[:, 1] < 0) or
                        np.any(eye[:, 1] >= h)):
                    reason = 'EYE_OUTSIDE_FRAME'
            if reason is None and face_bbox is not None:
                b = np.asarray(face_bbox[:4], dtype=np.float64)
                if face_height is None:
                    reason = 'BAD_FACE_BOX'
                else:
                    padx, pady = .12 * (b[2] - b[0]), .12 * (b[3] - b[1])
                    if (np.any(eye[:, 0] < b[0] - padx) or np.any(eye[:, 0] > b[2] + padx)
                            or np.any(eye[:, 1] < b[1] - pady) or np.any(eye[:, 1] > b[3] + pady)):
                        reason = 'EYE_OUTSIDE_FACE'
            raw[side] = dict(valid=False, status='UNKNOWN', p_closed=None,
                             reason=reason, width_px=max(width, 0.), ear=ear,
                             signature=signature, filtered_ear=None,
                             expected_width_px=None, face_scale=None,
                             landmark_deviation=None, closed_threshold=None,
                             open_threshold=None)
        if self.calibration_requested and not self.calibrated:
            accepted = self._collect(raw, face_height, frontal, now)
            for side in SIDES:
                raw[side]['calibration_sample_ok'] = accepted
                raw[side]['reason'] = 'CALIBRATING'
                raw[side].pop('signature', None)
            return raw
        result = {}
        for side in SIDES:
            item = raw[side]
            reason = item['reason']
            width = item['width_px']
            if reason is None and self.calibrated:
                if face_height is None:
                    reason = 'BAD_FACE_BOX'
                else:
                    scale = face_height / self.baseline_face_height
                    expected = self.baseline_width_px[side] * scale
                    item['expected_width_px'] = expected
                    item['face_scale'] = scale
                    floor = max(self.min_width, expected * self.width_ratio)
                    if width < floor or expected < self.min_width:
                        reason = 'SMALL_EYE_AT_DISTANCE'
                    elif abs(yaw) <= 20 and not .55 <= width / expected <= 1.75:
                        reason = 'UNSTABLE_EYE_SCALE'
                    else:
                        good, quality_reason, deviation = self._observe_quality(
                            side, width, item['signature'], expected, scale, now,
                            frontal=max(abs(yaw), abs(pitch), abs(roll)) <= 20)
                        item['landmark_deviation'] = deviation
                        if not good:
                            reason = quality_reason
            item.pop('signature', None)  # internal shape feature, not part of API output
            if reason is not None:
                # WAIT_STABLE and noisy shape/width must keep the sliding window:
                # otherwise every following frame would be WAIT_STABLE forever.
                if reason not in ('WAIT_STABLE_EYE', 'UNSTABLE_EYE_WIDTH',
                                  'UNSTABLE_LANDMARK_SHAPE'):
                    self._clear_runtime(side)
                # Do not propagate the last CLOSED through a long run of bad data.
                last_valid = self._last_valid_t[side]
                if last_valid is None or now - last_valid > self.calibration_gap:
                    self._state[side] = 'OPEN'
                self._candidate[side] = None
                self._candidate_count[side] = 0
                self._ear_history[side].clear()
                result[side] = dict(item, reason=reason, status='UNKNOWN', valid=False)
                continue
            if not self.calibrated:
                # Compatibility when callers intentionally bypass calibration.
                low, high = self._thresholds(side)
                if item['ear'] <= low:
                    self._state[side] = 'CLOSED'
                elif item['ear'] >= high:
                    self._state[side] = 'OPEN'
                status, filtered = self._state[side], item['ear']
            else:
                status, filtered, low, high = self._update_state(side, item['ear'])
            self._last_valid_t[side] = now
            result[side] = dict(item, status=status, valid=True, reason=None,
                                filtered_ear=filtered,
                                closed_threshold=low, open_threshold=high)
        return result
