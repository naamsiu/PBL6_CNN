"""Synthetic eye geometry tests; no ONNX model required."""
import numpy as np
import pytest

from app.drowsiness import DrowsinessEngine
from app.inference.eye_landmarks import LandmarkEyes

FACE = (0, 0, 120, 100)
FRAME = (160, 160, 3)
POSE = {'yaw': 0., 'pitch': 0., 'roll': 0.}


def points(ear, scale=1.0):
    p = np.zeros((98, 2), np.float32)
    for start, left in ((60, 10.), (68, 60.)):
        height = ear * 20 / 2
        p[start:start + 8] = np.array([
            [left, 30], [left + 5, 30-height], [left + 10, 30-height],
            [left + 15, 30-height], [left + 20, 30],
            [left + 15, 30+height], [left + 10, 30+height],
            [left + 5, 30+height]], np.float32) * scale
    return p


def calibrate(model, ear=0.28, begin=0.0, scale=1.0, face=FACE):
    for t in np.arange(begin, begin + model.calibration_seconds + 0.11, .1):
        obs = model.predict(points(ear, scale=scale), POSE, FRAME, face, now=float(t))
        assert all(item['status'] == 'UNKNOWN' for item in obs.values())
    assert model.calibrated


def test_narrow_eyes_calibrate_from_raw_ear_below_fixed_open_threshold():
    model = LandmarkEyes()
    model.start_calibration()
    calibrate(model, ear=.12)
    assert model.baseline_ear['left'] == pytest.approx(.12, abs=.003)
    assert model._thresholds('left')[1] < .22
    for t in (3.2, 3.3, 3.4):
        obs = model.predict(points(.12), POSE, FRAME, FACE, now=t)
    assert all(v['status'] == 'OPEN' for v in obs.values())
    for t in (3.5, 3.6, 3.7, 3.8, 3.9):
        obs = model.predict(points(.035), POSE, FRAME, FACE, now=t)
    assert all(v['status'] == 'CLOSED' for v in obs.values())


def test_distance_scales_quality_not_normalized_ear():
    model = LandmarkEyes()
    model.start_calibration()
    calibrate(model, ear=.28)
    # Face, eye width, and eyelid distance are all halved.
    for t in (3.0, 3.1, 3.2):
        obs = model.predict(points(.28, .5), POSE, FRAME, (0, 0, 60, 50), now=t)
    assert all(v['status'] == 'OPEN' for v in obs.values())
    assert obs['left']['face_scale'] == pytest.approx(.5)
    assert obs['left']['expected_width_px'] == pytest.approx(10)
    assert obs['left']['ear'] == pytest.approx(.28, abs=.001)
    # Physical resolution is insufficient at one-quarter of initial size.
    small = model.predict(points(.28, .25), POSE, FRAME, (0, 0, 30, 25), now=3.5)
    assert all(v['status'] == 'UNKNOWN' for v in small.values())
    assert all(not v['valid'] for v in small.values())


def test_pose_and_missing_detection_break_calibration_without_eye_alert():
    model = LandmarkEyes()
    model.start_calibration()
    for i in range(15):
        model.predict(points(.15), POSE, FRAME, FACE, now=i * .1)
    assert model.calibration_progress > .4
    bad = model.predict(points(.15), {'yaw': 55, 'pitch': 0, 'roll': 0}, FRAME, FACE, now=1.5)
    assert all(v['status'] == 'UNKNOWN' for v in bad.values())
    assert model.calibration_progress == 0
    for i in range(31):
        model.predict(points(.15), POSE, FRAME, FACE, now=2 + i * .1)
    assert model.calibrated


def test_calibration_disables_eye_alert_until_ready_and_resets_for_next_driver():
    model = LandmarkEyes()
    model.start_calibration()
    engine = DrowsinessEngine(clock=lambda: 0.)
    for i in range(31):
        now = i * .1
        obs = model.predict(points(.12), POSE, FRAME, FACE, now=now)
        assert all(v['status'] == 'UNKNOWN' for v in obs.values())
        fake_mouth = np.full((98, 2), .5, np.float32)
        for k in range(8):
            angle = np.pi * k / 4
            fake_mouth[88+k] = [.5-.1*np.cos(angle), .7-.01*np.sin(angle)]
        status = engine.update(fake_mouth, FACE, FRAME, pose=POSE,
                               eye_observation=obs, now=now)
        assert not status['drowsy'] and not status['eyes_closed']
    assert engine.calibrated   # pose/mouth baseline collected in parallel
    assert model.calibrated
    model.start_calibration()
    assert model.baseline_ear == {} and not model.calibrated
    calibrate(model, ear=.31, begin=10)
    assert model.baseline_ear['left'] == pytest.approx(.31, abs=.003)


def test_eye_width_jitter_is_rejected_when_face_box_size_is_constant():
    model = LandmarkEyes()
    model.start_calibration()
    calibrate(model, ear=.28)
    for now, scale in ((3.2, 1.), (3.3, 1.5), (3.4, .7), (3.5, 1.4)):
        obs = model.predict(points(.28, scale), POSE, FRAME, FACE, now=now)
    assert all(v['status'] == 'UNKNOWN' for v in obs.values())
    for now in (3.6, 3.7, 3.8, 3.9):
        obs = model.predict(points(.28), POSE, FRAME, FACE, now=now)
    assert all(v['status'] == 'OPEN' for v in obs.values())


def test_brief_natural_blink_is_skipped_during_calibration():
    model = LandmarkEyes()
    model.start_calibration()
    for i in range(31):
        ear = .025 if i in (10, 11) else .16
        model.predict(points(ear), POSE, FRAME, FACE, now=i * .1)
    assert model.calibrated
    assert model.baseline_ear['left'] == pytest.approx(.16, abs=.003)


def test_asymmetric_eyes_get_distinct_open_references():
    model = LandmarkEyes()
    model.start_calibration()
    p = points(.15)
    # Right eye's vertical opening is larger (right eye starts at 68).
    q = points(.24)
    p[68:76] = q[68:76]
    for i in range(31):
        model.predict(p, POSE, FRAME, FACE, now=i*.1)
    assert model.calibrated
    assert model.baseline_ear['left'] == pytest.approx(.15, abs=.003)
    assert model.baseline_ear['right'] == pytest.approx(.24, abs=.003)
    l_close, l_open = model._thresholds('left')
    r_close, r_open = model._thresholds('right')
    assert l_close < r_close and l_open < r_open
