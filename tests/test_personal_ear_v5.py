"""V5 regression: two-state hysteresis, personal statistics, face scale, tracking."""
import numpy as np
import pytest

from app.inference.eye_landmarks import LandmarkEyes, unknown_eyes
from app.eye_temporal import EyeTemporal
from test_eye_baseline import points, POSE, FRAME, FACE


def calibrated(ear=.24):
    model = LandmarkEyes()
    model.start_calibration()
    for i in range(31):
        model.predict(points(ear), POSE, FRAME, FACE, now=.1 * i)
    assert model.calibrated
    return model


def sequence(model, ears, start=3.1, dt=.1, face=FACE, scale=1.):
    last = None
    for index, ear in enumerate(ears):
        last = model.predict(points(ear, scale=scale), POSE, FRAME, face,
                             now=start + index * dt)
    return last


def test_four_calibration_features_are_exposed_separately_for_each_eye():
    model = LandmarkEyes()
    model.start_calibration()
    for i in range(31):
        p = points(.24 + .010 * np.sin(i))
        p[68:76] = points(.135 + .008 * np.cos(i))[68:76]
        # Small natural horizontal landmark jitter during calibration.
        p[61, 0] += .4 * np.sin(i)
        model.predict(p, POSE, FRAME, FACE, now=i*.1)
    assert model.calibrated
    info = model.calibration_info()
    assert info['ear']['left'] > info['ear']['right']
    assert info['variability']['left'] > 0
    assert info['variability']['right'] > 0
    assert info['width_px']['left'] == pytest.approx(20., abs=1)
    assert info['width_px']['right'] == pytest.approx(20., abs=1)
    assert info['face_height_px'] == pytest.approx(100.)
    assert info['landmark_stability']['left'] >= .005
    assert info['landmark_stability']['right'] >= .005


def test_midpoint_ear_is_not_unknown_and_retains_previous_state():
    model = calibrated(.24)
    # Baseline .24 -> 0.1632 close and 0.1776 reopen, .17 is in between.
    assert model._thresholds('left') == pytest.approx((.1632, .1776), abs=.003)
    sequence(model, [.24] * 5)
    for side, item in sequence(model, [.17] * 12, start=3.7).items():
        assert item['status'] == 'OPEN' and item['valid'] and item['reason'] is None
    out = sequence(model, [.12] * 6, start=5.0)
    assert {x['status'] for x in out.values()} == {'CLOSED'}
    out = sequence(model, [.17] * 8, start=5.7)
    assert {x['status'] for x in out.values()} == {'CLOSED'}
    out = sequence(model, [.23] * 6, start=6.6)
    assert {x['status'] for x in out.values()} == {'OPEN'}


def test_noisy_open_baseline_tightens_closed_decision():
    m = LandmarkEyes()
    m.start_calibration()
    for i in range(31):
        m.predict(points(.24 + .035 * np.sin(i*1.4)), POSE, FRAME, FACE, now=i*.1)
    assert m.calibrated
    assert m.baseline_variability['left'] >= .01
    low, high = m._thresholds('left')
    assert low <= m.baseline_ear['left'] * m.close_ratio + 1e-9
    assert high > low
    assert m.calibration_info()['variability']['left'] >= .01


def test_both_closed_required_but_one_open_prevents_false_alert():
    timing = EyeTemporal()
    both = {'left': {'valid':True, 'status':'CLOSED'},
            'right': {'valid':True, 'status':'CLOSED'}}
    for i in range(30):
        out = timing.update(both, i*.1)
    assert out['long_eye_closure']
    partly_open = {'left': {'valid':True,'status':'OPEN'},
                   'right': {'valid':False,'status':'UNKNOWN','reason':'BLURRED'}}
    out = timing.update(partly_open, 3.)
    assert out['eye_status'] == 'OPEN' and not out['long_eye_closure']
    half_closed = {'left': {'valid':True,'status':'CLOSED'},
                   'right': {'valid':False,'status':'UNKNOWN','reason':'BLURRED'}}
    out = timing.update(half_closed, 3.1)
    assert out['eye_status'] == 'UNKNOWN' and out['closed_seconds'] == 0


def test_distance_scale_does_not_multiply_personal_ear_thresholds():
    m = calibrated(.24)
    low0, high0 = m._thresholds('left')
    o = sequence(m, [.24]*5, face=(0, 0, 60, 50), scale=.5)
    assert o['left']['face_scale'] == pytest.approx(.5)
    assert o['left']['status'] == 'OPEN'
    assert m._thresholds('left') == (low0, high0)
    o = sequence(m, [.24]*2, start=4, face=(0,0,30,25), scale=.25)
    assert all(v['status']=='UNKNOWN' for v in o.values())


def test_landmark_horizontal_shape_instability_is_not_confused_with_a_blink():
    m = calibrated(.24)
    sequence(m, [.24]*5)
    # Move paired upper/lower x coordinates together; eyelid gap stays intact.
    out = None
    for i in range(5):
        p = points(.24)
        for first, second in ((61,67),(62,66),(63,65)):
            p[first,0] += 8
            p[second,0] += 8
        out = m.predict(p, POSE, FRAME, FACE, now=4 + i*.1)
    assert out['left']['status'] == 'UNKNOWN'
    assert out['left']['reason'] == 'UNSTABLE_LANDMARK_SHAPE'
    assert out['right']['status'] == 'OPEN'
    # Window can heal when tracking returns; never jam in WAIT_STABLE.
    out = sequence(m, [.24]*6, start=4.6)
    assert out['left']['status'] == 'OPEN'


def test_pose_gate_breaks_observation_and_no_stale_closed_accumulation():
    m = calibrated(.24)
    sequence(m, [.24]*5)
    out = sequence(m, [.10]*6, start=4)
    assert all(x['status']=='CLOSED' for x in out.values())
    bad = m.predict(points(.10), {'yaw':60,'pitch':0,'roll':0}, FRAME, FACE, now=4.7)
    assert all(x['status']=='UNKNOWN' for x in bad.values())
    out = sequence(m, [.17]*6, start=4.8)
    assert all(x['status']=='OPEN' for x in out.values())


def test_no_cross_driver_calibration_leak():
    m = calibrated(.12)
    low1, _ = m._thresholds('left')
    m.start_calibration()
    assert not m.calibrated and m.baseline_variability == {}
    for i in range(31):
        m.predict(points(.30), POSE, FRAME, FACE, now=10 + i*.1)
    assert m.calibrated
    assert m._thresholds('left')[0] > low1


def test_missing_eye_never_counts_as_both_closed():
    t = EyeTemporal()
    for i in range(60):
        result = t.update({'left': {'valid':True,'status':'CLOSED'},
                           'right': {'valid':False,'status':'UNKNOWN','reason':'MISSING'}},
                          i*.05)
    assert not result['eyes_closed']
    assert not result['long_eye_closure']
    assert result['perclos'] == 0
    assert result['eye_status'] == 'UNKNOWN'
