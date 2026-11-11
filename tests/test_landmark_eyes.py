import numpy as np
from app.inference.eye_landmarks import LandmarkEyes
from app.eye_temporal import EyeTemporal


def points(height):
    p = np.zeros((98, 2), np.float32)
    for start, offset in ((60, 0), (68, 40)):
        p[start:start+8] = np.array([
            [offset, 20], [offset+5, 20-height], [offset+10, 20-height],
            [offset+15, 20-height], [offset+20, 20], [offset+15, 20+height],
            [offset+10, 20+height], [offset+5, 20+height]], np.float32)
    return p


def test_open_and_closed_geometry():
    model = LandmarkEyes()
    assert all(v['status'] == 'OPEN' for v in model.predict(points(4)).values())
    assert all(v['status'] == 'CLOSED' for v in model.predict(points(1)).values())


def test_pose_gate_and_temporal():
    model = LandmarkEyes()
    assert all(v['status'] == 'UNKNOWN' for v in model.predict(points(4), {'yaw':50,'pitch':0}).values())
    engine = EyeTemporal()
    assert engine.update(model.predict(points(1)), 0)['eye_status'] == 'CLOSED'
    assert engine.update(model.predict(points(1)), 0.2)['eye_status'] == 'CLOSED'
    assert engine.update(model.predict(points(4)), 0.3)['eye_status'] == 'OPEN'


def test_personal_open_reference_and_no_eye_rectangles():
    model = LandmarkEyes()
    model.start_calibration()
    face = (0, 0, 120, 100)
    pose = {'yaw': 0, 'pitch': 0, 'roll': 0}
    for index in range(31):
        out = model.predict(points(4), pose, (100, 120, 3), face, now=index * .1)
        assert all('box' not in eye for eye in out.values())
        assert all(eye['status'] == 'UNKNOWN' for eye in out.values())
    assert model.calibrated
    # Personal baseline is EAR=0.4, so this EAR=0.18 is closed.
    for index in range(6):
        out = model.predict(points(1.8), pose, (100, 120, 3), face, now=3.2 + index * .1)
    assert all(eye['status'] == 'CLOSED' for eye in out.values())


def test_pose_and_visibility_guards_skip_without_a_closed_prediction():
    model = LandmarkEyes()
    good = points(4)
    bad_yaw = model.predict(good, {'yaw':65, 'pitch':0, 'roll':0})
    assert {e['status'] for e in bad_yaw.values()} == {'UNKNOWN'}
    assert {e['reason'] for e in bad_yaw.values()} == {'POSE_OUT_OF_RANGE'}
    bad_roll = model.predict(good, {'yaw':0, 'pitch':0, 'roll':50})
    assert all(not e['valid'] for e in bad_roll.values())
    p = good.copy()
    p[60:68, 0] += 200
    out = model.predict(p, {'yaw':0, 'pitch':0, 'roll':0}, frame_shape=(100,120,3))
    assert out['left']['status'] == 'UNKNOWN'
    assert out['right']['status'] == 'OPEN'
    tracker = EyeTemporal()
    state = tracker.update(out, 0.)
    assert state['eye_status'] == 'OPEN' and not state['long_eye_closure']  # one valid OPEN proves not both CLOSED


def test_unreliable_eye_frames_do_not_create_eye_alerts():
    from app.drowsiness import DrowsinessEngine
    now = 0.0
    engine = DrowsinessEngine(clock=lambda: now)
    landmarks = np.full((98, 2), .5, dtype=np.float32)
    for _ in range(70):
        state = engine.update(landmarks, face_bbox=[100,100,300,300],
                              frame_shape=(480,640,3),
                              pose={'yaw':0,'pitch':0,'roll':0},
                              eye_observation={side: dict(valid=False,status='UNKNOWN',
                                                      reason='POSE_OUT_OF_RANGE')
                                               for side in ('left','right')}, now=now)
        now += .05
        assert state['alert_level'] == 0 and state['alert_key'] is None
    assert state['state'] == 'MONITORING'
