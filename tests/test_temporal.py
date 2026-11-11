import numpy as np
import pytest

from app.eye_temporal import EyeTemporal, EyeTimingConfig
from app.drowsiness import DrowsinessEngine
from app.inference.eye_landmarks import unknown_eyes


def obs(left=0.02, right=None):
    right = left if right is None else right
    return {s: {'valid': p >= .7 or p <= .3, 'status': 'CLOSED' if p >= .7 else 'OPEN' if p <= .3 else 'UNKNOWN', 'p_closed': None, 'width_px': 24.0, 'box': None, 'reason': None}
            for s, p in [('left', left), ('right', right)]}


def landmarks(mar=0.1):
    lm = np.full((98, 2), 0.5, np.float32)
    for i in range(8):
        a = np.pi * i / 4
        lm[88+i] = [0.5 - 0.1*np.cos(a), 0.7 - mar*0.1*np.sin(a)]
    return lm


class Sequence:
    def __init__(self, engine=None):
        self.now = 0.0
        self.engine = engine or DrowsinessEngine(clock=lambda: self.now)
    def run(self, p=0.02, duration=1.0, pose=None, mar=0.1, observation=None):
        pose = {'yaw': 0., 'pitch': 0., 'roll': 0.} if pose is None else pose
        out = None
        for _ in range(round(duration/0.05)+1):
            out = self.engine.update(landmarks(mar), [100, 100, 300, 300], (480, 640, 3),
                                     pose=pose, eye_observation=obs(p) if observation is None else observation,
                                     now=self.now)
            self.now += 0.05
        return out


def test_long_closure_at_startup_without_calibration():
    s = Sequence()
    out = s.run(0.99, 1.45)
    assert not out['drowsy'] and not out['calibrated']
    out = s.run(0.99, 0.10)
    assert out['drowsy'] and out['alert_key'] == 'DROWSY:EYES'


def test_blink_returns_directly_to_awake_without_observing():
    s = Sequence()
    assert s.run(duration=.10)['state'] == 'AWAKE'
    blink = s.run(.99, .20)
    assert blink['state'] == 'BLINK_OR_CLOSING' and not blink['drowsy']
    out = s.run(.01, .05)
    assert out['state'] == 'AWAKE' and out['perclos'] == 0
    assert out['recovery_seconds'] < s.engine.eyes.config.recover_seconds


def test_mouth_open_is_not_awake_and_blocks_drowsy_recovery():
    s = Sequence()
    s.run(duration=2.2)  # calibrate mouth baseline
    assert s.run(duration=.2, mar=1.0)['state'] == 'MOUTH_OPEN'
    assert s.run(duration=.6, mar=1.0)['state'] == 'YAWNING'
    assert s.run(duration=.1, mar=.1)['state'] == 'AWAKE'
    s.run(.99, 1.6)
    assert s.engine._drowsy_active
    assert s.run(.01, 1.4, mar=1.0)['drowsy']  # opened eyes but yawning
    assert not s.run(.01, 1.2, mar=.1)['drowsy']


def test_minor_head_motion_does_not_reset_recovery():
    s = Sequence()
    s.run(.99, 1.6)
    assert s.engine._drowsy_active
    s.run(.01, .5, pose={'yaw':20.,'pitch':0.,'roll':0.})
    # Pitch is large enough to count as looking down, but eye ROIs are
    # within the CNN pose gate and the eye timer should still recover.
    out = s.run(.01, .65, pose={'yaw':20.,'pitch':-22.,'roll':0.})
    assert not out['drowsy']
    assert out['state'] != 'OBSERVING'


def test_extreme_head_turn_without_landmark_eye_data_is_not_awake():
    s = Sequence()
    s.run(duration=2.2)
    for _ in range(30):
        out = s.engine.update(landmarks(), [100, 100, 300, 300], (480, 640, 3),
                              pose={'yaw':50., 'pitch':0., 'roll':0.},
                              eye_observation=unknown_eyes('POSE'), now=s.now)
        s.now += .05
    assert out['state'] == 'LOOK_AWAY'
    assert out['eye_status'] == 'UNKNOWN'
    assert not out['drowsy']


@pytest.mark.parametrize('observation,reason', [(obs(.5),'NO_EYE_MEASUREMENT'),
                                                (unknown_eyes('SMALL_EYE'),'SMALL_EYE')])
def test_invalid_observation_never_means_awake(observation, reason):
    s = Sequence()
    out = s.run(duration=2.2, observation=observation)
    assert out['eye_status'] == 'UNKNOWN' and out['eye_quality_reason'] == reason
    assert out['state'] == 'MONITORING' and out['alert_level'] == 0 and out['alert_key'] is None
    assert out['closed_seconds'] == out['open_seconds'] == 0


def test_disagreeing_eyes_do_not_claim_bilateral_closure():
    s = Sequence()
    out = s.run(duration=2.2, observation=obs(.99,.01))
    assert out['eye_status'] == 'OPEN'
    assert out['eye_quality_reason'] == 'PARTIAL_EYE_OPEN'
    assert out['alert_level'] == 0 and out['closed_seconds'] == 0


def test_unknown_breaks_closure_and_suppresses_stale_alarm():
    s = Sequence()
    assert not s.run(.99,.8)['drowsy']
    s.run(duration=.1, observation=unknown_eyes('LANDMARK_ERROR'))
    assert not s.run(.99,.8)['drowsy']
    assert s.run(.99,.8)['drowsy']
    for reason in ('NO_FACE', 'LANDMARK_ERROR', 'CAMERA_ERROR'):
        s.now += .1
        out = s.engine.observation_failed(reason, s.now)
        assert not out['drowsy'] and out['alert_level'] == 0 and out['eye_status'] == 'UNKNOWN'
    s.engine.reset_calibration()
    assert not s.run(.99,.1)['drowsy']
    assert not s.run(.01,.5)['drowsy']
    assert not s.run(.01,.6)['drowsy']


def test_camera_gap_not_counted_as_eye_closure_or_recovery():
    s = Sequence()
    s.run(.99,.8)
    s.now += 10
    assert not s.run(.99,.4)['drowsy']
    assert s.run(.99,1.2)['drowsy']
    assert s.run(.01,.6)['drowsy']
    s.now += 10
    assert s.run(.01,.6)['drowsy']
    assert not s.run(.01,.5)['drowsy']


def test_eye_nod_combination_and_away():
    s = Sequence()
    assert s.run(duration=2.2)['calibrated']
    out = s.run(.99,.75, pose={'yaw':0.,'pitch':-25.,'roll':0.})
    assert out['drowsy'] and out['alert_key'] == 'DROWSY:EYES_NOD'
    assert out['closed_seconds'] < 1.5
    # With reliable OPEN eyes and a closed mouth, turning the head no longer
    # blocks the 1s DROWSY recovery. Distraction remains separate.
    recovered = s.run(.01,1.2,pose={'yaw':38.,'pitch':0.,'roll':0.})
    assert not recovered['drowsy'] and recovered['state'] == 'LOOK_AWAY'
    assert not s.run(.01,1.6)['drowsy']
    out = s.run(.01,2.2,pose={'yaw':38.,'pitch':0.,'roll':0.})
    assert out['state'] == 'LOOK_AWAY'


def test_mouth_independent_of_eye_roi_quality():
    s = Sequence()
    s.run(duration=2.2)
    bad = unknown_eyes('SMALL_EYE')
    a = s.run(duration=.9, mar=1., observation=bad)
    assert a['yawn_count'] == 1 and a['yawning']
    s.run(duration=.2, mar=.1, observation=bad)
    b = s.run(duration=.9, mar=1., observation=bad)
    assert b['yawn_count'] == 2 and b['drowsy'] and b['alert_key'] == 'DROWSY:YAWN'


def test_no_face_before_calibration_is_not_awake():
    s = Sequence()
    for i in range(50):
        out = s.engine.observation_failed('NO_FACE', i*.05)
    assert out['state'] == 'NO_FACE' and out['eye_status'] == 'UNKNOWN'


def test_repeated_short_blinks_excluded_from_perclos():
    tracker = EyeTemporal()
    now = 0.
    for _ in range(20):
        for value, n in ((.01, 16),(.99, 5)):
            for i in range(n):
                out = tracker.update(obs(value), now)
                now += .05
                assert not out['perclos_event']
    assert out['perclos_ready'] and out['perclos'] == 0.


def test_perclos_event_retained_but_not_immediately_retriggered_after_recovery():
    cfg = EyeTimingConfig(perclos_min_seconds=2., perclos_window=10.)
    s = Sequence(DrowsinessEngine(eye_config=cfg))
    s.run(duration=1.)
    for _ in range(4):
        s.run(.99,.65)
        s.run(.01,.25)
    assert s.engine._drowsy_active
    recovered = s.run(.01,1.1)
    assert not recovered['drowsy']
    assert recovered['perclos'] > cfg.perclos_threshold  # history was not erased
    assert not s.run(.01,.5)['drowsy']  # same old evidence cannot relatch
    assert s.run(.99,1.55)['drowsy']  # a new long closure still triggers


def test_unknown_time_excluded_and_window_clipped_exactly():
    cfg = EyeTimingConfig(perclos_min_seconds=1., perclos_window=2.)
    t = EyeTemporal(cfg)
    for now in np.arange(0, 2.01, .1):
        t.update(obs(.99), now)
    out = t.update(unknown_eyes('NO_FACE'), 2.3)
    assert out['perclos_valid_seconds'] == pytest.approx(1.7)
    assert out['perclos_coverage'] == pytest.approx(.85)
    assert not out['perclos_event']
    out = t.update(unknown_eyes('NO_FACE'), 5.)
    assert out['perclos_valid_seconds'] == 0 and not out['perclos_ready']


def test_missing_pose_prevents_recovery():
    s = Sequence()
    s.run(.99,1.55)
    for i in range(30):
        out = s.engine.update(landmarks(), eye_observation=obs(.01), pose=None, now=s.now)
        s.now += .05
    assert out['eye_status'] == 'UNKNOWN' and not out['drowsy'] and out['alert_level'] == 0


@pytest.mark.parametrize('fps',[5,15,30])
def test_duration_threshold_in_seconds_across_frame_rates(fps):
    tracker = EyeTemporal()
    first_alert = None
    for i in range(3*fps):
        now=i/fps
        out=tracker.update(obs(.99),now)
        if out['long_eye_closure']:
            first_alert=now
            break
    assert 1.5-1e-8 <= first_alert <= 1.5+1/fps


def test_pose_baseline_can_be_learned_after_pose_becomes_available():
    engine=DrowsinessEngine()
    now=0.
    for i in range(50):
        engine.update(landmarks(),[100,100,300,300],(480,640,3),eye_observation=obs(.01),now=now)
        now+=.05
    assert engine.calibrated and engine.pose0 is None
    for i in range(50):
        engine.update(landmarks(),[100,100,300,300],(480,640,3),pose={'yaw':0.,'pitch':0.,'roll':0.},
                      eye_observation=obs(.01),now=now)
        now+=.05
    assert engine.pose0 is not None
