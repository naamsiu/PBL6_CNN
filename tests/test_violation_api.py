import json
from email import policy
from email.parser import BytesParser
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
import requests

from app.api_client import FaceClient, ViolationOutbox


def setup_queue(tmp_path):
    client = FaceClient(base_url='https://example.test', api_key='test-key')
    queue = ViolationOutbox(client, tmp_path, start_worker=False)
    event = queue.enqueue(np.zeros((1200, 2000, 3), np.uint8), 120, .94)
    return client, queue, tmp_path / (event + '.json')


def test_multipart_contract(tmp_path):
    client, queue, path = setup_queue(tmp_path)
    response = Mock(status_code=201, headers={})
    response.json.return_value = {'code': 'VIOLATION_RECORDED'}
    client.session.send = Mock(return_value=response)
    queue.flush_once()
    assert not path.exists()
    request = client.session.send.call_args.args[0]
    assert request.url == 'https://example.test/api/v1/device/violations/'
    assert request.headers['Authorization'] == 'Bearer test-key'
    assert int(request.headers['Content-Length']) == len(request.body)
    assert len(request.body) <= 2 * 1024 * 1024 + 16 * 1024
    mime = BytesParser(policy=policy.default).parsebytes(
        ('Content-Type: ' + request.headers['Content-Type'] + '\r\n\r\n').encode() + request.body)
    parts = {part.get_param('name', header='content-disposition'): part for part in mime.iter_parts()}
    assert set(parts) == {'event_id', 'session_id', 'violation_code', 'detected_at', 'confidence', 'image'}
    assert parts['image'].get_content_type() == 'image/jpeg'
    image = cv2.imdecode(np.frombuffer(parts['image'].get_payload(decode=True), np.uint8), 1)
    assert image.shape[0] <= 1080 and image.shape[1] <= 1920


@pytest.mark.parametrize('failure', [requests.Timeout(), (503, {'code': 'STORAGE_UNAVAILABLE'}), (200, {'code':'UNEXPECTED'})])
def test_retry_preserves_payload_across_restart(tmp_path, failure):
    client, queue, path = setup_queue(tmp_path)
    if isinstance(failure, Exception):
        client.post_violation = Mock(side_effect=failure)
    else:
        client.post_violation = Mock(return_value=failure)
    queue.flush_once()
    original_args = client.post_violation.call_args.args
    entry = json.loads(path.read_text())
    assert not entry['blocked'] and entry['next_attempt'] > 0
    entry['next_attempt'] = 0
    queue._save(path, entry)
    queue2 = ViolationOutbox(client, tmp_path, start_worker=False)
    client.post_violation = Mock(return_value=(200, {'code':'EVENT_ALREADY_RECORDED', 'data':{'status':'APPROVED'}}))
    queue2.flush_once()
    assert client.post_violation.call_args.args == original_args
    assert not path.exists()


def test_rejected_retained_without_auto_retry(tmp_path):
    client, queue, path = setup_queue(tmp_path)
    client.post_violation = Mock(return_value=(400, {'code':'INVALID_SESSION'}))
    queue.flush_once()
    queue.flush_once()
    assert client.post_violation.call_count == 1
    assert json.loads(path.read_text())['blocked']


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -.1, 1.1])
def test_invalid_confidence(tmp_path, value):
    _, queue, _ = setup_queue(tmp_path)
    with pytest.raises(ValueError):
        queue.enqueue(np.zeros((10,10,3), np.uint8), 1, value)


def test_monitor_episode_and_cooldown():
    from app.monitor import FullMonitor
    from unittest.mock import patch
    monitor = FullMonitor.__new__(FullMonitor)
    monitor.violation_outbox = Mock()
    monitor.session_id = 120
    monitor._violation_events = {}
    monitor._no_face_since = None
    monitor._no_face_last = None
    frame = np.zeros((10, 10, 3), np.uint8)
    state = dict(state='DROWSY', drowsy=True, eye_reliable=True, observation_ok=True,
                 alert_key='DROWSY:EYES', p_closed_left=.94, p_closed_right=.96)
    with patch('app.monitor.time.monotonic', return_value=100):
        monitor._queue_violation(frame, state)
        monitor._queue_violation(frame, dict(state, alert_key='DROWSY:PERCLOS'))
    assert monitor.violation_outbox.enqueue.call_count == 1
    assert monitor.violation_outbox.enqueue.call_args.args[2] == 0.0
    with patch('app.monitor.time.monotonic', return_value=110):
        monitor._queue_violation(frame, {'state':'AWAKE', 'observation_ok':True})
        monitor._queue_violation(frame, state)
    assert monitor.violation_outbox.enqueue.call_count == 1
    with patch('app.monitor.time.monotonic', return_value=131):
        monitor._queue_violation(frame, state)
    assert monitor.violation_outbox.enqueue.call_count == 2


@pytest.fixture
def event_monitor():
    from app.monitor import FullMonitor
    monitor = FullMonitor.__new__(FullMonitor)
    monitor.violation_outbox = Mock()
    monitor.session_id = 120
    monitor._violation_events = {}
    monitor._no_face_since = monitor._no_face_last = None
    return monitor


def tick(monitor, now, **state):
    from unittest.mock import patch
    frame = np.zeros((20, 20, 3), np.uint8)
    with patch('app.monitor.time.monotonic', return_value=now):
        monitor._queue_violation(frame, state)


def sent_codes(monitor):
    return [call.kwargs['violation_code'] for call in monitor.violation_outbox.enqueue.call_args_list]


@pytest.mark.parametrize('state,code', [
    ({'distracted': True, 'away_reason':'YAW'}, 'DISTRACTED_LOOK_AWAY'),
    ({'distracted': True, 'away_reason':'LOOK_DOWN'}, 'DISTRACTED_LOOK_DOWN'),
    ({'yawning': True}, 'YAWNING'),
    ({'head_nod': True}, 'HEAD_NOD'),
])
def test_each_behavior_once_until_recovery(event_monitor, state, code):
    for now in [1, 2, 40]:
        tick(event_monitor, now, observation_ok=True, **state)
    assert sent_codes(event_monitor) == [code]
    for now in [41, 41.25, 41.5]:
        tick(event_monitor, now, observation_ok=True)
    tick(event_monitor, 42, observation_ok=True, **state)
    assert sent_codes(event_monitor) == [code, code]


def test_concurrent_behaviors_are_not_hidden_by_drowsy(event_monitor):
    tick(event_monitor, 1, observation_ok=True, state='DROWSY', drowsy=True,
         eye_reliable=True, distracted=True, away_reason='LOOK_DOWN', yawning=True, head_nod=True)
    assert set(sent_codes(event_monitor)) == {'DROWSINESS', 'DISTRACTED_LOOK_DOWN', 'YAWNING', 'HEAD_NOD'}


def test_no_face_continuous_and_real_frame(event_monitor):
    for now in [0, .4, .8, 1.2, 1.6]:
        tick(event_monitor, now, observation_reason='NO_FACE', drowsy=True)
    assert sent_codes(event_monitor) == []
    tick(event_monitor, 2, observation_reason='NO_FACE', drowsy=True)
    tick(event_monitor, 2.4, observation_reason='NO_FACE')
    assert sent_codes(event_monitor) == ['FACE_NOT_VISIBLE']


def test_no_face_gap_resets_timer(event_monitor):
    tick(event_monitor, 0, observation_reason='NO_FACE')
    tick(event_monitor, 3, observation_reason='NO_FACE')
    assert sent_codes(event_monitor) == []


@pytest.mark.parametrize('reason', ['CAMERA_ERROR', 'LANDMARK_ERROR', 'FACE_DETECTOR_ERROR'])
def test_sensor_failure_never_sends_latched_behaviors(event_monitor, reason):
    tick(event_monitor, 10, observation_reason=reason, drowsy=True, distracted=True,
         away_reason='YAW', yawning=True, head_nod=True)
    assert sent_codes(event_monitor) == []


def test_missing_image_does_not_enqueue(event_monitor):
    event_monitor._queue_violation(None, dict(observation_ok=True, drowsy=True, yawning=True))
    assert sent_codes(event_monitor) == []


def test_failed_code_does_not_block_other_codes(event_monitor):
    event_monitor.violation_outbox.enqueue.side_effect = [OSError('disk'), 'id']
    tick(event_monitor, 1, observation_ok=True, eye_reliable=True, drowsy=True, yawning=True)
    assert not event_monitor._violation_events['DROWSINESS']['sent']
    assert event_monitor._violation_events['YAWNING']['sent']
