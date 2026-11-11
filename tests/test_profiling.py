import csv

import numpy as np
import pytest

from app.inference.nanodet import NanoDetFaceDetector
from app.inference.landmark import Landmark98
from app.profiling import PerformanceLogger


class FakeDetectorSession:
    def run(self, requested, feeds):
        assert feeds['images'].shape == (1, 3, 512, 512)
        return []


def make_detector(profile=True, results=True):
    det = NanoDetFaceDetector.__new__(NanoDetFaceDetector)
    det.profile_enabled = profile
    det.center_zoom = 1.0
    det.input_size = 512
    det.score_threshold = 0.25
    det.nms_threshold = 0.5
    det.input_name = 'images'
    det.session = FakeDetectorSession()
    det._try_predecoded = lambda outputs: (
        np.array([[100, 100, 300, 300]], dtype=np.float32) if results else
        np.zeros((0, 4), dtype=np.float32),
        np.array([0.9], dtype=np.float32) if results else
        np.zeros((0,), dtype=np.float32),
    )
    return det


def test_nanodet_profile_records_true_session_run_duration():
    detector = make_detector()
    detections = detector.detect(np.zeros((512, 512, 3), dtype=np.uint8))
    assert len(detections) == 1
    t = detector.last_timing
    assert set(t) == {'nanodet_pre_ms','nanodet_onnx_ms','nanodet_post_ms','nanodet_total_ms'}
    assert all(0 <= x < 1000 for x in t.values())
    assert t['nanodet_total_ms'] == pytest.approx(
        t['nanodet_pre_ms'] + t['nanodet_onnx_ms'] + t['nanodet_post_ms'], abs=0.02)


def test_nanodet_records_timing_even_when_no_face():
    detector = make_detector(results=False)
    assert detector.detect(np.zeros((512,512,3),dtype=np.uint8)) == []
    assert detector.last_timing['nanodet_onnx_ms'] >= 0


def test_no_profiling_keeps_detector_outputs_unchanged():
    detector = make_detector(profile=False)
    assert len(detector.detect(np.zeros((512, 512, 3), dtype=np.uint8))) == 1
    assert detector.last_timing == {}


class FakeLandmarkSession:
    def run(self, requested, feeds):
        arr = feeds['images']
        assert arr.shape == (1, 3, 256, 256)
        return [np.full((1, 98, 2), 0.5, dtype=np.float32),
                np.array([[1.0, 2.0, 3.0]], dtype=np.float32)]


def make_landmark(profile=True):
    model = Landmark98.__new__(Landmark98)
    model.profile_enabled = profile
    model.crop_scale = 1.25
    model.crop_y_shift = 0
    model.input_size = 256
    model.input_name = 'images'
    model.session = FakeLandmarkSession()
    model.outputs_meta = [type('Meta', (), {'name': n}) for n in ('landmarks','pose')]
    return model


def test_landmark_profile_separates_pre_onnx_post():
    lmk = make_landmark()
    data = lmk.predict(np.zeros((480, 640, 3), dtype=np.uint8), (100, 100, 300, 300))
    assert data['frame'].shape == (98, 2)
    assert data['pose']['yaw'] == 1
    assert lmk.last_timing['landmark_total_ms'] == pytest.approx(
        sum(lmk.last_timing[k] for k in ('landmark_pre_ms','landmark_onnx_ms','landmark_post_ms')),
        abs=0.02)


def test_logger_csv_missing_landmark_is_blank_and_rolling_stats(tmp_path, capsys):
    path = tmp_path / 'perf.csv'
    logger = PerformanceLogger(every=2, csv_path=str(path))
    logger.record({'camera_read_ms': 10.0,'frame_total_ms': 28.0,'faces':1,'face_detected':True,
                   'nanodet_total_ms': 12.0,'nanodet_onnx_ms': 7.0,
                   'landmark_total_ms': 4.0,'landmark_onnx_ms':3.0})
    logger.record({'camera_read_ms': 10.0,'frame_total_ms': 22.0,'faces':0,'face_detected':False,
                   'nanodet_total_ms': 12.0,'nanodet_onnx_ms':7.0})
    logger.close()
    with path.open(newline='') as fd:
        records = list(csv.DictReader(fd))
    assert len(records) == 2
    assert float(records[0]['landmark_total_ms']) >= 0
    assert records[1]['landmark_total_ms'] == ''
    text = capsys.readouterr().out
    assert 'NanoDet total' in text and 'Landmark total' in text
    assert 'face found 1/2' in text
    assert 'FPS~40.0' in text


def test_profile_interval_must_be_positive(tmp_path):
    with pytest.raises(ValueError):
        PerformanceLogger(every=0,csv_path=str(tmp_path/'bad.csv'))
