import numpy as np
import pytest

from app.inference.nanodet import NanoDetFaceDetector
from app.inference.landmark import Landmark98
from app.inference.eye_landmarks import LandmarkEyes


def test_center_roi_is_full_frame_if_no_zoom():
    frame = np.zeros((720,1280,3), dtype=np.uint8)
    roi, ox, oy = NanoDetFaceDetector._center_crop_for_detection(frame, 1.0)
    assert roi.shape == frame.shape and (ox, oy) == (0, 0)


def test_center_roi_crop_and_bbox_mapping():
    frame = np.zeros((720,1280,3), dtype=np.uint8)
    roi, ox, oy = NanoDetFaceDetector._center_crop_for_detection(frame, 1.25)
    assert roi.shape == (576,1024,3)
    assert (ox,oy) == (128,72)
    detector = NanoDetFaceDetector.__new__(NanoDetFaceDetector)
    detector.center_zoom = 1.25
    detector.input_size = 512
    detector.score_threshold = .25
    detector.nms_threshold = .5
    detector.input_name = 'images'
    class FakeSession:
        def run(self, outputs, feed):
            assert feed['images'].shape == (1,3,512,512)
            return []
    detector.session = FakeSession()
    # Box in letterboxed 512 px coordinates for full-frame box
    # (428,272,528,422); after ROI offset, must restore full frame.
    detector._try_predecoded = lambda x: (
        np.array([[150,212,200,287]], dtype=np.float32),
        np.array([.9], dtype=np.float32),
    )
    result = detector.detect(frame)
    assert len(result) == 1
    assert result[0][:4] == pytest.approx((428,272,528,422), abs=1.0)
    assert result[0][4] == pytest.approx(.9, abs=1e-6)


def test_crop_extends_below_original_chin_bbox():
    lmk = Landmark98.__new__(Landmark98)
    lmk.crop_scale = 1.33
    lmk.crop_y_shift = .035
    lmk.input_size = 256
    frame = np.zeros((720,1280,3), dtype=np.uint8)
    cropped, t = lmk._crop_face(frame, [200,100,400,300])
    assert cropped.shape == (256,256,3)
    assert t['top'] + t['crop_size'] > 300
    assert t['top'] < 100


def test_eight_pixel_eye_remains_skippable_with_pose(monkeypatch):
    monkeypatch.setenv('EAR_MIN_WIDTH_PX','8')
    lmk = LandmarkEyes()
    p = np.zeros((98,2),dtype=np.float32)
    for start, offset in ((60,100),(68,130)):
        p[start:start+8] = np.array([
            [offset,40],[offset+2,38],[offset+4,38],[offset+6,38],
            [offset+8,40],[offset+6,42],[offset+4,42],[offset+2,42]
        ],dtype=np.float32)
    out=lmk.predict(p, pose={'yaw':0,'pitch':0,'roll':0}, frame_shape=(720,1280,3))
    assert all(x['valid'] and x['status']=='OPEN' for x in out.values())
    invalid=lmk.predict(p, pose={'yaw':55,'pitch':0,'roll':0}, frame_shape=(720,1280,3))
    assert all(x['status']=='UNKNOWN' for x in invalid.values())
