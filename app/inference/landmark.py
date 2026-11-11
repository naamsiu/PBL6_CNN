from pathlib import Path
import time
 
import cv2
import numpy as np
import onnxruntime as ort
 
from app.config import (
    LANDMARK_CROP_SCALE,
    LANDMARK_CROP_Y_SHIFT,
    LANDMARK_INPUT_SIZE,
    ORT_THREADS,
)
 
 
class Landmark98:
    """
    Preprocessing khớp notebook landmark:
 
    1. frame BGR -> RGB
    2. square crop quanh bbox
       size = max(w,h) * LANDMARK_CROP_SCALE (training baseline 1.25)
       optional small downward crop shift if detector cuts off chin
    3. pad đen nếu crop tràn frame
    4. resize 256x256
    5. float32 / 255
    6. không mean/std bổ sung
       (notebook cuối dùng mean=[0,0,0], std=[1,1,1])
    7. NCHW
 
    Output model:
      - landmarks : (1, 98, 2) toạ độ chuẩn hoá theo crop [0, 1]
      - pose      : (1, 3) [yaw, pitch, roll] đơn vị độ (nếu model có)
    Model cũ chỉ có landmark vẫn chạy được, pose sẽ là None.
    """
 
    def __init__(
        self,
        model_path,
        crop_scale=LANDMARK_CROP_SCALE,
        crop_y_shift=LANDMARK_CROP_Y_SHIFT,
    ):
        model_path = Path(model_path)
 
        if not model_path.exists():
            raise FileNotFoundError(
                f"Không thấy Landmark ONNX: {model_path}"
            )
 
        self.model_path = str(
            model_path
        )
        self.input_size = (
            LANDMARK_INPUT_SIZE
        )
        self.crop_scale = float(crop_scale)
        self.crop_y_shift = float(crop_y_shift)
        if (not np.isfinite(self.crop_scale) or not 1.0 <= self.crop_scale <= 1.6
                or not np.isfinite(self.crop_y_shift) or abs(self.crop_y_shift) > 0.15):
            raise ValueError("Invalid LANDMARK_CROP_SCALE / LANDMARK_CROP_Y_SHIFT")
 
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(
            1,
            ORT_THREADS,
        )
        options.inter_op_num_threads = 1
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
 
        self.session = (
            ort.InferenceSession(
                self.model_path,
                sess_options=options,
                providers=[
                    "CPUExecutionProvider"
                ],
            )
        )
 
        self.input = (
            self.session.get_inputs()[0]
        )
        self.input_name = (
            self.input.name
        )
        self.outputs_meta = (
            self.session.get_outputs()
        )
        self.has_pose = any(
            "pose" in o.name.lower()
            for o in self.outputs_meta
        ) or len(self.outputs_meta) >= 2
        self.profile_enabled = False
        self.last_timing = {}
 
    def info(self):
        return {
            "has_pose": self.has_pose,
            "model": self.model_path,
            "input": {
                "name": self.input.name,
                "shape": self.input.shape,
                "type": self.input.type,
            },
            "outputs": [
                {
                    "name": o.name,
                    "shape": o.shape,
                    "type": o.type,
                }
                for o in self.outputs_meta
            ],
        }
 
    def _crop_face(
        self,
        frame_bgr,
        bbox,
    ):
        image = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB,
        )
 
        h, w = image.shape[:2]
 
        x1, y1, x2, y2 = map(
            float,
            bbox[:4],
        )
 
        bbox_w = x2 - x1
        bbox_h = y2 - y1
 
        if (
            bbox_w <= 1
            or bbox_h <= 1
        ):
            raise ValueError(
                "Face bbox không hợp lệ."
            )
 
        cx = (x1 + x2) / 2.0
        # A small downward bias helps retain the chin when a detector box
        # ends too high. Keep very small to stay near the training crop.
        cy = (y1 + y2) / 2.0 + bbox_h * self.crop_y_shift
 
        base_size = max(
            bbox_w,
            bbox_h,
        )
 
        crop_size = (
            base_size
            * self.crop_scale
        )
 
        crop_size_int = max(
            2,
            int(
                np.ceil(
                    crop_size
                )
            ),
        )
 
        left = int(
            np.floor(
                cx
                - crop_size / 2.0
            )
        )
        top = int(
            np.floor(
                cy
                - crop_size / 2.0
            )
        )
 
        right = (
            left
            + crop_size_int
        )
        bottom = (
            top
            + crop_size_int
        )
 
        pad_left = max(
            0,
            -left,
        )
        pad_top = max(
            0,
            -top,
        )
        pad_right = max(
            0,
            right - w,
        )
        pad_bottom = max(
            0,
            bottom - h,
        )
 
        if any([
            pad_left,
            pad_top,
            pad_right,
            pad_bottom,
        ]):
            image = cv2.copyMakeBorder(
                image,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
 
        left_p = left + pad_left
        top_p = top + pad_top
 
        crop = image[
            top_p:top_p + crop_size_int,
            left_p:left_p + crop_size_int,
        ]
 
        if crop.size == 0:
            raise ValueError(
                "Face crop rỗng."
            )
 
        crop = cv2.resize(
            crop,
            (
                self.input_size,
                self.input_size,
            ),
            interpolation=cv2.INTER_LINEAR,
        )
 
        transform = {
            "left": left,
            "top": top,
            "crop_size": crop_size_int,
        }
 
        return crop, transform
 
    def _split_outputs(
        self,
        outputs,
    ):
        """
        Tách output thành landmark (196 số) và pose (3 số).
        Ưu tiên theo tên ('landmarks', 'pose'), không có tên thì theo kích thước.
        """
        landmarks = None
        pose_raw = None
 
        for meta, value in zip(
            self.outputs_meta,
            outputs,
        ):
            arr = np.asarray(
                value,
                dtype=np.float32,
            )
            name = meta.name.lower()
 
            if "pose" in name and arr.size == 3:
                pose_raw = arr
            elif "landmark" in name and arr.size == 98 * 2:
                landmarks = arr
 
        for value in outputs:
            arr = np.asarray(
                value,
                dtype=np.float32,
            )
 
            if landmarks is None and arr.size == 98 * 2:
                landmarks = arr
            elif pose_raw is None and arr.size == 3:
                pose_raw = arr
 
        if landmarks is None:
            shapes = [
                tuple(
                    np.asarray(x).shape
                )
                for x in outputs
            ]
 
            raise RuntimeError(
                "Không nhận ra output Landmark ONNX. "
                f"Output shapes: {shapes}"
            )
 
        pose = None
 
        if pose_raw is not None:
            yaw, pitch, roll = [
                float(v)
                for v in pose_raw.reshape(3)
            ]
 
            if np.isfinite([yaw, pitch, roll]).all():
                pose = {
                    "yaw": yaw,
                    "pitch": pitch,
                    "roll": roll,
                }
 
        return landmarks, pose
 
    def predict(
        self,
        frame_bgr,
        bbox,
    ):
        self.last_timing = {}
        start = time.perf_counter() if getattr(self, "profile_enabled", False) else None
        crop_rgb, transform = (
            self._crop_face(
                frame_bgr,
                bbox,
            )
        )
 
        blob = (
            crop_rgb.astype(
                np.float32
            )
            / 255.0
        )
 
        blob = np.transpose(
            blob,
            (2, 0, 1),
        )[None, ...]
 
        blob = np.ascontiguousarray(
            blob,
            dtype=np.float32,
        )
        prep_done = time.perf_counter() if getattr(self, "profile_enabled", False) else None

        outputs = self.session.run(
            None,
            {
                self.input_name: blob
            },
        )
        onnx_done = time.perf_counter() if getattr(self, "profile_enabled", False) else None

        if not outputs:
            raise RuntimeError(
                "Landmark ONNX không có output."
            )
 
        raw, pose = self._split_outputs(
            outputs
        )
 
        landmarks_norm = (
            raw.reshape(98, 2)
        )
 
        left = transform["left"]
        top = transform["top"]
        size = transform[
            "crop_size"
        ]
 
        landmarks_frame = np.empty(
            (98, 2),
            dtype=np.float32,
        )
 
        landmarks_frame[:, 0] = (
            left
            + landmarks_norm[:, 0]
            * size
        )
 
        landmarks_frame[:, 1] = (
            top
            + landmarks_norm[:, 1]
            * size
        )
 
        if getattr(self, "profile_enabled", False):
            end = time.perf_counter()
            self.last_timing = {
                "landmark_pre_ms": (prep_done - start) * 1000.0,
                "landmark_onnx_ms": (onnx_done - prep_done) * 1000.0,
                "landmark_post_ms": (end - onnx_done) * 1000.0,
                "landmark_total_ms": (end - start) * 1000.0,
            }
        return {
            "normalized": landmarks_norm,
            "frame": landmarks_frame,
            "crop_transform": transform,
            "pose": pose,
        }
 