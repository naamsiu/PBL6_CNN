from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort

from app.config import (
    NANODET_INPUT_SIZE,
    NANODET_CENTER_ZOOM,
    NANODET_NMS_THRESHOLD,
    NANODET_REG_MAX,
    NANODET_SCORE_THRESHOLD,
    ORT_THREADS,
)


def _sigmoid(x):
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x, axis=-1):
    x = x - np.max(
        x,
        axis=axis,
        keepdims=True,
    )

    exp_x = np.exp(x)

    return exp_x / np.sum(
        exp_x,
        axis=axis,
        keepdims=True,
    )


def _nms(
    boxes,
    scores,
    iou_threshold,
):
    if len(boxes) == 0:
        return np.empty(
            (0,),
            dtype=np.int64,
        )

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (
        np.maximum(
            0.0,
            x2 - x1,
        )
        * np.maximum(
            0.0,
            y2 - y1,
        )
    )

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = int(order[0])

        keep.append(i)

        if order.size == 1:
            break

        rest = order[1:]

        xx1 = np.maximum(
            x1[i],
            x1[rest],
        )

        yy1 = np.maximum(
            y1[i],
            y1[rest],
        )

        xx2 = np.minimum(
            x2[i],
            x2[rest],
        )

        yy2 = np.minimum(
            y2[i],
            y2[rest],
        )

        inter = (
            np.maximum(
                0.0,
                xx2 - xx1,
            )
            * np.maximum(
                0.0,
                yy2 - yy1,
            )
        )

        union = (
            areas[i]
            + areas[rest]
            - inter
        )

        iou = inter / np.maximum(
            union,
            1e-6,
        )

        order = rest[
            iou <= iou_threshold
        ]

    return np.asarray(
        keep,
        dtype=np.int64,
    )


class NanoDetFaceDetector:
    """
    NanoDet face detector.

    Input:
        OpenCV BGR frame

    Preprocess:
        BGR -> RGB
        letterbox -> 512x512
        padding = 114
        float32 / 255
        NCHW

    Model hiện tại của bạn trả:

        boxes  : (1, 14592, 4)
        scores : (1, 14592)

    Tức là bbox đã được decode sẵn.
    """

    def __init__(
        self,
        model_path,
        score_threshold=NANODET_SCORE_THRESHOLD,
        nms_threshold=NANODET_NMS_THRESHOLD,
        center_zoom=NANODET_CENTER_ZOOM,
    ):
        model_path = Path(
            model_path
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Không thấy NanoDet ONNX: "
                f"{model_path}"
            )

        self.model_path = str(
            model_path
        )

        self.input_size = (
            NANODET_INPUT_SIZE
        )

        self.reg_max = (
            NANODET_REG_MAX
        )

        self.score_threshold = (
            score_threshold
        )

        self.nms_threshold = (
            nms_threshold
        )
        self.center_zoom = float(center_zoom)
        if not np.isfinite(self.center_zoom) or not 1.0 <= self.center_zoom <= 2.0:
            raise ValueError("NANODET_CENTER_ZOOM must be between 1.0 and 2.0")

        options = (
            ort.SessionOptions()
        )

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
        # Set from the monitor only when profiling is explicitly enabled.
        self.profile_enabled = False
        self.last_timing = {}

    def info(self):
        return {
            "model": self.model_path,
            "input": {
                "name": self.input.name,
                "shape": self.input.shape,
                "type": self.input.type,
            },
            "outputs": [
                {
                    "name": output.name,
                    "shape": output.shape,
                    "type": output.type,
                }
                for output
                in self.outputs_meta
            ],
        }

    def _letterbox(
        self,
        frame_bgr,
    ):
        h, w = frame_bgr.shape[:2]

        scale = min(
            self.input_size / w,
            self.input_size / h,
        )

        new_w = int(
            round(
                w * scale
            )
        )

        new_h = int(
            round(
                h * scale
            )
        )

        rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB,
        )

        resized = cv2.resize(
            rgb,
            (
                new_w,
                new_h,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        canvas = np.full(
            (
                self.input_size,
                self.input_size,
                3,
            ),
            114,
            dtype=np.uint8,
        )

        pad_x = (
            self.input_size
            - new_w
        ) // 2

        pad_y = (
            self.input_size
            - new_h
        ) // 2

        canvas[
            pad_y:pad_y + new_h,
            pad_x:pad_x + new_w,
        ] = resized

        blob = (
            canvas.astype(
                np.float32
            )
            / 255.0
        )

        blob = np.transpose(
            blob,
            (
                2,
                0,
                1,
            ),
        )[None, ...]

        blob = np.ascontiguousarray(
            blob,
            dtype=np.float32,
        )

        return (
            blob,
            scale,
            pad_x,
            pad_y,
        )

    def _to_nchw(
        self,
        arr,
    ):
        arr = np.asarray(
            arr
        )

        if arr.ndim != 4:
            return arr

        channels = {
            1,
            4 * (
                self.reg_max + 1
            ),
        }

        if arr.shape[1] in channels:
            return arr

        if arr.shape[-1] in channels:
            return np.transpose(
                arr,
                (
                    0,
                    3,
                    1,
                    2,
                ),
            )

        return arr

    def _collect_raw_levels(
        self,
        outputs,
    ):
        """
        Fallback cho trường hợp model export
        raw cls/reg feature maps.
        """

        cls_by_hw = {}
        reg_by_hw = {}

        for meta, raw in zip(
            self.outputs_meta,
            outputs,
        ):
            arr = self._to_nchw(
                np.asarray(raw)
            )

            if arr.ndim != 4:
                continue

            _, c, h, w = (
                arr.shape
            )

            if c == 1:
                cls_by_hw[
                    (h, w)
                ] = arr

            elif (
                c
                == 4
                * (
                    self.reg_max
                    + 1
                )
            ):
                reg_by_hw[
                    (h, w)
                ] = arr

        common = (
            set(cls_by_hw)
            & set(reg_by_hw)
        )

        if not common:
            return []

        shapes = sorted(
            common,
            key=lambda hw: (
                hw[0]
                * hw[1]
            ),
            reverse=True,
        )

        return [
            (
                hw,
                cls_by_hw[hw],
                reg_by_hw[hw],
            )
            for hw in shapes
        ]

    def _decode_raw_levels(
        self,
        levels,
    ):
        """
        Fallback decoder nếu ONNX
        trả raw feature maps.
        """

        all_boxes = []
        all_scores = []

        bins = np.arange(
            self.reg_max + 1,
            dtype=np.float32,
        )

        for (
            (h, w),
            cls_arr,
            reg_arr,
        ) in levels:

            logits = (
                cls_arr[
                    0,
                    0,
                ]
                .reshape(-1)
                .astype(
                    np.float32
                )
            )

            scores = _sigmoid(
                logits
            )

            reg = (
                reg_arr[0]
                .transpose(
                    1,
                    2,
                    0,
                )
                .reshape(
                    h * w,
                    4,
                    self.reg_max + 1,
                )
                .astype(
                    np.float32
                )
            )

            prob = _softmax(
                reg,
                axis=-1,
            )

            distances = (
                prob
                * bins[
                    None,
                    None,
                    :
                ]
            ).sum(
                axis=-1
            )

            stride_x = (
                self.input_size
                / float(w)
            )

            stride_y = (
                self.input_size
                / float(h)
            )

            xs = (
                np.arange(
                    w,
                    dtype=np.float32,
                )
                + 0.5
            ) * stride_x

            ys = (
                np.arange(
                    h,
                    dtype=np.float32,
                )
                + 0.5
            ) * stride_y

            yy, xx = np.meshgrid(
                ys,
                xs,
                indexing="ij",
            )

            points_x = (
                xx.reshape(-1)
            )

            points_y = (
                yy.reshape(-1)
            )

            stride = (
                self.input_size
                / float(w)
            )

            distances = (
                distances
                * stride
            )

            x1 = (
                points_x
                - distances[:, 0]
            )

            y1 = (
                points_y
                - distances[:, 1]
            )

            x2 = (
                points_x
                + distances[:, 2]
            )

            y2 = (
                points_y
                + distances[:, 3]
            )

            boxes = np.stack(
                [
                    x1,
                    y1,
                    x2,
                    y2,
                ],
                axis=1,
            )

            all_boxes.append(
                boxes
            )

            all_scores.append(
                scores
            )

        boxes = np.concatenate(
            all_boxes,
            axis=0,
        )

        scores = np.concatenate(
            all_scores,
            axis=0,
        )

        return (
            boxes,
            scores,
        )

    def _try_predecoded(
        self,
        outputs,
    ):
        """
        Hỗ trợ model ONNX đã decode bbox.

        Model hiện tại:

            boxes
            shape = (1, 14592, 4)

            scores
            shape = (1, 14592)

        Ngoài ra vẫn hỗ trợ:

            [1, N, 5+]

        với:

            x1,y1,x2,y2,score
        """

        by_name = {
            meta.name: np.asarray(
                raw
            )
            for meta, raw in zip(
                self.outputs_meta,
                outputs,
            )
        }

        boxes_arr = None
        scores_arr = None

        # ====================================================
        # Trường hợp chính:
        #
        # boxes  -> [1,N,4]
        # scores -> [1,N]
        # ====================================================

        if "boxes" in by_name:
            candidate = (
                by_name["boxes"]
            )

            if (
                candidate.ndim == 3
                and candidate.shape[0] == 1
                and candidate.shape[-1] == 4
            ):
                boxes_arr = (
                    candidate[0]
                    .astype(
                        np.float32
                    )
                )

        if "scores" in by_name:
            candidate = (
                by_name["scores"]
            )

            if (
                candidate.ndim == 2
                and candidate.shape[0] == 1
            ):
                scores_arr = (
                    candidate[0]
                    .astype(
                        np.float32
                    )
                )

            elif (
                candidate.ndim == 3
                and candidate.shape[0] == 1
                and candidate.shape[-1] == 1
            ):
                scores_arr = (
                    candidate[
                        0,
                        :,
                        0,
                    ]
                    .astype(
                        np.float32
                    )
                )

        if (
            boxes_arr is not None
            and scores_arr is not None
        ):
            if (
                len(boxes_arr)
                != len(scores_arr)
            ):
                raise RuntimeError(
                    "NanoDet ONNX boxes/scores "
                    "không cùng số lượng: "
                    f"boxes={boxes_arr.shape}, "
                    f"scores={scores_arr.shape}"
                )

            finite_scores = (
                scores_arr[
                    np.isfinite(
                        scores_arr
                    )
                ]
            )

            # Nếu score còn là logits.
            if finite_scores.size:
                if (
                    finite_scores.min()
                    < 0.0
                    or finite_scores.max()
                    > 1.0
                ):
                    scores_arr = (
                        _sigmoid(
                            scores_arr
                        )
                    )

            finite_boxes = (
                boxes_arr[
                    np.isfinite(
                        boxes_arr
                    )
                ]
            )

            # Nếu bbox normalized 0..1.
            if (
                finite_boxes.size
                and np.max(
                    np.abs(
                        finite_boxes
                    )
                )
                <= 2.0
            ):
                boxes_arr = (
                    boxes_arr
                    * self.input_size
                )

            return (
                boxes_arr,
                scores_arr,
            )

        # ====================================================
        # Generic:
        #
        # tìm bất kỳ:
        # [1,N,4]
        # +
        # [1,N]
        # ====================================================

        generic_boxes = None
        generic_scores = None

        for raw in outputs:
            arr = np.asarray(
                raw
            )

            if (
                arr.ndim == 3
                and arr.shape[0] == 1
                and arr.shape[-1] == 4
            ):
                generic_boxes = (
                    arr[0]
                    .astype(
                        np.float32
                    )
                )

            elif (
                arr.ndim == 2
                and arr.shape[0] == 1
            ):
                generic_scores = (
                    arr[0]
                    .astype(
                        np.float32
                    )
                )

            elif (
                arr.ndim == 3
                and arr.shape[0] == 1
                and arr.shape[-1] == 1
            ):
                generic_scores = (
                    arr[
                        0,
                        :,
                        0,
                    ]
                    .astype(
                        np.float32
                    )
                )

        if (
            generic_boxes
            is not None
            and generic_scores
            is not None
            and len(
                generic_boxes
            )
            == len(
                generic_scores
            )
        ):
            finite_scores = (
                generic_scores[
                    np.isfinite(
                        generic_scores
                    )
                ]
            )

            if finite_scores.size:
                if (
                    finite_scores.min()
                    < 0.0
                    or finite_scores.max()
                    > 1.0
                ):
                    generic_scores = (
                        _sigmoid(
                            generic_scores
                        )
                    )

            finite_boxes = (
                generic_boxes[
                    np.isfinite(
                        generic_boxes
                    )
                ]
            )

            if (
                finite_boxes.size
                and np.max(
                    np.abs(
                        finite_boxes
                    )
                )
                <= 2.0
            ):
                generic_boxes = (
                    generic_boxes
                    * self.input_size
                )

            return (
                generic_boxes,
                generic_scores,
            )

        # ====================================================
        # Fallback:
        #
        # [1,N,5+]
        #
        # x1 y1 x2 y2 score
        # ====================================================

        for raw in outputs:
            arr = np.asarray(
                raw
            )

            if (
                arr.ndim == 3
                and arr.shape[0] == 1
                and arr.shape[-1] >= 5
            ):
                data = (
                    arr[0]
                    .astype(
                        np.float32
                    )
                )

                boxes = data[
                    :,
                    :4,
                ]

                scores = data[
                    :,
                    4,
                ]

                finite_scores = (
                    scores[
                        np.isfinite(
                            scores
                        )
                    ]
                )

                if finite_scores.size:
                    if (
                        finite_scores.min()
                        < 0.0
                        or finite_scores.max()
                        > 1.0
                    ):
                        scores = (
                            _sigmoid(
                                scores
                            )
                        )

                finite_boxes = (
                    boxes[
                        np.isfinite(
                            boxes
                        )
                    ]
                )

                if (
                    finite_boxes.size
                    and np.max(
                        np.abs(
                            finite_boxes
                        )
                    )
                    <= 2.0
                ):
                    boxes = (
                        boxes
                        * self.input_size
                    )

                return (
                    boxes,
                    scores,
                )

        return None

    @staticmethod
    def _center_crop_for_detection(frame_bgr, zoom):
        """Zoom detection via central ROI, keeping frame-coordinate offsets.

        This does NOT add real optical resolution. Never use the cropped
        region for Landmark, API evidence or head-position calculations.
        """
        h, w = frame_bgr.shape[:2]
        if h < 2 or w < 2 or zoom <= 1.0:
            return frame_bgr, 0, 0
        cw = max(2, min(w, int(round(w / zoom))))
        ch = max(2, min(h, int(round(h / zoom))))
        x0 = (w - cw) // 2
        y0 = (h - ch) // 2
        return frame_bgr[y0:y0 + ch, x0:x0 + cw], x0, y0

    def _profile_result(self, detections, start, prep_done, onnx_done):
        if getattr(self, "profile_enabled", False):
            end = time.perf_counter()
            self.last_timing = {
                "nanodet_pre_ms": (prep_done - start) * 1000.0,
                "nanodet_onnx_ms": (onnx_done - prep_done) * 1000.0,
                "nanodet_post_ms": (end - onnx_done) * 1000.0,
                "nanodet_total_ms": (end - start) * 1000.0,
            }
        return detections

    def detect(
        self,
        frame_bgr,
    ):
        self.last_timing = {}
        start = time.perf_counter() if getattr(self, "profile_enabled", False) else None
        original_h, original_w = frame_bgr.shape[:2]
        detection_frame, roi_x, roi_y = self._center_crop_for_detection(
            frame_bgr, self.center_zoom
        )
        (
            blob,
            scale,
            pad_x,
            pad_y,
        ) = self._letterbox(
            detection_frame
        )
        prep_done = time.perf_counter() if getattr(self, "profile_enabled", False) else None

        outputs = (
            self.session.run(
                None,
                {
                    self.input_name:
                    blob
                },
            )
        )
        onnx_done = time.perf_counter() if getattr(self, "profile_enabled", False) else None

        # ====================================================
        # Ưu tiên ONNX đã decode.
        #
        # Model hiện tại của bạn:
        # boxes  (1,14592,4)
        # scores (1,14592)
        # ====================================================

        decoded = (
            self._try_predecoded(
                outputs
            )
        )

        if decoded is not None:
            boxes, scores = (
                decoded
            )

        else:
            # Fallback cho model raw.
            levels = (
                self._collect_raw_levels(
                    outputs
                )
            )

            if not levels:
                shapes = [
                    (
                        meta.name,
                        tuple(
                            np.asarray(
                                output
                            ).shape
                        ),
                    )
                    for meta, output
                    in zip(
                        self.outputs_meta,
                        outputs,
                    )
                ]

                raise RuntimeError(
                    "Không nhận ra output "
                    "NanoDet ONNX. "
                    f"Outputs thực tế: "
                    f"{shapes}"
                )

            boxes, scores = (
                self._decode_raw_levels(
                    levels
                )
            )

        boxes = np.asarray(
            boxes,
            dtype=np.float32,
        )

        scores = np.asarray(
            scores,
            dtype=np.float32,
        ).reshape(-1)

        # Loại NaN/Inf.
        valid_finite = (
            np.isfinite(
                boxes
            ).all(axis=1)
            & np.isfinite(
                scores
            )
        )

        boxes = boxes[
            valid_finite
        ]

        scores = scores[
            valid_finite
        ]

        if len(boxes) == 0:
            return self._profile_result([], start, prep_done, onnx_done)

        # Bbox đang ở coordinate của
        # input 512x512.
        boxes[:, 0::2] = np.clip(
            boxes[:, 0::2],
            0,
            self.input_size,
        )

        boxes[:, 1::2] = np.clip(
            boxes[:, 1::2],
            0,
            self.input_size,
        )

        # Score threshold.
        keep = (
            scores
            >= self.score_threshold
        )

        boxes = boxes[
            keep
        ]

        scores = scores[
            keep
        ]

        if len(boxes) == 0:
            return self._profile_result([], start, prep_done, onnx_done)

        # Loại bbox lỗi.
        wh = (
            boxes[:, 2:4]
            - boxes[:, 0:2]
        )

        valid = (
            (wh[:, 0] > 1.0)
            & (wh[:, 1] > 1.0)
        )

        boxes = boxes[
            valid
        ]

        scores = scores[
            valid
        ]

        if len(boxes) == 0:
            return self._profile_result([], start, prep_done, onnx_done)

        # NMS.
        keep_idx = _nms(
            boxes,
            scores,
            self.nms_threshold,
        )

        boxes = boxes[
            keep_idx
        ]

        scores = scores[
            keep_idx
        ]

        # ====================================================
        # Letterbox 512 coordinate
        # -> original camera coordinate
        # ====================================================

        boxes[:, [0, 2]] = (
            boxes[:, [0, 2]]
            - pad_x
        ) / scale

        boxes[:, [1, 3]] = (
            boxes[:, [1, 3]]
            - pad_y
        ) / scale

        frame_h, frame_w = (
            detection_frame.shape[:2]
        )

        boxes[:, 0::2] = np.clip(
            boxes[:, 0::2],
            0,
            frame_w - 1,
        )

        boxes[:, 1::2] = np.clip(
            boxes[:, 1::2],
            0,
            frame_h - 1,
        )

        # Restore boxes to full-frame coordinate system. Consumers (landmark,
        # API, posture, authentication) must NOT see the zoomed coordinates.
        boxes[:, [0, 2]] += roi_x
        boxes[:, [1, 3]] += roi_y
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, original_w - 1)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, original_h - 1)

        detections = []

        for box, score in zip(
            boxes,
            scores,
        ):
            detections.append(
                (
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                    float(score),
                )
            )

        detections.sort(
            key=lambda x: x[4],
            reverse=True,
        )

        return self._profile_result(detections, start, prep_done, onnx_done)