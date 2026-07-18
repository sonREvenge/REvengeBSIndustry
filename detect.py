import os
import warnings

import cv2
import numpy as np
import onnxruntime as ort

from utils import load_toml_as_dict
from brawl_industry_logger import get_logger
from _paths import TRT_CACHE_DIR

log = get_logger("brawl_industry.detect")

warnings.filterwarnings(
    "ignore",
    message=".*'pin_memory' argument is set as true but no accelerator is found.*",
    category=UserWarning,
)

_ORT_THREADS = 2


def _build_providers(preferred: str) -> list:
    available = set(ort.get_available_providers())
    providers = []

    if preferred in ("gpu", "auto"):
        if "TensorrtExecutionProvider" in available:
            os.makedirs(TRT_CACHE_DIR, exist_ok=True)
            providers.append((
                "TensorrtExecutionProvider",
                {
                    "trt_fp16_enable": True,
                    "trt_max_workspace_size": 1 << 29,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": TRT_CACHE_DIR,
                },
            ))
        if "CUDAExecutionProvider" in available:
            providers.append(("CUDAExecutionProvider", {
                "cudnn_conv_algo_search": "DEFAULT",
                "cudagraphs": True,
            }))
        if "DmlExecutionProvider" in available:
            providers.append("DmlExecutionProvider")

    providers.append("CPUExecutionProvider")
    return providers


def _numpy_nms(boxes, scores, iou_threshold=0.6):
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)
    x1 = boxes[:, 0]; y1 = boxes[:, 1]
    x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int32)


def _normalize_yolo_output(raw_output):
    prediction = raw_output[0] if isinstance(raw_output, (list, tuple)) else raw_output
    prediction = np.asarray(prediction)
    if prediction.ndim == 3:
        prediction = prediction[0]
    if prediction.ndim != 2:
        raise ValueError(f"Unexpected YOLO output shape: {prediction.shape}")
    if prediction.shape[0] < prediction.shape[1] and prediction.shape[0] <= 256:
        prediction = prediction.T
    return prediction


def _postprocess_raw(raw_output, conf_thresh=0.6, iou_thresh=0.6):
    prediction = _normalize_yolo_output(raw_output)

    n_detections = prediction.shape[0]
    n_classes    = prediction.shape[1] - 4
    if n_classes <= 0:
        return []

    boxes_cxcywh = prediction[:, :4]
    class_scores = prediction[:, 4:]

    class_ids   = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(n_detections), class_ids]

    mask = confidences >= conf_thresh
    if not np.any(mask):
        return []

    boxes_cxcywh = boxes_cxcywh[mask]
    confidences  = confidences[mask]
    class_ids    = class_ids[mask]

    x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    x2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
    y2 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    results = []
    for cls in np.unique(class_ids):
        cls_mask   = class_ids == cls
        cls_boxes  = boxes_xyxy[cls_mask]
        cls_scores = confidences[cls_mask]
        keep = _numpy_nms(cls_boxes, cls_scores, iou_thresh)
        if len(keep) == 0:
            continue
        kept_boxes  = cls_boxes[keep]
        kept_scores = cls_scores[keep]
        kept_cls    = np.full((len(keep), 1), cls, dtype=np.float32)
        det = np.hstack([kept_boxes, kept_scores.reshape(-1, 1), kept_cls])
        results.append(det)

    return results


class Detect:
    def __init__(self, model_path, ignore_classes=None, classes=None, input_size=(640, 640)):
        self.preferred_device = load_toml_as_dict("cfg/general_config.toml").get("cpu_or_gpu", "auto")
        self.model_path       = model_path
        self.classes          = classes
        self.ignore_classes   = set(ignore_classes) if ignore_classes else set()
        self.input_size       = input_size

        if not os.path.exists(model_path):
            from _paths import MODELS_DIR
            raise FileNotFoundError(f"Model not found: {model_path}. Place .onnx files in {MODELS_DIR}")

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = _ORT_THREADS
        so.inter_op_num_threads = _ORT_THREADS
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")

        providers = _build_providers(self.preferred_device)
        self.model = ort.InferenceSession(model_path, sess_options=so, providers=providers)

        active_provider = self.model.get_providers()[0]
        log.info(f"Model {os.path.basename(model_path)} loaded on {active_provider}")

        self.input_name = self.model.get_inputs()[0].name

        self._padded_img_buffer = np.full(
            (1, 3, self.input_size[0], self.input_size[1]),
            128.0 / 255.0,
            dtype=np.float32,
        )

        self._last_resized_w = 0
        self._last_resized_h = 0

        self._use_iobinding = "CUDA" in active_provider or "Tensorrt" in active_provider
        if self._use_iobinding:
            try:
                self._io_binding = self.model.io_binding()
                self._input_ortvalue = ort.OrtValue.ortvalue_from_numpy(self._padded_img_buffer, "cuda", 0)
            except Exception as e:
                log.warning(f"IOBinding init failed ({e}), falling back to model.run()")
                self._use_iobinding = False

    def preprocess_image(self, img):
        h, w = img.shape[:2]
        scale = min(self.input_size[0] / h, self.input_size[1] / w)
        new_w = int(w * scale)
        new_h = int(h * scale)

        if new_w != self._last_resized_w or new_h != self._last_resized_h:
            self._padded_img_buffer[:] = 128.0 / 255.0
            self._last_resized_w = new_w
            self._last_resized_h = new_h

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        img_float = resized.astype(np.float32, copy=True)
        np.multiply(img_float, 1.0 / 255.0, out=img_float)
        self._padded_img_buffer[0, 0, :new_h, :new_w] = img_float[:, :, 0]
        self._padded_img_buffer[0, 1, :new_h, :new_w] = img_float[:, :, 1]
        self._padded_img_buffer[0, 2, :new_h, :new_w] = img_float[:, :, 2]

        return self._padded_img_buffer, new_w, new_h

    def postprocess(self, raw_output, orig_img_shape, resized_shape, conf_thresh=0.6):
        detections = _postprocess_raw(raw_output, conf_thresh=conf_thresh, iou_thresh=0.6)
        orig_h, orig_w       = orig_img_shape
        resized_w, resized_h = resized_shape
        scale_w = orig_w / resized_w
        scale_h = orig_h / resized_h

        results = []
        for det in detections:
            if len(det):
                det[:, 0] *= scale_w
                det[:, 1] *= scale_h
                det[:, 2] *= scale_w
                det[:, 3] *= scale_h
                results.append(det)
        return results

    def detect_objects(self, img, conf_thresh=0.6):
        orig_h, orig_w = img.shape[:2]
        preprocessed_img, resized_w, resized_h = self.preprocess_image(img)

        if self._use_iobinding:
            try:
                np.copyto(self._input_ortvalue.numpy(), preprocessed_img, casting="no")
                self._io_binding.bind_ortvalue_input(self.input_name, self._input_ortvalue)
                self._io_binding.bind_output("output0", "cuda")
                self.model.run_with_iobinding(self._io_binding)
                outputs = self._io_binding.copy_outputs_to_cpu()
            except Exception as e:
                log.warning(f"IOBinding run failed ({e}), disabling")
                self._use_iobinding = False
                outputs = self.model.run(None, {self.input_name: preprocessed_img})
        else:
            outputs = self.model.run(None, {self.input_name: preprocessed_img})

        detections = self.postprocess(outputs, (orig_h, orig_w), (resized_w, resized_h), conf_thresh)

        results = {}
        for detection in detections:
            for row in detection:
                x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                class_id = int(row[5])
                if class_id < 0 or class_id >= len(self.classes):
                    continue
                class_name = self.classes[class_id]
                if class_id in self.ignore_classes or class_name in self.ignore_classes:
                    continue
                results.setdefault(class_name, []).append([x1, y1, x2, y2])

        return results
