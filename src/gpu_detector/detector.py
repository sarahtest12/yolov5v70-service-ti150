"""YOLOv5 inference module for JPEG batches on a CoreX GPU."""

from __future__ import annotations

import time
from collections.abc import Sequence

import cv2
import numpy as np
import torch

from gpu_detector.config import ModelConfig
from gpu_detector.domain import Detection, EncodedFrame, InferenceOutput, InferenceTimings, InvalidFrameError
from models.common import DetectMultiBackend
from utils.dataloaders import letterbox
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device


class YoloDetector:
    """Load one model once and expose batch detection as the sole interface."""

    def __init__(self, config: ModelConfig):
        self._config = config
        self._device = select_device(config.device)
        self._model = DetectMultiBackend(
            config.weights,
            device=self._device,
            data=config.data,
            fp16=config.fp16,
        )
        self._image_size = int(check_img_size(config.image_size, s=self._model.stride))
        self._stride = int(self._model.stride)
        self._model.warmup(imgsz=(config.max_batch_size, 3, self._image_size, self._image_size))
        self._warmup_nms()

    @property
    def device_name(self) -> str:
        return torch.cuda.get_device_name(self._device)

    @torch.inference_mode()
    def detect_batch(self, frames: Sequence[EncodedFrame]) -> list[InferenceOutput]:
        if not frames:
            return []
        if len(frames) > self._config.max_batch_size:
            raise ValueError(
                f"batch has {len(frames)} frames, maximum is {self._config.max_batch_size}"
            )

        preprocess_started = time.perf_counter()
        originals: list[np.ndarray] = []
        tensors: list[np.ndarray] = []
        for frame in frames:
            original = self._decode(frame)
            resized = letterbox(
                original,
                (self._image_size, self._image_size),
                stride=self._stride,
                auto=False,
            )[0]
            rgb_chw = resized.transpose((2, 0, 1))[::-1]
            tensors.append(np.ascontiguousarray(rgb_chw))
            originals.append(original)

        tensor = torch.from_numpy(np.stack(tensors)).to(self._device)
        tensor = tensor.half() if self._model.fp16 else tensor.float()
        tensor /= 255
        self._synchronize()
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000

        inference_started = time.perf_counter()
        prediction = self._model(tensor)
        self._synchronize()
        inference_ms = (time.perf_counter() - inference_started) * 1000

        nms_started = time.perf_counter()
        prediction = non_max_suppression(
            prediction,
            self._config.confidence_threshold,
            self._config.iou_threshold,
            max_det=self._config.max_detections,
        )
        self._synchronize()
        nms_ms = (time.perf_counter() - nms_started) * 1000

        timings = InferenceTimings(
            preprocess_ms=preprocess_ms,
            inference_ms=inference_ms,
            nms_ms=nms_ms,
        )
        outputs: list[InferenceOutput] = []
        for detections, original in zip(prediction, originals):
            height, width = original.shape[:2]
            normalized: list[Detection] = []
            if len(detections):
                detections[:, :4] = scale_boxes(tensor.shape[2:], detections[:, :4], original.shape).round()
                for *xyxy, confidence, class_id in detections.cpu().tolist():
                    numeric_class_id = int(class_id)
                    normalized.append(Detection(
                        x1=self._normalize(xyxy[0], width),
                        y1=self._normalize(xyxy[1], height),
                        x2=self._normalize(xyxy[2], width),
                        y2=self._normalize(xyxy[3], height),
                        class_id=numeric_class_id,
                        label=str(self._model.names[numeric_class_id]),
                        confidence=min(1.0, max(0.0, float(confidence))),
                    ))
            outputs.append(InferenceOutput(
                width=width,
                height=height,
                detections=tuple(normalized),
                timings=timings,
            ))
        return outputs

    @staticmethod
    def _decode(frame: EncodedFrame) -> np.ndarray:
        if not frame.jpeg:
            raise InvalidFrameError("JPEG payload is empty")
        encoded = np.frombuffer(frame.jpeg, dtype=np.uint8)
        original = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if original is None:
            raise InvalidFrameError("JPEG payload cannot be decoded")

        height, width = original.shape[:2]
        if frame.declared_width != width or frame.declared_height != height:
            raise InvalidFrameError(
                f"declared dimensions {frame.declared_width}x{frame.declared_height} "
                f"do not match decoded dimensions {width}x{height}"
            )
        return original

    @staticmethod
    def _normalize(value: float, dimension: int) -> float:
        return min(1.0, max(0.0, float(value) / dimension))

    def _synchronize(self) -> None:
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    @torch.inference_mode()
    def _warmup_nms(self) -> None:
        class_count = len(self._model.names)
        prediction = torch.zeros(
            (self._config.max_batch_size, 2, 5 + class_count),
            device=self._device,
        )
        prediction = prediction.half() if self._model.fp16 else prediction.float()
        prediction[:, :, :4] = torch.tensor(
            [[100.0, 100.0, 50.0, 50.0], [105.0, 105.0, 50.0, 50.0]],
            device=self._device,
            dtype=prediction.dtype,
        )
        prediction[:, :, 4] = 0.9
        prediction[:, :, 5] = 0.9
        non_max_suppression(
            prediction,
            self._config.confidence_threshold,
            self._config.iou_threshold,
            max_det=self._config.max_detections,
        )
        self._synchronize()
