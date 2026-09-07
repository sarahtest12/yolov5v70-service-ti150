"""gRPC adapter mapping the shared detector contract to the inference seam."""

from __future__ import annotations

import hmac
import logging
import time
from collections.abc import Iterable, Iterator
from typing import Protocol

import grpc

from gpu_detector.domain import (
    EncodedFrame,
    FrameExpiredError,
    InferenceOutput,
    InvalidFrameError,
    OverloadedError,
    SchedulerClosedError,
)
from gpu_detector.metrics import DetectorMetrics
from detector_contract import detector_pb2, detector_pb2_grpc


LOGGER = logging.getLogger(__name__)


class InferenceSubmitter(Protocol):
    @property
    def queue_depth(self) -> int: ...

    def submit(self, frame: EncodedFrame) -> InferenceOutput: ...


class GrpcDetectorServicer(detector_pb2_grpc.DetectorServicer):
    def __init__(
        self,
        submitter: InferenceSubmitter,
        metrics: DetectorMetrics,
        *,
        max_frame_bytes: int,
        max_dimension: int,
        auth_token: str,
        max_frame_age_ms: int = 0,
    ):
        self._submitter = submitter
        self._metrics = metrics
        self._max_frame_bytes = max_frame_bytes
        self._max_dimension = max_dimension
        self._auth_token = auth_token
        self._max_frame_age_ms = max_frame_age_ms

    def Detect(
        self,
        request_iterator: Iterable[detector_pb2.DetectionFrame],
        context: grpc.ServicerContext,
    ) -> Iterator[detector_pb2.DetectionResult]:
        self._require_authentication(context)
        for request in request_iterator:
            started = time.perf_counter()
            try:
                self._validate_request(request)
                self._metrics.queue_depth.set(self._submitter.queue_depth)
                output = self._submitter.submit(EncodedFrame(
                    jpeg=request.jpeg,
                    declared_width=request.width,
                    declared_height=request.height,
                ))
                response = self._success(request, output)
                code_name = "ok"
            except FrameExpiredError as error:
                response = self._error(request, detector_pb2.RESULT_CODE_EXPIRED, str(error))
                code_name = "expired"
            except InvalidFrameError as error:
                response = self._error(request, detector_pb2.RESULT_CODE_INVALID_FRAME, str(error))
                code_name = "invalid_frame"
                LOGGER.warning(
                    "invalid detection frame",
                    extra={"stream_id": request.stream_id, "frame_id": request.frame_id, "result_code": code_name},
                )
            except OverloadedError:
                response = self._error(request, detector_pb2.RESULT_CODE_OVERLOADED, "inference queue is full")
                code_name = "overloaded"
                self._metrics.queue_drops.inc()
            except SchedulerClosedError:
                response = self._error(
                    request,
                    detector_pb2.RESULT_CODE_INFERENCE_ERROR,
                    "detector is shutting down",
                )
                code_name = "inference_error"
            except Exception:
                response = self._error(
                    request,
                    detector_pb2.RESULT_CODE_INFERENCE_ERROR,
                    "inference failed",
                )
                code_name = "inference_error"
                LOGGER.exception(
                    "detection inference failed",
                    extra={"stream_id": request.stream_id, "frame_id": request.frame_id, "result_code": code_name},
                )

            elapsed_ms = (time.perf_counter() - started) * 1000
            self._metrics.queue_depth.set(self._submitter.queue_depth)
            self._metrics.record_result(code_name, elapsed_ms, output if code_name == "ok" else None)
            yield response

    def _require_authentication(self, context: grpc.ServicerContext) -> None:
        if not self._auth_token:
            return
        metadata: dict[str, str] = {}
        for item in context.invocation_metadata():
            if hasattr(item, "key") and hasattr(item, "value"):
                key, value = item.key, item.value
            else:
                key, value = item
            metadata[str(key).lower()] = str(value)
        provided = metadata.get("authorization", "")
        expected = f"Bearer {self._auth_token}"
        if not hmac.compare_digest(provided, expected):
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "invalid bearer token")

    def _validate_request(self, request: detector_pb2.DetectionFrame) -> None:
        if not request.stream_id or len(request.stream_id) > 128:
            raise InvalidFrameError("stream_id must contain between 1 and 128 characters")
        if any(ord(character) < 32 for character in request.stream_id):
            raise InvalidFrameError("stream_id cannot contain control characters")
        if request.frame_id == 0:
            raise InvalidFrameError("frame_id must be greater than zero")
        if not request.jpeg:
            raise InvalidFrameError("JPEG payload is empty")
        if len(request.jpeg) > self._max_frame_bytes:
            raise InvalidFrameError(f"JPEG payload exceeds {self._max_frame_bytes} bytes")
        if not 0 < request.width <= self._max_dimension:
            raise InvalidFrameError(f"width must be between 1 and {self._max_dimension}")
        if not 0 < request.height <= self._max_dimension:
            raise InvalidFrameError(f"height must be between 1 and {self._max_dimension}")
        time_base_values = (request.time_base_num, request.time_base_den)
        if any(time_base_values) and not all(time_base_values):
            raise InvalidFrameError("time_base_num and time_base_den must both be zero or both be positive")
        if self._max_frame_age_ms and request.observed_at_unix_ms:
            age_ms = time.time_ns() // 1_000_000 - request.observed_at_unix_ms
            if age_ms > self._max_frame_age_ms:
                raise FrameExpiredError(
                    f"frame age {age_ms} ms exceeds the {self._max_frame_age_ms} ms limit"
                )

    @staticmethod
    def _success(
        request: detector_pb2.DetectionFrame,
        output: InferenceOutput,
    ) -> detector_pb2.DetectionResult:
        return detector_pb2.DetectionResult(
            stream_id=request.stream_id,
            frame_id=request.frame_id,
            observed_at_unix_ms=request.observed_at_unix_ms,
            completed_at_unix_ms=time.time_ns() // 1_000_000,
            detections=[
                detector_pb2.BoundingBox(
                    x1=detection.x1,
                    y1=detection.y1,
                    x2=detection.x2,
                    y2=detection.y2,
                    class_id=detection.class_id,
                    label=detection.label,
                    confidence=detection.confidence,
                )
                for detection in output.detections
            ],
            preprocess_ms=output.timings.preprocess_ms,
            inference_ms=output.timings.inference_ms,
            nms_ms=output.timings.nms_ms,
            code=detector_pb2.RESULT_CODE_OK,
            source_pts=request.source_pts,
            time_base_num=request.time_base_num,
            time_base_den=request.time_base_den,
        )

    @staticmethod
    def _error(
        request: detector_pb2.DetectionFrame,
        code: int,
        message: str,
    ) -> detector_pb2.DetectionResult:
        return detector_pb2.DetectionResult(
            stream_id=request.stream_id,
            frame_id=request.frame_id,
            observed_at_unix_ms=request.observed_at_unix_ms,
            completed_at_unix_ms=time.time_ns() // 1_000_000,
            code=code,
            error_message=message[:512],
            source_pts=request.source_pts,
            time_base_num=request.time_base_num,
            time_base_den=request.time_base_den,
        )
