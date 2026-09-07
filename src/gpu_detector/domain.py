"""Types forming the inference seam between transport and model code."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EncodedFrame:
    """One JPEG frame with dimensions declared by the sender."""

    jpeg: bytes
    declared_width: int
    declared_height: int


@dataclass(frozen=True, slots=True)
class Detection:
    """One normalized xyxy detection relative to the decoded source frame."""

    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    label: str
    confidence: float


@dataclass(frozen=True, slots=True)
class InferenceTimings:
    preprocess_ms: float
    inference_ms: float
    nms_ms: float


@dataclass(frozen=True, slots=True)
class InferenceOutput:
    width: int
    height: int
    detections: tuple[Detection, ...]
    timings: InferenceTimings


class InvalidFrameError(ValueError):
    """The encoded frame cannot be safely processed."""


class OverloadedError(RuntimeError):
    """The bounded inference queue cannot accept another frame."""


class FrameExpiredError(RuntimeError):
    """The frame is too old to be useful for real-time inference."""


class SchedulerClosedError(RuntimeError):
    """The inference scheduler is not accepting work."""
