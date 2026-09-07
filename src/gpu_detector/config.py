"""Environment-backed configuration for the GPU detector process."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

from detector_contract.config import AUTH_TOKEN_ENV, DEFAULT_TRANSPORT, TransportConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, found {value!r}")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    try:
        return default if value is None else int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, found {value!r}") from error


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    try:
        return default if value is None else float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number, found {value!r}") from error


@dataclass(frozen=True, slots=True)
class ModelConfig:
    weights: Path
    data: Path
    device: str
    image_size: int
    confidence_threshold: float
    iou_threshold: float
    max_detections: int
    fp16: bool
    max_batch_size: int


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    grpc_host: str = "0.0.0.0"
    grpc_port: int = DEFAULT_TRANSPORT.grpc_port
    health_host: str = "127.0.0.1"
    health_port: int = 8081
    device: str = "0"
    weights: Path = PROJECT_ROOT / "models/yolov5s.pt"
    weights_sha256: str = ""
    data: Path = PROJECT_ROOT / "config/coco80.yaml"
    image_size: int = 640
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 300
    fp16: bool = True
    batch_size: int = 1
    batch_wait_ms: float = 0.0
    queue_capacity: int = 8
    max_queue_wait_ms: float = 250.0
    max_frame_age_ms: int = 1000
    max_frame_bytes: int = DEFAULT_TRANSPORT.max_frame_bytes
    max_dimension: int = 16384
    grpc_max_message_bytes: int = DEFAULT_TRANSPORT.max_message_bytes
    grpc_workers: int = 16
    auth_token: str = field(default="", repr=False)
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "DetectorConfig":
        config = cls(
            grpc_host=os.getenv("DETECTOR_GRPC_HOST", "0.0.0.0"),
            grpc_port=_env_int("DETECTOR_GRPC_PORT", DEFAULT_TRANSPORT.grpc_port),
            health_host=os.getenv("DETECTOR_HEALTH_HOST", "127.0.0.1"),
            health_port=_env_int("DETECTOR_HEALTH_PORT", 8081),
            device=os.getenv("DETECTOR_DEVICE", "0"),
            weights=Path(os.getenv("DETECTOR_WEIGHTS", str(PROJECT_ROOT / "models/yolov5s.pt"))),
            weights_sha256=os.getenv("DETECTOR_WEIGHTS_SHA256", "").strip().lower(),
            data=Path(os.getenv("DETECTOR_DATA", str(PROJECT_ROOT / "config/coco80.yaml"))),
            image_size=_env_int("DETECTOR_IMAGE_SIZE", 640),
            confidence_threshold=_env_float("DETECTOR_CONF_THRESHOLD", 0.25),
            iou_threshold=_env_float("DETECTOR_IOU_THRESHOLD", 0.45),
            max_detections=_env_int("DETECTOR_MAX_DETECTIONS", 300),
            fp16=_env_bool("DETECTOR_FP16", True),
            batch_size=_env_int("DETECTOR_BATCH_SIZE", 1),
            batch_wait_ms=_env_float("DETECTOR_BATCH_WAIT_MS", 0.0),
            queue_capacity=_env_int("DETECTOR_QUEUE_CAPACITY", 8),
            max_queue_wait_ms=_env_float("DETECTOR_MAX_QUEUE_WAIT_MS", 250.0),
            max_frame_age_ms=_env_int("DETECTOR_MAX_FRAME_AGE_MS", 1000),
            max_frame_bytes=_env_int("DETECTOR_MAX_FRAME_BYTES", DEFAULT_TRANSPORT.max_frame_bytes),
            max_dimension=_env_int("DETECTOR_MAX_DIMENSION", 16384),
            grpc_max_message_bytes=_env_int("DETECTOR_GRPC_MAX_MESSAGE_BYTES", DEFAULT_TRANSPORT.max_message_bytes),
            grpc_workers=_env_int("DETECTOR_GRPC_WORKERS", 16),
            auth_token=os.getenv(AUTH_TOKEN_ENV, ""),
            log_level=os.getenv("DETECTOR_LOG_LEVEL", "INFO").upper(),
        )
        config.validate()
        return config

    @property
    def grpc_address(self) -> str:
        return f"{self.grpc_host}:{self.grpc_port}"

    def transport_config(self) -> TransportConfig:
        return TransportConfig(
            grpc_port=self.grpc_port,
            max_frame_bytes=self.max_frame_bytes,
            max_message_bytes=self.grpc_max_message_bytes,
        )

    def model_config(self) -> ModelConfig:
        return ModelConfig(
            weights=self.weights,
            data=self.data,
            device=self.device,
            image_size=self.image_size,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            max_detections=self.max_detections,
            fp16=self.fp16,
            max_batch_size=self.batch_size,
        )

    def validate(self) -> None:
        if not 0 <= self.grpc_port <= 65535:
            raise ValueError("DETECTOR_GRPC_PORT must be between 0 and 65535")
        if not 0 <= self.health_port <= 65535:
            raise ValueError("DETECTOR_HEALTH_PORT must be between 0 and 65535")
        if self.image_size <= 0:
            raise ValueError("DETECTOR_IMAGE_SIZE must be positive")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("DETECTOR_CONF_THRESHOLD must be between 0 and 1")
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("DETECTOR_IOU_THRESHOLD must be between 0 and 1")
        if self.max_detections <= 0:
            raise ValueError("DETECTOR_MAX_DETECTIONS must be positive")
        if self.batch_size <= 0:
            raise ValueError("DETECTOR_BATCH_SIZE must be positive")
        if self.batch_wait_ms < 0:
            raise ValueError("DETECTOR_BATCH_WAIT_MS cannot be negative")
        if self.queue_capacity <= 0:
            raise ValueError("DETECTOR_QUEUE_CAPACITY must be positive")
        if self.max_queue_wait_ms < 0:
            raise ValueError("DETECTOR_MAX_QUEUE_WAIT_MS cannot be negative")
        if self.max_frame_age_ms < 0:
            raise ValueError("DETECTOR_MAX_FRAME_AGE_MS cannot be negative")
        if self.max_frame_bytes <= 0:
            raise ValueError("DETECTOR_MAX_FRAME_BYTES must be positive")
        if self.max_dimension <= 0:
            raise ValueError("DETECTOR_MAX_DIMENSION must be positive")
        if self.grpc_max_message_bytes < self.max_frame_bytes:
            raise ValueError("DETECTOR_GRPC_MAX_MESSAGE_BYTES must be at least DETECTOR_MAX_FRAME_BYTES")
        if self.grpc_workers <= 0:
            raise ValueError("DETECTOR_GRPC_WORKERS must be positive")
        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("DETECTOR_LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")
        if self.weights_sha256 and len(self.weights_sha256) != 64:
            raise ValueError("DETECTOR_WEIGHTS_SHA256 must contain 64 hexadecimal characters")
        if self.weights_sha256:
            try:
                int(self.weights_sha256, 16)
            except ValueError as error:
                raise ValueError("DETECTOR_WEIGHTS_SHA256 must be hexadecimal") from error

    def verify_files(self) -> str:
        if not self.weights.is_file():
            raise FileNotFoundError(f"model weights not found: {self.weights}")
        if not self.data.is_file():
            raise FileNotFoundError(f"dataset metadata not found: {self.data}")

        digest = hashlib.sha256()
        with self.weights.open("rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_sha256 = digest.hexdigest()
        if self.weights_sha256 and actual_sha256 != self.weights_sha256:
            raise ValueError(
                f"model SHA-256 mismatch: expected {self.weights_sha256}, found {actual_sha256}"
            )
        return actual_sha256
