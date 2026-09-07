"""Process-level module wiring model, scheduler, transports, and health."""

from __future__ import annotations

import logging
from concurrent import futures

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from gpu_detector.config import DetectorConfig
from gpu_detector.detector import YoloDetector
from gpu_detector.grpc_server import GrpcDetectorServicer
from gpu_detector.health import HealthHttpServer, HealthState
from gpu_detector.metrics import DetectorMetrics
from gpu_detector.scheduler import InferenceScheduler
from detector_contract import detector_pb2_grpc


LOGGER = logging.getLogger(__name__)
DETECTOR_SERVICE_NAME = "detector.v1.Detector"


class GpuDetectorApplication:
    """Own the complete detector process lifecycle behind start/stop/wait."""

    def __init__(self, config: DetectorConfig):
        config.validate()
        self._config = config
        model_sha256 = config.verify_files()
        LOGGER.info("verified model", extra={"model_sha256": model_sha256})

        self._metrics = DetectorMetrics()
        self._state = HealthState()
        self._detector = YoloDetector(config.model_config())
        self._scheduler = InferenceScheduler(
            self._detector,
            queue_capacity=config.queue_capacity,
            batch_size=config.batch_size,
            batch_wait_ms=config.batch_wait_ms,
            max_queue_wait_ms=config.max_queue_wait_ms,
        )
        self._health_http = HealthHttpServer(
            config.health_host,
            config.health_port,
            self._state,
            self._metrics.registry,
        )
        self._grpc_health = health.HealthServicer()
        self._grpc_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=config.grpc_workers, thread_name_prefix="grpc-detector"),
            options=config.transport_config().grpc_options(),
        )
        detector_pb2_grpc.add_DetectorServicer_to_server(
            GrpcDetectorServicer(
                self._scheduler,
                self._metrics,
                max_frame_bytes=config.max_frame_bytes,
                max_dimension=config.max_dimension,
                auth_token=config.auth_token,
                max_frame_age_ms=config.max_frame_age_ms,
            ),
            self._grpc_server,
        )
        health_pb2_grpc.add_HealthServicer_to_server(self._grpc_health, self._grpc_server)
        self._bound_grpc_port = self._grpc_server.add_insecure_port(config.grpc_address)
        if not self._bound_grpc_port:
            raise RuntimeError(f"cannot bind gRPC server to {config.grpc_address}")
        self._started = False

    @property
    def grpc_port(self) -> int:
        return self._bound_grpc_port

    @property
    def health_address(self) -> tuple[str, int]:
        return self._health_http.address

    def start(self) -> None:
        if self._started:
            raise RuntimeError("application has already been started")
        self._state.set_live(True)
        self._health_http.start()
        try:
            self._scheduler.start()
            self._grpc_server.start()
        except BaseException:
            self._state.set_live(False)
            self._health_http.close()
            self._scheduler.close()
            raise

        self._grpc_health.set("", health_pb2.HealthCheckResponse.SERVING)
        self._grpc_health.set(DETECTOR_SERVICE_NAME, health_pb2.HealthCheckResponse.SERVING)
        self._state.set_ready(True)
        self._metrics.set_ready(True)
        self._started = True
        LOGGER.info(
            "GPU detector started",
            extra={
                "grpc_address": f"{self._config.grpc_host}:{self._bound_grpc_port}",
                "health_address": f"{self.health_address[0]}:{self.health_address[1]}",
                "device": self._detector.device_name,
            },
        )

    def wait(self) -> None:
        self._grpc_server.wait_for_termination()

    def stop(self, grace_seconds: float = 5.0) -> None:
        if not self._started:
            return
        self._started = False
        self._state.set_ready(False)
        self._metrics.set_ready(False)
        self._grpc_health.set("", health_pb2.HealthCheckResponse.NOT_SERVING)
        self._grpc_health.set(DETECTOR_SERVICE_NAME, health_pb2.HealthCheckResponse.NOT_SERVING)
        self._grpc_server.stop(grace_seconds).wait(timeout=grace_seconds + 1)
        self._scheduler.close()
        self._state.set_live(False)
        self._health_http.close()
        LOGGER.info("GPU detector stopped")
