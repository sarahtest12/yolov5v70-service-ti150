"""Low-cardinality Prometheus metrics for the GPU detector."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, REGISTRY

from gpu_detector.domain import InferenceOutput


class DetectorMetrics:
    def __init__(self, registry: CollectorRegistry = REGISTRY):
        self.registry = registry
        self.ready = Gauge("detector_ready", "Whether the detector can accept traffic", registry=registry)
        self.requests = Counter(
            "detector_requests_total",
            "Detection frames completed by result code",
            labelnames=("code",),
            registry=registry,
        )
        self.queue_depth = Gauge(
            "detector_queue_depth",
            "Frames currently waiting in the inference queue",
            registry=registry,
        )
        self.queue_drops = Counter(
            "detector_queue_drop_total",
            "Frames rejected because the inference queue is full",
            registry=registry,
        )
        self.detections = Histogram(
            "detector_detections_per_frame",
            "Number of detections returned for a successful frame",
            buckets=(0, 1, 2, 5, 10, 20, 50, 100, 300),
            registry=registry,
        )
        self.preprocess_ms = Histogram(
            "detector_preprocess_ms",
            "Batch preprocessing duration in milliseconds",
            registry=registry,
        )
        self.inference_ms = Histogram(
            "detector_inference_ms",
            "Batch model inference duration in milliseconds",
            registry=registry,
        )
        self.nms_ms = Histogram(
            "detector_nms_ms",
            "Batch non-maximum suppression duration in milliseconds",
            registry=registry,
        )
        self.end_to_end_ms = Histogram(
            "detector_end_to_end_ms",
            "Time from receiving a frame to creating its response",
            registry=registry,
        )

    def set_ready(self, ready: bool) -> None:
        self.ready.set(1 if ready else 0)

    def record_result(self, code: str, elapsed_ms: float, output: InferenceOutput | None = None) -> None:
        self.requests.labels(code=code).inc()
        self.end_to_end_ms.observe(elapsed_ms)
        if output is not None:
            self.detections.observe(len(output.detections))
            self.preprocess_ms.observe(output.timings.preprocess_ms)
            self.inference_ms.observe(output.timings.inference_ms)
            self.nms_ms.observe(output.timings.nms_ms)
