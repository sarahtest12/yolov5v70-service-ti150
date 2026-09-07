"""Send a paced, low-rate JPEG stream and report detector integration statistics."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import grpc
from google.protobuf.json_format import MessageToDict

from detector_contract import detector_pb2, detector_pb2_grpc
from detector_contract.config import AUTH_TOKEN_ENV, DEFAULT_TRANSPORT


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return round(ordered[index], 3)


@dataclass(slots=True)
class RunStatistics:
    completed: int = 0
    detections: int = 0
    result_codes: Counter[str] = field(default_factory=Counter)
    latency_ms: list[float] = field(default_factory=list)

    def observe(self, result: detector_pb2.DetectionResult, received_at_unix_ms: int) -> None:
        self.completed += 1
        self.detections += len(result.detections)
        try:
            code_name = detector_pb2.ResultCode.Name(result.code)
        except ValueError:
            code_name = f"RESULT_CODE_UNKNOWN_{result.code}"
        self.result_codes[code_name] += 1
        if result.observed_at_unix_ms:
            self.latency_ms.append(max(0.0, received_at_unix_ms - result.observed_at_unix_ms))

    def summary(self, *, sent: int, elapsed_seconds: float) -> dict[str, object]:
        return {
            "sent": sent,
            "completed": self.completed,
            "detections": self.detections,
            "result_codes": dict(sorted(self.result_codes.items())),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "completed_fps": round(self.completed / elapsed_seconds, 3) if elapsed_seconds else 0.0,
            "latency_ms": {
                "p50": _percentile(self.latency_ms, 0.50),
                "p95": _percentile(self.latency_ms, 0.95),
                "p99": _percentile(self.latency_ms, 0.99),
                "max": round(max(self.latency_ms), 3) if self.latency_ms else None,
            },
        }


class PacedFrameRequests:
    """Generate round-robin stream frames without catch-up bursts."""

    def __init__(
        self,
        *,
        jpeg: bytes,
        width: int,
        height: int,
        stream_count: int,
        fps_per_stream: float,
        duration_seconds: float,
        stream_prefix: str,
        stop_event: threading.Event,
    ):
        self._jpeg = jpeg
        self._width = width
        self._height = height
        self._stream_count = stream_count
        self._total_fps = stream_count * fps_per_stream
        self._duration_seconds = duration_seconds
        self._stream_prefix = stream_prefix
        self._stop_event = stop_event
        self.sent_count = 0

    def __iter__(self) -> Iterator[detector_pb2.DetectionFrame]:
        started = time.monotonic()
        deadline = started + self._duration_seconds
        interval = 1.0 / self._total_fps
        next_send = started
        frame_ids = [0] * self._stream_count

        while not self._stop_event.is_set():
            now = time.monotonic()
            if now >= deadline:
                break
            if self._stop_event.wait(max(0.0, next_send - now)):
                break

            now = time.monotonic()
            if now >= deadline:
                break
            stream_index = self.sent_count % self._stream_count
            frame_ids[stream_index] += 1
            self.sent_count += 1
            yield detector_pb2.DetectionFrame(
                stream_id=f"{self._stream_prefix}-{stream_index + 1}",
                frame_id=frame_ids[stream_index],
                observed_at_unix_ms=time.time_ns() // 1_000_000,
                width=self._width,
                height=self._height,
                jpeg=self._jpeg,
            )

            next_send += interval
            if next_send < now:
                next_send = now + interval


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=f"127.0.0.1:{DEFAULT_TRANSPORT.grpc_port}")
    parser.add_argument("--image", type=Path, default=PROJECT_ROOT / "tests/fixtures/bus.jpg")
    parser.add_argument("--streams", type=int, default=1, help="logical stream count on one gRPC connection")
    parser.add_argument("--fps", type=float, default=5.0, help="frames per second for each logical stream")
    parser.add_argument("--duration", type=float, default=10.0, help="send duration in seconds")
    parser.add_argument("--stream-prefix", default="integration-camera")
    parser.add_argument("--token", default=os.getenv(AUTH_TOKEN_ENV, ""))
    parser.add_argument("--rpc-timeout", type=float, default=0.0, help="0 uses duration plus 30 seconds")
    parser.add_argument("--print-results", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.streams <= 128:
        parser.error("--streams must be between 1 and 128")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.duration <= 0:
        parser.error("--duration must be positive")
    if args.rpc_timeout < 0:
        parser.error("--rpc-timeout cannot be negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError(f"OpenCV could not decode: {args.image}")
    height, width = image.shape[:2]
    stop_event = threading.Event()
    requests = PacedFrameRequests(
        jpeg=args.image.read_bytes(),
        width=width,
        height=height,
        stream_count=args.streams,
        fps_per_stream=args.fps,
        duration_seconds=args.duration,
        stream_prefix=args.stream_prefix,
        stop_event=stop_event,
    )
    statistics = RunStatistics()
    timeout = args.rpc_timeout or args.duration + 30.0
    metadata = (("authorization", f"Bearer {args.token}"),) if args.token else None
    options = DEFAULT_TRANSPORT.grpc_options()
    started = time.monotonic()

    try:
        with grpc.insecure_channel(args.target, options=options) as channel:
            grpc.channel_ready_future(channel).result(timeout=min(timeout, 30.0))
            call = detector_pb2_grpc.DetectorStub(channel).Detect(
                iter(requests),
                metadata=metadata,
                timeout=timeout,
            )
            try:
                for result in call:
                    statistics.observe(result, time.time_ns() // 1_000_000)
                    if args.print_results:
                        print(json.dumps(
                            MessageToDict(result, preserving_proto_field_name=True),
                            ensure_ascii=False,
                        ))
            except KeyboardInterrupt:
                stop_event.set()
                call.cancel()
    except grpc.RpcError as error:
        stop_event.set()
        print(json.dumps({
            "error": "grpc_failed",
            "code": error.code().name,
            "details": error.details(),
        }, ensure_ascii=False))
        return 2

    elapsed = time.monotonic() - started
    summary = statistics.summary(sent=requests.sent_count, elapsed_seconds=elapsed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    failed_codes = {
        name: count
        for name, count in statistics.result_codes.items()
        if name != "RESULT_CODE_OK" and count
    }
    return 1 if requests.sent_count != statistics.completed or failed_codes else 0


if __name__ == "__main__":
    raise SystemExit(main())
