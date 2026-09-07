from __future__ import annotations

import time
import unittest

import grpc
from prometheus_client import CollectorRegistry

from gpu_detector.domain import Detection, EncodedFrame, InferenceOutput, InferenceTimings
from gpu_detector.grpc_server import GrpcDetectorServicer
from gpu_detector.metrics import DetectorMetrics
from detector_contract import detector_pb2


class AbortedRpc(Exception):
    def __init__(self, code: grpc.StatusCode, details: str):
        super().__init__(details)
        self.code = code


class FakeContext:
    def __init__(self, metadata: tuple[tuple[str, str], ...] = ()):
        self._metadata = tuple(grpc.aio.Metadata(*metadata)) if metadata else ()

    def invocation_metadata(self) -> tuple[object, ...]:
        return self._metadata

    def abort(self, code: grpc.StatusCode, details: str) -> None:
        raise AbortedRpc(code, details)


class FakeSubmitter:
    def __init__(self) -> None:
        self.frames: list[EncodedFrame] = []

    @property
    def queue_depth(self) -> int:
        return 0

    def submit(self, frame: EncodedFrame) -> InferenceOutput:
        self.frames.append(frame)
        return InferenceOutput(
            width=10,
            height=20,
            detections=(Detection(0.1, 0.2, 0.8, 0.9, 5, "bus", 0.75),),
            timings=InferenceTimings(1.0, 2.0, 3.0),
        )


class GrpcDetectorServicerTest(unittest.TestCase):
    def make_servicer(
        self,
        submitter: FakeSubmitter,
        token: str = "",
        max_frame_age_ms: int = 0,
    ) -> GrpcDetectorServicer:
        return GrpcDetectorServicer(
            submitter,
            DetectorMetrics(CollectorRegistry()),
            max_frame_bytes=1024,
            max_dimension=100,
            max_frame_age_ms=max_frame_age_ms,
            auth_token=token,
        )

    @staticmethod
    def valid_request() -> detector_pb2.DetectionFrame:
        return detector_pb2.DetectionFrame(
            stream_id="camera-1",
            frame_id=7,
            observed_at_unix_ms=1000,
            width=10,
            height=20,
            jpeg=b"jpeg",
            source_pts=90,
            time_base_num=1,
            time_base_den=90000,
        )

    def test_maps_successful_inference_to_contract(self) -> None:
        submitter = FakeSubmitter()
        result = next(self.make_servicer(submitter).Detect(iter([self.valid_request()]), FakeContext()))

        self.assertEqual(result.code, detector_pb2.RESULT_CODE_OK)
        self.assertEqual(result.stream_id, "camera-1")
        self.assertEqual(result.frame_id, 7)
        self.assertEqual(result.source_pts, 90)
        self.assertEqual(len(result.detections), 1)
        self.assertEqual(result.detections[0].label, "bus")
        self.assertAlmostEqual(result.detections[0].x1, 0.1)
        self.assertEqual(len(submitter.frames), 1)

    def test_returns_frame_error_without_closing_stream(self) -> None:
        submitter = FakeSubmitter()
        invalid = self.valid_request()
        invalid.jpeg = b""
        valid = self.valid_request()
        results = list(self.make_servicer(submitter).Detect(iter([invalid, valid]), FakeContext()))

        self.assertEqual(results[0].code, detector_pb2.RESULT_CODE_INVALID_FRAME)
        self.assertEqual(results[1].code, detector_pb2.RESULT_CODE_OK)
        self.assertEqual(len(submitter.frames), 1)

    def test_rejects_invalid_bearer_token_at_connection_level(self) -> None:
        servicer = self.make_servicer(FakeSubmitter(), token="secret")
        context = FakeContext((("authorization", "Bearer wrong"),))
        with self.assertRaises(AbortedRpc) as captured:
            list(servicer.Detect(iter([self.valid_request()]), context))
        self.assertEqual(captured.exception.code, grpc.StatusCode.UNAUTHENTICATED)

    def test_returns_expired_without_submitting_and_keeps_stream_open(self) -> None:
        submitter = FakeSubmitter()
        stale = self.valid_request()
        stale.observed_at_unix_ms = time.time_ns() // 1_000_000 - 5_000
        current = self.valid_request()
        current.frame_id = 8
        current.observed_at_unix_ms = time.time_ns() // 1_000_000

        results = list(self.make_servicer(
            submitter,
            max_frame_age_ms=1_000,
        ).Detect(iter([stale, current]), FakeContext()))

        self.assertEqual(results[0].code, detector_pb2.RESULT_CODE_EXPIRED)
        self.assertEqual(results[1].code, detector_pb2.RESULT_CODE_OK)
        self.assertEqual([result.frame_id for result in results], [7, 8])
        self.assertEqual(len(submitter.frames), 1)


if __name__ == "__main__":
    unittest.main()
