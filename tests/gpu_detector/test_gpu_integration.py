from __future__ import annotations

import os
import time
import unittest
from pathlib import Path
from urllib.request import urlopen

import cv2
import grpc

from gpu_detector.application import GpuDetectorApplication
from gpu_detector.config import DetectorConfig
from detector_contract import detector_pb2, detector_pb2_grpc


RUN_GPU_INTEGRATION = os.getenv("RUN_GPU_INTEGRATION") == "1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(RUN_GPU_INTEGRATION, "set RUN_GPU_INTEGRATION=1 to load the CoreX model")
class GpuApplicationIntegrationTest(unittest.TestCase):
    def test_invalid_then_valid_frame_over_real_grpc(self) -> None:
        image_path = PROJECT_ROOT / "tests/fixtures/bus.jpg"
        image = cv2.imread(str(image_path))
        self.assertIsNotNone(image)
        height, width = image.shape[:2]

        config = DetectorConfig(
            grpc_host="127.0.0.1",
            grpc_port=0,
            health_host="127.0.0.1",
            health_port=0,
            auth_token="integration-secret",
            fp16=True,
        )
        application = GpuDetectorApplication(config)
        application.start()
        try:
            health_host, health_port = application.health_address
            with urlopen(f"http://{health_host}:{health_port}/health/ready", timeout=3) as response:
                self.assertEqual(response.status, 200)

            requests = iter((
                detector_pb2.DetectionFrame(
                    stream_id="camera-1",
                    frame_id=1,
                    width=width,
                    height=height,
                    jpeg=b"invalid",
                ),
                detector_pb2.DetectionFrame(
                    stream_id="camera-1",
                    frame_id=2,
                    observed_at_unix_ms=time.time_ns() // 1_000_000 - 5_000,
                    width=width,
                    height=height,
                    jpeg=image_path.read_bytes(),
                ),
                detector_pb2.DetectionFrame(
                    stream_id="camera-1",
                    frame_id=3,
                    observed_at_unix_ms=time.time_ns() // 1_000_000,
                    width=width,
                    height=height,
                    jpeg=image_path.read_bytes(),
                ),
            ))
            with grpc.insecure_channel(f"127.0.0.1:{application.grpc_port}") as channel:
                stub = detector_pb2_grpc.DetectorStub(channel)
                results = list(stub.Detect(
                    requests,
                    metadata=(("authorization", "Bearer integration-secret"),),
                    timeout=20,
                ))

            self.assertEqual(results[0].code, detector_pb2.RESULT_CODE_INVALID_FRAME)
            self.assertEqual(results[1].code, detector_pb2.RESULT_CODE_EXPIRED)
            self.assertEqual(results[2].code, detector_pb2.RESULT_CODE_OK)
            self.assertGreaterEqual(len(results[2].detections), 1)
            self.assertLess(results[2].nms_ms, 20)
        finally:
            application.stop()


if __name__ == "__main__":
    unittest.main()
