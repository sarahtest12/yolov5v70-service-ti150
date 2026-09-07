from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc
import cv2
import numpy as np

from detector_client import DetectorClient, frame_from_jpeg, pb
from detector_contract import detector_pb2_grpc
import detector_contract
from detector_contract.config import DEFAULT_TRANSPORT, TransportConfig
from demo import load_config


class FakeDetector(detector_pb2_grpc.DetectorServicer):
    def __init__(self):
        self.frames = []
        self.expire_first = True

    def Detect(self, requests, context):
        if dict(context.invocation_metadata()).get("authorization") != "Bearer test-token":
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "invalid bearer token")
        for request in requests:
            self.frames.append(request)
            yield pb.DetectionResult(
                stream_id=request.stream_id, frame_id=request.frame_id,
                observed_at_unix_ms=request.observed_at_unix_ms,
                source_pts=request.source_pts,
                time_base_num=request.time_base_num, time_base_den=request.time_base_den,
                code=pb.RESULT_CODE_EXPIRED if self.expire_first and request.frame_id == 1 else pb.RESULT_CODE_OK,
                detections=[] if self.expire_first and request.frame_id == 1 else [pb.BoundingBox(
                    x1=.1, y1=.2, x2=.5, y2=.8, class_id=0, label="person", confidence=.9,
                )],
            )


class ClientIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.server = grpc.server(self.pool)
        self.detector = FakeDetector()
        detector_pb2_grpc.add_DetectorServicer_to_server(self.detector, self.server)
        port = self.server.add_insecure_port("127.0.0.1:0")
        self.target = f"127.0.0.1:{port}"
        self.server.start()

    def tearDown(self):
        self.server.stop(0).wait()
        self.pool.shutdown(wait=True)

    def frame(self, frame_id):
        return frame_from_jpeg(
            b"test-jpeg-bytes", width=640, height=480,
            stream_id="camera-1", frame_id=frame_id, observed_at_unix_ms=123456,
            source_pts=90000, time_base_num=1, time_base_den=90000,
        )

    def test_stream_preserves_fields_and_continues_after_frame_error(self):
        with DetectorClient(self.target, token="test-token") as client:
            results = list(client.detect([self.frame(1), self.frame(2)], rpc_timeout_seconds=3))
        self.assertEqual([r.code for r in results], [pb.RESULT_CODE_EXPIRED, pb.RESULT_CODE_OK])
        self.assertEqual([r.frame_id for r in results], [1, 2])
        self.assertEqual(results[1].stream_id, "camera-1")
        self.assertEqual(results[1].observed_at_unix_ms, 123456)
        self.assertEqual(results[1].source_pts, 90000)
        self.assertEqual(results[1].time_base_den, 90000)
        self.assertEqual(results[1].detections[0].label, "person")
        self.assertAlmostEqual(results[1].detections[0].confidence, .9)
        self.assertEqual(self.detector.frames[1].jpeg, b"test-jpeg-bytes")
        self.assertEqual(self.detector.frames[1].width, 640)

    def test_bad_token_is_rpc_error(self):
        with DetectorClient(self.target, token="wrong") as client:
            with self.assertRaises(grpc.RpcError) as caught:
                list(client.detect([self.frame(1)], rpc_timeout_seconds=3))
        self.assertEqual(caught.exception.code(), grpc.StatusCode.UNAUTHENTICATED)
        self.assertEqual(self.detector.frames, [])

    def test_explicit_missing_timestamp_is_preserved(self):
        request = frame_from_jpeg(
            b"jpeg", width=1, height=1, stream_id="test", frame_id=1,
            observed_at_unix_ms=0,
        )
        self.assertEqual(request.observed_at_unix_ms, 0)

    def test_demo_runs_from_another_directory_with_unicode_png_and_env_token(self):
        self.detector.expire_first = False
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "config.example.json").read_text())
        config.update(target=self.target, image="测试图片.png", token="wrong", count=2, fps=20)
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8-sig")
            success, png = cv2.imencode(".png", np.zeros((24, 32, 3), dtype=np.uint8))
            self.assertTrue(success)
            (Path(directory) / config["image"]).write_bytes(png.tobytes())
            result = subprocess.run(
                [sys.executable, str(root / "demo.py"), "--config", str(config_path)],
                cwd=directory, env=dict(os.environ, DETECTOR_AUTH_TOKEN="test-token"),
                capture_output=True, text=True, timeout=15,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('"RESULT_CODE_OK": 2', result.stdout)
        self.assertNotIn("test-token", result.stdout)
        self.assertEqual(len(self.detector.frames), 2)
        self.assertEqual(self.detector.frames[0].width, 32)
        self.assertEqual(self.detector.frames[0].height, 24)
        self.assertTrue(self.detector.frames[0].jpeg.startswith(b"\xff\xd8"))


class ConfigurationTest(unittest.TestCase):
    def test_imports_the_shared_module_and_configuration(self):
        shared = Path(__file__).resolve().parents[2] / "shared/detector_contract"
        self.assertEqual(Path(detector_contract.__file__).resolve().parent, shared)
        self.assertIsInstance(DEFAULT_TRANSPORT, TransportConfig)
        self.assertEqual(DEFAULT_TRANSPORT.max_frame_bytes, 4 * 1024 * 1024)

    def test_rejects_nan_fps_and_rpc_timeout_shorter_than_send_duration(self):
        example = json.loads((Path(__file__).resolve().parents[1] / "config.example.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            for overrides in ({"fps": float("nan")}, {"rpc_timeout_seconds": 1}):
                with self.subTest(overrides=overrides):
                    config_path.write_text(json.dumps(example | overrides), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_config(config_path)


if __name__ == "__main__":
    unittest.main()
