from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gpu_detector.config import DetectorConfig


class DetectorConfigTest(unittest.TestCase):
    def test_from_env_parses_runtime_values(self) -> None:
        with patch.dict(os.environ, {
            "DETECTOR_GRPC_PORT": "50061",
            "DETECTOR_FP16": "false",
            "DETECTOR_BATCH_SIZE": "4",
            "DETECTOR_BATCH_WAIT_MS": "7.5",
            "DETECTOR_MAX_QUEUE_WAIT_MS": "125.5",
            "DETECTOR_MAX_FRAME_AGE_MS": "800",
        }, clear=False):
            config = DetectorConfig.from_env()

        self.assertEqual(config.grpc_port, 50061)
        self.assertFalse(config.fp16)
        self.assertEqual(config.batch_size, 4)
        self.assertEqual(config.batch_wait_ms, 7.5)
        self.assertEqual(config.max_queue_wait_ms, 125.5)
        self.assertEqual(config.max_frame_age_ms, 800)

    def test_from_env_rejects_invalid_boolean(self) -> None:
        with patch.dict(os.environ, {"DETECTOR_FP16": "sometimes"}, clear=False):
            with self.assertRaisesRegex(ValueError, "DETECTOR_FP16 must be true or false"):
                DetectorConfig.from_env()

    def test_rejects_negative_frame_time_limits(self) -> None:
        with self.assertRaisesRegex(ValueError, "DETECTOR_MAX_QUEUE_WAIT_MS cannot be negative"):
            DetectorConfig(max_queue_wait_ms=-1).validate()
        with self.assertRaisesRegex(ValueError, "DETECTOR_MAX_FRAME_AGE_MS cannot be negative"):
            DetectorConfig(max_frame_age_ms=-1).validate()

    def test_verify_files_returns_and_checks_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "model.pt"
            data = root / "data.yaml"
            weights.write_bytes(b"trusted model")
            data.write_text("names: []\n")
            expected = hashlib.sha256(b"trusted model").hexdigest()

            config = DetectorConfig(weights=weights, data=data, weights_sha256=expected)
            self.assertEqual(config.verify_files(), expected)

            invalid = DetectorConfig(weights=weights, data=data, weights_sha256="0" * 64)
            with self.assertRaisesRegex(ValueError, "model SHA-256 mismatch"):
                invalid.verify_files()


if __name__ == "__main__":
    unittest.main()
