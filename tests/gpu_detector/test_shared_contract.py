from pathlib import Path
import unittest

import detector_contract
from detector_contract.config import DEFAULT_TRANSPORT, TransportConfig
from gpu_detector.config import DetectorConfig


class SharedContractTest(unittest.TestCase):
    def test_gpu_imports_contract_from_the_only_shared_source(self):
        root = Path(__file__).resolve().parents[2]
        self.assertEqual(
            Path(detector_contract.__file__).resolve().parent,
            root / "shared/detector_contract",
        )
        for old_path in ("src/detector_contract", "cpu_client/detector_contract"):
            self.assertFalse(list((root / old_path).glob("*.py")))
            self.assertFalse((root / old_path / "detector.proto").exists())

    def test_server_defaults_and_overrides_use_shared_transport_config(self):
        self.assertEqual(DetectorConfig().transport_config(), DEFAULT_TRANSPORT)
        config = DetectorConfig(grpc_port=51000, max_frame_bytes=512, grpc_max_message_bytes=1024)
        transport = config.transport_config()
        self.assertIsInstance(transport, TransportConfig)
        self.assertEqual(transport.grpc_port, 51000)
        self.assertEqual(dict(transport.grpc_options())["grpc.max_receive_message_length"], 1024)

    def test_rejects_transport_message_limit_smaller_than_frame_limit(self):
        with self.assertRaises(ValueError):
            TransportConfig(max_frame_bytes=1024, max_message_bytes=512)
