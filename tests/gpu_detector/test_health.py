from __future__ import annotations

import json
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

from prometheus_client import CollectorRegistry, Gauge

from gpu_detector.health import HealthHttpServer, HealthState


class HealthHttpServerTest(unittest.TestCase):
    def test_health_and_metrics_follow_state(self) -> None:
        state = HealthState()
        registry = CollectorRegistry()
        Gauge("test_ready", "test gauge", registry=registry).set(1)
        server = HealthHttpServer("127.0.0.1", 0, state, registry)
        server.start()
        host, port = server.address
        base_url = f"http://{host}:{port}"
        try:
            with self.assertRaises(HTTPError) as captured:
                urlopen(f"{base_url}/health/ready", timeout=2)
            self.assertEqual(captured.exception.code, 503)

            state.set_live(True)
            state.set_ready(True)
            with urlopen(f"{base_url}/health/live", timeout=2) as response:
                self.assertEqual(json.load(response), {"status": "ok"})
            with urlopen(f"{base_url}/health/ready", timeout=2) as response:
                self.assertEqual(response.status, 200)
            with urlopen(f"{base_url}/metrics", timeout=2) as response:
                self.assertIn(b"test_ready 1.0", response.read())
        finally:
            server.close()


if __name__ == "__main__":
    unittest.main()
