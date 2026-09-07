from __future__ import annotations

import unittest

from detector_contract import detector_pb2
from gpu_detector.continuous_client import RunStatistics, _percentile


class ContinuousClientStatisticsTest(unittest.TestCase):
    def test_summarizes_result_codes_and_client_observed_latency(self) -> None:
        statistics = RunStatistics()
        statistics.observe(detector_pb2.DetectionResult(
            observed_at_unix_ms=1_000,
            code=detector_pb2.RESULT_CODE_OK,
            detections=[detector_pb2.BoundingBox(label="person")],
        ), received_at_unix_ms=1_010)
        statistics.observe(detector_pb2.DetectionResult(
            observed_at_unix_ms=2_000,
            code=detector_pb2.RESULT_CODE_EXPIRED,
        ), received_at_unix_ms=2_040)

        summary = statistics.summary(sent=2, elapsed_seconds=1.0)

        self.assertEqual(summary["completed"], 2)
        self.assertEqual(summary["detections"], 1)
        self.assertEqual(summary["result_codes"], {
            "RESULT_CODE_EXPIRED": 1,
            "RESULT_CODE_OK": 1,
        })
        self.assertEqual(summary["latency_ms"]["p50"], 10.0)
        self.assertEqual(summary["latency_ms"]["p95"], 40.0)

    def test_percentile_is_empty_safe(self) -> None:
        self.assertIsNone(_percentile([], 0.95))

    def test_preserves_unknown_result_code_in_summary(self) -> None:
        statistics = RunStatistics()

        statistics.observe(
            detector_pb2.DetectionResult(code=99),
            received_at_unix_ms=1_000,
        )

        summary = statistics.summary(sent=1, elapsed_seconds=1.0)
        self.assertEqual(summary["result_codes"], {"RESULT_CODE_UNKNOWN_99": 1})


if __name__ == "__main__":
    unittest.main()
