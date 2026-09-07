from __future__ import annotations

import threading
import time
import unittest
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from gpu_detector.domain import (
    EncodedFrame,
    FrameExpiredError,
    InferenceOutput,
    InferenceTimings,
    OverloadedError,
    SchedulerClosedError,
)
from gpu_detector.scheduler import InferenceScheduler


def output_for(frame: EncodedFrame) -> InferenceOutput:
    return InferenceOutput(
        width=frame.declared_width,
        height=frame.declared_height,
        detections=(),
        timings=InferenceTimings(1.0, 2.0, 3.0),
    )


class RecordingDetector:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def detect_batch(self, frames: Sequence[EncodedFrame]) -> list[InferenceOutput]:
        self.batch_sizes.append(len(frames))
        return [output_for(frame) for frame in frames]


class BlockingDetector(RecordingDetector):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def detect_batch(self, frames: Sequence[EncodedFrame]) -> list[InferenceOutput]:
        self.started.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test did not release fake detector")
        return super().detect_batch(frames)


class InferenceSchedulerTest(unittest.TestCase):
    frame = EncodedFrame(jpeg=b"jpeg", declared_width=10, declared_height=20)

    def test_forms_micro_batch_for_concurrent_submitters(self) -> None:
        detector = RecordingDetector()
        scheduler = InferenceScheduler(detector, queue_capacity=4, batch_size=2, batch_wait_ms=100)
        scheduler.start()
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                outputs = list(pool.map(scheduler.submit, (self.frame, self.frame)))
        finally:
            scheduler.close()

        self.assertEqual(detector.batch_sizes, [2])
        self.assertEqual([output.width for output in outputs], [10, 10])

    def test_rejects_when_pending_queue_is_full(self) -> None:
        detector = BlockingDetector()
        scheduler = InferenceScheduler(detector, queue_capacity=1, batch_size=1, batch_wait_ms=0)
        scheduler.start()
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(scheduler.submit, self.frame)
            self.assertTrue(detector.started.wait(timeout=2))
            second = pool.submit(scheduler.submit, self.frame)
            deadline = time.monotonic() + 2
            while scheduler.queue_depth != 1 and time.monotonic() < deadline:
                time.sleep(0.01)
            with self.assertRaises(OverloadedError):
                scheduler.submit(self.frame)
            detector.release.set()
            first.result(timeout=2)
            second.result(timeout=2)
        scheduler.close()

    def test_rejects_submit_after_close(self) -> None:
        scheduler = InferenceScheduler(RecordingDetector(), queue_capacity=1, batch_size=1, batch_wait_ms=0)
        scheduler.start()
        scheduler.close()
        with self.assertRaises(SchedulerClosedError):
            scheduler.submit(self.frame)

    def test_drops_frame_that_expires_while_waiting_for_gpu(self) -> None:
        detector = BlockingDetector()
        scheduler = InferenceScheduler(
            detector,
            queue_capacity=1,
            batch_size=1,
            batch_wait_ms=0,
            max_queue_wait_ms=25,
        )
        scheduler.start()
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                running = pool.submit(scheduler.submit, self.frame)
                self.assertTrue(detector.started.wait(timeout=2))
                with self.assertRaises(FrameExpiredError):
                    scheduler.submit(self.frame)
                detector.release.set()
                running.result(timeout=2)
        finally:
            detector.release.set()
            scheduler.close()

        self.assertEqual(detector.batch_sizes, [1])


if __name__ == "__main__":
    unittest.main()
