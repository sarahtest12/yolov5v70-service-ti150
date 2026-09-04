"""Bounded single-worker micro-batching for GPU inference."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Protocol

from gpu_detector.domain import EncodedFrame, InferenceOutput, OverloadedError, SchedulerClosedError


class BatchDetector(Protocol):
    def detect_batch(self, frames: Sequence[EncodedFrame]) -> list[InferenceOutput]: ...


@dataclass(slots=True)
class _WorkItem:
    frame: EncodedFrame
    future: Future[InferenceOutput]


_STOP = object()


class InferenceScheduler:
    """Serialize GPU access, bound pending work, and form optional micro-batches."""

    def __init__(
        self,
        detector: BatchDetector,
        *,
        queue_capacity: int,
        batch_size: int,
        batch_wait_ms: float,
    ):
        if queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_wait_ms < 0:
            raise ValueError("batch_wait_ms cannot be negative")

        self._detector = detector
        self._batch_size = batch_size
        self._batch_wait_seconds = batch_wait_ms / 1000
        self._queue: queue.Queue[_WorkItem | object] = queue.Queue(maxsize=queue_capacity)
        self._state_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._accepting = False

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    def start(self) -> None:
        with self._state_lock:
            if self._thread is not None:
                raise RuntimeError("scheduler has already been started")
            self._accepting = True
            self._thread = threading.Thread(target=self._run, name="gpu-inference", daemon=True)
            self._thread.start()

    def submit(self, frame: EncodedFrame) -> InferenceOutput:
        item = _WorkItem(frame=frame, future=Future())
        with self._state_lock:
            if not self._accepting:
                raise SchedulerClosedError("inference scheduler is not accepting work")
            try:
                self._queue.put_nowait(item)
            except queue.Full as error:
                raise OverloadedError("inference queue is full") from error
        return item.future.result()

    def close(self, timeout: float = 30.0) -> None:
        with self._state_lock:
            thread = self._thread
            if thread is None:
                return
            self._accepting = False

        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out while stopping inference scheduler")
            try:
                self._queue.put(_STOP, timeout=min(remaining, 0.1))
                break
            except queue.Full:
                continue

        thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if thread.is_alive():
            raise TimeoutError("inference scheduler worker did not stop")

    def _run(self) -> None:
        stop_after_batch = False
        try:
            while not stop_after_batch:
                first = self._queue.get()
                if first is _STOP:
                    self._queue.task_done()
                    break

                batch = [first]
                deadline = time.monotonic() + self._batch_wait_seconds
                while len(batch) < self._batch_size:
                    try:
                        if self._batch_wait_seconds:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                break
                            candidate = self._queue.get(timeout=remaining)
                        else:
                            candidate = self._queue.get_nowait()
                    except queue.Empty:
                        break

                    if candidate is _STOP:
                        self._queue.task_done()
                        stop_after_batch = True
                        break
                    batch.append(candidate)

                self._process_batch(batch)
                for _ in batch:
                    self._queue.task_done()
        finally:
            self._reject_remaining()

    def _process_batch(self, batch: list[_WorkItem]) -> None:
        try:
            outputs = self._detector.detect_batch([item.frame for item in batch])
            if len(outputs) != len(batch):
                raise RuntimeError(
                    f"detector returned {len(outputs)} results for a batch of {len(batch)}"
                )
        except BaseException as error:
            for item in batch:
                item.future.set_exception(error)
            return

        for item, output in zip(batch, outputs):
            item.future.set_result(output)

    def _reject_remaining(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            try:
                if item is not _STOP:
                    item.future.set_exception(SchedulerClosedError("inference scheduler stopped"))
            finally:
                self._queue.task_done()
