"""Command-line entry point for the GPU detector process."""

from __future__ import annotations

import logging
import signal
import threading

from gpu_detector.application import GpuDetectorApplication
from gpu_detector.config import DetectorConfig
from gpu_detector.logging_utils import configure_logging


LOGGER = logging.getLogger(__name__)


def main() -> int:
    config = DetectorConfig.from_env()
    configure_logging(config.log_level)
    if not config.auth_token:
        LOGGER.warning("DETECTOR_AUTH_TOKEN is empty; only use this configuration on a trusted development host")

    application = GpuDetectorApplication(config)
    stop_requested = threading.Event()

    def request_stop(signum: int, frame: object) -> None:
        LOGGER.info("shutdown requested", extra={"signal": signum})
        stop_requested.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    application.start()
    try:
        stop_requested.wait()
    finally:
        application.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
