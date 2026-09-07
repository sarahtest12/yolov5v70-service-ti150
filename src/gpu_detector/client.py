"""Send one or more JPEG frames to a running detector and print JSON results."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import grpc
from google.protobuf.json_format import MessageToDict

from detector_contract import detector_pb2, detector_pb2_grpc
from detector_contract.config import AUTH_TOKEN_ENV, DEFAULT_TRANSPORT


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=f"127.0.0.1:{DEFAULT_TRANSPORT.grpc_port}")
    parser.add_argument("--image", type=Path, default=PROJECT_ROOT / "tests/fixtures/bus.jpg")
    parser.add_argument("--stream-id", default="smoke-test")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--token", default=os.getenv(AUTH_TOKEN_ENV, ""))
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError(f"OpenCV could not decode: {args.image}")
    jpeg = args.image.read_bytes()
    height, width = image.shape[:2]

    def requests() -> Iterator[detector_pb2.DetectionFrame]:
        for frame_id in range(1, args.count + 1):
            yield detector_pb2.DetectionFrame(
                stream_id=args.stream_id,
                frame_id=frame_id,
                observed_at_unix_ms=time.time_ns() // 1_000_000,
                width=width,
                height=height,
                jpeg=jpeg,
            )

    options = DEFAULT_TRANSPORT.grpc_options()
    metadata = (("authorization", f"Bearer {args.token}"),) if args.token else None
    failures = 0
    with grpc.insecure_channel(args.target, options=options) as channel:
        grpc.channel_ready_future(channel).result(timeout=args.timeout)
        stub = detector_pb2_grpc.DetectorStub(channel)
        for response in stub.Detect(requests(), metadata=metadata, timeout=args.timeout):
            print(json.dumps(
                MessageToDict(response, preserving_proto_field_name=True),
                ensure_ascii=False,
                indent=2,
            ))
            if response.code != detector_pb2.RESULT_CODE_OK:
                failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
