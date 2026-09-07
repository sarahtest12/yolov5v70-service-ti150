"""读取配置，按指定帧率重复发送图片，打印每帧结果和联调统计。"""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
from collections import Counter
from pathlib import Path

import cv2
import grpc
import numpy as np
from google.protobuf.json_format import MessageToDict

from detector_client import DetectorClient, frame_from_jpeg, pb
from detector_contract.config import AUTH_TOKEN_ENV, DEFAULT_TRANSPORT


def load_config(path: Path) -> dict:
    # utf-8-sig 同时兼容 Windows 编辑器保存的带 BOM UTF-8 文件。
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    for key in ("target", "image", "stream_id"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"配置 {key} 必须是非空字符串")
    if not isinstance(config.get("token", ""), str):
        raise ValueError("配置 token 必须是字符串")
    count = config.get("count")
    if type(count) is not int or count <= 0:
        raise ValueError("配置 count 必须是正整数")
    for key in ("fps", "connect_timeout_seconds", "rpc_timeout_seconds"):
        value = config.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"配置 {key} 必须是有限的正数")
    if config["rpc_timeout_seconds"] <= (count - 1) / config["fps"]:
        raise ValueError("rpc_timeout_seconds 是整次调用时限，应大于发送时长并留出响应余量")
    if len(config["stream_id"]) > 128 or any(ord(c) < 32 for c in config["stream_id"]):
        raise ValueError("stream_id 最多 128 字符，且不能包含控制字符")
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        # 相对图片路径始终相对于配置文件，而不是终端工作目录。
        image_path = args.config.resolve().parent / config["image"]
        image = cv2.imdecode(np.frombuffer(image_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法解码图片：{image_path}")
        height, width = image.shape[:2]
        # 统一编码为 JPEG；输入 test.png 等图片也可用于联调。
        success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise ValueError("JPEG 编码失败")
        jpeg = encoded.tobytes()
        if len(jpeg) > DEFAULT_TRANSPORT.max_frame_bytes:
            raise ValueError(f"JPEG 超过默认 {DEFAULT_TRANSPORT.max_frame_bytes} 字节限制，请缩小测试图片")
    except (OSError, ValueError, TypeError, AttributeError, cv2.error) as error:
        print(json.dumps({"error": "config_or_image_error", "message": str(error)}, ensure_ascii=False))
        return 2

    sent = 0
    completed = 0
    codes: Counter[str] = Counter()
    latencies: list[float] = []
    stop = threading.Event()
    interval = 1.0 / config["fps"]

    def requests():
        nonlocal sent
        next_send = time.monotonic()
        for frame_id in range(1, config["count"] + 1):
            if stop.wait(max(0.0, next_send - time.monotonic())):
                return
            sent += 1
            yield frame_from_jpeg(
                jpeg, width=width, height=height,
                stream_id=config["stream_id"], frame_id=frame_id,
            )
            # 发送若被背压拖慢，从当前时间重新计时，不补发积压帧。
            next_send = max(next_send + interval, time.monotonic() + interval)

    started = time.monotonic()
    exit_code = 0
    try:
        with DetectorClient(
            config["target"],
            token=os.environ.get(AUTH_TOKEN_ENV, config.get("token", "")),
            connect_timeout_seconds=config["connect_timeout_seconds"],
        ) as client:
            for result in client.detect(requests(), rpc_timeout_seconds=config["rpc_timeout_seconds"]):
                completed += 1
                try:
                    code = pb.ResultCode.Name(result.code)
                except ValueError:
                    code = f"UNKNOWN_{result.code}"
                codes[code] += 1
                latencies.append(max(0, time.time_ns() // 1_000_000 - result.observed_at_unix_ms))
                print(json.dumps(MessageToDict(result, preserving_proto_field_name=True), ensure_ascii=False))
    except grpc.FutureTimeoutError:
        print(json.dumps({"error": "connect_timeout", "target": config["target"]}))
        exit_code = 2
    except grpc.RpcError as error:
        print(json.dumps({"error": error.code().name, "message": error.details()}, ensure_ascii=False))
        exit_code = 2
    except KeyboardInterrupt:
        exit_code = 130
    finally:
        stop.set()

    elapsed = time.monotonic() - started
    ordered = sorted(latencies)

    def percentile(p: float):
        return ordered[math.ceil(len(ordered) * p) - 1] if ordered else None

    print(json.dumps({
        "summary": {
            "sent": sent, "completed": completed, "result_codes": dict(codes),
            "elapsed_seconds": round(elapsed, 3),
            "completed_fps": round(completed / elapsed, 3) if elapsed else 0,
            "latency_ms": {"p50": percentile(.5), "p95": percentile(.95), "p99": percentile(.99)},
        },
    }, ensure_ascii=False, indent=2))
    if exit_code:
        return exit_code
    return 0 if sent == completed == config["count"] and codes["RESULT_CODE_OK"] == completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
