#!/usr/bin/env python3
"""Run one YOLOv5 image through the CoreX GPU and print detections as JSON."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVICE_SRC = PROJECT_ROOT / "src"
if str(SERVICE_SRC) not in sys.path:
    sys.path.insert(0, str(SERVICE_SRC))

from models.common import DetectMultiBackend  # noqa: E402
from utils.dataloaders import letterbox  # noqa: E402
from utils.general import check_img_size, non_max_suppression, scale_boxes  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "models/yolov5s.pt")
    parser.add_argument("--image", type=Path, default=PROJECT_ROOT / "tests/fixtures/bus.jpg")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "config/coco80.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--expect-min-detections", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(f"weights not found: {args.weights}")
    if not args.image.is_file():
        raise FileNotFoundError(f"image not found: {args.image}")

    device = select_device(args.device)
    model = DetectMultiBackend(args.weights, device=device, data=args.data, fp16=args.half)
    image_size = check_img_size(args.image_size, s=model.stride)

    original = cv2.imread(str(args.image))
    if original is None:
        raise ValueError(f"OpenCV could not decode: {args.image}")

    resized = letterbox(original, image_size, stride=int(model.stride), auto=model.pt)[0]
    rgb_chw = resized.transpose((2, 0, 1))[::-1]
    tensor = torch.from_numpy(np.ascontiguousarray(rgb_chw)).to(model.device)
    tensor = tensor.half() if model.fp16 else tensor.float()
    tensor /= 255
    tensor = tensor.unsqueeze(0)

    model.warmup(imgsz=tuple(tensor.shape))
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    prediction = model(tensor)
    torch.cuda.synchronize(device)
    inference_ms = (time.perf_counter() - started) * 1000

    prediction = non_max_suppression(prediction, args.confidence, args.iou)
    detections = prediction[0]
    results: list[dict[str, object]] = []
    if len(detections):
        detections[:, :4] = scale_boxes(tensor.shape[2:], detections[:, :4], original.shape).round()
        for *xyxy, confidence, class_id in detections.cpu().tolist():
            numeric_class_id = int(class_id)
            results.append({
                "class_id": numeric_class_id,
                "label": model.names[numeric_class_id],
                "confidence": round(float(confidence), 6),
                "box_xyxy": [int(value) for value in xyxy],
            })

    output = {
        "device": torch.cuda.get_device_name(device),
        "dtype": "float16" if model.fp16 else "float32",
        "image": str(args.image),
        "image_width": original.shape[1],
        "image_height": original.shape[0],
        "inference_ms": round(inference_ms, 3),
        "detection_count": len(results),
        "detections": results,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))

    if len(results) < args.expect_min_detections:
        print(
            f"Expected at least {args.expect_min_detections} detection(s), found {len(results)}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
