#!/usr/bin/env bash

set -euo pipefail

GPU_SERVICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$GPU_SERVICE_ROOT"

python -m grpc_tools.protoc \
  -Ishared \
  --python_out=shared \
  --grpc_python_out=shared \
  shared/detector_contract/detector.proto

python -m py_compile \
  shared/detector_contract/detector_pb2.py \
  shared/detector_contract/detector_pb2_grpc.py
