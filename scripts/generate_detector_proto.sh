#!/usr/bin/env bash

set -euo pipefail

GPU_SERVICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$GPU_SERVICE_ROOT"

python -m grpc_tools.protoc \
  -Isrc \
  --python_out=src \
  --grpc_python_out=src \
  src/detector_contract/detector.proto

python -m py_compile \
  src/detector_contract/detector_pb2.py \
  src/detector_contract/detector_pb2_grpc.py
