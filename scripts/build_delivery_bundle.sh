#!/usr/bin/env bash

set -euo pipefail

GPU_SERVICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_SERVICE_PARENT="$(dirname "$GPU_SERVICE_ROOT")"
GPU_SERVICE_NAME="$(basename "$GPU_SERVICE_ROOT")"
GPU_SERVICE_RELEASE_NAME="gpu-grpc-service"
GPU_SERVICE_VERSION="$(tr -d '[:space:]' < "$GPU_SERVICE_ROOT/VERSION")"
GPU_SERVICE_DIST="$GPU_SERVICE_ROOT/dist"
GPU_SERVICE_ARCHIVE="$GPU_SERVICE_DIST/${GPU_SERVICE_RELEASE_NAME}-${GPU_SERVICE_VERSION}.tar.gz"

for required_file in models/yolov5s.pt config/coco80.yaml LICENSE THIRD_PARTY_NOTICES.md \
  shared/pyproject.toml shared/detector_contract/__init__.py shared/detector_contract/config.py \
  shared/detector_contract/detector.proto shared/detector_contract/detector_pb2.py \
  shared/detector_contract/detector_pb2_grpc.py; do
  if [[ ! -f "$GPU_SERVICE_ROOT/$required_file" ]]; then
    printf 'Required delivery file is missing: %s\n' "$GPU_SERVICE_ROOT/$required_file" >&2
    exit 1
  fi
done

mkdir -p "$GPU_SERVICE_DIST"
tar \
  --exclude="${GPU_SERVICE_NAME}/.git" \
  --exclude="${GPU_SERVICE_NAME}/.venv" \
  --exclude="${GPU_SERVICE_NAME}/.venv.stale-*" \
  --exclude="${GPU_SERVICE_NAME}/cpu_client/.venv" \
  --exclude="${GPU_SERVICE_NAME}/cpu_client/config.json" \
  --exclude="${GPU_SERVICE_NAME}/dist" \
  --exclude='*.egg-info' \
  --exclude="${GPU_SERVICE_NAME}/shared/build" \
  --exclude="${GPU_SERVICE_NAME}/shared/dist" \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  -C "$GPU_SERVICE_PARENT" \
  --transform="s,^${GPU_SERVICE_NAME},${GPU_SERVICE_RELEASE_NAME}," \
  -czf "$GPU_SERVICE_ARCHIVE" \
  "$GPU_SERVICE_NAME"

(
  cd "$GPU_SERVICE_DIST"
  sha256sum "$(basename "$GPU_SERVICE_ARCHIVE")" > "$(basename "$GPU_SERVICE_ARCHIVE").sha256"
)
printf 'Created %s\n' "$GPU_SERVICE_ARCHIVE"
printf 'Created %s\n' "$GPU_SERVICE_ARCHIVE.sha256"
