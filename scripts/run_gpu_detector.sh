#!/usr/bin/env bash

set -euo pipefail

GPU_SERVICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$GPU_SERVICE_ROOT"

# shellcheck disable=SC1091
source scripts/corex_env.sh
exec python -m gpu_detector.main
