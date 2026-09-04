#!/usr/bin/env bash

set -euo pipefail

GPU_SERVICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$GPU_SERVICE_ROOT"

if [[ ! -d /usr/local/corex/lib64/python3/dist-packages ]]; then
  printf '%s\n' 'CoreX SDK was not found at /usr/local/corex.' >&2
  exit 1
fi

if [[ -e .venv ]] && ! python3 scripts/check_venv_location.py --quiet; then
  GPU_STALE_VENV="$GPU_SERVICE_ROOT/.venv.stale-$(date -u +%Y%m%dT%H%M%SZ)-$$"
  mv "$GPU_SERVICE_ROOT/.venv" "$GPU_STALE_VENV"
  printf 'Preserved stale virtual environment at %s\n' "$GPU_STALE_VENV"
fi

if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source scripts/corex_env.sh

python -m pip install --upgrade pip
python -m pip install -r requirements-corex.txt
python -m pip install --no-deps --editable .
scripts/generate_detector_proto.sh
python scripts/check_venv_location.py
python scripts/check_corex_env.py
