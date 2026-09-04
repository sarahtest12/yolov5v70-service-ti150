#!/usr/bin/env bash

set -euo pipefail

GPU_SERVICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COREX_HOME_FOR_TEST="${COREX_HOME:-/usr/local/corex}"

environment_output="$({
  env -i \
    PATH=/usr/bin:/bin \
    COREX_HOME="$COREX_HOME_FOR_TEST" \
    bash --noprofile --norc -c '
      set -euo pipefail
      cd "$1"
      source scripts/corex_env.sh
      python_path="$(command -v python || true)"
      if [[ -z "$python_path" ]]; then
        printf "%s\n" "python was not found after sourcing scripts/corex_env.sh" >&2
        exit 21
      fi
      printf "%s\n" "$python_path"
      python -c "import sys; print(sys.prefix)"
    ' bash "$GPU_SERVICE_ROOT"
} 2>&1)" || {
  printf 'FAIL: %s\n' "$environment_output" >&2
  exit 1
}

mapfile -t environment_lines <<< "$environment_output"
actual_python="$(readlink -f "${environment_lines[0]}")"
expected_python="$(readlink -f "$GPU_SERVICE_ROOT/.venv/bin/python")"
actual_prefix="${environment_lines[1]:-}"
expected_prefix="$GPU_SERVICE_ROOT/.venv"

if [[ "$actual_python" != "$expected_python" ]]; then
  printf 'FAIL: python resolved to %s, expected %s\n' "$actual_python" "$expected_python" >&2
  exit 1
fi

if [[ "$actual_prefix" != "$expected_prefix" ]]; then
  printf 'FAIL: sys.prefix is %s, expected %s\n' "$actual_prefix" "$expected_prefix" >&2
  exit 1
fi

printf '%s\n' 'CoreX environment relocation check: PASS'
