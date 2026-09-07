#!/usr/bin/env bash

# Load the standalone service venv together with the host-provided CoreX SDK.
# Usage: source scripts/corex_env.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf '%s\n' 'This script must be sourced: source scripts/corex_env.sh' >&2
  exit 2
fi

COREX_HOME="${COREX_HOME:-/usr/local/corex}"
GPU_SERVICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_SERVICE_VENV_DIR="${GPU_SERVICE_VENV_DIR:-$GPU_SERVICE_ROOT/.venv}"

if [[ ! -d "$COREX_HOME/lib64/python3/dist-packages" ]]; then
  printf 'CoreX Python packages not found under %s\n' "$COREX_HOME" >&2
  return 1
fi

if [[ ! -x "$GPU_SERVICE_VENV_DIR/bin/python" ]]; then
  printf 'Python venv not found at %s\n' "$GPU_SERVICE_VENV_DIR" >&2
  printf '%s\n' 'Run scripts/bootstrap_corex.sh first, or set GPU_SERVICE_VENV_DIR.' >&2
  return 1
fi

if ! python3 "$GPU_SERVICE_ROOT/scripts/check_venv_location.py" --venv "$GPU_SERVICE_VENV_DIR" --quiet; then
  printf 'Python venv at %s belongs to a different checkout or is incomplete.\n' "$GPU_SERVICE_VENV_DIR" >&2
  printf '%s\n' 'Run scripts/bootstrap_corex.sh to preserve it and create a valid local environment.' >&2
  return 1
fi

_corex_prepend_path() {
  local variable_name="$1"
  local directory="$2"
  local current_value="${!variable_name:-}"

  case ":$current_value:" in
    *":$directory:"*) ;;
    *) printf -v "$variable_name" '%s%s%s' "$directory" "${current_value:+:}" "$current_value" ;;
  esac
  export "$variable_name"
}

_corex_prepend_path PATH /usr/local/openmpi/bin
_corex_prepend_path PATH "$COREX_HOME/lib64/python3/dist-packages/bin"
_corex_prepend_path PATH "$COREX_HOME/bin"
_corex_prepend_path LD_LIBRARY_PATH /usr/local/lib
_corex_prepend_path LD_LIBRARY_PATH /usr/local/openmpi/lib
_corex_prepend_path LD_LIBRARY_PATH "$COREX_HOME/lib64"
_corex_prepend_path PYTHONPATH "$COREX_HOME/lib64/python3/dist-packages"
_corex_prepend_path PYTHONPATH "$GPU_SERVICE_ROOT/src"
_corex_prepend_path PYTHONPATH "$GPU_SERVICE_ROOT/shared"

export COREX_HOME
export GPU_SERVICE_ROOT
export GPU_SERVICE_VENV_DIR
export YOLOv5_AUTOINSTALL=false

# shellcheck disable=SC1090
source "$GPU_SERVICE_VENV_DIR/bin/activate"

unset -f _corex_prepend_path
