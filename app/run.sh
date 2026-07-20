#!/usr/bin/env bash
set -euo pipefail

VENV="${VENV:-.venv}"
PYTHON="${PYTHON:-}"
PY_LAUNCHER_ARGS=()
USE_VENV="${USE_VENV:-auto}" # auto|yes|no

OS="$(uname -s 2>/dev/null || echo unknown)"
OS_FAMILY="linux"
case "$OS" in
  Darwin*) OS_FAMILY="mac";;
  Linux*) OS_FAMILY="linux";;
  MINGW*|MSYS*|CYGWIN*) OS_FAMILY="windows";;
  *) OS_FAMILY="linux";;
esac

venv_python_path() {
  if [[ "$OS_FAMILY" == "windows" ]]; then
    echo "$VENV/Scripts/python.exe"
  else
    echo "$VENV/bin/python"
  fi
}

conda_python_path() {
  if [[ "$OS_FAMILY" == "windows" ]]; then
    echo "$CONDA_PREFIX/python.exe"
  else
    echo "$CONDA_PREFIX/bin/python"
  fi
}

python_exists() {
  local p="$1"
  if [[ "$OS_FAMILY" == "windows" ]]; then
    [[ -f "$p" ]]
  else
    [[ -x "$p" ]]
  fi
}

if [[ -z "$PYTHON" ]]; then
  if [[ "$USE_VENV" != "no" ]]; then
    PYTHON="$(venv_python_path)"
  fi
fi

if ! python_exists "$PYTHON"; then
  if [[ -n "${CONDA_PREFIX:-}" ]] && [[ "$USE_VENV" != "yes" ]]; then
    PYTHON="$(conda_python_path)"
  fi
fi

if ! python_exists "$PYTHON"; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
  elif [[ "$OS_FAMILY" == "windows" ]] && command -v py >/dev/null 2>&1; then
    PYTHON="py"
    PY_LAUNCHER_ARGS=(-3)
  fi
fi

if [[ -z "$PYTHON" ]] || ([[ "$PYTHON" != "py" ]] && ! python_exists "$PYTHON"); then
  echo "Error: Python not found at $PYTHON"
  echo "Tip: run ./install.sh first or set PYTHON to your interpreter."
  echo "Hint: if you are using conda, activate your env first."
  exit 1
fi

exec "$PYTHON" "${PY_LAUNCHER_ARGS[@]}" -m src.tracewave
