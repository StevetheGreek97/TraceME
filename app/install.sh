#!/usr/bin/env bash
set -euo pipefail

VENV="${VENV:-.venv}"
PYTHON="${PYTHON:-}"
PY_LAUNCHER_ARGS=()
ENV_MODE="auto" # auto|venv|conda|system
INSTALL_FULL=1
DOWNLOAD_WEIGHTS=1

SAM2_URL="https://huggingface.co/facebook/sam2-hiera-tiny/resolve/main/sam2_hiera_tiny.pt"
SAM2_WEIGHTS="src/sam2_configs/sam2_hiera_tiny.pt"

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

resolve_conda_cmd() {
  if [[ -n "${CONDA_EXE:-}" && -f "${CONDA_EXE}" ]]; then
    echo "$CONDA_EXE"
    return 0
  fi
  if command -v conda >/dev/null 2>&1; then
    echo "conda"
    return 0
  fi
  return 1
}

usage() {
  cat <<EOF
Usage: ./install.sh [options]

Options:
  --venv <path>     Virtual env directory (default: .venv)
  --no-venv         Use system/active Python (no venv)
  --conda           Use the active conda env (no venv)
  --python <path>   Python executable (default: python3)
  -h, --help        Show help

Environment:
  VENV, PYTHON can also be set via env vars.
  If a conda env is active and no mode is specified, conda is used.
  Note: sam2 is not available on conda channels; pip is used even in conda envs.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      ENV_MODE="venv"
      VENV="$2"
      shift
      ;;
    --no-venv)
      ENV_MODE="system"
      ;;
    --conda)
      ENV_MODE="conda"
      ;;
    --python)
      PYTHON="$2"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
  shift
done

if [[ "$ENV_MODE" == "auto" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    ENV_MODE="conda"
  else
    ENV_MODE="venv"
  fi
fi

if [[ "$ENV_MODE" == "conda" && -z "${CONDA_PREFIX:-}" ]]; then
  echo "Error: --conda requires an active conda env."
  echo "Example: conda create -n tracewave python=3.10 && conda activate tracewave"
  exit 1
fi

if [[ -z "$PYTHON" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
  elif [[ "$OS_FAMILY" == "windows" ]] && command -v py >/dev/null 2>&1; then
    PYTHON="py"
    PY_LAUNCHER_ARGS=(-3)
  else
    echo "Error: Python not found. Install Python 3.9+ and ensure it is on PATH."
    exit 1
  fi
fi

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Error: Python not found: $PYTHON"
  exit 1
fi

install_ffmpeg() {
  if command -v ffmpeg >/dev/null 2>&1 && command -v ffprobe >/dev/null 2>&1; then
    echo "FFmpeg and FFprobe already installed."
    return 0
  fi

  echo "FFmpeg/FFprobe not found. Installing..."

  if [[ "$ENV_MODE" != "venv" && -n "${CONDA_PREFIX:-}" ]]; then
    local conda_cmd=""
    if conda_cmd="$(resolve_conda_cmd)"; then
      "$conda_cmd" install -y -c conda-forge ffmpeg
      return 0
    fi
  fi

  if [[ "$OS_FAMILY" == "windows" ]]; then
    if command -v winget >/dev/null 2>&1; then
      winget install --id Gyan.FFmpeg -e
      return 0
    fi
    if command -v choco >/dev/null 2>&1; then
      choco install -y ffmpeg
      return 0
    fi
    echo "Error: FFmpeg not found and no supported installer (winget/choco) available."
    echo "Please install FFmpeg manually and ensure ffmpeg/ffprobe are on PATH."
    exit 1
  fi

  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y ffmpeg
    return 0
  fi

  if command -v brew >/dev/null 2>&1; then
    brew install ffmpeg
    return 0
  fi

  if command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y ffmpeg
    return 0
  fi

  if command -v yum >/dev/null 2>&1; then
    sudo yum install -y ffmpeg
    return 0
  fi

  if command -v pacman >/dev/null 2>&1; then
    sudo pacman -Sy --noconfirm ffmpeg
    return 0
  fi

  echo "Error: Unsupported package manager. Please install ffmpeg and ffprobe manually."
  echo "See: https://ffmpeg.org/download.html"
  exit 1
}

install_ffmpeg

BASE_PY_CMD=("$PYTHON" "${PY_LAUNCHER_ARGS[@]}")

if [[ "$ENV_MODE" == "venv" ]]; then
  if [[ ! -d "$VENV" ]]; then
    "${BASE_PY_CMD[@]}" -m venv "$VENV"
  fi
  PY_CMD=("$(venv_python_path)")
else
  PY_CMD=("${BASE_PY_CMD[@]}")
fi

PIP_CMD=("${PY_CMD[@]}" -m pip)

"${PIP_CMD[@]}" install --upgrade pip

if [[ "$INSTALL_FULL" -eq 1 ]]; then
  "${PIP_CMD[@]}" install -e ".[sam2,yaml]"
else
  "${PIP_CMD[@]}" install -e .
fi

if [[ "$DOWNLOAD_WEIGHTS" -eq 1 ]]; then
  mkdir -p "$(dirname "$SAM2_WEIGHTS")"
  if [[ -f "$SAM2_WEIGHTS" ]]; then
    echo "SAM2 weights already present at $SAM2_WEIGHTS"
  else
    echo "Downloading SAM2 weights to $SAM2_WEIGHTS"
    if command -v curl >/dev/null 2>&1; then
      curl -L -o "$SAM2_WEIGHTS" "$SAM2_URL"
    elif command -v wget >/dev/null 2>&1; then
      wget -O "$SAM2_WEIGHTS" "$SAM2_URL"
    elif [[ "$OS_FAMILY" == "windows" ]] && command -v powershell.exe >/dev/null 2>&1; then
      powershell.exe -NoProfile -Command "Invoke-WebRequest -Uri '$SAM2_URL' -OutFile '$SAM2_WEIGHTS'"
    else
      SAM2_URL="$SAM2_URL" SAM2_WEIGHTS="$SAM2_WEIGHTS" "${PY_CMD[@]}" - <<'PY'
import os
import urllib.request

url = os.environ["SAM2_URL"]
dst = os.environ["SAM2_WEIGHTS"]
os.makedirs(os.path.dirname(dst), exist_ok=True)
urllib.request.urlretrieve(url, dst)
PY
    fi
  fi
fi

echo "Done."
echo "Environment: $ENV_MODE"
if [[ "$ENV_MODE" == "venv" ]]; then
  echo "Activate with:"
  echo "  Linux/macOS: source \"$VENV/bin/activate\""
  echo "  Windows (PowerShell): .\\$VENV\\Scripts\\Activate.ps1"
elif [[ "$ENV_MODE" == "conda" ]]; then
  echo "Activate with: conda activate ${CONDA_DEFAULT_ENV:-<env>}"
else
  echo "Using system/active Python: $PYTHON"
fi
echo "Run with: ${PY_CMD[*]} -m src.tracewave"
