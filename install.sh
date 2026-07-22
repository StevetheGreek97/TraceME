#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# SAM2 install controls
INSTALL_SAM2="${INSTALL_SAM2:-1}"
SAM2_REPO_URL="${SAM2_REPO_URL:-https://github.com/facebookresearch/sam2.git}"
SAM2_ROOT="${SAM2_ROOT:-third_party/sam2}"
SAM2_REPO_REF="${SAM2_REPO_REF:-}"
SAM2_DOWNLOAD_CHECKPOINTS="${SAM2_DOWNLOAD_CHECKPOINTS:-1}"
SAM2_INSTALL_NOTEBOOKS="${SAM2_INSTALL_NOTEBOOKS:-0}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: $PYTHON_BIN not found. Set PYTHON_BIN or install Python 3." >&2
  exit 1
fi

echo "==> Creating virtual environment: $VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"

echo "==> Activating virtual environment"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

echo "==> Installing package + dependencies"
pip install -e .

if [[ "$INSTALL_SAM2" == "1" ]]; then
  if ! command -v git >/dev/null 2>&1; then
    echo "ERROR: git not found. Install git or set INSTALL_SAM2=0 to skip SAM2." >&2
    exit 1
  fi

  if [[ ! -d "$SAM2_ROOT" ]]; then
    echo "==> Cloning SAM2 into: $SAM2_ROOT"
    mkdir -p "$(dirname "$SAM2_ROOT")"
    git clone "$SAM2_REPO_URL" "$SAM2_ROOT"
    if [[ -n "$SAM2_REPO_REF" ]]; then
      echo "==> Checking out SAM2 ref: $SAM2_REPO_REF"
      (cd "$SAM2_ROOT" && git checkout "$SAM2_REPO_REF")
    fi
  fi

  if [[ "$SAM2_INSTALL_NOTEBOOKS" == "1" ]]; then
    echo "==> Installing SAM2 (with notebooks extras) from: $SAM2_ROOT"
    pip install -e "${SAM2_ROOT}[notebooks]"
  else
    echo "==> Installing SAM2 from: $SAM2_ROOT"
    pip install -e "$SAM2_ROOT"
  fi

  if [[ "$SAM2_DOWNLOAD_CHECKPOINTS" == "1" ]]; then
    if [[ -f "$SAM2_ROOT/checkpoints/download_ckpts.sh" ]]; then
      echo "==> Downloading SAM2 checkpoints"
      (cd "$SAM2_ROOT/checkpoints" && bash ./download_ckpts.sh)
    else
      echo "WARN: SAM2 checkpoint script not found at: $SAM2_ROOT/checkpoints/download_ckpts.sh"
      echo "      Check out the SAM2 repo or set SAM2_DOWNLOAD_CHECKPOINTS=0 to skip."
    fi
  else
    echo "NOTE: SAM2 checkpoints not downloaded (SAM2_DOWNLOAD_CHECKPOINTS=0)."
  fi
else
  echo "NOTE: SAM2 install skipped (INSTALL_SAM2=0)."
fi

echo "==> Done."
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Run with: traceme -i /path/to/frames -o /path/to/output -p /path/to/prompts.yaml"
