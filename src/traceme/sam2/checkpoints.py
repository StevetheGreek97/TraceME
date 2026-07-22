from __future__ import annotations

import os
from pathlib import Path
from typing import Dict
from urllib.request import urlopen

from traceme.core.logging import get_logger

log = get_logger("traceme.sam2.checkpoints")

_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"
_CHECKPOINT_FILES: Dict[str, str] = {
    "tiny": "sam2.1_hiera_tiny.pt",
    "small": "sam2.1_hiera_small.pt",
    "base_plus": "sam2.1_hiera_base_plus.pt",
    "large": "sam2.1_hiera_large.pt",
    "sam3": "sam3.pt",
}
# sam3.pt is gated on Hugging Face (needs `hf auth login`), so it cannot be
# auto-downloaded from a public URL like the SAM2 checkpoints.
_NO_AUTO_DOWNLOAD = {"sam3"}


def _truthy_env(name: str, default: str = "1") -> bool:
    val = os.getenv(name, default)
    return val.lower() not in {"0", "false", "no", "off"}


def default_checkpoint_dir() -> Path:
    xdg = os.getenv("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "traceme" / "sam2" / "checkpoints"


def checkpoint_filename(model: str) -> str:
    if model not in _CHECKPOINT_FILES:
        raise ValueError(f"Unknown model '{model}'. Choose from: {sorted(_CHECKPOINT_FILES)}")
    return _CHECKPOINT_FILES[model]


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    log.info("Downloading SAM2 checkpoint: %s", url)
    with urlopen(url) as resp, tmp.open("wb") as f:
        total = resp.headers.get("Content-Length")
        total_bytes = int(total) if total else None
        downloaded = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total_bytes and downloaded % (50 * 1024 * 1024) < len(chunk):
                pct = (downloaded / total_bytes) * 100.0
                log.info("Download progress: %.1f%%", pct)
    tmp.replace(dst)
    log.info("Checkpoint saved: %s", dst)


def download_checkpoint(model: str, *, checkpoint_dir: Path | None = None) -> Path:
    fname = checkpoint_filename(model)
    target_dir = checkpoint_dir or default_checkpoint_dir()
    target_path = target_dir / fname
    if model in _NO_AUTO_DOWNLOAD:
        raise FileNotFoundError(
            f"{fname} cannot be auto-downloaded: the SAM3 checkpoint is gated on "
            "Hugging Face. Log in with `hf auth login`, download it from "
            f"facebook/sam3, and place it at {target_path} (or point "
            "SAM2_CHECKPOINT / SAM2_CHECKPOINT_DIR at it)."
        )
    _download(f"{_BASE_URL}/{fname}", target_path)
    return target_path


def download_all_checkpoints(*, checkpoint_dir: Path | None = None) -> Path:
    target_dir = checkpoint_dir or default_checkpoint_dir()
    for model in _CHECKPOINT_FILES:
        if model in _NO_AUTO_DOWNLOAD:
            continue
        download_checkpoint(model, checkpoint_dir=target_dir)
    return target_dir


def resolve_checkpoint(
    model: str,
    *,
    sam2_root: Path | None = None,
    checkpoint_dir: Path | None = None,
    auto_download: bool | None = None,
) -> Path:
    env_ckpt = os.getenv("SAM2_CHECKPOINT")
    if env_ckpt:
        env_path = Path(env_ckpt).expanduser()
        if env_path.exists():
            return env_path
        raise FileNotFoundError(f"SAM2_CHECKPOINT set but missing: {env_path}")

    candidates = []
    if checkpoint_dir is not None:
        candidates.append(Path(checkpoint_dir).expanduser())
    if sam2_root is not None:
        candidates.append(Path(sam2_root) / "checkpoints")
    candidates.append(default_checkpoint_dir())

    fname = checkpoint_filename(model)
    for base in candidates:
        path = base / fname
        if path.exists():
            return path

    if auto_download is None:
        auto_download = _truthy_env("TRACEME_AUTO_DOWNLOAD", "1")

    if auto_download:
        target_dir = Path(checkpoint_dir).expanduser() if checkpoint_dir else default_checkpoint_dir()
        return download_checkpoint(model, checkpoint_dir=target_dir)

    raise FileNotFoundError(
        "SAM2 checkpoint not found. Set SAM2_CHECKPOINT or SAM2_CHECKPOINT_DIR, "
        "or run traceme-download-checkpoints."
    )
