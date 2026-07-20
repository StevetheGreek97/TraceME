from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import QThread, pyqtSignal


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mpeg", ".mpg", ".wmv"}


@dataclass
class VideoImportResult:
    source_path: Path
    frames_dir: Path
    frame_count: int
    fps: Optional[float]


def _safe_stem(path: Path) -> str:
    stem = path.stem.strip()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    return stem or "video"


def _unique_dir(root: Path, base_name: str) -> Path:
    candidate = root / base_name
    if not candidate.exists():
        return candidate
    i = 1
    while True:
        cand = root / f"{base_name}_{i}"
        if not cand.exists():
            return cand
        i += 1


def _probe_fps(video: Path) -> Optional[float]:
    if shutil.which("ffprobe") is None:
        return None
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=nw=1:nk=1",
            str(video),
        ]
        out = subprocess.check_output(cmd, text=True).strip()
        if "/" in out:
            num, den = out.split("/", 1)
            return float(num) / float(den)
        return float(out)
    except Exception:
        return None


class VideoImportThread(QThread):
    progress = pyqtSignal(str, int, int)
    error = pyqtSignal(str)
    finished = pyqtSignal(list)  # list[VideoImportResult]

    def __init__(self, videos: List[Path], frames_root: Path, quality: int = 2, threads: int = 4, parent=None):
        super().__init__(parent)
        self.videos = [Path(v) for v in videos]
        self.frames_root = Path(frames_root)
        self.quality = int(quality)
        self.threads = max(1, int(threads))

    def _extract(self, video: Path, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        output_pattern = str(output_dir / "%05d.jpg")
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-i", str(video),
            "-q:v", str(self.quality),
            "-start_number", "0",
            "-threads", str(self.threads),
            output_pattern,
        ]
        subprocess.run(cmd, check=True)

    def run(self):
        results: List[VideoImportResult] = []
        total = len(self.videos)
        self.frames_root.mkdir(parents=True, exist_ok=True)

        for idx, video in enumerate(self.videos, start=1):
            try:
                base = _safe_stem(video)
                out_dir = _unique_dir(self.frames_root, base)
                self.progress.emit(f"Extracting {video.name} ({idx}/{total})", idx - 1, total)
                self._extract(video, out_dir)
                frame_count = len([p for p in out_dir.iterdir() if p.suffix.lower() in {'.jpg', '.png'}])
                fps = _probe_fps(video)
                results.append(VideoImportResult(video, out_dir, frame_count, fps))
                self.progress.emit(f"Extracted {video.name} ({idx}/{total})", idx, total)
            except Exception as e:
                self.error.emit(f"{video.name}: {e}")
                self.progress.emit(f"Failed {video.name} ({idx}/{total})", idx, total)

        self.finished.emit(results)


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None
