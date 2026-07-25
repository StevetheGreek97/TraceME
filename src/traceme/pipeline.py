from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Literal, Optional, TypeVar
import argparse
import json
import logging
import shutil

import os

from traceme.core.logging import add_file_handler, get_logger, set_log_context, timer
from traceme.video.merge import merge_csv_chunks, merge_chunk_videos, merge_mask_chunks
from traceme.prompts.parser import YamlPromptParser
from traceme.video.chunker import VideoChunker

ChunkMode = Literal["auto", "load", "force"]
MODEL_CHOICES = ("tiny", "small", "base_plus", "large", "sam3")
T = TypeVar("T")


@dataclass(frozen=True)
class PipelineConfig:
    frame_dir: Path
    output_folder: Path
    prompt_file: Path
    chunk_size: int = 500
    overlap: int = 1
    fps: int = 60
    del_tmp: bool = False
    chunk_mode: ChunkMode = "auto"
    model: str | None = None
    resume: bool = True
    save_masks: bool = False


@dataclass(frozen=True)
class PipelinePaths:
    output_root: Path
    tmp_root: Path
    chunks_dir: Path
    files_dir: Path
    log_file: Path
    merged_csv: Path
    merged_video: Path
    merged_masks: Path
    run_summary: Path

    @classmethod
    def from_config(cls, cfg: PipelineConfig) -> "PipelinePaths":
        tmp_root = cfg.output_folder / f"{cfg.frame_dir.name}_tmp"
        return cls(
            output_root=cfg.output_folder,
            tmp_root=tmp_root,
            chunks_dir=tmp_root / "chunks",
            files_dir=tmp_root / "files",
            log_file=tmp_root / f"{cfg.frame_dir.name}_run.log",
            merged_csv=cfg.output_folder / f"{cfg.frame_dir.name}.csv",
            merged_video=cfg.output_folder / f"{cfg.frame_dir.name}.mp4",
            merged_masks=cfg.output_folder / f"{cfg.frame_dir.name}_masks.npz",
            run_summary=cfg.output_folder / f"{cfg.frame_dir.name}_run_summary.json",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full video chunking and processing pipeline.")
    parser.add_argument(
        "-i", "--frame_dir", required=True, type=Path,
        help="Path to the input FRAMES directory."
    )
    parser.add_argument(
        "-o", "--output_folder", required=True, type=Path,
        help="Path to the output directory."
    )
    parser.add_argument(
        "-p", "--prompt_file", required=True, type=Path,
        help="YAML prompt file path."
    )
    parser.add_argument(
        "-c", "--chunk_size", type=int, default=500,
        help="Number of frames per chunk (default: 500)."
    )
    parser.add_argument(
        "--overlap", type=int, default=1,
        help="Number of overlapping frames between chunks (default: 1)."
    )
    parser.add_argument(
        "--fps", type=int, default=60,
        help="FPS for output video (default: 60)."
    )
    parser.add_argument(
        "--chunk-mode",
        choices=("auto", "load", "force"),
        default="auto",
        help="Chunking mode (default: auto)."
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default=None,
        help="Model to use: a SAM2 size (tiny/small/base_plus/large) or 'sam3' "
             "(overrides SAM2_MODEL env var; sam3 requires the sam3 extra)."
    )
    parser.add_argument(
        "-d", "--del_tmp", action="store_true",
        help="If set, deletes temporary frame/chunk folders after processing."
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Reprocess all chunks even if completed outputs from a previous run exist."
    )
    parser.add_argument(
        "--save-masks", action="store_true",
        help="Persist per-frame object masks as a compressed .npz per chunk "
             "(<chunk>_masks.npz, bit-packed) for downstream shape analysis."
    )
    return parser


def parse_args(argv: Optional[Iterable[str]] = None) -> PipelineConfig:
    args = build_parser().parse_args(argv)
    return PipelineConfig(
        frame_dir=args.frame_dir,
        output_folder=args.output_folder,
        prompt_file=args.prompt_file,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        fps=args.fps,
        del_tmp=args.del_tmp,
        chunk_mode=args.chunk_mode,
        model=args.model,
        resume=not args.no_resume,
        save_masks=args.save_masks,
    )


def _attach_file_logger(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    add_file_handler(log_file)


def _validate_inputs(cfg: PipelineConfig, log: logging.Logger) -> None:
    if not cfg.frame_dir.exists() or not cfg.frame_dir.is_dir():
        log.error(f"frame_dir does not exist or is not a directory: {cfg.frame_dir}")
        raise SystemExit(2)
    if not cfg.prompt_file.exists():
        log.error(f"prompt_file not found: {cfg.prompt_file}")
        raise SystemExit(2)


def _log_config(cfg: PipelineConfig, log: logging.Logger) -> None:
    log.info(
        "Pipeline config: frame_dir=%s output=%s prompt=%s chunk_size=%s overlap=%s fps=%s "
        "chunk_mode=%s model=%s del_tmp=%s",
        cfg.frame_dir,
        cfg.output_folder,
        cfg.prompt_file,
        cfg.chunk_size,
        cfg.overlap,
        cfg.fps,
        cfg.chunk_mode,
        cfg.model or os.environ.get("SAM2_MODEL", "<env>"),
        cfg.del_tmp,
    )


def _run_step(log: logging.Logger, label: str, fn: Callable[[], T], *, fatal: bool = True) -> T | None:
    with timer(log, label):
        try:
            return fn()
        except Exception as e:
            log.exception(f"{label} failed: {e}")
            if fatal:
                raise
    return None


def run_pipeline(cfg: PipelineConfig) -> None:
    log = get_logger("traceme.pipeline")
    paths = PipelinePaths.from_config(cfg)

    paths.output_root.mkdir(parents=True, exist_ok=True)
    _attach_file_logger(paths.log_file)
    set_log_context(
        run_id=cfg.frame_dir.name,
        job_id=os.getenv("JOBBER_TASK_ID") or os.getenv("SLURM_PROCID") or os.getenv("SLURM_JOB_ID"),
    )
    _validate_inputs(cfg, log)
    _log_config(cfg, log)

    if cfg.model:
        os.environ["SAM2_MODEL"] = cfg.model

    try:
        from traceme.sam2 import config as sam2_config
    except Exception as e:
        log.error("Failed to import SAM2 config. Install SAM2 and its dependencies. Error: %s", e)
        raise

    sam2_config.log_runtime_summary(log)
    sam2_config.log_model_summary(log)
    sam2_config.log_precision_summary(log)

    chunker = VideoChunker(
        frame_dir=cfg.frame_dir,
        output_dir=paths.chunks_dir,
        chunk_size=cfg.chunk_size,
        overlap=cfg.overlap,
        action="symlink",
        remove_org=False,
    )

    _run_step(
        log,
        "Chunking frames",
        lambda: chunker.chunk_frames(mode=cfg.chunk_mode),
    )
    log.info(
        f"Chunks prepared: {chunker.count_chunks()} | "
        f"chunk_size={cfg.chunk_size} | overlap={cfg.overlap}"
    )

    prompts = _run_step(
        log,
        "Loading prompts",
        lambda: YamlPromptParser(cfg.prompt_file).load(strict_keys=False),
    ) or []
    by_chunk = YamlPromptParser.prompts_by_chunk(
        prompts,
        chunk_size=cfg.chunk_size,
        overlap=cfg.overlap,
        total_frames=len(chunker.get_all_frames_flat()) or None,
    )
    log.info(
        f"Prompts loaded: {sum(len(v) for v in by_chunk.values())} total "
        f"across {len(by_chunk)} chunks"
    )

    from traceme.sam2.runner import run_sam2

    summary = _run_step(
        log,
        "SAM2 run",
        lambda: run_sam2(
            chunker=chunker,
            by_chunk=by_chunk,
            output=paths.files_dir,
            video_fps=cfg.fps,
            prepare_chunks=False,
            resume=cfg.resume,
            save_masks=cfg.save_masks,
        ),
        fatal=True,
    ) or {}
    failed_chunks = summary.get("failed_chunks", [])

    _run_step(
        log,
        "Merging chunk CSVs",
        lambda: merge_csv_chunks(paths.files_dir, paths.merged_csv),
        fatal=False,
    )
    _run_step(
        log,
        "Merging chunk videos",
        lambda: merge_chunk_videos(paths.files_dir, paths.merged_video),
        fatal=False,
    )
    if cfg.save_masks:
        _run_step(
            log,
            "Merging chunk masks",
            lambda: merge_mask_chunks(paths.files_dir, paths.merged_masks),
            fatal=False,
        )

    _run_step(
        log,
        "Writing run summary",
        lambda: _write_run_summary(paths.run_summary, cfg, summary),
        fatal=False,
    )

    if cfg.del_tmp:
        if failed_chunks:
            log.warning(
                "Keeping temporary files despite --del_tmp: %d chunk(s) failed and "
                "the tmp folder is needed to resume.",
                len(failed_chunks),
            )
        else:
            def _cleanup() -> None:
                shutil.rmtree(paths.tmp_root, ignore_errors=True)
                log.info("Temporary files deleted.")

            _run_step(log, "Cleanup temporary files", _cleanup, fatal=False)

    if failed_chunks:
        log.error(
            "Pipeline finished with %d failed chunk(s): %s. Merged outputs are PARTIAL. "
            "Re-run the same command to retry only the failed chunks.",
            len(failed_chunks),
            failed_chunks,
        )
        raise SystemExit(1)

    log.info("Pipeline finished successfully.")


def _write_run_summary(path: Path, cfg: PipelineConfig, summary: dict) -> None:
    failed = summary.get("failed_chunks", [])
    payload = {
        "status": "partial" if failed else "complete",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        **summary,
        "config": {
            "frame_dir": str(cfg.frame_dir),
            "output_folder": str(cfg.output_folder),
            "prompt_file": str(cfg.prompt_file),
            "chunk_size": cfg.chunk_size,
            "overlap": cfg.overlap,
            "fps": cfg.fps,
            "chunk_mode": cfg.chunk_mode,
            "model": cfg.model or os.environ.get("SAM2_MODEL", "large"),
            "resume": cfg.resume,
            "save_masks": cfg.save_masks,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def cli_main(argv: Optional[Iterable[str]] = None) -> None:
    run_pipeline(parse_args(argv))
