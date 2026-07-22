from pathlib import Path
import subprocess
import shutil

import pandas as pd

from traceme.core.logging import get_logger
from traceme.video.chunker import numeric_sort_key

try:
    import imageio_ffmpeg
except Exception:  # pragma: no cover - optional dependency
    imageio_ffmpeg = None

log = get_logger("traceme.video.merge")


def _handle_single_file(files: list[Path], output: Path, kind: str) -> bool:
    if len(files) != 1:
        return False
    src = files[0]
    if src.resolve() == output.resolve():
        log.info(f"Single {kind} already correctly named → {output.name}")
        return True
    try:
        src.rename(output)
        log.info(f"Renamed single {kind} → {output.name}")
    except Exception as e:
        log.exception(f"Failed to rename {src.name} to {output.name}: {e}")
    return True


def _resolve_ffmpeg() -> str | None:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path and imageio_ffmpeg is not None:
        try:
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            log.info(f"Using bundled ffmpeg at: {ffmpeg_path}")
        except Exception as e:
            log.exception(f"imageio_ffmpeg.get_ffmpeg_exe() failed: {e}")
            ffmpeg_path = None
    return ffmpeg_path


def merge_csv_chunks(input_dir: Path, output_csv: Path) -> None:
    """
    Merges chunk_*.csv files into one combined CSV named after frame_dir.
    If only one CSV exists, it is simply renamed to match the frame_dir name.
    """
    csv_files = sorted(input_dir.glob("*.csv"), key=numeric_sort_key)
    if not csv_files:
        log.warning(f"No chunk_*.csv files found in {input_dir}")
        return

    if _handle_single_file(csv_files, output_csv, "CSV"):
        return


    # Otherwise, merge multiple CSVs
    log.info(f"Merging {len(csv_files)} CSV files...")

    frames = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            log.exception(f"Failed to read {csv_file}: {e}")
            continue

        if "global_frame_idx" not in df.columns:
            log.error(f"{csv_file} missing 'global_frame_idx'; skipping.")
            continue
        if "obj_id" not in df.columns:
            log.error(f"{csv_file} missing 'obj_id'; skipping.")
            continue

        frames.append(df)

    if not frames:
        log.warning("No valid rows found during merge; not writing merged file.")
        return

    merged_df = pd.concat(frames, ignore_index=True)
    # Dedup per (frame, object): boundary frames are duplicated across adjacent
    # chunks, and each frame can hold multiple tracked objects. `keep="first"`
    # preserves the earlier chunk's row, matching the previous single-row
    # dedup behavior. pandas treats NaN obj_id (frames with no objects) as
    # equal to itself here, so those still dedup correctly too.
    merged_df = merged_df.drop_duplicates(subset=["global_frame_idx", "obj_id"], keep="first")
    merged_df = merged_df.sort_values(["global_frame_idx", "obj_id"]).reset_index(drop=True)

    # A blank obj_id on any row forces that whole column to float64 on read;
    # restore clean integer formatting instead of "500.0" everywhere.
    for col in ("chunk_id", "global_frame_idx", "in_chunk_idx", "area_px"):
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].astype("int64")
    if "obj_id" in merged_df.columns:
        merged_df["obj_id"] = merged_df["obj_id"].astype("Int64")

    merged_df.to_csv(output_csv, index=False)

    log.info(
        f"[OK] Merged {len(csv_files)} chunk files → {output_csv.name} "
        f"({len(merged_df)} rows, {merged_df['global_frame_idx'].nunique()} unique frames)"
    )


def merge_chunk_videos(input_dir: Path, output_file: Path) -> None:
    """
    Concatenate all *.mp4 files in `input_dir` into `output_file`.
    Uses system ffmpeg if available; otherwise tries imageio-ffmpeg.
    If only one MP4 exists, it is renamed to `output_file`.
    """
    mp4_files = sorted(input_dir.glob("*.mp4"), key=numeric_sort_key)
    if not mp4_files:
        log.warning(f"No .mp4 files found in {input_dir}")
        return

    if _handle_single_file(mp4_files, output_file, "MP4"):
        return

    # Locate ffmpeg
    ffmpeg_path = _resolve_ffmpeg()

    if not ffmpeg_path:
        log.error(
            "No ffmpeg found in PATH and imageio-ffmpeg not available. "
            "Install with: pip install 'imageio[ffmpeg]'. Skipping merge."
        )
        return

    # Prepare concat list beside the output file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    concat_list = output_file.with_suffix(".concat.txt")

    def _ffconcat_line(p: Path) -> str:
        # Use absolute POSIX paths and escape single quotes for ffmpeg concat demuxer
        s = p.resolve().as_posix().replace("'", r"'\''")
        return f"file '{s}'\n"

    try:
        with concat_list.open("w", encoding="utf-8") as f:
            for p in mp4_files:
                f.write(_ffconcat_line(p))

        cmd = [
            str(ffmpeg_path),
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            str(output_file),
        ]

        log.info(f"ffmpeg concat → {output_file.name}")
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        log.info(f"[OK] merged video written: {output_file.name}")

    except subprocess.CalledProcessError as e:
        err = e.stderr.decode(errors="ignore") if e.stderr else str(e)
        log.exception(f"ffmpeg merge failed: {err}")
    finally:
        # Remove the temporary concat list
        try:
            concat_list.unlink(missing_ok=True)
        except Exception:
            pass
