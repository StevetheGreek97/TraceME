from pathlib import Path
import subprocess
import shutil

import pandas as pd

from tracewave.core.logging import get_logger

try:
    import imageio_ffmpeg
except Exception:  # pragma: no cover - optional dependency
    imageio_ffmpeg = None

log = get_logger("tracewave.video.merge")


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
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        log.warning(f"No chunk_*.csv files found in {input_dir}")
        return

    if _handle_single_file(csv_files, output_csv, "CSV"):
        return


    # Otherwise, merge multiple CSVs
    log.info(f"Merging {len(csv_files)} CSV files...")

    seen_frames = set()
    merged_rows = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            log.exception(f"Failed to read {csv_file}: {e}")
            continue

        if "global_frame_idx" not in df.columns:
            log.error(f"{csv_file} missing 'global_frame_idx'; skipping.")
            continue

        for _, row in df.iterrows():
            try:
                gidx = int(row["global_frame_idx"])
            except Exception:
                continue
            if gidx not in seen_frames:
                merged_rows.append(row)
                seen_frames.add(gidx)

    if not merged_rows:
        log.warning("No valid rows found during merge; not writing merged file.")
        return

    merged_df = pd.DataFrame(merged_rows)
    merged_df = merged_df.sort_values("global_frame_idx").reset_index(drop=True)
    merged_df.to_csv(output_csv, index=False)

    log.info(f"[OK] Merged {len(csv_files)} chunk files → {output_csv.name} ({len(merged_df)} unique frames)")


def merge_chunk_videos(input_dir: Path, output_file: Path) -> None:
    """
    Concatenate all *.mp4 files in `input_dir` into `output_file`.
    Uses system ffmpeg if available; otherwise tries imageio-ffmpeg.
    If only one MP4 exists, it is renamed to `output_file`.
    """
    mp4_files = sorted(input_dir.glob("*.mp4"))
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
