"""Integration tests of the full pipeline against the fake predictor:
outputs, resume, failure handling, retry."""
import json

import cv2
import numpy as np
import pandas as pd
import pytest

import fakes
from traceme.pipeline import PipelineConfig, PipelinePaths, run_pipeline
from traceme.sam2.io import _done_marker, _seed_file


@pytest.fixture
def env(tmp_path):
    frames = tmp_path / "clipA"
    frames.mkdir()
    for i in range(9):
        img = np.full((16, 16, 3), 30 * (i % 8), dtype=np.uint8)
        cv2.imwrite(str(frames / f"{i:05d}.jpg"), img)
    prompt_file = tmp_path / "prompts.yaml"
    prompt_file.write_text(
        "prompts:\n"
        "  - {frame_idx: 1, obj_id: 1, points: [[4, 4]], labels: [1]}\n"
        "  - {frame_idx: 2, obj_id: 2, points: [[5, 3]], labels: [1]}\n"
    )
    cfg = PipelineConfig(
        frame_dir=frames,
        output_folder=tmp_path / "out",
        prompt_file=prompt_file,
        chunk_size=5,
        overlap=1,
        fps=5,
    )
    return cfg, PipelinePaths.from_config(cfg)


def _summary(paths):
    return json.loads(paths.run_summary.read_text())


def test_full_run_outputs(env):
    cfg, paths = env
    run_pipeline(cfg)

    s = _summary(paths)
    assert s["status"] == "complete"
    assert s["processed_chunks"] == [0, 1] and s["failed_chunks"] == []
    assert fakes.STATE["builds"] == 1, "predictor must be built once per run"
    assert _seed_file(paths.chunks_dir, 0).exists(), "seeds must persist for resume"
    assert _done_marker(paths.chunks_dir, 0).exists()

    df = pd.read_csv(paths.merged_csv)
    assert sorted(df["global_frame_idx"].unique()) == list(range(9))
    # boundary frame deduped to one row per (frame, obj)
    f4 = df[df["global_frame_idx"] == 4]
    assert len(f4) == len(f4["obj_id"].unique())
    # obj 2 (prompted in chunk 0) must survive into chunk 1 via seeds
    c1 = df[df["chunk_id"] == 1]
    assert set(c1["obj_id"].dropna().astype(int)) == {1, 2}
    assert (df["area_px"] == 24).all()

    # merged video: exactly one frame per input frame (overlap skipped)
    cap = cv2.VideoCapture(str(paths.merged_video))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    assert n == 9


def test_resume_skips_completed(env):
    cfg, paths = env
    run_pipeline(cfg)
    builds = fakes.STATE["builds"]

    run_pipeline(cfg)
    assert fakes.STATE["builds"] == builds, "resume must not rebuild the predictor"
    s = _summary(paths)
    assert s["resumed_chunks"] == [0, 1] and s["processed_chunks"] == []


def test_no_resume_reprocesses(env):
    cfg, paths = env
    run_pipeline(cfg)
    builds = fakes.STATE["builds"]

    run_pipeline(PipelineConfig(**{**cfg.__dict__, "resume": False}))
    assert fakes.STATE["builds"] == builds + 1
    assert _summary(paths)["processed_chunks"] == [0, 1]


def test_failure_marks_partial_and_exits_nonzero(env):
    cfg, paths = env
    fakes.STATE["fail_chunk_dirs"] = {str(paths.chunks_dir / "chunk_001")}

    with pytest.raises(SystemExit) as exc:
        run_pipeline(cfg)
    assert exc.value.code == 1

    s = _summary(paths)
    assert s["status"] == "partial" and s["failed_chunks"] == [1]
    assert not _done_marker(paths.chunks_dir, 1).exists()
    # tmp dir must survive even if del_tmp was requested (needed for resume)
    assert paths.tmp_root.exists()


def test_retry_heals_only_failed_chunk(env):
    cfg, paths = env
    fakes.STATE["fail_chunk_dirs"] = {str(paths.chunks_dir / "chunk_001")}
    with pytest.raises(SystemExit):
        run_pipeline(cfg)
    builds = fakes.STATE["builds"]

    fakes.STATE["fail_chunk_dirs"] = set()
    run_pipeline(cfg)
    assert fakes.STATE["builds"] == builds + 1, "only the failed chunk is reprocessed"
    s = _summary(paths)
    assert s["status"] == "complete"
    assert s["resumed_chunks"] == [0] and s["processed_chunks"] == [1]
