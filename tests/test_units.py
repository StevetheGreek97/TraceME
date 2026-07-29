import numpy as np
import pandas as pd
import pytest

from traceme.sam2.io import (
    _mask_stats,
    _pack_mask_bool,
    _unpack_mask,
    _write_csv_for_chunk,
    global_to_inchunk_idx,
    CSV_HEADER,
)
from traceme.prompts.parser import Prompt, YamlPromptParser
from traceme.video.chunker import VideoChunker, numeric_sort_key
from traceme.video.merge import merge_csv_chunks


# ---------------- mask stats & packing ----------------

def test_mask_stats_rectangle():
    m = np.zeros((16, 16), dtype=np.uint8)
    m[2:6, 3:9] = 1
    area, cx, cy, bx, by, bw, bh = _mask_stats(m[None, :, :])
    assert area == 24
    assert (bx, by, bw, bh) == (3, 2, 6, 4)
    assert (cx, cy) == (5.5, 3.5)


def test_mask_stats_empty():
    assert _mask_stats(np.zeros((8, 8))) is None


def test_pack_unpack_roundtrip():
    rng = np.random.default_rng(42)
    for shape in [(16, 16), (17, 31), (1, 64, 65)]:
        m = rng.random(shape) > 0.5
        packed, shp = _pack_mask_bool(m)
        out = _unpack_mask(packed, shp)
        assert out.shape == m.squeeze().shape
        assert np.array_equal(out, m.squeeze().astype(bool))


# ---------------- chunk index math consistency ----------------

@pytest.mark.parametrize("cs,ov", [(5, 1), (500, 1), (100, 5), (10, 0)])
def test_chunk_math_agrees(cs, ov):
    for fi in list(range(0, 3 * cs)) + [7 * cs - 1, 7 * cs]:
        chunks = YamlPromptParser.frame_to_chunks(fi, chunk_size=cs, overlap=ov)
        assert chunks, fi
        for c in chunks:
            start, end = YamlPromptParser.chunk_span(c, chunk_size=cs, overlap=ov)
            assert start <= fi <= end, (fi, c, start, end)
            in_idx = global_to_inchunk_idx(fi, c, cs, ov)
            assert 0 <= in_idx <= end - start, (fi, c, in_idx)
            assert start + in_idx == fi


def test_numeric_sort_key_padding_rollover(tmp_path):
    names = [f"chunk_{i}" for i in (99, 100, 1000, 2)]
    paths = [tmp_path / f"{n}.csv" for n in names]
    ordered = sorted(paths, key=numeric_sort_key)
    assert [p.stem for p in ordered] == ["chunk_2", "chunk_99", "chunk_100", "chunk_1000"]


# ---------------- prompt parser ----------------

def _write_yaml(tmp_path, text):
    p = tmp_path / "prompts.yaml"
    p.write_text(text)
    return p


def test_parser_valid(tmp_path):
    p = _write_yaml(
        tmp_path,
        "prompts:\n"
        "  - frame_idx: 3\n"
        "    obj_id: 1\n"
        "    box: [10, 20, 30, 40]\n"
        "    points: [[1, 2], [3, 4]]\n"
        "    labels: [0, 1]\n",
    )
    prompts = YamlPromptParser(p).load()
    assert prompts == [
        Prompt(frame_idx=3, obj_id=1, box=(10, 20, 30, 40), points=((1, 2), (3, 4)), labels=(0, 1))
    ]


def test_parser_zero_size_box_dropped(tmp_path):
    p = _write_yaml(
        tmp_path,
        "prompts:\n"
        "  - {frame_idx: 0, obj_id: 1, box: [5, 5, 0, 10], points: [[1, 1]], labels: [1]}\n",
    )
    assert YamlPromptParser(p).load()[0].box is None


def test_parser_missing_key(tmp_path):
    p = _write_yaml(tmp_path, "prompts:\n  - {frame_idx: 0, obj_id: 1, points: []}\n")
    with pytest.raises(ValueError, match="labels"):
        YamlPromptParser(p).load()


def test_parser_strict_unknown_key(tmp_path):
    p = _write_yaml(
        tmp_path,
        "prompts:\n  - {frame_idx: 0, obj_id: 1, points: [], labels: [], bogus: 1}\n",
    )
    with pytest.raises(ValueError, match="bogus"):
        YamlPromptParser(p).load(strict_keys=True)


def test_prompts_by_chunk_boundary_duplication():
    prompts = [Prompt(frame_idx=499, obj_id=1, box=None, points=((1, 1),), labels=(1,))]
    # with the real frame count, the boundary prompt lands in both chunks
    buckets = YamlPromptParser.prompts_by_chunk(
        prompts, chunk_size=500, overlap=1, total_frames=600
    )
    assert set(buckets) == {0, 1}
    # without it, the chunk count is bounded by the highest prompted frame
    buckets = YamlPromptParser.prompts_by_chunk(prompts, chunk_size=500, overlap=1)
    assert set(buckets) == {0}


# ---------------- CSV writer + merge ----------------

def test_csv_write_and_merge(tmp_path):
    files_dir = tmp_path / "files"
    stats0 = {
        0: {1: (24, 5.5, 3.5, 3, 2, 6, 4)},
        1: {},                     # empty frame
        4: {1: (24, 5.5, 3.5, 3, 2, 6, 4), 2: None},  # vanished object
    }
    stats1 = {0: {1: (24, 6.5, 3.5, 4, 2, 6, 4)}}  # in-chunk 0 == global 4 (overlap dup)
    _write_csv_for_chunk(files_dir / "clip_chunk_000.csv", stats0, cid=0, cs=5, ov=1)
    _write_csv_for_chunk(files_dir / "clip_chunk_001.csv", stats1, cid=1, cs=5, ov=1)

    merged = tmp_path / "clip.csv"
    merge_csv_chunks(files_dir, merged)
    df = pd.read_csv(merged)

    assert list(df.columns) == CSV_HEADER
    # boundary frame 4 deduped: obj 1 kept once (first chunk's row wins)
    f4_obj1 = df[(df["global_frame_idx"] == 4) & (df["obj_id"] == 1)]
    assert len(f4_obj1) == 1
    assert f4_obj1.iloc[0]["chunk_id"] == 0
    # empty frame kept with area 0 and blank obj_id
    f1 = df[df["global_frame_idx"] == 1]
    assert len(f1) == 1 and f1.iloc[0]["area_px"] == 0 and pd.isna(f1.iloc[0]["obj_id"])
    # vanished (lost) object row: sentinel -1 across all stat columns
    van = df[(df["global_frame_idx"] == 4) & (df["obj_id"] == 2)]
    assert van.iloc[0]["area_px"] == -1 and van.iloc[0]["bbox_x"] == -1


# ---------------- chunker ----------------

def _make_frames(d, n):
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (d / f"{i:05d}.jpg").write_bytes(b"\xff\xd8fake")
    return d


def test_chunker_symlink_build_and_reuse(tmp_path):
    frames = _make_frames(tmp_path / "frames", 9)
    out = tmp_path / "chunks"
    ch = VideoChunker(frame_dir=frames, output_dir=out, chunk_size=5, overlap=1, action="symlink")
    dirs = ch.chunk_frames(mode="auto")
    assert [d.name for d in dirs] == ["chunk_000", "chunk_001"]
    assert len(ch.get_frame_paths(0)) == 5
    assert len(ch.get_frame_paths(1)) == 5  # 1 overlap + 4
    assert (out / "chunk_001" / "00004.jpg").is_symlink()

    # second run with identical params reuses the manifest
    ch2 = VideoChunker(frame_dir=frames, output_dir=out, chunk_size=5, overlap=1, action="symlink")
    assert ch2._manifest_reason()[0] == "valid"

    # param change invalidates
    ch3 = VideoChunker(frame_dir=frames, output_dir=out, chunk_size=4, overlap=1, action="symlink")
    assert ch3._manifest_reason()[0] == "param_mismatch"
