from __future__ import annotations

from pathlib import Path
import csv
import numpy as np

from traceme.sam2.config import SEED_DIRNAME


def _seed_file(out_root: Path, cid: int) -> Path:
    return out_root / SEED_DIRNAME / f"seed_chunk_{cid:03d}.npz"


def _done_marker(out_root: Path, cid: int) -> Path:
    return out_root / SEED_DIRNAME / f"chunk_{cid:03d}.done"


def _pack_mask_bool(m_bool: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Accept masks as (H,W), (1,H,W) or (H,W,1), bool/uint8/0-255.
    Returns (packed_bits, (H, W)).
    """
    m = np.asarray(m_bool)

    # Squeeze common singleton channels
    if m.ndim == 3 and m.shape[0] == 1:   # [1,H,W] -> [H,W]
        m = m[0]
    if m.ndim == 3 and m.shape[2] == 1:   # [H,W,1] -> [H,W]
        m = m[..., 0]

    # Ensure 2D
    if m.ndim != 2:
        raise ValueError(f"_pack_mask_bool expects 2D mask, got shape {m.shape}")

    # Binarize and pack
    if m.dtype != np.bool_:
        m = (m > 0)
    h, w = m.shape
    packed = np.packbits(m.astype(np.uint8), axis=1)  # pack along width
    return packed, (h, w)


def _unpack_mask(packed: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Unpack a mask from packed bits to boolean array."""
    packed = np.asarray(packed, dtype=np.uint8)  # ensure correct dtype
    h, w = shape
    u = np.unpackbits(packed, axis=1)
    return u[:, :w].reshape(h, w).astype(bool)


def global_to_inchunk_idx(global_idx: int, cid: int, chunk_size: int, overlap: int) -> int:
    start = cid * chunk_size
    ovl_start = max(0, start - overlap) if cid > 0 and overlap > 0 else start
    return global_idx - ovl_start


def _mask_stats(mask) -> tuple[int, float, float, int, int, int, int] | None:
    """
    Per-object mask stats: (area_px, centroid_x, centroid_y, bbox_x, bbox_y, bbox_w, bbox_h).
    Returns None for an empty mask. Accepts (H,W), (1,H,W) or (H,W,1).
    """
    m = np.asarray(mask)
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    if m.ndim == 3 and m.shape[2] == 1:
        m = m[..., 0]
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (
        int(xs.size),
        round(float(xs.mean()), 2),
        round(float(ys.mean()), 2),
        x0,
        y0,
        x1 - x0 + 1,
        y1 - y0 + 1,
    )


def _mask_archive_path(csv_path: Path) -> Path:
    return csv_path.with_name(csv_path.stem + "_masks.npz")


def _write_masks_for_chunk(
    path: Path,
    video_segments: dict[int, dict[int, np.ndarray]],
    *,
    cid: int,
    cs: int,
    ov: int,
) -> None:
    """Persist every non-empty object mask in a chunk as bit-packed arrays.

    One entry per (frame, object) pair. Reload with e.g.:
        data = np.load(path, allow_pickle=True)
        for gidx, oid, packed, shp in zip(
            data["global_frame_idx"], data["obj_id"], data["packed"], data["shape"]
        ):
            mask = _unpack_mask(packed, tuple(shp))
    """
    start = cid * cs
    ovl_start = max(0, start - ov) if cid > 0 and ov > 0 else start

    global_idx_list: list[int] = []
    obj_id_list: list[int] = []
    packed_list: list[np.ndarray] = []
    shape_list: list[tuple[int, int]] = []

    for in_idx in sorted(video_segments.keys()):
        global_idx = ovl_start + in_idx
        for obj_id, mask in video_segments[in_idx].items():
            if not np.any(mask):
                continue
            packed, shp = _pack_mask_bool(mask)
            global_idx_list.append(global_idx)
            obj_id_list.append(int(obj_id))
            packed_list.append(packed)
            shape_list.append(shp)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        global_frame_idx=np.array(global_idx_list, dtype=np.int32),
        obj_id=np.array(obj_id_list, dtype=np.int32),
        packed=np.array(packed_list, dtype=object),
        shape=np.array(shape_list, dtype=object),
    )


CSV_HEADER = [
    "chunk_id", "global_frame_idx", "in_chunk_idx", "obj_id",
    "area_px", "centroid_x", "centroid_y",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h",
]


def _write_csv_for_chunk(
    csv_path: Path,
    stats_per_frame: dict[int, dict[int, tuple | None]],
    *,
    cid: int,
    cs: int,
    ov: int,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        start = cid * cs
        ovl_start = max(0, start - ov) if cid > 0 and ov > 0 else start
        for in_idx in sorted(stats_per_frame.keys()):
            per_obj = stats_per_frame[in_idx]
            global_idx = ovl_start + in_idx
            if not per_obj:
                writer.writerow([cid, global_idx, in_idx, "", 0, "", "", "", "", "", ""])
                continue
            for obj_id in sorted(per_obj.keys()):
                stats = per_obj[obj_id]
                if stats is None:
                    # Object is tracked but its mask vanished (lost) in this frame.
                    writer.writerow([cid, global_idx, in_idx, obj_id, -1, -1, -1, -1, -1, -1, -1])
                else:
                    writer.writerow([cid, global_idx, in_idx, obj_id, *stats])
