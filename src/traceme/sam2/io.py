from __future__ import annotations

from pathlib import Path
import csv
import numpy as np

from traceme.sam2.config import SEED_DIRNAME


def _seed_file(out_root: Path, cid: int) -> Path:
    return out_root / SEED_DIRNAME / f"seed_chunk_{cid:03d}.npz"


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


def _write_csv_for_chunk(
    csv_path: Path,
    areas_per_frame: dict[int, dict[int, int]],
    *,
    cid: int,
    cs: int,
    ov: int,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chunk_id", "global_frame_idx", "in_chunk_idx", "obj_id", "area_px"])
        start = cid * cs
        ovl_start = max(0, start - ov) if cid > 0 and ov > 0 else start
        for in_idx in sorted(areas_per_frame.keys()):
            per_obj = areas_per_frame[in_idx]
            global_idx = ovl_start + in_idx
            if not per_obj:
                writer.writerow([cid, global_idx, in_idx, "", 0])
                continue
            for obj_id in sorted(per_obj.keys()):
                writer.writerow([cid, global_idx, in_idx, obj_id, int(per_obj[obj_id])])
