from pathlib import Path
import gc
import numpy as np
import torch
from contextlib import nullcontext
from time import sleep
from typing import Literal

from sam2.build_sam import build_sam2_video_predictor
from tracewave.sam2.config import SAM2_CHECKPOINT, MODEL_CFG, device, SEED_DIRNAME
from tracewave.sam2.io import (
    _seed_file,
    _unpack_mask,
    _pack_mask_bool,
    _write_csv_for_chunk,
    global_to_inchunk_idx,
)
from tracewave.core.logging import get_logger, timer
from tracewave.video.render import _render_chunk_video

log = get_logger("tracewave.sam2.runner")


def _safe_unlink(p: Path, retries=5, delay=0.1):
    for i in range(retries):
        try:
            p.unlink(missing_ok=True)
            return
        except PermissionError:
            gc.collect()
            sleep(delay * (i + 1))
    try:
        p.rename(p.with_suffix(p.suffix + ".stale"))
    except Exception:
        pass


def _cleanup_predictor(predictor, inf_state):
    if inf_state is not None:
        del inf_state
    if predictor is not None:
        del predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return None, None


def run_sam2(
    *,
    chunker,
    by_chunk: dict[int, list],
    output: str | Path,
    video_fps: int = 30,
    prepare_chunks: bool = True,
    chunk_mode: Literal["auto", "load", "force"] = "auto",
):
    out_root = chunker.output_dir
    (out_root / SEED_DIRNAME).mkdir(parents=True, exist_ok=True)
    if prepare_chunks:
        with timer(log, "Chunking preparation"):
            chunker.chunk_frames(mode=chunk_mode)

    cs, ov = chunker.chunk_size, chunker.overlap
    total_chunks = chunker.count_chunks()
    all_chunk_ids = list(range(total_chunks))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    # Base name for outputs (e.g., 'Control1_A7_100925')
    try:
        base_name = Path(chunker.frame_dir).name
    except Exception:
        base_name = output.name
    log.info(f"Base output name: {base_name}")
    log.info(f"Chunk size: {cs} | overlap: {ov} | total chunks: {total_chunks}")

    # Perf knobs (CUDA only)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        log.info("Enabled TF32 + cuDNN benchmark; using bfloat16 autocast")
    else:
        autocast_ctx = nullcontext()

    total_frames = 0
    total_objects = 0

    predictor = None
    inf_state = None

    try:
        for cid in all_chunk_ids:
            chunk_dir = chunker.get_chunk_dir(cid)
            frame_files = chunker.get_frame_paths(cid)
            n_frames = len(frame_files)
            log.info(f"[chunk {cid:03d}] dir={chunk_dir} | frames={n_frames}")
            if n_frames == 0:
                log.warning(f"[chunk {cid:03d}] No frames; skipping")
                sf = _seed_file(out_root, cid - 1)
                if sf.exists():
                    _safe_unlink(sf)
                continue

            # Build predictor per chunk
            predictor = build_sam2_video_predictor(
                MODEL_CFG,
                str(SAM2_CHECKPOINT),
                device=device,
            )

            try:
                # init state
                inf_state = predictor.init_state(
                    video_path=str(chunk_dir),
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=False,
                    async_loading_frames=True,
                )

                # ---- restore overlap seeds ----
                if ov > 0 and cid > 0:
                    prev_seed_path = _seed_file(out_root, cid - 1)
                    if prev_seed_path.exists():
                        with timer(log, f"[chunk {cid:03d}] restore_seeds"):
                            try:
                                with np.load(prev_seed_path, allow_pickle=True) as data:
                                    rel_indices = data["rel_indices"].copy()
                                    obj_ids_arr = data["obj_ids"].copy()
                                    packed_list = data["packed"].copy()
                                    shapes_list = data["shapes"].copy()

                                restored = 0
                                for r, oids, packed_masks, shapes in zip(
                                    rel_indices, obj_ids_arr, packed_list, shapes_list
                                ):
                                    if r >= n_frames:
                                        continue
                                    oids = np.asarray(oids, dtype=np.int32)
                                    packed_masks = [np.asarray(pm, dtype=np.uint8) for pm in packed_masks]
                                    shapes = [tuple(map(int, shp)) for shp in shapes]
                                    for oid, packed, shp in zip(oids, packed_masks, shapes):
                                        m = _unpack_mask(packed, shp)
                                        predictor.add_new_mask(
                                            inference_state=inf_state,
                                            frame_idx=int(r),
                                            obj_id=int(oid),
                                            mask=m,
                                        )
                                        restored += 1
                                log.info(f"[chunk {cid:03d}] restored {restored} seed masks from overlap")
                            finally:
                                _safe_unlink(prev_seed_path)

                # ---- apply prompts ----
                plist = by_chunk.get(cid, [])
                if plist:
                    with timer(log, f"[chunk {cid:03d}] apply_prompts"):
                        applied = 0
                        per_frame: dict[int, dict[int, dict[str, list]]] = {}
                        for p in plist:
                            i = global_to_inchunk_idx(p.frame_idx, cid, cs, ov)
                            if 0 <= i < n_frames:
                                rec = per_frame.setdefault(i, {}).setdefault(
                                    int(p.obj_id), {"points": [], "labels": [], "boxes": []}
                                )
                                for (x, y), lab in zip(p.points, p.labels):
                                    rec["points"].append([int(x), int(y)])
                                    rec["labels"].append(int(lab))
                                if p.box:
                                    x, y, w, h = map(int, p.box)
                                    rec["boxes"].append([x, y, x + w, y + h])
                        for i in sorted(per_frame.keys()):
                            for obj_id, pack in per_frame[i].items():
                                box_xyxy = pack["boxes"][0] if pack["boxes"] else None
                                predictor.add_new_points_or_box(
                                    inference_state=inf_state,
                                    frame_idx=i,
                                    obj_id=int(obj_id),
                                    points=pack["points"] or None,
                                    labels=pack["labels"] or None,
                                    clear_old_points=True,
                                    normalize_coords=True,
                                    box=box_xyxy,
                                )
                                applied += 1
                        log.info(f"[chunk {cid:03d}] applied {applied} prompts")

                # ---- propagate ----
                video_segments: dict[int, dict[int, np.ndarray]] = {}
                with timer(log, f"[chunk {cid:03d}] propagate_and_collect"):
                    with torch.inference_mode(), autocast_ctx:
                        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inf_state):
                            masks_gpu = out_mask_logits > 0
                            masks_cpu = masks_gpu.to("cpu", dtype=torch.uint8, non_blocking=True)
                            ids_cpu = torch.as_tensor(out_obj_ids, dtype=torch.int32).cpu()
                            segs = {int(ids_cpu[i]): masks_cpu[i].numpy() for i in range(masks_cpu.shape[0])}
                            video_segments[out_frame_idx] = segs

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                # ---- CSV (RENAMED) ----
                with timer(log, f"[chunk {cid:03d}] write_csv"):
                    areas_per_frame: dict[int, dict[int, int]] = {}
                    objects_in_chunk = 0
                    for idx, segs in video_segments.items():
                        if not segs:
                            areas_per_frame[idx] = {}
                        else:
                            apf = {oid: int(np.count_nonzero(mask)) for oid, mask in segs.items()}
                            objects_in_chunk += len(apf)
                            areas_per_frame[idx] = apf

                    csv_path = output / f"{base_name}_chunk_{cid:03d}.csv"
                    _write_csv_for_chunk(csv_path, areas_per_frame, cid=cid, cs=cs, ov=ov)
                    log.info(
                        f"[chunk {cid:03d}] wrote {csv_path.name} "
                        f"(frames={len(areas_per_frame)}, objs={objects_in_chunk})"
                    )
                    total_frames += len(areas_per_frame)
                    total_objects += objects_in_chunk

                # ---- video (RENAMED) ----
                with timer(log, f"[chunk {cid:03d}] render_video"):
                    mp4_path = output / f"{base_name}_chunk_{cid:03d}.mp4"
                    _render_chunk_video(mp4_path, frame_files, video_segments, fps=video_fps)
                    log.info(f"[chunk {cid:03d}] wrote {mp4_path.name}")
                # ---- overlap seeds ----
                if ov > 0:
                    with timer(log, f"[chunk {cid:03d}] write_overlap_seeds"):
                        tail_start = max(0, n_frames - ov)
                        tail_indices = list(range(tail_start, n_frames))
                        rel_indices, obj_ids_per_rel, packed_per_rel, shapes_per_rel = [], [], [], []
                        for rel, abs_idx in enumerate(tail_indices):
                            segs = video_segments.get(abs_idx, {})
                            if not segs:
                                continue
                            rel_indices.append(rel)
                            oids_row, packed_row, shapes_row = [], [], []
                            for oid, mask_bool in segs.items():
                                packed, shp = _pack_mask_bool(mask_bool)
                                oids_row.append(int(oid))
                                packed_row.append(packed)
                                shapes_row.append(shp)
                            obj_ids_per_rel.append(np.array(oids_row, dtype=np.int32))
                            packed_per_rel.append(np.array(packed_row, dtype=object))
                            shapes_per_rel.append(np.array(shapes_row, dtype=object))

                        if rel_indices:
                            sf = _seed_file(out_root, cid)
                            np.savez_compressed(
                                sf,
                                rel_indices=np.array(rel_indices, dtype=np.int32),
                                obj_ids=np.array(obj_ids_per_rel, dtype=object),
                                packed=np.array(packed_per_rel, dtype=object),
                                shapes=np.array(shapes_per_rel, dtype=object),
                            )
                            log.info(f"[chunk {cid:03d}] wrote seeds → {sf.name} (frames={len(rel_indices)})")

                log.info(f"[OK] chunk {cid:03d} complete")

            except Exception as e:
                log.exception(f"[chunk {cid:03d}] processing failed: {e}")

            finally:
                predictor, inf_state = _cleanup_predictor(predictor, inf_state)

    finally:
        predictor, inf_state = _cleanup_predictor(predictor, inf_state)
    log.info(f"Run complete: frames={total_frames}, objects={total_objects}, chunks={total_chunks}")
