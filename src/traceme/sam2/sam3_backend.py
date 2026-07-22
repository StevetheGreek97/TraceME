"""SAM3 backend: exposes Meta's SAM3 tracker behind the SAM2 video-predictor
interface used by traceme.sam2.runner.

sam3.pt ships the full detector+tracker video model. The pipeline's
points/boxes/mask workflow only needs the tracker (the SAM2-style PVS
component), so we build the standalone Sam3TrackerPredictor with its own
vision backbone and load the tracker weights plus the shared vision-backbone
weights out of the full checkpoint.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
import torch

from traceme.core.logging import get_logger

log = get_logger("traceme.sam2.sam3_backend")

_INSTALL_HINT = (
    "The 'sam3' package is required for --model sam3. "
    "Install it with: pip install 'traceme-pipeline[sam3]' or pip install sam3 "
    "(requires Python >= 3.12)."
)


@contextlib.contextmanager
def _cpu_construction_fallback():
    """sam3's PositionEmbeddingSine precomputes its cache with device='cuda'
    hardcoded; redirect those allocations to CPU when CUDA is unavailable so
    the model can at least be built on CPU-only machines."""
    if torch.cuda.is_available():
        yield
        return
    orig_zeros = torch.zeros

    def zeros(*args, **kw):
        if kw.get("device") in ("cuda", torch.device("cuda")):
            kw["device"] = "cpu"
        return orig_zeros(*args, **kw)

    torch.zeros = zeros
    try:
        yield
    finally:
        torch.zeros = orig_zeros


_CPU_RUNTIME_PATCHED = False


def _apply_cpu_runtime_fallback():
    """sam3 calls .cuda() and .pin_memory() unconditionally in its frame
    loading and tracking paths (io_utils.py, _get_image_feature,
    _get_tpos_enc, ...). On a CPU-only machine every such call would crash,
    so make them no-ops there. GPU machines are unaffected."""
    global _CPU_RUNTIME_PATCHED
    if torch.cuda.is_available() or _CPU_RUNTIME_PATCHED:
        return
    log.warning(
        "CUDA not available: patching Tensor.cuda/pin_memory to no-ops so the "
        "SAM3 tracker can run on CPU (expect very slow inference)."
    )
    torch.Tensor.cuda = lambda self, *a, **kw: self
    torch.Tensor.pin_memory = lambda self, *a, **kw: self
    _CPU_RUNTIME_PATCHED = True


class Sam3TrackerAdapter:
    """SAM2-compatible facade over sam3's Sam3TrackerPredictor.

    Differences bridged here:
    - sam3 expects relative [0,1] prompt coordinates (rel_coordinates=True);
      the pipeline works in absolute pixels.
    - sam3's propagate_in_video yields 5-tuples and only consolidates prompts
      when propagate_preflight=True; the runner expects SAM2's 3-tuple.
    - sam3's add_new_mask requires a 2D torch tensor.
    - sam3 hardcodes CUDA as the state storage device unless state offloading
      is enabled.
    """

    def __init__(self, tracker, device: torch.device):
        self._tracker = tracker
        self._device = device

    def init_state(
        self,
        *,
        video_path,
        offload_video_to_cpu=True,
        offload_state_to_cpu=False,
        async_loading_frames=False,
    ):
        if self._device.type != "cuda":
            offload_state_to_cpu = True
        return self._tracker.init_state(
            video_path=video_path,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
            async_loading_frames=async_loading_frames,
        )

    def add_new_mask(self, *, inference_state, frame_idx, obj_id, mask):
        m = torch.as_tensor(np.asarray(mask))
        if m.dim() == 3:
            m = m.squeeze(0)
        return self._tracker.add_new_mask(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            mask=m,
        )

    def add_new_points_or_box(
        self,
        *,
        inference_state,
        frame_idx,
        obj_id,
        points=None,
        labels=None,
        clear_old_points=True,
        normalize_coords=True,  # accepted for SAM2 signature compatibility
        box=None,
    ):
        w = float(inference_state["video_width"])
        h = float(inference_state["video_height"])
        pts = labs = None
        if points:
            pts = torch.tensor(points, dtype=torch.float32) / torch.tensor([w, h])
            labs = torch.tensor(labels, dtype=torch.int32)
        box_t = None
        if box is not None:
            box_t = torch.tensor(box, dtype=torch.float32) / torch.tensor([w, h, w, h])
        return self._tracker.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=pts,
            labels=labs,
            clear_old_points=clear_old_points,
            rel_coordinates=True,
            box=box_t,
        )

    def propagate_in_video(self, inference_state):
        it = self._tracker.propagate_in_video(
            inference_state,
            start_frame_idx=None,
            max_frame_num_to_track=None,
            reverse=False,
            tqdm_disable=True,
            propagate_preflight=True,
        )
        for frame_idx, obj_ids, _low_res_masks, video_res_masks, _obj_scores in it:
            yield frame_idx, obj_ids, video_res_masks


def build_sam3_tracker(checkpoint_path: str | Path, device: torch.device) -> Sam3TrackerAdapter:
    try:
        from sam3.model_builder import build_tracker
    except ImportError as e:
        raise ImportError(_INSTALL_HINT) from e

    _apply_cpu_runtime_fallback()
    with _cpu_construction_fallback():
        tracker = build_tracker(apply_temporal_disambiguation=True, with_backbone=True)

    sd = torch.load(str(checkpoint_path), map_location="cpu", mmap=True, weights_only=True)
    if "model" in sd:
        sd = sd["model"]
    remapped = {}
    for k, v in sd.items():
        if k.startswith("tracker."):
            remapped[k[len("tracker."):]] = v
        elif k.startswith("detector.backbone.vision_backbone."):
            remapped["backbone." + k[len("detector.backbone."):]] = v
    tracker.load_state_dict(remapped, strict=True)
    tracker = tracker.to(device).eval()
    log.info("SAM3 tracker loaded from %s (%d tensors)", checkpoint_path, len(remapped))
    return Sam3TrackerAdapter(tracker, device)
