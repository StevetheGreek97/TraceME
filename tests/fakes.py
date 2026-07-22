"""Fake SAM2 predictor used by the test suite.

Mimics the predictor interface the runner uses (init_state /
add_new_points_or_box / add_new_mask / propagate_in_video) without any model
weights: every prompted or seeded object yields a deterministic 4x6 rectangle
mask shifted by its obj_id.
"""
from pathlib import Path

import torch

STATE = {"builds": 0, "fail_chunk_dirs": set()}


def reset():
    STATE["builds"] = 0
    STATE["fail_chunk_dirs"] = set()


class FakePredictor:
    def init_state(self, video_path, **kw):
        p = Path(video_path)
        return {"n": len(list(p.glob("*.jpg"))), "objs": {}, "dir": str(p)}

    def add_new_mask(self, inference_state, frame_idx, obj_id, mask):
        inference_state["objs"][int(obj_id)] = True

    def add_new_points_or_box(self, inference_state, frame_idx, obj_id, **kw):
        inference_state["objs"][int(obj_id)] = True

    def propagate_in_video(self, state):
        if state["dir"] in STATE["fail_chunk_dirs"]:
            raise RuntimeError("injected chunk failure")
        ids = sorted(state["objs"])
        for i in range(state["n"]):
            logits = torch.full((len(ids), 1, 16, 16), -1.0)
            for k, oid in enumerate(ids):
                logits[k, 0, 2:6, 3 + oid:9 + oid] = 1.0
            yield i, ids, logits


def build_predictor(*args, **kwargs):
    STATE["builds"] += 1
    return FakePredictor()
