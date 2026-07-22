"""SAM3 adapter tests against the real sam3 package and checkpoint.

Skipped automatically when sam3 or the checkpoint is unavailable (e.g. CI).
Slow on CPU (~1 min); run explicitly with:  pytest tests/test_sam3_adapter.py
"""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")

_CKPT_CANDIDATES = [
    os.environ.get("SAM3_TEST_CHECKPOINT"),
    "/home/steve/mnt/cluster/data/sam2_checkpoints/sam3.pt",
    str(Path.home() / ".cache/traceme/sam2/checkpoints/sam3.pt"),
]
CKPT = next((p for p in _CKPT_CANDIDATES if p and Path(p).exists()), None)

pytestmark = [
    pytest.mark.skipif(importlib.util.find_spec("sam3") is None, reason="sam3 not installed"),
    pytest.mark.skipif(CKPT is None, reason="sam3.pt checkpoint not found"),
    pytest.mark.slow,
]


@pytest.fixture(scope="module")
def predictor():
    from traceme.sam2.sam3_backend import build_sam3_tracker

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return build_sam3_tracker(CKPT, device)


@pytest.fixture(scope="module")
def disk_frames(tmp_path_factory):
    root = tmp_path_factory.mktemp("sam3_frames")
    for i in range(3):
        img = np.full((256, 256, 3), 20, dtype=np.uint8)
        cv2.circle(img, (80 + 20 * i, 128), 40, (240, 240, 240), -1)
        cv2.imwrite(str(root / f"{i:05d}.jpg"), img)
    return root


def test_point_prompt_tracks_moving_disk(predictor, disk_frames):
    with torch.inference_mode():
        state = predictor.init_state(video_path=str(disk_frames))
        predictor.add_new_points_or_box(
            inference_state=state, frame_idx=0, obj_id=1,
            points=[[80, 128]], labels=[1], clear_old_points=True, box=None,
        )
        results = {}
        for fidx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            results[fidx] = (list(obj_ids), (mask_logits > 0).cpu().numpy())

    assert set(results) == {0, 1, 2}
    for fidx, (ids, m) in results.items():
        assert ids == [1]
        area = m[0, 0].sum()
        assert 2000 < area < 12000, f"frame {fidx}: implausible disk area {area}"
        ys, xs = np.nonzero(m[0, 0])
        assert abs(xs.mean() - (80 + 20 * fidx)) < 15


def test_numpy_mask_seed_propagates(predictor, disk_frames):
    with torch.inference_mode():
        state = predictor.init_state(video_path=str(disk_frames))
        seed = np.zeros((256, 256), dtype=bool)
        yy, xx = np.ogrid[:256, :256]
        seed[(xx - 80) ** 2 + (yy - 128) ** 2 <= 40**2] = True

        predictor.add_new_mask(inference_state=state, frame_idx=0, obj_id=7, mask=seed[None])
        n = 0
        for fidx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            m = (mask_logits > 0).cpu().numpy()
            assert list(obj_ids) == [7]
            assert m[0, 0].sum() > 2000
            n += 1
        assert n == 3
