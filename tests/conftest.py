"""Test setup: run against src/ directly and stub out the sam2 package and
traceme.sam2.config so no model weights, GPU, or checkpoint resolution are
needed. The sam3 adapter tests (test_sam3_adapter.py) bypass these stubs and
use the real sam3 package when available.
"""
import sys
import types
from pathlib import Path

import pytest
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
sys.path.insert(0, str(_HERE))

import fakes  # noqa: E402

# ---- stub the sam2 package (runner imports build_sam2_video_predictor) ----
_sam2_mod = types.ModuleType("sam2")
_build_mod = types.ModuleType("sam2.build_sam")
_build_mod.build_sam2_video_predictor = fakes.build_predictor
_sam2_mod.build_sam = _build_mod
sys.modules["sam2"] = _sam2_mod
sys.modules["sam2.build_sam"] = _build_mod

# ---- stub traceme.sam2.config (it resolves checkpoints at import time) ----
_cfg_mod = types.ModuleType("traceme.sam2.config")
_cfg_mod.SAM2_CHECKPOINT = "/fake/ckpt.pt"
_cfg_mod.MODEL_CFG = "fake.yaml"
_cfg_mod.IS_SAM3 = False
_cfg_mod.device = torch.device("cpu")
_cfg_mod.SEED_DIRNAME = "_seeds"
_cfg_mod.log_runtime_summary = lambda log: None
_cfg_mod.log_model_summary = lambda log: None
_cfg_mod.log_precision_summary = lambda log: None

import traceme.sam2 as _ts  # noqa: E402

sys.modules["traceme.sam2.config"] = _cfg_mod
_ts.config = _cfg_mod


@pytest.fixture(autouse=True)
def _reset_fakes():
    fakes.reset()
    yield
    fakes.reset()
