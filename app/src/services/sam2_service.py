from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import torch
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.build_sam import build_sam2
except Exception:  # pragma: no cover
    torch = None
    SAM2ImagePredictor = None
    build_sam2 = None


def get_resource_path(rel_path: str) -> str:
    if getattr(sys, "frozen", False):
        base_path = sys._MEIPASS  # type: ignore[attr-defined]
    else:
        # project root (three levels up from app/services)
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base_path, rel_path)


class Sam2Service:
    def __init__(self, device: str, config_name: str, weights_path: str):
        self.available = False
        self.error = ""
        self.device = device
        self.config_name = config_name
        self.weights_path = weights_path
        self.predictor: Optional[SAM2ImagePredictor] = None
        self._init_model()

    def _init_model(self):
        if torch is None or SAM2ImagePredictor is None or build_sam2 is None:
            self.error = "SAM2 dependencies are not available."
            return
        if cv2 is None:
            self.error = "OpenCV is required for SAM2 mask conversion."
            return
        dev = self.device
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
        try:
            model_path = self._resolve_weights_path(self.weights_path)
            sam2_model = build_sam2(self.config_name, model_path, device=dev)
            self.predictor = SAM2ImagePredictor(sam2_model)
            self.available = True
        except Exception as e:
            self.error = str(e)
            self.available = False

    @staticmethod
    def _resolve_weights_path(weights_path: str) -> str:
        if os.path.isabs(weights_path):
            return weights_path
        base_path = get_resource_path("")
        candidate = os.path.join(base_path, weights_path)
        if os.path.exists(candidate):
            return candidate
        # Common legacy path mapping: app/ -> src/
        if weights_path.startswith("app/"):
            alt = os.path.join(base_path, weights_path.replace("app/", "src/", 1))
            if os.path.exists(alt):
                return alt
        # Fallback to src/sam2_configs/<filename>
        fname = os.path.basename(weights_path)
        alt2 = os.path.join(base_path, "src", "sam2_configs", fname)
        if os.path.exists(alt2):
            return alt2
        return candidate

    @staticmethod
    def _mask_to_polygon(mask: np.ndarray) -> List[Tuple[int, int]]:
        if cv2 is None:
            return []
        m = mask.astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        cnt = max(contours, key=cv2.contourArea)
        poly = cnt.squeeze(axis=1)
        if poly.ndim != 2 or poly.shape[1] != 2:
            return []
        return [(int(x), int(y)) for x, y in poly.tolist()]

    def generate_polygon(
        self,
        image_rgb: np.ndarray,
        points_xy: Optional[List[Tuple[int, int]]],
        point_labels: Optional[List[int]],
        box_xywh: Optional[List[int]],
    ) -> List[Tuple[int, int]]:
        if not self.available or self.predictor is None:
            return []

        pts = points_xy or []
        labs = point_labels or []

        box_xyxy = None
        if box_xywh and len(box_xywh) == 4 and box_xywh[2] > 0 and box_xywh[3] > 0:
            x, y, w, h = box_xywh
            box_xyxy = np.array([x, y, x + w, y + h], dtype=np.int32)[None, :]

        point_coords = np.array(pts, dtype=np.float32) if pts else None
        point_labs = np.array(labs, dtype=np.int32) if labs else None

        self.predictor.set_image(image_rgb)
        masks, _, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labs,
            box=box_xyxy,
            multimask_output=False,
        )
        mask = masks[0]
        return self._mask_to_polygon(mask)
