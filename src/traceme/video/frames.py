from __future__ import annotations

import colorsys
from pathlib import Path
import subprocess

import cv2
import numpy as np
import torch


def _obj_color(obj_id: int) -> tuple[int, int, int]:
    """Deterministic bright color per obj_id, as BGR for OpenCV."""
    h = (obj_id * 0.1357) % 1.0
    s, v = 0.85, 1.0
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(b * 255), int(g * 255), int(r * 255)  # BGR


def _mask_to_bool(m):
    """Accepts torch tensor (1,H,W) or (H,W) or np; returns (H,W) bool."""
    if isinstance(m, torch.Tensor):
        m = m.detach().float().cpu().numpy()
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    return m > 0


def _annotate_frame(frame_bgr: np.ndarray, obj_ids: list[int], masks) -> np.ndarray:
    """
    masks: iterable aligned with obj_ids, each (1,H,W) or (H,W)
    Draws semi-transparent fill + contour + text label.
    """
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    overlay = out.copy()

    for k, oid in enumerate(obj_ids):
        m = _mask_to_bool(masks[k])
        if m.shape[:2] != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        if not m.any():
            continue

        color = _obj_color(int(oid))

        # fill
        overlay[m] = (0.6 * np.array(color) + 0.4 * overlay[m]).astype(np.uint8)

        # contour
        m8 = (m.astype(np.uint8) * 255)
        cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, color, thickness=2)

        # label: place near the largest contour if available
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            moments = cv2.moments(c)
            cx = int(moments["m10"] / (moments["m00"] + 1e-6))
            cy = int(moments["m01"] / (moments["m00"] + 1e-6))
        else:
            # fall back to top-left-ish
            ys, xs = np.where(m)
            cy = int(ys.mean()) if ys.size else 20
            cx = int(xs.mean()) if xs.size else 20

        txt = f"id:{oid}"
        cv2.putText(
            overlay,
            txt,
            (max(2, cx - 20), max(15, cy - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            txt,
            (max(2, cx - 20), max(15, cy - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return cv2.addWeighted(overlay, 0.9, out, 0.1, 0.0)


def xywh_to_xyxy(box_xywh):
    x, y, w, h = map(int, box_xywh)
    return [x, y, x + w, y + h]


def extract_frames(input_file, output_dir, quality: int = 1, start_number: int = 0, threads: int = 10):
    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_pattern = str(output_path / "%05d.jpg")

    command = [
        "ffmpeg",
        "-i", str(input_path),
        "-q:v", str(quality),
        "-start_number", str(start_number),
        output_pattern,
        "-threads", str(threads),
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Frames extracted to: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")
        raise


def save_mask_png(mask: torch.Tensor | np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m = mask.detach().float().cpu().numpy() if isinstance(mask, torch.Tensor) else mask
    if m.ndim == 3 and m.shape[0] == 1:  # (1,H,W) -> (H,W)
        m = m[0]
    cv2.imwrite(str(out_path), ((m > 0).astype(np.uint8) * 255))


def save_overlay(
    img_path: Path,
    mask: torch.Tensor | np.ndarray,
    out_path: Path,
    alpha: float = 0.5,
):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return
    m = mask.detach().float().cpu().numpy() if isinstance(mask, torch.Tensor) else mask
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    m = m > 0
    overlay = img.copy()
    color = np.zeros_like(img)
    color[..., 2] = 255
    overlay[m] = (alpha * color[m] + (1 - alpha) * img[m]).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)
