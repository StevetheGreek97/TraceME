from __future__ import annotations

import cv2
import numpy as np


def _render_chunk_video(out_path, frame_files, video_segments, fps: int = 30):
    if not frame_files:
        return

    first = cv2.imread(str(frame_files[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Failed to read first frame: {frame_files[0]}")
    h0, w0 = first.shape[:2]

    def _ensure_mask_u8_hw(mask_bool_like, w, h):
        """
        Ensure a 2D uint8 mask with shape (h, w), resizing with nearest-neighbor if needed.
        Accepts bool/uint8, 2D or 3D with a singleton channel.
        """
        m = mask_bool_like
        if m is None:
            return np.zeros((h, w), dtype=np.uint8)
        m = np.asarray(m)
        if m.ndim == 3 and m.shape[0] == 1:   # [1,h,w] -> [h,w]
            m = m[0]
        if m.ndim == 3 and m.shape[2] == 1:   # [h,w,1] -> [h,w]
            m = m[..., 0]
        if m.dtype != np.uint8:
            # convert bool/other to uint8 {0,1}
            m = (m > 0).astype(np.uint8)
        if m.shape[0] != h or m.shape[1] != w:
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            # keep binary {0,1}
            m = (m > 0).astype(np.uint8)
        return m

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w0, h0), True)

    for idx, fpath in enumerate(frame_files):
        frame = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
        if frame is None:
            frame = np.zeros((h0, w0, 3), dtype=np.uint8)
        if (frame.shape[1], frame.shape[0]) != (w0, h0):
            frame = cv2.resize(frame, (w0, h0), interpolation=cv2.INTER_NEAREST)

        segs = video_segments.get(idx, {})

        if segs:
            overlay = frame.copy()
            union = np.zeros((h0, w0), dtype=np.uint8)
            for m in segs.values():
                mu = _ensure_mask_u8_hw(m, w0, h0)
                union = np.bitwise_or(union, mu)
            mask_u8 = union * 255  # 0/255
            # apply simple single-color overlay
            sel = mask_u8 > 0
            if np.any(sel):
                color = np.array((0, 255, 255), dtype=np.uint8)
                overlay[sel] = (0.6 * color + 0.4 * overlay[sel]).astype(np.uint8)
                frame = cv2.addWeighted(overlay, 0.9, frame, 0.1, 0.0)

        vw.write(frame)

    vw.release()
