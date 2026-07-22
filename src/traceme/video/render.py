from __future__ import annotations

import cv2
import numpy as np

from traceme.video.frames import _annotate_frame


def _render_chunk_video(out_path, frame_files, video_segments, fps: int = 30, skip_first: int = 0):
    """
    skip_first: number of leading frames to omit from the video (overlap frames
    already rendered as the previous chunk's tail). They are still consumed for
    mask lookup, so video_segments stays indexed by in-chunk frame position.
    """
    if not frame_files:
        return

    first = cv2.imread(str(frame_files[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Failed to read first frame: {frame_files[0]}")
    h0, w0 = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w0, h0), True)

    for idx, fpath in enumerate(frame_files):
        if idx < skip_first:
            continue
        frame = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
        if frame is None:
            frame = np.zeros((h0, w0, 3), dtype=np.uint8)
        if (frame.shape[1], frame.shape[0]) != (w0, h0):
            frame = cv2.resize(frame, (w0, h0), interpolation=cv2.INTER_NEAREST)

        segs = video_segments.get(idx, {})
        if segs:
            obj_ids = sorted(segs.keys())
            frame = _annotate_frame(frame, obj_ids, [segs[oid] for oid in obj_ids])

        vw.write(frame)

    vw.release()
