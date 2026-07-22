# traceme/prompt_parser_yaml.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Literal

try:
    import yaml  # PyYAML
except Exception as e:  # pragma: no cover
    yaml = None

__all__ = [
    "Prompt",
    "YamlPromptParser",
]


# --------------------------- Data model ---------------------------

@dataclass(frozen=True)
class Prompt:
    """
    One interactive prompt annotation.

    Attributes
    ----------
    frame_idx : int
        Global frame index (0-based).
    obj_id : int
        Object identifier in that frame (0-based or 1-based; your choice).
    box : (x, y, w, h)
        Bounding box as integers.
    points : ((x1, y1), (x2, y2), ...)
        Point prompts.
    labels : (int, ...)
        Labels for points (e.g., 0=neg, 1=pos).
    """
    frame_idx: int
    obj_id: int
    box: Optional[Tuple[int, int, int, int]]  
    points: Tuple[Tuple[int, int], ...]
    labels: Tuple[int, ...]
    polygon: Optional[Tuple[Tuple[int, int], ...]] = None


# --------------------------- Parser ---------------------------

class YamlPromptParser:
    """
    Parser for YAML shaped like:

    prompts:
      - frame_idx: 0
        obj_id: 1
        box: [672, 914, 74, 110]
        points: [[486, 1094], [512, 1111]]
        labels: [0, 1]
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

    # ---------- Load & validate ----------

    def load(self, *, strict_keys: bool = False) -> List[Prompt]:
        """Load YAML and return a list of Prompt objects."""
        data = self._read_yaml(self.path)
        if not isinstance(data, dict) or "prompts" not in data:
            raise ValueError("YAML must contain a top-level 'prompts' list.")
        raw_list = data["prompts"]
        if not isinstance(raw_list, list):
            raise ValueError("'prompts' must be a list.")

        prompts: List[Prompt] = []
        for i, item in enumerate(raw_list):
            try:
                prompts.append(self._coerce_prompt(item, strict_keys=strict_keys))
            except Exception as e:
                raise ValueError(f"Invalid prompt at index {i}: {e}") from e
        return prompts

    # ---------- Convenience ----------

    @staticmethod
    def by_frame(prompts: Iterable[Prompt]) -> Dict[int, List[Prompt]]:
        out: Dict[int, List[Prompt]] = {}
        for p in prompts:
            out.setdefault(p.frame_idx, []).append(p)
        return out

    @staticmethod
    def filter_frames(
        prompts: Iterable[Prompt], *, min_idx: Optional[int] = None, max_idx: Optional[int] = None
    ) -> List[Prompt]:
        res: List[Prompt] = []
        for p in prompts:
            if min_idx is not None and p.frame_idx < min_idx:
                continue
            if max_idx is not None and p.frame_idx > max_idx:
                continue
            res.append(p)
        return res

    @staticmethod
    def to_jsonl(prompts: Iterable[Prompt], out_path: str | Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for p in prompts:
                row = {
                    "frame_idx": p.frame_idx,
                    "obj_id": p.obj_id,
                    "points": [list(pt) for pt in p.points],
                    "labels": list(p.labels),
                }
                if p.box is not None:
                    row["box"] = list(p.box)
                f.write(json.dumps(row) + "\n")
        return out_path

    # ---------- Chunk helpers (aligned with VideoChunker) ----------

    @staticmethod
    def frame_to_chunks(
        frame_idx: int,
        *,
        chunk_size: int,
        overlap: int = 0,
        total_frames: Optional[int] = None,
        total_chunks: Optional[int] = None
    ) -> List[int]:
        """
        Return the chunk index(es) containing this frame.
        Mirrors VideoChunker behavior:
          - chunk i covers frames [i*chunk_size - overlap, (i+1)*chunk_size - 1] (i>0, clipped at 0)
        Frames in the last `overlap` positions of chunk i also appear in chunk i+1.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if frame_idx < 0:
            raise ValueError("frame_idx must be >= 0")

        base = frame_idx // chunk_size
        chunks = [base]

        if overlap > 0 and (frame_idx % chunk_size) >= (chunk_size - overlap):
            nxt = base + 1
            if total_chunks is not None:
                if 0 <= nxt < total_chunks:
                    chunks.append(nxt)
            elif total_frames is not None:
                max_chunk = (total_frames + chunk_size - 1) // chunk_size
                if nxt < max_chunk:
                    chunks.append(nxt)
            else:
                chunks.append(nxt)

        return chunks

    @staticmethod
    def chunk_span(
        chunk_index: int,
        *,
        chunk_size: int,
        overlap: int = 0,
        total_frames: Optional[int] = None
    ) -> Tuple[int, int]:
        """
        Inclusive [start, end] frame index span covered by `chunk_index`.
        Matches how VideoChunker constructs overlap.
        """
        if chunk_index < 0:
            raise ValueError("chunk_index must be >= 0")
        start = chunk_index * chunk_size
        ovl_start = max(0, start - overlap) if chunk_index > 0 and overlap > 0 else start
        end = (chunk_index + 1) * chunk_size - 1
        if total_frames is not None:
            end = min(end, total_frames - 1)
        return ovl_start, end

    @staticmethod
    def prompts_by_chunk(
        prompts: Iterable[Prompt],
        *,
        chunk_size: int,
        overlap: int = 0,
        policy: Literal["both", "first", "last"] = "both",
        total_frames: Optional[int] = None
    ) -> Dict[int, List[Prompt]]:
        """
        Group prompts into chunk buckets.

        policy:
          - "both"  : duplicate boundary prompts into both chunks (default).
          - "first" : keep only the earlier chunk for boundary frames.
          - "last"  : keep only the later chunk for boundary frames.
        """
        # An explicit total_frames is authoritative. Only when it is unknown
        # do we bound the chunk count by the highest prompted frame (which
        # can under-estimate: a boundary prompt on the last prompted frame
        # would otherwise not be duplicated into the following chunk).
        total_chunks = None
        if total_frames is None:
            max_frame = max((p.frame_idx for p in prompts), default=-1)
            total_chunks = (max_frame + 1 + chunk_size - 1) // chunk_size if max_frame >= 0 else None

        buckets: Dict[int, List[Prompt]] = {}
        for p in prompts:
            candidates = YamlPromptParser.frame_to_chunks(
                p.frame_idx,
                chunk_size=chunk_size,
                overlap=overlap,
                total_frames=total_frames,
                total_chunks=total_chunks,
            )
            if policy == "first":
                candidates = candidates[:1]
            elif policy == "last":
                candidates = candidates[-1:]

            for c in candidates:
                if c < 0:
                    continue
                buckets.setdefault(c, []).append(p)

        return buckets

    # ---------- internals ----------

    @staticmethod
    def _coerce_prompt(item: Any, *, strict_keys: bool) -> Prompt:
        if not isinstance(item, dict):
            raise TypeError("prompt must be a mapping/object")

        required = {"frame_idx", "obj_id", "points", "labels"}
        optional = {"box", "polygon"}
        missing = required - set(item)
        if missing:
            raise KeyError(f"missing keys: {sorted(missing)}")

        if strict_keys:
            extra = set(item) - (required | optional)
            if extra:
                raise ValueError(f"unknown keys: {sorted(extra)}")

        frame_idx = _as_int(item["frame_idx"], name="frame_idx", minv=0)
        obj_id    = _as_int(item["obj_id"],    name="obj_id",    minv=0)

        # --- optional box ---
        box_val = item.get("box", None)
        bx: Optional[Tuple[int,int,int,int]] = None
        if box_val is not None:
            if not (isinstance(box_val, (list, tuple)) and len(box_val) == 4):
                raise TypeError("box must be a list of 4 integers [x, y, w, h]")
            bx_tmp = tuple(_as_int(v, name=f"box[{i}]", minv=0) for i, v in enumerate(box_val))
            # treat zero-size boxes as not provided
            if bx_tmp[2] > 0 and bx_tmp[3] > 0:
                bx = bx_tmp

        points = item["points"]
        if not isinstance(points, (list, tuple)):
            raise TypeError("points must be a list of [x, y] pairs")
        pts: List[Tuple[int, int]] = []
        for j, pt in enumerate(points):
            if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
                raise TypeError(f"points[{j}] must be [x, y]")
            x = _as_int(pt[0], name=f"points[{j}][0]", minv=0)
            y = _as_int(pt[1], name=f"points[{j}][1]", minv=0)
            pts.append((x, y))

        labels = item["labels"]
        if not isinstance(labels, (list, tuple)):
            raise TypeError("labels must be a list of integers")
        labs = tuple(_as_int(v, name=f"labels[{i}]", minv=0) for i, v in enumerate(labels))

        polygon_val = item.get("polygon", None)
        poly: Optional[Tuple[Tuple[int, int], ...]] = None
        if polygon_val is not None:
            if not isinstance(polygon_val, (list, tuple)):
                raise TypeError("polygon must be a list of [x, y] pairs")
            poly_pts: List[Tuple[int, int]] = []
            for j, pt in enumerate(polygon_val):
                if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
                    raise TypeError(f"polygon[{j}] must be [x, y]")
                x = _as_int(pt[0], name=f"polygon[{j}][0]", minv=0)
                y = _as_int(pt[1], name=f"polygon[{j}][1]", minv=0)
                poly_pts.append((x, y))
            poly = tuple(poly_pts)

        return Prompt(
            frame_idx=frame_idx,
            obj_id=obj_id,
            box=bx,
            points=tuple(pts),
            labels=labs,
            polygon=poly,
        )

    @staticmethod
    def _read_yaml(path: Path) -> Dict[str, Any]:
        if yaml is None:
            raise RuntimeError("PyYAML is required. Install with `pip install pyyaml`.")
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError("Top-level YAML must be a mapping/object.")
        return data


# --------------------------- helpers ---------------------------

def _as_int(v: Any, *, name: str, minv: Optional[int] = None, maxv: Optional[int] = None) -> int:
    try:
        iv = int(v)
    except Exception:
        raise TypeError(f"{name} must be an integer, got {type(v).__name__}")
    if minv is not None and iv < minv:
        raise ValueError(f"{name} must be >= {minv}, got {iv}")
    if maxv is not None and iv > maxv:
        raise ValueError(f"{name} must be <= {maxv}, got {iv}")
    return iv


# --------------------------- CLI ---------------------------

def _main() -> None:  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser(description="Parse YAML prompts and (optionally) export JSONL.")
    ap.add_argument("infile", type=str, help="Path to YAML file.")
    ap.add_argument("--strict", action="store_true", help="Fail on unknown keys.")
    ap.add_argument("--jsonl", type=str, default="", help="Optional output JSONL path.")
    ap.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for grouping.")
    ap.add_argument("--overlap", type=int, default=2, help="Overlap for grouping.")
    args = ap.parse_args()

    parser = YamlPromptParser(args.infile)
    prompts = parser.load(strict_keys=args.strict)
    print(f"Loaded {len(prompts)} prompts from {args.infile}")

    # quick peek: unique frames
    frames = sorted(set(p.frame_idx for p in prompts))
    print(f"Frames present: {len(frames)} (min={frames[0] if frames else 'n/a'}, max={frames[-1] if frames else 'n/a'})")

    # group by chunk (summary)
    by_chunk = YamlPromptParser.prompts_by_chunk(
        prompts,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        policy="both",
    )
    print(f"Chunks with prompts: {len(by_chunk)}")
    for cid in sorted(by_chunk.keys())[:10]:
        span = YamlPromptParser.chunk_span(cid, chunk_size=args.chunk_size, overlap=args.overlap,
                                           total_frames=(frames[-1] + 1) if frames else None)
        print(f"  chunk {cid:03d} span={span} prompts={len(by_chunk[cid])}")

    if args.jsonl:
        outp = YamlPromptParser.to_jsonl(prompts, args.jsonl)
        print(f"Wrote JSONL to {outp}")

if __name__ == "__main__":  # pragma: no cover
    _main()
