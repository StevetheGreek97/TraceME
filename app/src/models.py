from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import re
from typing import Dict, List, Optional, Tuple, Any

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ANNOT_SCHEMA_VERSION = 1


def _numeric_key(p: Path) -> int:
    s = p.stem
    if s.isdigit():
        return int(s)
    m = re.findall(r"\d+", s)
    return int(m[-1]) if m else 0


@dataclass
class ObjectAnno:
    box: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
    points: List[Tuple[int, int]] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)  # 1=positive, 0=negative
    polygon: Optional[List[Tuple[int, int]]] = None
    class_name: Optional[str] = None

    def add_point(self, x: int, y: int, label: int):
        self.points.append((int(x), int(y)))
        self.labels.append(int(label))

    def remove_point(self, x: int, y: int, label: int) -> bool:
        for i, ((px, py), lab) in enumerate(zip(self.points, self.labels)):
            if px == int(x) and py == int(y) and lab == int(label):
                del self.points[i]
                del self.labels[i]
                return True
        return False

    def set_box(self, x: int, y: int, w: int, h: int):
        self.box = (int(x), int(y), int(w), int(h))

    def set_polygon(self, polygon: Optional[List[Tuple[int, int]]]):
        if polygon:
            self.polygon = [(int(x), int(y)) for x, y in polygon]
        else:
            self.polygon = None

    def set_class(self, name: Optional[str]):
        self.class_name = name


@dataclass
class FrameAnno:
    objects: Dict[int, ObjectAnno] = field(default_factory=dict)

    def ensure_object(self, obj_id: int) -> ObjectAnno:
        if obj_id not in self.objects:
            self.objects[obj_id] = ObjectAnno()
        return self.objects[obj_id]


class AnnotationModel:
    def __init__(self, frames_dir: Path):
        self.frames_dir = Path(frames_dir)
        if self.frames_dir.exists():
            self.frames = sorted(
                [p for p in self.frames_dir.iterdir() if p.suffix.lower() in IMG_EXTS],
                key=_numeric_key,
            )
        else:
            self.frames = []
        self.index: int = 0
        self.ann: Dict[int, FrameAnno] = {}
        self.dirty: bool = False

    def clamp_index(self, i: int) -> int:
        return max(0, min(i, len(self.frames) - 1)) if self.frames else 0

    def set_index(self, i: int) -> int:
        self.index = self.clamp_index(i)
        return self.index

    def _frame(self, fidx: Optional[int] = None) -> FrameAnno:
        idx = self.index if fidx is None else fidx
        if idx not in self.ann:
            self.ann[idx] = FrameAnno()
        return self.ann[idx]

    def add_point(self, obj_id: int, x: int, y: int, label: int):
        obj = self._frame().ensure_object(obj_id)
        obj.add_point(x, y, label)
        self.dirty = True

    def remove_point(self, obj_id: int, x: int, y: int, label: int) -> bool:
        obj = self._frame().ensure_object(obj_id)
        removed = obj.remove_point(x, y, label)
        if removed:
            self.dirty = True
        return removed

    def set_box(self, obj_id: int, x: int, y: int, w: int, h: int):
        obj = self._frame().ensure_object(obj_id)
        obj.set_box(x, y, w, h)
        self.dirty = True

    def set_polygon(self, obj_id: int, polygon: Optional[List[Tuple[int, int]]]):
        obj = self._frame().ensure_object(obj_id)
        obj.set_polygon(polygon)
        self.dirty = True

    def set_class(self, obj_id: int, class_name: Optional[str]):
        obj = self._frame().ensure_object(obj_id)
        obj.set_class(class_name)
        self.dirty = True

    def clear_object(self, obj_id: int):
        if self.index not in self.ann:
            return
        if obj_id in self.ann[self.index].objects:
            self.ann[self.index].objects[obj_id] = ObjectAnno()
            self.dirty = True

    def get_object(self, fidx: int, obj_id: int) -> Optional[ObjectAnno]:
        fr = self.ann.get(fidx)
        if not fr:
            return None
        return fr.objects.get(obj_id)

    def is_frame_annotated(self, fidx: int) -> bool:
        fr = self.ann.get(fidx)
        if not fr:
            return False
        for obj in fr.objects.values():
            has_box = isinstance(obj.box, tuple) and len(obj.box) == 4 and obj.box[2] > 0 and obj.box[3] > 0
            has_points = len(obj.points) > 0
            has_poly = isinstance(obj.polygon, list) and len(obj.polygon) >= 3
            if has_box or has_points or has_poly:
                return True
        return False

    # ---------- persistence ----------
    def to_records(self, video_id: str) -> List[Dict[str, Any]]:
        records = []
        for fidx in sorted(self.ann.keys()):
            fr = self.ann[fidx]
            for oid in sorted(fr.objects.keys()):
                o = fr.objects[oid]
                has_box = isinstance(o.box, tuple) and len(o.box) == 4 and o.box[2] > 0 and o.box[3] > 0
                has_points = len(o.points) > 0
                has_poly = isinstance(o.polygon, list) and len(o.polygon) >= 3
                if not (has_box or has_points or has_poly):
                    continue
                rec: Dict[str, Any] = {
                    "video_id": video_id,
                    "frame_idx": int(fidx),
                    "obj_id": int(oid),
                    "points": [[int(x), int(y)] for (x, y) in o.points],
                    "labels": [int(v) for v in o.labels],
                }
                if o.class_name:
                    rec["class"] = o.class_name
                if has_box:
                    rec["box"] = [int(v) for v in o.box]
                if has_poly:
                    rec["polygon"] = [[int(x), int(y)] for (x, y) in o.polygon]
                records.append(rec)
        return records

    def load_records(self, records: List[Dict[str, Any]]):
        for rec in records:
            try:
                fidx = int(rec.get("frame_idx", 0))
                oid = int(rec.get("obj_id", 1))
                obj = self._frame(fidx).ensure_object(oid)
                pts = rec.get("points", []) or []
                labs = rec.get("labels", []) or []
                obj.points = [(int(x), int(y)) for x, y in pts]
                obj.labels = [int(v) for v in labs]
                box = rec.get("box", None)
                if isinstance(box, list) and len(box) == 4:
                    obj.box = tuple(int(v) for v in box)
                poly = rec.get("polygon", None)
                if isinstance(poly, list) and len(poly) >= 3:
                    obj.polygon = [(int(x), int(y)) for x, y in poly]
                obj.class_name = rec.get("class", None)
            except Exception:
                continue


def load_annotations(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("annotations", [])
    by_video: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        vid = rec.get("video_id", "")
        if not vid:
            continue
        by_video.setdefault(vid, []).append(rec)
    return by_video


def save_annotations(path: Path, models: Dict[str, AnnotationModel]):
    all_records: List[Dict[str, Any]] = []
    for vid, model in models.items():
        all_records.extend(model.to_records(vid))
        model.dirty = False
    payload = {"schema_version": ANNOT_SCHEMA_VERSION, "annotations": all_records}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def export_yaml(path: Path, models: Dict[str, AnnotationModel]) -> Tuple[bool, str]:
    if yaml is None:
        return False, "PyYAML not installed. Run: pip install pyyaml"
    items = []
    for vid, model in models.items():
        for rec in model.to_records(vid):
            entry = {
                "frame_idx": rec["frame_idx"],
                "obj_id": rec["obj_id"],
                "points": rec.get("points", []),
                "labels": rec.get("labels", []),
            }
            if "box" in rec:
                entry["box"] = rec["box"]
            if "polygon" in rec:
                entry["polygon"] = rec["polygon"]
            items.append(entry)
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump({"prompts": items}, f, sort_keys=False)
        return True, f"Saved {path}"
    except Exception as e:
        return False, str(e)
