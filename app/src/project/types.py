from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = 1


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _from_rel(path_str: str, root: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (root / p).resolve()


@dataclass
class Sam2Settings:
    config_name: str = "sam2_hiera_t.yaml"
    weights_path: str = "src/sam2_configs/sam2_hiera_tiny.pt"
    auto_run: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_name": self.config_name,
            "weights_path": self.weights_path,
            "auto_run": self.auto_run,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Sam2Settings":
        return cls(
            config_name=str(data.get("config_name", cls().config_name)),
            weights_path=str(data.get("weights_path", cls().weights_path)),
            auto_run=bool(data.get("auto_run", cls().auto_run)),
        )


@dataclass
class UIState:
    last_video_id: Optional[str] = None
    last_frame_index: int = 0
    mode: str = "box"
    show_only_annotated: bool = False
    last_class: Optional[str] = None
    last_obj_id: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "last_video_id": self.last_video_id,
            "last_frame_index": int(self.last_frame_index),
            "mode": self.mode,
            "show_only_annotated": bool(self.show_only_annotated),
            "last_class": self.last_class,
            "last_obj_id": int(self.last_obj_id),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UIState":
        return cls(
            last_video_id=data.get("last_video_id"),
            last_frame_index=int(data.get("last_frame_index", 0)),
            mode=str(data.get("mode", "box")),
            show_only_annotated=bool(data.get("show_only_annotated", False)),
            last_class=data.get("last_class"),
            last_obj_id=int(data.get("last_obj_id", 1)),
        )


@dataclass
class ClassLabel:
    name: str
    color: str  # hex string

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "color": self.color}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClassLabel":
        return cls(name=str(data.get("name", "class")), color=str(data.get("color", "#00FF00")))


@dataclass
class VideoItem:
    id: str
    name: str
    source_path: str
    frames_dir: str  # relative to project root
    frame_count: int = 0
    fps: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "source_path": self.source_path,
            "frames_dir": self.frames_dir,
            "frame_count": int(self.frame_count),
            "fps": self.fps,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VideoItem":
        return cls(
            id=str(data.get("id", "")),
            name=str(data.get("name", "")),
            source_path=str(data.get("source_path", "")),
            frames_dir=str(data.get("frames_dir", "")),
            frame_count=int(data.get("frame_count", 0)),
            fps=data.get("fps", None),
        )


@dataclass
class Project:
    root: Path
    name: str
    created_at: str = field(default_factory=_now_iso)
    last_opened: str = field(default_factory=_now_iso)
    frames_root: Path = field(default_factory=lambda: Path("frames"))
    annotations_path: Path = field(default_factory=lambda: Path("annotations.json"))
    videos: List[VideoItem] = field(default_factory=list)
    classes: List[ClassLabel] = field(default_factory=list)
    ui_state: UIState = field(default_factory=UIState)
    sam2: Sam2Settings = field(default_factory=Sam2Settings)

    @property
    def config_path(self) -> Path:
        return self.root / "project.json"

    def frames_root_abs(self) -> Path:
        return (self.root / self.frames_root).resolve()

    def annotations_path_abs(self) -> Path:
        return (self.root / self.annotations_path).resolve()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "project_name": self.name,
            "created_at": self.created_at,
            "last_opened": self.last_opened,
            "frames_root": self.frames_root.as_posix(),
            "annotations_path": self.annotations_path.as_posix(),
            "videos": [v.to_dict() for v in self.videos],
            "classes": [c.to_dict() for c in self.classes],
            "ui_state": self.ui_state.to_dict(),
            "sam2": self.sam2.to_dict(),
        }

    @classmethod
    def from_dict(cls, root: Path, data: Dict[str, Any]) -> "Project":
        frames_root = Path(data.get("frames_root", "frames"))
        annotations_path = Path(data.get("annotations_path", "annotations.json"))
        videos = [VideoItem.from_dict(v) for v in data.get("videos", [])]
        classes = [ClassLabel.from_dict(c) for c in data.get("classes", [])]
        ui_state = UIState.from_dict(data.get("ui_state", {}))
        sam2 = Sam2Settings.from_dict(data.get("sam2", {}))
        return cls(
            root=root,
            name=str(data.get("project_name", root.name)),
            created_at=str(data.get("created_at", _now_iso())),
            last_opened=str(data.get("last_opened", _now_iso())),
            frames_root=frames_root,
            annotations_path=annotations_path,
            videos=videos,
            classes=classes,
            ui_state=ui_state,
            sam2=sam2,
        )

    def resolve_video_frames_dir(self, video: VideoItem) -> Path:
        return _from_rel(video.frames_dir, self.root)

    def set_default_classes(self):
        if not self.classes:
            self.classes = [
                ClassLabel(name="default", color="#00FF00"),
                ClassLabel(name="object", color="#FFD700"),
            ]
