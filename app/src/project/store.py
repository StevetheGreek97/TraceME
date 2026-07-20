from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

from .types import Project, SCHEMA_VERSION, _now_iso


def create_project(root: Path, name: str) -> Project:
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    proj = Project(root=root, name=name)
    proj.set_default_classes()
    proj.frames_root_abs().mkdir(parents=True, exist_ok=True)
    proj.annotations_path_abs().parent.mkdir(parents=True, exist_ok=True)
    save_project(proj)
    if not proj.annotations_path_abs().exists():
        with open(proj.annotations_path_abs(), "w", encoding="utf-8") as f:
            json.dump({"schema_version": 1, "annotations": []}, f, indent=2)
    return proj


def load_project(config_path: Path) -> Tuple[Project, str]:
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(str(config_path))

    root = config_path.parent
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    schema = int(data.get("schema_version", 0))
    if schema != SCHEMA_VERSION:
        # allow forward compatibility by loading anyway, but return warning
        warning = f"Project schema {schema} differs from app schema {SCHEMA_VERSION}."
    else:
        warning = ""

    proj = Project.from_dict(root=root, data=data)
    proj.last_opened = _now_iso()
    proj.set_default_classes()
    return proj, warning


def save_project(project: Project) -> None:
    project.last_opened = _now_iso()
    payload = project.to_dict()
    with open(project.config_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
