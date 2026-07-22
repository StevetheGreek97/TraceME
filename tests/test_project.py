from src.project.store import create_project, load_project, save_project
from src.project.types import ClassLabel, Project, UIState, VideoItem


def test_project_round_trip(tmp_path):
    proj = create_project(tmp_path / "proj", "myproj")
    proj.add_object("cell", "#FF0000")
    proj.videos.append(
        VideoItem(id="vid_1", name="v1", source_path="/x/v1.mp4", frames_dir="frames/v1", frame_count=9, fps=25.0)
    )
    save_project(proj)

    loaded, warning = load_project(proj.config_path)
    assert warning == ""
    assert loaded.name == "myproj"
    assert [c.name for c in loaded.classes] == ["cell"]
    assert loaded.videos[0].fps == 25.0
    assert loaded.next_obj_id == 2
    assert (tmp_path / "proj" / "annotations.json").exists()


def test_legacy_zero_obj_id_classes_dropped(tmp_path):
    data = Project(root=tmp_path, name="p").to_dict()
    data["classes"] = [
        {"obj_id": 0, "name": "legacy", "color": "#00FF00"},
        {"obj_id": 3, "name": "kept", "color": "#0000FF"},
    ]
    data["next_obj_id"] = 1
    proj = Project.from_dict(root=tmp_path, data=data)
    assert [c.name for c in proj.classes] == ["kept"]
    assert proj.next_obj_id == 4  # bumped past the highest surviving obj_id


def test_ui_state_ignores_removed_mode_key():
    st = UIState.from_dict(
        {"last_video_id": "v", "last_frame_index": 3, "mode": "box", "show_only_annotated": True}
    )
    assert st.last_frame_index == 3
    assert st.show_only_annotated is True
    assert not hasattr(st, "mode")
    assert "mode" not in st.to_dict()


def test_schema_mismatch_warns(tmp_path):
    proj = create_project(tmp_path / "proj", "p")
    data = proj.to_dict()
    data["schema_version"] = 999
    proj.config_path.write_text(__import__("json").dumps(data))
    _, warning = load_project(proj.config_path)
    assert "schema" in warning.lower()
