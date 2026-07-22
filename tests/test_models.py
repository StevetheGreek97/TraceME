from pathlib import Path

import pytest

from src.models import (
    AnnotationModel,
    ObjectAnno,
    export_yaml,
    load_annotations,
    save_annotations,
)


def make_frames(tmp_path: Path, names) -> Path:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for n in names:
        (frames_dir / n).write_bytes(b"\xff\xd8\xff\xd9")
    return frames_dir


def test_editing_prompts_clears_polygon():
    obj = ObjectAnno()
    obj.set_polygon([(0, 0), (10, 0), (10, 10)])
    obj.add_point(5, 5, 1)
    assert obj.polygon is None

    obj.set_polygon([(0, 0), (10, 0), (10, 10)])
    obj.set_box(1, 2, 3, 4)
    assert obj.polygon is None

    obj.set_polygon([(0, 0), (10, 0), (10, 10)])
    assert obj.remove_point(5, 5, 1)
    assert obj.polygon is None
    assert not obj.remove_point(99, 99, 1)


def test_frames_sorted_numerically(tmp_path):
    frames_dir = make_frames(tmp_path, ["10.jpg", "2.jpg", "1.jpg", "frame_00003.jpg"])
    model = AnnotationModel(frames_dir)
    assert [p.name for p in model.frames] == ["1.jpg", "2.jpg", "frame_00003.jpg", "10.jpg"]


def test_records_round_trip(tmp_path):
    frames_dir = make_frames(tmp_path, ["00000.jpg", "00001.jpg"])
    model = AnnotationModel(frames_dir)
    model.add_point(1, 10, 20, 1)
    model.add_point(1, 30, 40, 0)
    model.set_box(2, 5, 6, 7, 8)
    model.set_polygon_at(1, 1, [(0, 0), (4, 0), (4, 4)])
    # an object with no usable prompts must not be exported
    model.set_box(3, 0, 0, 0, 0)

    records = model.to_records("vid_a")
    keys = {(r["frame_idx"], r["obj_id"]) for r in records}
    assert keys == {(0, 1), (0, 2), (1, 1)}
    by_key = {(r["frame_idx"], r["obj_id"]): r for r in records}
    assert by_key[(0, 1)]["points"] == [[10, 20], [30, 40]]
    assert by_key[(0, 1)]["labels"] == [1, 0]
    assert by_key[(0, 2)]["box"] == [5, 6, 7, 8]
    assert by_key[(1, 1)]["polygon"] == [[0, 0], [4, 0], [4, 4]]

    clone = AnnotationModel(frames_dir)
    clone.load_records(records)
    assert clone.to_records("vid_a") == records


def test_set_polygon_at_ignores_current_index(tmp_path):
    frames_dir = make_frames(tmp_path, ["00000.jpg", "00001.jpg", "00002.jpg"])
    model = AnnotationModel(frames_dir)
    model.set_index(2)
    model.set_polygon_at(0, 7, [(0, 0), (2, 0), (2, 2)])
    assert model.get_object(0, 7).polygon == [(0, 0), (2, 0), (2, 2)]
    assert model.get_object(2, 7) is None


def test_is_frame_annotated(tmp_path):
    frames_dir = make_frames(tmp_path, ["00000.jpg"])
    model = AnnotationModel(frames_dir)
    assert not model.is_frame_annotated(0)
    model.set_box(1, 0, 0, 0, 0)  # zero-size box does not count
    assert not model.is_frame_annotated(0)
    model.add_point(1, 3, 3, 1)
    assert model.is_frame_annotated(0)


def test_save_load_annotations_file(tmp_path):
    frames_dir = make_frames(tmp_path, ["00000.jpg"])
    model = AnnotationModel(frames_dir)
    model.add_point(1, 1, 2, 1)
    path = tmp_path / "annotations.json"
    save_annotations(path, {"vid_a": model})
    assert not model.dirty

    by_video = load_annotations(path)
    assert set(by_video) == {"vid_a"}
    assert by_video["vid_a"][0]["points"] == [[1, 2]]
    assert load_annotations(tmp_path / "missing.json") == {}


def test_export_yaml_shape(tmp_path):
    yaml = pytest.importorskip("yaml")
    frames_dir = make_frames(tmp_path, ["00000.jpg"])
    model = AnnotationModel(frames_dir)
    model.add_point(1, 10, 20, 1)
    model.set_box(1, 5, 6, 7, 8)

    out = tmp_path / "video.yaml"
    ok, msg = export_yaml(out, {"vid_a": model})
    assert ok, msg
    data = yaml.safe_load(out.read_text())
    assert list(data) == ["prompts"]
    (entry,) = data["prompts"]
    assert entry == {
        "frame_idx": 0,
        "obj_id": 1,
        "points": [[10, 20]],
        "labels": [1],
        "box": [5, 6, 7, 8],
    }
