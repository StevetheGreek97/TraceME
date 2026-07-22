import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.services.sam2_service import Sam2PredictThread


class FakeService:
    """Stands in for Sam2Service; returns a fixed polygon."""

    available = True
    error = ""

    def __init__(self, polygon=None, exc=None):
        self.polygon = polygon or [(0, 0), (10, 0), (10, 10)]
        self.exc = exc
        self.calls = []

    def generate_polygon(self, image_rgb, points_xy, point_labels, box_xywh):
        if self.exc:
            raise self.exc
        self.calls.append((image_rgb.shape, points_xy, point_labels, box_xywh))
        return self.polygon


@pytest.fixture
def frame_path(tmp_path):
    path = tmp_path / "frame.jpg"
    cv2.imwrite(str(path), np.zeros((32, 32, 3), dtype=np.uint8))
    return path


def run_thread(qapp, thread):
    results, errors = [], []
    thread.done.connect(results.append)
    thread.failed.connect(errors.append)
    thread.start()
    assert thread.wait(10000)
    qapp.processEvents()
    return results, errors


def test_predict_thread_success(qapp, frame_path):
    svc = FakeService()
    t = Sam2PredictThread(svc, frame_path, [(1, 2)], [1], [0, 0, 5, 5], "vid_a", 3, 7)
    results, errors = run_thread(qapp, t)
    assert errors == []
    (res,) = results
    assert res["video_id"] == "vid_a"
    assert res["frame_idx"] == 3
    assert res["obj_id"] == 7
    assert res["points"] == [(1, 2)]
    assert res["labels"] == [1]
    assert res["box"] == [0, 0, 5, 5]
    assert res["polygon"] == svc.polygon
    # image decoded and converted before reaching the service
    assert svc.calls[0][0] == (32, 32, 3)


def test_predict_thread_missing_image(qapp, tmp_path):
    t = Sam2PredictThread(FakeService(), tmp_path / "nope.jpg", [], [], None, "v", 0, 1)
    results, errors = run_thread(qapp, t)
    assert results == []
    assert len(errors) == 1 and "Failed to read" in errors[0]


def test_predict_thread_exception_reported(qapp, frame_path):
    t = Sam2PredictThread(FakeService(exc=RuntimeError("boom")), frame_path, [], [], None, "v", 0, 1)
    results, errors = run_thread(qapp, t)
    assert results == []
    assert errors == ["boom"]
