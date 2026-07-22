import pytest
from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt6.QtGui import QMouseEvent, QPixmap, QWheelEvent

from src.ui.image_view import ImageView


@pytest.fixture
def view(qapp):
    v = ImageView()
    v.resize(400, 300)
    v.show()
    pix = QPixmap(800, 600)
    pix.fill(Qt.GlobalColor.gray)
    v.set_image(pix)
    yield v
    v.close()


def wheel(view, delta):
    ev = QWheelEvent(
        QPointF(200, 150),
        QPointF(200, 150),
        QPoint(0, 0),
        QPoint(0, delta),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    view.wheelEvent(ev)


def mouse(etype, btn, pos):
    return QMouseEvent(etype, QPointF(*pos), QPointF(*pos), btn, btn, Qt.KeyboardModifier.NoModifier)


def test_zoom_kept_across_same_size_frames(view):
    fit_scale = view.transform().m11()
    wheel(view, 120)
    wheel(view, 120)
    assert view._user_zoomed
    zoomed = view.transform().m11()
    assert zoomed > fit_scale

    pix2 = QPixmap(800, 600)
    pix2.fill(Qt.GlobalColor.darkGray)
    view.set_image(pix2)
    assert view._user_zoomed
    assert view.transform().m11() == pytest.approx(zoomed)


def test_refit_on_different_size_image(view):
    wheel(view, 120)
    pix = QPixmap(400, 400)
    pix.fill(Qt.GlobalColor.white)
    view.set_image(pix)
    assert not view._user_zoomed


def test_zoom_clamped(view):
    for _ in range(200):
        wheel(view, -120)
    assert view.transform().m11() >= ImageView.MIN_SCALE * 0.999
    for _ in range(200):
        wheel(view, 120)
    assert view.transform().m11() <= ImageView.MAX_SCALE * 1.001


def test_keyboard_zoom(view):
    fit_scale = view.transform().m11()
    view.zoom_in()
    assert view._user_zoomed
    assert view.transform().m11() == pytest.approx(fit_scale * view.zoom_factor)
    view.zoom_out()
    assert view.transform().m11() == pytest.approx(fit_scale)
    view.fit_to_view()
    assert not view._user_zoomed


def test_middle_drag_pans_without_adding_points(view):
    from PyQt6.QtWidgets import QApplication

    view.fit_to_view()
    for _ in range(4):
        wheel(view, 120)
    assert view.horizontalScrollBar().maximum() > 0

    h0 = view.horizontalScrollBar().value()
    view.mousePressEvent(mouse(QEvent.Type.MouseButtonPress, Qt.MouseButton.MiddleButton, (200, 150)))
    assert view._pan_last is not None
    # pan uses the app override cursor, never the view/viewport cursor, so
    # QGraphicsView's item-cursor caching can't resurrect the closed hand later
    assert QApplication.overrideCursor() is not None
    assert QApplication.overrideCursor().shape() == Qt.CursorShape.ClosedHandCursor
    view.mouseMoveEvent(mouse(QEvent.Type.MouseMove, Qt.MouseButton.MiddleButton, (150, 150)))
    view.mouseReleaseEvent(mouse(QEvent.Type.MouseButtonRelease, Qt.MouseButton.MiddleButton, (150, 150)))
    assert view._pan_last is None
    assert QApplication.overrideCursor() is None
    assert view.horizontalScrollBar().value() == h0 + 50
    assert view.point_items == []
