from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from PyQt6 import QtCore, QtGui
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QPen, QColor, QPolygonF, QBrush
from PyQt6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsPolygonItem,
)


_BASE_COLORS = [
    (255, 215, 0),
    (0, 170, 255),
    (255, 120, 0),
    (170, 255, 0),
    (255, 0, 200),
    (0, 255, 170),
    (255, 80, 80),
    (160, 100, 255),
]


def color_for_obj(obj_id: int) -> QColor:
    r, g, b = _BASE_COLORS[(max(0, obj_id - 1)) % len(_BASE_COLORS)]
    return QColor(r, g, b)


class ImageView(QGraphicsView):
    pointRadius = 4

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHints(
            QtGui.QPainter.RenderHint.Antialiasing
            | QtGui.QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setMouseTracking(True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.zoom_factor = 1.25
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.pix_item: Optional[QGraphicsPixmapItem] = None

        self.mode = "box"  # "box" | "point"
        self._box_start: Optional[QtCore.QPointF] = None
        self._temp_box_item: Optional[QGraphicsRectItem] = None
        self._box_override_active = False

        self.box_items_by_obj: Dict[int, QGraphicsRectItem] = {}
        self.polygon_items_by_obj: Dict[int, QGraphicsPolygonItem] = {}
        self.point_items: List[Tuple[QGraphicsEllipseItem, int]] = []

        self.on_add_point = None
        self.on_set_box = None
        self.on_remove_point = None
        self.get_current_obj_id = None

    def set_image(self, pix: QPixmap):
        self.scene.clear()
        self.point_items.clear()
        self.polygon_items_by_obj.clear()
        self.box_items_by_obj.clear()
        self._temp_box_item = None
        self.pix_item = self.scene.addPixmap(pix)
        self.scene.setSceneRect(self.pix_item.boundingRect())
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def set_mode(self, mode: str):
        self.mode = mode

    def _start_box(self, scene_pos: QtCore.QPointF):
        x, y = int(scene_pos.x()), int(scene_pos.y())
        self._box_start = scene_pos
        if self._temp_box_item:
            try:
                self.scene.removeItem(self._temp_box_item)
            except Exception:
                pass
            self._temp_box_item = None
        pen = QPen(QColor(255, 215, 0), 2, Qt.PenStyle.DashLine)
        rect_item = self.scene.addRect(x, y, 1, 1, pen)
        rect_item.setZValue(9)
        self._temp_box_item = rect_item

    def wheelEvent(self, event: QtGui.QWheelEvent):
        zoom = self.zoom_factor if event.angleDelta().y() > 0 else 1 / self.zoom_factor
        self.scale(zoom, zoom)

    # ----- points -----
    def add_point_visual(self, x: int, y: int, label: int, obj_id: int = 1):
        r = self.pointRadius
        fill = QColor(0, 200, 0) if label == 1 else QColor(255, 0, 0)
        ring = color_for_obj(obj_id)
        pen = QPen(ring, 2)
        item = self.scene.addEllipse(
            x - r, y - r, 2 * r, 2 * r,
            pen,
            QBrush(fill)
        )
        item.setZValue(10)
        item.setCursor(Qt.CursorShape.PointingHandCursor)
        item.setData(0, int(x))
        item.setData(1, int(y))
        item.setData(2, int(label))
        item.setData(3, int(obj_id))
        self.point_items.append((item, label))

    def remove_point_visual(self, item: QGraphicsEllipseItem):
        try:
            self.scene.removeItem(item)
        except Exception:
            pass
        self.point_items = [(it, lab) for (it, lab) in self.point_items if it is not item]

    def remove_points_for_obj(self, obj_id: int):
        to_remove = [it for (it, _) in self.point_items if int(it.data(3) or 0) == int(obj_id)]
        for it in to_remove:
            try:
                self.scene.removeItem(it)
            except Exception:
                pass
        self.point_items = [(it, lab) for (it, lab) in self.point_items if int(it.data(3) or 0) != int(obj_id)]

    # ----- boxes -----
    def set_box_visual(self, x: int, y: int, w: int, h: int, obj_id: int = 1):
        prev = self.box_items_by_obj.pop(int(obj_id), None)
        if prev is not None:
            try:
                self.scene.removeItem(prev)
            except Exception:
                pass
        color = color_for_obj(obj_id)
        pen = QPen(color, 2)
        box_item = self.scene.addRect(x, y, w, h, pen)
        box_item.setZValue(9)
        self.box_items_by_obj[int(obj_id)] = box_item

    def remove_box_for_obj(self, obj_id: int):
        prev = self.box_items_by_obj.pop(int(obj_id), None)
        if prev is not None:
            try:
                self.scene.removeItem(prev)
            except Exception:
                pass

    # ----- polygons -----
    def add_polygon_visual(self, polygon_points: List[Tuple[int, int]], obj_id: int = 1):
        if not polygon_points:
            return
        self.remove_polygon_for_obj(obj_id)
        color = color_for_obj(obj_id)
        poly = QPolygonF([QtCore.QPointF(int(x), int(y)) for x, y in polygon_points])
        pen = QPen(color)
        pen.setWidth(2)
        item = QGraphicsPolygonItem(poly)
        item.setPen(pen)
        item.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 60)))
        item.setZValue(8)
        self.scene.addItem(item)
        self.polygon_items_by_obj[int(obj_id)] = item

    def remove_polygon_for_obj(self, obj_id: int):
        prev = self.polygon_items_by_obj.pop(int(obj_id), None)
        if prev is not None:
            try:
                self.scene.removeItem(prev)
            except Exception:
                pass

    def remove_all_for_obj(self, obj_id: int):
        self.remove_points_for_obj(obj_id)
        self.remove_box_for_obj(obj_id)
        self.remove_polygon_for_obj(obj_id)

    # ---------- mouse handling ----------
    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        if not self.pix_item:
            return
        scene_pos = self.mapToScene(ev.pos())
        x, y = int(scene_pos.x()), int(scene_pos.y())

        if (ev.modifiers() & Qt.KeyboardModifier.ControlModifier):
            for it in self.items(ev.pos()):
                if isinstance(it, QGraphicsEllipseItem):
                    px = int(it.data(0)) if it.data(0) is not None else x
                    py = int(it.data(1)) if it.data(1) is not None else y
                    plab = int(it.data(2)) if it.data(2) is not None else 1
                    if self.on_remove_point:
                        self.on_remove_point(px, py, plab)
                    self.remove_point_visual(it)
                    ev.accept()
                    return

        if ev.button() == Qt.MouseButton.LeftButton and (ev.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self._box_override_active = True
            self._start_box(scene_pos)
            ev.accept()
            return

        if ev.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            label = 1 if ev.button() == Qt.MouseButton.LeftButton else 0
            if self.on_add_point:
                self.on_add_point(x, y, label)
            curr_obj = self.get_current_obj_id() if callable(self.get_current_obj_id) else 1
            self.add_point_visual(x, y, label, obj_id=int(curr_obj))
            ev.accept()
            return

        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent):
        if self._box_start is not None and self._temp_box_item is not None:
            scene_pos = self.mapToScene(ev.pos())
            x0, y0 = self._box_start.x(), self._box_start.y()
            w = scene_pos.x() - x0
            h = scene_pos.y() - y0
            rect = QtCore.QRectF(min(x0, x0 + w), min(y0, y0 + h), abs(w), abs(h))
            self._temp_box_item.setRect(rect)
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent):
        if self._box_start is not None and self._temp_box_item is not None:
            rectf = self._temp_box_item.rect()
            x, y, w, h = int(rectf.x()), int(rectf.y()), int(rectf.width()), int(rectf.height())
            try:
                self.scene.removeItem(self._temp_box_item)
            except Exception:
                pass
            self._temp_box_item = None
            min_drag = 3
            if self._box_override_active and w < min_drag and h < min_drag:
                scene_pos = self.mapToScene(ev.pos())
                px, py = int(scene_pos.x()), int(scene_pos.y())
                if self.on_add_point:
                    self.on_add_point(px, py, 1)
                curr_obj = self.get_current_obj_id() if callable(self.get_current_obj_id) else 1
                self.add_point_visual(px, py, 1, obj_id=int(curr_obj))
            else:
                if self.on_set_box:
                    self.on_set_box(x, y, w, h)
            self._box_start = None
            self._box_override_active = False
        super().mouseReleaseEvent(ev)
