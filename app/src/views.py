# ================================
# File: views.py
# Qt widgets & signals. The view never mutates the model.
# ================================
from typing import List, Tuple, Optional
from pathlib import Path
from PyQt6 import QtGui, QtCore, QtWidgets
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPen, QPixmap, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QSpinBox,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsEllipseItem, QGraphicsRectItem,
    QMessageBox, QSlider, QFileDialog
)


class ImageView(QGraphicsView):
    pointRadius = 4

    # Signals (view-only)
    pointAdded = pyqtSignal(int, int, int)         # x, y, label
    pointRemoved = pyqtSignal(int, int, int)       # x, y, label
    boxSet = pyqtSignal(int, int, int, int)        # x, y, w, h

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
        self.box_item: Optional[QGraphicsRectItem] = None
        self.point_items: List[Tuple[QGraphicsEllipseItem, int]] = []  # (item, label)

    # ------- public view API -------
    def set_image(self, pix: QPixmap):
        self.scene.clear()
        self.point_items.clear()
        self.box_item = None
        self.pix_item = self.scene.addPixmap(pix)
        self.scene.setSceneRect(self.pix_item.boundingRect())
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def set_mode(self, mode: str):
        self.mode = mode

    def add_point_visual(self, x: int, y: int, label: int):
        r = self.pointRadius
        item = self.scene.addEllipse(
            x - r, y - r, 2 * r, 2 * r,
            QPen(Qt.GlobalColor.black, 1),
            QtGui.QBrush(QColor(0, 200, 0) if label == 1 else QColor(255, 0, 0))
        )
        item.setZValue(10)
        item.setCursor(Qt.CursorShape.PointingHandCursor)
        item.setData(0, int(x))
        item.setData(1, int(y))
        item.setData(2, int(label))
        self.point_items.append((item, label))

    def remove_point_visual(self, item: QGraphicsEllipseItem):
        try:
            self.scene.removeItem(item)
        except Exception:
            pass
        self.point_items = [(it, lab) for (it, lab) in self.point_items if it is not item]

    def set_box_visual(self, x: int, y: int, w: int, h: int):
        if self.box_item:
            self.scene.removeItem(self.box_item)
            self.box_item = None
        pen = QPen(QColor(255, 215, 0), 2)
        self.box_item = self.scene.addRect(x, y, w, h, pen)
        self.box_item.setZValue(9)

    # ------- events -------
    def wheelEvent(self, event: QtGui.QWheelEvent):
        self.scale(self.zoom_factor if event.angleDelta().y() > 0 else 1 / self.zoom_factor, 
                   self.zoom_factor if event.angleDelta().y() > 0 else 1 / self.zoom_factor)

    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        if not self.pix_item:
            return
        scene_pos = self.mapToScene(ev.pos())
        x, y = int(scene_pos.x()), int(scene_pos.y())

        if self.mode == "box":
            if ev.button() == Qt.MouseButton.LeftButton:
                self._box_start = scene_pos
                if self.box_item:
                    self.scene.removeItem(self.box_item)
                    self.box_item = None
                pen = QPen(QColor(255, 215, 0), 2, Qt.PenStyle.DashLine)
                self.box_item = self.scene.addRect(x, y, 1, 1, pen)
                self.box_item.setZValue(9)
        else:  # point mode
            if (ev.modifiers() & Qt.KeyboardModifier.ControlModifier):
                for it in self.items(ev.pos()):
                    if isinstance(it, QGraphicsEllipseItem):
                        px = int(it.data(0)) if it.data(0) is not None else x
                        py = int(it.data(1)) if it.data(1) is not None else y
                        plab = int(it.data(2)) if it.data(2) is not None else 1
                        self.pointRemoved.emit(px, py, plab)
                        self.remove_point_visual(it)
                        ev.accept()
                        return
            else:
                if ev.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
                    label = 1 if ev.button() == Qt.MouseButton.LeftButton else 0
                    self.pointAdded.emit(x, y, label)
                    self.add_point_visual(x, y, label)
                    ev.accept()
                    return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent):
        if self.mode == "box" and self._box_start is not None and self.box_item is not None:
            scene_pos = self.mapToScene(ev.pos())
            x0, y0 = self._box_start.x(), self._box_start.y()
            w = scene_pos.x() - x0
            h = scene_pos.y() - y0
            rect = QtCore.QRectF(min(x0, x0 + w), min(y0, y0 + h), abs(w), abs(h))
            self.box_item.setRect(rect)
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent):
        if self.mode == "box" and self._box_start is not None and self.box_item is not None:
            rectf = self.box_item.rect()
            x, y, w, h = int(rectf.x()), int(rectf.y()), int(rectf.width()), int(rectf.height())
            self.scene.removeItem(self.box_item)
            self.box_item = None
            self.set_box_visual(x, y, w, h)
            self.boxSet.emit(x, y, w, h)
            self._box_start = None
        super().mouseReleaseEvent(ev)


class MainWindowView(QMainWindow):
    # top-level view signals
    undoRequested = pyqtSignal()
    redoRequested = pyqtSignal()
    # top-level view signals
    modeChanged = pyqtSignal(str)             # "box" | "point"
    objectIdChanged = pyqtSignal(int)
    exportRequested = pyqtSignal()
    frameChangeRequested = pyqtSignal(int)    # index

    def __init__(self, frames: List[Path]):
        super().__init__()
        self.setWindowTitle("Frame Annotator (MVC)")
        self.frames = frames
        self.idx = 0

        central = QWidget()
        self.setCentralWidget(central)
        v = QVBoxLayout(central)

        # toolbar row
        hb = QHBoxLayout()
        self.btn_prev = QPushButton("← Prev")
        self.btn_next = QPushButton("Next →")
        self.btn_box = QPushButton("1) Box")
        self.btn_point = QPushButton("2) Point (+ L / - R, Ctrl+Click to delete)")
        self.btn_export = QPushButton("4) Export YAML")
        hb.addWidget(self.btn_prev)
        hb.addWidget(self.btn_next)
        hb.addSpacing(12)
        hb.addWidget(self.btn_box)
        hb.addWidget(self.btn_point)
        hb.addSpacing(12)
        hb.addWidget(QLabel("obj_id:"))
        self.obj_spin = QSpinBox()
        self.obj_spin.setRange(1, 9999)
        self.obj_spin.setValue(1)
        hb.addWidget(self.obj_spin)
        hb.addStretch(1)
        hb.addWidget(self.btn_export)
        v.addLayout(hb)

        # image view
        self.view = ImageView()
        v.addWidget(self.view, 1)

        # slider with debounce
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.frames) - 1)
        self.slider.setValue(0)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.setTickInterval(1)

        self._slide_timer = QTimer(self)
        self._slide_timer.setSingleShot(True)
        self._slide_timer.setInterval(120)
        self._pending_slider_index = 0
        self._dragging_slider = False

        self.slider.valueChanged.connect(self._on_slider_value_changed)
        self._slide_timer.timeout.connect(self._on_slider_timeout)
        self.slider.sliderPressed.connect(self._on_slider_pressed)
        self.slider.sliderReleased.connect(self._on_slider_released)

        v.addWidget(self.slider)

        # status bar
        self.status = self.statusBar()
        self.status.showMessage("Use ←/→ or slider. Box / Point. Ctrl+Click a point to delete. Ctrl+Z/Y = Undo/Redo")

        # wire buttons
        self.btn_box.clicked.connect(lambda: self._emit_mode("box"))
        self.btn_point.clicked.connect(lambda: self._emit_mode("point"))
        self.btn_export.clicked.connect(self.exportRequested.emit)
        self.btn_prev.clicked.connect(lambda: self.frameChangeRequested.emit(self.idx - 1))
        self.btn_next.clicked.connect(lambda: self.frameChangeRequested.emit(self.idx + 1))
        self.obj_spin.valueChanged.connect(lambda v: self.objectIdChanged.emit(int(v)))

        # keyboard shortcuts (work while window is active)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, activated=lambda: self.frameChangeRequested.emit(self.idx + 1))
        QShortcut(QKeySequence(Qt.Key.Key_Left),  self, activated=lambda: self.frameChangeRequested.emit(self.idx - 1))
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self._undo_requested)
        QShortcut(QKeySequence("Ctrl+Y"), self, activated=self._redo_requested)
        QShortcut(QKeySequence(Qt.Key.Key_Home),  self, activated=lambda: self.frameChangeRequested.emit(0))
        QShortcut(QKeySequence(Qt.Key.Key_End),   self, activated=lambda: self.frameChangeRequested.emit(len(self.frames) - 1))

        # menu
        file_menu = self.menuBar().addMenu("File")
        act_export = file_menu.addAction("Export YAML…")
        act_export.triggered.connect(self.exportRequested.emit)
        act_open = file_menu.addAction("Open folder…")
        act_open.triggered.connect(self._not_supported_yet)

        edit_menu = self.menuBar().addMenu("Edit")
        act_undo = edit_menu.addAction("Undo")
        act_undo.setShortcut("Ctrl+Z")
        act_undo.triggered.connect(self._undo_requested)
        act_redo = edit_menu.addAction("Redo")
        act_redo.setShortcut("Ctrl+Y")
        act_redo.triggered.connect(self._redo_requested)

    # ------- private helpers -------
    def _not_supported_yet(self):
        QMessageBox.information(self, "Coming soon", "Switching folders at runtime will be added later.")

    def _emit_mode(self, mode: str):
        self.view.set_mode(mode)
        self.modeChanged.emit(mode)
        self.status.showMessage(f"Mode: {mode}")

    def _on_slider_value_changed(self, i: int):
        self._pending_slider_index = i
        self._slide_timer.start()

    def _on_slider_timeout(self):
        self.frameChangeRequested.emit(self._pending_slider_index)

    def _on_slider_pressed(self):
        self._dragging_slider = True

    def _on_slider_released(self):
        self._dragging_slider = False
        self._slide_timer.stop()
        self.frameChangeRequested.emit(self.slider.value())

    def _undo_requested(self):
        self.undoRequested.emit()

    def _redo_requested(self):
        self.redoRequested.emit()

    # ------- public view API used by controller -------
    def ask_export_path(self, default_dir: Path) -> Optional[Path]:
        default_path = str(default_dir / "prompts.yaml")
        path, _ = QFileDialog.getSaveFileName(self, "Export YAML", default_path, "YAML (*.yaml *.yml)")
        return Path(path) if path else None

    def set_title_for_frame(self, idx: int):
        p = self.frames[idx]
        self.setWindowTitle(f"Annotator — {p.name}  [{idx + 1}/{len(self.frames)}]")

    def set_slider(self, idx: int):
        if self.slider.value() != idx:
            self.slider.blockSignals(True)
            self.slider.setValue(idx)
            self.slider.blockSignals(False)

    def set_pixmap(self, pix: QPixmap):
        self.view.set_image(pix)

    def redraw_for_frame(self, entries: List[Tuple[Optional[Tuple[int,int,int,int]], List[Tuple[int,int]], List[int]]]):
        # entries: list of (box, points, labels)
        for box, points, labels in entries:
            if box:
                self.view.set_box_visual(*box)
            for (x, y), lab in zip(points, labels):
                self.view.add_point_visual(x, y, lab)
