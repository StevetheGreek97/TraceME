from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Dict, List, Optional

try:
    import cv2
except Exception:
    cv2 = None
from PyQt6 import QtGui
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QKeySequence, QShortcut, QPixmap, QColor, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QColorDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSlider,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QSplitter,
    QInputDialog,
    QDialog,
    QDialogButtonBox,
    QPlainTextEdit,
)

from src.models import AnnotationModel, load_annotations, save_annotations, export_yaml
from src.project.store import create_project, load_project, save_project
from src.project.types import ClassLabel, Project, VideoItem
from src.services.sam2_service import Sam2Service
from src.services.video_importer import VideoImportThread, VideoImportResult, has_ffmpeg, VIDEO_EXTS
from src.ui.image_view import BASE_COLORS, ImageView


class MainWindow(QMainWindow):
    AUTOSAVE_MS = 800

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TraceWave")

        self.project: Optional[Project] = None
        self.models: Dict[str, AnnotationModel] = {}
        self.current_video_id: Optional[str] = None
        self.sam2: Optional[Sam2Service] = None
        self._import_thread: Optional[VideoImportThread] = None

        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(self.AUTOSAVE_MS)
        self._autosave_timer.timeout.connect(self.save_current_project)

        self._build_ui()
        self._build_menu()
        self._wire_shortcuts()
        self._set_project_enabled(False)

    # ---------------- UI ----------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # LEFT: project library
        left = QWidget()
        left_v = QVBoxLayout(left)
        self.project_label = QLabel("Project: (none)")
        left_v.addWidget(self.project_label)

        self.video_tree = QTreeWidget()
        self.video_tree.setHeaderHidden(True)
        self.video_tree.itemSelectionChanged.connect(self._on_tree_selection_changed)
        left_v.addWidget(self.video_tree, 1)

        left_v.addWidget(QLabel("Objects"))
        self.objects_list = QListWidget()
        self.objects_list.itemSelectionChanged.connect(self._on_object_selection_changed)
        left_v.addWidget(self.objects_list, 1)

        objects_btn_row = QHBoxLayout()
        self.btn_add_object = QPushButton("+ Add")
        self.btn_rename_object = QPushButton("Rename")
        self.btn_recolor_object = QPushButton("Recolor")
        self.btn_delete_object = QPushButton("Delete")
        for b in (self.btn_add_object, self.btn_rename_object, self.btn_recolor_object, self.btn_delete_object):
            objects_btn_row.addWidget(b)
        left_v.addLayout(objects_btn_row)

        self.btn_add_object.clicked.connect(self.action_add_object)
        self.btn_rename_object.clicked.connect(self.action_rename_object)
        self.btn_recolor_object.clicked.connect(self.action_recolor_object)
        self.btn_delete_object.clicked.connect(self.action_delete_object)

        splitter.addWidget(left)

        # RIGHT: controls + view
        right = QWidget()
        right_v = QVBoxLayout(right)

        # content splitter: frames list + image view
        self.content_splitter = QSplitter(Qt.Orientation.Horizontal)

        frames_panel = QWidget()
        frames_v = QVBoxLayout(frames_panel)
        self.list_label = QLabel("Frames")
        frames_v.addWidget(self.list_label)
        self.frame_list = QListWidget()
        self.frame_list.itemClicked.connect(self._on_frame_list_clicked)
        frames_v.addWidget(self.frame_list, 1)

        self.btn_toggle_annotated = QPushButton("Show only annotated")
        self.btn_toggle_annotated.setCheckable(True)
        self.btn_toggle_annotated.toggled.connect(self.update_frame_list)
        frames_v.addWidget(self.btn_toggle_annotated)

        self.content_splitter.addWidget(frames_panel)

        image_panel = QWidget()
        image_v = QVBoxLayout(image_panel)
        self.view = ImageView()
        image_v.addWidget(self.view, 1)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setValue(0)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.setTickInterval(1)
        image_v.addWidget(self.slider)

        self.content_splitter.addWidget(image_panel)
        self.content_splitter.setStretchFactor(1, 1)
        right_v.addWidget(self.content_splitter, 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

        # status
        self.status = self.statusBar()
        self.status.showMessage("Create or open a project to begin.")
        self.import_progress = QProgressBar()
        self.import_progress.setVisible(False)
        self.import_progress.setMaximumHeight(12)
        self.status.addPermanentWidget(self.import_progress)

        # empty panel
        self.empty_panel = QWidget()
        emp_layout = QVBoxLayout(self.empty_panel)
        emp_layout.addStretch(1)
        lbl = QLabel("No project loaded")
        lbl.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        lab2 = QLabel("Tip: Open a project to resume where you left off.")
        lab2.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        emp_layout.addWidget(lbl)
        emp_layout.addWidget(lab2)
        emp_layout.addStretch(2)
        right_v.addWidget(self.empty_panel)
        self.empty_panel.hide()

        # wire UI
        self.view.on_add_point = self.add_point
        self.view.on_set_box = self.set_box
        self.view.on_remove_point = self.remove_point
        self.view.get_current_obj_id = lambda: self.current_obj_id()
        self.view.get_object_color = self._object_color
        self.view.get_object_name = self._object_name

        self.slider.valueChanged.connect(self._on_slider_value_changed)

    def _build_menu(self):
        file_menu = self.menuBar().addMenu("File")
        self.act_new_project = file_menu.addAction("New Project...")
        self.act_open_project = file_menu.addAction("Open Project...")
        self.act_save_project = file_menu.addAction("Save Project")
        file_menu.addSeparator()
        self.act_import_videos = file_menu.addAction("Import Videos...")
        self.act_export_yaml = file_menu.addAction("Export YAMLs")

        self.act_new_project.triggered.connect(self.action_new_project)
        self.act_open_project.triggered.connect(self.action_open_project)
        self.act_save_project.triggered.connect(self.save_current_project)
        self.act_import_videos.triggered.connect(self.action_import_videos)
        self.act_export_yaml.triggered.connect(self.action_export_yaml)

        help_menu = self.menuBar().addMenu("Help")
        self.act_help = help_menu.addAction("How to Use TraceWave")
        self.act_help.triggered.connect(self.show_help)

    def _wire_shortcuts(self):
        self._shortcuts = []

        def add_shortcut(seq: str, handler):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.activated.connect(handler)
            self._shortcuts.append(sc)

        add_shortcut("Right", lambda: self.load_frame(self.current_index() + 1))
        add_shortcut("Left", lambda: self.load_frame(self.current_index() - 1))
        add_shortcut("Home", lambda: self.load_frame(0))
        add_shortcut("End", lambda: self.load_frame(self.current_frame_count() - 1))
        add_shortcut("E", self.run_sam2)
        add_shortcut("D", self.clear_mask_for_current)

    # ---------------- Project lifecycle ----------------
    def action_new_project(self):
        base_dir = QFileDialog.getExistingDirectory(self, "Choose Project Location", str(Path.cwd()))
        if not base_dir:
            return
        name, ok = QInputDialog.getText(self, "Project Name", "Enter project name:")
        if not ok or not name:
            return
        root = Path(base_dir) / name
        if root.exists() and any(root.iterdir()):
            resp = QMessageBox.question(
                self,
                "Folder not empty",
                f"'{root}' is not empty. Use this folder anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if resp != QMessageBox.StandardButton.Yes:
                return
        project = create_project(root, name)
        self.load_project(project)

    def action_open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open project.json", str(Path.cwd()), "JSON (*.json)")
        if not path:
            return
        try:
            project, warning = load_project(Path(path))
        except Exception as e:
            QMessageBox.critical(self, "Failed to open project", str(e))
            return
        if warning:
            QMessageBox.warning(self, "Project version", warning)
        self.load_project(project)

    def load_project(self, project: Project):
        self.project = project
        self.project_label.setText(f"Project: {project.name}")
        project.frames_root_abs().mkdir(parents=True, exist_ok=True)

        self.sam2 = Sam2Service(
            device="cuda",
            config_name=project.sam2.config_name,
            weights_path=project.sam2.weights_path,
        )
        if not self.sam2.available:
            self.status.showMessage(f"SAM2 disabled: {self.sam2.error}")

        # load annotations
        records_by_video = load_annotations(project.annotations_path_abs())
        self.models.clear()
        for vid in project.videos:
            frames_dir = project.resolve_video_frames_dir(vid)
            if not frames_dir.exists():
                self.status.showMessage(f"Missing frames folder: {frames_dir}")
            model = AnnotationModel(frames_dir)
            if vid.id in records_by_video:
                model.load_records(records_by_video[vid.id])
            self.models[vid.id] = model

        self._migrate_legacy_objects()
        self._refresh_video_tree(select_id=project.ui_state.last_video_id)
        self._set_project_enabled(True)
        self.apply_ui_state()

    def _migrate_legacy_objects(self):
        """Backfill an object registry entry for any obj_id already used in
        annotations but missing from project.classes (e.g. projects saved
        before the Objects panel existed)."""
        if not self.project:
            return
        used_ids = set()
        for model in self.models.values():
            for fr in model.ann.values():
                used_ids.update(fr.objects.keys())
        known_ids = {c.obj_id for c in self.project.classes}
        missing = sorted(used_ids - known_ids)
        if not missing:
            return
        for oid in missing:
            self.project.classes.append(ClassLabel(obj_id=oid, name=f"Object {oid}", color=self._next_palette_color()))
        self.project.next_obj_id = max(self.project.next_obj_id, max(missing) + 1)

    def save_current_project(self):
        if not self.project:
            return
        self.update_ui_state()
        save_project(self.project)
        save_annotations(self.project.annotations_path_abs(), self.models)
        self.status.showMessage("Project saved.")

    def update_ui_state(self):
        if not self.project:
            return
        state = self.project.ui_state
        state.last_video_id = self.current_video_id
        state.last_frame_index = self.current_index()
        state.mode = self.view.mode
        state.show_only_annotated = self.btn_toggle_annotated.isChecked()
        oid = self.current_obj_id()
        if oid is not None:
            state.last_obj_id = oid

    def apply_ui_state(self):
        if not self.project:
            return
        state = self.project.ui_state
        self._refresh_objects_list(select_obj_id=state.last_obj_id)
        self.set_mode(state.mode or "box")
        self.btn_toggle_annotated.setChecked(state.show_only_annotated)

        if state.last_video_id and state.last_video_id in self.models:
            self.load_video(state.last_video_id)
            self.load_frame(state.last_frame_index)
        elif self.project.videos:
            self.load_video(self.project.videos[0].id)

    # ---------------- Video import ----------------
    def action_import_videos(self):
        if not self.project:
            QMessageBox.information(self, "No project", "Create or open a project first.")
            return
        if not has_ffmpeg():
            QMessageBox.critical(self, "FFmpeg not found", "FFmpeg is required to extract frames.")
            return

        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Videos",
            str(Path.cwd()),
            "Videos (*.mp4 *.mov *.avi *.mkv *.m4v *.mpeg *.mpg *.wmv)",
        )
        if not paths:
            return
        videos = [Path(p) for p in paths if Path(p).suffix.lower() in VIDEO_EXTS]
        if not videos:
            QMessageBox.information(self, "No videos", "No supported video files were selected.")
            return

        self._import_thread = VideoImportThread(videos, self.project.frames_root_abs(), quality=2, threads=4, parent=self)
        self._import_thread.progress.connect(self._on_import_progress)
        self._import_thread.error.connect(self._on_import_error)
        self._import_thread.finished.connect(self._on_import_finished)
        self._import_thread.start()

        self.import_progress.setVisible(True)
        self.import_progress.setRange(0, len(videos))
        self.import_progress.setValue(0)
        self.act_import_videos.setEnabled(False)

    def _on_import_progress(self, message: str, current: int, total: int):
        if total > 0:
            self.import_progress.setMaximum(total)
            self.import_progress.setValue(current)
        self.status.showMessage(message)

    def _on_import_error(self, message: str):
        self.status.showMessage(f"Import error: {message}")

    def _on_import_finished(self, results: List[VideoImportResult]):
        self.import_progress.setVisible(False)
        self.act_import_videos.setEnabled(True)
        self._import_thread = None

        if not self.project:
            return

        last_vid_id: Optional[str] = None
        for res in results:
            vid_id = f"vid_{uuid.uuid4().hex[:8]}"
            name = res.frames_dir.name
            frames_rel = res.frames_dir.resolve().relative_to(self.project.root.resolve()).as_posix()
            item = VideoItem(
                id=vid_id,
                name=name,
                source_path=str(res.source_path),
                frames_dir=frames_rel,
                frame_count=res.frame_count,
                fps=res.fps,
            )
            self.project.videos.append(item)
            self.models[vid_id] = AnnotationModel(res.frames_dir)
            last_vid_id = vid_id

        self._refresh_video_tree(select_id=last_vid_id)
        self.save_current_project()

    # ---------------- UI helpers ----------------
    def _set_project_enabled(self, enabled: bool):
        for w in [
            self.objects_list,
            self.btn_add_object,
            self.frame_list,
            self.slider,
            self.btn_toggle_annotated,
        ]:
            w.setEnabled(enabled)
        if not enabled:
            self.btn_rename_object.setEnabled(False)
            self.btn_recolor_object.setEnabled(False)
            self.btn_delete_object.setEnabled(False)
        self.content_splitter.setVisible(enabled)
        self.empty_panel.setVisible(not enabled)
        self.act_save_project.setEnabled(enabled)
        self.act_import_videos.setEnabled(enabled)
        self.act_export_yaml.setEnabled(enabled)

    def _refresh_video_tree(self, select_id: Optional[str] = None):
        self.video_tree.blockSignals(True)
        self.video_tree.clear()

        if not self.project:
            self.video_tree.blockSignals(False)
            return

        root_item = QTreeWidgetItem([self.project.name])
        root_item.setFlags(root_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        self.video_tree.addTopLevelItem(root_item)
        root_item.setExpanded(True)

        for vid in self.project.videos:
            item = QTreeWidgetItem([vid.name])
            item.setData(0, Qt.ItemDataRole.UserRole, vid.id)
            root_item.addChild(item)

        self.video_tree.blockSignals(False)

        if select_id:
            self._select_video_by_id(select_id)
        elif self.project.videos:
            self._select_video_by_id(self.project.videos[0].id)

    def _select_video_by_id(self, vid_id: str):
        root = self.video_tree.topLevelItem(0)
        if not root:
            return
        for i in range(root.childCount()):
            child = root.child(i)
            if child.data(0, Qt.ItemDataRole.UserRole) == vid_id:
                self.video_tree.setCurrentItem(child)
                return

    # ---------------- Frame + annotations ----------------
    def current_model(self) -> Optional[AnnotationModel]:
        if not self.current_video_id:
            return None
        return self.models.get(self.current_video_id)

    def current_index(self) -> int:
        model = self.current_model()
        return model.index if model else 0

    def current_frame_count(self) -> int:
        model = self.current_model()
        return len(model.frames) if model else 0

    def current_obj_id(self) -> Optional[int]:
        items = self.objects_list.selectedItems()
        if not items:
            return None
        return int(items[0].data(Qt.ItemDataRole.UserRole))

    def _object_color(self, obj_id: int) -> Optional[QColor]:
        if not self.project:
            return None
        obj = self.project.get_object(obj_id)
        return QColor(obj.color) if obj else None

    def _object_name(self, obj_id: int) -> str:
        if not self.project:
            return f"Object {obj_id}"
        obj = self.project.get_object(obj_id)
        return obj.name if obj else f"Object {obj_id}"

    def _swatch_icon(self, color: QColor) -> QIcon:
        pix = QPixmap(14, 14)
        pix.fill(color)
        return QIcon(pix)

    def _next_palette_color(self) -> str:
        if not self.project:
            r, g, b = BASE_COLORS[0]
            return f"#{r:02X}{g:02X}{b:02X}"
        used = {c.color.strip().lower() for c in self.project.classes}
        n = len(BASE_COLORS)
        start = len(self.project.classes) % n
        for i in range(n):
            r, g, b = BASE_COLORS[(start + i) % n]
            hexs = f"#{r:02X}{g:02X}{b:02X}"
            if hexs.lower() not in used:
                return hexs
        r, g, b = BASE_COLORS[start % n]
        return f"#{r:02X}{g:02X}{b:02X}"

    def _refresh_objects_list(self, select_obj_id: Optional[int] = None):
        self.objects_list.blockSignals(True)
        self.objects_list.clear()
        if self.project:
            for c in self.project.classes:
                item = QListWidgetItem(c.name)
                item.setIcon(self._swatch_icon(QColor(c.color)))
                item.setData(Qt.ItemDataRole.UserRole, c.obj_id)
                self.objects_list.addItem(item)
        self.objects_list.blockSignals(False)
        selected = self._select_object_by_id(select_obj_id) if select_obj_id is not None else False
        if not selected and self.objects_list.count():
            self.objects_list.setCurrentRow(0)
        self._on_object_selection_changed()

    def _select_object_by_id(self, obj_id: int) -> bool:
        for row in range(self.objects_list.count()):
            item = self.objects_list.item(row)
            if int(item.data(Qt.ItemDataRole.UserRole)) == int(obj_id):
                self.objects_list.setCurrentItem(item)
                return True
        return False

    def _on_object_selection_changed(self):
        has_selection = self.current_obj_id() is not None
        self.btn_rename_object.setEnabled(has_selection)
        self.btn_recolor_object.setEnabled(has_selection)
        self.btn_delete_object.setEnabled(has_selection)

    # ---------------- Object management ----------------
    def action_add_object(self):
        if not self.project:
            return
        name, ok = QInputDialog.getText(self, "New Object", "Object name:")
        if not ok:
            return
        name = name.strip()
        if not name:
            QMessageBox.warning(self, "Invalid name", "Object name cannot be empty.")
            return
        if self.project.object_name_exists(name):
            QMessageBox.warning(self, "Duplicate name", f"An object named '{name}' already exists.")
            return
        obj = self.project.add_object(name, self._next_palette_color())
        self._refresh_objects_list(select_obj_id=obj.obj_id)
        self.save_current_project()
        self.status.showMessage(f"Added object '{name}'.")

    def action_rename_object(self):
        if not self.project:
            return
        oid = self.current_obj_id()
        if oid is None:
            return
        obj = self.project.get_object(oid)
        if not obj:
            return
        name, ok = QInputDialog.getText(self, "Rename Object", "Object name:", text=obj.name)
        if not ok:
            return
        name = name.strip()
        if not name:
            QMessageBox.warning(self, "Invalid name", "Object name cannot be empty.")
            return
        if self.project.object_name_exists(name, exclude_obj_id=oid):
            QMessageBox.warning(self, "Duplicate name", f"An object named '{name}' already exists.")
            return
        obj.name = name
        self._refresh_objects_list(select_obj_id=oid)
        self.redraw_annotations_for_current()
        self.save_current_project()

    def action_recolor_object(self):
        if not self.project:
            return
        oid = self.current_obj_id()
        if oid is None:
            return
        obj = self.project.get_object(oid)
        if not obj:
            return
        color = QColorDialog.getColor(QColor(obj.color), self, "Choose Object Color")
        if not color.isValid():
            return
        obj.color = color.name().upper()
        self._refresh_objects_list(select_obj_id=oid)
        self.redraw_annotations_for_current()
        self.save_current_project()

    def action_delete_object(self):
        if not self.project:
            return
        oid = self.current_obj_id()
        if oid is None:
            return
        obj = self.project.get_object(oid)
        if not obj:
            return
        resp = QMessageBox.question(
            self,
            "Delete object",
            f"Delete '{obj.name}' and all of its annotations across every video?\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if resp != QMessageBox.StandardButton.Yes:
            return
        for model in self.models.values():
            model.delete_object(oid)
        self.project.remove_object(oid)
        self.view.remove_all_for_obj(oid)
        self._refresh_objects_list()
        self.update_frame_list()
        self.save_current_project()
        self.status.showMessage(f"Deleted object '{obj.name}'.")

    def _on_tree_selection_changed(self):
        items = self.video_tree.selectedItems()
        if not items:
            return
        item = items[0]
        vid_id = item.data(0, Qt.ItemDataRole.UserRole)
        if not vid_id:
            return
        self.load_video(str(vid_id))

    def load_video(self, vid_id: str):
        if not self.project or vid_id not in self.models:
            return
        self.current_video_id = vid_id
        model = self.models[vid_id]
        self.list_label.setText(f"Frames - {self._video_name(vid_id)}")
        self.update_frame_list()
        self.load_frame(model.index)

    def _video_name(self, vid_id: str) -> str:
        if not self.project:
            return ""
        for v in self.project.videos:
            if v.id == vid_id:
                return v.name
        return ""

    def load_frame(self, i: int):
        model = self.current_model()
        if not model or not model.frames:
            return
        i = model.clamp_index(i)
        model.index = i

        img_path = model.frames[i]
        pix = QPixmap(str(img_path))
        if pix.isNull():
            QMessageBox.critical(self, "Error", f"Failed to load {img_path}")
            return
        self.view.set_image(pix)
        self.redraw_annotations_for_current()
        self.setWindowTitle(f"TraceWave - {img_path.name} [{i + 1}/{len(model.frames)}]")

        # sync slider
        if self.slider.maximum() != len(model.frames) - 1:
            self.slider.blockSignals(True)
            self.slider.setMaximum(len(model.frames) - 1)
            self.slider.blockSignals(False)
        if self.slider.value() != i:
            self.slider.blockSignals(True)
            self.slider.setValue(i)
            self.slider.blockSignals(False)

        # sync list selection
        row = self.visible_index_from_frame_index(i)
        if row is not None:
            self.frame_list.blockSignals(True)
            self.frame_list.setCurrentRow(row)
            self.frame_list.blockSignals(False)

    def redraw_annotations_for_current(self):
        model = self.current_model()
        if not model:
            return
        fidx = model.index
        if self.view.pix_item:
            pix = self.view.pix_item.pixmap()
            self.view.set_image(pix)
        fr = model.ann.get(fidx)
        if not fr:
            return
        for oid, obj in fr.objects.items():
            if obj.box and len(obj.box) == 4:
                self.view.set_box_visual(*[int(v) for v in obj.box], obj_id=int(oid))
            for (x, y), lab in zip(obj.points, obj.labels):
                self.view.add_point_visual(int(x), int(y), int(lab), obj_id=int(oid))
            if obj.polygon and len(obj.polygon) >= 3:
                self.view.add_polygon_visual([(int(x), int(y)) for x, y in obj.polygon], obj_id=int(oid))

    def update_frame_list(self):
        model = self.current_model()
        if not model:
            return
        self.frame_list.blockSignals(True)
        self.frame_list.clear()
        show_only = self.btn_toggle_annotated.isChecked()
        for idx, p in enumerate(model.frames):
            annotated = model.is_frame_annotated(idx)
            if show_only and not annotated:
                continue
            item = QListWidgetItem(p.name)
            item.setCheckState(Qt.CheckState.Checked if annotated else Qt.CheckState.Unchecked)
            if not annotated:
                item.setForeground(QtGui.QBrush(QtGui.QColor(120, 120, 120)))
            item.setData(Qt.ItemDataRole.UserRole, idx)
            item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            self.frame_list.addItem(item)
        self.frame_list.blockSignals(False)
        row = self.visible_index_from_frame_index(model.index)
        if row is not None:
            self.frame_list.setCurrentRow(row)

    def visible_index_from_frame_index(self, frame_idx: int) -> Optional[int]:
        for row in range(self.frame_list.count()):
            it = self.frame_list.item(row)
            if it.data(Qt.ItemDataRole.UserRole) == frame_idx:
                return row
        return None

    def refresh_list_item(self, fidx: int):
        row = self.visible_index_from_frame_index(fidx)
        if row is None:
            self.update_frame_list()
            return
        it = self.frame_list.item(row)
        model = self.current_model()
        annotated = model.is_frame_annotated(fidx) if model else False
        it.setCheckState(Qt.CheckState.Checked if annotated else Qt.CheckState.Unchecked)
        it.setForeground(QtGui.QBrush(QtGui.QColor(0, 0, 0) if annotated else QtGui.QColor(120, 120, 120)))

    def _on_frame_list_clicked(self, item: QListWidgetItem):
        idx = int(item.data(Qt.ItemDataRole.UserRole))
        self.load_frame(idx)

    def _on_slider_value_changed(self, i: int):
        self.load_frame(i)

    # ---------- annotation callbacks ----------
    def add_point(self, x: int, y: int, label: int):
        model = self.current_model()
        if not model:
            return
        oid = self.current_obj_id()
        if oid is None:
            self.status.showMessage("Add an object to track first.")
            return
        model.add_point(oid, x, y, label)
        self.view.remove_polygon_for_obj(oid)
        self.refresh_list_item(model.index)
        self.mark_dirty()

    def remove_point(self, x: int, y: int, label: int):
        model = self.current_model()
        if not model:
            return
        oid = self.current_obj_id()
        if oid is None:
            return
        model.remove_point(oid, x, y, label)
        self.view.remove_polygon_for_obj(oid)
        self.refresh_list_item(model.index)
        self.mark_dirty()

    def set_box(self, x: int, y: int, w: int, h: int):
        model = self.current_model()
        if not model:
            return
        oid = self.current_obj_id()
        if oid is None:
            self.status.showMessage("Add an object to track first.")
            return
        model.set_box(oid, x, y, w, h)
        self.view.set_box_visual(x, y, w, h, obj_id=oid)
        self.view.remove_polygon_for_obj(oid)
        self.refresh_list_item(model.index)
        self.mark_dirty()

    def clear_mask_for_current(self):
        model = self.current_model()
        if not model:
            return
        oid = self.current_obj_id()
        if oid is None:
            return
        model.set_polygon(oid, None)
        self.view.remove_polygon_for_obj(oid)
        self.refresh_list_item(model.index)
        self.mark_dirty()

    def run_sam2(self):
        model = self.current_model()
        if not model:
            return
        oid = self.current_obj_id()
        if oid is None:
            self.status.showMessage("Add an object to track first.")
            return
        if not self.sam2 or not self.sam2.available:
            self.status.showMessage(f"SAM2 unavailable: {self.sam2.error if self.sam2 else 'not initialized'}")
            return
        if cv2 is None:
            self.status.showMessage("OpenCV not available; SAM2 disabled.")
            return
        fidx = model.index
        obj = model.get_object(fidx, oid)
        if not obj:
            return
        if not obj.points and not (obj.box and obj.box[2] > 0 and obj.box[3] > 0):
            self.status.showMessage("Add points or a box before running SAM2.")
            return

        img_path = model.frames[fidx]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            self.status.showMessage(f"Failed to read {img_path}")
            return
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        poly = self.sam2.generate_polygon(
            img,
            points_xy=[(int(x), int(y)) for x, y in obj.points],
            point_labels=[int(v) for v in obj.labels],
            box_xywh=list(obj.box) if obj.box else None,
        )
        if not poly:
            self.status.showMessage("SAM2 returned an empty mask.")
            return
        model.set_polygon(oid, poly)
        self.view.add_polygon_visual(poly, obj_id=oid)
        self.refresh_list_item(fidx)
        self.mark_dirty()

    # ---------- modes / state ----------
    def set_mode(self, mode: str):
        self.view.set_mode(mode)
        self.status.showMessage(f"Mode: {mode}")
        self.mark_dirty()

    def mark_dirty(self):
        if not self.project:
            return
        self._autosave_timer.stop()
        self._autosave_timer.start()

    def action_export_yaml(self):
        if not self.project:
            return
        if not self.project.videos:
            QMessageBox.information(self, "No videos", "There are no videos to export.")
            return

        errors = []
        saved = 0
        for vid in self.project.videos:
            model = self.models.get(vid.id)
            if not model:
                continue
            frames_dir = self.project.resolve_video_frames_dir(vid)
            frames_dir.mkdir(parents=True, exist_ok=True)
            source_name = Path(vid.source_path).name
            base_name = source_name or vid.name or vid.id
            out_name = Path(base_name).with_suffix(".yaml").name
            out_path = frames_dir / out_name
            ok, msg = export_yaml(out_path, {vid.id: model})
            if ok:
                saved += 1
            else:
                errors.append(f"{vid.name or vid.id}: {msg}")

        if errors:
            QMessageBox.critical(
                self,
                "Export failed",
                "Some videos failed to export:\n" + "\n".join(errors),
            )
            return
        QMessageBox.information(self, "Exported", f"Exported YAML for {saved} video(s).")

    def show_help(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("TraceWave Help")
        dlg.setModal(True)
        layout = QVBoxLayout(dlg)

        help_text = (
            "Quick Start\n"
            "1. File > New Project (or Open Project).\n"
            "2. File > Import Videos to extract frames.\n"
            "3. Objects panel (left): + Add an object to track, giving it a name.\n"
            "4. Select a video from the tree on the left.\n"
            "5. Annotate frames in the main view.\n"
            "6. File > Export YAMLs to save per-video YAML files.\n\n"
            "Navigation\n"
            "- Right/Left arrows: next/previous frame.\n"
            "- Home/End: first/last frame.\n"
            "- Slider or frame list: jump to a frame.\n\n"
            "Objects\n"
            "- Objects panel: manage the things you're tracking, each with its\n"
            "  own name and color, shared across every video in the project.\n"
            "- + Add creates a new object with an auto-assigned color.\n"
            "- Rename / Recolor change the selected object.\n"
            "- Delete permanently removes the selected object and ALL of its\n"
            "  annotations across every frame and video (cannot be undone; its\n"
            "  id is never reused).\n"
            "- The selected object in the list is the one you're annotating.\n"
            "  Its name and color are shown on the canvas next to whatever\n"
            "  you draw for it.\n\n"
            "Annotation Basics\n"
            "- Point mode (default):\n"
            "  - Left-click: positive point.\n"
            "  - Right-click: negative point.\n"
            "  - Ctrl-click a point: remove it.\n"
            "- Box override:\n"
            "  - Hold Shift and drag with left mouse to draw a box.\n"
            "- Editing an object's points/box after a mask exists clears that\n"
            "  mask automatically, since it no longer matches - press E again\n"
            "  to regenerate it.\n\n"
            "Masks & SAM2\n"
            "- Press E to run SAM2 on the current object.\n"
            "- Press D to clear the current object's mask polygon.\n\n"
            "Export\n"
            "- Export YAMLs saves one YAML per video.\n"
            "- The YAML filename matches the source video name.\n"
            "- Files are saved inside each video's frames folder.\n"
        )

        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setPlainText(help_text)
        layout.addWidget(text, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        dlg.resize(720, 560)
        dlg.exec()

    def closeEvent(self, event: QtGui.QCloseEvent):
        try:
            self.save_current_project()
        except Exception:
            pass
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1500, 900)
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
