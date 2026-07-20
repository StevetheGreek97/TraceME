import numpy as np
import cv2
import torch
from PyQt6.QtGui import QPen, QColor, QPolygonF, QBrush
from PyQt6.QtWidgets import QGraphicsEllipseItem, QGraphicsPolygonItem, QGraphicsRectItem
from PyQt6.QtCore import pyqtSignal, QObject, QPointF, QRectF, Qt
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2
import sys, os
def get_resource_path(rel_path):
    """Get the absolute path to a resource, adjusting for executable and normal script execution."""
    if getattr(sys, 'frozen', False):
        # If the application is running as an executable (PyInstaller)
        base_path = sys._MEIPASS  
    else:
        # Normal script execution: move one directory up to get out of 'services'
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    return os.path.join(base_path, rel_path)


class SamBoxMasker(QObject):
    """
    A class for creating masks interactively using the SAM 2 model with box and point input.
    """

    mask_added = pyqtSignal(str, np.ndarray)

    def __init__(self, parent, device: str = "cpu"):
        """
        Initialize the SamBoxMasker class.

        Args:
            parent: Reference to the parent ImageDisplay instance.
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        super().__init__()
        self.parent = parent
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load the SAM 2 model
        self.predictor = self._load_sam2_model()

        # Interactive elements
        self.foreground_points = []
        self.background_points = []
        self.foreground_items = []
        self.background_items = []
        self.box = None  # (x1, y1, x2, y2)
        self.box_item = None  # Reference to the drawn box
        self.current_polygon_item = None  # Reference to displayed polygon
        self.mask = None
        self.is_drawing_box = False  # ✅ Initialize the variable
        self.box_start = None  # ✅ Store box start position

    def _load_sam2_model(self):
        """Load the SAM 2 model once."""
        model_path = get_resource_path("/home/steve/Documents/Projects/HeartBit/app/sam2_configs/sam2_hiera_tiny.pt")
        sam2_model = build_sam2("sam2_hiera_t.yaml", model_path, device=self.device)
        return SAM2ImagePredictor(sam2_model)

    def _add_graphics_point(self, point: tuple, label: int):
        """Helper function to add a point to the scene."""
        color = QColor(0, 255, 0) if label == 1 else QColor(255, 0, 0)
        pen = QPen(color, 2)

        ellipse = QGraphicsEllipseItem(point[0] - 2, point[1] - 2, 4, 4)
        ellipse.setPen(pen)
        ellipse.setBrush(color)
        self.parent.scene.addItem(ellipse)

        return ellipse

    def add_point(self, point: tuple, label: int):
        """
        Add a point to the appropriate list (foreground or background).

        Args:
            point (tuple): The point to add as (x, y).
            label (int): The label for the point (1 for foreground, 0 for background).
        """
        ellipse = self._add_graphics_point(point, label)

        if label == 1:
            self.foreground_points.append(point)
            self.foreground_items.append(ellipse)
        else:
            self.background_points.append(point)
            self.background_items.append(ellipse)


    def add_box(self, start_point: tuple, end_point: tuple):
        """
        Add a bounding box to the scene.

        Args:
            start_point (tuple): Top-left corner of the box (x1, y1).
            end_point (tuple): Bottom-right corner of the box (x2, y2).
        """
        self.clear_box()

        x1, y1 = start_point
        x2, y2 = end_point
        self.box = np.array([x1, y1, x2, y2])

        rect = QRectF(x1, y1, x2 - x1, y2 - y1)
        pen = QPen(QColor(0, 0, 255), 2)  # Blue box
        self.box_item = QGraphicsRectItem(rect)
        self.box_item.setPen(pen)
        self.parent.scene.addItem(self.box_item)

 

    def clear_box(self):
        """Remove any existing bounding box."""
        if self.box_item:
            self.parent.scene.removeItem(self.box_item)
            self.box_item = None
        self.box = None

    def display_polygon(self, polygon_points: list, color=QColor(0, 255, 0), line_width: int = 2):
        if self.current_polygon_item:
            self.parent.scene.removeItem(self.current_polygon_item)
            self.current_polygon_item = None

        if not polygon_points:

            return

        polygon = QPolygonF([QPointF(x, y) for x, y in polygon_points])
        pen = QPen(color)
        pen.setWidth(line_width)

        polygon_item = QGraphicsPolygonItem(polygon)
        polygon_item.setPen(pen)
        polygon_item.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 60)))  # translucent fill
        self.parent.scene.addItem(polygon_item)

        self.current_polygon_item = polygon_item
        self.mask = None

    def generate_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Generate a mask using the SAM 2 model based on added points and/or bounding box.

        Args:
            image (numpy.ndarray): The input image for mask generation.

        Returns:
            numpy.ndarray: Generated mask.
        """
        if self.box is None and not self.foreground_points:
            return None

        points = np.array(self.foreground_points + self.background_points)
        labels = np.array([1] * len(self.foreground_points) + [0] * len(self.background_points))

        self.predictor.set_image(image)
        masks, scores, _ = self.predictor.predict(
            point_coords=points if len(points) > 0 else None,
            point_labels=labels if len(labels) > 0 else None,
            box=self.box[None, :] if self.box is not None else None,
            multimask_output=False
        )
        mask = masks[0]

        # Extract contours for visualization
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygon = contours[0].squeeze(axis=1).tolist() if contours else []

        self.display_polygon(polygon)
        self.mask = np.array(polygon, dtype=np.float32)

        return self.mask

    def clear_temp_items(self):
        """
        Clear temporary points, lines, and polygons.
        """
        for item in self.foreground_items + self.background_items:
            self.parent.scene.removeItem(item)

        self.foreground_items.clear()
        self.background_items.clear()
        self.foreground_points.clear()
        self.background_points.clear()

        self.clear_box()

        if self.current_polygon_item:
            self.parent.scene.removeItem(self.current_polygon_item)
            self.current_polygon_item = None

        self.mask = None


    def complete_mask(self):
        """
        Save the generated mask to the database.
        """
        if self.mask is None or self.mask.shape[0] == 0:

            return
        if not self.parent.parent.sidebar.has_valid_class_selection():
            self.clear_temp_items()
            return  # ❌ Cancel saving if no valid class is selected

        class_name, selected_color = self.parent.parent.sidebar.get_selected_class_color()
        self.parent.parent.state_manager.mask_manager.save_mask(
            self.mask, self.parent.parent.state_manager.current_image_name, class_name
        )

        self.mask_added.emit(self.parent.parent.state_manager.current_image_name, self.mask)
        self.clear_temp_items()
    def update_box_preview(self, start_point, end_point):
        """
        Update the visual preview of the bounding box while the user is drawing.
        """
        self.clear_box()  # Remove the old preview

        x1, y1 = start_point
        x2, y2 = end_point
        self.box = np.array([x1, y1, x2, y2])

        rect = QRectF(x1, y1, x2 - x1, y2 - y1)
        pen = QPen(QColor(0, 0, 255), 2, Qt.PenStyle.DashLine)  # Dashed blue box for preview
        self.box_item = QGraphicsRectItem(rect)
        self.box_item.setPen(pen)
        self.parent.scene.addItem(self.box_item)

