# gui/workflow_monitor_window.py - Graphics Window for Workflow Visualization
import sys
from PySide6.QtCore import Qt, QPointF, Slot, QTimer
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (
    QMainWindow, QGraphicsView, QWidget, QVBoxLayout, QHBoxLayout,
    QFrame, QLabel
)

from gui.components import Colors, Typography, ModernButton
from gui.workflow_monitor_scene import WorkflowMonitorScene


class WorkflowMonitorWindow(QMainWindow):
    """
    Window for visualizing the AI agent workflow in real-time.
    Contains the graphics scene and view, along with control widgets.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AvA - Visual Workflow Monitor")
        self.setGeometry(150, 150, 1000, 700)

        # Core components
        self.scene = WorkflowMonitorScene()
        self.view = QGraphicsView(self.scene)
        self.agent_status_widgets = {}

        self._init_ui()
        self._apply_style()

        # Initialize with a standard layout
        self.scene.setup_standard_workflow()

        # FIX: Setup control panel immediately to avoid race conditions.
        self._setup_control_panel()

        self.center_view()

    def _init_ui(self):
        """Initialize the UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Configure the graphics view
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Create the control panel
        self.control_panel = QWidget()
        self.control_panel.setMinimumWidth(250)
        self.control_panel.setMaximumWidth(300)
        self.control_panel.setStyleSheet(
            f"background: {Colors.SECONDARY_BG}; border-left: 1px solid {Colors.BORDER_DEFAULT};")

        main_layout.addWidget(self.view, 5)
        main_layout.addWidget(self.control_panel, 1)

    def _setup_control_panel(self):
        """Setup the control panel UI. This is now called directly."""
        # Clear any existing layout
        if self.control_panel.layout():
            for i in reversed(range(self.control_panel.layout().count())):
                item = self.control_panel.layout().itemAt(i)
                if item.widget():
                    item.widget().setParent(None)
            self.control_panel.layout().deleteLater()

        layout = QVBoxLayout(self.control_panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Workflow Status Section
        status_frame = QFrame()
        status_frame_layout = QVBoxLayout(status_frame)
        status_frame_layout.setSpacing(6)

        status_header = QLabel("Agent Status")
        status_header.setFont(Typography.heading_medium())
        status_frame_layout.addWidget(status_header)

        # Agent status displays
        self.agent_status_widgets = {}
        for agent_id, agent_node in self.scene._agent_nodes.items():
            agent_panel = QFrame()
            agent_panel_layout = QVBoxLayout(agent_panel)
            agent_panel.setStyleSheet(f"background: {Colors.ELEVATED_BG}; border-radius: 6px; padding: 8px;")

            agent_name = QLabel(f"{agent_node.icon} {agent_node.agent_name}")
            agent_name.setFont(Typography.body())

            status_text = QLabel("Status: Idle\nReady")
            status_text.setFont(Typography.body_small())
            status_text.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
            status_text.setWordWrap(True)

            agent_panel_layout.addWidget(agent_name)
            agent_panel_layout.addWidget(status_text)

            status_frame_layout.addWidget(agent_panel)
            self.agent_status_widgets[agent_id] = status_text

        layout.addWidget(status_frame)
        layout.addStretch(1)

        # Controls Section
        controls_header = QLabel("Controls")
        controls_header.setFont(Typography.heading_medium())
        layout.addWidget(controls_header)

        reset_zoom_btn = ModernButton("🔍 Reset Zoom")
        reset_zoom_btn.clicked.connect(self.view.resetTransform)
        layout.addWidget(reset_zoom_btn)

        center_view_btn = ModernButton("🎯 Center View")
        center_view_btn.clicked.connect(self.center_view)
        layout.addWidget(center_view_btn)

        refresh_btn = ModernButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.refresh_workflow)
        layout.addWidget(refresh_btn)

    def _apply_style(self):
        """Apply modern styling."""
        self.setStyleSheet(f"background-color: {Colors.PRIMARY_BG};")
        self.view.setStyleSheet("border: none;")

    # Public Slots for AvAApplication to connect to
    @Slot(str, str, str)
    def update_agent_status(self, agent_id: str, status: str, status_text: str):
        self.scene.update_agent_status(agent_id, status, status_text)
        if agent_id in self.agent_status_widgets:
            status_label = self.agent_status_widgets[agent_id]
            status_label.setText(f"Status: {status.title()}\n{status_text}")
            color = Colors.TEXT_SECONDARY
            if status == "working":
                color = Colors.ACCENT_ORANGE
            elif status == "success":
                color = Colors.ACCENT_GREEN
            elif status == "error":
                color = Colors.ACCENT_RED
            status_label.setStyleSheet(f"color: {color};")

    @Slot(str, str)
    def activate_connection(self, from_agent_id: str, to_agent_id: str):
        self.scene.activate_connection(from_agent_id, to_agent_id)

    @Slot(str, str)
    def deactivate_connection(self, from_agent_id: str, to_agent_id: str):
        self.scene.deactivate_connection(from_agent_id, to_agent_id)

    @Slot()
    def refresh_workflow(self):
        self.scene.refresh_workflow()
        self._setup_control_panel()  # Re-setup panel to match refreshed scene
        for agent_id in self.agent_status_widgets.keys():
            self.update_agent_status(agent_id, "idle", "Ready")

    @Slot()
    def center_view(self):
        """Centers the view on the scene's content."""
        self.view.fitInView(self.scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        """Handle zooming with the mouse wheel."""
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor
        if event.angleDelta().y() > 0:
            self.view.scale(zoom_in_factor, zoom_in_factor)
        else:
            self.view.scale(zoom_out_factor, zoom_out_factor)