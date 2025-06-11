# gui/workflow_monitor_window.py - Visual Workflow Monitor Window

from typing import Dict, Optional
from PySide6.QtCore import Qt, Slot, QTimer
from PySide6.QtGui import QAction, QKeySequence, QPainter
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGraphicsView,
    QLabel, QFrame, QPushButton, QSizePolicy, QStatusBar
)

from gui.workflow_monitor_scene import WorkflowMonitorScene
from gui.components import Colors, Typography, ModernButton


class WorkflowGraphicsView(QGraphicsView):
    """
    Custom graphics view with smooth scrolling and zoom capabilities.
    """

    def __init__(self, scene: WorkflowMonitorScene):
        super().__init__(scene)

        # Setup view properties
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Enable zoom with mouse wheel
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        # Apply dark styling
        self.setStyleSheet(f"""
            QGraphicsView {{
                background: {Colors.PRIMARY_BG};
                border: 1px solid {Colors.BORDER_DEFAULT};
                border-radius: 8px;
            }}
            QScrollBar:vertical {{
                background: {Colors.SECONDARY_BG};
                width: 12px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background: {Colors.BORDER_DEFAULT};
                border-radius: 6px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {Colors.ACCENT_BLUE};
            }}
            QScrollBar:horizontal {{
                background: {Colors.SECONDARY_BG};
                height: 12px;
                border-radius: 6px;
            }}
            QScrollBar::handle:horizontal {{
                background: {Colors.BORDER_DEFAULT};
                border-radius: 6px;
                min-width: 20px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background: {Colors.ACCENT_BLUE};
            }}
        """)

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Zoom with Ctrl+Wheel
            zoom_factor = 1.15
            if event.angleDelta().y() > 0:
                self.scale(zoom_factor, zoom_factor)
            else:
                self.scale(1.0 / zoom_factor, 1.0 / zoom_factor)
        else:
            # Normal scrolling
            super().wheelEvent(event)


class WorkflowStatusPanel(QFrame):
    """
    Side panel showing workflow status and controls.
    """

    def __init__(self):
        super().__init__()

        self.setFixedWidth(280)
        self.setFrameStyle(QFrame.Shape.NoFrame)
        self.setStyleSheet(f"""
            WorkflowStatusPanel {{
                background: {Colors.SECONDARY_BG};
                border-left: 1px solid {Colors.BORDER_DEFAULT};
            }}
        """)

        self._setup_ui()

        # Update timer
        self._update_timer = QTimer()
        self._update_timer.timeout.connect(self._update_display)
        self._update_timer.start(1000)  # Update every second

    def _setup_ui(self):
        """Setup the status panel UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Title
        title_label = QLabel("Workflow Status")
        title_label.setFont(Typography.heading_medium())
        title_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; margin-bottom: 8px;")
        layout.addWidget(title_label)

        # Status indicators
        self._create_status_section(layout)

        # Controls
        self._create_controls_section(layout)

        # Stretch to push everything to top
        layout.addStretch(1)

        self.setLayout(layout)

    def _create_status_section(self, layout: QVBoxLayout):
        """Create the status indicators section"""
        # Section header
        status_header = QLabel("Agent Status")
        status_header.setFont(Typography.heading_small())
        status_header.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; margin-top: 8px;")
        layout.addWidget(status_header)

        # Status list container
        self.status_container = QWidget()
        status_layout = QVBoxLayout(self.status_container)
        status_layout.setContentsMargins(0, 8, 0, 8)
        status_layout.setSpacing(6)

        # Placeholder status items (will be updated dynamically)
        self.status_items = {}

        layout.addWidget(self.status_container)

    def _create_controls_section(self, layout: QVBoxLayout):
        """Create the controls section"""
        # Section header
        controls_header = QLabel("Controls")
        controls_header.setFont(Typography.heading_small())
        controls_header.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; margin-top: 16px;")
        layout.addWidget(controls_header)

        # Control buttons
        self.reset_zoom_btn = ModernButton("🔍 Reset Zoom", button_type="secondary")
        self.reset_zoom_btn.clicked.connect(self._reset_zoom)
        layout.addWidget(self.reset_zoom_btn)

        self.center_view_btn = ModernButton("🎯 Center View", button_type="secondary")
        self.center_view_btn.clicked.connect(self._center_view)
        layout.addWidget(self.center_view_btn)

        self.refresh_btn = ModernButton("🔄 Refresh", button_type="secondary")
        self.refresh_btn.clicked.connect(self._refresh_display)
        layout.addWidget(self.refresh_btn)

    def update_agent_status(self, agent_id: str, agent_name: str, status: str, status_text: str):
        """Update the status display for an agent"""
        if agent_id not in self.status_items:
            # Create new status item
            item_widget = QWidget()
            item_layout = QVBoxLayout(item_widget)
            item_layout.setContentsMargins(8, 6, 8, 6)
            item_layout.setSpacing(2)

            # Agent name label
            name_label = QLabel(agent_name)
            name_label.setFont(Typography.body())
            name_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-weight: 500;")

            # Status label
            status_label = QLabel()
            status_label.setFont(Typography.body_small())
            status_label.setWordWrap(True)

            item_layout.addWidget(name_label)
            item_layout.addWidget(status_label)

            # Style the container
            item_widget.setStyleSheet(f"""
                QWidget {{
                    background: {Colors.ELEVATED_BG};
                    border: 1px solid {Colors.BORDER_DEFAULT};
                    border-radius: 6px;
                }}
            """)

            self.status_items[agent_id] = {
                'widget': item_widget,
                'name_label': name_label,
                'status_label': status_label
            }

            self.status_container.layout().addWidget(item_widget)

        # Update the status
        item = self.status_items[agent_id]
        status_label = item['status_label']

        # Update text and color based on status
        status_colors = {
            'idle': Colors.TEXT_MUTED,
            'working': Colors.ACCENT_ORANGE,
            'success': Colors.ACCENT_GREEN,
            'error': Colors.ACCENT_RED
        }

        color = status_colors.get(status, Colors.TEXT_SECONDARY)
        status_label.setText(f"Status: {status}\n{status_text}")
        status_label.setStyleSheet(f"color: {color};")

    def _update_display(self):
        """Periodic update of the display"""
        # This can be connected to get real-time updates from the scene
        pass

    def _reset_zoom(self):
        """Reset zoom to default level"""
        if hasattr(self.parent(), 'graphics_view'):
            self.parent().graphics_view.resetTransform()

    def _center_view(self):
        """Center the view on the workflow"""
        if hasattr(self.parent(), 'graphics_view'):
            self.parent().graphics_view.centerOn(0, 0)

    def _refresh_display(self):
        """Refresh the entire display"""
        if hasattr(self.parent(), 'refresh_workflow'):
            self.parent().refresh_workflow()


class WorkflowMonitorWindow(QMainWindow):
    """
    Main window for the Visual Workflow Monitor.
    Provides real-time visualization of the AI workflow process.
    """

    def __init__(self):
        super().__init__()

        # Initialize components
        self.monitor_scene = WorkflowMonitorScene()
        self.graphics_view = WorkflowGraphicsView(self.monitor_scene)
        self.status_panel = WorkflowStatusPanel()

        # Setup window
        self._setup_window()
        self._setup_ui()
        self._setup_menu_bar()
        self._setup_status_bar()

        # Initialize with standard workflow
        self.refresh_workflow()

    def _setup_window(self):
        """Setup window properties"""
        self.setWindowTitle("AvA - Visual Workflow Monitor")
        self.setMinimumSize(1000, 700)
        self.resize(1400, 900)

        # Apply dark theme
        self.setStyleSheet(f"""
            QMainWindow {{
                background: {Colors.PRIMARY_BG};
                color: {Colors.TEXT_PRIMARY};
            }}
            QMenuBar {{
                background: {Colors.SECONDARY_BG};
                color: {Colors.TEXT_PRIMARY};
                border-bottom: 1px solid {Colors.BORDER_DEFAULT};
                padding: 4px;
            }}
            QMenuBar::item {{
                background: transparent;
                padding: 4px 8px;
                border-radius: 4px;
            }}
            QMenuBar::item:selected {{
                background: {Colors.HOVER_BG};
            }}
        """)

    def _setup_ui(self):
        """Setup the main UI layout"""
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Main graphics view
        main_layout.addWidget(self.graphics_view, 1)

        # Status panel
        main_layout.addWidget(self.status_panel)

        self.setCentralWidget(central_widget)

    def _setup_menu_bar(self):
        """Setup the menu bar"""
        menubar = self.menuBar()

        # View menu
        view_menu = menubar.addMenu("View")

        # Reset zoom action
        reset_zoom_action = QAction("Reset Zoom", self)
        reset_zoom_action.setShortcut(QKeySequence("Ctrl+0"))
        reset_zoom_action.triggered.connect(self._reset_zoom)
        view_menu.addAction(reset_zoom_action)

        # Center view action
        center_action = QAction("Center View", self)
        center_action.setShortcut(QKeySequence("Ctrl+Shift+C"))
        center_action.triggered.connect(self._center_view)
        view_menu.addAction(center_action)

        view_menu.addSeparator()

        # Refresh action
        refresh_action = QAction("Refresh", self)
        refresh_action.setShortcut(QKeySequence("F5"))
        refresh_action.triggered.connect(self.refresh_workflow)
        view_menu.addAction(refresh_action)

    def _setup_status_bar(self):
        """Setup the status bar"""
        status_bar = QStatusBar()
        status_bar.setStyleSheet(f"""
            QStatusBar {{
                background: {Colors.SECONDARY_BG};
                color: {Colors.TEXT_SECONDARY};
                border-top: 1px solid {Colors.BORDER_DEFAULT};
                padding: 4px;
            }}
        """)

        status_bar.showMessage("Workflow Monitor Ready")
        self.setStatusBar(status_bar)

    @Slot(str, str, str)
    def update_agent_status(self, agent_id: str, status: str, status_text: str = ""):
        """Update the status of a specific agent"""
        # Update scene
        self.monitor_scene.update_agent_status(agent_id, status, status_text)

        # Update status panel
        agent_node = self.monitor_scene.get_agent_node(agent_id)
        if agent_node:
            self.status_panel.update_agent_status(
                agent_id,
                agent_node.agent_name,
                status,
                status_text
            )

        # Update status bar
        active_agents = [
            node.agent_name for node in self.monitor_scene._agent_nodes.values()
            if node.get_status() == "working"
        ]

        if active_agents:
            self.statusBar().showMessage(f"Active: {', '.join(active_agents)}")
        else:
            self.statusBar().showMessage("Workflow Monitor Ready")

    @Slot(str, str)
    def activate_connection(self, from_agent_id: str, to_agent_id: str):
        """Activate data flow animation between agents"""
        self.monitor_scene.activate_connection(from_agent_id, to_agent_id, True)

    @Slot(str, str)
    def deactivate_connection(self, from_agent_id: str, to_agent_id: str):
        """Deactivate data flow animation between agents"""
        self.monitor_scene.activate_connection(from_agent_id, to_agent_id, False)

    @Slot()
    def refresh_workflow(self):
        """Refresh the workflow display"""
        self.monitor_scene.setup_standard_workflow()
        self._center_view()

        # Update status panel for all agents
        for agent_id, node in self.monitor_scene._agent_nodes.items():
            self.status_panel.update_agent_status(
                agent_id,
                node.agent_name,
                node.get_status(),
                node.get_status_text()
            )

    def _reset_zoom(self):
        """Reset zoom to default level"""
        self.graphics_view.resetTransform()

    def _center_view(self):
        """Center the view on the workflow"""
        self.graphics_view.fitInView(
            self.monitor_scene.itemsBoundingRect(),
            Qt.AspectRatioMode.KeepAspectRatio
        )

    def closeEvent(self, event):
        """Handle window close event"""
        # Clean up any running animations
        for connection in self.monitor_scene._connections:
            connection.set_active(False)

        super().closeEvent(event)