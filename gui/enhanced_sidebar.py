# gui/enhanced_sidebar.py - Updated with READABLE font sizes

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSlider,
    QListWidget, QListWidgetItem, QProgressBar, QFrame, QPushButton,
    QScrollArea, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFont

from gui.components import ModernButton, StatusIndicator


class StyledPanel(QFrame):
    """Compact panel with clean styling and READABLE fonts"""

    def __init__(self, title="", collapsible=False, initially_collapsed=False):
        super().__init__()
        self.is_collapsed = initially_collapsed
        self.collapsible = collapsible

        self.setFrameStyle(QFrame.Shape.NoFrame)
        self.setStyleSheet("""
            StyledPanel {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2a2a2e, stop:1 #252526);
                border: 2px solid #00d7ff;
                border-radius: 8px;
                margin: 2px;
            }
        """)

        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(10, 8, 10, 10)  # Increased padding
        self.main_layout.setSpacing(8)  # Increased spacing

        if title:
            self._create_header(title)

        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(8)  # Increased spacing

        self.main_layout.addWidget(self.content_widget)
        self.setLayout(self.main_layout)

        if self.collapsible and self.is_collapsed:
            self.content_widget.hide()

    def _create_header(self, title):
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 6)

        self.title_label = QLabel(title)
        self.title_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))  # Readable size
        self.title_label.setStyleSheet("color: #00d7ff; background: transparent;")

        header_layout.addWidget(self.title_label)
        header_layout.addStretch()

        if self.collapsible:
            self.collapse_btn = QPushButton("+" if self.is_collapsed else "−")
            self.collapse_btn.setFixedSize(20, 20)  # Slightly larger button
            self.collapse_btn.setStyleSheet("""
                QPushButton {
                    background: #00d7ff; color: #1e1e1e; border: none;
                    border-radius: 10px; font-weight: bold; font-size: 12px;
                } QPushButton:hover { background: #40e0ff; }
            """)
            self.collapse_btn.clicked.connect(self._toggle_collapse)
            header_layout.addWidget(self.collapse_btn)

        self.main_layout.insertLayout(0, header_layout)

    def add_widget(self, widget):
        self.content_layout.addWidget(widget)

    def add_layout(self, layout):
        self.content_layout.addLayout(layout)

    def _toggle_collapse(self):
        if not self.collapsible:
            return
        self.is_collapsed = not self.is_collapsed
        self.content_widget.setVisible(not self.is_collapsed)
        if hasattr(self, 'collapse_btn'):
            self.collapse_btn.setText("+" if self.is_collapsed else "−")


class ProjectControlPanel(StyledPanel):
    """Panel for the New Project button"""
    new_project_clicked = Signal()

    def __init__(self):
        super().__init__(title="Project Management", collapsible=False)
        self._init_ui()

    def _init_ui(self):
        self.new_project_btn = ModernButton("📁 New Project", button_type="primary")
        self.new_project_btn.setMinimumHeight(36)  # Larger button
        self.new_project_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0078d4, stop:1 #005a9e);
                border: 1px solid #004578;
                border-radius: 6px;
                color: white;
                padding: 8px 14px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #106ebe, stop:1 #005a9e);
                border-color: #0078d4;
            }
            QPushButton:pressed {
                background: #004578;
            }
        """)
        self.new_project_btn.clicked.connect(self.new_project_clicked.emit)
        self.add_widget(self.new_project_btn)


class ModelConfigPanel(StyledPanel):
    """Panel for Model Configuration with READABLE text"""
    model_config_requested = Signal()

    def __init__(self):
        super().__init__("AI Model Configuration", collapsible=False)
        self._init_ui()

    def _init_ui(self):
        # Model status display
        self.model_status_layout = QVBoxLayout()
        self.model_status_layout.setSpacing(6)

        # Current model assignments (readable display)
        self.planner_status = QLabel("🧠 Planner: Not configured")
        self.planner_status.setStyleSheet("color: #888; font-size: 11px;")  # Increased from 8px

        self.coder_status = QLabel("⚙️ Coder: Not configured")
        self.coder_status.setStyleSheet("color: #888; font-size: 11px;")  # Increased from 8px

        self.assembler_status = QLabel("📄 Assembler: Not configured")
        self.assembler_status.setStyleSheet("color: #888; font-size: 11px;")  # Increased from 8px

        self.model_status_layout.addWidget(self.planner_status)
        self.model_status_layout.addWidget(self.coder_status)
        self.model_status_layout.addWidget(self.assembler_status)

        self.add_layout(self.model_status_layout)

        # Configuration button
        self.config_button = ModernButton("⚙️ Configure Models", button_type="accent")
        self.config_button.setMinimumHeight(36)  # Larger button
        self.config_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #00d7ff, stop:1 #0078d4);
                border: 1px solid #0078d4;
                border-radius: 6px;
                color: #1e1e1e;
                padding: 8px 14px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #40e0ff, stop:1 #0078d4);
            }
            QPushButton:pressed {
                background: #0078d4;
            }
        """)
        self.config_button.clicked.connect(self.model_config_requested.emit)
        self.add_widget(self.config_button)

    def update_model_status(self, config_summary: dict):
        """Update the display with current model assignments"""
        planner = config_summary.get('planner', 'Not configured')
        coder = config_summary.get('coder', 'Not configured')
        assembler = config_summary.get('assembler', 'Not configured')

        # Truncate long model names for display but keep readable
        def truncate_model(name):
            if len(name) > 25:  # Increased from 20
                return name[:22] + "..."
            return name

        self.planner_status.setText(f"🧠 Planner: {truncate_model(planner)}")
        self.coder_status.setText(f"⚙️ Coder: {truncate_model(coder)}")
        self.assembler_status.setText(f"📄 Assembler: {truncate_model(assembler)}")

        # Update colors based on configuration status
        color = "#4ade80" if "Not configured" not in planner else "#888"
        self.planner_status.setStyleSheet(f"color: {color}; font-size: 11px;")

        color = "#4ade80" if "Not configured" not in coder else "#888"
        self.coder_status.setStyleSheet(f"color: {color}; font-size: 11px;")

        color = "#4ade80" if "Not configured" not in assembler else "#888"
        self.assembler_status.setStyleSheet(f"color: {color}; font-size: 11px;")


class KnowledgeBasePanel(StyledPanel):
    # Signal for RAG actions
    scan_directory_requested = Signal()

    def __init__(self):
        super().__init__("Knowledge Base (RAG)", collapsible=True, initially_collapsed=False)
        self._init_ui()

    def _init_ui(self):
        self.scan_btn = ModernButton("🌐 Scan Directory (Global)", button_type="secondary")
        self.scan_btn.setMinimumHeight(32)  # Larger button
        self.scan_btn.setStyleSheet("""
            QPushButton {
                background: #2d2d30; border: 1px solid #404040; border-radius: 5px;
                color: #cccccc; padding: 6px 10px; font-weight: 500; font-size: 11px;
            }
            QPushButton:hover { background: #3e3e42; border-color: #00d7ff; }
        """)
        self.scan_btn.clicked.connect(self.scan_directory_requested.emit)
        self.add_widget(self.scan_btn)

        rag_status_layout = QHBoxLayout()
        rag_status_layout.setContentsMargins(0, 6, 0, 0)

        rag_label = QLabel("RAG:")
        rag_label.setFont(QFont("Segoe UI", 10))  # Increased from 8
        rag_label.setStyleSheet("color: #cccccc;")

        self.rag_status_display_label = QLabel("Initializing embedder...")
        self.rag_status_display_label.setStyleSheet("color: #ffb900; font-size: 10px;")  # Increased from 8px

        rag_status_layout.addWidget(rag_label)
        rag_status_layout.addWidget(self.rag_status_display_label, 1)
        self.add_layout(rag_status_layout)


class ChatActionsPanel(StyledPanel):
    action_triggered = Signal(str)

    def __init__(self):
        super().__init__("Chat Actions", collapsible=True, initially_collapsed=False)
        self._init_ui()

    def _init_ui(self):
        buttons = [
            ("💬 New Session", "new_session"),
            ("📊 View LLM Log", "view_log"),
            ("📟 Open Terminal", "open_terminal"),
            ("📄 Open Code Viewer", "open_code_viewer"),
            ("⚡ View Generated Code", "view_code"),
            ("🔨 Force Code Gen", "force_gen"),
            ("🔄 Check for Updates", "check_updates")
        ]

        for text, action in buttons:
            btn = ModernButton(text, button_type="secondary")
            btn.setMinimumHeight(30)  # Larger buttons
            btn.setStyleSheet("""
                QPushButton {
                    background: #2d2d30; border: 1px solid #404040; border-radius: 5px;
                    color: #cccccc; padding: 6px 8px; font-weight: 500; font-size: 10px;
                }
                QPushButton:hover { background: #3e3e42; border-color: #00d7ff; }
            """)
            btn.clicked.connect(lambda checked, a=action: self.action_triggered.emit(a))
            self.add_widget(btn)


class AvALeftSidebar(QWidget):
    # Signals from child panels
    action_triggered = Signal(str)
    new_project_requested = Signal()
    scan_directory_requested = Signal()
    model_config_requested = Signal()

    def __init__(self):
        super().__init__()
        self.setFixedWidth(300)  # Increased from 280 for better readability
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(6, 6, 6, 6)  # Increased margins
        main_layout.setSpacing(0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea { background: transparent; border: none; }
            QScrollBar:vertical {
                background: #2d2d30; width: 8px; border-radius: 4px; margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #404040; border-radius: 4px; min-height: 20px;
            }
            QScrollBar::handle:vertical:hover { background: #00d7ff; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px; border: none; background: none;
            }
        """)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(6)  # Increased spacing

        # Create panels
        self.project_control_panel = ProjectControlPanel()
        self.model_config_panel = ModelConfigPanel()
        self.rag_panel = KnowledgeBasePanel()
        self.actions_panel = ChatActionsPanel()

        content_layout.addWidget(self.project_control_panel)
        content_layout.addWidget(self.model_config_panel)
        content_layout.addWidget(self.rag_panel)
        content_layout.addWidget(self.actions_panel)
        content_layout.addStretch()

        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)

        self.setStyleSheet("""
            AvALeftSidebar {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e1e1e, stop:1 #252526);
                border-right: 2px solid #00d7ff;
            }
        """)

    def _connect_signals(self):
        # Connect signals from child panels to the sidebar's own signals
        self.project_control_panel.new_project_clicked.connect(self.new_project_requested)
        self.model_config_panel.model_config_requested.connect(self.model_config_requested)
        self.rag_panel.scan_directory_requested.connect(self.scan_directory_requested)
        self.actions_panel.action_triggered.connect(self.action_triggered)

    def update_sidebar_rag_status(self, status_text: str, color_hex: str):
        """Update RAG status display in sidebar"""
        if hasattr(self.rag_panel, 'rag_status_display_label'):
            self.rag_panel.rag_status_display_label.setText(status_text)
            self.rag_panel.rag_status_display_label.setStyleSheet(f"color: {color_hex}; font-size: 10px;")  # Increased from 8px

    def update_model_status_display(self, config_summary: dict):
        """Update model configuration status display"""
        self.model_config_panel.update_model_status(config_summary)