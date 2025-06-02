# gui/enhanced_sidebar.py - Compact Layout with Everything Visible

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSlider,
    QListWidget, QListWidgetItem, QProgressBar, QFrame, QPushButton,
    QScrollArea, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFont

from gui.components import ModernButton, StatusIndicator


class StyledPanel(QFrame):
    """Compact panel with clean styling"""

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
        self.main_layout.setContentsMargins(8, 6, 8, 8)  # Reduced padding
        self.main_layout.setSpacing(6)  # Reduced spacing

        if title:
            self._create_header(title)

        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(6)

        self.main_layout.addWidget(self.content_widget)
        self.setLayout(self.main_layout)

        if self.collapsible and self.is_collapsed:
            self.content_widget.hide()

    def _create_header(self, title):
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 4)

        self.title_label = QLabel(title)
        self.title_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))  # Smaller font
        self.title_label.setStyleSheet("color: #00d7ff; background: transparent;")

        header_layout.addWidget(self.title_label)
        header_layout.addStretch()

        if self.collapsible:
            self.collapse_btn = QPushButton("+" if self.is_collapsed else "−")
            self.collapse_btn.setFixedSize(18, 18)  # Smaller button
            self.collapse_btn.setStyleSheet("""
                QPushButton {
                    background: #00d7ff; color: #1e1e1e; border: none;
                    border-radius: 9px; font-weight: bold; font-size: 11px;
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
        super().__init__(title="Project Management", collapsible=False) # Can be collapsible if desired
        self._init_ui()

    def _init_ui(self):
        self.new_project_btn = ModernButton("📁 New Project", button_type="primary")
        self.new_project_btn.setMinimumHeight(32)
        self.new_project_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0078d4, stop:1 #005a9e);
                border: 1px solid #004578;
                border-radius: 5px;
                color: white;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 11px;
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


class LLMConfigPanel(StyledPanel):
    model_changed = Signal(str, str)
    temperature_changed = Signal(float)

    def __init__(self):
        super().__init__("LLM Configuration", collapsible=True, initially_collapsed=False)
        self._init_ui()

    def _init_ui(self):
        # Compact Chat LLM
        chat_layout = QHBoxLayout()
        chat_layout.setSpacing(6)

        chat_label = QLabel("Chat LLM:")
        chat_label.setFont(QFont("Segoe UI", 8))
        chat_label.setStyleSheet("color: #cccccc;")
        chat_label.setMinimumWidth(60)

        self.chat_combo = QComboBox()
        self.chat_combo.addItems([
            "Gemini: gemini-2.5-pro-preview",
            "OpenAI: gpt-4o",
            "Anthropic: claude-3.5-sonnet",
            "DeepSeek: deepseek-chat"
        ])
        self.chat_combo.setStyleSheet("""
            QComboBox {
                background: #1e1e1e; border: 1px solid #404040; border-radius: 3px;
                padding: 4px 8px; color: #cccccc; min-width: 100px; font-size: 8px;
                min-height: 20px;
            }
            QComboBox:hover { border-color: #00d7ff; }
            QComboBox::drop-down { border: none; width: 16px; }
            QComboBox::down-arrow {
                image: none; border-left: 3px solid transparent; border-right: 3px solid transparent;
                border-top: 3px solid #cccccc; margin-right: 4px;
            }
            QComboBox QAbstractItemView {
                background: #2d2d30; border: 1px solid #00d7ff;
                selection-background-color: #00d7ff; selection-color: #1e1e1e; color: #cccccc;
            }
        """)

        chat_status = StatusIndicator("ready")

        chat_layout.addWidget(chat_label)
        chat_layout.addWidget(self.chat_combo, 1)
        chat_layout.addWidget(chat_status)
        self.add_layout(chat_layout)

        # Compact Code LLM
        code_label = QLabel("Specialized LLM (Code Gen):")
        code_label.setFont(QFont("Segoe UI", 8))
        code_label.setStyleSheet("color: #cccccc; margin: 4px 0 2px 0;")
        self.add_widget(code_label)

        code_layout = QHBoxLayout()
        code_layout.setSpacing(6)

        self.code_combo = QComboBox()
        self.code_combo.addItems([
            "Ollama (Gen): qwen2.5-coder",
            "OpenAI: gpt-4o",
            "Anthropic: claude-3.5-sonnet",
            "DeepSeek: deepseek-coder-v2"
        ])
        self.code_combo.setStyleSheet(self.chat_combo.styleSheet())

        code_status = StatusIndicator("ready")

        code_layout.addWidget(self.code_combo, 1)
        code_layout.addWidget(code_status)
        self.add_layout(code_layout)

        # Compact Temperature
        temp_header = QHBoxLayout()
        temp_header.setContentsMargins(0, 6, 0, 2)

        temp_label = QLabel("Temperature (Chat):")
        temp_label.setFont(QFont("Segoe UI", 8))
        temp_label.setStyleSheet("color: #cccccc;")

        self.temp_value = QLabel("0.70")
        self.temp_value.setStyleSheet("color: #00d7ff; font-weight: bold; font-size: 8px;")

        temp_header.addWidget(temp_label)
        temp_header.addStretch()
        temp_header.addWidget(self.temp_value)
        self.add_layout(temp_header)

        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setRange(0, 100)
        self.temp_slider.setValue(70)
        self.temp_slider.setFixedHeight(20)  # Smaller slider
        self.temp_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #404040; height: 4px; background: #1e1e1e; border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #00d7ff; border: 1px solid #00d7ff; width: 12px; height: 12px;
                border-radius: 6px; margin: -4px 0;
            }
            QSlider::handle:horizontal:hover { background: #40e0ff; border-color: #40e0ff; }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00d7ff, stop:1 #0078d4);
                border-radius: 2px;
            }
        """)
        self.temp_slider.valueChanged.connect(self._on_temperature_changed)
        self.add_widget(self.temp_slider)

        # Compact Persona button
        self.persona_btn = ModernButton("🎭 Configure Persona", button_type="secondary")
        self.persona_btn.setMinimumHeight(26)  # Smaller button
        self.persona_btn.setStyleSheet("""
            QPushButton {
                background: #2d2d30; border: 1px solid #404040; border-radius: 4px;
                color: #cccccc; padding: 4px 8px; font-weight: 500; font-size: 9px;
                min-height: 26px;
            }
            QPushButton:hover { background: #3e3e42; border-color: #00d7ff; color: white; }
            QPushButton:pressed { background: #1e1e1e; }
        """)
        self.add_widget(self.persona_btn)

    def _on_temperature_changed(self, value):
        temp_val = value / 100.0
        self.temp_value.setText(f"{temp_val:.2f}")
        self.temperature_changed.emit(temp_val)


class KnowledgeBasePanel(StyledPanel):
    # Signal for RAG actions
    scan_directory_requested = Signal()
    # add_files_requested = Signal() # If you bring back the add files button

    def __init__(self):
        super().__init__("Knowledge Base (RAG)", collapsible=True, initially_collapsed=False)
        self._init_ui()

    def _init_ui(self):
        self.scan_btn = ModernButton("🌐 Scan Directory (Global)", button_type="secondary")
        self.scan_btn.setMinimumHeight(26)
        self.scan_btn.setStyleSheet("""
            QPushButton {
                background: #2d2d30; border: 1px solid #404040; border-radius: 4px;
                color: #cccccc; padding: 4px 8px; font-weight: 500; font-size: 9px;
            }
            QPushButton:hover { background: #3e3e42; border-color: #00d7ff; }
        """)
        self.scan_btn.clicked.connect(self.scan_directory_requested.emit) # Emit signal
        self.add_widget(self.scan_btn)

        rag_status_layout = QHBoxLayout()
        rag_status_layout.setContentsMargins(0, 4, 0, 0)

        rag_label = QLabel("RAG:")
        rag_label.setFont(QFont("Segoe UI", 8))
        rag_label.setStyleSheet("color: #cccccc;")

        self.rag_status_display_label = QLabel("Initializing embedder...")
        self.rag_status_display_label.setStyleSheet("color: #ffb900; font-size: 8px;")

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
            btn.setMinimumHeight(24)  # Smaller buttons
            btn.setStyleSheet("""
                QPushButton {
                    background: #2d2d30; border: 1px solid #404040; border-radius: 4px;
                    color: #cccccc; padding: 3px 6px; font-weight: 500; font-size: 8px;
                }
                QPushButton:hover { background: #3e3e42; border-color: #00d7ff; }
            """)
            btn.clicked.connect(lambda checked, a=action: self.action_triggered.emit(a))
            self.add_widget(btn)


class AvALeftSidebar(QWidget):
    # Signals from child panels
    model_changed = Signal(str, str)
    temperature_changed = Signal(float)
    action_triggered = Signal(str) # From ChatActionsPanel
    new_project_requested = Signal() # From ProjectControlPanel
    scan_directory_requested = Signal() # From KnowledgeBasePanel

    def __init__(self):
        super().__init__()
        self.setFixedWidth(280)
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea { background: transparent; border: none; }
            QScrollBar:vertical {
                background: #2d2d30; width: 6px; border-radius: 3px; margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #404040; border-radius: 3px; min-height: 15px;
            }
            QScrollBar::handle:vertical:hover { background: #00d7ff; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px; border: none; background: none;
            }
        """)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(3)

        # Create panels
        self.project_control_panel = ProjectControlPanel() # New panel for "New Project"
        self.llm_panel = LLMConfigPanel()
        self.rag_panel = KnowledgeBasePanel()
        self.actions_panel = ChatActionsPanel()

        content_layout.addWidget(self.project_control_panel) # Add new project panel
        content_layout.addWidget(self.llm_panel)
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
        self.llm_panel.model_changed.connect(self.model_changed)
        self.llm_panel.temperature_changed.connect(self.temperature_changed)
        self.rag_panel.scan_directory_requested.connect(self.scan_directory_requested)
        self.actions_panel.action_triggered.connect(self.action_triggered)

    def get_current_models(self):
        return {
            "chat_model": self.llm_panel.chat_combo.currentText(),
            "code_model": self.llm_panel.code_combo.currentText(),
            "temperature": self.llm_panel.temp_slider.value() / 100.0
        }

    def update_sidebar_rag_status(self, status_text: str, color_hex: str):
        if hasattr(self.rag_panel, 'rag_status_display_label'):
            self.rag_panel.rag_status_display_label.setText(status_text)
            self.rag_panel.rag_status_display_label.setStyleSheet(f"color: {color_hex}; font-size: 8px;")