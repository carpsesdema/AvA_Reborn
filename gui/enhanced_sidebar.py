# gui/enhanced_sidebar.py - Simple Clean Fix

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSlider,
    QListWidget, QListWidgetItem, QProgressBar, QFrame, QPushButton,
    QScrollArea, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFont

from gui.components import ModernButton, StatusIndicator


class StyledPanel(QFrame):
    """Base panel with clean styling"""

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
                margin: 3px;
            }
        """)

        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(12, 10, 12, 12)
        self.main_layout.setSpacing(10)

        if title:
            self._create_header(title)

        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(8)

        self.main_layout.addWidget(self.content_widget)
        self.setLayout(self.main_layout)

        if self.collapsible and self.is_collapsed:
            self.content_widget.hide()

    def _create_header(self, title):
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 8)

        self.title_label = QLabel(title)
        self.title_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.title_label.setStyleSheet("color: #00d7ff; background: transparent;")

        header_layout.addWidget(self.title_label)
        header_layout.addStretch()

        if self.collapsible:
            self.collapse_btn = QPushButton("+" if self.is_collapsed else "−")
            self.collapse_btn.setFixedSize(20, 20)
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


class ProjectSessionPanel(StyledPanel):
    project_changed = Signal(str)
    session_changed = Signal(str)

    def __init__(self):
        super().__init__(title="", collapsible=False)
        self._init_ui()

    def _init_ui(self):
        # Tab buttons - cleaner layout
        tab_layout = QHBoxLayout()
        tab_layout.setContentsMargins(0, 0, 0, 15)
        tab_layout.setSpacing(8)

        self.projects_btn = QPushButton("Projects")
        self.sessions_btn = QPushButton("Sessions")

        # Better tab styling
        for btn in [self.projects_btn, self.sessions_btn]:
            btn.setMinimumHeight(32)
            btn.setCheckable(True)

        self.projects_btn.setChecked(True)
        self._update_tab_styles()

        self.projects_btn.clicked.connect(self._switch_to_projects)
        self.sessions_btn.clicked.connect(self._switch_to_sessions)

        tab_layout.addWidget(self.projects_btn)
        tab_layout.addWidget(self.sessions_btn)
        self.add_layout(tab_layout)

        # Projects section
        projects_label = QLabel("Active Projects:")
        projects_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Medium))
        projects_label.setStyleSheet("color: #cccccc; margin: 5px 0;")
        self.add_widget(projects_label)

        # Simple project list - no custom widgets
        self.project_list = QListWidget()
        self.project_list.setStyleSheet("""
            QListWidget {
                background: #1e1e1e; 
                border: 1px solid #404040; 
                border-radius: 6px; 
                color: #cccccc; 
                outline: none;
                padding: 4px;
                font-size: 12px;
            }
            QListWidget::item { 
                padding: 10px 12px; 
                border-bottom: 1px solid #2d2d30; 
                border-radius: 4px; 
                margin: 1px;
                min-height: 20px;
            }
            QListWidget::item:selected { 
                background: #00d7ff; 
                color: #1e1e1e; 
                font-weight: bold; 
            }
            QListWidget::item:hover { 
                background: #2d2d30; 
            }
        """)

        # Add simple text items
        self.project_list.addItem("Default Project")
        self.project_list.addItem("Web Scraper")
        self.project_list.addItem("Discord Bot")
        self.project_list.setCurrentRow(0)

        self.project_list.setMinimumHeight(120)
        self.project_list.setMaximumHeight(140)
        self.add_widget(self.project_list)

        # Bigger, better New Project button
        self.new_project_btn = ModernButton("📁 New Project", button_type="primary")
        self.new_project_btn.setMinimumHeight(40)
        self.new_project_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0078d4, stop:1 #005a9e);
                border: 1px solid #004578;
                border-radius: 6px;
                color: white;
                padding: 10px 15px;
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
        self.add_widget(self.new_project_btn)

        # Spacer
        spacer = QWidget()
        spacer.setFixedHeight(15)
        self.add_widget(spacer)

        # Sessions section
        sessions_label = QLabel("Sessions (Current Project):")
        sessions_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Medium))
        sessions_label.setStyleSheet("color: #cccccc; margin: 5px 0;")
        self.add_widget(sessions_label)

        # Simple progress bar
        self.session_progress = QProgressBar()
        self.session_progress.setValue(45)
        self.session_progress.setFixedHeight(8)
        self.session_progress.setStyleSheet("""
            QProgressBar {
                background: #2d2d30;
                border: none;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00d7ff, stop:1 #0078d4);
                border-radius: 4px;
            }
        """)
        self.add_widget(self.session_progress)

        # Simple session list
        self.session_list = QListWidget()
        self.session_list.setStyleSheet(self.project_list.styleSheet())
        self.session_list.addItem("Main Chat")
        self.session_list.addItem("Code Review")
        self.session_list.addItem("Bug Fixes")
        self.session_list.setCurrentRow(0)

        self.session_list.setMinimumHeight(80)
        self.session_list.setMaximumHeight(100)
        self.add_widget(self.session_list)

    def _update_tab_styles(self):
        active_style = """
            QPushButton {
                background: #00d7ff; color: #1e1e1e; border: 1px solid #00d7ff;
                border-radius: 6px; padding: 8px 16px; font-weight: bold; font-size: 11px;
            }
        """
        inactive_style = """
            QPushButton {
                background: #2d2d30; color: #cccccc; border: 1px solid #404040;
                border-radius: 6px; padding: 8px 16px; font-weight: 500; font-size: 11px;
            } QPushButton:hover { background: #3e3e42; color: white; }
        """

        if self.projects_btn.isChecked():
            self.projects_btn.setStyleSheet(active_style)
            self.sessions_btn.setStyleSheet(inactive_style)
        else:
            self.sessions_btn.setStyleSheet(active_style)
            self.projects_btn.setStyleSheet(inactive_style)

    def _switch_to_projects(self):
        self.projects_btn.setChecked(True)
        self.sessions_btn.setChecked(False)
        self._update_tab_styles()

    def _switch_to_sessions(self):
        self.sessions_btn.setChecked(True)
        self.projects_btn.setChecked(False)
        self._update_tab_styles()


class LLMConfigPanel(StyledPanel):
    model_changed = Signal(str, str)
    temperature_changed = Signal(float)

    def __init__(self):
        super().__init__("LLM Configuration", collapsible=True, initially_collapsed=False)
        self._init_ui()

    def _init_ui(self):
        # Chat LLM
        chat_layout = QHBoxLayout()
        chat_layout.setSpacing(8)

        chat_label = QLabel("Chat LLM:")
        chat_label.setFont(QFont("Segoe UI", 9))
        chat_label.setStyleSheet("color: #cccccc;")
        chat_label.setMinimumWidth(70)

        self.chat_combo = QComboBox()
        self.chat_combo.addItems([
            "Gemini: gemini-2.5-pro-preview",
            "OpenAI: gpt-4o",
            "Anthropic: claude-3.5-sonnet",
            "DeepSeek: deepseek-chat"
        ])
        self.chat_combo.setStyleSheet("""
            QComboBox {
                background: #1e1e1e; border: 1px solid #404040; border-radius: 4px;
                padding: 6px 10px; color: #cccccc; min-width: 120px; font-size: 9px;
            }
            QComboBox:hover { border-color: #00d7ff; }
            QComboBox::drop-down { border: none; width: 20px; }
            QComboBox::down-arrow {
                image: none; border-left: 4px solid transparent; border-right: 4px solid transparent;
                border-top: 4px solid #cccccc; margin-right: 6px;
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

        # Code LLM
        code_label = QLabel("Specialized LLM (Code Gen):")
        code_label.setFont(QFont("Segoe UI", 9))
        code_label.setStyleSheet("color: #cccccc; margin: 8px 0 4px 0;")
        self.add_widget(code_label)

        code_layout = QHBoxLayout()
        code_layout.setSpacing(8)

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

        # Temperature
        temp_header = QHBoxLayout()
        temp_header.setContentsMargins(0, 12, 0, 4)

        temp_label = QLabel("Temperature (Chat):")
        temp_label.setFont(QFont("Segoe UI", 9))
        temp_label.setStyleSheet("color: #cccccc;")

        self.temp_value = QLabel("0.70")
        self.temp_value.setStyleSheet("color: #00d7ff; font-weight: bold;")

        temp_header.addWidget(temp_label)
        temp_header.addStretch()
        temp_header.addWidget(self.temp_value)
        self.add_layout(temp_header)

        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setRange(0, 100)
        self.temp_slider.setValue(70)
        self.temp_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #404040; height: 6px; background: #1e1e1e; border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #00d7ff; border: 2px solid #00d7ff; width: 16px; height: 16px;
                border-radius: 8px; margin: -6px 0;
            }
            QSlider::handle:horizontal:hover { background: #40e0ff; border-color: #40e0ff; }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00d7ff, stop:1 #0078d4);
                border-radius: 3px;
            }
        """)
        self.temp_slider.valueChanged.connect(self._on_temperature_changed)
        self.add_widget(self.temp_slider)

        # Persona button
        spacer = QWidget()
        spacer.setFixedHeight(8)
        self.add_widget(spacer)

        self.persona_btn = ModernButton("🎭 Configure Persona", button_type="secondary")
        self.persona_btn.setMinimumHeight(32)
        self.add_widget(self.persona_btn)

    def _on_temperature_changed(self, value):
        temp_val = value / 100.0
        self.temp_value.setText(f"{temp_val:.2f}")
        self.temperature_changed.emit(temp_val)


class KnowledgeBasePanel(StyledPanel):
    def __init__(self):
        super().__init__("Knowledge Base (RAG)", collapsible=True, initially_collapsed=True)
        self._init_ui()

    def _init_ui(self):
        self.scan_btn = ModernButton("🌐 Scan Directory (Global)", button_type="secondary")
        self.scan_btn.setMinimumHeight(32)
        self.add_widget(self.scan_btn)

        spacer = QWidget()
        spacer.setFixedHeight(6)
        self.add_widget(spacer)

        self.add_files_btn = ModernButton("📄 Add Files (Project)", button_type="secondary")
        self.add_files_btn.setMinimumHeight(32)
        self.add_widget(self.add_files_btn)

        rag_status_layout = QHBoxLayout()
        rag_status_layout.setContentsMargins(0, 8, 0, 0)

        rag_label = QLabel("RAG:")
        rag_label.setFont(QFont("Segoe UI", 9))
        rag_label.setStyleSheet("color: #cccccc;")

        self.rag_status_display_label = QLabel("Initializing embedder...")
        self.rag_status_display_label.setStyleSheet("color: #ffb900; font-size: 9px;")

        rag_status_layout.addWidget(rag_label)
        rag_status_layout.addWidget(self.rag_status_display_label, 1)
        self.add_layout(rag_status_layout)


class ChatActionsPanel(StyledPanel):
    action_triggered = Signal(str)

    def __init__(self):
        super().__init__("Chat Actions", collapsible=True, initially_collapsed=True)
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

        for i, (text, action) in enumerate(buttons):
            btn = ModernButton(text, button_type="secondary")
            btn.setMinimumHeight(30)
            btn.clicked.connect(lambda checked, a=action: self.action_triggered.emit(a))
            self.add_widget(btn)

            if i < len(buttons) - 1:
                spacer = QWidget()
                spacer.setFixedHeight(3)
                self.add_widget(spacer)


class AvALeftSidebar(QWidget):
    model_changed = Signal(str, str)
    temperature_changed = Signal(float)
    action_triggered = Signal(str)

    def __init__(self):
        super().__init__()
        self.setFixedWidth(280)
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(4)

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
        content_layout.setSpacing(6)

        # Create panels
        self.project_panel = ProjectSessionPanel()
        self.llm_panel = LLMConfigPanel()
        self.rag_panel = KnowledgeBasePanel()
        self.actions_panel = ChatActionsPanel()

        content_layout.addWidget(self.project_panel)
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
        self.llm_panel.model_changed.connect(self.model_changed)
        self.llm_panel.temperature_changed.connect(self.temperature_changed)
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
            self.rag_panel.rag_status_display_label.setStyleSheet(f"color: {color_hex}; font-size: 9px;")