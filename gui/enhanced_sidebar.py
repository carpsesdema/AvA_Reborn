# gui/enhanced_sidebar.py - Fixed Project Panel Layout

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSlider,
    QListWidget, QListWidgetItem, QProgressBar, QFrame, QPushButton,
    QScrollArea, QSizePolicy, QLayout
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFont

from gui.components import ModernButton, StatusIndicator


class StyledPanel(QFrame):
    """Base panel with AvA styling matching your design"""

    def __init__(self, title="", collapsible=False, initially_collapsed=False):
        super().__init__()
        self.is_collapsed = initially_collapsed
        self.collapsible = collapsible
        self._title_text = title

        self.setFrameStyle(QFrame.Shape.NoFrame)
        self.setStyleSheet("""
            StyledPanel {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2a2a2e, stop:1 #252526);
                border: 2px solid #00d7ff;
                border-radius: 8px;
                margin: 4px 2px;
            }
        """)

        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(12, 8, 12, 12)
        self.main_layout.setSpacing(8)

        if title:
            self._create_header(title)

        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(8)

        self.main_layout.addWidget(self.content_widget)
        self.setLayout(self.main_layout)
        self._apply_current_visual_state()

    def _create_header(self, title):
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 6)

        self.title_label = QLabel(title)
        self.title_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.title_label.setStyleSheet("""
            QLabel {
                color: #00d7ff;
                background: transparent;
                border: none;
                padding: 4px 0px;
            }
        """)

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

    def add_widget(self, widget: QWidget):
        self.content_layout.addWidget(widget)

    def add_layout(self, layout: QHBoxLayout | QVBoxLayout):
        self.content_layout.addLayout(layout)

    def _apply_current_visual_state(self):
        self.content_widget.setVisible(not self.is_collapsed if self.collapsible else True)

        if self.collapsible and hasattr(self, 'collapse_btn'):
            self.collapse_btn.setText("+" if self.is_collapsed else "−")

        if self.collapsible and self.is_collapsed:
            self.content_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            self.content_widget.setFixedHeight(0)
        else:
            self.content_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
            self.content_widget.setMinimumHeight(50)
            self.content_widget.setMaximumHeight(16777215)

        if self.layout():
            self.layout().activate()
        self.updateGeometry()

    def _toggle_collapse(self):
        if not self.collapsible:
            return
        self.is_collapsed = not self.is_collapsed
        self._apply_current_visual_state()


class ProjectSessionPanel(StyledPanel):
    project_changed = Signal(str)
    session_changed = Signal(str)

    def __init__(self, collapsible=False, initially_collapsed=False):
        super().__init__(title="", collapsible=collapsible, initially_collapsed=initially_collapsed)
        self._init_ui()

    def _init_ui(self):
        # Tab buttons for Projects/Sessions with better spacing
        tab_layout = QHBoxLayout()
        tab_layout.setContentsMargins(0, 0, 0, 12)  # More bottom spacing
        tab_layout.setSpacing(6)

        self.projects_btn = QPushButton("Projects")
        self.sessions_btn = QPushButton("Sessions")

        tab_style = """
            QPushButton {
                background: #2d2d30; color: #cccccc; border: 1px solid #404040;
                border-radius: 6px; padding: 8px 16px; font-weight: 500; font-size: 11px;
                min-height: 28px;
            } QPushButton:hover { background: #3e3e42; color: white; }
        """
        active_tab_style = """
            QPushButton {
                background: #00d7ff; color: #1e1e1e; border: 1px solid #00d7ff;
                border-radius: 6px; padding: 8px 16px; font-weight: bold; font-size: 11px;
                min-height: 28px;
            }
        """
        self.projects_btn.setStyleSheet(active_tab_style)
        self.sessions_btn.setStyleSheet(tab_style)
        self.projects_btn.setCheckable(True)
        self.sessions_btn.setCheckable(True)
        self.projects_btn.setChecked(True)
        self.projects_btn.clicked.connect(self._switch_to_projects)
        self.sessions_btn.clicked.connect(self._switch_to_sessions)
        tab_layout.addWidget(self.projects_btn)
        tab_layout.addWidget(self.sessions_btn)
        self.add_layout(tab_layout)

        # Projects section with better spacing
        projects_label = QLabel("Active Projects:")
        projects_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Medium))
        projects_label.setStyleSheet("color: #cccccc; margin: 4px 0px;")
        self.add_widget(projects_label)

        # Project list with better sizing
        self.project_list = QListWidget()
        self.project_list.setStyleSheet("""
            QListWidget {
                background: #1e1e1e; 
                border: 1px solid #404040; 
                border-radius: 6px; 
                color: #cccccc; 
                outline: none;
                padding: 4px;
            }
            QListWidget::item { 
                padding: 8px 10px; 
                border-bottom: 1px solid #2d2d30; 
                border-radius: 4px; 
                margin: 2px 1px;
                min-height: 24px;
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
        self._add_sample_projects()
        self.project_list.setMinimumHeight(100)  # Better minimum height
        self.project_list.setMaximumHeight(120)  # Controlled max height
        self.add_widget(self.project_list)

        # New Project button with better sizing
        self.new_project_btn = ModernButton("📁 New Project", button_type="primary")
        self.new_project_btn.setMinimumHeight(36)  # Bigger button
        self.new_project_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0078d4, stop:1 #005a9e);
                border: 1px solid #004578;
                border-radius: 6px;
                color: white;
                padding: 8px 12px;
                font-weight: 600;
                font-size: 11px;
                min-height: 36px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #106ebe, stop:1 #005a9e);
                border-color: #0078d4;
            }
            QPushButton:pressed {
                background: #004578;
                border-color: #003d5b;
            }
        """)
        self.add_widget(self.new_project_btn)

        # Add spacing before sessions
        spacing_widget = QWidget()
        spacing_widget.setFixedHeight(8)
        self.add_widget(spacing_widget)

        # Sessions section with better spacing
        sessions_label = QLabel("Sessions (Current Project):")
        sessions_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Medium))
        sessions_label.setStyleSheet("color: #cccccc; margin: 4px 0px;")
        self.add_widget(sessions_label)

        # Session progress with better styling
        self.session_progress = QProgressBar()
        self.session_progress.setValue(45)
        self.session_progress.setFixedHeight(8)  # Slightly thicker
        self.session_progress.setStyleSheet("""
            QProgressBar {
                background: #2d2d30;
                border: none;
                border-radius: 4px;
                margin: 2px 0px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00d7ff, stop:1 #0078d4);
                border-radius: 4px;
            }
        """)
        self.add_widget(self.session_progress)

        # Session list with better sizing
        self.session_list = QListWidget()
        self.session_list.setStyleSheet(self.project_list.styleSheet())
        self._add_sample_sessions()
        self.session_list.setMinimumHeight(60)  # Better minimum height
        self.session_list.setMaximumHeight(80)  # Controlled max height
        self.add_widget(self.session_list)

    def _add_sample_projects(self):
        projects = [("Default Project", 75), ("Web Scraper", 23), ("Discord Bot", 91)]
        for name, prog in projects:
            item = QListWidgetItem()
            item_widget = QWidget()
            item_layout = QVBoxLayout(item_widget)
            item_layout.setContentsMargins(6, 4, 6, 4)
            item_layout.setSpacing(3)

            name_lbl = QLabel(name)
            name_lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.Medium))
            name_lbl.setStyleSheet("color: white; background: transparent;")

            prog_bar = QProgressBar()
            prog_bar.setValue(prog)
            prog_bar.setFixedHeight(4)
            prog_bar.setStyleSheet("""
                QProgressBar {
                    background: #404040;
                    border: none;
                    border-radius: 2px;
                }
                QProgressBar::chunk {
                    background: #00d7ff;
                    border-radius: 2px;
                }
            """)

            item_layout.addWidget(name_lbl)
            item_layout.addWidget(prog_bar)
            item.setSizeHint(item_widget.sizeHint())
            self.project_list.addItem(item)
            self.project_list.setItemWidget(item, item_widget)

        if self.project_list.count() > 0:
            self.project_list.setCurrentRow(0)

    def _add_sample_sessions(self):
        sessions = ["Main Chat", "Code Review", "Bug Fixes"]
        for session_name in sessions:
            item = QListWidgetItem(session_name)
            item.setFont(QFont("Segoe UI", 10))
            self.session_list.addItem(item)
        if self.session_list.count() > 0:
            self.session_list.setCurrentRow(0)

    def _switch_to_projects(self):
        active = """
            QPushButton {
                background: #00d7ff; color: #1e1e1e; border: 1px solid #00d7ff;
                border-radius: 6px; padding: 8px 16px; font-weight: bold; font-size: 11px;
                min-height: 28px;
            }
        """
        inactive = """
            QPushButton {
                background: #2d2d30; color: #cccccc; border: 1px solid #404040;
                border-radius: 6px; padding: 8px 16px; font-weight: 500; font-size: 11px;
                min-height: 28px;
            } QPushButton:hover { background: #3e3e42; color: white; }
        """
        self.projects_btn.setStyleSheet(active)
        self.sessions_btn.setStyleSheet(inactive)
        self.projects_btn.setChecked(True)
        self.sessions_btn.setChecked(False)

    def _switch_to_sessions(self):
        active = """
            QPushButton {
                background: #00d7ff; color: #1e1e1e; border: 1px solid #00d7ff;
                border-radius: 6px; padding: 8px 16px; font-weight: bold; font-size: 11px;
                min-height: 28px;
            }
        """
        inactive = """
            QPushButton {
                background: #2d2d30; color: #cccccc; border: 1px solid #404040;
                border-radius: 6px; padding: 8px 16px; font-weight: 500; font-size: 11px;
                min-height: 28px;
            } QPushButton:hover { background: #3e3e42; color: white; }
        """
        self.sessions_btn.setStyleSheet(active)
        self.projects_btn.setStyleSheet(inactive)
        self.sessions_btn.setChecked(True)
        self.projects_btn.setChecked(False)


class LLMConfigPanel(StyledPanel):
    model_changed = Signal(str, str)
    temperature_changed = Signal(float)

    def __init__(self, collapsible=False, initially_collapsed=False):
        super().__init__("LLM Configuration", collapsible=collapsible, initially_collapsed=initially_collapsed)
        self._init_ui()

    def _init_ui(self):
        # Chat LLM Section with better spacing
        chat_layout = QHBoxLayout()
        chat_layout.setContentsMargins(0, 4, 0, 8)
        chat_layout.setSpacing(8)

        chat_label = QLabel("Chat LLM:")
        chat_label.setFont(QFont("Segoe UI", 9))
        chat_label.setStyleSheet("color: #cccccc; background: transparent;")
        chat_label.setMinimumWidth(80)

        self.chat_combo = QComboBox()
        self.chat_combo.addItems([
            "Gemini: gemini-2.5-pro-preview",
            "OpenAI: gpt-4o",
            "Anthropic: claude-3.5-sonnet",
            "DeepSeek: deepseek-chat"
        ])
        self.chat_combo.setStyleSheet("""
            QComboBox {
                background: #1e1e1e;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 6px 10px;
                color: #cccccc;
                min-width: 140px;
                font-size: 9px;
                min-height: 24px;
            }
            QComboBox:hover {
                border-color: #00d7ff;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 4px solid #cccccc;
                margin-right: 6px;
            }
            QComboBox QAbstractItemView {
                background: #2d2d30;
                border: 1px solid #00d7ff;
                selection-background-color: #00d7ff;
                selection-color: #1e1e1e;
                color: #cccccc;
            }
        """)

        chat_status = StatusIndicator("ready")

        chat_layout.addWidget(chat_label)
        chat_layout.addWidget(self.chat_combo, 1)
        chat_layout.addWidget(chat_status)
        self.add_layout(chat_layout)

        # Specialized LLM Section
        code_label = QLabel("Specialized LLM (Code Gen):")
        code_label.setFont(QFont("Segoe UI", 9))
        code_label.setStyleSheet("color: #cccccc; background: transparent; margin: 4px 0px;")
        self.add_widget(code_label)

        code_row = QHBoxLayout()
        code_row.setSpacing(8)
        self.code_combo = QComboBox()
        self.code_combo.addItems([
            "Ollama (Gen): qwen2.5-coder",
            "OpenAI: gpt-4o",
            "Anthropic: claude-3.5-sonnet",
            "DeepSeek: deepseek-coder-v2"
        ])
        self.code_combo.setStyleSheet(self.chat_combo.styleSheet())
        code_status = StatusIndicator("ready")
        code_row.addWidget(self.code_combo, 1)
        code_row.addWidget(code_status)
        self.add_layout(code_row)

        # Temperature Slider Section with better spacing
        temp_label_layout = QHBoxLayout()
        temp_label_layout.setContentsMargins(0, 12, 0, 4)
        temp_label = QLabel("Temperature (Chat):")
        temp_label.setFont(QFont("Segoe UI", 9))
        temp_label.setStyleSheet("color: #cccccc; background: transparent;")
        self.temp_value = QLabel("0.70")
        self.temp_value.setStyleSheet("color: #00d7ff; font-weight: bold; background: transparent;")
        temp_label_layout.addWidget(temp_label)
        temp_label_layout.addStretch()
        temp_label_layout.addWidget(self.temp_value)
        self.add_layout(temp_label_layout)

        # Temperature slider with better styling
        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setRange(0, 100)
        self.temp_slider.setValue(70)
        self.temp_slider.setMinimumHeight(24)
        self.temp_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #404040;
                height: 6px;
                background: #1e1e1e;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #00d7ff;
                border: 2px solid #00d7ff;
                width: 16px;
                height: 16px;
                border-radius: 8px;
                margin: -6px 0;
            }
            QSlider::handle:horizontal:hover {
                background: #40e0ff;
                border-color: #40e0ff;
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00d7ff, stop:1 #0078d4);
                border-radius: 3px;
            }
        """)
        self.temp_slider.valueChanged.connect(self._on_temperature_changed)
        self.add_widget(self.temp_slider)

        # Configure Persona Button with spacing
        persona_spacing = QWidget()
        persona_spacing.setFixedHeight(8)
        self.add_widget(persona_spacing)

        self.persona_btn = ModernButton("🎭 Configure Persona", button_type="secondary")
        self.persona_btn.setMinimumHeight(32)
        self.add_widget(self.persona_btn)

    def _on_temperature_changed(self, value):
        temp_val = value / 100.0
        self.temp_value.setText(f"{temp_val:.2f}")
        self.temperature_changed.emit(temp_val)


class KnowledgeBasePanel(StyledPanel):
    def __init__(self, collapsible=False, initially_collapsed=False):
        super().__init__("Knowledge Base (RAG)", collapsible=collapsible, initially_collapsed=initially_collapsed)
        self._init_ui()

    def _init_ui(self):
        # Scan Directory button with better spacing
        self.scan_btn = ModernButton("🌐 Scan Directory (Global)", button_type="secondary")
        self.scan_btn.setMinimumHeight(32)
        self.add_widget(self.scan_btn)

        # Add spacing
        spacing_widget = QWidget()
        spacing_widget.setFixedHeight(6)
        self.add_widget(spacing_widget)

        # Add Files button
        self.add_files_btn = ModernButton("📄 Add Files (Project)", button_type="secondary")
        self.add_files_btn.setMinimumHeight(32)
        self.add_widget(self.add_files_btn)

        # RAG Status with better spacing
        rag_status_layout = QHBoxLayout()
        rag_status_layout.setContentsMargins(0, 8, 0, 0)

        rag_status_label = QLabel("RAG:")
        rag_status_label.setFont(QFont("Segoe UI", 9))
        rag_status_label.setStyleSheet("color: #cccccc; background: transparent;")

        self.rag_status_display_label = QLabel("Initializing embedder...")
        self.rag_status_display_label.setStyleSheet("color: #ffb900; font-size: 9px; background: transparent;")

        rag_status_layout.addWidget(rag_status_label)
        rag_status_layout.addWidget(self.rag_status_display_label, 1)
        self.add_layout(rag_status_layout)


class ChatActionsPanel(StyledPanel):
    action_triggered = Signal(str)

    def __init__(self, collapsible=False, initially_collapsed=False):
        super().__init__("Chat Actions", collapsible=collapsible, initially_collapsed=initially_collapsed)
        self._init_ui()

    def _init_ui(self):
        # Create buttons with consistent styling and better spacing
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
            btn.setMinimumHeight(32)
            btn.clicked.connect(lambda checked, a=action: self.action_triggered.emit(a))
            self.add_widget(btn)

            # Add small spacing between buttons (except last one)
            if i < len(buttons) - 1:
                spacing_widget = QWidget()
                spacing_widget.setFixedHeight(3)
                self.add_widget(spacing_widget)


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
        main_layout.setContentsMargins(8, 8, 8, 8)  # Better outer margins
        main_layout.setSpacing(6)  # Consistent spacing between panels

        # Create scrollable area for panels
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background: #2d2d30;
                width: 10px;
                border-radius: 5px;
                margin: 2px;
            }
            QScrollBar::handle:vertical {
                background: #404040;
                border-radius: 5px;
                min-height: 24px;
            }
            QScrollBar::handle:vertical:hover {
                background: #00d7ff;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
                border: none;
                background: none;
            }
            QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {
                background: none;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        """)

        # Content widget for scroll area
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(8)  # Better spacing between panels

        # Create all panels with proper sizing
        self.project_panel = ProjectSessionPanel(collapsible=False)
        self.llm_panel = LLMConfigPanel(collapsible=True, initially_collapsed=False)
        self.rag_panel = KnowledgeBasePanel(collapsible=True, initially_collapsed=True)
        self.actions_panel = ChatActionsPanel(collapsible=True, initially_collapsed=True)

        # Add panels to content layout
        content_layout.addWidget(self.project_panel)
        content_layout.addWidget(self.llm_panel)
        content_layout.addWidget(self.rag_panel)
        content_layout.addWidget(self.actions_panel)
        content_layout.addStretch()  # Push everything to the top

        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)

        # Apply overall styling
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
            self.rag_panel.rag_status_display_label.setStyleSheet(
                f"color: {color_hex}; font-size: 9px; background: transparent;"
            )