# gui/enhanced_sidebar.py - Complete Left Sidebar matching your design

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSlider,
    QListWidget, QListWidgetItem, QProgressBar, QFrame, QPushButton,
    QScrollArea, QSizePolicy  # Added QSizePolicy
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFont

from gui.components import ModernButton, StatusIndicator


class StyledPanel(QFrame):
    """Base panel with AvA styling matching your design"""

    def __init__(self, title="", collapsible=False, initially_collapsed=False):  # Added initially_collapsed
        super().__init__()
        self.is_collapsed = initially_collapsed  # Set initial logical state
        self.collapsible = collapsible
        self._title_text = title

        self.setFrameStyle(QFrame.Shape.NoFrame)  # NoFrame, styling is via stylesheet
        self.setStyleSheet("""
            StyledPanel {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2a2a2e, stop:1 #252526);
                border: 2px solid #00d7ff;
                border-radius: 8px;
                margin: 2px 2px; 
            }
        """)

        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(10, 6, 10, 8)
        self.main_layout.setSpacing(6)

        if title:  # Create header only if title is provided
            self._create_header(title)

        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(6)
        self.main_layout.addWidget(self.content_widget)

        self.setLayout(self.main_layout)

        if self.collapsible and self.is_collapsed:
            self.content_widget.setVisible(False)
            if hasattr(self, 'collapse_btn'):  # Ensure button exists
                self.collapse_btn.setText("+")

    def _create_header(self, title):
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 4)  # Small bottom margin for spacing

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
                    background: #00d7ff;
                    color: #1e1e1e;
                    border: none;
                    border-radius: 10px;
                    font-weight: bold;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background: #40e0ff;
                }
            """)
            self.collapse_btn.clicked.connect(self._toggle_collapse)
            header_layout.addWidget(self.collapse_btn)

        self.main_layout.insertLayout(0, header_layout)

    def _toggle_collapse(self):
        if not self.collapsible:
            return

        self.is_collapsed = not self.is_collapsed
        self.content_widget.setVisible(not self.is_collapsed)
        if hasattr(self, 'collapse_btn'):
            self.collapse_btn.setText("+" if self.is_collapsed else "−")

        if self.is_collapsed:
            self.content_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            self.content_widget.setFixedHeight(0)
        else:
            self.content_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            # Calculate minimum height based on content, or set a reasonable default
            min_h = self.content_layout.sizeHint().height() if self.content_layout.count() > 0 else 50
            self.content_widget.setMinimumHeight(min_h)
            self.content_widget.setMaximumHeight(16777215)

        if self.parentWidget() and self.parentWidget().layout():
            self.parentWidget().layout().activate()
        elif self.parentWidget():
            self.parentWidget().updateGeometry()

    def add_widget(self, widget):
        self.content_layout.addWidget(widget)

    def add_layout(self, layout):
        self.content_layout.addLayout(layout)

    def set_initial_collapse_state(self, collapsed: bool):
        if self.collapsible and collapsed != self.is_collapsed:
            self._toggle_collapse()
        elif self.collapsible and collapsed:
            self.is_collapsed = True  # Ensure logical state is correct
            self.content_widget.setVisible(False)
            if hasattr(self, 'collapse_btn'):
                self.collapse_btn.setText("+")
        elif self.collapsible and not collapsed:
            self.is_collapsed = False  # Ensure logical state is correct
            self.content_widget.setVisible(True)
            if hasattr(self, 'collapse_btn'):
                self.collapse_btn.setText("−")


class ProjectSessionPanel(StyledPanel):
    """Projects and Sessions panel matching your design"""

    project_changed = Signal(str)
    session_changed = Signal(str)

    def __init__(self, collapsible=False, initially_collapsed=False):  # Added parameters
        super().__init__(title="", collapsible=collapsible, initially_collapsed=initially_collapsed)  # Pass to super
        self._init_ui()
        if self.collapsible and self.is_collapsed:
            self.set_initial_collapse_state(True)

    def _init_ui(self):
        # Tab buttons for Projects/Sessions
        tab_layout = QHBoxLayout()
        tab_layout.setContentsMargins(0, 0, 0, 8)

        self.projects_btn = QPushButton("Projects")
        self.sessions_btn = QPushButton("Sessions")

        # Style tab buttons
        tab_style = """
            QPushButton {
                background: #2d2d30; color: #cccccc; border: 1px solid #404040;
                border-radius: 4px; padding: 6px 12px; font-weight: 500; font-size: 11px;
            } QPushButton:hover { background: #3e3e42; color: white; }
        """
        active_tab_style = """
            QPushButton {
                background: #00d7ff; color: #1e1e1e; border: 1px solid #00d7ff;
                border-radius: 4px; padding: 6px 12px; font-weight: bold; font-size: 11px;
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
        self.add_layout(tab_layout)  # Use the panel's add_layout method

        projects_label = QLabel("Projects:")
        projects_label.setFont(QFont("Segoe UI", 9, QFont.Weight.Medium))
        projects_label.setStyleSheet("color: #cccccc; margin-top: 4px;")
        self.add_widget(projects_label)
        self.project_list = QListWidget()
        self.project_list.setStyleSheet("""
            QListWidget { background: #1e1e1e; border: 1px solid #404040; border-radius: 4px; color: #cccccc; outline: none; }
            QListWidget::item { padding: 6px 8px; border-bottom: 1px solid #2d2d30; border-radius: 2px; margin: 1px; }
            QListWidget::item:selected { background: #00d7ff; color: #1e1e1e; font-weight: bold; }
            QListWidget::item:hover { background: #2d2d30; }""")
        self._add_sample_projects()
        self.project_list.setMaximumHeight(90)
        self.add_widget(self.project_list)
        self.new_project_btn = ModernButton("📁 New Project", button_type="primary")
        self.new_project_btn.setMinimumHeight(30)
        self.add_widget(self.new_project_btn)

        sessions_label = QLabel("Sessions (Current Project):")
        sessions_label.setFont(QFont("Segoe UI", 9, QFont.Weight.Medium))
        sessions_label.setStyleSheet("color: #cccccc; margin-top: 8px;")
        self.add_widget(sessions_label)
        self.session_progress = QProgressBar()
        self.session_progress.setValue(45)
        self.session_progress.setMaximumHeight(6)
        self.session_progress.setStyleSheet(
            "QProgressBar{background:#2d2d30;border:none;border-radius:3px;}QProgressBar::chunk{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #00d7ff,stop:1 #0078d4);border-radius:3px;}")
        self.add_widget(self.session_progress)
        self.session_list = QListWidget()
        self.session_list.setStyleSheet(self.project_list.styleSheet())
        self._add_sample_sessions()
        self.session_list.setMaximumHeight(70)
        self.add_widget(self.session_list)

    def _add_sample_projects(self):
        projects = [("Default Project", 75), ("365_crawl", 23)]
        for name, prog in projects:
            item = QListWidgetItem()
            item_widget = QWidget()
            item_layout = QVBoxLayout(item_widget)
            item_layout.setContentsMargins(4, 2, 4, 2)
            item_layout.setSpacing(2)
            name_lbl = QLabel(name)
            name_lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.Medium))
            name_lbl.setStyleSheet("color:white;background:transparent;")
            prog_bar = QProgressBar()
            prog_bar.setValue(prog)
            prog_bar.setMaximumHeight(3)
            prog_bar.setStyleSheet(
                "QProgressBar{background:#404040;border:none;border-radius:1px;}QProgressBar::chunk{background:#00d7ff;border-radius:1px;}")
            item_layout.addWidget(name_lbl)
            item_layout.addWidget(prog_bar)
            item.setSizeHint(item_widget.sizeHint())
            self.project_list.addItem(item)
            self.project_list.setItemWidget(item, item_widget)
        if self.project_list.count() > 0: self.project_list.setCurrentRow(0)

    def _add_sample_sessions(self):
        for session_name in ["Main Chat"]: self.session_list.addItem(QListWidgetItem(session_name))
        if self.session_list.count() > 0: self.session_list.setCurrentRow(0)

    def _switch_to_projects(self):
        active = "QPushButton{background:#00d7ff;color:#1e1e1e;border:1px solid #00d7ff;border-radius:4px;padding:6px 12px;font-weight:bold;font-size:11px;}"
        inactive = "QPushButton{background:#2d2d30;color:#cccccc;border:1px solid #404040;border-radius:4px;padding:6px 12px;font-weight:500;font-size:11px;}"
        self.projects_btn.setStyleSheet(active)
        self.sessions_btn.setStyleSheet(inactive)
        self.projects_btn.setChecked(True)
        self.sessions_btn.setChecked(False)

    def _switch_to_sessions(self):
        active = "QPushButton{background:#00d7ff;color:#1e1e1e;border:1px solid #00d7ff;border-radius:4px;padding:6px 12px;font-weight:bold;font-size:11px;}"
        inactive = "QPushButton{background:#2d2d30;color:#cccccc;border:1px solid #404040;border-radius:4px;padding:6px 12px;font-weight:500;font-size:11px;}"
        self.sessions_btn.setStyleSheet(active)
        self.projects_btn.setStyleSheet(inactive)
        self.sessions_btn.setChecked(True)
        self.projects_btn.setChecked(False)


class LLMConfigPanel(StyledPanel):
    model_changed = Signal(str, str)
    temperature_changed = Signal(float)

    def __init__(self, collapsible=False, initially_collapsed=False):  # Added parameters
        super().__init__("LLM Configuration", collapsible=collapsible,
                         initially_collapsed=initially_collapsed)  # Pass to super
        self._init_ui()
        if self.collapsible and self.is_collapsed:
            self.set_initial_collapse_state(True)

    def _init_ui(self):
        chat_layout = QHBoxLayout()
        chat_layout.setContentsMargins(0, 4, 0, 4)
        chat_label = QLabel("Chat LLM:")
        chat_label.setFont(QFont("Segoe UI", 9))
        chat_label.setStyleSheet("color:#cccccc;background:transparent;")
        chat_label.setMinimumWidth(70)
        self.chat_combo = QComboBox()
        self.chat_combo.addItems(["Gemini: gemini-2.5-pro-preview", "OpenAI: gpt-4o", "Anthropic: claude-3.5-sonnet",
                                  "DeepSeek: deepseek-chat"])
        self.chat_combo.setStyleSheet(
            "QComboBox{background:#1e1e1e;border:1px solid #404040;border-radius:4px;padding:4px 8px;color:#cccccc;min-width:120px;font-size:9px;} QComboBox:hover{border-color:#00d7ff;} QComboBox::drop-down{border:none;width:18px;} QComboBox::down-arrow{image:none;border-left:4px solid transparent;border-right:4px solid transparent;border-top:4px solid #cccccc;margin-right:6px;} QComboBox QAbstractItemView{background:#2d2d30;border:1px solid #00d7ff;selection-background-color:#00d7ff;selection-color:#1e1e1e;color:#cccccc;}")
        chat_status = StatusIndicator("ready")
        chat_layout.addWidget(chat_label)
        chat_layout.addWidget(self.chat_combo, 1)
        chat_layout.addWidget(chat_status)
        self.add_layout(chat_layout)
        code_label = QLabel("Specialized LLM (Code Gen):")
        code_label.setFont(QFont("Segoe UI", 9))
        code_label.setStyleSheet("color:#cccccc;background:transparent;")
        self.add_widget(code_label)
        code_row = QHBoxLayout()
        self.code_combo = QComboBox()
        self.code_combo.addItems(["Ollama (Gen): qwen2.5-coder", "OpenAI: gpt-4o", "Anthropic: claude-3.5-sonnet",
                                  "DeepSeek: deepseek-coder-v2"])
        self.code_combo.setStyleSheet(self.chat_combo.styleSheet())
        code_status = StatusIndicator("ready")
        code_row.addWidget(self.code_combo, 1)
        code_row.addWidget(code_status)
        self.add_layout(code_row)
        temp_label_layout = QHBoxLayout()
        temp_label = QLabel("Temperature (Chat):")
        temp_label.setFont(QFont("Segoe UI", 9))
        temp_label.setStyleSheet("color:#cccccc;background:transparent;")
        self.temp_value = QLabel("0.70")
        self.temp_value.setStyleSheet("color:#00d7ff;font-weight:bold;background:transparent;")
        temp_label_layout.addWidget(temp_label)
        temp_label_layout.addStretch()
        temp_label_layout.addWidget(self.temp_value)
        self.add_layout(temp_label_layout)
        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setRange(0, 100)
        self.temp_slider.setValue(70)
        self.temp_slider.setStyleSheet(
            "QSlider::groove:horizontal{border:1px solid #404040;height:4px;background:#1e1e1e;border-radius:2px;} QSlider::handle:horizontal{background:#00d7ff;border:2px solid #00d7ff;width:14px;height:14px;border-radius:7px;margin:-6px 0;} QSlider::handle:horizontal:hover{background:#40e0ff;border-color:#40e0ff;} QSlider::sub-page:horizontal{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #00d7ff,stop:1 #0078d4);border-radius:2px;}")
        self.temp_slider.valueChanged.connect(self._on_temperature_changed)
        self.add_widget(self.temp_slider)
        self.persona_btn = ModernButton("🎭 Configure Persona", button_type="secondary")
        self.persona_btn.setMinimumHeight(28)
        self.add_widget(self.persona_btn)

    def _on_temperature_changed(self, value):
        temp_val = value / 100.0
        self.temp_value.setText(f"{temp_val:.2f}")
        self.temperature_changed.emit(temp_val)


class KnowledgeBasePanel(StyledPanel):
    def __init__(self, collapsible=False, initially_collapsed=False):  # Added parameters
        super().__init__("Knowledge Base (RAG)", collapsible=collapsible,
                         initially_collapsed=initially_collapsed)  # Pass to super
        self._init_ui()
        if self.collapsible and self.is_collapsed:
            self.set_initial_collapse_state(True)

    def _init_ui(self):
        self.scan_btn = ModernButton("🌐 Scan Directory (Global)", button_type="secondary")
        self.scan_btn.setMinimumHeight(28)
        self.add_widget(self.scan_btn)
        self.add_files_btn = ModernButton("📄 Add Files (Project)", button_type="secondary")
        self.add_files_btn.setMinimumHeight(28)
        self.add_widget(self.add_files_btn)
        rag_status_layout = QHBoxLayout()
        rag_status_layout.setContentsMargins(0, 4, 0, 0)
        rag_status_label = QLabel("RAG:")
        rag_status_label.setFont(QFont("Segoe UI", 9))
        rag_status_label.setStyleSheet("color:#cccccc;background:transparent;")
        self.rag_status_display_label = QLabel("Initializing embedder...")  # Made instance variable
        self.rag_status_display_label.setStyleSheet("color:#ffb900;font-size:9px;background:transparent;")
        rag_status_layout.addWidget(rag_status_label)
        rag_status_layout.addWidget(self.rag_status_display_label, 1)
        self.add_layout(rag_status_layout)


class ChatActionsPanel(StyledPanel):
    action_triggered = Signal(str)

    def __init__(self, collapsible=False, initially_collapsed=False):  # Added parameters
        super().__init__("Chat Actions", collapsible=collapsible,
                         initially_collapsed=initially_collapsed)  # Pass to super
        self._init_ui()
        if self.collapsible and self.is_collapsed:
            self.set_initial_collapse_state(True)

    def _init_ui(self):
        buttons = [("💬 New Session", "new_session"), ("📊 View LLM Log", "view_log"),
                   ("📟 Open Terminal", "open_terminal"), ("📄 Open Code Viewer", "open_code_viewer"),
                   ("⚡ View Generated Code", "view_code"), ("🔨 Force Code Gen", "force_gen"),
                   ("🔄 Check for Updates", "check_updates")]
        for text, action_id in buttons:
            btn = ModernButton(text, button_type="secondary")
            btn.setMinimumHeight(28)
            btn.clicked.connect(lambda checked=False, a=action_id: self.action_triggered.emit(a))
            self.add_widget(btn)


class AvALeftSidebar(QWidget):
    """Complete left sidebar containing all panels"""
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
        main_layout.setSpacing(0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea { background: transparent; border: none; }
            QScrollBar:vertical { background: #2d2d30; width: 8px; border-radius: 4px; margin: 0px; }
            QScrollBar::handle:vertical { background: #404040; border-radius: 4px; min-height: 20px; }
            QScrollBar::handle:vertical:hover { background: #00d7ff; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; border: none; background: none; }
            QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical { background: none; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
        """)

        content_widget = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(6)

        # Instantiate panels with collapsible properties
        self.project_panel = ProjectSessionPanel(collapsible=False)
        self.llm_panel = LLMConfigPanel(collapsible=True, initially_collapsed=False)
        self.rag_panel = KnowledgeBasePanel(collapsible=True, initially_collapsed=True)
        self.actions_panel = ChatActionsPanel(collapsible=True, initially_collapsed=True)

        content_layout.addWidget(self.project_panel)
        content_layout.addWidget(self.llm_panel)
        content_layout.addWidget(self.rag_panel)
        content_layout.addWidget(self.actions_panel)
        content_layout.addStretch()

        content_widget.setLayout(content_layout)
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

        # Ensure initial visual state for panels that start collapsed
        # The StyledPanel __init__ and set_initial_collapse_state should handle this.
        # This is an explicit call to ensure it if there were any issues with initial sizing.
        if self.rag_panel.is_collapsed:  # is_collapsed is set by initially_collapsed=True
            self.rag_panel.set_initial_collapse_state(True)
        if self.actions_panel.is_collapsed:
            self.actions_panel.set_initial_collapse_state(True)

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

    def update_sidebar_rag_status(self, status_text: str, color_hex: str):  # Added method
        if hasattr(self.rag_panel, 'rag_status_display_label'):
            self.rag_panel.rag_status_display_label.setText(status_text)
            self.rag_panel.rag_status_display_label.setStyleSheet(
                f"color: {color_hex}; font-size: 9px; background: transparent;"
            )