# gui/panels.py - Professional UI Panels

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSlider,
    QListWidget, QListWidgetItem, QProgressBar, QLineEdit, QFrame
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFont

from gui.components import ModernButton, ModernPanel, StatusIndicator


class LLMConfigPanel(ModernPanel):
    """LLM Configuration panel matching the old design"""

    model_changed = Signal(str, str)  # (model_type, model_name)

    def __init__(self):
        super().__init__("LLM Configuration")
        self._init_ui()

    def _init_ui(self):
        # Chat LLM Section
        chat_layout = QHBoxLayout()
        chat_label = QLabel("Chat LLM:")
        chat_label.setFont(QFont("Segoe UI", 9))
        chat_label.setStyleSheet("color: #cccccc;")

        self.chat_combo = QComboBox()
        self.chat_combo.addItems([
            "gemini-2.5-pro-preview",
            "gpt-4o",
            "claude-3.5-sonnet",
            "deepseek-chat"
        ])
        self.chat_combo.setStyleSheet("""
            QComboBox {
                background: #1e1e1e;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 6px 8px;
                color: #cccccc;
                min-width: 200px;
            }
            QComboBox:hover {
                border-color: #0078d4;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-style: none;
            }
        """)

        chat_status = StatusIndicator("ready")

        chat_layout.addWidget(chat_label)
        chat_layout.addWidget(self.chat_combo, 1)
        chat_layout.addWidget(chat_status)
        self.add_widget(QWidget())
        self.content_layout.itemAt(-1).widget().setLayout(chat_layout)

        # Specialized LLM Section
        code_layout = QHBoxLayout()
        code_label = QLabel("Specialized LLM (Code Gen):")
        code_label.setFont(QFont("Segoe UI", 9))
        code_label.setStyleSheet("color: #cccccc;")

        self.code_combo = QComboBox()
        self.code_combo.addItems([
            "qwen2.5-coder-32b-instruct",
            "codellama-13b-instruct",
            "deepseek-coder-v2",
            "claude-3.5-sonnet"
        ])
        self.code_combo.setStyleSheet(self.chat_combo.styleSheet())

        code_status = StatusIndicator("ready")

        code_layout.addWidget(code_label)
        code_layout.addWidget(self.code_combo, 1)
        code_layout.addWidget(code_status)
        self.add_widget(QWidget())
        self.content_layout.itemAt(-1).widget().setLayout(code_layout)

        # Temperature Slider
        temp_layout = QVBoxLayout()
        temp_label_layout = QHBoxLayout()
        temp_label = QLabel("Temperature (Chat):")
        temp_label.setFont(QFont("Segoe UI", 9))
        temp_label.setStyleSheet("color: #cccccc;")

        self.temp_value = QLabel("0.70")
        self.temp_value.setStyleSheet("color: #0078d4; font-weight: bold;")

        temp_label_layout.addWidget(temp_label)
        temp_label_layout.addStretch()
        temp_label_layout.addWidget(self.temp_value)

        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setRange(0, 100)
        self.temp_slider.setValue(70)
        self.temp_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #404040;
                height: 6px;
                background: #1e1e1e;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #0078d4;
                border: 2px solid #0078d4;
                width: 16px;
                height: 16px;
                border-radius: 8px;
                margin: -6px 0;
            }
            QSlider::handle:horizontal:hover {
                background: #00d7ff;
                border-color: #00d7ff;
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0078d4, stop:1 #00d7ff);
                border-radius: 3px;
            }
        """)
        self.temp_slider.valueChanged.connect(lambda v: self.temp_value.setText(f"{v / 100:.2f}"))

        temp_layout.addLayout(temp_label_layout)
        temp_layout.addWidget(self.temp_slider)
        self.add_widget(QWidget())
        self.content_layout.itemAt(-1).widget().setLayout(temp_layout)

        # Configure Persona Button
        self.persona_btn = ModernButton("🎭 Configure Persona", button_type="secondary")
        self.add_widget(self.persona_btn)


class ProjectPanel(ModernPanel):
    """Project management panel"""

    project_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        # Projects/Sessions tabs
        tab_layout = QHBoxLayout()
        tab_layout.setContentsMargins(0, 0, 0, 8)

        self.projects_tab = ModernButton("Projects", button_type="accent")
        self.sessions_tab = ModernButton("Sessions", button_type="secondary")

        self.projects_tab.setCheckable(True)
        self.sessions_tab.setCheckable(True)
        self.projects_tab.setChecked(True)

        tab_layout.addWidget(self.projects_tab)
        tab_layout.addWidget(self.sessions_tab)
        tab_layout.addStretch()

        self.add_widget(QWidget())
        self.content_layout.itemAt(-1).widget().setLayout(tab_layout)

        # Projects section
        projects_label = QLabel("Projects:")
        projects_label.setFont(QFont("Segoe UI", 9))
        projects_label.setStyleSheet("color: #cccccc; margin-top: 8px;")
        self.add_widget(projects_label)

        # Project list with progress bar
        self.project_list = QListWidget()
        self.project_list.setStyleSheet("""
            QListWidget {
                background: #1e1e1e;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 4px;
                color: #cccccc;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #2d2d30;
                border-radius: 4px;
                margin: 1px 0;
            }
            QListWidget::item:selected {
                background: #0078d4;
                color: white;
            }
            QListWidget::item:hover {
                background: #2d2d30;
            }
        """)

        # Add sample project with progress
        project_item = QListWidgetItem()
        project_widget = QWidget()
        project_layout = QVBoxLayout()
        project_layout.setContentsMargins(0, 0, 0, 0)
        project_layout.setSpacing(4)

        project_name = QLabel("Default Project")
        project_name.setFont(QFont("Segoe UI", 10, QFont.Weight.Medium))
        project_name.setStyleSheet("color: white;")

        progress_bar = QProgressBar()
        progress_bar.setValue(75)
        progress_bar.setMaximumHeight(4)
        progress_bar.setStyleSheet("""
            QProgressBar {
                background: #2d2d30;
                border: none;
                border-radius: 2px;
            }
            QProgressBar::chunk {
                background: #00d7ff;
                border-radius: 2px;
            }
        """)

        project_layout.addWidget(project_name)
        project_layout.addWidget(progress_bar)
        project_widget.setLayout(project_layout)

        project_item.setSizeHint(project_widget.sizeHint())
        self.project_list.addItem(project_item)
        self.project_list.setItemWidget(project_item, project_widget)

        self.project_list.setMaximumHeight(120)
        self.add_widget(self.project_list)

        # New Project button
        self.new_project_btn = ModernButton("📁 New Project", button_type="primary")
        self.add_widget(self.new_project_btn)

        # Sessions section
        sessions_label = QLabel("Sessions (Current Project):")
        sessions_label.setFont(QFont("Segoe UI", 9))
        sessions_label.setStyleSheet("color: #cccccc; margin-top: 12px;")
        self.add_widget(sessions_label)

        # Session progress bar
        session_progress = QProgressBar()
        session_progress.setValue(45)
        session_progress.setStyleSheet(progress_bar.styleSheet())
        self.add_widget(session_progress)


class RAGPanel(ModernPanel):
    """Knowledge Base (RAG) panel"""

    def __init__(self):
        super().__init__("Knowledge Base (RAG)")
        self._init_ui()

    def _init_ui(self):
        # Scan Directory button
        self.scan_btn = ModernButton("🌐 Scan Directory (Global)", button_type="secondary")
        self.add_widget(self.scan_btn)

        # Add Files button
        self.add_files_btn = ModernButton("📄 Add Files (Project)", button_type="secondary")
        self.add_widget(self.add_files_btn)

        # RAG Status
        rag_status_layout = QHBoxLayout()
        rag_status_label = QLabel("RAG:")
        rag_status_label.setFont(QFont("Segoe UI", 9))
        rag_status_label.setStyleSheet("color: #cccccc;")

        rag_status_text = QLabel("Initializing embedder...")
        rag_status_text.setStyleSheet("color: #ffb900; font-size: 9px;")

        rag_status_layout.addWidget(rag_status_label)
        rag_status_layout.addWidget(rag_status_text, 1)

        self.add_widget(QWidget())
        self.content_layout.itemAt(-1).widget().setLayout(rag_status_layout)


class ChatActionsPanel(ModernPanel):
    """Chat Actions panel"""

    action_triggered = Signal(str)

    def __init__(self):
        super().__init__("Chat Actions")
        self._init_ui()

    def _init_ui(self):
        # New Session
        self.new_session_btn = ModernButton("💬 New Session", button_type="secondary")
        self.new_session_btn.clicked.connect(lambda: self.action_triggered.emit("new_session"))
        self.add_widget(self.new_session_btn)

        # View LLM Log
        self.view_log_btn = ModernButton("📊 View LLM Log", button_type="secondary")
        self.view_log_btn.clicked.connect(lambda: self.action_triggered.emit("view_log"))
        self.add_widget(self.view_log_btn)

        # View Generated Code
        self.view_code_btn = ModernButton("⚡ View Generated Code", button_type="secondary")
        self.view_code_btn.clicked.connect(lambda: self.action_triggered.emit("view_code"))
        self.add_widget(self.view_code_btn)

        # Force Code Gen
        self.force_gen_btn = ModernButton("🔨 Force Code Gen", button_type="accent")
        self.force_gen_btn.clicked.connect(lambda: self.action_triggered.emit("force_gen"))
        self.add_widget(self.force_gen_btn)

        # Check for Updates
        self.update_btn = ModernButton("🔄 Check for Updates", button_type="secondary")
        self.update_btn.clicked.connect(lambda: self.action_triggered.emit("check_updates"))
        self.add_widget(self.update_btn)


class ChatInterface(QWidget):
    """Professional chat interface"""

    message_sent = Signal(str)

    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Chat input area
        input_frame = QFrame()
        input_frame.setStyleSheet("""
            QFrame {
                background: #2d2d30;
                border: 1px solid #404040;
                border-radius: 8px;
            }
        """)

        input_layout = QHBoxLayout()
        input_layout.setContentsMargins(16, 12, 16, 12)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Ask AvA anything...")
        self.input_field.setStyleSheet("""
            QLineEdit {
                background: transparent;
                border: none;
                color: white;
                font-size: 14px;
                font-family: "Segoe UI";
            }
            QLineEdit::placeholder {
                color: #6d6d6d;
            }
        """)
        self.input_field.returnPressed.connect(self._send_message)

        self.send_btn = ModernButton("Send", button_type="primary")
        self.send_btn.clicked.connect(self._send_message)
        self.send_btn.setMinimumWidth(80)

        input_layout.addWidget(self.input_field, 1)
        input_layout.addWidget(self.send_btn)
        input_frame.setLayout(input_layout)

        # Status bar
        status_layout = QHBoxLayout()
        status_layout.setContentsMargins(16, 8, 16, 8)

        self.status_indicator = StatusIndicator("ready")
        self.status_text = QLabel("Ready (gemini-2.5-pro-preview)")
        self.status_text.setStyleSheet("color: #6d6d6d; font-size: 12px;")

        status_layout.addWidget(self.status_indicator)
        status_layout.addWidget(self.status_text)
        status_layout.addStretch()

        layout.addWidget(input_frame)

        status_widget = QWidget()
        status_widget.setLayout(status_layout)
        layout.addWidget(status_widget)

        self.setLayout(layout)

    def _send_message(self):
        message = self.input_field.text().strip()
        if message:
            self.message_sent.emit(message)
            self.input_field.clear()
            self.status_text.setText("Processing...")
            self.status_indicator.update_status("working")