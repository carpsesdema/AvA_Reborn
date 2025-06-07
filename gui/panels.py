# gui/panels.py - Modernized panels with sleek design

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSlider,
    QListWidget, QListWidgetItem, QProgressBar, QFrame, QPushButton,
    QScrollArea
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFont

from gui.components import ModernButton, StatusIndicator, Colors, Typography


class StyledPanel(QFrame):
    """Modern panel with sleek styling and proper typography"""

    def __init__(self, title="", collapsible=False, initially_collapsed=False):
        super().__init__()
        self.is_collapsed = initially_collapsed
        self.collapsible = collapsible

        self.setFrameStyle(QFrame.Shape.NoFrame)
        self.setStyleSheet(f"""
            StyledPanel {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {Colors.SECONDARY_BG}, stop:1 {Colors.ELEVATED_BG});
                border: 1px solid {Colors.BORDER_DEFAULT};
                border-radius: 12px;
                margin: 4px;
            }}
            StyledPanel:hover {{
                border-color: {Colors.BORDER_ACCENT};
            }}
        """)

        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(16, 12, 16, 16)
        self.main_layout.setSpacing(12)

        if title:
            self._create_header(title)

        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(10)
        self.content_layout.addStretch(1) # This will push the panel's content to the top

        self.main_layout.addWidget(self.content_widget)
        self.setLayout(self.main_layout)

        if self.collapsible and self.is_collapsed:
            self.content_widget.hide()

    def _create_header(self, title):
        """Create modern header with title"""
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 8)

        title_label = QLabel(title)
        title_label.setFont(Typography.heading_small())
        title_label.setStyleSheet(f"""
            QLabel {{
                color: {Colors.TEXT_PRIMARY};
                background: transparent;
                border: none;
                font-weight: 600;
            }}
        """)

        header_layout.addWidget(title_label)

        if self.collapsible:
            # Add collapse button if needed
            pass

        self.main_layout.addLayout(header_layout)

    def add_widget(self, widget):
        """Add widget to content area, before the stretch"""
        # Insert widget before the last item, which is the stretch
        self.content_layout.insertWidget(self.content_layout.count() - 1, widget)

    def add_layout(self, layout):
        """Add layout to content area, before the stretch"""
        # Insert layout before the last item, which is the stretch
        self.content_layout.insertLayout(self.content_layout.count() - 1, layout)


class ModernPanel(StyledPanel):
    """Alias for backward compatibility"""
    pass


class LLMConfigPanel(StyledPanel):
    """Modern LLM Configuration panel"""

    model_changed = Signal(str, str)
    temperature_changed = Signal(float)

    def __init__(self):
        super().__init__("AI Model Configuration")
        self._init_ui()

    def _init_ui(self):
        # Chat LLM Section
        chat_layout = QHBoxLayout()
        chat_layout.setContentsMargins(0, 4, 0, 4)

        chat_label = QLabel("Chat LLM:")
        chat_label.setFont(Typography.body())
        chat_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; min-width: 80px;")

        self.chat_combo = QComboBox()
        self.chat_combo.addItems([
            "Gemini: gemini-2.5-pro-preview",
            "OpenAI: gpt-4o",
            "Anthropic: claude-3.5-sonnet",
            "DeepSeek: deepseek-chat"
        ])
        self.chat_combo.setFont(Typography.body_small())
        self.chat_combo.setStyleSheet(f"""
            QComboBox {{
                background: {Colors.ELEVATED_BG};
                border: 1px solid {Colors.BORDER_DEFAULT};
                border-radius: 6px;
                padding: 6px 12px;
                color: {Colors.TEXT_PRIMARY};
                min-width: 180px;
            }}
            QComboBox:hover {{
                border-color: {Colors.BORDER_ACCENT};
            }}
            QComboBox:focus {{
                border-color: {Colors.ACCENT_BLUE};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 4px solid {Colors.TEXT_SECONDARY};
                margin-right: 6px;
            }}
            QComboBox QAbstractItemView {{
                background: {Colors.ELEVATED_BG};
                border: 1px solid {Colors.BORDER_ACCENT};
                border-radius: 6px;
                selection-background-color: {Colors.HOVER_BG};
                selection-color: {Colors.TEXT_PRIMARY};
                color: {Colors.TEXT_PRIMARY};
                padding: 4px;
            }}
            QComboBox QAbstractItemView::item {{
                padding: 8px 12px;
                border-radius: 4px;
                margin: 2px;
            }}
        """)

        chat_status = StatusIndicator("ready")

        chat_layout.addWidget(chat_label)
        chat_layout.addWidget(self.chat_combo, 1)
        chat_layout.addWidget(chat_status)
        self.add_layout(chat_layout)

        # Specialists Section
        specialists_label = QLabel("AI Specialists:")
        specialists_label.setFont(Typography.body())
        specialists_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; margin-top: 8px;")
        self.add_widget(specialists_label)

        self.specialists_combo = QComboBox()
        self.specialists_combo.addItems([
            "Auto-select best models",
            "Use Chat LLM for all",
            "Custom configuration"
        ])
        self.specialists_combo.setFont(Typography.body_small())
        self.specialists_combo.setStyleSheet(self.chat_combo.styleSheet())
        self.add_widget(self.specialists_combo)

        # Temperature Control
        temp_header = QLabel("Temperature Control")
        temp_header.setFont(Typography.heading_small())
        temp_header.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; margin-top: 12px; margin-bottom: 4px;")
        self.add_widget(temp_header)

        temp_layout = QHBoxLayout()
        temp_layout.setContentsMargins(0, 4, 0, 4)

        temp_label = QLabel("Temp:")
        temp_label.setFont(Typography.body_small())
        temp_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; min-width: 50px;")

        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setMinimum(0)
        self.temp_slider.setMaximum(100)
        self.temp_slider.setValue(70)
        self.temp_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                background: {Colors.BORDER_DEFAULT};
                height: 6px;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {Colors.ACCENT_BLUE};
                border: 2px solid {Colors.PRIMARY_BG};
                width: 18px;
                height: 18px;
                border-radius: 9px;
                margin: -6px 0;
            }}
            QSlider::handle:horizontal:hover {{
                background: #6cb6ff;
                border-color: {Colors.ACCENT_BLUE};
            }}
            QSlider::sub-page:horizontal {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {Colors.ACCENT_BLUE}, stop:1 #6cb6ff);
                border-radius: 3px;
            }}
        """)
        self.temp_slider.valueChanged.connect(self._on_temperature_changed)

        self.temp_value = QLabel("0.70")
        self.temp_value.setFont(Typography.body_small())
        self.temp_value.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; min-width: 40px; font-weight: 500;")

        temp_layout.addWidget(temp_label)
        temp_layout.addWidget(self.temp_slider, 1)
        temp_layout.addWidget(self.temp_value)
        self.add_layout(temp_layout)

        # Configure Persona Button
        self.persona_btn = ModernButton("🎭 Configure Persona", button_type="secondary")
        self.add_widget(self.persona_btn)

    def _on_temperature_changed(self, value):
        """Handle temperature slider changes"""
        temp_val = value / 100.0
        self.temp_value.setText(f"{temp_val:.2f}")
        self.temperature_changed.emit(temp_val)


class KnowledgeBasePanel(StyledPanel):
    """Modern Knowledge Base (RAG) panel"""

    scan_directory_requested = Signal()

    def __init__(self):
        super().__init__("Knowledge Base (RAG)")
        self._init_ui()

    def _init_ui(self):
        # Scan Directory button
        self.scan_btn = ModernButton("🌐 Scan Directory (Global)", button_type="secondary")
        self.scan_btn.clicked.connect(self.scan_directory_requested.emit)
        self.add_widget(self.scan_btn)

        # Add Files button
        self.add_files_btn = ModernButton("📄 Add Files (Project)", button_type="secondary")
        self.add_widget(self.add_files_btn)

        # RAG Status
        rag_status_layout = QHBoxLayout()
        rag_status_layout.setContentsMargins(0, 8, 0, 0)

        rag_status_label = QLabel("RAG:")
        rag_status_label.setFont(Typography.body())
        rag_status_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")

        self.rag_status_display_label = QLabel("Initializing embedder...")
        self.rag_status_display_label.setFont(Typography.body_small())
        self.rag_status_display_label.setStyleSheet(f"color: {Colors.ACCENT_ORANGE};")

        rag_status_layout.addWidget(rag_status_label)
        rag_status_layout.addWidget(self.rag_status_display_label, 1)
        self.add_layout(rag_status_layout)

    def update_rag_status(self, status_text: str):
        """Update RAG status display"""
        self.rag_status_display_label.setText(status_text)

        # Update color based on status
        if "ready" in status_text.lower() or "complete" in status_text.lower():
            color = Colors.ACCENT_GREEN
        elif "error" in status_text.lower() or "failed" in status_text.lower():
            color = Colors.ACCENT_RED
        elif "initializing" in status_text.lower() or "loading" in status_text.lower():
            color = Colors.ACCENT_ORANGE
        else:
            color = Colors.TEXT_SECONDARY

        self.rag_status_display_label.setStyleSheet(f"color: {color};")


class ChatActionsPanel(StyledPanel):
    """Modern Chat Actions panel"""

    action_triggered = Signal(str)

    def __init__(self):
        super().__init__("Chat Actions")
        self._init_ui()

    def _init_ui(self):
        # Create buttons with modern styling and better organization
        button_groups = [
            # Session Management
            [
                ("💬 New Session", "new_session"),
                ("📊 View LLM Log", "view_log"),
            ],
            # Development Tools
            [
                ("📟 Open Terminal", "open_terminal"),
                ("📄 Open Code Viewer", "open_code_viewer"),
                ("🎛️ Toggle AI Monitor", "toggle_feedback_panel"),
            ],
            # Code Generation
            [
                ("⚡ View Generated Code", "view_code"),
                ("🔨 Force Code Gen", "force_gen"),
            ],
            # System
            [
                ("🔄 Check for Updates", "check_updates"),
            ]
        ]

        for group_index, group in enumerate(button_groups):
            # Add visual separator between groups
            if group_index > 0:
                separator = QFrame()
                separator.setFixedHeight(1)
                separator.setStyleSheet(f"""
                    QFrame {{
                        background: {Colors.BORDER_DEFAULT};
                        border: none;
                        margin: 8px 0px;
                    }}
                """)
                self.add_widget(separator)

            for text, action in group:
                btn = ModernButton(text, button_type="secondary")
                btn.clicked.connect(lambda checked, a=action: self.action_triggered.emit(a))
                self.add_widget(btn)


# Enhanced Sidebar Panel for Model Status
class ModelStatusPanel(StyledPanel):
    """Panel showing AI specialist model assignments"""

    def __init__(self):
        super().__init__("AI Specialists Status")
        self._init_ui()

    def _init_ui(self):
        """Initialize model status displays"""
        specialists = [
            ("🧠 Planner", "planner_status"),
            ("⚙️ Coder", "coder_status"),
            ("📄 Assembler", "assembler_status"),
            ("🧐 Reviewer", "reviewer_status"),
            ("💬 Chat", "chat_status")
        ]

        for label_text, attr_name in specialists:
            status_layout = QHBoxLayout()
            status_layout.setContentsMargins(0, 4, 0, 4)

            # Status indicator
            indicator = StatusIndicator("offline")

            # Status label
            status_label = QLabel(f"{label_text}: Not configured")
            status_label.setFont(Typography.body_small())
            status_label.setStyleSheet(f"color: {Colors.TEXT_MUTED};")

            status_layout.addWidget(indicator)
            status_layout.addWidget(status_label, 1)

            self.add_layout(status_layout)

            # Store references for updating
            setattr(self, attr_name, status_label)
            setattr(self, f"{attr_name}_indicator", indicator)

    def update_model_status_display(self, config_summary: dict):
        """Update the model status display"""

        def truncate_model(name, length=22):
            return name[:length - 3] + "..." if name and len(name) > length else name or "Not configured"

        status_map = {
            "planner": (self.planner_status, self.planner_status_indicator, "🧠 Planner"),
            "coder": (self.coder_status, self.coder_status_indicator, "⚙️ Coder"),
            "assembler": (self.assembler_status, self.assembler_status_indicator, "📄 Assembler"),
            "reviewer": (self.reviewer_status, self.reviewer_status_indicator, "🧐 Reviewer"),
            "chat": (self.chat_status, self.chat_status_indicator, "💬 Chat")
        }

        for role_str, (label_widget, indicator_widget, prefix) in status_map.items():
            model_name = config_summary.get(role_str, "Not configured")
            display_name = truncate_model(model_name)
            label_widget.setText(f"{prefix}: {display_name}")

            if model_name and model_name != "Not configured":
                label_widget.setStyleSheet(f"color: {Colors.ACCENT_GREEN};")
                indicator_widget.update_status("success")
            else:
                label_widget.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
                indicator_widget.update_status("offline")


# Project Management Panel
class ProjectManagementPanel(StyledPanel):
    """Modern project management panel"""

    new_project_requested = Signal()

    def __init__(self):
        super().__init__("Project Management")
        self._init_ui()

    def _init_ui(self):
        # New Project button - make it prominent
        self.new_project_btn = ModernButton("📁 New Project", button_type="primary")
        self.new_project_btn.clicked.connect(self.new_project_requested.emit)
        self.add_widget(self.new_project_btn)

        # Current project display
        project_layout = QHBoxLayout()
        project_layout.setContentsMargins(0, 8, 0, 0)

        project_label = QLabel("Project:")
        project_label.setFont(Typography.body())
        project_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")

        self.project_display = QLabel("Default Project")
        self.project_display.setFont(Typography.body())
        self.project_display.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-weight: 500;")

        project_layout.addWidget(project_label)
        project_layout.addWidget(self.project_display, 1)

        self.add_layout(project_layout)

    def update_project_display(self, project_name: str):
        """Update the project display"""
        self.project_display.setText(project_name)