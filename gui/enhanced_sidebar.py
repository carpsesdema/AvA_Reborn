# gui/enhanced_sidebar.py - V4 with consolidated Architect Role

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QSizePolicy
)

from gui.components import ModernButton, StatusIndicator, Colors, Typography
from gui.panels import StyledPanel, ProjectManagementPanel


class AvALeftSidebar(QWidget):
    """Modern AvA left sidebar with all panels"""

    # Signals
    new_project_requested = Signal()
    load_project_requested = Signal()
    model_config_requested = Signal()
    scan_directory_requested = Signal()
    action_triggered = Signal(str)

    def __init__(self):
        super().__init__()
        self._init_ui()
        self._apply_style()

    def _init_ui(self):
        """Initialize the sidebar UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Project Management Panel
        self.project_panel = ProjectManagementPanel()
        self.project_panel.new_project_requested.connect(self.new_project_requested.emit)
        self.project_panel.load_project_requested.connect(self.load_project_requested.emit)
        layout.addWidget(self.project_panel)

        # AI Model Configuration Panel
        self.model_config_panel = AIModelConfigPanel()
        self.model_config_panel.model_config_requested.connect(self.model_config_requested.emit)
        layout.addWidget(self.model_config_panel)

        # Knowledge Base Panel
        self.knowledge_panel = KnowledgeBasePanel()
        self.knowledge_panel.scan_directory_requested.connect(self.scan_directory_requested.emit)
        self.knowledge_panel.add_files_btn.clicked.connect(lambda: self.action_triggered.emit("add_project_files")) # Connect the add files button
        layout.addWidget(self.knowledge_panel)

        # Chat Actions Panel
        self.actions_panel = ChatActionsPanel()
        self.actions_panel.action_triggered.connect(self.action_triggered.emit)
        layout.addWidget(self.actions_panel)

        layout.addStretch(1)

        self.setLayout(layout)

    def _apply_style(self):
        """Apply modern sidebar styling"""
        self.setFixedWidth(340)
        self.setStyleSheet(f"""
            AvALeftSidebar {{
                background: {Colors.SECONDARY_BG};
                border-right: 1px solid {Colors.BORDER_DEFAULT};
            }}
        """)

    def update_project_display(self, project_name: str):
        self.project_panel.update_project_display(project_name)

    def update_model_status_display(self, config_summary: dict):
        self.model_config_panel.update_model_status_display(config_summary)


class AIModelConfigPanel(StyledPanel):
    """Modern AI Model Configuration panel showing V4 roles."""

    model_config_requested = Signal()

    def __init__(self):
        super().__init__("AI Model Configuration")
        self._init_ui()

    def _init_ui(self):
        """Initialize the UI components with the new role structure."""
        self.config_btn = ModernButton("⚙️ Configure Models", button_type="primary")
        self.config_btn.clicked.connect(self.model_config_requested.emit)
        self.add_widget(self.config_btn)

        specialists_header = QLabel("AI Specialists Status")
        specialists_header.setFont(Typography.heading_small())
        specialists_header.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; margin: 12px 0px 8px 0px; font-weight: 600;")
        self.add_widget(specialists_header)

        # NEW V4 ROLES
        specialists = [
            ("🏛️ Architect", "architect_status"),
            ("⚙️ Coder", "coder_status"),
            ("🧩 Assembler", "assembler_status"),
            ("🧐 Reviewer", "reviewer_status"),
            ("💬 Chat", "chat_status")
        ]

        for label_text, attr_name in specialists:
            status_layout = QHBoxLayout()
            status_layout.setContentsMargins(0, 4, 0, 4)
            indicator = StatusIndicator("offline")
            status_label = QLabel(f"{label_text}: Not configured")
            status_label.setFont(Typography.body_small())
            status_label.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
            status_layout.addWidget(indicator)
            status_layout.addWidget(status_label, 1)
            self.add_layout(status_layout)
            setattr(self, attr_name, status_label)
            setattr(self, f"{attr_name}_indicator", indicator)

    def update_model_status_display(self, config_summary: dict):
        """Update the model status display for the V4 roles."""

        def truncate_model(name, length=22):
            return name[:length - 3] + "..." if name and len(name) > length else name or "Not configured"

        # NEW V4 ROLE MAPPING
        status_map = {
            "architect": (self.architect_status, self.architect_status_indicator, "🏛️ Architect"),
            "coder": (self.coder_status, self.coder_status_indicator, "⚙️ Coder"),
            "assembler": (self.assembler_status, self.assembler_status_indicator, "🧩 Assembler"),
            "reviewer": (self.reviewer_status, self.reviewer_status_indicator, "🧐 Reviewer"),
            "chat": (self.chat_status, self.chat_status_indicator, "💬 Chat")
        }

        for role_str, (label_widget, indicator_widget, prefix) in status_map.items():
            model_name = config_summary.get(role_str, "Not configured")
            display_name = truncate_model(model_name)
            label_widget.setText(f"{prefix}: {display_name}")
            if model_name and model_name != "Not configured":
                label_widget.setStyleSheet(f"color: {Colors.ACCENT_GREEN}; font-weight: 500;")
                indicator_widget.update_status("success")
            else:
                label_widget.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
                indicator_widget.update_status("offline")


class KnowledgeBasePanel(StyledPanel):
    # ... (This class remains unchanged)
    scan_directory_requested = Signal()

    def __init__(self):
        super().__init__("Knowledge Base (RAG)")
        self._init_ui()

    def _init_ui(self):
        self.scan_btn = ModernButton("🌐 Scan Directory (Global)", button_type="secondary")
        self.scan_btn.clicked.connect(self.scan_directory_requested.emit)
        self.add_widget(self.scan_btn)

        self.add_files_btn = ModernButton("📄 Add Files (Project)", button_type="secondary")
        self.add_widget(self.add_files_btn)

        # RAG status display removed.


class ChatActionsPanel(StyledPanel):
    action_triggered = Signal(str)

    def __init__(self):
        super().__init__("Chat Actions")
        self._init_ui()

    def _init_ui(self):
        button_groups = [
            ("Session", [
                ("💾 Save Session", "save_session"),
                ("📂 Load Session", "load_session"),
                ("💬 New Session", "new_session"),
            ]),
            ("Tools", [
                ("📊 View LLM Log", "view_log"),
                ("📈 Workflow Monitor", "open_workflow_monitor"),
                ("📄 Open Code Viewer", "open_code_viewer"),
            ]),
            ("System", [
                ("🔄 Check for Updates", "check_updates"),
            ])
        ]
        for group_index, (group_name, buttons) in enumerate(button_groups):
            if group_index > 0:
                separator = QFrame()
                separator.setFixedHeight(1)
                separator.setStyleSheet(f"background: {Colors.BORDER_DEFAULT}; border: none; margin: 8px 0px;")
                self.add_widget(separator)

            group_label = QLabel(group_name)
            group_label.setFont(Typography.body_small())
            group_label.setStyleSheet(
                f"color: {Colors.TEXT_MUTED}; font-weight: 600; margin: 4px 0px 2px 0px; text-transform: uppercase; font-size: 10px; letter-spacing: 0.5px;")
            self.add_widget(group_label)

            for text, action in buttons:
                btn = ModernButton(text, button_type="secondary")
                btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
                btn.clicked.connect(lambda checked, a=action: self.action_triggered.emit(a))
                self.add_widget(btn)