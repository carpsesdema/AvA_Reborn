# gui/enhanced_sidebar.py - Updated with Modern Design System

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame
)

from gui.components import ModernButton, StatusIndicator, Colors, Typography
from gui.panels import StyledPanel, ProjectManagementPanel


class AvALeftSidebar(QWidget):
    """Modern AvA left sidebar with all panels"""

    # Signals
    new_project_requested = Signal()
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
        layout.addWidget(self.project_panel)

        # AI Model Configuration Panel
        self.model_config_panel = AIModelConfigPanel()
        self.model_config_panel.model_config_requested.connect(self.model_config_requested.emit)
        layout.addWidget(self.model_config_panel)

        # Knowledge Base Panel
        self.knowledge_panel = KnowledgeBasePanel()
        self.knowledge_panel.scan_directory_requested.connect(self.scan_directory_requested.emit)
        layout.addWidget(self.knowledge_panel)

        # Chat Actions Panel
        self.actions_panel = ChatActionsPanel()
        self.actions_panel.action_triggered.connect(self.action_triggered.emit)
        layout.addWidget(self.actions_panel)

        # By removing the stretch here, the panels will expand to fill the vertical space
        # layout.addStretch(1) # <-- REMOVED!

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
        """Update project display"""
        self.project_panel.update_project_display(project_name)

    def update_model_status_display(self, config_summary: dict):
        """Update model status display"""
        self.model_config_panel.update_model_status_display(config_summary)

    def update_rag_status_display(self, status_text: str):
        """Update RAG status display"""
        self.knowledge_panel.update_rag_status(status_text)


class AIModelConfigPanel(StyledPanel):
    """Modern AI Model Configuration panel"""

    model_config_requested = Signal()

    def __init__(self):
        super().__init__("AI Model Configuration")
        self._init_ui()

    def _init_ui(self):
        """Initialize the UI components"""
        # Configure Models button - make it prominent
        self.config_btn = ModernButton("⚙️ Configure Models", button_type="primary")
        self.config_btn.clicked.connect(self.model_config_requested.emit)
        self.add_widget(self.config_btn)

        # AI Specialists Status section
        specialists_header = QLabel("AI Specialists Status")
        specialists_header.setFont(Typography.heading_small())
        specialists_header.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; margin: 12px 0px 8px 0px; font-weight: 600;")
        self.add_widget(specialists_header)

        # Create status displays for each specialist
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
                label_widget.setStyleSheet(f"color: {Colors.ACCENT_GREEN}; font-weight: 500;")
                indicator_widget.update_status("success")
            else:
                label_widget.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
                indicator_widget.update_status("offline")


class KnowledgeBasePanel(StyledPanel):
    """Modern Knowledge Base (RAG) panel"""

    scan_directory_requested = Signal()

    def __init__(self):
        super().__init__("Knowledge Base (RAG)")
        self._init_ui()

    def _init_ui(self):
        """Initialize the UI components"""
        # Scan Directory button
        self.scan_btn = ModernButton("🌐 Scan Directory (Global)", button_type="secondary")
        self.scan_btn.clicked.connect(self.scan_directory_requested.emit)
        self.add_widget(self.scan_btn)

        # Add Files button
        self.add_files_btn = ModernButton("📄 Add Files (Project)", button_type="secondary")
        self.add_widget(self.add_files_btn)

        # RAG status display
        rag_status_layout = QHBoxLayout()
        rag_status_layout.setContentsMargins(0, 12, 0, 0)

        rag_label = QLabel("RAG:")
        rag_label.setFont(Typography.body())
        rag_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")

        self.rag_status_display_label = QLabel("Initializing embedder...")
        self.rag_status_display_label.setFont(Typography.body_small())
        self.rag_status_display_label.setStyleSheet(f"color: {Colors.ACCENT_ORANGE};")

        rag_status_layout.addWidget(rag_label)
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
        """Initialize the UI components"""
        # Organize buttons into logical groups
        button_groups = [
            # Session Management
            ("Session", [
                ("💬 New Session", "new_session"),
                ("📊 View LLM Log", "view_log"),
            ]),
            # Development Tools
            ("Tools", [
                ("📟 Open Terminal", "open_terminal"),
                ("📄 Open Code Viewer", "open_code_viewer"),
            ]),
            # Code Operations (Now Empty)
            ("Code", []),
            # System
            ("System", [
                ("🔄 Check for Updates", "check_updates"),
            ])
        ]

        # Filter out empty groups before creating UI
        active_button_groups = [group for group in button_groups if group[1]]

        for group_index, (group_name, buttons) in enumerate(active_button_groups):
            # Add group separator (except for first group)
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

            # Add group label
            if len(active_button_groups) > 1:  # Only show group labels if multiple groups
                group_label = QLabel(group_name)
                group_label.setFont(Typography.body_small())
                group_label.setStyleSheet(f"""
                    color: {Colors.TEXT_MUTED}; 
                    font-weight: 600; 
                    margin: 4px 0px 2px 0px;
                    text-transform: uppercase;
                    font-size: 10px;
                    letter-spacing: 0.5px;
                """)
                self.add_widget(group_label)

            # Add buttons in group
            for text, action in buttons:
                btn = ModernButton(text, button_type="secondary")
                btn.clicked.connect(lambda checked, a=action: self.action_triggered.emit(a))
                self.add_widget(btn)


# For backward compatibility, maintain the old structure
class AIModelConfigurationPanel(StyledPanel):
    """Legacy name for AIModelConfigPanel"""

    model_config_requested = Signal()
    temperature_changed = Signal(float)

    def __init__(self):
        super().__init__("AI Model Configuration")
        self._init_ui()

    def _init_ui(self):
        """Initialize with basic structure for compatibility"""
        # Configure Models button
        self.config_btn = ModernButton("⚙️ Configure Models", button_type="primary")
        self.config_btn.clicked.connect(self.model_config_requested.emit)
        self.add_widget(self.config_btn)

        # Status displays - simplified for compatibility
        self.planner_status = QLabel("🧠 Planner: Not configured")
        self.planner_status.setFont(Typography.body_small())
        self.planner_status.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
        self.add_widget(self.planner_status)

        self.coder_status = QLabel("⚙️ Coder: Not configured")
        self.coder_status.setFont(Typography.body_small())
        self.coder_status.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
        self.add_widget(self.coder_status)

        self.assembler_status = QLabel("📄 Assembler: Not configured")
        self.assembler_status.setFont(Typography.body_small())
        self.assembler_status.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
        self.add_widget(self.assembler_status)

        self.reviewer_status = QLabel("🧐 Reviewer: Not configured")
        self.reviewer_status.setFont(Typography.body_small())
        self.reviewer_status.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
        self.add_widget(self.reviewer_status)

        self.chat_status = QLabel("💬 Chat: Not configured")
        self.chat_status.setFont(Typography.body_small())
        self.chat_status.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
        self.add_widget(self.chat_status)

    def update_model_status_display(self, config_summary: dict):
        """Update model status display with modern styling"""

        def truncate_model(name, length=22):
            return name[:length - 3] + "..." if name and len(name) > length else name or "Not configured"

        status_widgets = {
            "planner": (self.planner_status, "🧠 Planner"),
            "coder": (self.coder_status, "⚙️ Coder"),
            "assembler": (self.assembler_status, "📄 Assembler"),
            "reviewer": (self.reviewer_status, "🧐 Reviewer"),
            "chat": (self.chat_status, "💬 Chat")
        }

        for role_str, (label_widget, prefix) in status_widgets.items():
            model_name = config_summary.get(role_str, "Not configured")
            display_name = truncate_model(model_name)
            label_widget.setText(f"{prefix}: {display_name}")

            if model_name and model_name != "Not configured":
                label_widget.setStyleSheet(f"color: {Colors.ACCENT_GREEN}; font-weight: 500;")
            else:
                label_widget.setStyleSheet(f"color: {Colors.TEXT_MUTED};")