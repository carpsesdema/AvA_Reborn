# gui/main_window.py - COMPLETE WORKING REPLACEMENT
# This version uses the AvALeftSidebar from gui.enhanced_sidebar

from PySide6.QtCore import Signal, Slot, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QFrame, QLabel, QTextEdit, QScrollArea, QComboBox, QSlider, QPushButton
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt

from gui.components import ModernButton, StatusIndicator
from gui.enhanced_sidebar import AvALeftSidebar  # Uses the enhanced sidebar


class ChatDisplay(QTextEdit):
    """Actual chat display"""

    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setStyleSheet("""
            QTextEdit {
                background: #1e1e1e;
                border: 1px solid #3e3e42;
                border-radius: 8px;
                color: #cccccc;
                padding: 12px;
                font-family: "Segoe UI";
                font-size: 14px;
            }
        """)
        self.append(self._format_message("AvA",
                                         "Hello! I'm AvA. I can help you build applications or just chat. What would you like to work on?",
                                         "assistant"))

    def add_user_message(self, message: str):
        self.append(self._format_message("You", message, "user"))

    def add_assistant_message(self, message: str):
        self.append(self._format_message("AvA", message, "assistant"))

    def _format_message(self, sender: str, message: str, role: str) -> str:
        if role == "user":
            color = "#00d7ff"
            bg = "#2d2d30"
        else:  # assistant
            color = "#3fb950"  # Green for AvA's responses
            bg = "#252526"

        return f"""
        <div style="margin: 8px 0; padding: 8px 12px; background: {bg}; border-radius: 8px; border-left: 3px solid {color};">
            <div style="font-weight: bold; color: {color}; margin-bottom: 4px;">{sender}:</div>
            <div style="color: #cccccc; line-height: 1.4;">{message}</div>
        </div>
        """


class ChatInterface(QWidget):
    message_sent = Signal(str)

    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        self.chat_display = ChatDisplay()
        layout.addWidget(self.chat_display, 1)

        input_layout = QHBoxLayout()
        input_layout.setSpacing(10)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Chat with AvA... (type 'build' or 'create' to start a project)")
        self.input_field.setMinimumHeight(45)
        self.input_field.setStyleSheet("""
            QLineEdit {
                background: #2d2d30;
                border: 2px solid #404040;
                border-radius: 8px;
                color: white;
                font-size: 14px;
                padding: 12px 16px;
            }
            QLineEdit:focus {
                border-color: #00d7ff;
            }
            QLineEdit::placeholder {
                color: #888;
            }
        """)
        self.input_field.returnPressed.connect(self._send_message)

        self.send_btn = ModernButton("Send", button_type="accent")
        self.send_btn.clicked.connect(self._send_message)
        self.send_btn.setMinimumHeight(45)
        self.send_btn.setMinimumWidth(80)

        input_layout.addWidget(self.input_field, 1)
        input_layout.addWidget(self.send_btn)

        # Status Bar Layout
        status_bar_layout = QHBoxLayout()
        status_bar_layout.setContentsMargins(0, 8, 0, 0)

        self.llm_status_indicator = StatusIndicator("offline")
        self.llm_status_text = QLabel("LLM: Initializing...")
        self.llm_status_text.setStyleSheet("color: #888; font-size: 11px; margin-left: 5px;")
        status_bar_layout.addWidget(self.llm_status_indicator)
        status_bar_layout.addWidget(self.llm_status_text)
        status_bar_layout.addStretch(1)

        self.rag_status_indicator = StatusIndicator("offline")
        self.rag_status_text_label = QLabel("RAG: Initializing...")
        self.rag_status_text_label.setStyleSheet("color: #888; font-size: 11px; margin-left: 5px;")
        status_bar_layout.addWidget(self.rag_status_indicator)
        status_bar_layout.addWidget(self.rag_status_text_label)

        layout.addLayout(input_layout)
        layout.addLayout(status_bar_layout)
        self.setLayout(layout)

    def _send_message(self):
        message = self.input_field.text().strip()
        if not message:
            return
        self.chat_display.add_user_message(message)
        self.input_field.clear()
        self.message_sent.emit(message)

    def update_llm_status(self, text: str, indicator_status: str = "ready"):
        self.llm_status_text.setText(text)
        self.llm_status_indicator.update_status(indicator_status)

    def update_rag_ui_status(self, text: str, color_or_key: str):
        self.rag_status_text_label.setText(text)

        text_color_hex = "#888888"
        indicator_key = "offline"

        if color_or_key.startswith("#"):
            text_color_hex = color_or_key
            hex_to_key_map = {
                "#4ade80": "success", "#3fb950": "success",
                "#ffb900": "working",
                "#ef4444": "error", "#f85149": "error",
                "#00d7ff": "ready",
                "#6a9955": "success",  # Another green from screenshot
            }
            indicator_key = hex_to_key_map.get(color_or_key.lower(), "offline")
        else:
            indicator_key = color_or_key
            key_to_hex_map = {
                "ready": "#00d7ff", "success": "#4ade80", "working": "#ffb900",
                "error": "#ef4444", "offline": "#888888", "grey": "#888888"
            }
            text_color_hex = key_to_hex_map.get(color_or_key, "#888888")

        self.rag_status_text_label.setStyleSheet(f"color: {text_color_hex}; font-size: 11px; margin-left: 5px;")
        self.rag_status_indicator.update_status(indicator_key)


class AvAMainWindow(QMainWindow):
    workflow_requested = Signal(str)
    new_project_requested = Signal()

    def __init__(self, ava_app=None, config=None):
        super().__init__()
        self.ava_app = ava_app

        self.setWindowTitle("AvA - AI Development Assistant")
        self.setGeometry(100, 100, 1400, 900)
        self._apply_theme()
        self._init_ui()
        self._connect_signals()

        if self.ava_app:
            # Connect signals from AvAApplication
            self.ava_app.rag_status_changed.connect(self.update_rag_status_display)
            self.ava_app.workflow_started.connect(self.on_workflow_started)  # Connect this
            self.ava_app.workflow_completed.connect(self.on_workflow_completed)  # Connect this
            self.ava_app.error_occurred.connect(self.on_app_error_occurred)  # Connect this
            self.ava_app.project_loaded.connect(self.update_project_display)  # Connect this

            # Defer initial UI status update slightly to ensure all components are ready
            QTimer.singleShot(150, self._update_initial_ui_status)
        else:
            self._update_initial_ui_status()  # Fallback if no ava_app

    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background: #1e1e1e;
                color: #cccccc;
            }
        """)

    def _init_ui(self):
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.sidebar = AvALeftSidebar()
        self.chat_interface = ChatInterface()

        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.chat_interface, 1)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def _connect_signals(self):
        self.chat_interface.message_sent.connect(self.workflow_requested)
        self.sidebar.action_triggered.connect(self._handle_sidebar_action)

        if hasattr(self.sidebar, 'project_panel') and hasattr(self.sidebar.project_panel, 'new_project_btn'):
            self.sidebar.project_panel.new_project_btn.clicked.connect(self.new_project_requested.emit)

        self.sidebar.temperature_changed.connect(self._on_temperature_changed)
        self.sidebar.model_changed.connect(self._on_model_changed)

    def _update_initial_ui_status(self):
        llm_model_text = "LLM: Unknown"
        llm_indicator_status = "offline"
        if self.ava_app and self.ava_app.current_config:
            chat_model_name = self.ava_app.current_config.get("chat_model", "LLM").split(':')[-1].strip()
            current_temp = self.ava_app.current_config.get("temperature", 0.0)
            llm_model_text = f"LLM: {chat_model_name} (Temp: {current_temp:.2f})"
            if self.ava_app.llm_client and self.ava_app.llm_client.get_available_models() and \
                    self.ava_app.llm_client.get_available_models() != ["No LLM services available"]:
                llm_indicator_status = "ready"

        self.chat_interface.update_llm_status(llm_model_text, llm_indicator_status)

        rag_text = "RAG: Unknown"
        rag_color_key_for_indicator = "offline"

        if self.ava_app:
            # Get status directly from ava_app which should reflect RAGManager's current state
            app_status = self.ava_app.get_status()
            rag_info = app_status.get("rag", {})
            rag_text = rag_info.get("status_text", "RAG: Unknown")

            if rag_info.get("ready"):
                rag_color_key_for_indicator = "success"
            elif not rag_info.get("available", True):  # If RAG is not available (e.g. import failed)
                rag_color_key_for_indicator = "offline"
            elif "Initializing" in rag_text or "loading" in rag_text.lower() or "Embedder" in rag_text:
                rag_color_key_for_indicator = "working"
            elif "Error" in rag_text or "Fail" in rag_text or "Missing" in rag_text:
                rag_color_key_for_indicator = "error"

        self.update_rag_status_display(rag_text, rag_color_key_for_indicator)

        project_name_to_display = "Default Project"
        if self.ava_app and hasattr(self.ava_app, 'current_project'):
            project_name_to_display = self.ava_app.current_project
        self.update_project_display(project_name_to_display)

    def _on_temperature_changed(self, temp: float):
        if self.ava_app:
            self.ava_app.update_configuration({"temperature": temp})
            chat_model_name = self.ava_app.current_config.get("chat_model", "LLM").split(':')[-1].strip()
            self.chat_interface.update_llm_status(f"LLM: {chat_model_name} (Temp: {temp:.2f})", "ready")

    def _on_model_changed(self, model_type: str, model_name: str):
        if self.ava_app:
            new_config = {}
            if model_type == "chat":
                new_config["chat_model"] = model_name
            elif model_type == "code":
                new_config["code_model"] = model_name
            self.ava_app.update_configuration(new_config)

            chat_model_display_name = self.ava_app.current_config.get("chat_model", model_name).split(':')[-1].strip()
            current_temp = self.ava_app.current_config.get("temperature", 0.0)
            self.chat_interface.update_llm_status(f"LLM: {chat_model_display_name} (Temp: {current_temp:.2f})", "ready")

    def _handle_sidebar_action(self, action: str):
        if not self.ava_app:
            print(f"AvAApp not available to handle action: {action}")
            return

        if action == "open_terminal" or action == "view_log":
            self.ava_app._open_terminal()
        elif action == "open_code_viewer" or action == "view_code":
            self.ava_app._open_code_viewer()
        elif action == "new_session":
            self.chat_interface.chat_display.clear()  # Clear current chat
            self.chat_interface.chat_display.add_assistant_message("New session started! How can I help you today?")
            if hasattr(self.ava_app, 'current_session'): self.ava_app.current_session = "New Session"
            self.update_project_display(
                self.ava_app.current_project if hasattr(self.ava_app, 'current_project') else "Default Project")
        elif action == "scan_directory":
            if self.ava_app.rag_manager:
                self.ava_app.rag_manager.scan_directory_dialog(parent_widget=self)
            else:
                self.chat_interface.add_assistant_message("RAG Manager not available for scanning.")
        elif action == "add_files":
            if self.ava_app.rag_manager:
                self.ava_app.rag_manager.add_files_dialog(parent_widget=self)
            else:
                self.chat_interface.add_assistant_message("RAG Manager not available for adding files.")
        elif action == "force_gen":
            self.chat_interface.add_assistant_message("Force code generation triggered (logic to be implemented).")
        elif action == "check_updates":
            self.chat_interface.add_assistant_message("Checking for updates (feature not yet implemented).")
        else:
            print(f"Unknown sidebar action: {action}")

    @Slot(str)
    def on_workflow_started(self, prompt: str):
        llm_name = "LLM"
        if self.ava_app and self.ava_app.current_config:
            llm_name = self.ava_app.current_config.get("chat_model", "LLM").split(':')[-1].strip()
        self.chat_interface.update_llm_status(f"{llm_name}: Working on: {prompt[:25]}...", "working")

    @Slot(dict)
    def on_workflow_completed(self, result: dict):
        success = result.get("success", False)
        llm_name = "LLM";
        temp = 0.0
        if self.ava_app and self.ava_app.current_config:
            llm_name = self.ava_app.current_config.get("chat_model", "LLM").split(':')[-1].strip()
            temp = self.ava_app.current_config.get("temperature", 0.0)

        if success:
            project_name = result.get("project_name", "your project")
            num_files = result.get("file_count", 0)
            message = f"✅ Workflow for '{project_name}' completed! Generated {num_files} files. View them in the Code Viewer."
            self.chat_interface.add_assistant_message(message)
            self.chat_interface.update_llm_status(f"LLM: {llm_name} (Temp: {temp:.2f})", "success")
        else:
            error_msg = result.get("error", "An unknown error occurred.")
            self.chat_interface.add_assistant_message(f"❌ Workflow failed: {error_msg}")
            self.chat_interface.update_llm_status(f"LLM: {llm_name} (Temp: {temp:.2f})", "error")

    @Slot(str, str)
    def on_app_error_occurred(self, component: str, error_message: str):
        self.chat_interface.add_assistant_message(f"⚠️ Error in {component}: {error_message}")
        llm_name = "LLM";
        temp = 0.0
        if self.ava_app and self.ava_app.current_config:
            llm_name = self.ava_app.current_config.get("chat_model", "LLM").split(':')[-1].strip()
            temp = self.ava_app.current_config.get("temperature", 0.0)
        self.chat_interface.update_llm_status(f"LLM: {llm_name} (Temp: {temp:.2f}) - Error", "error")

    @Slot(str, str)
    def update_rag_status_display(self, status_text: str, color_or_key: str):
        self.chat_interface.update_rag_ui_status(status_text, color_or_key)

        text_color_hex = "#888888"
        if color_or_key.startswith("#"):
            text_color_hex = color_or_key
        else:
            key_to_hex_map = {
                "ready": "#4ade80", "success": "#4ade80", "working": "#ffb900",
                "error": "#ef4444", "offline": "#888888", "grey": "#888888",
                # Adding keys that RAGManager might emit for its status_changed signal
                "error": "#ef4444", "warning": "#ffb900", "success": "#4ade80"
            }
            text_color_hex = key_to_hex_map.get(color_or_key, "#888888")

        if hasattr(self.sidebar, 'update_sidebar_rag_status'):
            self.sidebar.update_sidebar_rag_status(status_text, text_color_hex)

    @Slot(str)  # Connected to ava_app.project_loaded
    def update_project_display(self, project_name_or_path: str):
        project_name = project_name_or_path  # Could be full path from project_loaded
        if "/" in project_name or "\\" in project_name:  # Basic check if it's a path
            from pathlib import Path
            project_name = Path(project_name_or_path).name

        session_name = "Main Chat"
        chat_model_from_config = "N/A"

        if self.ava_app:
            session_name = self.ava_app.current_session if hasattr(self.ava_app, 'current_session') else "Main Chat"
            if self.ava_app.current_config:
                chat_model_from_config = self.ava_app.current_config.get("chat_model", "N/A").split(':')[-1].strip()

        self.setWindowTitle(f"AvA [{project_name}] - Session: {session_name} (LLM: {chat_model_from_config})")

        if hasattr(self.sidebar, 'project_panel'):
            # If project_panel's list needs dynamic update and selection:
            # self.sidebar.project_panel.update_and_select_project(project_name)
            pass