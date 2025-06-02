# gui/main_window.py - Enhanced with Model Configuration Dialog

import asyncio
import inspect

from PySide6.QtCore import Signal, Slot, QTimer
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QLabel, QTextEdit
)

from gui.components import ModernButton, StatusIndicator
from gui.enhanced_sidebar import AvALeftSidebar
from gui.model_config_dialog import ModelConfigurationDialog  # NEW IMPORT

# Import LLMRole for chat functionality
try:
    from core.llm_client import LLMRole
except ImportError:
    # Fallback if enhanced client is not available
    class LLMRole:
        CHAT = "chat"


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
                                         "Hello! I'm AvA with enhanced AI specialists. I can help you build applications using my Planner, Coder, and Assembler AIs, or just chat. What would you like to work on?",
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

        # Enhanced Status Bar Layout with AI Specialist Info
        status_bar_layout = QHBoxLayout()
        status_bar_layout.setContentsMargins(0, 8, 0, 0)

        # LLM Status
        self.llm_status_indicator = StatusIndicator("offline")
        self.llm_status_text = QLabel("LLM: Initializing...")
        self.llm_status_text.setStyleSheet("color: #888; font-size: 11px; margin-left: 5px;")
        status_bar_layout.addWidget(self.llm_status_indicator)
        status_bar_layout.addWidget(self.llm_status_text)

        # AI Specialists Status (enhanced)
        self.specialists_status_indicator = StatusIndicator("offline")
        self.specialists_status_text = QLabel("AI Specialists: Initializing...")
        self.specialists_status_text.setStyleSheet("color: #888; font-size: 11px; margin-left: 5px;")
        status_bar_layout.addWidget(self.specialists_status_indicator)
        status_bar_layout.addWidget(self.specialists_status_text)

        status_bar_layout.addStretch(1)

        # RAG Status
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

    def update_specialists_status(self, text: str, indicator_status: str = "ready"):
        """Update AI specialists status"""
        self.specialists_status_text.setText(text)
        self.specialists_status_indicator.update_status(indicator_status)

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
                "#6a9955": "success",
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

        self.setWindowTitle("AvA - Enhanced AI Development Assistant")
        self.setGeometry(100, 100, 1400, 900)
        self._apply_theme()
        self._init_ui()
        self._connect_signals()

        if self.ava_app:
            # Connect signals from AvAApplication
            self.ava_app.rag_status_changed.connect(self.update_rag_status_display)
            self.ava_app.workflow_started.connect(self.on_workflow_started)
            self.ava_app.workflow_completed.connect(self.on_workflow_completed)
            self.ava_app.error_occurred.connect(self.on_app_error_occurred)
            self.ava_app.project_loaded.connect(self.update_project_display)

            # Defer initial UI status update
            QTimer.singleShot(150, self._update_initial_ui_status)
        else:
            self._update_initial_ui_status()

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
        self.chat_interface.message_sent.connect(self.handle_user_message)

        # Connect signals from the sidebar directly
        self.sidebar.new_project_requested.connect(self.new_project_requested.emit)
        self.sidebar.scan_directory_requested.connect(self._handle_rag_scan_directory)
        self.sidebar.action_triggered.connect(self._handle_sidebar_action)

        # NEW: Connect model configuration signal
        self.sidebar.model_config_requested.connect(self._open_model_config_dialog)

    def _open_model_config_dialog(self):
        """NEW: Open the model configuration dialog"""
        if not self.ava_app or not self.ava_app.llm_client:
            self.chat_interface.chat_display.add_assistant_message(
                "⚠️ LLM client is not available. Cannot configure models."
            )
            return

        # Create and show the model configuration dialog
        dialog = ModelConfigurationDialog(
            llm_client=self.ava_app.llm_client,
            parent=self
        )

        # Connect the configuration applied signal
        dialog.configuration_applied.connect(self._on_model_configuration_applied)

        # Show the dialog
        dialog.exec()

    def _on_model_configuration_applied(self, new_config: dict):
        """NEW: Handle when model configuration is applied"""
        # Update the sidebar display with new configuration
        if hasattr(self.ava_app.llm_client, 'get_role_assignments'):
            assignments = self.ava_app.llm_client.get_role_assignments()

            # Create display summary
            config_summary = {}
            for role, model_key in assignments.items():
                if model_key and hasattr(self.ava_app.llm_client, 'models'):
                    model_config = self.ava_app.llm_client.models.get(model_key)
                    if model_config:
                        display_name = f"{model_config.provider}/{model_config.model}"

                        if role.value == "planner":
                            config_summary['planner'] = display_name
                        elif role.value == "coder":
                            config_summary['coder'] = display_name
                        elif role.value == "assembler":
                            config_summary['assembler'] = display_name

            # Update sidebar display
            self.sidebar.update_model_status_display(config_summary)

        # Update specialist status in chat interface
        self._update_specialists_status()

        # Add confirmation message to chat
        self.chat_interface.chat_display.add_assistant_message(
            "✅ Model configuration updated! Your AI specialists are now ready with their optimized models."
        )

    def handle_user_message(self, message: str):
        """Handle user messages - decide between casual chat vs workflow"""
        if self._is_build_request(message):
            # This is a build request - send to workflow
            self.workflow_requested.emit(message)
        else:
            # This is casual chat - handle directly
            self._handle_casual_chat(message)

    def _is_build_request(self, prompt: str) -> bool:
        """Determine if this is a build request"""
        prompt_lower = prompt.lower().strip()

        # IGNORE casual chat
        casual_phrases = [
            'hi', 'hello', 'hey', 'thanks', 'thank you', 'ok', 'okay',
            'yes', 'no', 'sure', 'cool', 'nice', 'good', 'great'
        ]

        if prompt_lower in casual_phrases:
            return False

        # IGNORE questions without build intent
        question_words = ['what', 'how', 'why', 'when', 'where', 'who']
        if any(prompt_lower.startswith(word) for word in question_words):
            build_question_patterns = ['how to build', 'how to create', 'how to make', 'what should i build']
            if not any(pattern in prompt_lower for pattern in build_question_patterns):
                return False

        # REQUIRE explicit build keywords
        build_keywords = [
            'build', 'create', 'make', 'generate', 'develop', 'code',
            'implement', 'write', 'design', 'construct', 'program',
            'application', 'app', 'website', 'tool', 'script', 'project'
        ]

        has_build_keyword = any(keyword in prompt_lower for keyword in build_keywords)
        is_substantial = len(prompt.split()) >= 3

        return has_build_keyword and is_substantial

    def _handle_casual_chat(self, message: str):
        """Handle casual chat directly with LLM using CHAT role"""
        if not self.ava_app or not self.ava_app.llm_client:
            self.chat_interface.chat_display.add_assistant_message(
                "Sorry, LLM client is not available right now."
            )
            return

        # Update status to show we're chatting
        self.chat_interface.update_llm_status("LLM: Chatting...", "working")

        # Create casual chat prompt
        chat_prompt = f"""
You are AvA, a friendly AI development assistant with specialized AI agents. The user said: "{message}"

Respond naturally and helpfully. If they're just greeting you or making casual conversation, 
respond warmly. If they ask about your capabilities, mention:
- You have specialized AI agents: Planner AI, Coder AI, and Assembler AI
- You can build applications using a micro-task workflow
- You have access to a knowledge base for code examples and best practices
- You can work with multiple programming languages

Keep responses conversational and under 2-3 sentences unless they ask for detailed information.
"""

        # Handle chat asynchronously to avoid blocking UI
        asyncio.create_task(self._async_casual_chat(chat_prompt))

    async def _async_casual_chat(self, prompt: str):
        """Async casual chat handler with role-based LLM"""
        try:
            # Use CHAT role for casual conversation
            response_chunks = []

            if hasattr(self.ava_app.llm_client, 'stream_chat') and len(
                    inspect.signature(self.ava_app.llm_client.stream_chat).parameters) > 1:
                # Enhanced LLM client with role support
                async for chunk in self.ava_app.llm_client.stream_chat(prompt, LLMRole.CHAT):
                    response_chunks.append(chunk)
            else:
                # Fallback to old LLM client
                async for chunk in self.ava_app.llm_client.stream_chat(prompt):
                    response_chunks.append(chunk)

            response = ''.join(response_chunks).strip()

            # Add to chat display
            self.chat_interface.chat_display.add_assistant_message(response)

            # Reset status
            self._update_chat_llm_status()

        except Exception as e:
            self.chat_interface.chat_display.add_assistant_message(
                f"Sorry, I encountered an error: {e}"
            )
            self.chat_interface.update_llm_status("LLM: Error", "error")

    def _update_initial_ui_status(self):
        """Update initial UI status with enhanced information"""
        # Update main LLM status
        self._update_chat_llm_status()

        # Update AI specialists status
        self._update_specialists_status()

        # Update model configuration display
        self._update_model_config_display()

        # Update RAG status
        rag_text = "RAG: Unknown"
        rag_color_key_for_indicator = "offline"

        if self.ava_app:
            app_status = self.ava_app.get_status()
            rag_info = app_status.get("rag", {})
            rag_text = rag_info.get("status_text", "RAG: Unknown")

            if rag_info.get("ready"):
                rag_color_key_for_indicator = "success"
            elif not rag_info.get("available", True):
                rag_color_key_for_indicator = "offline"
            elif "Initializing" in rag_text or "loading" in rag_text.lower() or "Embedder" in rag_text:
                rag_color_key_for_indicator = "working"
            elif "Error" in rag_text or "Fail" in rag_text or "Missing" in rag_text:
                rag_color_key_for_indicator = "error"

        self.update_rag_status_display(rag_text, rag_color_key_for_indicator)

        # Update project display
        project_name_to_display = "Default Project"
        if self.ava_app and hasattr(self.ava_app, 'current_project'):
            project_name_to_display = self.ava_app.current_project
        self.update_project_display(project_name_to_display)

    def _update_chat_llm_status(self):
        """Update chat LLM status display"""
        llm_model_text = "LLM: Unknown"
        llm_indicator_status = "offline"

        if self.ava_app and self.ava_app.llm_client:
            if hasattr(self.ava_app.llm_client, 'get_role_assignments'):
                assignments = self.ava_app.llm_client.get_role_assignments()
                chat_model_key = assignments.get('chat', 'Unknown')

                if chat_model_key and hasattr(self.ava_app.llm_client, 'models'):
                    model_config = self.ava_app.llm_client.models.get(chat_model_key)
                    if model_config:
                        chat_model_name = model_config.model
                        current_temp = model_config.temperature
                        llm_model_text = f"Chat: {chat_model_name} (T: {current_temp:.2f})"
                        llm_indicator_status = "ready"

        self.chat_interface.update_llm_status(llm_model_text, llm_indicator_status)

    def _update_specialists_status(self):
        """Update AI specialists status display"""
        specialists_text = "AI Specialists: Unknown"
        specialists_status = "offline"

        if self.ava_app and self.ava_app.llm_client:
            if hasattr(self.ava_app.llm_client, 'get_role_assignments'):
                # Enhanced LLM client with role assignments
                assignments = self.ava_app.llm_client.get_role_assignments()
                planner_model = assignments.get('planner', 'N/A').split('_')[-1] if assignments.get(
                    'planner') else 'N/A'
                coder_model = assignments.get('coder', 'N/A').split('_')[-1] if assignments.get('coder') else 'N/A'

                specialists_text = f"Specialists: P:{planner_model[:8]} C:{coder_model[:8]}"
                specialists_status = "ready"
            else:
                # Legacy LLM client
                specialists_text = "AI Specialists: Legacy Mode"
                specialists_status = "working"

        self.chat_interface.update_specialists_status(specialists_text, specialists_status)

    def _update_model_config_display(self):
        """NEW: Update model configuration display in sidebar"""
        if not (self.ava_app and self.ava_app.llm_client):
            return

        if hasattr(self.ava_app.llm_client, 'get_role_assignments'):
            assignments = self.ava_app.llm_client.get_role_assignments()

            config_summary = {}
            for role, model_key in assignments.items():
                if model_key and hasattr(self.ava_app.llm_client, 'models'):
                    model_config = self.ava_app.llm_client.models.get(model_key)
                    if model_config:
                        display_name = f"{model_config.provider}/{model_config.model}"

                        if role.value == "planner":
                            config_summary['planner'] = display_name
                        elif role.value == "coder":
                            config_summary['coder'] = display_name
                        elif role.value == "assembler":
                            config_summary['assembler'] = display_name

            self.sidebar.update_model_status_display(config_summary)

    def _handle_sidebar_action(self, action: str):
        """Handles actions from the ChatActionsPanel in the sidebar."""
        if not self.ava_app:
            print(f"AvAApp not available to handle action: {action}")
            return

        if action == "open_terminal" or action == "view_log":
            self.ava_app._open_terminal()
        elif action == "open_code_viewer" or action == "view_code":
            self.ava_app._open_code_viewer()
        elif action == "new_session":
            self.chat_interface.chat_display.clear()
            self.chat_interface.chat_display.add_assistant_message("New session started! How can I help you today?")
            if hasattr(self.ava_app, 'current_session'):
                self.ava_app.current_session = "New Session"
            self.update_project_display(
                self.ava_app.current_project if hasattr(self.ava_app, 'current_project') else "Default Project")
        elif action == "force_gen":
            self.chat_interface.chat_display.add_assistant_message(
                "Force code generation triggered (logic to be implemented).")
        elif action == "check_updates":
            self.chat_interface.chat_display.add_assistant_message(
                "Checking for updates (feature not yet implemented).")
        else:
            print(f"Unknown sidebar action from ChatActionsPanel: {action}")

    def _handle_rag_scan_directory(self):
        if self.ava_app and self.ava_app.rag_manager:
            self.ava_app.rag_manager.scan_directory_dialog(parent_widget=self)
        else:
            self.chat_interface.chat_display.add_assistant_message("RAG Manager is not available to scan directory.")

    @Slot(str)
    def on_workflow_started(self, prompt: str):
        self.chat_interface.update_llm_status("Workflow: Starting specialists...", "working")
        self.chat_interface.update_specialists_status("AI Specialists: Active", "working")

    @Slot(dict)
    def on_workflow_completed(self, result: dict):
        success = result.get("success", False)

        if success:
            project_name = result.get("project_name", "your project")
            num_files = result.get("file_count", 0)
            message = f"✅ Enhanced workflow for '{project_name}' completed! Generated {num_files} files with AI specialists. View them in the Code Viewer."
            self.chat_interface.chat_display.add_assistant_message(message)
            self.chat_interface.update_specialists_status("AI Specialists: Completed", "success")
        else:
            error_msg = result.get("error", "An unknown error occurred.")
            self.chat_interface.chat_display.add_assistant_message(f"❌ Enhanced workflow failed: {error_msg}")
            self.chat_interface.update_specialists_status("AI Specialists: Error", "error")

        # Reset chat LLM status
        self._update_chat_llm_status()

    @Slot(str, str)
    def on_app_error_occurred(self, component: str, error_message: str):
        self.chat_interface.chat_display.add_assistant_message(f"⚠️ Error in {component}: {error_message}")
        if "workflow" in component.lower() or "specialist" in component.lower():
            self.chat_interface.update_specialists_status("AI Specialists: Error", "error")
        self._update_chat_llm_status()

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
                "warning": "#ffb900"
            }
            text_color_hex = key_to_hex_map.get(color_or_key, "#888888")

        if hasattr(self.sidebar, 'update_sidebar_rag_status'):
            self.sidebar.update_sidebar_rag_status(status_text, text_color_hex)

    @Slot(str)
    def update_project_display(self, project_name_or_path: str):
        project_name = project_name_or_path
        if "/" in project_name or "\\" in project_name:
            from pathlib import Path
            project_name = Path(project_name_or_path).name

        session_name = "Main Chat"  # Default session name
        if self.ava_app and hasattr(self.ava_app, 'current_session'):
            # Check if AvAApplication exists and has current_session
            current_session_val = getattr(self.ava_app, 'current_session', None)
            if current_session_val:  # Ensure it's not None or empty
                session_name = current_session_val

        # Enhanced title with AI specialist info
        if self.ava_app and hasattr(self.ava_app.llm_client, 'get_role_assignments'):
            assignments = self.ava_app.llm_client.get_role_assignments()
            planner = assignments.get('planner', 'N/A').split('-')[-1][:8] if assignments.get('planner') else 'N/A'
            coder = assignments.get('coder', 'N/A').split('-')[-1][:8] if assignments.get('coder') else 'N/A'
            self.setWindowTitle(f"AvA Enhanced [{project_name}] - Session: {session_name} (P:{planner} C:{coder})")
        else:
            self.setWindowTitle(f"AvA [{project_name}] - Session: {session_name}")