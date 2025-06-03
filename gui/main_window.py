# gui/main_window.py - Enhanced with Feedback Panel and AI Collaboration

import asyncio
import inspect

from PySide6.QtCore import Signal, Slot, QTimer, Qt
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QLabel, QTextEdit, QSplitter, QTabWidget
)

from gui.components import ModernButton, StatusIndicator
from gui.enhanced_sidebar import AvALeftSidebar
from gui.model_config_dialog import ModelConfigurationDialog

# NEW: Import feedback panel
try:
    from gui.feedback_panel import FeedbackPanel

    FEEDBACK_PANEL_AVAILABLE = True
except ImportError:
    print("Feedback panel not available - enhanced features disabled")
    FEEDBACK_PANEL_AVAILABLE = False

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

        # Enhanced welcome message
        welcome_msg = "Hello! I'm AvA with enhanced AI specialists. I can help you build applications using my Planner, Coder, and Assembler AIs"
        if FEEDBACK_PANEL_AVAILABLE:
            welcome_msg += " with real-time collaboration monitoring"
        welcome_msg += ". What would you like to work on?"

        self.append(self._format_message("AvA", welcome_msg, "assistant"))

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

        # Sanitize message for HTML
        import html
        escaped_message = html.escape(message)

        return f"""
        <div style="margin: 8px 0; padding: 8px 12px; background: {bg}; border-radius: 8px; border-left: 3px solid {color};">
            <div style="font-weight: bold; color: {color}; margin-bottom: 4px;">{sender}:</div>
            <div style="color: #cccccc; line-height: 1.4; white-space: pre-wrap;">{escaped_message}</div>
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
        placeholder_text = "Chat with AvA... (type 'build' or 'create' to start a project"
        if FEEDBACK_PANEL_AVAILABLE:
            placeholder_text += " with AI collaboration)"
        else:
            placeholder_text += ")"
        self.input_field.setPlaceholderText(placeholder_text)
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

    # NEW: Enhanced workflow signals
    feedback_settings_changed = Signal(dict)
    user_feedback_added = Signal(str, str, int, str)  # type, content, rating, file_path
    iteration_requested = Signal(str, str)  # file_path, feedback

    def __init__(self, ava_app=None, config=None):
        super().__init__()
        self.ava_app = ava_app
        self.show_feedback_panel = FEEDBACK_PANEL_AVAILABLE

        # Enhanced title based on available features
        title = "AvA - Enhanced AI Development Assistant"
        if FEEDBACK_PANEL_AVAILABLE:
            title += " with AI Collaboration"
        self.setWindowTitle(title)

        self.setGeometry(100, 100, 1600, 900)  # Wider to accommodate feedback panel
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

            # NEW: Connect enhanced workflow signals if available
            if hasattr(self.ava_app, 'ai_collaboration_started'):
                self.ava_app.ai_collaboration_started.connect(self.on_ai_collaboration_started)
            if hasattr(self.ava_app, 'ai_feedback_received'):
                self.ava_app.ai_feedback_received.connect(self.on_ai_feedback_received)
            if hasattr(self.ava_app, 'iteration_completed'):
                self.ava_app.iteration_completed.connect(self.on_iteration_completed)
            if hasattr(self.ava_app, 'quality_check_completed'):
                self.ava_app.quality_check_completed.connect(self.on_quality_check_completed)

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

        # Left sidebar
        self.sidebar = AvALeftSidebar()

        # NEW: Create main content area with optional feedback panel
        if FEEDBACK_PANEL_AVAILABLE:
            # Use splitter for resizable layout with feedback panel
            self.main_splitter = QSplitter()
            self.main_splitter.setOrientation(Qt.Orientation.Horizontal)  # Horizontal orientation

            # Chat interface
            self.chat_interface = ChatInterface()

            # NEW: Feedback panel
            self.feedback_panel = FeedbackPanel()
            self.feedback_panel.setMinimumWidth(350)
            self.feedback_panel.setMaximumWidth(450)

            # Add to splitter
            self.main_splitter.addWidget(self.chat_interface)
            self.main_splitter.addWidget(self.feedback_panel)

            # Set initial splitter sizes (chat interface gets more space)
            self.main_splitter.setSizes([800, 350])
            self.main_splitter.setCollapsible(0, False)  # Chat interface not collapsible
            self.main_splitter.setCollapsible(1, True)  # Feedback panel can be collapsed

            main_layout.addWidget(self.sidebar)
            main_layout.addWidget(self.main_splitter, 1)
        else:
            # Standard layout without feedback panel
            self.chat_interface = ChatInterface()
            self.feedback_panel = None

            main_layout.addWidget(self.sidebar)
            main_layout.addWidget(self.chat_interface, 1)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def _toggle_feedback_panel(self):
        """NEW: Toggle feedback panel visibility"""
        if FEEDBACK_PANEL_AVAILABLE and hasattr(self, 'feedback_panel'):
            current_visibility = self.feedback_panel.isVisible()
            self.feedback_panel.setVisible(not current_visibility)

            if not current_visibility:
                # Show feedback panel
                self.main_splitter.setSizes([800, 350])
            else:
                # Hide feedback panel
                self.main_splitter.setSizes([1150, 0])

    def _connect_signals(self):
        self.chat_interface.message_sent.connect(self.handle_user_message)

        # Connect signals from the sidebar directly
        self.sidebar.new_project_requested.connect(self.new_project_requested.emit)
        self.sidebar.scan_directory_requested.connect(self._handle_rag_scan_directory)
        self.sidebar.action_triggered.connect(self._handle_sidebar_action)

        # Connect model configuration signal
        self.sidebar.model_config_requested.connect(self._open_model_config_dialog)

        # NEW: Connect feedback panel signals if available
        if FEEDBACK_PANEL_AVAILABLE and hasattr(self, 'feedback_panel'):
            self.feedback_panel.feedback_settings_changed.connect(self._on_feedback_settings_changed)
            self.feedback_panel.user_feedback_added.connect(self._on_user_feedback_added)
            self.feedback_panel.iteration_requested.connect(self._on_iteration_requested)

    # NEW: Enhanced workflow signal handlers
    @Slot(dict)
    def _on_feedback_settings_changed(self, settings: dict):
        """Handle feedback settings changes"""
        if self.ava_app and hasattr(self.ava_app, 'update_workflow_settings'):
            self.ava_app.update_workflow_settings(settings)
        self.feedback_settings_changed.emit(settings)

    @Slot(str, str, int, str)
    def _on_user_feedback_added(self, feedback_type: str, content: str, rating: int, file_path: str):
        """Handle user feedback"""
        if self.ava_app and hasattr(self.ava_app, 'add_user_feedback'):
            self.ava_app.add_user_feedback(feedback_type, content, rating, file_path)
        self.user_feedback_added.emit(feedback_type, content, rating, file_path)

    @Slot(str, str)
    def _on_iteration_requested(self, file_path: str, feedback: str):
        """Handle iteration request"""
        if self.ava_app and hasattr(self.ava_app, 'request_file_iteration'):
            success = self.ava_app.request_file_iteration(file_path, feedback)
            if success:
                self.chat_interface.chat_display.add_assistant_message(
                    f"🔄 Iteration requested for {file_path}. Processing your feedback..."
                )
        self.iteration_requested.emit(file_path, feedback)

    # NEW: Enhanced workflow event handlers
    @Slot(str)
    def on_ai_collaboration_started(self, session_id: str):
        """Handle AI collaboration session start"""
        if FEEDBACK_PANEL_AVAILABLE and hasattr(self, 'feedback_panel'):
            self.feedback_panel.add_ai_collaboration_message(
                "system", "", f"🤝 Collaboration session started: {session_id}"
            )

    @Slot(str, str, str)
    def on_ai_feedback_received(self, from_ai: str, to_ai: str, content: str):
        """Handle AI feedback messages"""
        if FEEDBACK_PANEL_AVAILABLE and hasattr(self, 'feedback_panel'):
            self.feedback_panel.add_ai_collaboration_message(from_ai, content, to_ai)  # Corrected order

    @Slot(str, int)
    def on_iteration_completed(self, file_path: str, iteration_number: int):
        """Handle iteration completion"""
        if FEEDBACK_PANEL_AVAILABLE and hasattr(self, 'feedback_panel'):
            self.feedback_panel.update_iteration_count(iteration_number)

        # Update available files for iteration requests
        if self.ava_app and hasattr(self.ava_app, 'project_state_manager'):
            if self.ava_app.project_state_manager:
                available_files = list(self.ava_app.project_state_manager.files.keys())
                if FEEDBACK_PANEL_AVAILABLE and hasattr(self, 'feedback_panel'):
                    self.feedback_panel.update_available_files(available_files)

    @Slot(str, bool, str)
    def on_quality_check_completed(self, file_path: str, approved: bool, feedback: str):
        """Handle quality check completion"""
        if FEEDBACK_PANEL_AVAILABLE and hasattr(self, 'feedback_panel'):
            self.feedback_panel.show_file_completed(file_path, approved, 1)  # Iteration number might need to be tracked

    def _open_model_config_dialog(self):
        """Open the model configuration dialog"""
        if not self.ava_app or not self.ava_app.llm_client:
            self.chat_interface.chat_display.add_assistant_message(
                "⚠️ LLM client is not available. Cannot configure models."
            )
            return

        dialog = ModelConfigurationDialog(llm_client=self.ava_app.llm_client, parent=self)
        dialog.configuration_applied.connect(self._on_model_configuration_applied)
        dialog.exec()

    def _on_model_configuration_applied(self, new_config: dict):
        """Handle when model configuration is applied"""
        if hasattr(self.ava_app.llm_client, 'get_role_assignments'):
            assignments = self.ava_app.llm_client.get_role_assignments()  # Returns Dict[str, str]
            config_summary = {}
            for role_str_key, model_name_key in assignments.items():  # role_str_key is "planner", "coder" etc.
                if model_name_key and hasattr(self.ava_app.llm_client, 'models'):
                    model_config = self.ava_app.llm_client.models.get(model_name_key)
                    if model_config:
                        display_name = f"{model_config.provider}/{model_config.model}"
                        # Use the string key directly
                        if role_str_key == LLMRole.PLANNER.value:  # Compare with enum's value
                            config_summary['planner'] = display_name
                        elif role_str_key == LLMRole.CODER.value:
                            config_summary['coder'] = display_name
                        elif role_str_key == LLMRole.ASSEMBLER.value:
                            config_summary['assembler'] = display_name
            self.sidebar.update_model_status_display(config_summary)
        self._update_specialists_status()
        self.chat_interface.chat_display.add_assistant_message(
            "✅ Model configuration updated! Your AI specialists are now ready with their optimized models."
        )

    def handle_user_message(self, message: str):
        """Handle user messages - decide between casual chat vs workflow"""
        if self._is_build_request(message):
            if FEEDBACK_PANEL_AVAILABLE and hasattr(self, 'feedback_panel'):
                self.feedback_panel.clear_display()
                self.feedback_panel.update_workflow_stage("initializing", "Starting enhanced workflow...")
            self.workflow_requested.emit(message)
        else:
            self._handle_casual_chat(message)

    def _is_build_request(self, prompt: str) -> bool:
        prompt_lower = prompt.lower().strip()
        casual_phrases = ['hi', 'hello', 'hey', 'thanks', 'thank you', 'ok', 'okay', 'yes', 'no', 'sure', 'cool',
                          'nice', 'good', 'great']
        if prompt_lower in casual_phrases: return False
        question_words = ['what', 'how', 'why', 'when', 'where', 'who']
        if any(prompt_lower.startswith(word) for word in question_words):
            build_question_patterns = ['how to build', 'how to create', 'how to make', 'what should i build']
            if not any(pattern in prompt_lower for pattern in build_question_patterns): return False
        build_keywords = ['build', 'create', 'make', 'generate', 'develop', 'code', 'implement', 'write', 'design',
                          'construct', 'program', 'application', 'app', 'website', 'tool', 'script', 'project']
        has_build_keyword = any(keyword in prompt_lower for keyword in build_keywords)
        is_substantial = len(prompt.split()) >= 3
        return has_build_keyword and is_substantial

    def _handle_casual_chat(self, message: str):
        if not self.ava_app or not self.ava_app.llm_client:
            self.chat_interface.chat_display.add_assistant_message("Sorry, LLM client is not available right now.")
            return
        self.chat_interface.update_llm_status("LLM: Chatting...", "working")
        enhanced_features = "I also have real-time AI collaboration monitoring and user feedback controls. " if FEEDBACK_PANEL_AVAILABLE else ""
        chat_prompt = f"""
You are AvA, a friendly AI development assistant with specialized AI agents. User said: "{message}"
Respond naturally. Capabilities: Planner, Coder, Assembler AIs; micro-task workflow; {enhanced_features}knowledge base; iterative improvements.
Keep responses conversational and brief unless asked for details.
"""
        asyncio.create_task(self._async_casual_chat(chat_prompt))

    async def _async_casual_chat(self, prompt: str):
        try:
            response_chunks = []
            # Use CHAT role for casual conversation
            llm_client = self.ava_app.llm_client
            if hasattr(llm_client, 'stream_chat') and 'role' in inspect.signature(llm_client.stream_chat).parameters:
                async for chunk in llm_client.stream_chat(prompt, LLMRole.CHAT):  # Pass the enum member
                    response_chunks.append(chunk)
            else:  # Fallback for older client signature
                async for chunk in llm_client.stream_chat(prompt):
                    response_chunks.append(chunk)
            response = ''.join(response_chunks).strip()
            self.chat_interface.chat_display.add_assistant_message(response)
            self._update_chat_llm_status()
        except Exception as e:
            self.chat_interface.chat_display.add_assistant_message(f"Sorry, I encountered an error: {e}")
            self.chat_interface.update_llm_status("LLM: Error", "error")

    def _update_initial_ui_status(self):
        self._update_chat_llm_status()
        self._update_specialists_status()
        self._update_model_config_display()
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
        project_name_to_display = "Default Project"
        if self.ava_app and hasattr(self.ava_app,
                                    'current_project'): project_name_to_display = self.ava_app.current_project
        self.update_project_display(project_name_to_display)
        if FEEDBACK_PANEL_AVAILABLE and hasattr(self, 'feedback_panel'):
            self.feedback_panel.update_workflow_stage("idle", "Ready for enhanced AI collaboration")
            if (self.ava_app and hasattr(self.ava_app, 'use_enhanced_workflow') and
                    self.ava_app.use_enhanced_workflow and hasattr(self.ava_app, 'project_state_manager')):
                if self.ava_app.project_state_manager:
                    available_files = list(self.ava_app.project_state_manager.files.keys())
                    self.feedback_panel.update_available_files(available_files)

    def _update_chat_llm_status(self):
        llm_model_text = "LLM: Unknown"
        llm_indicator_status = "offline"
        if self.ava_app and self.ava_app.llm_client and hasattr(self.ava_app.llm_client, 'get_role_assignments'):
            assignments = self.ava_app.llm_client.get_role_assignments()  # Dict[str, str]
            chat_model_key = assignments.get(LLMRole.CHAT.value)  # Use .value for string key lookup
            if chat_model_key and hasattr(self.ava_app.llm_client, 'models'):
                model_config = self.ava_app.llm_client.models.get(chat_model_key)
                if model_config:
                    chat_model_name = model_config.model
                    current_temp = model_config.temperature
                    llm_model_text = f"Chat: {chat_model_name} (T: {current_temp:.2f})"
                    llm_indicator_status = "ready"
        self.chat_interface.update_llm_status(llm_model_text, llm_indicator_status)

    def _update_specialists_status(self):
        specialists_text = "AI Specialists: Unknown"
        specialists_status = "offline"
        if self.ava_app and self.ava_app.llm_client and hasattr(self.ava_app.llm_client, 'get_role_assignments'):
            assignments = self.ava_app.llm_client.get_role_assignments()  # Dict[str, str]
            planner_model_key = assignments.get(LLMRole.PLANNER.value, 'N/A')
            coder_model_key = assignments.get(LLMRole.CODER.value, 'N/A')

            def get_short_name(model_key_str):
                if model_key_str == 'N/A' or not hasattr(self.ava_app.llm_client, 'models'): return 'N/A'
                mc = self.ava_app.llm_client.models.get(model_key_str)
                return mc.model.split('/')[-1][:8] if mc else 'N/A'

            planner_model_short = get_short_name(planner_model_key)
            coder_model_short = get_short_name(coder_model_key)

            specialists_text = f"Specialists: P:{planner_model_short} C:{coder_model_short}"
            specialists_status = "ready" if planner_model_key != 'N/A' and coder_model_key != 'N/A' else "offline"
        self.chat_interface.update_specialists_status(specialists_text, specialists_status)

    def _update_model_config_display(self):
        if not (self.ava_app and self.ava_app.llm_client and hasattr(self.ava_app.llm_client,
                                                                     'get_role_assignments')): return
        assignments = self.ava_app.llm_client.get_role_assignments()  # Dict[str, str]
        config_summary = {}
        for role_str_key, model_name_key in assignments.items():  # role_str_key is "planner", "coder"
            if model_name_key and hasattr(self.ava_app.llm_client, 'models'):
                model_config = self.ava_app.llm_client.models.get(model_name_key)
                if model_config:
                    display_name = f"{model_config.provider}/{model_config.model}"
                    # Use the string key directly
                    if role_str_key == LLMRole.PLANNER.value:
                        config_summary['planner'] = display_name
                    elif role_str_key == LLMRole.CODER.value:
                        config_summary['coder'] = display_name
                    elif role_str_key == LLMRole.ASSEMBLER.value:
                        config_summary['assembler'] = display_name
        self.sidebar.update_model_status_display(config_summary)

    def _handle_sidebar_action(self, action: str):
        if not self.ava_app: print(f"AvAApp not available for action: {action}"); return
        if action == "open_terminal" or action == "view_log":
            self.ava_app._open_terminal()
        elif action == "open_code_viewer" or action == "view_code":
            self.ava_app._open_code_viewer()
        elif action == "toggle_feedback_panel":
            self._toggle_feedback_panel()
        elif action == "new_session":
            self.chat_interface.chat_display.clear()
            self.chat_interface.chat_display.add_assistant_message("New session started! How can I help you today?")
            if hasattr(self.ava_app, 'current_session'): self.ava_app.current_session = "New Session"
            self.update_project_display(
                self.ava_app.current_project if hasattr(self.ava_app, 'current_project') else "Default Project")
            if FEEDBACK_PANEL_AVAILABLE and hasattr(self, 'feedback_panel'): self.feedback_panel.clear_display()
        elif action == "force_gen":
            self.chat_interface.chat_display.add_assistant_message("Force code generation triggered (logic TBI).")
        elif action == "check_updates":
            self.chat_interface.chat_display.add_assistant_message("Checking for updates (feature TBI).")
        else:
            print(f"Unknown sidebar action: {action}")

    def _handle_rag_scan_directory(self):
        if self.ava_app and self.ava_app.rag_manager:
            self.ava_app.rag_manager.scan_directory_dialog(parent_widget=self)
        else:
            self.chat_interface.chat_display.add_assistant_message("RAG Manager is not available.")

    @Slot(str)
    def on_workflow_started(self, prompt: str):
        self.chat_interface.update_llm_status("Workflow: Starting...", "working")
        self.chat_interface.update_specialists_status("AI Specialists: Active", "working")
        if FEEDBACK_PANEL_AVAILABLE and hasattr(self, 'feedback_panel'):
            self.feedback_panel.update_workflow_stage("planning", "Starting enhanced AI collaboration...")

    @Slot(dict)
    def on_workflow_completed(self, result: dict):
        success = result.get("success", False)
        if success:
            project_name = result.get("project_name", "your project")
            num_files = result.get("file_count", 0)
            message = f"✅ Enhanced workflow for '{project_name}' completed! Generated {num_files} files."
            if FEEDBACK_PANEL_AVAILABLE: message += " Check AI Monitor for details."
            self.chat_interface.chat_display.add_assistant_message(message)
            self.chat_interface.update_specialists_status("AI Specialists: Completed", "success")
            if FEEDBACK_PANEL_AVAILABLE and hasattr(self, 'feedback_panel'): self.feedback_panel.show_workflow_complete(
                result)
        else:
            error_msg = result.get("error", "Unknown error.")
            self.chat_interface.chat_display.add_assistant_message(f"❌ Enhanced workflow failed: {error_msg}")
            self.chat_interface.update_specialists_status("AI Specialists: Error", "error")
        self._update_chat_llm_status()

    @Slot(str, str)
    def on_app_error_occurred(self, component: str, error_message: str):
        self.chat_interface.chat_display.add_assistant_message(f"⚠️ Error in {component}: {error_message}")
        if "workflow" in component.lower() or "specialist" in component.lower(): self.chat_interface.update_specialists_status(
            "AI Specialists: Error", "error")
        self._update_chat_llm_status()

    @Slot(str, str)
    def update_rag_status_display(self, status_text: str, color_or_key: str):
        self.chat_interface.update_rag_ui_status(status_text, color_or_key)
        text_color_hex = "#888888"
        if color_or_key.startswith("#"):
            text_color_hex = color_or_key
        else:
            key_to_hex_map = {"ready": "#4ade80", "success": "#4ade80", "working": "#ffb900", "error": "#ef4444",
                              "offline": "#888888", "grey": "#888888", "warning": "#ffb900"}
            text_color_hex = key_to_hex_map.get(color_or_key, "#888888")
        if hasattr(self.sidebar, 'update_sidebar_rag_status'): self.sidebar.update_sidebar_rag_status(status_text,
                                                                                                      text_color_hex)

    @Slot(str)
    def update_project_display(self, project_name_or_path: str):
        project_name = project_name_or_path
        if "/" in project_name or "\\" in project_name: from pathlib import Path; project_name = Path(
            project_name_or_path).name
        session_name = "Main Chat"
        current_session_val = getattr(self.ava_app, 'current_session', None) if self.ava_app else None
        if current_session_val: session_name = current_session_val
        base_title = f"AvA [{project_name}] - Session: {session_name}"
        if self.ava_app and hasattr(self.ava_app.llm_client, 'get_role_assignments'):
            assignments = self.ava_app.llm_client.get_role_assignments()  # Dict[str,str]

            def get_short_name_from_key(role_val_str):
                model_key = assignments.get(role_val_str)
                if not model_key or not hasattr(self.ava_app.llm_client, 'models'): return 'N/A'
                mc = self.ava_app.llm_client.models.get(model_key)
                return mc.model.split('/')[-1][:8] if mc else 'N/A'

            planner_short = get_short_name_from_key(LLMRole.PLANNER.value)
            coder_short = get_short_name_from_key(LLMRole.CODER.value)

            title_suffix = f" (Enhanced: P:{planner_short} C:{coder_short})" if FEEDBACK_PANEL_AVAILABLE and hasattr(
                self.ava_app,
                'use_enhanced_workflow') and self.ava_app.use_enhanced_workflow else f" (P:{planner_short} C:{coder_short})"
            base_title += title_suffix
        self.setWindowTitle(base_title)