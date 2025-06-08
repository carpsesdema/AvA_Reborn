# gui/main_window.py - Enhanced with Modern Chat Bubbles and Sleek Design

from PySide6.QtCore import Signal, Slot, QTimer
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout
)
from rich import text

from gui.chat_interface import ChatInterface  # <-- The star of the show!
from gui.components import Colors
from gui.enhanced_sidebar import AvALeftSidebar
from gui.model_config_dialog import ModelConfigurationDialog

# Import LLMRole for chat functionality
try:
    from core.llm_client import LLMRole
except ImportError:
    class LLMRole:
        PLANNER = "planner"
        CODER = "coder"
        ASSEMBLER = "assembler"
        REVIEWER = "reviewer"
        CHAT = "chat"


class AvAMainWindow(QMainWindow):
    """Enhanced main window with modern design"""

    # Signals for workflow integration
    new_project_requested = Signal()
    load_project_requested = Signal()
    workflow_requested_with_context = Signal(str, list)
    # Compatibility signal for the old application core
    workflow_requested = Signal(str)

    def __init__(self, ava_app=None):
        super().__init__()
        self.ava_app = ava_app

        # Setup window
        self.setWindowTitle("AvA - Fast Professional AI Development")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        self._init_ui()
        self._apply_theme()
        self._connect_signals()

        # Connect to AvA app if available
        if self.ava_app:
            self._connect_ava_signals()
            QTimer.singleShot(150, self._update_initial_ui_status)
        else:
            self._update_initial_ui_status()

    def _apply_theme(self):
        """Apply modern dark theme"""
        self.setStyleSheet(f"""
            QMainWindow {{ 
                background: {Colors.PRIMARY_BG}; 
                color: {Colors.TEXT_PRIMARY}; 
            }}
        """)

    def _init_ui(self):
        """Initialize the main UI layout"""
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar and chat interface
        self.sidebar = AvALeftSidebar()
        self.chat_interface = ChatInterface()  # <-- Using the new widget-based interface

        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.chat_interface, 1)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def _connect_signals(self):
        """Connect UI signals"""
        self.chat_interface.message_sent.connect(self.handle_user_message)
        self.sidebar.new_project_requested.connect(self.new_project_requested.emit)
        self.sidebar.load_project_requested.connect(self.load_project_requested.emit)
        self.sidebar.scan_directory_requested.connect(self._handle_rag_scan_directory)
        self.sidebar.action_triggered.connect(self._handle_sidebar_action)
        self.sidebar.model_config_requested.connect(self._open_model_config_dialog)

    def _connect_ava_signals(self):
        """Connect to AvA app signals"""
        try:
            self.ava_app.rag_status_changed.connect(self.update_rag_status_display)
            # self.ava_app.workflow_started.connect(self.on_workflow_started) # BUG FIX: This connection is redundant and causes the double message.
            self.ava_app.workflow_completed.connect(self.on_workflow_completed)
            self.ava_app.error_occurred.connect(self.on_app_error_occurred)
            self.ava_app.project_loaded.connect(self.update_project_display)

            # Enhanced workflow progress
            if hasattr(self.ava_app, 'workflow_engine') and self.ava_app.workflow_engine:
                # This is the single source of truth for workflow progress now.
                self.ava_app.workflow_engine.workflow_started.connect(self.on_workflow_started)
                if hasattr(self.ava_app.workflow_engine, 'workflow_progress'):
                    self.ava_app.workflow_engine.workflow_progress.connect(self.on_workflow_progress)
        except AttributeError:
            pass  # Some signals might not be available

    def _open_model_config_dialog(self):
        """Open modern model configuration dialog"""
        if not self.ava_app or not self.ava_app.llm_client:
            self.chat_interface.add_assistant_response("⚠️ LLM client unavailable.")
            return

        dialog = ModelConfigurationDialog(llm_client=self.ava_app.llm_client, parent=self)
        dialog.configuration_applied.connect(self._on_model_configuration_applied)
        dialog.exec()

    def _on_model_configuration_applied(self, applied_config_summary: dict):
        """Handle model configuration updates"""
        if hasattr(self.ava_app.llm_client, 'get_role_assignments'):
            current_assignments = self.ava_app.llm_client.get_role_assignments()
            display_summary = {}
            for role_str_key, model_name_key in current_assignments.items():
                if model_name_key and hasattr(self.ava_app.llm_client, 'models'):
                    model_config = self.ava_app.llm_client.models.get(model_name_key)
                    if model_config:
                        display_summary[role_str_key] = f"{model_config.provider}/{model_config.model}"
            self.sidebar.update_model_status_display(display_summary)
        self._update_specialists_status()
        self.chat_interface.add_assistant_response("✅ Model configuration updated!")

    def handle_user_message(self, message: str):
        """Enhanced message handling with streaming workflow integration"""

        # The chat_interface now handles adding the user message bubble itself.
        # We just need to decide what to do with the message.
        if self._is_build_request(message):
            self.chat_interface.add_workflow_status("Analyzing your request...")
            # This is the new, correct signal with chat history!
            self.workflow_requested_with_context.emit(message, self.chat_interface.conversation_history.copy())
        else:
            self._handle_casual_chat(message)

    def _is_build_request(self, prompt: str) -> bool:
        """Determine if message is a build request"""
        prompt_lower = prompt.lower().strip()
        casual_phrases = ['hi', 'hello', 'hey', 'thanks', 'thank you', 'ok', 'okay', 'yes', 'no', 'sure', 'cool',
                          'nice', 'good', 'great']
        if prompt_lower in casual_phrases:
            return False

        question_words = ['what', 'how', 'why', 'when', 'where', 'who']
        if any(prompt_lower.startswith(word) for word in question_words):
            build_question_patterns = ['how to build', 'how to create', 'how to make', 'what should i build']
            if not any(pattern in prompt_lower for pattern in build_question_patterns):
                return False

        build_keywords = ['build', 'create', 'make', 'generate', 'develop', 'code', 'implement', 'write', 'design',
                          'construct', 'program', 'application', 'app', 'website', 'tool', 'script', 'project',
                          'calculator']
        has_build_keyword = any(keyword in prompt_lower for keyword in build_keywords)
        is_substantial = len(prompt.split()) >= 3

        return has_build_keyword and is_substantial

    def _handle_casual_chat(self, message: str):
        """Handle casual chat messages"""
        if not self.ava_app or not self.ava_app.llm_client:
            self.chat_interface.add_assistant_response("Sorry, LLM client is not available right now.")
            return

        self.chat_interface.update_specialists_status("AI Specialists: Thinking...", "working")

        # Get chat response
        try:
            if hasattr(self.ava_app, 'handle_casual_chat'):
                response = self.ava_app.handle_casual_chat(message)
                self.chat_interface.add_assistant_response(response)
            else:
                self.chat_interface.add_assistant_response(
                    "I'm ready to help you build something! Just describe what you'd like to create.")
        except Exception as e:
            self.chat_interface.add_assistant_response(f"Sorry, I encountered an error: {str(e)}")

        self.chat_interface.update_specialists_status("AI Specialists: Ready", "ready")

    def _handle_rag_scan_directory(self):
        """Handle RAG directory scanning"""
        if self.ava_app and hasattr(self.ava_app, 'scan_directory'):
            self.chat_interface.add_workflow_status("Scanning directory for knowledge...")
            try:
                self.ava_app.scan_directory()
                self.chat_interface.add_assistant_response("📚 Directory scan complete! Knowledge base updated.")
            except Exception as e:
                self.chat_interface.add_assistant_response(f"❌ Scan failed: {str(e)}")
        else:
            self.chat_interface.add_assistant_response("⚠️ Directory scanning not available.")

    def _handle_sidebar_action(self, action: str):
        """Handle sidebar action triggers"""
        # NEW: Handle save and load actions
        if action == "save_session":
            if hasattr(self.ava_app, 'save_session'):
                self.ava_app.save_session()
            return
        if action == "load_session":
            if hasattr(self.ava_app, 'load_session'):
                self.ava_app.load_session()
            return

        action_messages = {
            "new_session": "🔄 Starting new session...",
            "view_log": "📊 Opening LLM log viewer...",
            "open_terminal": "📟 Opening terminal...",
            "open_code_viewer": "📄 Opening code viewer...",
            "check_updates": "🔄 Checking for updates..."
        }

        message = action_messages.get(action, f"Action '{action}' not implemented yet.")
        self.chat_interface.add_assistant_response(message)

    # Status update methods
    @Slot(str, str)
    def on_workflow_started(self, workflow_type: str, description: str = ""):
        """Handle workflow start"""
        self.chat_interface.update_specialists_status("AI Specialists: Working...", "working")
        self.chat_interface.update_performance_status("Performance: Processing...", "working")

        if description:
            self.chat_interface.add_workflow_status(f"Starting {workflow_type}: {description}")
        else:
            self.chat_interface.add_workflow_status(f"Starting {workflow_type} workflow...")

    @Slot(str)
    def on_workflow_progress(self, update: str):
        """Handle workflow progress updates"""
        self.chat_interface.add_workflow_status(update)

    @Slot(dict)
    def on_workflow_completed(self, result: dict):
        """Handle workflow completion"""
        if result.get("success", False):
            project_name = result.get("project_name", "Unknown")
            num_files = result.get("num_files", 0)
            strategy = result.get("strategy", "Standard")
            elapsed_time = result.get("elapsed_time", 0)

            message = f"""✅ **Workflow Complete!**

🎯 **Project:** {project_name}
📁 **Files Generated:** {num_files}
⚡ **Strategy:** {strategy}
⏱️ **Time:** {elapsed_time:.1f}s
📂 **Location:** {result.get('project_dir', 'Generated')}

Your professional code is ready! Check the Code Viewer to explore the generated files."""

            self.chat_interface.add_assistant_response(message)
            self.chat_interface.update_specialists_status("AI Specialists: Complete", "success")
            self.chat_interface.update_performance_status(f"Performance: {elapsed_time:.1f}s", "success")

        else:
            error_msg = result.get("error", "Unknown error.")
            elapsed_time = result.get("elapsed_time", 0)

            failure_message = f"""❌ **Workflow Failed**

⚠️ **Error:** {error_msg}
⏱️ **Time:** {elapsed_time:.1f}s

Let me know if you'd like to try again or need help with something else."""

            self.chat_interface.add_assistant_response(failure_message)
            self.chat_interface.update_specialists_status("AI Specialists: Error", "error")
            self.chat_interface.update_performance_status("Performance: Error", "error")

        self._update_chat_llm_status()

    @Slot(str, str)
    def on_app_error_occurred(self, component: str, error_message: str):
        """Handle application errors"""
        error_text = f"⚠️ **System Error**\n\n**Component:** {component}\n**Error:** {error_message}"
        self.chat_interface.add_assistant_response(error_text)

        if "workflow" in component.lower() or "specialist" in component.lower():
            self.chat_interface.update_specialists_status("AI Specialists: Error", "error")

        self._update_chat_llm_status()

    def _update_initial_ui_status(self):
        """Update initial UI status"""
        self._update_chat_llm_status()
        self._update_specialists_status()
        self._update_model_config_display()

        # RAG status
        rag_text = "RAG: Unknown"
        rag_color_key = "offline"
        if self.ava_app:
            try:
                app_status = self.ava_app.get_status()
                rag_info = app_status.get("rag", {})
                rag_text = rag_info.get("status_text", "RAG: Unknown")
                if rag_info.get("ready"):
                    rag_color_key = "success"
                elif not rag_info.get("available", True):
                    rag_color_key = "offline"
                elif "Initializing" in rag_text:
                    rag_color_key = "working"
                elif "Error" in rag_text or "Missing" in rag_text:
                    rag_color_key = "error"
            except:
                pass

        self.update_rag_status_display(rag_text, rag_color_key)

        # Project display
        project_name = "Default Project"
        if self.ava_app and hasattr(self.ava_app, 'current_project'):
            project_name = self.ava_app.current_project
        self.update_project_display(project_name)

    def _update_chat_llm_status(self):
        """Update chat LLM status"""
        # Implementation depends on your LLM client structure
        pass

    def _update_specialists_status(self):
        """Update AI specialists status"""
        status = "AI Specialists: Ready"
        color = "ready"

        if self.ava_app and hasattr(self.ava_app, 'llm_client'):
            try:
                if hasattr(self.ava_app.llm_client, 'get_role_assignments'):
                    assignments = self.ava_app.llm_client.get_role_assignments()
                    configured_count = sum(1 for v in assignments.values() if v)
                    total_count = len(assignments)

                    if configured_count == 0:
                        status = "AI Specialists: Not Configured"
                        color = "error"
                    elif configured_count < total_count:
                        status = f"AI Specialists: {configured_count}/{total_count} Configured"
                        color = "working"
                    else:
                        status = "AI Specialists: Ready"
                        color = "success"
            except:
                pass

        self.chat_interface.update_specialists_status(status, color)

    def _update_model_config_display(self):
        """Update model configuration display"""
        if self.ava_app and hasattr(self.ava_app, 'llm_client'):
            try:
                if hasattr(self.ava_app.llm_client, 'get_role_assignments'):
                    current_assignments = self.ava_app.llm_client.get_role_assignments()
                    display_summary = {}

                    for role_str_key, model_name_key in current_assignments.items():
                        if model_name_key and hasattr(self.ava_app.llm_client, 'models'):
                            model_config = self.ava_app.llm_client.models.get(model_name_key)
                            if model_config:
                                display_summary[role_str_key] = f"{model_config.provider}/{model_config.model}"

                    self.sidebar.update_model_status_display(display_summary)
            except:
                pass

    # Public interface methods
    def update_project_display(self, project_name: str):
        """Update project display"""
        if hasattr(self.sidebar, 'update_project_display'):
            self.sidebar.update_project_display(project_name)

    def update_rag_status_display(self, status_text: str, status_color: str = "working"):
        """Update RAG status display"""
        if hasattr(self.sidebar, 'update_rag_status_display'):
            self.sidebar.update_rag_status_display(status_text)
        self.chat_interface.update_rag_status(status_text, status_color)

    def load_chat_history(self, history: list):
        """
        Clears the current chat and loads a new conversation history.
        This is a public method called by AvAApplication when a session is loaded.
        """
        self.chat_interface.conversation_history = history

        # Clear the visual chat bubbles
        while self.chat_interface.chat_scroll.content_layout.count() > 0:
            item = self.chat_interface.chat_scroll.content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Re-add the stretch
        self.chat_interface.chat_scroll.content_layout.addStretch()

        # Re-populate the chat with the loaded history
        for message_data in history:
            role = message_data.get("role")
            message = message_data.get("message")

            # Recreate the bubble. This assumes the bubble can be created from this data.
            # You might need to adjust the ChatBubble constructor or how you store timestamps.
            if role == "user":
                self.chat_interface.add_user_message(message)
            elif role == "assistant":
                self.chat_interface.add_assistant_response(message)
            elif role == "streaming":  # Or 'workflow', depending on your history
                self.chat_interface.add_workflow_status(message)