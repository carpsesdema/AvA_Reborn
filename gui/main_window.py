# gui/main_window.py - Enhanced with Modern Chat Bubbles and Sleek Design

import asyncio
import inspect
import html
from datetime import datetime

from PySide6.QtCore import Signal, Slot, QTimer, Qt
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QLabel, QTextEdit, QSplitter, QTabWidget, QFrame
)
from PySide6.QtGui import QFont

from gui.components import ModernButton, StatusIndicator, Colors, Typography
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


class ChatDisplay(QTextEdit):
    """Modern chat display with beautiful message bubbles"""

    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setStyleSheet(f"""
            QTextEdit {{
                background: {Colors.PRIMARY_BG};
                border: 1px solid {Colors.BORDER_DEFAULT};
                border-radius: 12px;
                color: {Colors.TEXT_PRIMARY};
                padding: 16px;
                font-family: "Segoe UI";
                font-size: 13px;
                line-height: 1.6;
            }}
            QScrollBar:vertical {{
                background: {Colors.SECONDARY_BG};
                width: 8px;
                border-radius: 4px;
                margin: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: {Colors.BORDER_DEFAULT};
                border-radius: 4px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {Colors.ACCENT_BLUE};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                border: none;
                background: none;
                height: 0px;
            }}
        """)

        welcome_msg = """Hello! I'm AvA, your fast professional AI development assistant.

🚀 **Ready to build something amazing?**

I use specialized AI agents:
- **Planner** - Creates smart project architecture
- **Coder** - Generates clean, professional code  
- **Assembler** - Integrates everything seamlessly
- **Reviewer** - Ensures quality and best practices

✨ **Just tell me what you want to build!**
Examples: "Create a calculator GUI", "Build a web API", "Make a file organizer tool" """

        self.append(self._format_message("AvA", welcome_msg, "assistant"))

    def add_user_message(self, message: str):
        """Add a user message with modern bubble styling"""
        self.append(self._format_message("You", message, "user"))

    def add_assistant_message(self, message: str):
        """Add assistant response with modern bubble styling"""
        self.append(self._format_message("AvA", message, "assistant"))

    def add_streaming_message(self, message: str):
        """Add streaming message with special formatting"""
        self.append(self._format_message("AvA", f"🔄 {message}", "streaming"))

    def _format_message(self, sender: str, message: str, role: str) -> str:
        """Format message with modern bubble design"""
        timestamp = datetime.now().strftime("%H:%M")

        # Define bubble styling based on role
        if role == "user":
            bubble_bg = f"""
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {Colors.ACCENT_BLUE}, stop:1 #1f6feb);
            """
            text_color = Colors.TEXT_PRIMARY
            sender_color = Colors.TEXT_PRIMARY
            time_color = "rgba(240, 246, 252, 0.8)"
            icon = "👤"
            align = "margin-left: 60px; margin-right: 20px;"
            border_radius = "18px 18px 4px 18px"

        elif role == "streaming":
            bubble_bg = f"""
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {Colors.ACCENT_ORANGE}, stop:1 #d18616);
            """
            text_color = Colors.TEXT_PRIMARY
            sender_color = Colors.TEXT_PRIMARY
            time_color = "rgba(240, 246, 252, 0.8)"
            icon = "⚡"
            align = "margin-left: 20px; margin-right: 60px;"
            border_radius = "18px 18px 18px 4px"

        else:  # assistant
            bubble_bg = f"""
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {Colors.ELEVATED_BG}, stop:1 {Colors.SECONDARY_BG});
            """
            text_color = Colors.TEXT_PRIMARY
            sender_color = Colors.ACCENT_GREEN
            time_color = Colors.TEXT_MUTED
            icon = "🤖"
            align = "margin-left: 20px; margin-right: 60px;"
            border_radius = "18px 18px 18px 4px"

        # Escape HTML in message content
        escaped_message = html.escape(message).replace("\n", "<br>")

        # Create modern bubble HTML
        bubble_html = f"""
        <div style="{align} margin-bottom: 16px;">
            <div style="
                {bubble_bg}
                border: 1px solid {Colors.BORDER_DEFAULT};
                border-radius: {border_radius};
                padding: 16px 20px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
                position: relative;
            ">
                <div style="
                    display: flex; 
                    align-items: center; 
                    justify-content: space-between; 
                    margin-bottom: 8px;
                    font-weight: 600;
                ">
                    <span style="color: {sender_color}; display: flex; align-items: center;">
                        <span style="margin-right: 8px; font-size: 14px;">{icon}</span>
                        <span style="font-size: 13px;">{sender}</span>
                    </span>
                    <span style="color: {time_color}; font-size: 11px; font-weight: normal;">
                        {timestamp}
                    </span>
                </div>
                <div style="
                    color: {text_color}; 
                    line-height: 1.5;
                    font-size: 13px;
                    word-wrap: break-word;
                ">
                    {escaped_message}
                </div>
            </div>
        </div>
        """

        return bubble_html


class ChatInterface(QWidget):
    """Modern chat interface with improved styling"""

    message_sent = Signal(str)
    workflow_requested = Signal(str, list)

    def __init__(self):
        super().__init__()
        self.conversation_history = []
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # Chat display
        self.chat_display = ChatDisplay()
        layout.addWidget(self.chat_display, 1)

        # Input area with modern styling
        input_frame = QFrame()
        input_frame.setStyleSheet(f"""
            QFrame {{
                background: {Colors.SECONDARY_BG};
                border: 1px solid {Colors.BORDER_DEFAULT};
                border-radius: 12px;
                padding: 4px;
            }}
            QFrame:focus-within {{
                border-color: {Colors.ACCENT_BLUE};
            }}
        """)

        input_layout = QHBoxLayout(input_frame)
        input_layout.setContentsMargins(12, 8, 8, 8)
        input_layout.setSpacing(12)

        # Message input field
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type your message...")
        self.input_field.setFont(Typography.body())
        self.input_field.setStyleSheet(f"""
            QLineEdit {{
                background: transparent;
                border: none;
                color: {Colors.TEXT_PRIMARY};
                padding: 8px 12px;
                font-size: 13px;
                selection-background-color: {Colors.ACCENT_BLUE};
            }}
            QLineEdit::placeholder {{
                color: {Colors.TEXT_MUTED};
            }}
        """)
        self.input_field.returnPressed.connect(self._send_message)

        # Send button
        self.send_button = ModernButton("Send", button_type="primary")
        self.send_button.setMaximumWidth(80)
        self.send_button.clicked.connect(self._send_message)

        input_layout.addWidget(self.input_field, 1)
        input_layout.addWidget(self.send_button)

        layout.addWidget(input_frame)

        # Modern status bar
        status_bar_layout = QHBoxLayout()
        status_bar_layout.setContentsMargins(0, 12, 0, 0)
        status_bar_layout.setSpacing(16)

        # AI Specialists status
        self.specialists_indicator = StatusIndicator("ready")
        self.specialists_status_text = QLabel("AI Specialists: Ready")
        self.specialists_status_text.setFont(Typography.body_small())
        self.specialists_status_text.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")

        specialist_layout = QHBoxLayout()
        specialist_layout.setSpacing(6)
        specialist_layout.addWidget(self.specialists_indicator)
        specialist_layout.addWidget(self.specialists_status_text)

        status_bar_layout.addLayout(specialist_layout)
        status_bar_layout.addStretch(1)

        # Performance indicator
        self.performance_indicator = StatusIndicator("offline")
        self.performance_text = QLabel("Performance: Ready")
        self.performance_text.setFont(Typography.body_small())
        self.performance_text.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")

        performance_layout = QHBoxLayout()
        performance_layout.setSpacing(6)
        performance_layout.addWidget(self.performance_indicator)
        performance_layout.addWidget(self.performance_text)

        status_bar_layout.addLayout(performance_layout)

        # RAG Status
        self.rag_status_indicator = StatusIndicator("working")
        self.rag_status_text_label = QLabel("RAG: Initializing...")
        self.rag_status_text_label.setFont(Typography.body_small())
        self.rag_status_text_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")

        rag_layout = QHBoxLayout()
        rag_layout.setSpacing(6)
        rag_layout.addWidget(self.rag_status_indicator)
        rag_layout.addWidget(self.rag_status_text_label)

        status_bar_layout.addLayout(rag_layout)

        layout.addLayout(status_bar_layout)
        self.setLayout(layout)

        # Apply main styling
        self.setStyleSheet(f"""
            ChatInterface {{
                background: {Colors.PRIMARY_BG};
            }}
        """)

    def _send_message(self):
        """Send message with improved flow"""
        message = self.input_field.text().strip()
        if not message:
            return

        # Add to chat display
        self.chat_display.add_user_message(message)
        self.input_field.clear()

        # Store in conversation history
        self.conversation_history.append({
            "role": "user",
            "message": message,
            "timestamp": datetime.now().isoformat()
        })

        # Keep last 10 messages for context
        if len(self.conversation_history) > 10:
            self.conversation_history.pop(0)

        # Emit with conversation context
        self.message_sent.emit(message)

    def add_assistant_response(self, response: str):
        """Add assistant response to chat and history"""
        self.chat_display.add_assistant_message(response)

        # Store in conversation history
        self.conversation_history.append({
            "role": "assistant",
            "message": response,
            "timestamp": datetime.now().isoformat()
        })

    def add_workflow_status(self, status: str):
        """Add workflow status update"""
        self.chat_display.add_streaming_message(status)

    def update_specialists_status(self, text: str, status: str = "ready"):
        """Update specialists status"""
        self.specialists_status_text.setText(text)
        self.specialists_indicator.update_status(status)

    def update_performance_status(self, text: str, status: str = "ready"):
        """Update performance status"""
        self.performance_text.setText(text)
        self.performance_indicator.update_status(status)

    def update_rag_status(self, text: str, status: str = "working"):
        """Update RAG status"""
        self.rag_status_text_label.setText(text)
        self.rag_status_indicator.update_status(status)


class AvAMainWindow(QMainWindow):
    """Enhanced main window with modern design"""

    # Signals for workflow integration
    new_project_requested = Signal()
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
        self.chat_interface = ChatInterface()

        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.chat_interface, 1)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def _connect_signals(self):
        """Connect UI signals"""
        self.chat_interface.message_sent.connect(self.handle_user_message)
        self.sidebar.new_project_requested.connect(self.new_project_requested.emit)
        self.sidebar.scan_directory_requested.connect(self._handle_rag_scan_directory)
        self.sidebar.action_triggered.connect(self._handle_sidebar_action)
        self.sidebar.model_config_requested.connect(self._open_model_config_dialog)

    def _connect_ava_signals(self):
        """Connect to AvA app signals"""
        try:
            self.ava_app.rag_status_changed.connect(self.update_rag_status_display)
            self.ava_app.workflow_started.connect(self.on_workflow_started)
            self.ava_app.workflow_completed.connect(self.on_workflow_completed)
            self.ava_app.error_occurred.connect(self.on_app_error_occurred)
            self.ava_app.project_loaded.connect(self.update_project_display)

            # Enhanced workflow progress
            if hasattr(self.ava_app, 'workflow_engine') and self.ava_app.workflow_engine:
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

        if self._is_build_request(message):
            # Show immediate feedback
            self.chat_interface.add_workflow_status("Analyzing your request...")
            # Start workflow with conversation context
            self.workflow_requested_with_context.emit(message, self.chat_interface.conversation_history.copy())
            # Emit the simpler signal for the core application to connect to
            self.workflow_requested.emit(message)
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