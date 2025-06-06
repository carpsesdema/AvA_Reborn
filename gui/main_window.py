# gui/main_window.py - Enhanced with Real-time Streaming Integration

import asyncio
import inspect
import html
from datetime import datetime

from PySide6.QtCore import Signal, Slot, QTimer, Qt
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QLabel, QTextEdit, QSplitter, QTabWidget
)

from gui.components import ModernButton, StatusIndicator
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
    """Enhanced chat display with streaming support"""

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
                line-height: 1.5;
            }
            QScrollBar:vertical {
                background: #2d2d30;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #484f58;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #00d7ff;
            }
        """)

        welcome_msg = """Hello! I'm AvA, your fast professional AI development assistant.

🚀 **Ready to build something amazing?**

I use specialized AI agents:
- **Planner** - Creates smart project architecture
- **Coder** - Generates clean, professional code  
- **Assembler** - Integrates everything seamlessly
- **Reviewer** - Ensures quality and best practices

✨ **Just tell me what you want to build!**
Examples: "Create a calculator GUI", "Build a web API", "Make a file organizer tool"
"""
        self.append(self._format_message("AvA", welcome_msg, "assistant"))

    def add_user_message(self, message: str):
        self.append(self._format_message("You", message, "user"))

    def add_assistant_message(self, message: str):
        self.append(self._format_message("AvA", message, "assistant"))

    def add_streaming_message(self, message: str):
        """Add streaming message with special formatting"""
        self.append(self._format_message("AvA", f"🔄 {message}", "streaming"))

    def _format_message(self, sender: str, message: str, role: str) -> str:
        if role == "user":
            color = "#00d7ff"
            bg = "#2d2d30"
            icon = "👤"
        elif role == "streaming":
            color = "#ffb900"
            bg = "#2a2a2a"
            icon = "⚡"
        else:  # assistant
            color = "#3fb950"
            bg = "#252526"
            icon = "🤖"

        escaped_message = html.escape(message).replace("\n", "<br>")

        return f"""
        <div style="margin: 8px 0; padding: 12px; background: {bg}; border-radius: 8px; border-left: 3px solid {color};">
            <div style="font-weight: bold; color: {color}; margin-bottom: 6px; display: flex; align-items: center;">
                <span style="margin-right: 8px;">{icon}</span>
                <span>{sender}</span>
                <span style="margin-left: auto; font-size: 11px; color: #888;">{datetime.now().strftime('%H:%M:%S')}</span>
            </div>
            <div style="color: #cccccc; line-height: 1.5;">{escaped_message}</div>
        </div>
        """


class ChatInterface(QWidget):
    message_sent = Signal(str)
    workflow_requested = Signal(str, list)  # message, conversation_history

    def __init__(self):
        super().__init__()
        self.conversation_history = []
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        self.chat_display = ChatDisplay()
        layout.addWidget(self.chat_display, 1)

        # Enhanced input section
        input_layout = QHBoxLayout()
        input_layout.setSpacing(10)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("💬 Describe what you want to build... (try 'create a calculator app')")
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
                background: #353538;
            }
            QLineEdit::placeholder { 
                color: #888; 
                font-style: italic;
            }
        """)
        self.input_field.returnPressed.connect(self._send_message)

        self.send_btn = ModernButton("Send", button_type="accent")
        self.send_btn.clicked.connect(self._send_message)
        self.send_btn.setMinimumHeight(45)
        self.send_btn.setMinimumWidth(80)

        input_layout.addWidget(self.input_field, 1)
        input_layout.addWidget(self.send_btn)

        # Enhanced status bar with more information
        status_bar_layout = QHBoxLayout()
        status_bar_layout.setContentsMargins(0, 8, 0, 0)

        # LLM Status
        self.llm_status_indicator = StatusIndicator("offline")
        self.llm_status_text = QLabel("Chat: Initializing...")
        self.llm_status_text.setStyleSheet("color: #888; font-size: 11px; margin-left: 5px;")
        status_bar_layout.addWidget(self.llm_status_indicator)
        status_bar_layout.addWidget(self.llm_status_text)

        # AI Specialists Status
        self.specialists_status_indicator = StatusIndicator("offline")
        self.specialists_status_text = QLabel("AI Specialists: Initializing...")
        self.specialists_status_text.setStyleSheet("color: #888; font-size: 11px; margin-left: 5px;")
        status_bar_layout.addWidget(self.specialists_status_indicator)
        status_bar_layout.addWidget(self.specialists_status_text)

        status_bar_layout.addStretch(1)

        # Performance indicator
        self.performance_indicator = StatusIndicator("offline")
        self.performance_text = QLabel("Performance: Ready")
        self.performance_text.setStyleSheet("color: #888; font-size: 11px; margin-left: 5px;")
        status_bar_layout.addWidget(self.performance_indicator)
        status_bar_layout.addWidget(self.performance_text)

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

    def update_llm_status(self, text: str, indicator_status: str = "ready"):
        self.llm_status_text.setText(text)
        self.llm_status_indicator.update_status(indicator_status)

    def update_specialists_status(self, text: str, indicator_status: str = "ready"):
        self.specialists_status_text.setText(text)
        self.specialists_status_indicator.update_status(indicator_status)

    def update_performance_status(self, text: str, indicator_status: str = "ready"):
        self.performance_text.setText(text)
        self.performance_indicator.update_status(indicator_status)

    def update_rag_ui_status(self, text: str, color_or_key: str):
        self.rag_status_text_label.setText(text)
        text_color_hex = "#888"
        indicator_key = "offline"

        if color_or_key.startswith("#"):
            text_color_hex = color_or_key
            hex_map = {"#4ade80": "success", "#3fb950": "success", "#ffb900": "working",
                       "#ef4444": "error", "#f85149": "error", "#00d7ff": "ready", "#6a9955": "success"}
            indicator_key = hex_map.get(color_or_key.lower(), "offline")
        else:
            indicator_key = color_or_key
            key_map = {"ready": "#00d7ff", "success": "#4ade80", "working": "#ffb900",
                       "error": "#ef4444", "offline": "#888", "grey": "#888"}
            text_color_hex = key_map.get(color_or_key, "#888")

        self.rag_status_text_label.setStyleSheet(f"color: {text_color_hex}; font-size: 11px; margin-left: 5px;")
        self.rag_status_indicator.update_status(indicator_key)


class AvAMainWindow(QMainWindow):
    """Enhanced main window with streaming workflow integration"""

    # Streamlined signals
    workflow_requested = Signal(str)
    workflow_requested_with_context = Signal(str, list)
    new_project_requested = Signal()

    def __init__(self, ava_app=None, config=None):
        super().__init__()
        self.ava_app = ava_app

        self.setWindowTitle("AvA - Fast Professional AI Development")
        self.setGeometry(100, 100, 1400, 900)
        self._apply_theme()
        self._init_ui()
        self._connect_signals()

        if self.ava_app:
            self.ava_app.rag_status_changed.connect(self.update_rag_status_display)
            self.ava_app.workflow_started.connect(self.on_workflow_started)
            self.ava_app.workflow_completed.connect(self.on_workflow_completed)
            self.ava_app.error_occurred.connect(self.on_app_error_occurred)
            self.ava_app.project_loaded.connect(self.update_project_display)

            # NEW: Connect to enhanced workflow progress
            if hasattr(self.ava_app, 'workflow_engine') and self.ava_app.workflow_engine:
                if hasattr(self.ava_app.workflow_engine, 'workflow_progress'):
                    self.ava_app.workflow_engine.workflow_progress.connect(self.on_workflow_progress)

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

        # Sidebar + enhanced chat interface
        self.sidebar = AvALeftSidebar()
        self.chat_interface = ChatInterface()

        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.chat_interface, 1)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def _connect_signals(self):
        self.chat_interface.message_sent.connect(self.handle_user_message)
        self.sidebar.new_project_requested.connect(self.new_project_requested.emit)
        self.sidebar.scan_directory_requested.connect(self._handle_rag_scan_directory)
        self.sidebar.action_triggered.connect(self._handle_sidebar_action)
        self.sidebar.model_config_requested.connect(self._open_model_config_dialog)

    def _open_model_config_dialog(self):
        if not self.ava_app or not self.ava_app.llm_client:
            self.chat_interface.add_assistant_response("⚠️ LLM client unavailable.")
            return
        dialog = ModelConfigurationDialog(llm_client=self.ava_app.llm_client, parent=self)
        dialog.configuration_applied.connect(self._on_model_configuration_applied)
        dialog.exec()

    def _on_model_configuration_applied(self, applied_config_summary: dict):
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
        else:
            self._handle_casual_chat(message)

    def _is_build_request(self, prompt: str) -> bool:
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
        if not self.ava_app or not self.ava_app.llm_client:
            self.chat_interface.add_assistant_response("Sorry, LLM client is not available right now.")
            return

        self.chat_interface.update_llm_status("Chat: Responding...", "working")

        chat_prompt = f"""
You are AvA, a friendly AI development assistant with specialized AI agents for building applications.

User said: "{message}"

Respond naturally and conversationally. Keep it brief unless they ask for details.

Your capabilities:
- Fast professional code generation
- Specialized AI agents (Planner, Coder, Assembler, Reviewer)  
- Real-time streaming workflows
- Project-aware development

If they seem interested in building something, encourage them to describe what they want to create.
"""
        asyncio.create_task(self._async_casual_chat(chat_prompt, message))

    async def _async_casual_chat(self, prompt: str, original_message: str):
        try:
            response_chunks = []
            llm_client = self.ava_app.llm_client

            if hasattr(llm_client, 'stream_chat') and 'role' in inspect.signature(llm_client.stream_chat).parameters:
                async for chunk in llm_client.stream_chat(prompt, LLMRole.CHAT):
                    response_chunks.append(chunk)
            else:
                async for chunk in llm_client.stream_chat(prompt):
                    response_chunks.append(chunk)

            response = ''.join(response_chunks).strip()
            self.chat_interface.add_assistant_response(response)
            self._update_chat_llm_status()

        except Exception as e:
            self.chat_interface.add_assistant_response(f"Sorry, I encountered an error: {e}")
            self.chat_interface.update_llm_status("Chat: Error", "error")

    @Slot(str, str)
    def on_workflow_progress(self, stage: str, description: str):
        """Handle workflow progress updates with streaming display"""

        # Update chat with progress
        stage_messages = {
            "initializing": f"🔄 {description}",
            "context_discovery": f"🔍 {description}",
            "planning": f"🧠 {description}",
            "generation": f"⚡ {description}",
            "finalization": f"📄 {description}",
            "complete": f"✅ {description}",
            "error": f"❌ {description}"
        }

        if stage in stage_messages:
            self.chat_interface.add_workflow_status(stage_messages[stage])

        # Update performance status
        if stage == "generation":
            self.chat_interface.update_performance_status("Performance: Generating...", "working")
        elif stage == "complete":
            self.chat_interface.update_performance_status("Performance: Complete", "success")
        elif stage == "error":
            self.chat_interface.update_performance_status("Performance: Error", "error")

    @Slot(str)
    def on_workflow_started(self, prompt: str):
        self.chat_interface.update_llm_status("Workflow: Starting...", "working")
        self.chat_interface.update_specialists_status("AI Specialists: Active", "working")
        self.chat_interface.update_performance_status("Performance: Starting...", "working")

    @Slot(dict)
    def on_workflow_completed(self, result: dict):
        success = result.get("success", False)
        elapsed_time = result.get("elapsed_time", 0)

        if success:
            project_name = result.get("project_name", "your project")
            num_files = result.get("file_count", 0)
            strategy = result.get("strategy_used", "standard")

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
        error_text = f"⚠️ **System Error**\n\n**Component:** {component}\n**Error:** {error_message}"
        self.chat_interface.add_assistant_response(error_text)

        if "workflow" in component.lower() or "specialist" in component.lower():
            self.chat_interface.update_specialists_status("AI Specialists: Error", "error")

        self._update_chat_llm_status()

    def _update_initial_ui_status(self):
        self._update_chat_llm_status()
        self._update_specialists_status()
        self._update_model_config_display()

        # RAG status
        rag_text = "RAG: Unknown"
        rag_color_key = "offline"
        if self.ava_app:
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

        self.update_rag_status_display(rag_text, rag_color_key)

        # Project display
        project_name = "Default Project"
        if self.ava_app and hasattr(self.ava_app, 'current_project'):
            project_name = self.ava_app.current_project
        self.update_project_display(project_name)

    def _update_chat_llm_status(self):
        llm_model_text = "Chat: Unknown"
        llm_indicator_status = "offline"

        if self.ava_app and self.ava_app.llm_client and hasattr(self.ava_app.llm_client, 'get_role_assignments'):
            assignments = self.ava_app.llm_client.get_role_assignments()
            chat_model_key = assignments.get(LLMRole.CHAT.value)

            if chat_model_key and hasattr(self.ava_app.llm_client, 'models'):
                model_config = self.ava_app.llm_client.models.get(chat_model_key)
                if model_config:
                    chat_model_name = model_config.model
                    current_temp = model_config.temperature
                    llm_model_text = f"Chat: {chat_model_name.split('/')[-1][:15]} (T:{current_temp:.1f})"
                    llm_indicator_status = "ready"

        self.chat_interface.update_llm_status(llm_model_text, llm_indicator_status)

    def _update_specialists_status(self):
        specialists_text = "AI Specialists: Unknown"
        specialists_status = "offline"

        if self.ava_app and self.ava_app.llm_client and hasattr(self.ava_app.llm_client, 'get_role_assignments'):
            assignments = self.ava_app.llm_client.get_role_assignments()

            def get_short_model_name(role_value_str):
                model_key = assignments.get(role_value_str)
                if not model_key or not hasattr(self.ava_app.llm_client, 'models'):
                    return 'N/A'
                mc = self.ava_app.llm_client.models.get(model_key)
                return mc.model.split('/')[-1][:6] if mc else 'N/A'

            p_model = get_short_model_name(LLMRole.PLANNER.value)
            c_model = get_short_model_name(LLMRole.CODER.value)
            a_model = get_short_model_name(LLMRole.ASSEMBLER.value)
            r_model = get_short_model_name(LLMRole.REVIEWER.value)

            specialists_text = f"P:{p_model} C:{c_model} A:{a_model} R:{r_model}"
            all_assigned = all(m != 'N/A' for m in [p_model, c_model, a_model, r_model])
            specialists_status = "ready" if all_assigned else "working"
            if not any(m != 'N/A' for m in [p_model, c_model, a_model, r_model]):
                specialists_status = "offline"

        self.chat_interface.update_specialists_status(specialists_text, specialists_status)

    def _update_model_config_display(self):
        if not (self.ava_app and self.ava_app.llm_client and hasattr(self.ava_app.llm_client, 'get_role_assignments')):
            return

        assignments = self.ava_app.llm_client.get_role_assignments()
        config_summary = {}

        for role_enum_member in LLMRole:
            role_str_key = role_enum_member.value
            model_name_key = assignments.get(role_str_key)
            if model_name_key and hasattr(self.ava_app.llm_client, 'models'):
                model_config = self.ava_app.llm_client.models.get(model_name_key)
                if model_config:
                    config_summary[role_str_key] = f"{model_config.provider}/{model_config.model}"
            else:
                config_summary[role_str_key] = "Not Configured"

        self.sidebar.update_model_status_display(config_summary)

    def _handle_sidebar_action(self, action: str):
        if not self.ava_app:
            return

        if action == "open_terminal":
            self.ava_app._open_terminal()
        elif action == "open_code_viewer":
            self.ava_app._open_code_viewer()
        elif action == "new_session":
            self.chat_interface.chat_display.clear()
            self.chat_interface.chat_display.add_assistant_response("""🆕 **New Session Started!**

Ready to build something amazing? Just describe what you want to create:

💡 **Quick Examples:**
- "Create a calculator with GUI"
- "Build a file organizer tool"  
- "Make a web scraper"
- "Design a password generator"

What would you like to work on?""")
            self.chat_interface.conversation_history.clear()
        else:
            self.chat_interface.add_assistant_response(f"Action '{action}' triggered.")

    def _handle_rag_scan_directory(self):
        if self.ava_app and self.ava_app.rag_manager:
            self.ava_app.rag_manager.scan_directory_dialog(parent_widget=self)
        else:
            self.chat_interface.add_assistant_response("RAG Manager is not available.")

    @Slot(str, str)
    def update_rag_status_display(self, status_text: str, color_or_key: str):
        self.chat_interface.update_rag_ui_status(status_text, color_or_key)

        text_color_hex = "#888888"
        if color_or_key.startswith("#"):
            text_color_hex = color_or_key
        else:
            key_to_hex_map = {"ready": "#4ade80", "success": "#4ade80", "working": "#ffb900",
                              "error": "#ef4444", "offline": "#888888", "grey": "#888888"}
            text_color_hex = key_to_hex_map.get(color_or_key, "#888888")

        if hasattr(self.sidebar, 'update_sidebar_rag_status'):
            self.sidebar.update_sidebar_rag_status(status_text, text_color_hex)

    @Slot(str)
    def update_project_display(self, project_name_or_path: str):
        """Update project display in window title"""
        project_name = project_name_or_path
        if "/" in project_name or "\\" in project_name:
            from pathlib import Path
            project_name = Path(project_name_or_path).name

        session_name = "Main Chat"
        current_session_val = getattr(self.ava_app, 'current_session', None) if self.ava_app else None
        if current_session_val:
            session_name = current_session_val

        base_title = f"AvA [{project_name}] - Session: {session_name}"

        if self.ava_app and hasattr(self.ava_app.llm_client, 'get_role_assignments'):
            assignments = self.ava_app.llm_client.get_role_assignments()

            def get_short_model_name(role_value_str_key):
                model_key = assignments.get(role_value_str_key)
                if not model_key or not hasattr(self.ava_app.llm_client, 'models'):
                    return 'N/A'
                mc = self.ava_app.llm_client.models.get(model_key)
                return mc.model.split('/')[-1][:6] if mc else 'N/A'

            p_short = get_short_model_name(LLMRole.PLANNER.value)
            c_short = get_short_model_name(LLMRole.CODER.value)
            a_short = get_short_model_name(LLMRole.ASSEMBLER.value)
            r_short = get_short_model_name(LLMRole.REVIEWER.value)

            base_title += f" (P:{p_short} C:{c_short} A:{a_short} R:{r_short})"

        self.setWindowTitle(base_title)