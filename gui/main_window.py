# gui/main_window.py - Replace your existing file with this

from PySide6.QtCore import Signal, Slot, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QFrame, QLabel, QTextEdit, QScrollArea, QComboBox
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt

from gui.components import ModernButton, StatusIndicator


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
        else:
            color = "#3fb950"
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

        status_layout = QHBoxLayout()
        status_layout.setContentsMargins(0, 8, 0, 0)

        self.status_indicator = StatusIndicator("ready")
        self.status_text = QLabel("Ready • Gemini 2.5 Pro")
        self.status_text.setStyleSheet("color: #888; font-size: 11px;")

        status_layout.addWidget(self.status_indicator)
        status_layout.addWidget(self.status_text)
        status_layout.addStretch()

        layout.addLayout(input_layout)
        layout.addLayout(status_layout)
        self.setLayout(layout)

    def _send_message(self):
        message = self.input_field.text().strip()
        if not message:
            return

        self.chat_display.add_user_message(message)
        self.input_field.clear()

        # Check if workflow request
        workflow_keywords = ['build', 'create', 'make', 'generate', 'develop', 'code']
        is_workflow_request = any(keyword in message.lower() for keyword in workflow_keywords)

        if is_workflow_request:
            self.chat_display.add_assistant_message("I'll help you build that! Opening the development workflow...")
            self.status_text.setText("🚀 Starting development workflow...")
            self.status_indicator.update_status("working")
            self.message_sent.emit(message)
        else:
            self._handle_chat_response(message)

    def _handle_chat_response(self, message: str):
        message_lower = message.lower()

        if any(greeting in message_lower for greeting in ['hi', 'hello', 'hey']):
            response = "Hello! I'm here to help you with development projects. You can ask me to build applications, explain code concepts, or help with programming questions. What would you like to work on?"
        elif any(word in message_lower for word in ['help', 'what', 'how']):
            response = "I can help you with:\n• Building complete applications from scratch\n• Explaining code concepts\n• Debugging and improving existing code\n• Setting up project structures\n\nJust describe what you'd like to build or ask any programming question!"
        elif 'thanks' in message_lower or 'thank you' in message_lower:
            response = "You're welcome! Is there anything else I can help you build or explain?"
        else:
            response = f"I understand you said '{message}'. I'm designed to help with software development. If you'd like me to build something, just describe your project idea and I'll get started!"

        self.chat_display.add_assistant_message(response)

    def update_status(self, text: str, status: str = "ready"):
        self.status_text.setText(text)
        self.status_indicator.update_status(status)


class AvALeftSidebar(QWidget):
    model_changed = Signal(str, str)
    temperature_changed = Signal(float)
    action_triggered = Signal(str)

    def __init__(self):
        super().__init__()
        self.setFixedWidth(280)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Simple clean panels
        self._add_projects_panel(layout)
        self._add_llm_panel(layout)
        self._add_actions_panel(layout)

        layout.addStretch()
        self.setLayout(layout)

        self.setStyleSheet("""
            AvALeftSidebar {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e1e1e, stop:1 #252526);
                border-right: 2px solid #00d7ff;
            }
        """)

    def _add_projects_panel(self, layout):
        panel = self._create_panel("Projects & Sessions")

        projects_label = QLabel("Projects:")
        projects_label.setStyleSheet("color: #cccccc; font-weight: bold;")
        panel.addWidget(projects_label)

        project_item = QFrame()
        project_item.setStyleSheet("""
            QFrame {
                background: #0078d4;
                border-radius: 4px;
                padding: 8px;
                margin: 2px 0;
            }
        """)
        project_layout = QVBoxLayout()
        project_layout.setContentsMargins(4, 4, 4, 4)
        project_name = QLabel("Default Project")
        project_name.setStyleSheet("color: white; font-weight: bold;")
        project_layout.addWidget(project_name)
        project_item.setLayout(project_layout)
        panel.addWidget(project_item)

        new_project_btn = ModernButton("📁 New Project", button_type="primary")
        panel.addWidget(new_project_btn)

        layout.addWidget(panel)

    def _add_llm_panel(self, layout):
        panel = self._create_panel("LLM Configuration")

        # Chat LLM
        chat_layout = QHBoxLayout()
        chat_label = QLabel("Chat LLM:")
        chat_label.setStyleSheet("color: #cccccc;")

        self.chat_combo = QComboBox()
        self.chat_combo.addItems([
            "Gemini: gemini-2.5-pro",
            "OpenAI: gpt-4o",
            "Anthropic: claude-3.5-sonnet"
        ])
        self.chat_combo.setStyleSheet("""
            QComboBox {
                background: #1e1e1e;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 4px 8px;
                color: #cccccc;
                font-size: 10px;
            }
            QComboBox:hover {
                border-color: #00d7ff;
            }
        """)

        chat_status = StatusIndicator("ready")
        chat_layout.addWidget(chat_label)
        chat_layout.addWidget(self.chat_combo, 1)
        chat_layout.addWidget(chat_status)
        panel.addLayout(chat_layout)

        # Code LLM
        code_label = QLabel("Code LLM:")
        code_label.setStyleSheet("color: #cccccc; margin-top: 8px;")
        panel.addWidget(code_label)

        code_layout = QHBoxLayout()
        self.code_combo = QComboBox()
        self.code_combo.addItems([
            "Ollama: qwen2.5-coder",
            "OpenAI: gpt-4o",
            "Anthropic: claude-3.5-sonnet"
        ])
        self.code_combo.setStyleSheet(self.chat_combo.styleSheet())

        code_status = StatusIndicator("ready")
        code_layout.addWidget(self.code_combo, 1)
        code_layout.addWidget(code_status)
        panel.addLayout(code_layout)

        layout.addWidget(panel)

    def _add_actions_panel(self, layout):
        panel = self._create_panel("Actions")

        buttons = [
            ("💬 New Session", "new_session"),
            ("📟 Open Terminal", "open_terminal"),
            ("📄 Open Code Viewer", "open_code_viewer"),
            ("🔨 Force Generation", "force_gen"),
        ]

        for text, action in buttons:
            btn = ModernButton(text, button_type="secondary")
            btn.clicked.connect(lambda checked, a=action: self.action_triggered.emit(a))
            panel.addWidget(btn)

        layout.addWidget(panel)

    def _create_panel(self, title: str) -> QVBoxLayout:
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2a2a2e, stop:1 #252526);
                border: 2px solid #00d7ff;
                border-radius: 8px;
                margin: 2px;
            }
        """)

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 8, 12, 12)
        layout.setSpacing(6)

        title_label = QLabel(title)
        title_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #00d7ff; margin-bottom: 6px; background: transparent; border: none;")
        layout.addWidget(title_label)

        frame.setLayout(layout)
        return layout

    def get_current_models(self):
        return {
            "chat_model": self.chat_combo.currentText(),
            "code_model": self.code_combo.currentText(),
            "temperature": 0.7
        }


class AvAMainWindow(QMainWindow):
    workflow_requested = Signal(str)

    def __init__(self, ava_app=None, config=None):
        super().__init__()
        self.ava_app = ava_app
        self.config = config

        self.setWindowTitle("AvA - AI Development Assistant")
        self.setGeometry(100, 100, 1400, 900)
        self._apply_theme()
        self._init_ui()
        self._connect_signals()

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
        self.sidebar.action_triggered.connect(self._handle_action)

    def _handle_action(self, action):
        if action == "open_terminal":
            if self.ava_app:
                self.ava_app._open_terminal()
        elif action == "open_code_viewer":
            if self.ava_app:
                self.ava_app._open_code_viewer()
        elif action == "new_session":
            self.chat_interface.chat_display.clear()
            self.chat_interface.chat_display.add_assistant_message("New session started! How can I help you today?")

    def on_workflow_started(self, prompt, metadata=None):
        self.chat_interface.update_status("🚀 Building your project... Check terminal for progress", "working")

    def on_workflow_completed(self, result):
        if hasattr(result, 'success') and result.success:
            file_count = len(result.files_generated) if hasattr(result, 'files_generated') else 0
            self.chat_interface.update_status(f"✅ Generated {file_count} files successfully", "success")
            self.chat_interface.chat_display.add_assistant_message(
                f"Perfect! I've generated {file_count} files for your project. Check the code viewer to see the results!")
        else:
            self.chat_interface.update_status("❌ Generation failed", "error")
            self.chat_interface.chat_display.add_assistant_message(
                "Sorry, something went wrong during generation. Check the terminal for details.")

    def on_error_occurred(self, component, message, context=None):
        self.chat_interface.update_status(f"❌ Error: {message}", "error")
        self.chat_interface.chat_display.add_assistant_message(f"I encountered an error: {message}")