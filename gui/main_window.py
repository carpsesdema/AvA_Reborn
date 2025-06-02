# gui/main_window.py - Updated with Enhanced Sidebar

from PySide6.QtCore import Signal, Slot, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QFrame, QLabel
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt

from gui.panels import AvALeftSidebar
from gui.components import ModernButton, StatusIndicator


class ChatInterface(QWidget):
    """Clean chat interface for the main window"""

    message_sent = Signal(str)

    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(0)

        # REMOVED: All that welcome bullshit taking up space
        # This is a CHAT WINDOW - just chat!

        layout.addStretch()  # Push input to bottom

        # Simple chat input at bottom - like every chat app
        input_layout = QHBoxLayout()
        input_layout.setSpacing(10)

        # Input field
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Ask AvA to build something...")
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

        # Send button - next to input like normal chat
        self.send_btn = ModernButton("Send", button_type="accent")
        self.send_btn.clicked.connect(self._send_message)
        self.send_btn.setMinimumHeight(45)
        self.send_btn.setMinimumWidth(80)

        input_layout.addWidget(self.input_field, 1)  # Input takes most space
        input_layout.addWidget(self.send_btn)  # Send button on right

        # Simple status
        status_layout = QHBoxLayout()
        status_layout.setContentsMargins(0, 8, 0, 0)

        self.status_indicator = StatusIndicator("ready")
        self.status_text = QLabel("Ready")
        self.status_text.setStyleSheet("color: #888; font-size: 11px;")

        status_layout.addWidget(self.status_indicator)
        status_layout.addWidget(self.status_text)
        status_layout.addStretch()

        layout.addLayout(input_layout)
        layout.addLayout(status_layout)

        self.setLayout(layout)

    def _send_message(self):
        """Handle sending message"""
        message = self.input_field.text().strip()
        if message:
            self.message_sent.emit(message)
            self.input_field.clear()
            self.status_text.setText("🚀 Starting workflow...")
            self.status_indicator.update_status("working")

    def update_status(self, text: str, status: str = "ready"):
        """Update status display"""
        self.status_text.setText(text)
        self.status_indicator.update_status(status)


class AvAMainWindow(QMainWindow):
    """Updated AvA Main Window with enhanced sidebar"""

    workflow_requested = Signal(str)

    def __init__(self, ava_app=None, config=None):
        super().__init__()
        self.ava_app = ava_app
        self.config = config

        self.setWindowTitle(
            "AvA: PySide6 Rebuild - [Project: Default Project, Session: Main Chat, LLM: gemini-2.5-pro-prev]")
        self.setGeometry(100, 100, 1400, 900)
        self._apply_professional_theme()
        self._init_ui()
        self._connect_signals()

    def _apply_professional_theme(self):
        """Apply the professional dark theme"""
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e1e1e, stop:1 #252526);
                color: #cccccc;
            }
            * {
                font-family: "Segoe UI", "Roboto", sans-serif;
            }
        """)

    def _init_ui(self):
        """Initialize the UI with sidebar and main area"""
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left sidebar with all controls
        self.sidebar = AvALeftSidebar()

        # Main content area
        main_content = QWidget()
        main_content.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e1e1e, stop:1 #252526);
            }
        """)

        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(20, 20, 20, 20)

        # Chat interface
        self.chat_interface = ChatInterface()
        content_layout.addWidget(self.chat_interface)

        main_content.setLayout(content_layout)

        # Add to main layout
        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(main_content, 1)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def _connect_signals(self):
        """Connect all signals"""
        # Chat interface
        self.chat_interface.message_sent.connect(self._handle_chat_message)

        # Sidebar signals
        self.sidebar.model_changed.connect(self._handle_model_change)
        self.sidebar.temperature_changed.connect(self._handle_temperature_change)
        self.sidebar.action_triggered.connect(self._handle_action)

        # No need for terminal/code buttons in chat - they're in sidebar actions

    @Slot(str)
    def _handle_chat_message(self, message):
        """Handle chat message from user"""
        print(f"Chat message: {message}")
        self.workflow_requested.emit(message)

        # Update status
        self.chat_interface.update_status("🧠 Planning project...", "working")

    @Slot(str, str)
    def _handle_model_change(self, model_type, model_name):
        """Handle model selection changes"""
        print(f"Model changed: {model_type} -> {model_name}")
        # Update status with current models
        models = self.sidebar.get_current_models()
        status_text = f"Ready • {models['chat_model'].split(':')[0]} • {models['code_model'].split(':')[0]}"
        self.chat_interface.update_status(status_text)

    @Slot(float)
    def _handle_temperature_change(self, temperature):
        """Handle temperature changes"""
        print(f"Temperature changed: {temperature}")

    @Slot(str)
    def _handle_action(self, action):
        """Handle action button clicks from sidebar"""
        print(f"Action triggered: {action}")

        if action == "new_session":
            self.chat_interface.update_status("New session started", "ready")
        elif action == "view_log":
            self._open_terminal()
        elif action == "open_terminal":  # ADDED: Handle terminal button
            self._open_terminal()
        elif action == "open_code_viewer":  # ADDED: Handle code viewer button
            self._open_code_viewer()
        elif action == "view_code":
            self._open_code_viewer()
        elif action == "force_gen":
            self.chat_interface.update_status("Force generating...", "working")
        elif action == "check_updates":
            self.chat_interface.update_status("Checking for updates...", "working")

    def _open_terminal(self):
        """Open terminal window"""
        if self.ava_app:
            self.ava_app._open_terminal()

    def _open_code_viewer(self):
        """Open code viewer window"""
        if self.ava_app:
            self.ava_app._open_code_viewer()

    # Event handlers for workflow events
    @Slot(str, dict)
    def on_workflow_started(self, prompt, metadata):
        """Called when workflow starts"""
        self.chat_interface.update_status("🚀 Workflow started • Check terminal for progress", "working")

    @Slot(object)  # WorkflowResult
    def on_workflow_completed(self, result):
        """Called when workflow completes"""
        if hasattr(result, 'success') and result.success:
            file_count = len(result.files_generated) if hasattr(result, 'files_generated') else 0
            self.chat_interface.update_status(f"✅ Generated {file_count} files successfully", "success")
        else:
            self.chat_interface.update_status("❌ Workflow failed • Check terminal for details", "error")

    @Slot(str, str, dict)
    def on_error_occurred(self, component, message, context):
        """Called when an error occurs"""
        self.chat_interface.update_status(f"❌ Error in {component}: {message}", "error")

    def update_window_title(self, project=None, session=None, model=None):
        """Update window title with current context"""
        project = project or "Default Project"
        session = session or "Main Chat"
        model = model or "gemini-2.5-pro-prev"

        self.setWindowTitle(f"AvA: PySide6 Rebuild - [Project: {project}, Session: {session}, LLM: {model}]")

    def get_sidebar_state(self):
        """Get current sidebar configuration"""
        return self.sidebar.get_current_models()


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = AvAMainWindow()
    window.show()
    sys.exit(app.exec())