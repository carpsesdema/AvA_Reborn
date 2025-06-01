# windows/terminal_window.py - LLM terminal ONLY

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from datetime import datetime


class LLMTerminalWindow(QMainWindow):
    """
    SINGLE RESPONSIBILITY: Display streaming LLM workflow logs (like Aider)
    Does NOT handle workflow logic, LLM calls, or other functionality
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AvA - LLM Terminal (Streaming Workflow)")
        self.setGeometry(600, 100, 800, 600)
        self._init_ui()
        self._apply_theme()

    def _init_ui(self):
        """Initialize terminal UI"""
        central_widget = QWidget()
        layout = QVBoxLayout()

        # Terminal display
        self.terminal_text = QTextEdit()
        self.terminal_text.setReadOnly(True)
        self.terminal_text.setStyleSheet("""
            QTextEdit {
                background: #0d1117;
                color: #58a6ff;
                border: 1px solid #30363d;
                font-family: "JetBrains Mono", "Consolas", monospace;
                font-size: 12px;
                padding: 10px;
                line-height: 1.4;
            }
        """)

        # Controls
        controls = QHBoxLayout()

        self.clear_btn = QPushButton("Clear")
        self.save_btn = QPushButton("Save Log")

        button_style = """
            QPushButton {
                background: #2d2d30;
                border: 1px solid #404040;
                border-radius: 4px;
                color: #cccccc;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background: #3e3e42;
            }
        """

        self.clear_btn.setStyleSheet(button_style)
        self.save_btn.setStyleSheet(button_style)

        self.clear_btn.clicked.connect(self.clear_terminal)
        # TODO: Implement save_log functionality

        controls.addWidget(self.clear_btn)
        controls.addWidget(self.save_btn)
        controls.addStretch()

        layout.addWidget(self.terminal_text)
        layout.addLayout(controls)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Initial welcome message
        self._show_welcome()

    def _apply_theme(self):
        """Apply terminal theme"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0d1117;
                color: #c9d1d9;
            }
        """)

    def _show_welcome(self):
        """Show welcome message"""
        self.log("╭─────────────────────────────────────────╮")
        self.log("│        AvA LLM Terminal Ready           │")
        self.log("│     Waiting for workflow requests      │")
        self.log("╰─────────────────────────────────────────╯")

    def log(self, message: str, message_type: str = "info"):
        """
        Add message to terminal with timestamp and color coding
        This is the MAIN responsibility of this class
        """
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Color coding based on message type and content
        if "🚀" in message or "started" in message.lower():
            color = "#3fb950"  # Green
        elif "🧠" in message or "planner" in message.lower():
            color = "#a5a5f0"  # Purple
        elif "⚙️" in message or "assembler" in message.lower():
            color = "#58a6ff"  # Blue
        elif "📄" in message or "file" in message.lower():
            color = "#f0883e"  # Orange
        elif "✅" in message or "completed" in message.lower():
            color = "#3fb950"  # Green
        elif "❌" in message or "error" in message.lower():
            color = "#f85149"  # Red
        else:
            color = "#c9d1d9"  # Default

        # Format message with HTML for color
        formatted_message = f'<span style="color: #6e7681;">[{timestamp}]</span> <span style="color: {color};">{message}</span>'

        # Add to terminal
        self.terminal_text.append(formatted_message)

        # Auto-scroll to bottom
        scrollbar = self.terminal_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

        # Force UI update
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()

    def clear_terminal(self):
        """Clear terminal and show welcome message"""
        self.terminal_text.clear()
        self._show_welcome()