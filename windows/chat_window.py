# windows/chat_window.py - Chat interface ONLY

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QLabel, QFrame
)
from PySide6.QtCore import Qt, pyqtSignal
from PySide6.QtGui import QFont


class ChatWindow(QMainWindow):
    """
    SINGLE RESPONSIBILITY: Chat interface for user interaction with Planner AI
    Does NOT handle workflow, LLM calls, or other windows
    """

    workflow_requested = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AvA - Chat with Planner AI")
        self.setGeometry(100, 100, 500, 300)
        self._init_ui()
        self._apply_theme()

    def _init_ui(self):
        """Initialize UI components"""
        central_widget = QWidget()
        layout = QVBoxLayout()

        # Title
        title = QLabel("💬 Chat with AvA Planner")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #00d7ff; margin: 10px 0;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Instructions
        instructions = QLabel("Describe your project and AvA will plan and build it:")
        instructions.setStyleSheet("color: #cccccc; margin-bottom: 15px;")
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(instructions)

        # Input area
        input_frame = QFrame()
        input_frame.setStyleSheet("""
            QFrame {
                background: #2d2d30;
                border: 2px solid #0078d4;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        input_layout = QVBoxLayout()

        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText(
            "Example: Build a web scraper for news headlines\n"
            "Example: Create a Discord bot with commands\n"
            "Example: Make a GUI file manager"
        )
        self.input_text.setStyleSheet("""
            QTextEdit {
                background: transparent;
                border: none;
                color: white;
                font-size: 13px;
            }
        """)
        self.input_text.setMaximumHeight(80)

        # Buttons
        button_layout = QHBoxLayout()

        self.plan_btn = QPushButton("🧠 Plan & Build")
        self.plan_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #00d7ff, stop:1 #0078d4);
                border: none;
                border-radius: 6px;
                color: #1e1e1e;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #40e0ff, stop:1 #0078d4);
            }
        """)
        self.plan_btn.clicked.connect(self._start_workflow)

        self.terminal_btn = QPushButton("📊 Open Terminal")
        self.code_btn = QPushButton("📄 Open Code Viewer")

        secondary_style = """
            QPushButton {
                background: #2d2d30;
                border: 1px solid #404040;
                border-radius: 6px;
                color: #cccccc;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background: #3e3e42;
                border-color: #0078d4;
            }
        """

        self.terminal_btn.setStyleSheet(secondary_style)
        self.code_btn.setStyleSheet(secondary_style)

        button_layout.addWidget(self.plan_btn)
        button_layout.addWidget(self.terminal_btn)
        button_layout.addWidget(self.code_btn)

        input_layout.addWidget(self.input_text)
        input_layout.addLayout(button_layout)
        input_frame.setLayout(input_layout)
        layout.addWidget(input_frame)

        # Status
        self.status_label = QLabel("Ready to plan your project...")
        self.status_label.setStyleSheet("color: #888; font-size: 11px; margin-top: 10px;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def _apply_theme(self):
        """Apply dark theme"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                color: #cccccc;
            }
        """)

    def _start_workflow(self):
        """Handle workflow start - ONLY emits signal"""
        prompt = self.input_text.toPlainText().strip()
        if prompt:
            self.workflow_requested.emit(prompt)
            self.status_label.setText("🚀 Workflow started! Check terminal for progress...")
            self.input_text.clear()