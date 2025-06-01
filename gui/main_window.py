# gui/main_window.py - Professional Main Window

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLineEdit, QLabel, QFrame, QScrollArea,
    QGroupBox, QSlider, QComboBox, QProgressBar, QListWidget, QListWidgetItem,
    QSplitter, QTabWidget, QTreeWidget, QTreeWidgetItem, QCheckBox, QSpinBox
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer, QPropertyAnimation, QEasingCurve, QRect
from PySide6.QtGui import QFont, QPalette, QColor, QPixmap, QIcon, QPainter, QLinearGradient

from gui.components import ModernButton, ModernPanel, StatusIndicator
from gui.panels import LLMConfigPanel, ProjectPanel, RAGPanel, ChatActionsPanel, ChatInterface


class AvAMainWindow(QMainWindow):
    """Professional AvA Main Window matching the original design"""

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

    def _apply_professional_theme(self):
        """Apply the professional dark theme"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                color: #cccccc;
            }
            * {
                font-family: "Segoe UI", "Roboto", sans-serif;
            }
        """)

    def _init_ui(self):
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # Left sidebar with all panels
        left_sidebar = QWidget()
        left_sidebar.setFixedWidth(280)
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(12)

        # Add all panels
        self.project_panel = ProjectPanel()
        self.llm_panel = LLMConfigPanel()
        self.rag_panel = RAGPanel()
        self.actions_panel = ChatActionsPanel()

        sidebar_layout.addWidget(self.project_panel)
        sidebar_layout.addWidget(self.llm_panel)
        sidebar_layout.addWidget(self.rag_panel)
        sidebar_layout.addWidget(self.actions_panel)
        sidebar_layout.addStretch()

        left_sidebar.setLayout(sidebar_layout)

        # Right side - Chat interface (minimal like original)
        right_side = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Chat interface at bottom
        self.chat_interface = ChatInterface()

        right_layout.addStretch()  # Push chat to bottom
        right_layout.addWidget(self.chat_interface)

        right_side.setLayout(right_layout)

        # Add to main layout
        main_layout.addWidget(left_sidebar)
        main_layout.addWidget(right_side, 1)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Connect signals
        self._connect_signals()

    def _connect_signals(self):
        """Connect all panel signals"""
        self.chat_interface.message_sent.connect(self._handle_chat_message)
        self.actions_panel.action_triggered.connect(self._handle_action)

    @Slot(str)
    def _handle_chat_message(self, message):
        """Handle chat message from user"""
        print(f"Chat message: {message}")
        self.workflow_requested.emit(message)

        # Update UI to show processing
        self.chat_interface.status_indicator.update_status("working")
        self.chat_interface.status_text.setText("Processing...")

    @Slot(str)
    def _handle_action(self, action):
        """Handle action button clicks"""
        print(f"Action triggered: {action}")
        # TODO: Implement action handlers

    # Signal handlers for core application events
    @Slot(str, dict)
    def on_workflow_started(self, prompt, metadata):
        """Called when workflow starts"""
        self.chat_interface.status_indicator.update_status("working")
        self.chat_interface.status_text.setText("Generating code...")

    @Slot(object)  # WorkflowResult
    def on_workflow_completed(self, result):
        """Called when workflow completes"""
        if result.success:
            self.chat_interface.status_indicator.update_status("success")
            self.chat_interface.status_text.setText(f"Generated {len(result.files_generated)} files")
        else:
            self.chat_interface.status_indicator.update_status("error")
            self.chat_interface.status_text.setText("Workflow failed")

    @Slot(str, str, dict)
    def on_error_occurred(self, component, message, context):
        """Called when an error occurs"""
        self.chat_interface.status_indicator.update_status("error")
        self.chat_interface.status_text.setText(f"Error: {message}")


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = AvAMainWindow()
    window.show()
    sys.exit(app.exec())