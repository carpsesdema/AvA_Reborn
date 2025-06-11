# gui/main_window.py - Final Version with RAG status connection

from PySide6.QtCore import Signal, Slot, QTimer
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QMessageBox
)

from gui.chat_interface import ChatInterface
from gui.components import Colors
from gui.enhanced_sidebar import AvALeftSidebar
from gui.model_config_dialog import ModelConfigurationDialog


class AvAMainWindow(QMainWindow):
    """
    The main window of the AvA application, coordinating the UI components.
    """
    new_project_requested = Signal()
    load_project_requested = Signal()
    workflow_requested_with_context = Signal(str, list)
    workflow_requested = Signal(str)

    def __init__(self, ava_app=None):
        super().__init__()
        self.ava_app = ava_app

        self.setWindowTitle("AvA - AI Development Assistant")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        # UI must be initialized first
        self._init_ui()
        self._apply_theme()

        # Connect signals BEFORE showing the window or running async tasks
        self._connect_signals()

        if self.ava_app:
            self._connect_ava_signals()

    def _apply_theme(self):
        self.setStyleSheet(f"QMainWindow {{ background: {Colors.PRIMARY_BG}; color: {Colors.TEXT_PRIMARY}; }}")

    def _init_ui(self):
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.sidebar = AvALeftSidebar()
        self.chat_interface = ChatInterface()

        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.chat_interface, 1)
        self.setCentralWidget(central_widget)

    def _connect_signals(self):
        self.chat_interface.message_sent.connect(self.handle_user_message)
        self.sidebar.action_triggered.connect(self._handle_sidebar_action)
        self.sidebar.new_project_requested.connect(self.new_project_requested.emit)
        self.sidebar.load_project_requested.connect(self.load_project_requested.emit)
        self.sidebar.model_config_requested.connect(self._open_model_config_dialog)
        self.sidebar.scan_directory_requested.connect(self._handle_rag_scan_request)

    def _handle_sidebar_action(self, action: str):
        """Handle sidebar actions - delegate to ava_app if available"""
        if self.ava_app:
            self.ava_app._handle_sidebar_action(action)

    def _connect_ava_signals(self):
        if not self.ava_app:
            return

        if self.ava_app.workflow_engine:
            self.ava_app.workflow_engine.workflow_started.connect(self.on_workflow_started)
            self.ava_app.workflow_engine.workflow_completed.connect(self.on_workflow_completed)

        # --- FINAL FIX: THIS WHOLE BLOCK IS THE PROBLEM ---
        # The sidebar itself doesn't have an `update_rag_status` method.
        # This was the source of the crash. Removing it for good.
        # Any RAG status updates should be connected directly to the relevant panel if needed.
        # if hasattr(self.ava_app, 'rag_status_changed'):
        #     self.ava_app.rag_status_changed.connect(self.sidebar.update_rag_status)

    def handle_user_message(self, message: str):
        """
        Handle user message from chat interface.
        """
        self.workflow_requested_with_context.emit(message, [])

    def _open_model_config_dialog(self):
        """
        Open model configuration dialog.
        """
        if self.ava_app and self.ava_app.llm_client:
            dialog = ModelConfigurationDialog(self.ava_app.llm_client, self)
            if dialog.exec():
                # After configuration, tell the app to update the status display
                self.ava_app.update_model_status()
        else:
            QMessageBox.warning(self, "Models Not Ready", "The LLM client has not been initialized yet.")

    def _handle_rag_scan_request(self):
        """Handle RAG directory scan request"""
        if self.ava_app and self.ava_app.rag_manager:
            self.ava_app.rag_manager.scan_directory_dialog(self)

    def update_project_display(self, project_name: str):
        """Update the project display in sidebar"""
        self.sidebar.update_project_display(project_name)

    @Slot(dict)
    def update_model_status_display(self, assignments: dict):
        """Update the model status display in the sidebar."""
        self.sidebar.update_model_status_display(assignments)

    @Slot(str, str)
    def on_workflow_started(self, workflow_type: str, project_name: str):
        """Handle workflow started signal"""
        self.chat_interface.add_workflow_status(f"🚀 Started {workflow_type} workflow for '{project_name}'...")

    @Slot(dict)
    def on_workflow_completed(self, result: dict):
        """Handle workflow completion"""
        success = result.get('success', True)

        if success:
            project_dir = result.get('project_directory', 'your workspace.')
            message = f"✅ Workflow completed successfully! Your project is ready at:\n`{project_dir}`"
            self.chat_interface.add_assistant_response(message)
        else:
            error = result.get('error', 'Unknown error')
            self.chat_interface.add_assistant_response(f"❌ Workflow failed: {error}")