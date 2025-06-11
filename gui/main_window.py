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

        # --- MODIFIED: Connect signals BEFORE showing the window or running async tasks ---
        self._connect_signals()

        if self.ava_app:
            self._connect_ava_signals()
            # The timer is fine here, as it will fire after the event loop starts
            QTimer.singleShot(100, self.update_all_status_displays)

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
        self.sidebar.new_project_requested.connect(self.new_project_requested.emit)
        self.sidebar.load_project_requested.connect(self.load_project_requested.emit)
        self.sidebar.model_config_requested.connect(self._open_model_config_dialog)
        self.sidebar.action_triggered.connect(self._handle_sidebar_action)
        self.sidebar.scan_directory_requested.connect(self._handle_rag_scan_request)

    def _connect_ava_signals(self):
        if not self.ava_app: return

        if self.ava_app.workflow_engine:
            self.ava_app.workflow_engine.workflow_started.connect(self.on_workflow_started)
            self.ava_app.workflow_engine.workflow_progress.connect(self.on_workflow_progress)
            self.ava_app.workflow_engine.workflow_completed.connect(self.on_workflow_completed)

        self.ava_app.error_occurred.connect(self.on_app_error_occurred)
        self.ava_app.project_loaded.connect(self.update_project_display)

    @Slot(str)
    def handle_user_message(self, message: str):
        self.workflow_requested_with_context.emit(message, self.chat_interface.conversation_history)

    @Slot()
    def _open_model_config_dialog(self):
        if not (self.ava_app and self.ava_app.llm_client):
            QMessageBox.warning(self, "Unavailable", "LLM Client is not available.")
            return
        dialog = ModelConfigurationDialog(self.ava_app.llm_client, self)
        dialog.configuration_applied.connect(self._on_model_configuration_applied)
        dialog.exec()

    @Slot()
    def _handle_rag_scan_request(self):
        if self.ava_app and self.ava_app.rag_manager:
            self.ava_app.rag_manager.scan_directory_dialog(self)
        else:
            QMessageBox.warning(self, "RAG Unavailable", "The RAG (Knowledge Base) system is not available.")

    @Slot(str)
    def _handle_sidebar_action(self, action: str):
        if self.ava_app:
            self.ava_app._handle_sidebar_action(action)

    @Slot()
    def update_all_status_displays(self):
        self._update_model_config_display()
        if self.ava_app and self.ava_app.rag_manager:
            # RAG status is now handled internally by the RAGManager's signals to the app,
            # which no longer needs to propagate to the main window's sidebar.
            pass

    @Slot(dict)
    def _on_model_configuration_applied(self, config_summary: dict):
        self._update_model_config_display()
        self.chat_interface.add_assistant_response("✅ Model configuration has been updated!")

    def _update_model_config_display(self):
        if not (self.ava_app and self.ava_app.llm_client): return
        assignments = self.ava_app.llm_client.get_role_assignments()
        display_summary = {}
        for role_key, model_key in assignments.items():
            model_config = self.ava_app.llm_client.models.get(model_key)
            if model_config:
                display_summary[role_key] = f"{model_config.provider}/{model_config.model}"
            else:
                display_summary[role_key] = "Not Set"
        self.sidebar.update_model_status_display(display_summary)

    @Slot(str)
    def update_project_display(self, project_name: str):
        self.sidebar.update_project_display(project_name)

    @Slot(str, str)
    def on_workflow_started(self, workflow_type: str, description: str = ""):
        self.chat_interface.add_workflow_status(f"Starting {workflow_type}: {description}")

    @Slot(str, str)
    def on_workflow_progress(self, stage: str, description: str):
        self.chat_interface.add_workflow_status(f"Progress: {description}")

    @Slot(dict)
    def on_workflow_completed(self, result: dict):
        if result.get("success", False):
            msg = f"✅ **Workflow Complete!**\n\nProject '{result.get('project_name', 'N/A')}' is ready.\n\nCheck the Code Viewer to see the {result.get('num_files', 0)} generated files and use the 'Run Project' button to test it!"
            self.chat_interface.add_assistant_response(msg)
        else:
            self.on_app_error_occurred("Workflow", result.get("error", "An unknown error occurred."))

    @Slot(str, str)
    def on_app_error_occurred(self, component: str, error_message: str):
        error_text = f"❌ **Error in {component}**\n\n{error_message}"
        self.chat_interface.add_assistant_response(error_text)