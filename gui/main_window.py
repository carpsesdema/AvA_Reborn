# gui/main_window.py - Final Version with RAG status connection

from PySide6.QtCore import Signal, Slot
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
    This class is now primarily a "View" and delegates logic to AvAApplication.
    """
    # Signals are now handled directly by the AvAApplication controller for a cleaner design.
    # No need for this window to re-emit signals.

    def __init__(self):
        super().__init__()
        self.ava_app = None  # Reference to the main application logic

        self.setWindowTitle("AvA - AI Development Assistant")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        self._init_ui()
        self._apply_theme()
        # No signal connections needed here anymore, they are managed in AvAApplication.

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

    @Slot()
    def _open_model_config_dialog(self):
        """Opens the model configuration dialog. Called by AvAApplication."""
        if self.ava_app and self.ava_app.llm_client:
            dialog = ModelConfigurationDialog(self.ava_app.llm_client, self)
            if dialog.exec():
                self.ava_app.update_model_status()
        else:
            QMessageBox.warning(self, "Models Not Ready", "The LLM client has not been initialized yet.")

    @Slot(str)
    def update_project_display(self, project_name: str):
        """Update the project display in the sidebar."""
        self.sidebar.update_project_display(project_name)

    @Slot(dict)
    def update_model_status_display(self, assignments: dict):
        """Update the model status display in the sidebar."""
        self.sidebar.update_model_status_display(assignments)

    @Slot(str, str)
    def on_workflow_started(self, workflow_type: str, project_name: str):
        """Handle workflow started signal."""
        self.chat_interface.add_workflow_status(f"🚀 Started {workflow_type} workflow for '{project_name}'...")

    @Slot(dict)
    def on_workflow_completed(self, result: dict):
        """Handle workflow completion."""
        success = result.get('success', True)

        if success:
            project_dir = result.get('project_directory', 'your workspace.')
            message = f"✅ Workflow completed successfully! Your project is ready at:\n`{project_dir}`"
            self.chat_interface.add_assistant_response(message)
        else:
            error = result.get('error', 'An unknown error occurred.')
            self.chat_interface.add_assistant_response(f"❌ Workflow failed: {error}")