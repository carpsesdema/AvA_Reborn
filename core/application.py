# core/application.py

import asyncio
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from PySide6.QtCore import QObject, Signal, QTimer, QProcess, Slot
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QInputDialog

from core.config import ConfigManager
from core.enhanced_workflow_engine import EnhancedWorkflowEngine
from core.llm_client import EnhancedLLMClient, LLMRole
from core.project_state_manager import ProjectStateManager
from gui.code_viewer import CodeViewerWindow
from gui.main_window import AvAMainWindow
from gui.terminals import StreamingTerminal
from gui.workflow_monitor_window import WorkflowMonitorWindow
from utils.logger import get_logger

try:
    from core.rag_manager import RAGManager

    RAG_MANAGER_AVAILABLE = True
except ImportError:
    RAG_MANAGER_AVAILABLE = False
    RAGManager = None


class AvAApplication(QObject):
    """Enhanced AvA Application Controller with RAG integration"""

    # ALL REQUIRED SIGNALS
    fully_initialized_signal = Signal()
    workflow_started = Signal(str, str)
    workflow_completed = Signal(dict)
    error_occurred = Signal(str, str)
    rag_status_changed = Signal(str, str)
    project_loaded = Signal(str)
    model_status_updated = Signal(dict)

    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)
        self.config = ConfigManager()

        # Core components
        self.main_window: Optional[AvAMainWindow] = None
        self.streaming_terminal: Optional[StreamingTerminal] = None
        self.code_viewer: Optional[CodeViewerWindow] = None
        self.workflow_monitor: Optional[WorkflowMonitorWindow] = None
        self.workflow_engine: Optional[EnhancedWorkflowEngine] = None
        self.llm_client: Optional[EnhancedLLMClient] = None
        self.rag_manager: Optional[RAGManager] = None

        # Project state
        self.current_project = "Default Project"
        self.workspace_dir = Path(self.config.app_config.workspace_path)
        self.current_project_path = self.workspace_dir

        self.project_state_manager = ProjectStateManager(self.workspace_dir)

        self.logger.info("🚀 AvA Application Controller initialized")

    async def initialize(self):
        """Initialize all components asynchronously"""
        try:
            self.logger.info("🚀 Initializing AvA Application...")

            # Initialize core services first
            self._initialize_core_services()

            # Initialize GUI components
            self._initialize_gui_components()

            # Initialize workflow engine with references
            self._initialize_workflow_engine()

            # Connect all components
            self._connect_components()

            # Show main window
            self.main_window.show()

            # Start async initialization
            await self._initialize_async_components()

            # After all initialization, update the UI with the final status
            self.update_model_status()

            # Emit fully initialized signal
            self.fully_initialized_signal.emit()
            self.logger.info("✅ AvA Application initialized successfully")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize components: {e}", exc_info=True)
            self.error_occurred.emit("initialization", str(e))

    def _initialize_core_services(self):
        """Initialize core services"""
        self.logger.info("Initializing core services...")
        self.llm_client = EnhancedLLMClient()

    def _initialize_gui_components(self):
        """Initialize GUI windows and dialogs"""
        self.main_window = AvAMainWindow(ava_app=self)
        self.streaming_terminal = StreamingTerminal()
        self.code_viewer = CodeViewerWindow()
        self.workflow_monitor = WorkflowMonitorWindow()

    def _initialize_workflow_engine(self):
        """Initialize workflow engine"""
        self.workflow_engine = EnhancedWorkflowEngine(self.llm_client, self.streaming_terminal, self.code_viewer,
                                                      self.rag_manager)

    async def _initialize_async_components(self):
        """Initialize async components"""
        self.logger.info("Initializing async components...")
        if RAG_MANAGER_AVAILABLE:
            try:
                self.rag_manager = RAGManager()
                await self.rag_manager.async_initialize()
                self.rag_manager.status_changed.connect(self._handle_rag_status_changed)
            except Exception as e:
                self.logger.error(f"RAG manager initialization failed: {e}", exc_info=True)
                self.rag_manager = None

    def _connect_components(self):
        """Connect all component signals"""
        self.logger.info("Connecting components...")
        if self.main_window:
            self.main_window.workflow_requested_with_context.connect(self._schedule_workflow_request)
            self.main_window.new_project_requested.connect(self.create_new_project_dialog)
            self.main_window.load_project_requested.connect(self.load_existing_project_dialog)
            # Connect the new signal to the main window's handler
            self.model_status_updated.connect(self.main_window.update_model_status_display)

        if self.workflow_engine:
            self.workflow_engine.workflow_started.connect(self._on_workflow_started)
            self.workflow_engine.workflow_completed.connect(self._on_workflow_completed)
            self.workflow_engine.file_generated.connect(self._on_file_generated)
            self.workflow_engine.project_loaded.connect(self._on_project_loaded)
            if self.workflow_monitor:
                self.workflow_engine.node_status_changed.connect(self.workflow_monitor.update_agent_status)
                self.workflow_engine.data_flow_started.connect(self.workflow_monitor.activate_connection)
                self.workflow_engine.data_flow_completed.connect(self.workflow_monitor.deactivate_connection)
                self.workflow_engine.workflow_reset.connect(self.workflow_monitor.refresh_workflow)

        if self.code_viewer:
            self.code_viewer.run_project_requested.connect(self.run_current_project)
            self.code_viewer.terminal.command_completed.connect(self._on_command_completed)

        self.error_occurred.connect(self._on_error_occurred)
        self.logger.info("✅ Components connected.")

    def update_model_status(self):
        """Gets the model status and emits a signal to update the UI."""
        if self.llm_client:
            assignments = self.llm_client.get_role_assignments()
            self.model_status_updated.emit(assignments)

    def _handle_sidebar_action(self, action: str):
        """Handle sidebar actions"""
        self.logger.info(f"Sidebar action: {action}")
        if action == "view_log":
            if self.streaming_terminal:
                self.streaming_terminal.show()
                self.streaming_terminal.raise_()
                self.streaming_terminal.activateWindow()
        elif action == "open_workflow_monitor":
            if self.workflow_monitor:
                self.workflow_monitor.show()
                self.workflow_monitor.raise_()
                self.workflow_monitor.activateWindow()
        elif action == "open_code_viewer":
            if self.code_viewer:
                self.code_viewer.show()
                self.code_viewer.raise_()
                self.code_viewer.activateWindow()
        elif action == "save_session":
            self.save_session()
        elif action == "load_session":
            self.load_session()
        elif action == "new_session":
            self.new_session()

    def get_status(self) -> Dict[str, Any]:
        """
        Get application status for main.py.
        """
        status = {
            'llm_models': [],
            'rag': {'ready': False, 'status_text': 'RAG: Not Available', 'available': False}
        }

        if self.llm_client:
            try:
                # Use the helper method from llm_client to get the role assignments
                assignments = self.llm_client.get_role_assignments()
                # Format them nicely for the status check
                status['llm_models'] = [
                    f"{role.capitalize()}: {model_name}" for role, model_name in assignments.items()
                ]
            except Exception as e:
                self.logger.error(f"Error getting LLM status: {e}")

        if self.rag_manager:
            try:
                status['rag'] = {
                    'ready': self.rag_manager.is_ready,
                    'status_text': self.rag_manager.current_status,
                    'available': True
                }
            except Exception as e:
                self.logger.error(f"Error getting RAG status: {e}")

        return status

    def _handle_rag_status_changed(self, status_text: str, color_key: str):
        """Handle RAG status changes"""
        self.rag_status_changed.emit(status_text, color_key)

    @Slot(str, list)
    def _schedule_workflow_request(self, user_input: str, context_files: list):
        """
        --- FINAL FIX: PROPERLY SCHEDULE THE ASYNC WORKFLOW ---
        This function is a Qt Slot, which is a regular (synchronous) function.
        To run an `async def` function from here, we must schedule it on the
        running asyncio event loop using `asyncio.create_task`.
        """
        self.logger.info("Scheduling async workflow request...")
        asyncio.create_task(self._handle_enhanced_workflow_request(user_input, context_files))

    async def _handle_enhanced_workflow_request(self, user_input: str, context_files: list):
        """Handle workflow request with context and enhanced engine"""
        if not self.workflow_engine:
            self.logger.error("Workflow engine not initialized")
            return

        try:
            # Set the project state before running the workflow
            if not self.project_state_manager or self.project_state_manager.project_root != self.current_project_path:
                self.project_state_manager = ProjectStateManager(self.current_project_path)

            self.workflow_engine.set_project_state(self.project_state_manager)

            # Let the workflow engine handle execution
            await self.workflow_engine.execute_enhanced_workflow(
                user_prompt=user_input,
                conversation_context=context_files  # Assuming context_files is the conversation history
            )

            self.logger.info(f"Workflow initiated for prompt: {user_input[:50]}...")

        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}", exc_info=True)
            if self.main_window:
                QMessageBox.critical(self.main_window, "Workflow Error", f"Workflow failed: {str(e)}")

    def _on_workflow_started(self, workflow_type: str, project_name: str):
        """Handle workflow start event"""
        self.logger.info(f"Workflow started: {workflow_type} for project {project_name}")
        self.workflow_started.emit(workflow_type, project_name)

    def _on_workflow_completed(self, result: dict):
        """Handle workflow completion"""
        self.logger.info(f"Workflow completed: {result}")
        self.workflow_completed.emit(result)
        # If the workflow was successful, auto-load the project into the code viewer
        if result.get("success") and result.get("project_directory"):
            self.load_project(Path(result["project_directory"]))

    def _on_file_generated(self, file_path: str):
        """Handle file generation event"""
        self.logger.info(f"File generated: {file_path}")
        if self.code_viewer:
            self.code_viewer.auto_open_file(file_path)

    def _on_project_loaded(self, project_path: str):
        """Handle project load event"""
        self.logger.info(f"Project loaded: {project_path}")
        self.project_loaded.emit(project_path)

    def _on_command_completed(self, exit_code: int):
        """Handle command completion from code viewer's terminal"""
        if exit_code != 0:
            self.logger.error(f"Command failed with exit code {exit_code}")
        else:
            self.logger.info(f"Command completed successfully")

    def _on_error_occurred(self, context: str, error_message: str):
        """Handle application errors"""
        self.logger.error(f"Application error in {context}: {error_message}")
        if self.main_window:
            QMessageBox.critical(self.main_window, f"Error in {context}", error_message)

    def new_session(self):
        """Start a new session"""
        if self.main_window:
            self.main_window.chat_interface.clear_chat()
            self.main_window.chat_interface._add_welcome_message()
            self.current_project = "Default Project"
            self.current_project_path = self.workspace_dir
            self.main_window.update_project_display(self.current_project)
            self.project_state_manager = ProjectStateManager(self.workspace_dir)
        self.logger.info("New session started.")

    def save_session(self):
        """Save current session state"""
        self.logger.info("Session save requested - not yet implemented")

    def load_session(self):
        """Load a saved session"""
        self.logger.info("Session load requested - not yet implemented")

    def create_new_project_dialog(self):
        """Create a new project with dialog"""
        if not self.main_window:
            return

        project_name, ok = QInputDialog.getText(
            self.main_window, "New Project", "Enter project name:"
        )
        if not ok or not project_name.strip():
            return

        project_name = project_name.strip()

        # Start the workflow to create the new project
        asyncio.create_task(self._handle_enhanced_workflow_request(
            f"Create a new project named '{project_name}'.", []
        ))

    def load_existing_project_dialog(self):
        """Load an existing project with dialog"""
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly)
        dialog.setWindowTitle("Select Existing Project Directory")
        if dialog.exec():
            selected_dirs = dialog.selectedFiles()
            if selected_dirs:
                project_path = Path(selected_dirs[0])
                asyncio.create_task(self.workflow_engine.analyze_existing_project(str(project_path)))
                self.load_project(project_path)

    def load_project(self, project_path: Path):
        """Load a specific project"""
        self.current_project_path = project_path
        self.current_project = self.current_project_path.name
        self.project_state_manager = ProjectStateManager(self.current_project_path)

        if self.main_window:
            self.main_window.update_project_display(self.current_project)
        if self.code_viewer:
            self.code_viewer.load_project(str(project_path))

        self._on_project_loaded(str(project_path))

    def run_current_project(self):
        """Run the current project"""
        self.logger.info(f"Running project: {self.current_project}")

    def shutdown(self):
        """Clean shutdown of the application"""
        self.logger.info("Shutting down AvA application...")
        if self.rag_manager:
            pass
        QApplication.quit()