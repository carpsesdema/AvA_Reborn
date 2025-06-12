# core/application.py

import asyncio
import logging
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from core.config import ConfigManager
from core.llm_client import EnhancedLLMClient
from core.rag_manager import RAGManager
from core.project_state_manager import ProjectStateManager
from core.enhanced_workflow_engine import HybridWorkflowEngine
from services.rag_sync_service import RagSyncService

# Import GUI windows to be managed by the application
from gui.main_window import AvAMainWindow
from gui.code_viewer import CodeViewerWindow
from gui.workflow_monitor_window import WorkflowMonitorWindow
from gui.terminals import StreamingTerminal, TerminalWindow


class AvAApplication(QObject):
    """
    Main application class for AvA Reborn.
    Manages initialization, component wiring, and workflow execution.
    """
    initialization_complete = Signal()
    model_status_updated = Signal(dict)

    def __init__(self, main_window: AvAMainWindow):
        super().__init__()
        self.main_window = main_window
        self.main_window.ava_app = self  # Give window a reference back to this

        # Core components
        self.config_manager = ConfigManager()
        self.llm_client: EnhancedLLMClient = None
        self.rag_manager: RAGManager = None
        self.workflow_engine: HybridWorkflowEngine = None
        self.project_state_manager: ProjectStateManager = None
        self.rag_sync_service: RagSyncService = None

        # Managed sub-windows
        self.code_viewer_window = CodeViewerWindow()
        self.workflow_monitor_window = WorkflowMonitorWindow()
        self.streaming_terminal = self.code_viewer_window.terminal  # Use terminal from code_viewer

        logging.info("AvAApplication initialized.")

    async def initialize_async(self):
        """Asynchronous initialization of all application components."""
        logging.info("Starting async initialization...")
        await self.config_manager.load()

        self.llm_client = EnhancedLLMClient()

        self.rag_manager = RAGManager()
        await self.rag_manager.async_initialize()

        self.rag_sync_service = RagSyncService()
        if self.rag_manager.upload_service:
            self.rag_sync_service.upload_service = self.rag_manager.upload_service

        self.workflow_engine = HybridWorkflowEngine(
            llm_client=self.llm_client,
            terminal_window=self.streaming_terminal,
            code_viewer=self.code_viewer_window,
            rag_manager=self.rag_manager
        )

        self.connect_signals()
        self.update_model_status()

        logging.info("✅ Async initialization complete.")
        self.initialization_complete.emit()

    def connect_signals(self):
        """Connect signals between all application components."""
        # --- Main Window to App ---
        self.main_window.workflow_requested_with_context.connect(self.run_workflow)
        self.main_window.sidebar.new_project_requested.connect(self.new_project)
        self.main_window.sidebar.load_project_requested.connect(self.load_project_dialog)
        self.main_window.sidebar.model_config_requested.connect(self.main_window._open_model_config_dialog)
        self.main_window.sidebar.scan_directory_requested.connect(self.rag_manager.scan_directory_dialog)
        self.main_window.sidebar.action_triggered.connect(self.handle_sidebar_action)
        self.model_status_updated.connect(self.main_window.update_model_status_display)

        # --- Workflow Engine to GUI ---
        self.workflow_engine.workflow_completed.connect(self.main_window.on_workflow_completed)
        self.workflow_engine.workflow_started.connect(self.main_window.on_workflow_started)
        # Note: The code_viewer is part of the CodeViewerWindow, not the main window.
        self.workflow_engine.file_generated.connect(self.code_viewer_window.auto_open_file)
        self.workflow_engine.project_loaded.connect(self.main_window.update_project_display)

        # --- Workflow Engine to Monitor Window ---
        self.workflow_engine.node_status_changed.connect(self.workflow_monitor_window.update_agent_status)
        self.workflow_engine.data_flow_started.connect(self.workflow_monitor_window.activate_connection)
        self.workflow_engine.data_flow_completed.connect(self.workflow_monitor_window.deactivate_connection)
        self.workflow_engine.workflow_reset.connect(self.workflow_monitor_window.refresh_workflow)

        # --- RAG Manager to GUI ---
        if hasattr(self.main_window.sidebar, 'knowledge_panel'):
            self.rag_manager.status_changed.connect(
                self.main_window.sidebar.knowledge_panel.update_rag_status
            )
        logging.info("All application signals connected.")

    @Slot()
    def update_model_status(self):
        """Fetches model assignments and emits them for the GUI to display."""
        if self.llm_client:
            assignments = self.llm_client.get_role_assignments()
            self.model_status_updated.emit(assignments)

    @Slot(str, list)
    def run_workflow(self, user_prompt: str, conversation_context: list):
        """Runs the main development workflow."""
        if not self.workflow_engine:
            logging.error("Workflow engine not initialized!")
            return

        # Create a new project state if one doesn't exist
        if not self.workflow_engine.project_state_manager:
            logging.warning("No project loaded. Creating a temporary project state.")
            # Create a temporary directory or use a default workspace
            temp_project_path = Path("./workspace/new_project")
            temp_project_path.mkdir(parents=True, exist_ok=True)
            self.project_state_manager = ProjectStateManager(str(temp_project_path))
            self.workflow_engine.set_project_state(self.project_state_manager)

        asyncio.create_task(self.workflow_engine.execute_enhanced_workflow(user_prompt, conversation_context))

    @Slot(str)
    def handle_sidebar_action(self, action: str):
        """Handle generic actions from the sidebar."""
        action_map = {
            "new_session": self.new_session,
            "open_workflow_monitor": self.workflow_monitor_window.show,
            "open_code_viewer": self.code_viewer_window.show,
            "add_project_files": self.add_project_to_rag,
            # Add other actions here...
            "view_log": self.view_log
        }
        handler = action_map.get(action)
        if handler:
            handler()
        else:
            logging.warning(f"No handler found for sidebar action: {action}")

    # --- Specific Action Handlers ---
    @Slot()
    def new_project(self):
        logging.info("New project requested.")
        self.project_state_manager = None
        self.workflow_engine.project_state_manager = None
        self.main_window.update_project_display("New Project")
        self.main_window.chat_interface.clear_chat()
        self.main_window.chat_interface.add_assistant_response("Ready to build a new project! What should we create?")

    @Slot()
    def load_project_dialog(self):
        from PySide6.QtWidgets import QFileDialog
        directory = QFileDialog.getExistingDirectory(self.main_window, "Select Project Folder")
        if directory:
            self.load_project(directory)
            self.code_viewer_window.load_project(directory)

    def load_project(self, project_path: str):
        """Loads an existing project into the application state."""
        try:
            logging.info(f"Loading project from: {project_path}")
            self.project_state_manager = ProjectStateManager(project_path)
            self.workflow_engine.set_project_state(self.project_state_manager)
            self.main_window.update_project_display(Path(project_path).name)
            # You might want to automatically analyze the project here
            # asyncio.create_task(self.workflow_engine.analyze_existing_project(project_path))
        except Exception as e:
            logging.error(f"Failed to load project: {e}", exc_info=True)

    @Slot()
    def new_session(self):
        logging.info("New chat session requested.")
        self.main_window.chat_interface.clear_chat()
        self.main_window.chat_interface.add_assistant_response("New session started. How can I help you?")

    @Slot()
    def add_project_to_rag(self):
        if self.project_state_manager:
            project_path = str(self.project_state_manager.project_root)
            logging.info(f"Adding project '{project_path}' to RAG.")
            asyncio.create_task(self.rag_sync_service.manual_sync_directory(project_path))
        else:
            logging.warning("Cannot add project to RAG: No project loaded.")

    @Slot()
    def view_log(self):
        # This could open a new window or just show the terminal
        self.code_viewer_window.show()
        self.code_viewer_window.raise_()
        self.code_viewer_window.activateWindow()