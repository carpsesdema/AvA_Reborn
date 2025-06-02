# core/application.py - Fixed for Windows and Missing RAG Services

import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio

from PySide6.QtCore import QObject, Signal, QTimer  # Ensure QTimer is imported
from PySide6.QtWidgets import QApplication

# Import the components
from gui.main_window import AvAMainWindow
from windows.terminal_window import LLMTerminalWindow
from windows.code_viewer import CodeViewerWindow
from core.workflow_engine import WorkflowEngine
from core.llm_client import LLMClient

# Try to import RAG manager - gracefully handle if not available
try:
    from core.rag_manager import RAGManager

    RAG_MANAGER_AVAILABLE = True
except ImportError as e:
    RAG_MANAGER_AVAILABLE = False
    print(f"RAG Manager not available: {e}")


class AvAApplication(QObject):
    """
    AvA Application - Windows Compatible with Optional RAG
    """
    fully_initialized_signal = Signal()  # Signal to indicate all async init is done

    # Signals for status updates
    workflow_started = Signal(str)
    workflow_completed = Signal(dict)
    error_occurred = Signal(str, str)
    rag_status_changed = Signal(str, str)  # For RAGManager to emit to this app instance

    def __init__(self):
        super().__init__()

        # Initialize logging (Windows-compatible)
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

        # Core components
        self.llm_client = None
        self.main_window = None
        self.terminal_window = None
        self.code_viewer = None
        self.workflow_engine = None
        self.rag_manager = None  # Will be initialized in async_initialize

        # Application state
        self.workspace_dir = Path("./workspace")
        self.workspace_dir.mkdir(exist_ok=True)

        self.current_project = "Default Project"
        self.current_session = "Main Chat"
        self.active_workflows = {}

        # Configuration
        self.current_config = {
            "chat_model": "Gemini: gemini-2.5-pro",
            "code_model": "Ollama: qwen2.5-coder",
            "temperature": 0.7
        }

    def _setup_logging(self):
        """Setup Windows-compatible logging without Unicode emojis"""
        log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True)

        # Create a custom formatter that handles Unicode safely
        class SafeFormatter(logging.Formatter):
            def format(self, record):
                # Replace Unicode emojis with safe ASCII
                msg = super().format(record)
                emoji_replacements = {
                    '🚀': '[LAUNCH]',
                    '✅': '[OK]',
                    '❌': '[ERROR]',
                    '⚠️': '[WARNING]',
                    '🧠': '[BRAIN]',
                    '📊': '[STATS]',
                    '🔧': '[TOOL]',
                    '📄': '[FILE]',
                    '📝': '[EDIT]',
                    '📂': '[FOLDER]'
                }
                for emoji, replacement in emoji_replacements.items():
                    msg = msg.replace(emoji, replacement)
                return msg

        formatter = SafeFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # File handler with UTF-8 encoding
        file_handler = logging.FileHandler(log_dir / "ava.log", encoding='utf-8')
        file_handler.setFormatter(formatter)

        # Console handler with safe encoding
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            handlers=[file_handler, console_handler]
        )

    def initialize(self):
        """
        Synchronous part of initialization. UI elements are created here.
        Async components are scheduled via QTimer to ensure the event loop is running.
        """
        self.logger.info("[LAUNCH] Initializing AvA Application (sync part)...")

        self.main_window = AvAMainWindow(ava_app=self)
        self.logger.info("[OK] Main window initialized (sync part)")
        self.terminal_window = LLMTerminalWindow()
        self.logger.info("[OK] Terminal window initialized (sync part)")
        self.code_viewer = CodeViewerWindow()
        self.logger.info("[OK] Code viewer initialized (sync part)")

        # Initialize core services (LLMClient is synchronous)
        self._initialize_core_services()

        # Show the main window early so the user sees something.
        if self.main_window and not self.main_window.isVisible():
            self.main_window.show()
            self.logger.info("Main window shown.")

        # Schedule the async initialization to run after the current event processing.
        # This ensures the asyncio event loop (managed by qasync) is fully up and running.
        QTimer.singleShot(0, lambda: asyncio.create_task(self.async_initialize_components()))
        self.logger.info("Scheduled async_initialize_components.")

    async def async_initialize_components(self):
        """Asynchronously initialize all application components."""
        self.logger.info("[LAUNCH] Initializing AvA Application (async part)...")

        try:
            # Initialize RAG manager (now async)
            await self._initialize_rag_manager_async()

            # Initialize workflow engine (depends on LLMClient and RAGManager)
            self._initialize_workflow_engine()

            # Connect components
            self._connect_components()

            # Setup window behaviors
            self._setup_window_behaviors()

            # Check system status
            status = self._check_system_status()  # get_status uses self.rag_manager
            self.logger.info(f"System status: {status}")

            self.logger.info("[OK] AvA Application async components initialized successfully")
            self.fully_initialized_signal.emit()


        except Exception as e:
            self.logger.error(f"[ERROR] Failed to initialize AvA async components: {e}", exc_info=True)
            self.error_occurred.emit("async_initialization", str(e))

    def _initialize_core_services(self):
        """Initialize core business logic services (synchronous part)"""
        self.logger.info("Initializing core services...")

        self.llm_client = LLMClient()
        available_models = self.llm_client.get_available_models()
        self.logger.info(f"Available LLM models: {available_models}")

        if not available_models or available_models == ["No LLM services available"]:
            self.logger.warning("[WARNING] No LLM services available!")

    async def _initialize_rag_manager_async(self):
        self.logger.info("Initializing RAG manager (async)...")
        if RAG_MANAGER_AVAILABLE:
            try:
                self.rag_manager = RAGManager()
                self.rag_manager.status_changed.connect(self._on_rag_status_changed)
                self.rag_manager.upload_completed.connect(self._on_rag_upload_completed)
                await self.rag_manager.async_initialize()
                self.logger.info("[OK] RAG manager async initialization complete.")
            except Exception as e:
                self.logger.error(f"[ERROR] RAG manager async initialization failed: {e}", exc_info=True)
                self.rag_manager = None
                self._on_rag_status_changed(f"RAG Init Error: {e}", "error")
        else:
            self.logger.info("RAG services not available - running without RAG functionality")
            self.rag_manager = None
            self._on_rag_status_changed("RAG: Not Available", "grey")

    def _initialize_workflow_engine(self):
        self.logger.info("Initializing workflow engine...")
        self.workflow_engine = WorkflowEngine(
            self.llm_client,
            self.terminal_window,
            self.code_viewer,
            self.rag_manager
        )
        self.logger.info("[OK] Workflow engine initialized")

    def _connect_components(self):
        self.logger.info("Connecting components...")
        if self.main_window:
            self.main_window.workflow_requested.connect(self._handle_workflow_request)
            # Connect the new_project_requested signal
            if hasattr(self.main_window, 'new_project_requested') and self.main_window.new_project_requested:
                self.main_window.new_project_requested.connect(self.create_new_project_dialog)

        self.workflow_started.connect(self._on_workflow_started)
        self.workflow_completed.connect(self._on_workflow_completed)
        self.error_occurred.connect(self._on_error_occurred)

        if self.workflow_engine and self.main_window:
            self.workflow_engine.workflow_started.connect(self.main_window.on_workflow_started)
            self.workflow_engine.workflow_completed.connect(self.main_window.on_workflow_completed)
            self.workflow_engine.file_generated.connect(self._on_file_generated)
            self.workflow_engine.project_loaded.connect(self._on_project_loaded)

    def _setup_window_behaviors(self):
        if self.main_window:
            self.main_window.setWindowTitle(f"AvA - AI Development Assistant")
        if self.terminal_window:
            self.terminal_window.setWindowTitle("AvA - LLM Workflow Terminal")
        if self.code_viewer:
            self.code_viewer.setWindowTitle("AvA - Code Viewer")
        self._position_windows()

    def _position_windows(self):
        screen_geo = QApplication.primaryScreen().geometry()
        if self.main_window:
            self.main_window.setGeometry(100, 50, 1400, 900)
        if self.terminal_window:
            self.terminal_window.setGeometry(
                screen_geo.width() - 850,
                screen_geo.height() - 400,
                800, 350
            )
        if self.code_viewer:
            self.code_viewer.setGeometry(
                screen_geo.width() - 1000,
                50,
                950, 700
            )

    def _check_system_status(self) -> Dict[str, Any]:  # Renamed from get_status to avoid conflict if inherited
        """Check system status and dependencies"""
        rag_is_ready = False
        rag_current_status = "RAG: Unknown"
        rag_collections = {}

        if self.rag_manager and hasattr(self.rag_manager, 'is_ready'):
            rag_is_ready = self.rag_manager.is_ready
            if hasattr(self.rag_manager, 'current_status'):
                rag_current_status = self.rag_manager.current_status
            if hasattr(self.rag_manager, 'get_collection_info'):
                rag_collections = self.rag_manager.get_collection_info()

        status = {
            "llm_available": self.llm_client is not None and len(self.llm_client.get_available_models()) > 0,
            "workspace": self.workspace_dir.exists(),
            "rag_ready": rag_is_ready,
            "rag_current_status": rag_current_status,
            "rag_collections": rag_collections,
            "rag_available": RAG_MANAGER_AVAILABLE and self.rag_manager is not None,
            "windows_ready": all([
                self.main_window is not None,
                self.terminal_window is not None,
                self.code_viewer is not None,
                self.workflow_engine is not None
            ])
        }
        return status

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive application status for external calls (like from main.py)."""
        # This method can call _check_system_status or be a simplified version
        # if _check_system_status is too detailed or for internal use.
        # For now, let's make it comprehensive.

        # Core LLM status
        llm_models_list = []
        if self.llm_client:
            llm_models_list = self.llm_client.get_available_models()

        # RAG Status from RAGManager
        rag_info = {
            "ready": False,
            "status_text": "RAG: Not Initialized",
            "available": RAG_MANAGER_AVAILABLE,
            "collections": {}
        }
        if RAG_MANAGER_AVAILABLE:
            if self.rag_manager:  # If RAGManager instance exists
                rag_info["ready"] = self.rag_manager.is_ready
                rag_info["status_text"] = self.rag_manager.current_status
                if self.rag_manager.is_ready:
                    rag_info["collections"] = self.rag_manager.get_collection_info()
            else:  # RAG is available by import, but manager instance doesn't exist (init failed early)
                rag_info["status_text"] = "RAG: Initialization Failed"
        else:  # RAG not available due to import error
            rag_info["status_text"] = "RAG: Dependencies Missing"
            rag_info["reason"] = "RAG services not installed or critical import failed."

        return {
            "ready": self.workflow_engine is not None,  # Example: app is "ready" if workflow engine is up
            "llm_models": llm_models_list,
            "workspace": str(self.workspace_dir),
            "current_project": self.current_project,
            "current_session": self.current_session,
            "configuration": self.current_config,
            "rag": rag_info,
            "windows": {
                "main": self.main_window.isVisible() if self.main_window else False,
                "terminal": self.terminal_window.isVisible() if self.terminal_window else False,
                "code_viewer": self.code_viewer.isVisible() if self.code_viewer else False
            },
            "active_workflows": len(self.active_workflows)
        }

    def _handle_workflow_request(self, user_prompt: str):
        self.logger.info(f"Workflow request received: {user_prompt[:100]}...")
        current_status = self._check_system_status()
        if not current_status["llm_available"]:
            error_msg = "No LLM services available. Please configure API keys."
            if self.terminal_window: self.terminal_window.log(f"[ERROR] Error: {error_msg}")
            self.error_occurred.emit("llm", error_msg)
            return
        if not self.workflow_engine:
            error_msg = "Workflow engine not initialized. Please wait or restart."
            if self.terminal_window: self.terminal_window.log(f"[ERROR] Error: {error_msg}")
            self.error_occurred.emit("workflow_engine", error_msg)
            return

        self._open_terminal()
        workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.active_workflows[workflow_id] = {
            "prompt": user_prompt, "start_time": datetime.now(), "status": "running"
        }
        self.workflow_started.emit(user_prompt)
        try:
            self.workflow_engine.execute_workflow(user_prompt)
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}", exc_info=True)
            self.error_occurred.emit("workflow", str(e))

    def _open_terminal(self):
        if self.terminal_window:
            self.terminal_window.show()
            self.terminal_window.raise_()
            self.terminal_window.activateWindow()

    def _open_code_viewer(self):
        if self.code_viewer:
            self.code_viewer.show()
            self.code_viewer.raise_()
            self.code_viewer.activateWindow()

    def _on_file_generated(self, file_path: str):
        self.logger.info(f"File generated: {file_path}")
        if self.code_viewer:
            self.code_viewer.auto_open_file(file_path)

    def _on_project_loaded(self, project_path: str):
        self.logger.info(f"Project loaded: {project_path}")
        self.current_project = Path(project_path).name
        self._open_code_viewer()

    def _on_workflow_started(self, prompt: str):
        self.logger.info(f"[OK] Workflow started: {prompt}")

    def _on_workflow_completed(self, result: dict):
        self.logger.info(f"[OK] Workflow completed: {result}")
        for workflow_id, workflow in self.active_workflows.items():
            if workflow["status"] == "running":
                workflow["status"] = "completed"
                workflow["end_time"] = datetime.now()
                workflow["duration"] = (workflow["end_time"] - workflow["start_time"]).total_seconds()
                break

    def _on_error_occurred(self, component: str, error_message: str):
        self.logger.error(f"[ERROR] Error in {component}: {error_message}")

    def _on_rag_status_changed(self, status_text: str, color: str):
        self.logger.info(f"RAG Status: {status_text} (Color: {color})")
        if self.main_window and hasattr(self.main_window, 'update_rag_status'):
            self.main_window.update_rag_status(status_text, color)
        self.rag_status_changed.emit(status_text, color)

    def _on_rag_upload_completed(self, collection_id: str, files_processed: int):
        self.logger.info(f"RAG Upload completed: {files_processed} files processed in {collection_id}")
        if self.main_window and hasattr(self.main_window, 'update_rag_status'):
            self.main_window.update_rag_status(f"RAG: {files_processed} files added to {collection_id}", "#4ade80")

    def update_configuration(self, config_updates: dict):
        self.current_config.update(config_updates)
        self.logger.info(f"Configuration updated: {config_updates}")

    def create_new_project_dialog(self):
        """Handles the logic for creating a new project."""
        # This is where you'd open a QFileDialog or a custom dialog
        # to get the new project's name and path.
        # For simplicity, let's just log it for now.
        self.logger.info("Request to create a new project received.")
        if self.terminal_window:
            self.terminal_window.log("✨ New Project creation requested (dialog not yet implemented).")

        # Example: Using QFileDialog to get a directory
        from PySide6.QtWidgets import QFileDialog
        if self.main_window:  # Ensure a parent window exists for the dialog
            project_dir = QFileDialog.getExistingDirectory(
                self.main_window,
                "Create or Select Project Directory",
                str(self.workspace_dir)  # Start in the default workspace
            )
            if project_dir:
                project_path = Path(project_dir)
                project_name = project_path.name
                self.current_project = project_name
                # Further logic to "create" or "load" this project
                # For example, update UI, create a .ava project file, etc.
                self.logger.info(f"New project selected/created at: {project_path}")
                if self.main_window and hasattr(self.main_window, 'update_project_display'):
                    self.main_window.update_project_display(project_name)
                self.project_loaded.emit(str(project_path))  # Assuming project_loaded signal is for this
            else:
                self.logger.info("New project creation cancelled by user.")

    def shutdown(self):
        self.logger.info("Shutting down AvA Application...")
        if self.main_window: self.main_window.close()
        if self.terminal_window: self.terminal_window.close()
        if self.code_viewer: self.code_viewer.close()
        if self.rag_manager and hasattr(self.rag_manager, 'shutdown'):
            asyncio.create_task(self.rag_manager.shutdown())
        self.logger.info("AvA Application shutdown complete")