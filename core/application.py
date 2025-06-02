import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio # Ensure asyncio is imported

from PySide6.QtCore import QObject, Signal, QTimer
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
    project_loaded = Signal(str)

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
        self.rag_manager = None

        # Application state
        self.workspace_dir = Path("./workspace")
        self.workspace_dir.mkdir(exist_ok=True)

        self.current_project = "Default Project"
        self.current_session = "Main Chat"
        self.active_workflows = {}

        # Configuration
        self.current_config = {
            "chat_model": "Gemini: gemini-2.5-pro-preview-05-06",
            "code_model": "qwen2.5-coder:14b",
            "temperature": 0.7
        }

    def _setup_logging(self):
        """Setup Windows-compatible logging without Unicode emojis"""
        log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True)

        class SafeFormatter(logging.Formatter):
            def format(self, record):
                msg = super().format(record)
                emoji_replacements = {
                    '🚀': '[LAUNCH]', '✅': '[OK]', '❌': '[ERROR]', '⚠️': '[WARNING]',
                    '🧠': '[BRAIN]', '📊': '[STATS]', '🔧': '[TOOL]', '📄': '[FILE]',
                    '📝': '[EDIT]', '📂': '[FOLDER]'
                }
                for emoji, replacement in emoji_replacements.items():
                    msg = msg.replace(emoji, replacement)
                return msg

        formatter = SafeFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(log_dir / "ava.log", encoding='utf-8')
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

    async def initialize(self):
        self.logger.info("[LAUNCH] Initializing AvA Application (UI and core services part)...")

        self.main_window = AvAMainWindow(ava_app=self)
        self.logger.info("[OK] Main window initialized (sync part)")
        self.terminal_window = LLMTerminalWindow()
        self.logger.info("[OK] Terminal window initialized (sync part)")
        self.code_viewer = CodeViewerWindow()
        self.logger.info("[OK] Code viewer initialized (sync part)")

        self._initialize_core_services()

        if self.main_window and not self.main_window.isVisible():
            self.main_window.show()
            self.logger.info("Main window shown.")
            await asyncio.sleep(0.01) # Slightly longer sleep to ensure UI processes events

        self.logger.info("About to start and await async_initialize_components...")
        await self.async_initialize_components() # Directly await the async initialization
        self.logger.info("Finished awaiting async_initialize_components in initialize method.")
        # fully_initialized_signal is emitted from async_initialize_components

    async def async_initialize_components(self):
        self.logger.info("[LAUNCH] Initializing AvA Application (async components part)...")
        try:
            await self._initialize_rag_manager_async()
            self.logger.info(f"RAG Manager initialization attempt complete. RAG ready: {self.rag_manager.is_ready if self.rag_manager and hasattr(self.rag_manager, 'is_ready') else 'N/A or Manager not created'}")

            self._initialize_workflow_engine()
            self._connect_components()
            self._setup_window_behaviors()

            status = self.get_status()
            self.logger.info(f"System status after async init: {status}")
            self.logger.info("[OK] AvA Application async components initialized successfully.")

        except Exception as e:
            self.logger.error(f"[ERROR] Failed to initialize AvA async components: {e}", exc_info=True)
            self.error_occurred.emit("async_initialization", str(e))
        finally:
            # Ensure this signal is emitted regardless of success or failure in this block,
            # so main.py (or any listener) knows this phase is over.
            self.fully_initialized_signal.emit()


    def _initialize_core_services(self):
        self.logger.info("Initializing core services...")
        self.llm_client = LLMClient()
        available_models = self.llm_client.get_available_models()
        self.logger.info(f"Available LLM models: {available_models}")
        if not available_models or available_models == ["No LLM services available"]:
            self.logger.warning("[WARNING] No LLM services available!")

    async def _initialize_rag_manager_async(self):
        self.logger.info("Attempting to initialize RAG manager (async)...")
        if RAG_MANAGER_AVAILABLE:
            try:
                self.rag_manager = RAGManager()
                self.rag_manager.status_changed.connect(self._on_rag_status_changed)
                self.rag_manager.upload_completed.connect(self._on_rag_upload_completed)
                await self.rag_manager.async_initialize() # This now waits for embedder
                self.logger.info(f"[OK] RAG manager async_initialize call completed. RAG ready: {self.rag_manager.is_ready}")
            except Exception as e:
                self.logger.error(f"[ERROR] RAG manager async initialization threw an exception: {e}", exc_info=True)
                self.rag_manager = None # Ensure rag_manager is None if init fails
                self._on_rag_status_changed(f"RAG Init Exception: {e}", "error")
        else:
            self.logger.info("RAG services not available (import failed) - running without RAG functionality.")
            self.rag_manager = None
            self._on_rag_status_changed("RAG: Not Available (Import Fail)", "grey")

    def _initialize_workflow_engine(self):
        self.logger.info("Initializing workflow engine...")
        self.workflow_engine = WorkflowEngine(
            self.llm_client,
            self.terminal_window,
            self.code_viewer,
            self.rag_manager
        )
        self.logger.info("[OK] Workflow engine initialized.")

    def _connect_components(self):
        self.logger.info("Connecting components...")
        if self.main_window:
            self.main_window.workflow_requested.connect(self._handle_workflow_request)
            if hasattr(self.main_window, 'new_project_requested'): # Check before connecting
                self.main_window.new_project_requested.connect(self.create_new_project_dialog)

        self.workflow_started.connect(self._on_workflow_started)
        self.workflow_completed.connect(self._on_workflow_completed)
        self.error_occurred.connect(self._on_error_occurred)

        if self.workflow_engine: # Check if workflow_engine was initialized
            if self.main_window: # Check if main_window exists
                self.workflow_engine.workflow_started.connect(self.main_window.on_workflow_started)
                self.workflow_engine.workflow_completed.connect(self.main_window.on_workflow_completed)
            self.workflow_engine.file_generated.connect(self._on_file_generated)
            self.workflow_engine.project_loaded.connect(self._on_project_loaded)
        self.logger.info("[OK] Components connected.")


    def _setup_window_behaviors(self):
        if self.main_window:
            self.main_window.setWindowTitle(f"AvA - AI Development Assistant")
        if self.terminal_window:
            self.terminal_window.setWindowTitle("AvA - LLM Workflow Terminal")
        if self.code_viewer:
            self.code_viewer.setWindowTitle("AvA - Code Viewer")
        self._position_windows()
        self.logger.info("[OK] Window behaviors set up.")

    def _position_windows(self):
        screen_geo = QApplication.primaryScreen().geometry()
        if self.main_window: self.main_window.setGeometry(100, 50, 1400, 900)
        if self.terminal_window: self.terminal_window.setGeometry(screen_geo.width() - 850, screen_geo.height() - 400, 800, 350)
        if self.code_viewer: self.code_viewer.setGeometry(screen_geo.width() - 1000, 50, 950, 700)

    def get_status(self) -> Dict[str, Any]:
        llm_models_list = self.llm_client.get_available_models() if self.llm_client else ["LLM Client not init"]
        rag_info = {"ready": False, "status_text": "RAG: Not Initialized", "available": RAG_MANAGER_AVAILABLE, "collections": {}}
        if RAG_MANAGER_AVAILABLE:
            if self.rag_manager and hasattr(self.rag_manager, 'is_ready') and hasattr(self.rag_manager, 'current_status'):
                rag_info["ready"] = self.rag_manager.is_ready
                rag_info["status_text"] = self.rag_manager.current_status
                if hasattr(self.rag_manager, 'get_collection_info') and callable(getattr(self.rag_manager, 'get_collection_info')):
                    rag_info["collections"] = self.rag_manager.get_collection_info()
            elif self.rag_manager: # Manager exists but might not have all attributes (e.g. during init fail)
                 rag_info["status_text"] = "RAG: Manager exists, status uncertain"
            else: # RAG_MANAGER_AVAILABLE is true, but self.rag_manager is None (init failed badly)
                rag_info["status_text"] = "RAG: Initialization Failed (Manager None)"
        else: # RAG_MANAGER_AVAILABLE is false
            rag_info["status_text"] = "RAG: Dependencies Missing / Not Available"

        return {
            "ready": self.workflow_engine is not None,
            "llm_models": llm_models_list,
            "workspace": str(self.workspace_dir), "current_project": self.current_project,
            "current_session": self.current_session, "configuration": self.current_config,
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
        current_status = self.get_status()
        if not current_status["llm_models"] or current_status["llm_models"] == ["LLM Client not init"] or current_status["llm_models"] == ["No LLM services available"]:
            error_msg = "No LLM services available. Please configure API keys."
            if self.terminal_window: self.terminal_window.log(f"[ERROR] Workflow Halted: {error_msg}")
            self.error_occurred.emit("llm_unavailable", error_msg)
            return
        if not self.workflow_engine:
            error_msg = "Workflow engine not initialized. Cannot process request."
            if self.terminal_window: self.terminal_window.log(f"[ERROR] Workflow Halted: {error_msg}")
            self.error_occurred.emit("workflow_engine_unavailable", error_msg)
            return

        self._open_terminal()
        workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.active_workflows[workflow_id] = {"prompt": user_prompt, "start_time": datetime.now(), "status": "running"}
        self.workflow_started.emit(user_prompt)
        try:
            self.workflow_engine.execute_workflow(user_prompt)
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}", exc_info=True)
            self.error_occurred.emit("workflow_execution_error", str(e))
            self.active_workflows[workflow_id]["status"] = "failed"
            self.active_workflows[workflow_id]["error"] = str(e)


    def _open_terminal(self):
        if self.terminal_window: self.terminal_window.show(); self.terminal_window.raise_(); self.terminal_window.activateWindow()
    def _open_code_viewer(self):
        if self.code_viewer: self.code_viewer.show(); self.code_viewer.raise_(); self.code_viewer.activateWindow()

    def _on_file_generated(self, file_path: str):
        self.logger.info(f"File generated: {file_path}")
        if self.code_viewer: self.code_viewer.auto_open_file(file_path)

    def _on_project_loaded(self, project_path: str):
        self.logger.info(f"Project loaded: {project_path}")
        self.current_project = Path(project_path).name
        if self.main_window and hasattr(self.main_window, 'update_project_display'):
            self.main_window.update_project_display(self.current_project)
        self._open_code_viewer()

    def _on_workflow_started(self, prompt: str):
        self.logger.info(f"[OK] Workflow started (app signal): {prompt}")
        if self.main_window and hasattr(self.main_window, 'on_workflow_started'):
            self.main_window.on_workflow_started(prompt)

    def _on_workflow_completed(self, result: dict):
        self.logger.info(f"[OK] Workflow completed (app signal): {result.get('project_name', 'N/A')}")
        for wf_id, wf_data in self.active_workflows.items():
            if wf_data["status"] == "running": # Assuming only one runs at a time for this logic
                wf_data["status"] = "completed" if result.get("success") else "failed"
                wf_data["end_time"] = datetime.now()
                # ... (duration calculation)
                break
        if self.main_window and hasattr(self.main_window, 'on_workflow_completed'):
            self.main_window.on_workflow_completed(result)

    def _on_error_occurred(self, component: str, error_message: str):
        self.logger.error(f"[ERROR] App Error Signal in {component}: {error_message}")
        if self.main_window and hasattr(self.main_window, 'on_app_error_occurred'):
            self.main_window.on_app_error_occurred(component, error_message)

    def _on_rag_status_changed(self, status_text: str, color: str):
        self.logger.info(f"RAG Status Update (internal slot): {status_text} (Color: {color})")
        self.rag_status_changed.emit(status_text, color) # Re-emit for main_window

    def _on_rag_upload_completed(self, collection_id: str, files_processed: int):
        msg = f"RAG: {files_processed} files added to {collection_id}"
        self.logger.info(f"RAG Upload Completed (internal slot): {msg}")
        self.rag_status_changed.emit(msg, "#4ade80")

    def update_configuration(self, config_updates: dict):
        self.current_config.update(config_updates)
        self.logger.info(f"Configuration updated: {config_updates}")
        if self.main_window and hasattr(self.main_window, '_update_initial_ui_status'):
             self.main_window._update_initial_ui_status()

    def create_new_project_dialog(self):
        self.logger.info("Request to create a new project received.")
        if self.terminal_window: self.terminal_window.log("✨ New Project creation requested.")
        from PySide6.QtWidgets import QFileDialog
        if self.main_window:
            project_dir = QFileDialog.getExistingDirectory(self.main_window, "Create or Select Project Directory", str(self.workspace_dir))
            if project_dir:
                project_path = Path(project_dir)
                self.current_project = project_path.name
                self.logger.info(f"New project selected/created at: {project_path}")
                if hasattr(self.main_window, 'update_project_display'): self.main_window.update_project_display(self.current_project)
                self.project_loaded.emit(str(project_path)) # Ensure this signal is connected if used
            else: self.logger.info("New project creation cancelled by user.")

    def shutdown(self):
        self.logger.info("Shutting down AvA Application...")
        # Add proper async shutdown for RAG if needed
        if self.main_window: self.main_window.close()
        if self.terminal_window: self.terminal_window.close()
        if self.code_viewer: self.code_viewer.close()
        self.logger.info("AvA Application shutdown complete.")

