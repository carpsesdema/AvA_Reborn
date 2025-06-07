# core/application.py - Streamlined for Fast Professional Results

import asyncio
import logging
import json  # New import for session handling
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from PySide6.QtCore import QObject, Signal, QTimer
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QInputDialog  # New imports for dialogs

from core.llm_client import EnhancedLLMClient
from core.enhanced_workflow_engine import EnhancedWorkflowEngine

# Import the components
from gui.main_window import AvAMainWindow
from windows.code_viewer import CodeViewerWindow
from gui.terminals import TerminalWindow  # This is actually StreamingTerminal

# Try to import RAG manager - gracefully handle if not available
try:
    from core.rag_manager import RAGManager

    RAG_MANAGER_AVAILABLE = True
except ImportError as e:
    RAG_MANAGER_AVAILABLE = False
    print(f"RAG Manager not available: {e}")


class AvAApplication(QObject):
    """
    AvA Application - Streamlined for Fast Professional Results
    """
    fully_initialized_signal = Signal()

    # Basic signals only
    workflow_started = Signal(str)
    workflow_completed = Signal(dict)
    error_occurred = Signal(str, str)
    rag_status_changed = Signal(str, str)
    project_loaded = Signal(str)

    def __init__(self):
        super().__init__()

        # Initialize logging
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
        self.current_project_path = self.workspace_dir  # Keep track of the project path
        self.current_session = "Main Chat"
        self.active_workflows = {}

        # Streamlined Configuration
        self.current_config = {
            "chat_model": "gemini-2.5-pro-preview-06-05",
            "code_model": "qwen2.5-coder:14b",
            "temperature": 0.7
        }

        # Performance monitoring timer
        self.performance_timer = QTimer()
        self.performance_timer.timeout.connect(self._update_performance_stats)
        self.performance_timer.start(5000)

    def _setup_logging(self):
        """Setup streamlined logging"""
        log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True)

        class SafeFormatter(logging.Formatter):
            def format(self, record):
                msg = super().format(record)
                emoji_replacements = {
                    '🚀': '[START]', '✅': '[OK]', '❌': '[ERROR]', '⚠️': '[WARN]',
                    '🧠': '[AI]', '📊': '[STATS]', '🔧': '[TOOL]', '📄': '[FILE]'
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
        self.logger.info("[START] Initializing Streamlined AvA Application...")

        self.main_window = AvAMainWindow(ava_app=self)
        self.logger.info("[OK] Main window initialized")

        self.terminal_window = TerminalWindow()
        self.logger.info("[OK] Terminal window initialized")

        self.code_viewer = CodeViewerWindow()
        self.logger.info("[OK] Code viewer initialized")

        self._initialize_core_services()

        if self.main_window and not self.main_window.isVisible():
            self.main_window.show()
            self.logger.info("Main window shown.")
            await asyncio.sleep(0.01)

        self.logger.info("Starting async component initialization...")
        await self.async_initialize_components()
        self.logger.info("Async component initialization completed.")

    async def async_initialize_components(self):
        self.logger.info("[START] Initializing streamlined components...")
        try:
            await self._initialize_rag_manager_async()
            self.logger.info("RAG Manager initialization complete.")

            # Always use streamlined workflow engine
            self._initialize_workflow_engine()

            self._connect_components()
            self._setup_window_behaviors()

            status = self.get_status()
            self.logger.info(f"System status: {status}")
            self.logger.info("[OK] Streamlined AvA Application initialized successfully.")

        except Exception as e:
            self.logger.error(f"[ERROR] Failed to initialize components: {e}", exc_info=True)
            self.error_occurred.emit("async_initialization", str(e))
        finally:
            self.fully_initialized_signal.emit()

    def _initialize_core_services(self):
        self.logger.info("Initializing core services...")
        self.llm_client = EnhancedLLMClient()
        available_models = self.llm_client.get_available_models()
        self.logger.info(f"Available LLM models: {available_models}")
        if not available_models or "No LLM services available" in available_models[0]:
            self.logger.warning("[WARN] No LLM services available!")

    async def _initialize_rag_manager_async(self):
        self.logger.info("Attempting to initialize RAG manager...")
        if RAG_MANAGER_AVAILABLE:
            try:
                self.rag_manager = RAGManager()
                self.rag_manager.status_changed.connect(self._on_rag_status_changed)
                self.rag_manager.upload_completed.connect(self._on_rag_upload_completed)
                await self.rag_manager.async_initialize()
                self.logger.info(f"[OK] RAG manager initialized. Ready: {self.rag_manager.is_ready}")
            except Exception as e:
                self.logger.error(f"[ERROR] RAG manager initialization failed: {e}", exc_info=True)
                self.rag_manager = None
                self._on_rag_status_changed(f"RAG Init Exception: {e}", "error")
        else:
            self.logger.info("RAG services not available - running without RAG functionality.")
            self.rag_manager = None
            self._on_rag_status_changed("RAG: Not Available", "grey")

    def _initialize_workflow_engine(self):
        """Initialize streamlined workflow engine"""
        self.logger.info("Initializing streamlined workflow engine...")
        self.workflow_engine = EnhancedWorkflowEngine(
            self.llm_client,
            self.terminal_window,
            self.code_viewer,
            self.rag_manager
        )
        self.logger.info("[OK] Streamlined workflow engine initialized.")

    def _connect_components(self):
        self.logger.info("Connecting components...")
        if self.main_window:
            # Connect workflow requests
            self.main_window.workflow_requested.connect(self._handle_workflow_request)
            if hasattr(self.main_window, 'workflow_requested_with_context'):
                self.main_window.workflow_requested_with_context.connect(
                    self._handle_enhanced_workflow_request
                )
            if hasattr(self.main_window, 'new_project_requested'):
                self.main_window.new_project_requested.connect(self.create_new_project_dialog)
            if hasattr(self.main_window.sidebar, 'action_triggered'):
                self.main_window.sidebar.action_triggered.connect(self._handle_sidebar_action)

        # Connect workflow signals to the main window
        if self.workflow_engine and self.main_window:
            self.workflow_engine.workflow_completed.connect(self.main_window.on_workflow_completed)

        self.workflow_started.connect(self._on_workflow_started)
        self.workflow_completed.connect(self._on_workflow_completed)
        self.error_occurred.connect(self._on_error_occurred)

        if self.workflow_engine:
            self.workflow_engine.file_generated.connect(self._on_file_generated)
            self.workflow_engine.project_loaded.connect(self._on_project_loaded)

        self.logger.info("[OK] Components connected.")

    def _setup_window_behaviors(self):
        if self.main_window:
            self.main_window.setWindowTitle("AvA - Fast Professional AI Development")
        if self.terminal_window:
            self.terminal_window.setWindowTitle("AvA - Workflow Terminal")
        if self.code_viewer:
            self.code_viewer.setWindowTitle("AvA - Code Viewer")
        self._position_windows()
        self.logger.info("[OK] Window behaviors set up.")

    def _position_windows(self):
        screen_geo = QApplication.primaryScreen().geometry()
        if self.main_window:
            self.main_window.setGeometry(100, 50, 1400, 900)
        if self.terminal_window:
            # Adjusted width for potential scrollbar, and height for more content
            self.terminal_window.setGeometry(screen_geo.width() - 920, screen_geo.height() - 450, 900, 400)
        if self.code_viewer:
            self.code_viewer.setGeometry(screen_geo.width() - 1000, 50, 950, 700)

    def _update_performance_stats(self):
        """Update performance statistics"""
        if not (self.terminal_window and self.workflow_engine):
            return

        try:
            stats = self.workflow_engine.get_workflow_stats()
            cache_stats = stats.get("cache_stats", {})

            cache_size = cache_stats.get("cache_size", 0)
            hit_rate = cache_stats.get("hit_rate", 0.0)

            if hasattr(self.terminal_window, 'update_cache_status'):
                self.terminal_window.update_cache_status(cache_size, hit_rate)

            workflow_state = stats.get("workflow_state", {})
            if workflow_state.get("stage") not in ["idle", "complete", "error"]:
                completed = workflow_state.get("completed_tasks", 0)
                total = workflow_state.get("total_tasks", 0)
                if hasattr(self.terminal_window, 'update_task_progress'):
                    self.terminal_window.update_task_progress(completed, total)

        except Exception as e:
            self.logger.debug(f"Performance stats update failed: {e}")

    def get_status(self) -> Dict[str, Any]:
        llm_models_list = self.llm_client.get_available_models() if self.llm_client else ["LLM Client not init"]
        rag_info = {"ready": False, "status_text": "RAG: Not Initialized", "available": RAG_MANAGER_AVAILABLE,
                    "collections": {}}

        if RAG_MANAGER_AVAILABLE:
            if self.rag_manager and hasattr(self.rag_manager, 'is_ready') and hasattr(self.rag_manager,
                                                                                      'current_status'):
                rag_info["ready"] = self.rag_manager.is_ready
                rag_info["status_text"] = self.rag_manager.current_status
                if hasattr(self.rag_manager, 'get_collection_info') and callable(
                        getattr(self.rag_manager, 'get_collection_info')):
                    rag_info["collections"] = self.rag_manager.get_collection_info()
            elif self.rag_manager:
                rag_info["status_text"] = "RAG: Manager exists, status uncertain"
            else:
                rag_info["status_text"] = "RAG: Initialization Failed"
        else:
            rag_info["status_text"] = "RAG: Dependencies Missing"

        performance_stats = {}
        if self.workflow_engine:
            try:
                workflow_stats = self.workflow_engine.get_workflow_stats()
                performance_stats = {
                    "cache_stats": workflow_stats.get("cache_stats", {}),
                    "workflow_state": workflow_stats.get("workflow_state", {})
                }
            except Exception:
                pass

        return {
            "ready": self.workflow_engine is not None,
            "streamlined": True,
            "llm_models": llm_models_list,
            "workspace": str(self.workspace_dir),
            "current_project": self.current_project,
            "current_session": self.current_session,
            "configuration": self.current_config,
            "rag": rag_info,
            "performance": performance_stats,
            "windows": {
                "main": self.main_window.isVisible() if self.main_window else False,
                "terminal": self.terminal_window.isVisible() if self.terminal_window else False,
                "code_viewer": self.code_viewer.isVisible() if self.code_viewer else False
            },
            "active_workflows": len(self.active_workflows)
        }

    def _handle_enhanced_workflow_request(self, user_prompt: str, conversation_history: List[Dict]):
        self.logger.info(f"Fast workflow request: {user_prompt[:100]}...")
        self.logger.info(f"Context: {len(conversation_history)} messages")
        current_status = self.get_status()

        if not current_status["llm_models"] or "No LLM services available" in current_status["llm_models"][0]:
            error_msg = "No LLM services available. Please configure API keys."
            if self.terminal_window and hasattr(self.terminal_window, 'stream_log_rich'):
                self.terminal_window.stream_log_rich("Application", "error", f"Workflow Halted: {error_msg}", "0")
            self.error_occurred.emit("llm_unavailable", error_msg)
            return

        if not self.workflow_engine:
            error_msg = "Workflow engine not initialized."
            if self.terminal_window and hasattr(self.terminal_window, 'stream_log_rich'):
                self.terminal_window.stream_log_rich("Application", "error", f"Workflow Halted: {error_msg}", "0")
            self.error_occurred.emit("workflow_engine_unavailable", error_msg)
            return

        self._open_terminal()
        workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.active_workflows[workflow_id] = {
            "prompt": user_prompt, "conversation_history": conversation_history,
            "start_time": datetime.now(), "status": "running"
        }
        self.workflow_started.emit(user_prompt)

        try:
            self.logger.info("Starting FAST professional workflow...")
            if hasattr(self.workflow_engine, 'execute_enhanced_workflow'):
                asyncio.create_task(self.workflow_engine.execute_enhanced_workflow(
                    user_prompt, conversation_context=conversation_history
                ))
            else:  # Fallback
                self.workflow_engine.execute_workflow(user_prompt)
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}", exc_info=True)
            self.error_occurred.emit("workflow_execution_error", str(e))
            if workflow_id in self.active_workflows:  # Check if key exists before updating
                self.active_workflows[workflow_id]["status"] = "failed"
                self.active_workflows[workflow_id]["error"] = str(e)

    def _handle_workflow_request(self, user_prompt: str):
        self.logger.info(f"Standard workflow request: {user_prompt[:100]}...")
        self._handle_enhanced_workflow_request(user_prompt, [])

    def _handle_sidebar_action(self, action: str):
        """Handle sidebar action triggers from the main window."""
        if action == "save_session":
            self.save_session()
        elif action == "load_session":
            self.load_session()
        elif action == "new_session":
            self.new_session()
        elif action == "open_terminal":
            self._open_terminal()
        elif action == "open_code_viewer":
            self._open_code_viewer()
        else:
            message = f"Action '{action}' not implemented yet."
            if self.main_window:
                self.main_window.chat_interface.add_assistant_response(message)
            self.logger.warning(message)

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
        self.current_project_path = Path(project_path)
        self.current_project = self.current_project_path.name
        if self.main_window and hasattr(self.main_window, 'update_project_display'):
            self.main_window.update_project_display(self.current_project)
        self._open_code_viewer()

    def _on_workflow_started(self, prompt: str):
        self.logger.info(f"[OK] Workflow started: {prompt}")

    def _on_workflow_completed(self, result: dict):
        self.logger.info(f"[OK] Workflow completed: {result.get('project_name', 'N/A')}")
        # Update the current project path upon successful workflow completion
        if result.get("success") and "project_dir" in result:
            self._on_project_loaded(result["project_dir"])
        for wf_id, wf_data in self.active_workflows.items():
            if wf_data["status"] == "running":
                wf_data["status"] = "completed" if result.get("success") else "failed"
                wf_data["end_time"] = datetime.now()
                break

    def _on_error_occurred(self, component: str, error_message: str):
        self.logger.error(f"[ERROR] App Error in {component}: {error_message}")
        if self.terminal_window and hasattr(self.terminal_window, 'stream_log_rich'):
            self.terminal_window.stream_log_rich("Application", "error", f"Error in {component}: {error_message}", "0")

    def _on_rag_status_changed(self, status_text: str, color: str):
        self.logger.info(f"RAG Status: {status_text}")
        self.rag_status_changed.emit(status_text, color)

    def _on_rag_upload_completed(self, collection_id: str, files_processed: int):
        msg = f"RAG: {files_processed} files added to {collection_id}"
        self.logger.info(f"RAG Upload: {msg}")
        self.rag_status_changed.emit(msg, "#4ade80")

    def new_session(self):
        """Clears the chat and starts a fresh session."""
        if self.main_window:
            self.main_window.chat_interface.clear_chat()
            self.main_window.chat_interface._add_welcome_message()
            self.current_project = "Default Project"
            self.current_project_path = self.workspace_dir
            self.main_window.update_project_display(self.current_project)
            QMessageBox.information(self.main_window, "New Session", "Chat history has been cleared.")
        self.logger.info("New session started.")

    def save_session(self):
        """Saves the current chat session to a file."""
        if not self.main_window: return

        history = self.main_window.chat_interface.conversation_history
        session_data = {
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "project_name": self.current_project,
            "project_path": str(self.current_project_path),
            "conversation_history": history
        }

        # If a real project is active, save inside it. Otherwise, ask where to save.
        if self.current_project != "Default Project" and self.current_project_path.exists():
            session_dir = self.current_project_path / ".sessions"
            session_dir.mkdir(exist_ok=True)
            file_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            save_path = session_dir / file_name
        else:
            file_name, _ = QFileDialog.getSaveFileName(
                self.main_window,
                "Save Session",
                str(self.workspace_dir),
                "JSON Files (*.json)"
            )
            if not file_name:
                return
            save_path = Path(file_name)

        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2)
            QMessageBox.information(self.main_window, "Session Saved", f"Session saved to:\n{save_path}")
            self.logger.info(f"Session saved to {save_path}")
        except Exception as e:
            QMessageBox.critical(self.main_window, "Error", f"Failed to save session: {e}")
            self.logger.error(f"Failed to save session: {e}", exc_info=True)

    def load_session(self):
        """Loads a chat session from a file."""
        if not self.main_window: return

        file_name, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Load Session",
            str(self.workspace_dir),
            "JSON Files (*.json)"
        )

        if not file_name:
            return

        try:
            with open(file_name, 'r', encoding='utf-8') as f:
                session_data = json.load(f)

            # Validate session data
            if "conversation_history" not in session_data or "project_path" not in session_data:
                raise ValueError("Invalid session file format.")

            # Load the chat history into the UI
            self.main_window.chat_interface.load_history(session_data["conversation_history"])

            # Restore project context
            project_path = session_data.get("project_path")
            self._on_project_loaded(project_path)

            # Update code viewer if it exists
            if self.code_viewer:
                self.code_viewer.load_project(project_path)

            QMessageBox.information(self.main_window, "Session Loaded", f"Session loaded from:\n{file_name}")
            self.logger.info(f"Session loaded from {file_name}")

        except Exception as e:
            QMessageBox.critical(self.main_window, "Error", f"Failed to load session: {e}")
            self.logger.error(f"Failed to load session: {e}", exc_info=True)

    def update_configuration(self, config_updates: dict):
        self.current_config.update(config_updates)
        self.logger.info(f"Configuration updated: {config_updates}")

    def create_new_project_dialog(self):
        self.logger.info("Creating new project...")
        if self.terminal_window and hasattr(self.terminal_window, 'stream_log_rich'):
            self.terminal_window.stream_log_rich("Application", "info", "✨ New Project creation started.", "0")

        if not self.main_window: return

        try:
            project_name, ok = QInputDialog.getText(
                self.main_window, 'New Project', 'Enter project name:', text='my_new_project'
            )
            if not ok or not project_name.strip(): return
            project_name = project_name.strip()

            base_dir = QFileDialog.getExistingDirectory(
                self.main_window, 'Select Directory for New Project', str(self.workspace_dir)
            )
            if not base_dir: return

            project_path = Path(base_dir) / project_name
            if project_path.exists():
                reply = QMessageBox.question(
                    self.main_window, 'Directory Exists',
                    f'Directory "{project_name}" exists. Use it anyway?',
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No: return
            else:
                project_path.mkdir(parents=True, exist_ok=True)

            self.current_project = project_name
            self.current_project_path = project_path  # Update current project path

            (project_path / "README.md").write_text(f"# {project_name}\n\nA new project created with AvA.\n")
            (project_path / "main.py").write_text(
                f'"""{project_name} - Main Entry Point"""\n\ndef main():\n    print("Hello from {project_name}!")\n\nif __name__ == "__main__":\n    main()\n'
            )

            success_msg = f"✅ Project '{project_name}' created!"
            if self.terminal_window and hasattr(self.terminal_window, 'stream_log_rich'):
                self.terminal_window.stream_log_rich("Application", "success", success_msg, "0")

            if self.main_window:
                # Start a new session for the new project
                self.new_session()
                self.main_window.chat_interface.add_assistant_response(
                    f"{success_msg}\n\nLocation: {project_path}\n\nReady to build! Describe what you want to create."
                )
                if hasattr(self.main_window, 'update_project_display'):
                    self.main_window.update_project_display(project_name)

            self._on_project_loaded(str(project_path))

            if self.code_viewer and hasattr(self.code_viewer, 'load_project'):
                self.code_viewer.load_project(str(project_path))

            self.project_loaded.emit(str(project_path))

        except Exception as e:
            error_msg = f"Project creation failed: {e}"
            self.logger.error(error_msg, exc_info=True)
            if self.terminal_window and hasattr(self.terminal_window, 'stream_log_rich'):
                self.terminal_window.stream_log_rich("Application", "error", f"❌ {error_msg}", "0")

    def shutdown(self):
        self.logger.info("Shutting down streamlined AvA Application...")
        if self.performance_timer: self.performance_timer.stop()
        if self.main_window: self.main_window.close()
        if self.terminal_window: self.terminal_window.close()
        if self.code_viewer: self.code_viewer.close()
        self.logger.info("Shutdown complete.")