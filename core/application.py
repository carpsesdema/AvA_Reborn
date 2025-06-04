# core/application.py - Enhanced with Conversation Context

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from PySide6.QtCore import QObject, Signal, QTimer
from PySide6.QtWidgets import QApplication

from core.llm_client import LLMClient
from core.workflow_engine import WorkflowEngine

# SIMPLIFIED: Remove complex enhanced workflow imports
# These were causing the performance issues
ENHANCED_WORKFLOW_AVAILABLE = False

# Import the components
from gui.main_window import AvAMainWindow
from windows.code_viewer import CodeViewerWindow
from gui.terminals import TerminalWindow

# Try to import RAG manager - gracefully handle if not available
try:
    from core.rag_manager import RAGManager

    RAG_MANAGER_AVAILABLE = True
except ImportError as e:
    RAG_MANAGER_AVAILABLE = False
    print(f"RAG Manager not available: {e}")


class AvAApplication(QObject):
    """
    AvA Application - Enhanced with Conversation Context
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

        # SIMPLIFIED: No enhanced workflow components
        self.use_enhanced_workflow = False

        # Application state
        self.workspace_dir = Path("./workspace")
        self.workspace_dir.mkdir(exist_ok=True)

        self.current_project = "Default Project"
        self.current_session = "Main Chat"
        self.active_workflows = {}

        # SIMPLIFIED Configuration
        self.current_config = {
            "chat_model": "Gemini: gemini-2.5-pro-preview-05-06",
            "code_model": "qwen2.5-coder:14b",
            "temperature": 0.7
        }

        # Performance monitoring timer
        self.performance_timer = QTimer()
        self.performance_timer.timeout.connect(self._update_performance_stats)
        self.performance_timer.start(5000)

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
                    '📝': '[EDIT]', '📂': '[FOLDER]', '🤖': '[AI]', '🤝': '[COLLAB]'
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
        self.logger.info("[LAUNCH] Initializing AvA Application (Enhanced Version)...")

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
        self.logger.info("[LAUNCH] Initializing enhanced components...")
        try:
            await self._initialize_rag_manager_async()
            self.logger.info(f"RAG Manager initialization complete.")

            # SIMPLIFIED: Always use standard workflow engine
            self._initialize_workflow_engine()

            self._connect_components()
            self._setup_window_behaviors()

            status = self.get_status()
            self.logger.info(f"System status after initialization: {status}")
            self.logger.info("[OK] AvA Application (Enhanced) initialized successfully.")

        except Exception as e:
            self.logger.error(f"[ERROR] Failed to initialize AvA components: {e}", exc_info=True)
            self.error_occurred.emit("async_initialization", str(e))
        finally:
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
                await self.rag_manager.async_initialize()
                self.logger.info(f"[OK] RAG manager initialized. Ready: {self.rag_manager.is_ready}")
            except Exception as e:
                self.logger.error(f"[ERROR] RAG manager initialization failed: {e}", exc_info=True)
                self.rag_manager = None
                self._on_rag_status_changed(f"RAG Init Exception: {e}", "error")
        else:
            self.logger.info("RAG services not available - running without RAG functionality.")
            self.rag_manager = None
            self._on_rag_status_changed("RAG: Not Available (Import Fail)", "grey")

    def _initialize_workflow_engine(self):
        """Initialize workflow engine with enhanced context support"""
        self.logger.info("Initializing enhanced workflow engine...")
        self.workflow_engine = WorkflowEngine(
            self.llm_client,
            self.terminal_window,
            self.code_viewer,
            self.rag_manager
        )
        self.logger.info("[OK] Enhanced workflow engine initialized.")

    def _connect_components(self):
        self.logger.info("Connecting components...")
        if self.main_window:
            # ORIGINAL connection for compatibility
            self.main_window.workflow_requested.connect(self._handle_workflow_request)

            # NEW enhanced connection with conversation context
            if hasattr(self.main_window, 'workflow_requested_with_context'):
                self.main_window.workflow_requested_with_context.connect(
                    self._handle_enhanced_workflow_request
                )

            if hasattr(self.main_window, 'new_project_requested'):
                self.main_window.new_project_requested.connect(self.create_new_project_dialog)

        # Connect workflow signals
        self.workflow_started.connect(self._on_workflow_started)
        self.workflow_completed.connect(self._on_workflow_completed)
        self.error_occurred.connect(self._on_error_occurred)

        if self.workflow_engine:
            if self.main_window:
                self.workflow_engine.workflow_started.connect(self.main_window.on_workflow_started)
                self.workflow_engine.workflow_completed.connect(self.main_window.on_workflow_completed)

            # Connect progress updates to terminal
            if self.terminal_window and hasattr(self.terminal_window, 'update_workflow_progress'):
                self.workflow_engine.workflow_progress.connect(self.terminal_window.update_workflow_progress)
                self.logger.info("[OK] Workflow progress connected to terminal")

            self.workflow_engine.file_generated.connect(self._on_file_generated)
            self.workflow_engine.project_loaded.connect(self._on_project_loaded)

        self.logger.info("[OK] Components connected.")

    def _setup_window_behaviors(self):
        if self.main_window:
            self.main_window.setWindowTitle("AvA - AI Development Assistant (Enhanced Mode)")
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
            self.terminal_window.setGeometry(screen_geo.width() - 850, screen_geo.height() - 400, 900, 700)
        if self.code_viewer:
            self.code_viewer.setGeometry(screen_geo.width() - 1000, 50, 950, 700)

    def _update_performance_stats(self):
        """Update performance statistics in terminal"""
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
                rag_info["status_text"] = "RAG: Initialization Failed (Manager None)"
        else:
            rag_info["status_text"] = "RAG: Dependencies Missing / Not Available"

        # Include performance stats
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
            "enhanced_workflow": True,  # Now always enhanced
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
        """Enhanced workflow handler with full conversation context"""
        self.logger.info(f"Enhanced workflow request: {user_prompt[:100]}...")
        self.logger.info(f"Conversation context: {len(conversation_history)} messages")

        current_status = self.get_status()

        # Check for LLM availability
        if not current_status["llm_models"] or current_status["llm_models"] == ["LLM Client not init"] or \
                current_status["llm_models"] == ["No LLM services available"]:
            error_msg = "No LLM services available. Please configure API keys."
            if self.terminal_window:
                self.terminal_window.log(f"[ERROR] Enhanced Workflow Halted: {error_msg}")
            self.error_occurred.emit("llm_unavailable", error_msg)
            return

        if not self.workflow_engine:
            error_msg = "Workflow engine not initialized. Cannot process request."
            if self.terminal_window:
                self.terminal_window.log(f"[ERROR] Enhanced Workflow Halted: {error_msg}")
            self.error_occurred.emit("workflow_engine_unavailable", error_msg)
            return

        self._open_terminal()
        workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.active_workflows[workflow_id] = {
            "prompt": user_prompt,
            "conversation_history": conversation_history,
            "start_time": datetime.now(),
            "status": "running",
            "enhanced": True
        }

        self.workflow_started.emit(user_prompt)

        try:
            # ENHANCED: Use context-aware workflow execution
            self.logger.info("Starting ENHANCED workflow with conversation context...")
            self.workflow_engine.execute_enhanced_workflow(user_prompt, conversation_history)

        except Exception as e:
            self.logger.error(f"Enhanced workflow execution failed: {e}", exc_info=True)
            self.error_occurred.emit("enhanced_workflow_execution_error", str(e))
            self.active_workflows[workflow_id]["status"] = "failed"
            self.active_workflows[workflow_id]["error"] = str(e)

    def _handle_workflow_request(self, user_prompt: str):
        """Original workflow handler for compatibility"""
        self.logger.info(f"Standard workflow request received: {user_prompt[:100]}...")

        # Use enhanced workflow with empty conversation history
        self._handle_enhanced_workflow_request(user_prompt, [])

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
        if self.main_window and hasattr(self.main_window, 'update_project_display'):
            self.main_window.update_project_display(self.current_project)
        self._open_code_viewer()

    def _on_workflow_started(self, prompt: str):
        self.logger.info(f"[OK] Workflow started: {prompt}")
        if self.main_window and hasattr(self.main_window, 'on_workflow_started'):
            self.main_window.on_workflow_started(prompt)

    def _on_workflow_completed(self, result: dict):
        self.logger.info(f"[OK] Workflow completed: {result.get('project_name', 'N/A')}")
        for wf_id, wf_data in self.active_workflows.items():
            if wf_data["status"] == "running":
                wf_data["status"] = "completed" if result.get("success") else "failed"
                wf_data["end_time"] = datetime.now()
                break
        if self.main_window and hasattr(self.main_window, 'on_workflow_completed'):
            self.main_window.on_workflow_completed(result)

    def _on_error_occurred(self, component: str, error_message: str):
        self.logger.error(f"[ERROR] App Error in {component}: {error_message}")
        if self.main_window and hasattr(self.main_window, 'on_app_error_occurred'):
            self.main_window.on_app_error_occurred(component, error_message)

    def _on_rag_status_changed(self, status_text: str, color: str):
        self.logger.info(f"RAG Status Update: {status_text} (Color: {color})")
        self.rag_status_changed.emit(status_text, color)

    def _on_rag_upload_completed(self, collection_id: str, files_processed: int):
        msg = f"RAG: {files_processed} files added to {collection_id}"
        self.logger.info(f"RAG Upload Completed: {msg}")
        self.rag_status_changed.emit(msg, "#4ade80")

    def update_configuration(self, config_updates: dict):
        self.current_config.update(config_updates)
        self.logger.info(f"Configuration updated: {config_updates}")
        if self.main_window and hasattr(self.main_window, '_update_initial_ui_status'):
            self.main_window._update_initial_ui_status()

    def create_new_project_dialog(self):
        """Complete project creation dialog implementation"""
        self.logger.info("Request to create a new project received.")
        if self.terminal_window:
            self.terminal_window.log("✨ New Project creation requested.")

        from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

        if not self.main_window:
            self.logger.error("Main window not available for project creation dialog.")
            return

        try:
            # Step 1: Get project name from user
            project_name, ok = QInputDialog.getText(
                self.main_window,
                'New Project',
                'Enter project name:',
                text='my_new_project'
            )

            if not ok or not project_name.strip():
                self.logger.info("Project creation cancelled by user.")
                if self.terminal_window:
                    self.terminal_window.log("❌ Project creation cancelled by user.")
                return

            project_name = project_name.strip()

            # Step 2: Let user select project directory
            base_dir = QFileDialog.getExistingDirectory(
                self.main_window,
                'Select Directory for New Project',
                str(self.workspace_dir)
            )

            if not base_dir:
                self.logger.info("Project creation cancelled - no directory selected.")
                if self.terminal_window:
                    self.terminal_window.log("❌ Project creation cancelled - no directory selected.")
                return

            # Step 3: Create project directory
            project_path = Path(base_dir) / project_name

            if project_path.exists():
                reply = QMessageBox.question(
                    self.main_window,
                    'Directory Exists',
                    f'Directory "{project_name}" already exists in the selected location.\n\nDo you want to use it anyway?',
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )

                if reply == QMessageBox.StandardButton.No:
                    self.logger.info("Project creation cancelled - directory exists and user declined.")
                    if self.terminal_window:
                        self.terminal_window.log("❌ Project creation cancelled - directory exists.")
                    return
            else:
                try:
                    project_path.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"Created project directory: {project_path}")
                except Exception as e:
                    error_msg = f"Failed to create project directory: {e}"
                    self.logger.error(error_msg)
                    QMessageBox.critical(self.main_window, 'Error', error_msg)
                    return

            # Step 4: Update application state
            self.current_project = project_name
            self.workspace_dir = project_path

            # Step 5: Create basic project structure
            try:
                # Create basic project files
                (project_path / "README.md").write_text(f"# {project_name}\n\nA new project created with AvA.\n")
                (project_path / ".gitignore").write_text("__pycache__/\n*.pyc\n*.pyo\n*.egg-info/\n.env\n")

                # Create basic Python structure if it looks like a Python project
                if not any((project_path / ext).exists() for ext in ["package.json", "pom.xml", "Cargo.toml"]):
                    (project_path / "main.py").write_text(
                        f'"""\n{project_name}\n\nMain application entry point.\n"""\n\ndef main():\n    print(f"Hello from {project_name}!")\n\nif __name__ == "__main__":\n    main()\n')
                    (project_path / "requirements.txt").write_text("# Add your dependencies here\n")

                self.logger.info(f"Created basic project structure in {project_path}")

            except Exception as e:
                self.logger.warning(f"Failed to create basic project structure: {e}")
                # Continue anyway since directory was created successfully

            # Step 6: Update UI and notify user
            success_msg = f"✅ Project '{project_name}' created successfully!"
            self.logger.info(success_msg)

            if self.terminal_window:
                self.terminal_window.log(success_msg)
                self.terminal_window.log(f"📁 Project location: {project_path}")

            if self.main_window:
                self.main_window.chat_interface.chat_display.add_assistant_message(
                    f"{success_msg}\n\nProject location: {project_path}\n\nYou can now start building by describing what you want to create!"
                )

                # Update project display
                if hasattr(self.main_window, 'update_project_display'):
                    self.main_window.update_project_display(project_name)

            # Step 7: Open code viewer and load project
            self._open_code_viewer()
            if self.code_viewer and hasattr(self.code_viewer, 'load_project'):
                self.code_viewer.load_project(str(project_path))

            # Emit project loaded signal
            self.project_loaded.emit(str(project_path))

            # Step 8: Suggest next steps
            if self.main_window:
                self.main_window.chat_interface.chat_display.add_assistant_message(
                    "💡 **Next steps:**\n"
                    "• Describe what you want to build (e.g., 'Create a calculator app with PySide6')\"• I'll use my enhanced AI specialists with your conversation context\n"
                    "• Use the Code Viewer to explore generated files\n"
                    "• Use the Terminal to monitor progress"
                )

        except Exception as e:
            error_msg = f"Unexpected error during project creation: {e}"
            self.logger.error(error_msg, exc_info=True)

            if self.terminal_window:
                self.terminal_window.log(f"❌ {error_msg}")

            QMessageBox.critical(
                self.main_window if self.main_window else None,
                'Project Creation Error',
                error_msg
            )

    def shutdown(self):
        self.logger.info("Shutting down AvA Application...")

        # Stop performance monitoring
        if self.performance_timer:
            self.performance_timer.stop()

        # Close windows
        if self.main_window:
            self.main_window.close()
        if self.terminal_window:
            self.terminal_window.close()
        if self.code_viewer:
            self.code_viewer.close()

        self.logger.info("AvA Application shutdown complete.")