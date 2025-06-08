# core/application.py - Streamlined for Fast Professional Results

import asyncio
import logging
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
# noinspection PyUnresolvedReferences
from functools import partial

from PySide6.QtCore import QObject, Signal, QTimer
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QInputDialog

from core.llm_client import EnhancedLLMClient
from core.enhanced_workflow_engine import EnhancedWorkflowEngine
from core.execution_engine import ExecutionEngine

from gui.main_window import AvAMainWindow
from gui.code_viewer import CodeViewerWindow
from gui.terminals import TerminalWindow

try:
    from core.rag_manager import RAGManager

    RAG_MANAGER_AVAILABLE = True
except ImportError as e:
    RAG_MANAGER_AVAILABLE = False
    print(f"RAG Manager not available: {e}")


class AvAApplication(QObject):
    fully_initialized_signal = Signal()
    workflow_started = Signal(str)
    workflow_completed = Signal(dict)
    error_occurred = Signal(str, str)
    rag_status_changed = Signal(str, str)
    project_loaded = Signal(str)

    def __init__(self):
        super().__init__()
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

        self.llm_client = None
        self.main_window = None
        self.terminal_window = None
        self.code_viewer = None
        self.workflow_engine = None
        self.rag_manager = None
        self.execution_engine = None

        self.workspace_dir = Path("./workspace")
        self.workspace_dir.mkdir(exist_ok=True)
        self.current_project = "Default Project"
        self.current_project_path = self.workspace_dir
        self.current_session = "Main Chat"
        self.active_workflows = {}

        self.current_config = {"chat_model": "gemini-2.5-pro-preview-06-05", "code_model": "qwen2.5-coder:14b",
                               "temperature": 0.7}
        self.performance_timer = QTimer()
        self.performance_timer.timeout.connect(self._update_performance_stats)
        self.performance_timer.start(5000)

    def _setup_logging(self):
        log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True)

        class SafeFormatter(logging.Formatter):
            def format(self, record):
                msg = super().format(record)
                emoji_replacements = {'🚀': '[START]', '✅': '[OK]', '❌': '[ERROR]', '⚠️': '[WARN]', '🧠': '[AI]',
                                      '📊': '[STATS]', '🔧': '[TOOL]', '📄': '[FILE]'}
                for emoji, replacement in emoji_replacements.items(): msg = msg.replace(emoji, replacement)
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
            self._initialize_workflow_engine()
            self.execution_engine = ExecutionEngine(None,
                                                    self.code_viewer.interactive_terminal if self.code_viewer else None)
            self.logger.info("[OK] Execution engine initialized.")
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
        self.logger.info("Initializing streamlined workflow engine...")
        self.workflow_engine = EnhancedWorkflowEngine(self.llm_client, self.terminal_window, self.code_viewer,
                                                      self.rag_manager)
        self.logger.info("[OK] Streamlined workflow engine initialized.")

    def _connect_components(self):
        self.logger.info("Connecting components...")
        if self.main_window:
            self.main_window.workflow_requested.connect(self._handle_workflow_request)
            if hasattr(self.main_window, 'workflow_requested_with_context'):
                self.main_window.workflow_requested_with_context.connect(self._handle_enhanced_workflow_request)
            self.main_window.new_project_requested.connect(self.create_new_project_dialog)
            self.main_window.load_project_requested.connect(self.load_existing_project_dialog)
            if hasattr(self.main_window.sidebar, 'action_triggered'):
                self.main_window.sidebar.action_triggered.connect(self._handle_sidebar_action)
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
        if self.main_window: self.main_window.setWindowTitle("AvA - Fast Professional AI Development")
        if self.terminal_window: self.terminal_window.setWindowTitle("AvA - Workflow Terminal")
        if self.code_viewer: self.code_viewer.setWindowTitle("AvA - Code Viewer & IDE")
        self._position_windows()
        self.logger.info("[OK] Window behaviors set up.")

    def _position_windows(self):
        screen_geo = QApplication.primaryScreen().geometry()
        if self.main_window: self.main_window.setGeometry(100, 50, 1400, 900)
        if self.terminal_window: self.terminal_window.setGeometry(screen_geo.width() - 920, screen_geo.height() - 450,
                                                                  900, 400)
        if self.code_viewer: self.code_viewer.setGeometry(screen_geo.width() - 1000, 50, 950, 700)

    def _update_performance_stats(self):
        pass

    def get_status(self) -> Dict[str, Any]:
        llm_models_list = self.llm_client.get_available_models() if self.llm_client else ["LLM Client not init"]
        rag_info = {"ready": False, "status_text": "RAG: Not Initialized", "available": RAG_MANAGER_AVAILABLE,
                    "collections": {}}
        if RAG_MANAGER_AVAILABLE:
            if self.rag_manager and hasattr(self.rag_manager, 'is_ready'):
                rag_info["ready"] = self.rag_manager.is_ready
                rag_info["status_text"] = self.rag_manager.current_status
                if hasattr(self.rag_manager, 'get_collection_info'): rag_info[
                    "collections"] = self.rag_manager.get_collection_info()
            else:
                rag_info["status_text"] = "RAG: Initialization Failed"
        else:
            rag_info["status_text"] = "RAG: Dependencies Missing"
        return {"ready": self.workflow_engine is not None, "streamlined": True, "llm_models": llm_models_list,
                "workspace": str(self.workspace_dir), "current_project": self.current_project,
                "current_session": self.current_session, "configuration": self.current_config, "rag": rag_info,
                "active_workflows": len(self.active_workflows)}

    def _handle_enhanced_workflow_request(self, user_prompt: str, conversation_history: List[Dict]):
        self.logger.info(f"Fast workflow request: {user_prompt[:100]}...")
        if not self.workflow_engine:
            self.error_occurred.emit("workflow", "Engine not available.")
            return
        self._open_terminal()
        asyncio.create_task(self.workflow_engine.execute_enhanced_workflow(user_prompt, conversation_history))

    def run_project_in_terminal(self, project_path: Optional[Path] = None):
        """Executes the specified project, creating a venv and installing dependencies if necessary."""
        path_to_run = (project_path or self.current_project_path).resolve()

        if str(path_to_run) == str(self.workspace_dir.resolve()) and self.current_project == "Default Project":
            self.main_window.chat_interface.add_assistant_response("Please load or create a project first!")
            return

        if not self.code_viewer or not hasattr(self.code_viewer, 'interactive_terminal'):
            self.error_occurred.emit("execution", "Interactive terminal is not available.")
            return

        main_file = path_to_run / "main.py"
        req_file = path_to_run / "requirements.txt"

        if not main_file.exists():
            self.main_window.chat_interface.add_assistant_response(
                f"Cannot find `main.py` in '{path_to_run.name}' to run.")
            return

        self.main_window.chat_interface.add_assistant_response(f"🚀 Preparing to run `{path_to_run.name}`...")
        self._open_code_viewer()
        self.code_viewer.main_tabs.setCurrentWidget(self.code_viewer.interactive_terminal)
        self.code_viewer.interactive_terminal.set_working_directory(str(path_to_run))

        # --- VENV and Dependency Logic ---
        commands = []
        # Platform-specific venv Python executable path
        if sys.platform == "win32":
            venv_python = path_to_run / "venv" / "Scripts" / "python.exe"
            # Use '&&' to chain commands on Windows
            joiner = " && "
        else:
            venv_python = path_to_run / "venv" / "bin" / "python"
            # Use '&&' for Linux/macOS as well
            joiner = " && "

        if req_file.exists():
            self.main_window.chat_interface.add_assistant_response(
                "   - Found requirements.txt. Setting up virtual environment...")
            # 1. Create venv if it doesn't exist. Use the base python interpreter.
            commands.append(f'"{sys.executable}" -m venv venv')
            # 2. Install dependencies using the venv's python.
            commands.append(f'"{venv_python}" -m pip install -r "{req_file.name}"')

        # 3. Always add the final command to run the script using the venv python if it exists/was created
        final_run_command = f'"{venv_python}" "{main_file.name}"'
        if not req_file.exists():  # If no reqs, use the base python
            final_run_command = f'"{sys.executable}" "{main_file.name}"'

        commands.append(final_run_command)

        # Join all commands into a single string for the shell to execute
        full_command = joiner.join(commands)

        self.logger.info(f"Executing chained command: {full_command}")
        self.code_viewer.interactive_terminal.input_line.setText(full_command)
        self.code_viewer.interactive_terminal.run_command()

    def _handle_workflow_request(self, user_prompt: str):
        self._handle_enhanced_workflow_request(user_prompt, [])

    def _handle_sidebar_action(self, action: str):
        if action == "run_project":
            self.run_project_in_terminal()
        elif action == "open_terminal":
            self._open_terminal()
        elif action == "open_code_viewer":
            self._open_code_viewer()
        else:
            self.logger.warning(f"Sidebar action '{action}' not implemented yet.")

    def _open_terminal(self):
        if self.terminal_window: self.terminal_window.show(); self.terminal_window.raise_()

    def _open_code_viewer(self):
        if self.code_viewer: self.code_viewer.show(); self.code_viewer.raise_()

    def _on_file_generated(self, file_path: str):
        if self.code_viewer: self.code_viewer.auto_open_file(file_path)

    def _on_project_loaded(self, project_path: str):
        self.logger.info(f"Project loaded: {project_path}")
        self.current_project_path = Path(project_path)
        self.current_project = self.current_project_path.name
        if self.main_window: self.main_window.update_project_display(self.current_project)
        self._open_code_viewer()
        if self.code_viewer: self.code_viewer.load_project(project_path)

    def _on_workflow_started(self, prompt: str):
        self.logger.info(f"Workflow started: {prompt}")

    def _on_workflow_completed(self, result: dict):
        self.logger.info(f"Workflow completed: {result.get('project_name', 'N/A')}")
        if result.get("success"):
            new_project_dir_str = result.get("project_dir")
            if new_project_dir_str:
                self._on_project_loaded(new_project_dir_str)
                # No need for logic here anymore, MainWindow handles it

    def _on_error_occurred(self, component: str, error_message: str):
        self.logger.error(f"Error in {component}: {error_message}")

    def _on_rag_status_changed(self, status_text: str, color: str):
        if self.main_window: self.main_window.update_rag_status_display(status_text, color)

    def _on_rag_upload_completed(self, collection_id: str, files_processed: int):
        msg = f"RAG: Added {files_processed} files to {collection_id}"
        self.logger.info(msg)
        if self.main_window: self.main_window.update_rag_status_display(msg, "success")

    def create_new_project_dialog(self):
        self.logger.info("Creating new project...")
        if not self.main_window: return
        project_name, ok = QInputDialog.getText(self.main_window, 'New Project', 'Enter project name:',
                                                text='my-ava-project')
        if not (ok and project_name.strip()): return
        base_dir_str = QFileDialog.getExistingDirectory(self.main_window, 'Select Directory', str(self.workspace_dir))
        if not base_dir_str: return
        project_path = Path(base_dir_str) / project_name.strip()
        if project_path.exists():
            reply = QMessageBox.question(self.main_window, 'Directory Exists',
                                         f'Directory "{project_name.strip()}" exists. Use it anyway?',
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No: return
        else:
            project_path.mkdir(parents=True, exist_ok=True)
        self.current_project, self.current_project_path = project_name.strip(), project_path
        (project_path / "main.py").write_text(f'print("Hello from {project_name.strip()}!")')
        self._on_project_loaded(str(project_path))

    def load_existing_project_dialog(self):
        if not self.main_window: return
        folder_path_str = QFileDialog.getExistingDirectory(self.main_window, "Load Existing Project",
                                                           str(self.workspace_dir))
        if not folder_path_str: return
        self.logger.info(f"User selected project: {folder_path_str}")
        if self.workflow_engine:
            self.main_window.chat_interface.add_workflow_status(f"🧠 Analyzing '{Path(folder_path_str).name}'...")
            self._open_terminal()
            asyncio.create_task(self.workflow_engine.execute_analysis_workflow(folder_path_str))
        else:
            self.error_occurred.emit("workflow_engine", "Engine not available for analysis.")

    def shutdown(self):
        self.logger.info("Shutting down...")
        if self.performance_timer: self.performance_timer.stop()
        if self.main_window: self.main_window.close()
        if self.terminal_window: self.terminal_window.close()
        if self.code_viewer: self.code_viewer.close()