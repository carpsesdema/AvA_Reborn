# core/application.py - Streamlined for Fast Professional Results

import asyncio
import logging
import json
import sys
import shutil
import subprocess  # Import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
# noinspection PyUnresolvedReferences
from functools import partial

# Import the necessary Qt components
from PySide6.QtCore import QObject, Signal, QTimer, Qt, Q_ARG, QMetaObject
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QInputDialog, QWidget

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


# ============================================================================
class AvAApplication(QObject):
    fully_initialized_signal = Signal()
    workflow_started = Signal(str)
    workflow_completed = Signal(dict)
    error_occurred = Signal(str, str)
    rag_status_changed = Signal(str, str)
    project_loaded = Signal(str)

    # --- Signals for thread-safe terminal updates ---
    terminal_command_to_run = Signal(str)
    terminal_output = Signal(str)
    terminal_error = Signal(str)
    terminal_system_message = Signal(str)
    terminal_focus_requested = Signal()
    chat_message_received = Signal(str)

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
        asyncio.create_task(self._initialize_async_components())
        return True

    def _initialize_core_services(self):
        self.logger.info("Initializing core services...")
        self.llm_client = EnhancedLLMClient()
        self.logger.info(f"Available LLM models: {self.llm_client.get_available_models()}")
        print("Assigning default models for consolidated roles...")
        self.execution_engine = ExecutionEngine(project_state_manager=None, terminal=self.terminal_window)
        self.logger.info("[OK] Core services initialized.")

    async def _initialize_async_components(self):
        self.logger.info("Starting async component initialization...")
        self.logger.info("[START] Initializing streamlined components...")
        await self._initialize_rag_manager()
        self._initialize_workflow_engine()
        self._initialize_execution_engine()
        self._connect_components()
        self._setup_window_behaviors()
        self.logger.info("Async component initialization completed.")
        self.fully_initialized_signal.emit()
        self.logger.info("[OK] Streamlined AvA Application initialized successfully.")

    async def _initialize_rag_manager(self):
        if RAG_MANAGER_AVAILABLE:
            try:
                self.logger.info("Attempting to initialize RAG manager...")
                self.rag_manager = RAGManager()

                if hasattr(self.rag_manager, 'status_changed'):
                    self.rag_manager.status_changed.connect(self._on_rag_status_changed)
                elif hasattr(self.rag_manager, 'rag_status_changed'):
                    self.rag_manager.rag_status_changed.connect(self._on_rag_status_changed)

                if hasattr(self.rag_manager, 'upload_completed'):
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

        if self.code_viewer and hasattr(self.code_viewer, 'interactive_terminal'):
            terminal = self.code_viewer.interactive_terminal
            terminal.force_run_requested.connect(
                lambda: asyncio.create_task(asyncio.to_thread(self.run_project_in_terminal))
            )
            self.terminal_command_to_run.connect(terminal.append_command)
            self.terminal_output.connect(terminal.append_output)
            self.terminal_error.connect(terminal.append_error)
            self.terminal_system_message.connect(terminal.append_system_message)
            self.terminal_focus_requested.connect(self.code_viewer.focus_terminal_tab)

        if self.main_window:
            self.chat_message_received.connect(self.main_window.chat_interface.add_assistant_response)

        if self.workflow_engine and self.main_window:
            # --- THE FIX ---
            self.workflow_engine.workflow_completed.connect(self._on_workflow_completed)
        self.workflow_started.connect(self._on_workflow_started)
        self.workflow_completed.connect(self._on_workflow_completed)
        self.error_occurred.connect(self._on_error_occurred)
        if self.workflow_engine:
            self.workflow_engine.file_generated.connect(self._on_file_generated)
            self.workflow_engine.project_loaded.connect(self.project_loaded.emit)
        self.project_loaded.connect(self._on_project_loaded)
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

    def _find_system_python(self) -> str:
        # This function is unchanged
        import subprocess, os
        ava_venv_marker, current_executable = ".venv", sys.executable
        potential_pythons = []
        if sys.platform == "win32":
            potential_pythons = ["python", "py", "py -3", r"C:\Python312\python.exe", r"C:\Python311\python.exe"]
            user_appdata = os.environ.get('LOCALAPPDATA', '')
            if user_appdata:
                for version in ['312', '311']: potential_pythons.append(
                    f"{user_appdata}\\Programs\\Python\\Python{version}\\python.exe")
        else:
            potential_pythons = ["python3", "python", "/usr/bin/python3"]
        for python_cmd in potential_pythons:
            try:
                cmd_parts = python_cmd.split()
                result = subprocess.run(cmd_parts + ["--version"], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    result2 = subprocess.run(cmd_parts + ["-c", "import sys; print(sys.executable)"],
                                             capture_output=True, text=True, timeout=10)
                    if result2.returncode == 0:
                        python_path = result2.stdout.strip()
                        if ava_venv_marker not in python_path and python_path != current_executable:
                            return python_cmd
            except(Exception):
                continue
        return "python"

    def run_project_in_terminal(self, project_path_str: Optional[str] = None):
        """
        Synchronous, smart, and snappy method to run the project.
        """
        path_to_run = Path(project_path_str) if project_path_str else self.current_project_path
        path_to_run = path_to_run.resolve()

        def run_command_and_wait(command_parts: List[str], timeout_override: int = None) -> bool:
            command_str = ' '.join(f'"{part}"' if ' ' in part else part for part in command_parts)
            self.terminal_command_to_run.emit(command_str)
            try:
                result = subprocess.run(
                    command_parts, capture_output=True, text=True, cwd=str(path_to_run),
                    timeout=timeout_override or 300, check=False, encoding='utf-8', errors='ignore'
                )
                if result.stdout: self.terminal_output.emit(result.stdout)
                if result.stderr: self.terminal_error.emit(result.stderr)
                self.terminal_system_message.emit(f"Process finished with exit code {result.returncode}")
                return result.returncode == 0
            except subprocess.TimeoutExpired:
                msg = f"Command '{command_str[:50]}...' timed out!"
                self.terminal_error.emit(msg)
                self.chat_message_received.emit(f"❌ {msg}")
                return False
            except Exception as e:
                msg = f"Command '{command_str[:50]}...' failed: {e}"
                self.terminal_error.emit(msg)
                self.chat_message_received.emit(f"❌ {msg}")
                return False

        self.chat_message_received.emit(f"🚀 Preparing to run `{path_to_run.name}`...")
        self.terminal_focus_requested.emit()

        main_file = path_to_run / "main.py"
        req_file = path_to_run / "requirements.txt"
        venv_dir = path_to_run / "venv"

        if sys.platform == "win32":
            venv_python = venv_dir / "Scripts" / "python.exe"
        else:
            venv_python = venv_dir / "bin" / "python"

        system_python_cmd = self._find_system_python()
        self.chat_message_received.emit(f"Using system Python: {system_python_cmd}")

        if req_file.exists():
            if not venv_dir.exists() or not venv_python.exists():
                self.chat_message_received.emit("   - Virtual environment not found. Creating...")
                venv_success = run_command_and_wait(system_python_cmd.split() + ['-m', 'venv', 'venv'])
                if not venv_success:
                    self.chat_message_received.emit("❌ Failed to create virtual environment. Aborting run.")
                    return
                self.chat_message_received.emit("   - Installing dependencies...")
                pip_install_success = run_command_and_wait(
                    [str(venv_python), "-m", "pip", "install", "-r", req_file.name])
                if not pip_install_success:
                    self.chat_message_received.emit("❌ Failed to install dependencies. Check terminal for errors.")
            else:
                self.chat_message_received.emit("   - Virtual environment found. Checking dependencies...")
                check_script = "import pkg_resources;" + "".join(
                    f"pkg_resources.require('{line.strip()}');" for line in req_file.read_text().splitlines() if
                    line.strip() and not line.strip().startswith('#'))
                reqs_met = run_command_and_wait([str(venv_python), "-c", check_script])

                if not reqs_met:
                    self.chat_message_received.emit("   - Dependencies missing or outdated. Installing...")
                    pip_install_success = run_command_and_wait(
                        [str(venv_python), "-m", "pip", "install", "-r", req_file.name])
                    if not pip_install_success:
                        self.chat_message_received.emit("❌ Failed to install dependencies. Check terminal for errors.")
                else:
                    self.chat_message_received.emit("   - ✅ Dependencies are up to date!")

        if not main_file.exists():
            self.chat_message_received.emit("❌ No main.py file found!")
            return

        self.chat_message_received.emit("   - Executing main script...")
        python_to_use = str(venv_python) if venv_python.exists() else self._find_system_python()

        execution_success = run_command_and_wait([python_to_use, main_file.name])

        if execution_success:
            self.chat_message_received.emit("✅ Project execution completed successfully!")
        else:
            self.chat_message_received.emit("❌ Project execution encountered errors. Check terminal output above.")

    def _initialize_execution_engine(self):
        self.execution_engine = ExecutionEngine(project_state_manager=None, terminal=self.terminal_window)
        self.logger.info("[OK] Execution engine initialized.")

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
        if self.main_window:
            QMetaObject.invokeMethod(self.main_window, 'update_project_display', Qt.QueuedConnection,
                                     Q_ARG(str, self.current_project))
        if self.code_viewer:
            QMetaObject.invokeMethod(self.code_viewer, 'load_project', Qt.QueuedConnection, Q_ARG(str, project_path))

    def _on_workflow_started(self, prompt: str):
        self.logger.info(f"Workflow started: {prompt}")

    def _on_workflow_completed(self, result: dict):
        self.logger.info(f"Workflow completed: {result.get('project_name', 'N/A')}")
        self.workflow_completed.emit(result)

    def _on_error_occurred(self, component: str, error_message: str):
        self.logger.error(f"Error in {component}: {error_message}")

    def _on_rag_status_changed(self, status_text: str, color: str):
        if self.main_window:
            QMetaObject.invokeMethod(self.main_window, 'update_rag_status_display', Qt.QueuedConnection,
                                     Q_ARG(str, status_text), Q_ARG(str, color))

    def _on_rag_upload_completed(self, collection_id: str, files_processed: int):
        msg = f"RAG: Added {files_processed} files to {collection_id}"
        self.logger.info(msg)
        if self.main_window:
            QMetaObject.invokeMethod(self.main_window, 'update_rag_status_display', Qt.QueuedConnection,
                                     Q_ARG(str, msg), Q_ARG(str, "success"))

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
        self.project_loaded.emit(str(project_path))

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

    def _handle_workflow_request(self, user_prompt: str):
        self._handle_enhanced_workflow_request(user_prompt, [])

    def _handle_sidebar_action(self, action: str):
        if action == "run_project":
            asyncio.create_task(asyncio.to_thread(self.run_project_in_terminal))
        elif action == "open_terminal":
            self._open_terminal()
        elif action == "open_code_viewer":
            self._open_code_viewer()
        else:
            self.logger.warning(f"Sidebar action '{action}' not implemented yet.")

    def shutdown(self):
        self.logger.info("Shutting down...")
        if self.performance_timer: self.performance_timer.stop()
        if self.main_window: self.main_window.close()
        if self.terminal_window: self.terminal_window.close()
        if self.code_viewer: self.code_viewer.close()