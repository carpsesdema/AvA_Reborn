# core/application.py - V2.2 with full implementations

import asyncio
import logging
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from PySide6.QtCore import QObject, Signal, QTimer, Slot
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QInputDialog

from core.llm_client import EnhancedLLMClient
from core.enhanced_workflow_engine import EnhancedWorkflowEngine
from core.project_state_manager import ProjectStateManager

# Import our UI components
from gui.main_window import AvAMainWindow
from gui.code_viewer import CodeViewerWindow
from gui.terminals import StreamingTerminal
from gui.interactive_terminal import InteractiveTerminal

try:
    from core.rag_manager import RAGManager

    RAG_MANAGER_AVAILABLE = True
except ImportError as e:
    RAG_MANAGER_AVAILABLE = False
    print(f"RAG Manager not available: {e}")


class AvAApplication(QObject):
    """
    AvA Application - The central coordinator for all operations.
    """
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
        self.code_viewer = None
        self.workflow_engine = None
        self.rag_manager = None
        self.streaming_terminal = None

        self.workspace_dir = Path("./workspace")
        self.workspace_dir.mkdir(exist_ok=True)
        self.current_project = "Default Project"
        self.current_project_path = self.workspace_dir
        self.current_session = "Main Chat"
        self.active_workflows = {}
        self.current_config = {
            "chat_model": "gemini-1.5-pro-preview-0409",
            "code_model": "deepseek-coder",
        }

    def _setup_logging(self):
        log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True)

        class SafeFormatter(logging.Formatter):
            def format(self, record):
                msg = super().format(record)
                emoji_replacements = {
                    '🚀': '[START]', '✅': '[OK]', '❌': '[ERROR]', '⚠️': '[WARN]',
                    '🧠': '[AI]', '▶️': '[RUN]', '📂': '[FILE]', '📄': '[DOC]'
                }
                safe_msg = msg
                for emoji, replacement in emoji_replacements.items():
                    safe_msg = safe_msg.replace(emoji, replacement)
                return safe_msg

        formatter = SafeFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(log_dir / "ava.log", encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

    async def initialize(self):
        self.logger.info("🚀 Initializing AvA Application...")
        self.main_window = AvAMainWindow(ava_app=self)
        self.code_viewer = CodeViewerWindow()
        self.streaming_terminal = StreamingTerminal()
        self._initialize_core_services()
        self.main_window.show()
        await asyncio.sleep(0.01)
        await self.async_initialize_components()

    async def async_initialize_components(self):
        self.logger.info("Initializing async components...")
        try:
            await self._initialize_rag_manager_async()
            self._initialize_workflow_engine()
            self._connect_components()
            self.fully_initialized_signal.emit()
            self.logger.info("✅ AvA Application initialized successfully.")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize components: {e}", exc_info=True)
            self.error_occurred.emit("async_initialization", str(e))

    def _initialize_core_services(self):
        self.logger.info("Initializing core services...")
        self.llm_client = EnhancedLLMClient()

    async def _initialize_rag_manager_async(self):
        if RAG_MANAGER_AVAILABLE:
            try:
                self.rag_manager = RAGManager()
                await self.rag_manager.async_initialize()
                self.rag_manager.status_changed.connect(self.rag_status_changed.emit)
            except Exception as e:
                self.logger.error(f"RAG manager initialization failed: {e}", exc_info=True)
                self.rag_manager = None

    def _initialize_workflow_engine(self):
        self.workflow_engine = EnhancedWorkflowEngine(
            self.llm_client, self.streaming_terminal,
            self.code_viewer, self.rag_manager
        )

    def _connect_components(self):
        self.logger.info("Connecting components...")
        if self.main_window:
            self.main_window.workflow_requested_with_context.connect(self._handle_enhanced_workflow_request)
            self.main_window.new_project_requested.connect(self.create_new_project_dialog)
            self.main_window.load_project_requested.connect(self.load_existing_project_dialog)
            self.main_window.sidebar.action_triggered.connect(self._handle_sidebar_action)

        if self.workflow_engine and self.main_window:
            self.workflow_engine.workflow_completed.connect(self.main_window.on_workflow_completed)

        if self.workflow_engine:
            self.workflow_engine.file_generated.connect(self._on_file_generated)
            self.workflow_engine.project_loaded.connect(self._on_project_loaded)

        if self.code_viewer:
            self.code_viewer.run_project_requested.connect(self.run_current_project)

        self.error_occurred.connect(self._on_error_occurred)
        self.logger.info("✅ Components connected.")

    @Slot()
    def run_current_project(self):
        if not self.current_project_path or self.current_project == "Default Project":
            self._terminal_log("error", "No project loaded to run.")
            return
        asyncio.create_task(self._run_project_sequence())

    async def _run_project_sequence(self):
        project_path = self.current_project_path
        terminal = self.code_viewer.terminal
        terminal.clear_terminal()
        self._terminal_log("info", f"▶️ Starting execution for project: {project_path.name}")
        try:
            venv_path = project_path / 'venv'
            if not await self._ensure_venv(venv_path): return
            requirements_file = project_path / 'requirements.txt'
            if not await self._install_requirements(requirements_file, venv_path):
                self._terminal_log("warning", "Continuing despite requirement installation issues.")
            if not await self._execute_main_file(project_path, venv_path):
                self._terminal_log("error", "Could not find or execute a main entry point.")
        except Exception as e:
            self._terminal_log("error", f"An unexpected error occurred during project execution: {e}")
            self.logger.error("Project execution sequence failed", exc_info=True)

    async def _run_command_in_terminal(self, command: List[str], cwd: Path) -> bool:
        self._terminal_log("command", f"$ {' '.join(command)}")
        process = await asyncio.create_subprocess_exec(*command, cwd=cwd, stdout=subprocess.PIPE,
                                                       stderr=subprocess.PIPE)

        async def read_stream(stream, log_func):
            while True:
                line = await stream.readline()
                if not line: break
                log_func(line.decode(errors='ignore').strip())

        await asyncio.gather(
            read_stream(process.stdout, lambda line: self._terminal_log("stdout", line)),
            read_stream(process.stderr, lambda line: self._terminal_log("stderr", line))
        )
        await process.wait()
        if process.returncode != 0:
            self._terminal_log("error", f"Command failed with exit code {process.returncode}")
            return False
        self._terminal_log("success", "Command completed successfully.")
        return True

    async def _ensure_venv(self, venv_path: Path) -> bool:
        if venv_path.exists() and (venv_path / 'pyvenv.cfg').exists():
            self._terminal_log("info", "Virtual environment found.")
            return True
        self._terminal_log("info", "Virtual environment not found. Creating...")
        command = [sys.executable, '-m', 'venv', str(venv_path)]
        return await self._run_command_in_terminal(command, venv_path.parent)

    async def _install_requirements(self, req_file: Path, venv_path: Path) -> bool:
        if not req_file.exists():
            self._terminal_log("info", "No requirements.txt found. Skipping dependency installation.")
            return True
        self._terminal_log("info", "requirements.txt found. Installing dependencies...")
        pip_executable = str(venv_path / 'Scripts' / 'pip') if sys.platform == 'win32' else str(
            venv_path / 'bin' / 'pip')
        command = [pip_executable, 'install', '-r', str(req_file)]
        return await self._run_command_in_terminal(command, req_file.parent)

    async def _execute_main_file(self, project_path: Path, venv_path: Path) -> bool:
        main_file = project_path / 'main.py'
        if not main_file.exists():
            for py_file in project_path.rglob("*.py"):
                if 'if __name__ == "__main__":' in py_file.read_text():
                    main_file = py_file
                    break
        if not main_file.exists(): return False
        self._terminal_log("info", f"Executing entry point: {main_file.name}...")
        python_executable = str(venv_path / 'Scripts' / 'python') if sys.platform == 'win32' else str(
            venv_path / 'bin' / 'python')
        command = [python_executable, str(main_file)]
        return await self._run_command_in_terminal(command, project_path)

    def _terminal_log(self, log_type: str, message: str):
        if self.code_viewer and self.code_viewer.terminal:
            terminal = self.code_viewer.terminal
            log_methods = {
                "info": terminal.append_system_message, "error": terminal.append_error,
                "warning": terminal.append_system_message, "success": terminal.append_system_message,
                "command": terminal.append_command, "stdout": terminal.append_output, "stderr": terminal.append_error
            }
            prefix_map = {
                "info": "ℹ️ ", "error": "❌ ERROR: ", "warning": "⚠️ WARN: ", "success": "✅ SUCCESS: ",
                "stdout": "", "stderr": "", "command": ""
            }
            log_method = log_methods.get(log_type, terminal.append_output)
            prefix = prefix_map.get(log_type, "")
            if log_type in ["stdout", "stderr"]:
                log_method(message + "\n")
            else:
                log_method(prefix + message)

    def get_status(self) -> Dict[str, Any]:
        llm_models_list = self.llm_client.get_available_models() if self.llm_client else ["LLM Client not init"]
        rag_info = {"ready": False, "status_text": "RAG: Not Initialized", "available": RAG_MANAGER_AVAILABLE}
        if RAG_MANAGER_AVAILABLE and self.rag_manager:
            rag_info["ready"] = self.rag_manager.is_ready
            rag_info["status_text"] = self.rag_manager.current_status
        return {
            "ready": self.workflow_engine is not None, "llm_models": llm_models_list,
            "workspace": str(self.workspace_dir), "current_project": self.current_project,
            "current_session": self.current_session, "rag": rag_info,
            "windows": {
                "main": self.main_window.isVisible() if self.main_window else False,
                "code_viewer": self.code_viewer.isVisible() if self.code_viewer else False,
                "streaming_terminal": self.streaming_terminal.isVisible() if self.streaming_terminal else False,
            }
        }

    @Slot(str)
    def _on_file_generated(self, file_path: str):
        self.logger.info(f"File generated: {file_path}")
        if self.code_viewer: self.code_viewer.auto_open_file(file_path)

    @Slot(str)
    def _on_project_loaded(self, project_path: str):
        self.logger.info(f"Project loaded: {project_path}")
        self.current_project_path = Path(project_path)
        self.current_project = self.current_project_path.name
        if self.main_window: self.main_window.update_project_display(self.current_project)
        if self.code_viewer:
            self.code_viewer.load_project(project_path)
            self.code_viewer.show();
            self.code_viewer.raise_();
            self.code_viewer.activateWindow()

    def _handle_sidebar_action(self, action: str):
        self.logger.info(f"Sidebar action: {action}")
        if action == "view_log":
            if self.streaming_terminal:
                self.streaming_terminal.show()
                self.streaming_terminal.raise_()
                self.streaming_terminal.activateWindow()
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

    def new_session(self):
        if self.main_window:
            self.main_window.chat_interface.clear_chat()
            self.main_window.chat_interface._add_welcome_message()
            self.current_project = "Default Project";
            self.current_project_path = self.workspace_dir
            self.main_window.update_project_display(self.current_project)
        self.logger.info("New session started.")

    def save_session(self):
        if not self.main_window: return
        history = self.main_window.chat_interface.conversation_history
        session_data = {
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "project_name": self.current_project,
            "project_path": str(self.current_project_path),
            "conversation_history": history
        }
        if self.current_project != "Default Project" and self.current_project_path.exists():
            session_dir = self.current_project_path / ".sessions"
            session_dir.mkdir(exist_ok=True)
            file_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            save_path = session_dir / file_name
        else:
            save_path, _ = QFileDialog.getSaveFileName(self.main_window, "Save Session", str(self.workspace_dir),
                                                       "JSON Files (*.json)")
            if not save_path: return
            save_path = Path(save_path)
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2)
            QMessageBox.information(self.main_window, "Session Saved", f"Session saved to:\n{save_path}")
            self.logger.info(f"Session saved to {save_path}")
        except Exception as e:
            QMessageBox.critical(self.main_window, "Error", f"Failed to save session: {e}")

    def load_session(self):
        if not self.main_window: return
        file_name, _ = QFileDialog.getOpenFileName(self.main_window, "Load Session", str(self.workspace_dir),
                                                   "JSON Files (*.json)")
        if not file_name: return
        try:
            with open(file_name, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            if "conversation_history" not in session_data or "project_path" not in session_data:
                raise ValueError("Invalid session file format.")
            self.main_window.chat_interface.load_history(session_data["conversation_history"])
            self._on_project_loaded(session_data.get("project_path"))
            QMessageBox.information(self.main_window, "Session Loaded", f"Session loaded from:\n{file_name}")
        except Exception as e:
            QMessageBox.critical(self.main_window, "Error", f"Failed to load session: {e}")

    def create_new_project_dialog(self):
        if not self.main_window: return
        project_name, ok = QInputDialog.getText(self.main_window, 'New Project', 'Enter project name:',
                                                text='my-ava-project')
        if not (ok and project_name.strip()): return
        project_name = project_name.strip()
        base_dir_str = QFileDialog.getExistingDirectory(self.main_window, 'Select Directory', str(self.workspace_dir))
        if not base_dir_str: return
        project_path = Path(base_dir_str) / project_name
        if project_path.exists():
            reply = QMessageBox.question(self.main_window, 'Directory Exists',
                                         f'Directory "{project_name}" exists. Use it anyway?',
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No: return
        else:
            project_path.mkdir(parents=True, exist_ok=True)
        (project_path / "main.py").write_text(f'if __name__ == "__main__":\n    print("Hello from {project_name}!")\n')
        self._on_project_loaded(str(project_path))
        self.main_window.chat_interface.add_assistant_response(
            f"Project '{project_name}' created at {project_path}. What should we build in it?")

    def load_existing_project_dialog(self):
        if not self.main_window: return
        folder_path_str = QFileDialog.getExistingDirectory(self.main_window, "Load Existing Project")
        if not folder_path_str: return
        self.logger.info(f"User selected existing project to load: {folder_path_str}")
        if self.workflow_engine:
            self.main_window.chat_interface.add_workflow_status(
                f"🧠 Analyzing project '{Path(folder_path_str).name}'...")
            if self.streaming_terminal: self.streaming_terminal.show()
            asyncio.create_task(self.workflow_engine.execute_analysis_workflow(folder_path_str))

    def _on_error_occurred(self, component: str, error_message: str):
        self.logger.error(f"[ERROR] App Error in {component}: {error_message}")
        if self.main_window: self.main_window.on_app_error_occurred(component, error_message)

    def _handle_enhanced_workflow_request(self, user_prompt: str, conversation_history: List[Dict]):
        self.logger.info(f"Enhanced workflow request: {user_prompt[:100]}...")
        if not self.workflow_engine:
            self.error_occurred.emit("workflow_engine", "Workflow engine not initialized.")
            return
        if self.streaming_terminal: self.streaming_terminal.show()
        asyncio.create_task(self.workflow_engine.execute_enhanced_workflow(user_prompt, conversation_history))

    def shutdown(self):
        self.logger.info("Shutting down AvA Application...")
        if self.main_window: self.main_window.close()
        if self.code_viewer: self.code_viewer.close()
        if self.streaming_terminal: self.streaming_terminal.close()
        self.logger.info("Shutdown complete.")