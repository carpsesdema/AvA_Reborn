# core/application.py - V2.4 with venv and GDD creation

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from PySide6.QtCore import QObject, Signal, Slot, QProcess
from PySide6.QtWidgets import QFileDialog, QMessageBox, QInputDialog

from core.enhanced_workflow_engine import EnhancedWorkflowEngine
from core.llm_client import EnhancedLLMClient, LLMRole
from gui.code_viewer import CodeViewerWindow
# Import our UI components
from gui.main_window import AvAMainWindow
from gui.terminals import StreamingTerminal

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

        # --- NEW: State machine for project execution ---
        self._execution_queue = []
        # ---

        self.llm_client = None
        self.main_window = None
        self.code_viewer = None
        self.workflow_engine = None
        self.rag_manager = None
        self.streaming_terminal = None

        self.workspace_dir = Path("C:/Projects/AvA_Reborn/workspace")
        self.workspace_dir.mkdir(exist_ok=True)
        self.current_project = "Default Project"
        self.current_project_path = self.workspace_dir
        self.current_session = "Main Chat"
        self.active_workflows = {}

    def _setup_logging(self):
        log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True)

        class SafeFormatter(logging.Formatter):
            def format(self, record):
                msg = super().format(record)
                emoji_replacements = {'🚀': '[START]', '✅': '[OK]', '❌': '[ERROR]', '⚠️': '[WARN]', '🧠': '[AI]',
                                      '▶️': '[RUN]', '📂': '[FILE]', '📄': '[DOC]'}
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
        self.workflow_engine = EnhancedWorkflowEngine(self.llm_client, self.streaming_terminal, self.code_viewer,
                                                      self.rag_manager)

    def _connect_components(self):
        self.logger.info("Connecting components...")
        if self.main_window:
            self.main_window.workflow_requested_with_context.connect(self._handle_enhanced_workflow_request)
            self.main_window.new_project_requested.connect(self.create_new_project_dialog)
            self.main_window.load_project_requested.connect(self.load_existing_project_dialog)
            self.main_window.sidebar.action_triggered.connect(self._handle_sidebar_action)

        if self.workflow_engine:
            self.workflow_engine.workflow_completed.connect(self.main_window.on_workflow_completed)
            self.workflow_engine.file_generated.connect(self._on_file_generated)
            self.workflow_engine.project_loaded.connect(self._on_project_loaded)

        if self.code_viewer:
            self.code_viewer.run_project_requested.connect(self.run_current_project)
            self.code_viewer.terminal.command_completed.connect(self._on_command_completed)

        self.error_occurred.connect(self._on_error_occurred)
        self.logger.info("✅ Components connected.")

    @Slot()
    def run_current_project(self):
        if not self.current_project_path or self.current_project == "Default Project":
            self._terminal_log("error", "No project loaded to run.")
            return

        terminal = self.code_viewer.terminal
        terminal.clear_terminal()
        self._terminal_log("info", f"▶️ Starting execution for project: {self.current_project_path.name}")

        self._execution_queue = [
            self._ensure_venv,
            self._install_requirements,
            self._execute_main_file
        ]
        self._execute_next_step_in_queue()

    @Slot(int)
    def _on_command_completed(self, exit_code: int):
        if exit_code == 0:
            self._execute_next_step_in_queue()
        else:
            self._terminal_log("error", "Execution sequence halted due to an error.")
            self._execution_queue.clear()

    def _execute_next_step_in_queue(self):
        if self._execution_queue:
            next_step = self._execution_queue.pop(0)
            next_step()
        else:
            self._terminal_log("info", "Execution sequence finished.")

    def _ensure_venv(self):
        venv_path = self.current_project_path / 'venv'
        if venv_path.exists() and (venv_path / 'pyvenv.cfg').exists():
            self._terminal_log("info", "Virtual environment found.")
            self._on_command_completed(0)
            return

        self._terminal_log("info", "Virtual environment not found. Creating...")
        program = sys.executable
        args = ['-m', 'venv', 'venv']
        self.code_viewer.terminal.execute_command(program, args)

    def _install_requirements(self):
        req_file = self.current_project_path / 'requirements.txt'
        if not req_file.exists():
            self._terminal_log("info", "No requirements.txt found. Skipping dependency installation.")
            self._on_command_completed(0)
            return

        self._terminal_log("info", "requirements.txt found. Installing dependencies...")
        venv_path = self.current_project_path / 'venv'
        pip_executable = str(venv_path / 'Scripts' / 'pip') if sys.platform == 'win32' else str(
            venv_path / 'bin' / 'pip')
        args = ['install', '-r', str(req_file)]
        self.code_viewer.terminal.execute_command(pip_executable, args)

    def _execute_main_file(self):
        main_file = self.current_project_path / 'main.py'
        if not main_file.exists():
            for py_file in self.current_project_path.rglob("*.py"):
                if 'if __name__ == "__main__":' in py_file.read_text(errors='ignore'):
                    main_file = py_file
                    break

        if not main_file.exists():
            self._terminal_log("error",
                               "Could not find a main entry point (main.py or file with __name__ == '__main__').")
            self._on_command_completed(1)
            return

        self._terminal_log("info", f"Executing entry point: {main_file.name}...")
        venv_path = self.current_project_path / 'venv'
        python_executable = str(venv_path / 'Scripts' / 'python') if sys.platform == 'win32' else str(
            venv_path / 'bin' / 'python')
        args = [str(main_file)]
        self.code_viewer.terminal.execute_command(python_executable, args)

    def _terminal_log(self, log_type: str, message: str):
        if self.code_viewer and self.code_viewer.terminal:
            terminal = self.code_viewer.terminal
            log_methods = {"info": terminal.append_system_message, "error": terminal.append_error,
                           "warning": terminal.append_system_message, "success": terminal.append_system_message,
                           "command": terminal.append_command, "stdout": terminal.append_output,
                           "stderr": terminal.append_error}
            prefix_map = {"info": "ℹ️ ", "error": "❌ ERROR: ", "warning": "⚠️ WARN: ", "success": "✅ SUCCESS: ",
                          "stdout": "", "stderr": "", "command": ""}
            log_method = log_methods.get(log_type, terminal.append_output)
            prefix = prefix_map.get(log_type, "")
            if log_type in ["stdout", "stderr"]:
                log_method(message)
            else:
                log_method(prefix + message)

    def get_status(self) -> Dict[str, Any]:
        llm_models_list = self.llm_client.get_available_models() if self.llm_client else ["LLM Client not init"]
        rag_info = {"ready": False, "status_text": "RAG: Not Initialized", "available": RAG_MANAGER_AVAILABLE}
        if RAG_MANAGER_AVAILABLE and self.rag_manager:
            rag_info["ready"] = self.rag_manager.is_ready
            rag_info["status_text"] = self.rag_manager.current_status
        return {"ready": self.workflow_engine is not None, "llm_models": llm_models_list,
                "workspace": str(self.workspace_dir), "current_project": self.current_project,
                "current_session": self.current_session, "rag": rag_info,
                "windows": {"main": self.main_window.isVisible() if self.main_window else False,
                            "code_viewer": self.code_viewer.isVisible() if self.code_viewer else False,
                            "streaming_terminal": self.streaming_terminal.isVisible() if self.streaming_terminal else False}}

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
            self.code_viewer.show()
            self.code_viewer.raise_()
            self.code_viewer.activateWindow()

    def _handle_sidebar_action(self, action: str):
        self.logger.info(f"Sidebar action: {action}")
        if action == "view_log":
            if self.streaming_terminal: self.streaming_terminal.show(); self.streaming_terminal.raise_(); self.streaming_terminal.activateWindow()
        elif action == "open_code_viewer":
            if self.code_viewer: self.code_viewer.show(); self.code_viewer.raise_(); self.code_viewer.activateWindow()
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
            self.current_project = "Default Project"
            self.current_project_path = self.workspace_dir
            self.main_window.update_project_display(self.current_project)
        self.logger.info("New session started.")

    def save_session(self):
        if not self.main_window: return
        history = self.main_window.chat_interface.conversation_history
        session_data = {"version": "1.0", "timestamp": datetime.now().isoformat(), "project_name": self.current_project,
                        "project_path": str(self.current_project_path), "conversation_history": history}
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
            if "conversation_history" not in session_data or "project_path" not in session_data: raise ValueError(
                "Invalid session file format.")
            self.main_window.chat_interface.load_history(session_data["conversation_history"])
            self._on_project_loaded(session_data.get("project_path"))
            QMessageBox.information(self.main_window, "Session Loaded", f"Session loaded from:\n{file_name}")
        except Exception as e:
            QMessageBox.critical(self.main_window, "Error", f"Failed to load session: {e}")

    ### MODIFIED ###
    def create_new_project_dialog(self):
        """Creates a new project with a directory, main.py, GDD, and venv."""
        if not self.main_window: return

        project_name_raw, ok = QInputDialog.getText(self.main_window, 'New Project', 'Enter project name:',
                                                    text='my-ava-project')
        if not (ok and project_name_raw.strip()): return

        # Sanitize project name for use in filenames and paths
        project_name = "".join(c for c in project_name_raw.strip() if c.isalnum() or c in ('_', '-')).rstrip()
        if not project_name: project_name = "ava_project"

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

        # --- MODIFIED: More robust GDD creation ---
        try:
            # Create a placeholder main.py
            (project_path / "main.py").write_text(
                f'# main.py for {project_name}\n\nif __name__ == "__main__":\n    print("Hello from {project_name}!")\n')

            # Create the initial GDD file
            gdd_file_path = project_path / f"{project_name}_GDD.md"
            gdd_template = f"""# Game Design Document: {project_name_raw}

## 1. Project Vision
> Initial idea: {project_name_raw}

## 2. Core Gameplay Loop
_(To be defined)_

## 3. Key Features
_(To be defined)_

## 4. Implemented Systems
_(This section will be populated as you build out the project.)_

---

## Development Log
- **{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**: Project initialized by AvA.
"""
            gdd_file_path.write_text(gdd_template.strip(), encoding='utf-8')

            # Create the virtual environment using a blocking QProcess
            self._terminal_log("info", f"Creating virtual environment for '{project_name}'...")
            QMessageBox.information(self.main_window, "Project Setup",
                                    "Creating virtual environment... This may take a moment.")

            process = QProcess()
            process.setWorkingDirectory(str(project_path))
            process.start(sys.executable, ['-m', 'venv', 'venv'])
            if not process.waitForFinished(30000):  # 30-second timeout
                raise Exception("Virtual environment creation timed out.")
            if process.exitCode() != 0:
                raise Exception(f"Failed to create virtual environment. Exit code: {process.exitCode()}")

            self._terminal_log("success", "Virtual environment created successfully.")

        except Exception as e:
            QMessageBox.critical(self.main_window, "Project Setup Error", f"Failed to set up project files: {e}")
            self.logger.error(f"Failed during project creation for '{project_name}': {e}", exc_info=True)
            return

        self._on_project_loaded(str(project_path))
        self.main_window.chat_interface.add_assistant_response(
            f"✅ Project '{project_name}' created at {project_path}.\n\nA virtual environment and a GDD file have been set up. What should we build first?")

    def load_existing_project_dialog(self):
        if not self.main_window: return
        folder_path_str = QFileDialog.getExistingDirectory(self.main_window, "Load Existing Project",
                                                           str(self.workspace_dir))
        if not folder_path_str: return
        if self.workflow_engine:
            self.main_window.chat_interface.add_workflow_status(
                f"🧠 Analyzing project '{Path(folder_path_str).name}'...")
            if self.streaming_terminal: self.streaming_terminal.show()
            asyncio.create_task(self.workflow_engine.execute_analysis_workflow(folder_path_str))

    @Slot(str, str)
    def _on_error_occurred(self, component: str, error_message: str):
        self.logger.error(f"[ERROR] App Error in {component}: {error_message}")
        if self.main_window: self.main_window.on_app_error_occurred(component, error_message)

    @Slot(str, list)
    def _handle_enhanced_workflow_request(self, user_prompt: str, conversation_history: List[Dict]):
        """
        Slot to receive the user request signal. It schedules the async processing
        of the request without blocking the UI thread.
        """
        asyncio.create_task(self._process_user_request_async(user_prompt, conversation_history))

    async def _process_user_request_async(self, user_prompt: str, conversation_history: List[Dict]):
        """
        Handles a user request by first triaging it as a 'chat' or 'workflow'
        and then routing it to the appropriate handler.
        """
        self.logger.info(f"Handling user request: {user_prompt[:100]}...")

        # --- Triage Step ---
        triage_prompt = f"""
        Analyze the user's prompt. Is it a simple conversational message (like a greeting, a question about you, a casual remark) or is it a request to create, build, modify, or analyze a software project?

        Respond with only a single word: "chat" or "workflow".

        User Prompt: "{user_prompt}"
        """
        try:
            # Use a fast model for this simple task
            triage_response = await self.llm_client.chat(triage_prompt, role=LLMRole.CHAT)
            decision = triage_response.strip().lower()
            self.logger.info(f"Triage decision: '{decision}' for prompt: '{user_prompt}'")
        except Exception as e:
            self.logger.error(f"Triage failed: {e}. Defaulting to workflow.", exc_info=True)
            decision = "workflow"  # Fail-safe to old behavior

        # --- Route based on decision ---
        if "workflow" in decision:
            self.logger.info("Routing to Enhanced Workflow Engine.")
            if not self.workflow_engine:
                self.error_occurred.emit("workflow_engine", "Workflow engine not initialized.")
                return
            if self.streaming_terminal:
                self.streaming_terminal.show()
            # This is already an async method, so we can await the task
            await self.workflow_engine.execute_enhanced_workflow(user_prompt, conversation_history)
        else:  # "chat" or any other response
            self.logger.info("Routing to Simple Chat Handler.")
            await self._handle_simple_chat(user_prompt, conversation_history)

    async def _handle_simple_chat(self, user_prompt: str, conversation_history: List[Dict]):
        """
        Handles a simple conversational message by generating a direct chat response.
        """
        try:
            # Use the last 5 messages for context to keep it brief
            history_for_prompt = conversation_history[-5:]
            # Build a simple prompt with history for the chat model
            history_str = "\n".join(
                [f"{msg['role']}: {msg['message']}" for msg in history_for_prompt if msg.get('message')])

            chat_prompt = f"""
            You are AvA, a friendly and helpful AI development assistant. Continue the conversation naturally based on the recent history.

            Recent Conversation History:
            {history_str}

            Current User Prompt:
            user: {user_prompt}
            assistant: """

            response = await self.llm_client.chat(chat_prompt, role=LLMRole.CHAT)
            self.main_window.chat_interface.add_assistant_response(response)
        except Exception as e:
            self.logger.error(f"Simple chat handler failed: {e}", exc_info=True)
            self.error_occurred.emit("simple_chat", str(e))

    def shutdown(self):
        self.logger.info("Shutting down AvA Application...")
        if self.main_window: self.main_window.close()
        if self.code_viewer: self.code_viewer.close()
        if self.streaming_terminal: self.streaming_terminal.close()
        self.logger.info("Shutdown complete.")