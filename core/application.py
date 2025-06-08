# core/application.py - Enhanced Error-Aware Terminal Integration

import asyncio
import logging
import json
import sys
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
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


class ErrorContext:
    """Stores execution error context for Ava to analyze"""

    def __init__(self):
        self.last_command = ""
        self.stdout = ""
        self.stderr = ""
        self.exit_code = 0
        self.working_directory = ""
        self.timestamp = None
        self.file_states = {}  # Track which files existed before error

    def capture_execution(self, command: str, result: subprocess.CompletedProcess, cwd: str):
        """Capture full execution context"""
        self.last_command = command
        self.stdout = result.stdout or ""
        self.stderr = result.stderr or ""
        self.exit_code = result.returncode
        self.working_directory = cwd
        self.timestamp = datetime.now()

        # Capture file states for context
        if Path(cwd).exists():
            try:
                self.file_states = {
                    str(f.relative_to(cwd)): f.stat().st_mtime
                    for f in Path(cwd).rglob("*.py")
                    if f.is_file()
                }
            except Exception:
                self.file_states = {}

    def has_error(self) -> bool:
        """Check if this execution had errors"""
        return self.exit_code != 0 or "error" in self.stderr.lower() or "traceback" in self.stderr.lower()

    def get_error_summary(self) -> str:
        """Get a concise error summary for Ava"""
        if not self.has_error():
            return "No errors detected"

        summary = f"Command: {self.last_command}\n"
        summary += f"Exit Code: {self.exit_code}\n"
        summary += f"Working Dir: {self.working_directory}\n"

        if self.stderr:
            # Extract key error information
            lines = self.stderr.split('\n')
            important_lines = []
            for line in lines:
                line = line.strip()
                if any(keyword in line.lower() for keyword in
                       ['error', 'exception', 'traceback', 'failed', 'not found']):
                    important_lines.append(line)

            if important_lines:
                summary += f"Key Errors:\n" + "\n".join(important_lines[:5])
            else:
                summary += f"Stderr:\n{self.stderr[:500]}"

        if self.stdout and "error" in self.stdout.lower():
            summary += f"\nStdout (contains errors):\n{self.stdout[:300]}"

        return summary


class AvAApplication(QObject):
    fully_initialized_signal = Signal()
    workflow_started = Signal(str)
    workflow_completed = Signal(dict)
    error_occurred = Signal(str, str)
    rag_status_changed = Signal(str, str)
    project_loaded = Signal(str)

    # --- Enhanced terminal signals ---
    terminal_command_to_run = Signal(str)
    terminal_output = Signal(str)
    terminal_error = Signal(str)
    terminal_system_message = Signal(str)
    terminal_focus_requested = Signal()
    chat_message_received = Signal(str)

    # --- New error signals for Ava awareness ---
    execution_error_detected = Signal(object)  # ErrorContext object
    error_analysis_requested = Signal(str)  # Error summary for Ava

    def __init__(self):
        super().__init__()
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

        # Core components
        self.llm_client = None
        self.main_window = None
        self.terminal_window = None
        self.code_viewer = None
        self.workflow_engine = None
        self.rag_manager = None
        self.execution_engine = None

        # Project state
        self.workspace_dir = Path("./workspace")
        self.workspace_dir.mkdir(exist_ok=True)
        self.current_project = "Default Project"
        self.current_project_path = self.workspace_dir
        self.current_session = "Main Chat"
        self.active_workflows = {}

        # Error tracking for Ava
        self.last_error_context = None
        self.error_history = []  # Keep last 5 errors for pattern analysis

        self.current_config = {
            "chat_model": "gemini-2.5-pro-preview-06-05",
            "code_model": "qwen2.5-coder:14b",
            "temperature": 0.7
        }

        self.performance_timer = QTimer()
        self.performance_timer.timeout.connect(self._update_performance_stats)
        self.performance_timer.start(5000)

    def _setup_logging(self):
        log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True)

        class SafeFormatter(logging.Formatter):
            def format(self, record):
                msg = super().format(record)
                emoji_replacements = {
                    '🚀': '[START]', '✅': '[OK]', '❌': '[ERROR]',
                    '⚠️': '[WARN]', '🧠': '[AI]', '📊': '[STATS]',
                    '🔧': '[TOOL]', '📄': '[FILE]'
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
        self.logger.info("[START] Initializing Error-Aware AvA Application...")
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
        self.execution_engine = ExecutionEngine(project_state_manager=None, terminal=self.terminal_window)
        self.logger.info("[OK] Core services initialized.")

    async def _initialize_async_components(self):
        self.logger.info("Starting async component initialization...")
        await self._initialize_rag_manager()
        self._initialize_workflow_engine()
        self._initialize_execution_engine()
        self._connect_components()
        self._setup_window_behaviors()
        self.logger.info("Async component initialization completed.")
        self.fully_initialized_signal.emit()
        self.logger.info("[OK] Error-Aware AvA Application initialized successfully.")

    async def _initialize_rag_manager(self):
        if RAG_MANAGER_AVAILABLE:
            try:
                self.rag_manager = RAGManager()
                await self.rag_manager.async_initialize()
                self._on_rag_status_changed("RAG: Ready", "green")
                self.logger.info("[OK] RAG Manager initialized")
            except Exception as e:
                self.logger.error(f"RAG Manager initialization failed: {e}", exc_info=True)
                self.rag_manager = None
                self._on_rag_status_changed("RAG: Failed", "red")
        else:
            self.rag_manager = None
            self._on_rag_status_changed("RAG: Not Available", "grey")

    def _initialize_workflow_engine(self):
        self.logger.info("Initializing workflow engine...")
        self.workflow_engine = EnhancedWorkflowEngine(
            self.llm_client,
            self.terminal_window,
            self.code_viewer,
            self.rag_manager
        )
        self.logger.info("[OK] Workflow engine initialized.")

    def _connect_components(self):
        self.logger.info("Connecting components...")

        # Main window connections
        if self.main_window:
            self.main_window.workflow_requested.connect(self._handle_workflow_request)
            if hasattr(self.main_window, 'workflow_requested_with_context'):
                self.main_window.workflow_requested_with_context.connect(self._handle_enhanced_workflow_request)
            self.main_window.new_project_requested.connect(self.create_new_project_dialog)
            self.main_window.load_project_requested.connect(self.load_existing_project_dialog)
            if hasattr(self.main_window.sidebar, 'action_triggered'):
                self.main_window.sidebar.action_triggered.connect(self._handle_sidebar_action)

        # Terminal connections with error awareness
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

        # Chat connections
        if self.main_window:
            self.chat_message_received.connect(self.main_window.chat_interface.add_assistant_response)

        # Workflow connections
        if self.workflow_engine and self.main_window:
            self.workflow_engine.workflow_completed.connect(self._on_workflow_completed)

        # Error awareness connections
        self.execution_error_detected.connect(self._handle_execution_error)
        self.error_analysis_requested.connect(self._send_error_to_ava)

        # Standard workflow signals
        self.workflow_started.connect(self._on_workflow_started)
        # This signal is now only connected from the engine, not to itself
        self.error_occurred.connect(self._on_error_occurred)

        if self.workflow_engine:
            self.workflow_engine.file_generated.connect(self._on_file_generated)
            self.workflow_engine.project_loaded.connect(self.project_loaded.emit)

        self.project_loaded.connect(self._on_project_loaded)
        self.logger.info("[OK] Components connected with error awareness.")

    def _setup_window_behaviors(self):
        if self.main_window:
            self.main_window.setWindowTitle("AvA - Error-Aware AI Development")
        if self.terminal_window:
            self.terminal_window.setWindowTitle("AvA - Execution Terminal")
        if self.code_viewer:
            self.code_viewer.setWindowTitle("AvA - Code Viewer & IDE")
        self._position_windows()
        self.logger.info("[OK] Window behaviors set up.")

    def _position_windows(self):
        screen_geo = QApplication.primaryScreen().geometry()
        if self.main_window:
            self.main_window.setGeometry(100, 50, 1400, 900)
        if self.terminal_window:
            self.terminal_window.setGeometry(
                screen_geo.width() - 920, screen_geo.height() - 450, 900, 400
            )
        if self.code_viewer:
            self.code_viewer.setGeometry(screen_geo.width() - 1000, 50, 950, 700)

    def run_project_in_terminal(self, project_path_str: Optional[str] = None) -> bool:
        """
        Enhanced manual execution with comprehensive error capture for Ava
        Returns True if successful, False if errors occurred
        """
        path_to_run = Path(project_path_str) if project_path_str else self.current_project_path
        path_to_run = path_to_run.resolve()

        self.chat_message_received.emit(f"🚀 Running project: {path_to_run.name}")
        self.terminal_focus_requested.emit()

        def run_command_and_capture(command_parts: List[str], timeout_override: int = None) -> Tuple[
            bool, subprocess.CompletedProcess]:
            """Run command with full error capture"""
            command_str = ' '.join(f'"{part}"' if ' ' in part else part for part in command_parts)
            self.terminal_command_to_run.emit(command_str)

            try:
                result = subprocess.run(
                    command_parts,
                    capture_output=True,
                    text=True,
                    cwd=str(path_to_run),
                    timeout=timeout_override or 300,
                    check=False,
                    encoding='utf-8',
                    errors='ignore'
                )

                # Always emit output to terminal
                if result.stdout:
                    self.terminal_output.emit(result.stdout)
                if result.stderr:
                    self.terminal_error.emit(result.stderr)

                # Create error context for Ava
                error_context = ErrorContext()
                error_context.capture_execution(command_str, result, str(path_to_run))

                # Check for errors and notify Ava if needed
                if error_context.has_error():
                    self.last_error_context = error_context
                    self.error_history.append(error_context)
                    if len(self.error_history) > 5:  # Keep only last 5 errors
                        self.error_history.pop(0)

                    self.execution_error_detected.emit(error_context)
                    self.terminal_system_message.emit(
                        f"❌ Process finished with exit code {result.returncode} - Error captured for Ava analysis"
                    )
                    return False, result
                else:
                    self.terminal_system_message.emit(
                        f"✅ Process finished successfully with exit code {result.returncode}"
                    )
                    return True, result

            except subprocess.TimeoutExpired:
                msg = f"Command '{command_str[:50]}...' timed out!"
                self.terminal_error.emit(msg)
                self.terminal_system_message.emit("❌ Execution timed out")

                # Create timeout error context
                error_context = ErrorContext()
                error_context.last_command = command_str
                error_context.stderr = msg
                error_context.exit_code = -1
                error_context.working_directory = str(path_to_run)
                error_context.timestamp = datetime.now()

                self.last_error_context = error_context
                self.execution_error_detected.emit(error_context)
                return False, None
            except Exception as e:
                error_msg = f"Unexpected error running command: {e}"
                self.terminal_error.emit(error_msg)
                self.terminal_system_message.emit("❌ Execution failed unexpectedly")

                # Create exception error context
                error_context = ErrorContext()
                error_context.last_command = command_str
                error_context.stderr = error_msg
                error_context.exit_code = -2
                error_context.working_directory = str(path_to_run)
                error_context.timestamp = datetime.now()

                self.last_error_context = error_context
                self.execution_error_detected.emit(error_context)
                return False, None

        # Check for main.py
        main_file = path_to_run / "main.py"
        if not main_file.exists():
            self.chat_message_received.emit("❌ No main.py file found!")

            # Create "no main file" error context
            error_context = ErrorContext()
            error_context.last_command = "check for main.py"
            error_context.stderr = f"main.py not found in {path_to_run}"
            error_context.exit_code = -3
            error_context.working_directory = str(path_to_run)
            error_context.timestamp = datetime.now()

            self.last_error_context = error_context
            self.execution_error_detected.emit(error_context)
            return False

        # Check for virtual environment or requirements
        venv_python = path_to_run / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        req_file = path_to_run / "requirements.txt"

        # Handle dependencies if requirements.txt exists
        if req_file.exists():
            self.chat_message_received.emit("📦 Installing dependencies...")
            python_for_install = str(venv_python) if venv_python.exists() else self._find_system_python()

            success, result = run_command_and_capture([python_for_install, "-m", "pip", "install", "-r", req_file.name])
            if not success:
                self.chat_message_received.emit("❌ Failed to install dependencies. Check terminal for errors.")
                return False
            else:
                self.chat_message_received.emit("✅ Dependencies installed successfully!")

        # Execute main script
        self.chat_message_received.emit("🏃 Executing main script...")
        python_to_use = str(venv_python) if venv_python.exists() else self._find_system_python()

        success, result = run_command_and_capture([python_to_use, main_file.name])

        if success:
            self.chat_message_received.emit("✅ Project execution completed successfully!")
            return True
        else:
            self.chat_message_received.emit("❌ Project execution encountered errors. Check terminal output above.")
            self.chat_message_received.emit(
                "💡 Ask Ava to analyze the error: 'Can you fix the error that just occurred?'")
            return False

    def _find_system_python(self) -> str:
        """Find system Python (outside of AvA's virtual environment)"""
        import subprocess, os
        ava_venv_marker, current_executable = ".venv", sys.executable
        potential_pythons = []

        if sys.platform == "win32":
            potential_pythons = ["python", "py", "py -3", r"C:\Python312\python.exe", r"C:\Python311\python.exe"]
            user_appdata = os.environ.get('LOCALAPPDATA', '')
            if user_appdata:
                for version in ['312', '311']:
                    potential_pythons.append(f"{user_appdata}\\Programs\\Python\\Python{version}\\python.exe")
        else:
            potential_pythons = ["python3", "python", "/usr/bin/python3"]

        for python_cmd in potential_pythons:
            try:
                cmd_parts = python_cmd.split()
                result = subprocess.run(cmd_parts + ["--version"], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    result2 = subprocess.run(
                        cmd_parts + ["-c", "import sys; print(sys.executable)"],
                        capture_output=True, text=True, timeout=10
                    )
                    if result2.returncode == 0:
                        python_path = result2.stdout.strip()
                        if ava_venv_marker not in python_path and python_path != current_executable:
                            return python_cmd
            except Exception:
                continue
        return "python"

    def _handle_execution_error(self, error_context: ErrorContext):
        """Handle execution errors - log and prepare for Ava analysis"""
        self.logger.error(f"Execution error captured: {error_context.get_error_summary()}")

        # Auto-suggest Ava can help
        suggestion_msg = "\n🤖 Ava can help! Try asking:\n"
        suggestion_msg += "• 'Can you fix the error that just occurred?'\n"
        suggestion_msg += "• 'What went wrong with the execution?'\n"
        suggestion_msg += "• 'Debug the last error'"

        self.chat_message_received.emit(suggestion_msg)

    def _send_error_to_ava(self, user_message: str):
        """Send error context to Ava for analysis"""
        if not self.last_error_context:
            self.chat_message_received.emit("No recent execution errors to analyze.")
            return

        # Prepare comprehensive error context for Ava
        error_summary = self.last_error_context.get_error_summary()

        # Add file context if available
        context_message = f"Here's the error that occurred:\n\n{error_summary}\n\n"

        if self.last_error_context.file_states:
            context_message += "Project files present:\n"
            for file_path in sorted(self.last_error_context.file_states.keys()):
                context_message += f"  - {file_path}\n"

        # Combine user message with error context
        full_prompt = f"{user_message}\n\nError Context:\n{context_message}"

        # Send to workflow engine for analysis
        if self.workflow_engine:
            asyncio.create_task(
                self.workflow_engine.execute_enhanced_workflow(full_prompt, [])
            )

    def get_last_error_for_ava(self) -> Optional[str]:
        """Get the last error in a format suitable for Ava analysis"""
        if not self.last_error_context:
            return None
        return self.last_error_context.get_error_summary()

    def get_error_patterns_for_ava(self) -> str:
        """Analyze error patterns for Ava"""
        if not self.error_history:
            return "No error history available."

        analysis = f"Error History ({len(self.error_history)} recent errors):\n\n"
        for i, error in enumerate(self.error_history[-3:], 1):  # Last 3 errors
            analysis += f"Error {i}:\n"
            analysis += f"  Command: {error.last_command}\n"
            analysis += f"  Exit Code: {error.exit_code}\n"
            analysis += f"  Time: {error.timestamp.strftime('%H:%M:%S') if error.timestamp else 'Unknown'}\n"
            if error.stderr:
                # Get first error line
                first_error = error.stderr.split('\n')[0][:100]
                analysis += f"  Error: {first_error}\n"
            analysis += "\n"

        return analysis

    # --- Rest of the original methods remain the same ---

    def _initialize_execution_engine(self):
        self.execution_engine = ExecutionEngine(project_state_manager=None, terminal=self.terminal_window)
        self.logger.info("[OK] Execution engine initialized.")

    def _open_terminal(self):
        if self.terminal_window:
            self.terminal_window.show()
            self.terminal_window.raise_()

    def _open_code_viewer(self):
        if self.code_viewer:
            self.code_viewer.show()
            self.code_viewer.raise_()

    def _on_file_generated(self, file_path: str):
        if self.code_viewer:
            self.code_viewer.auto_open_file(file_path)

    def _on_project_loaded(self, project_path: str):
        self.logger.info(f"Project loaded: {project_path}")
        self.current_project_path = Path(project_path)
        self.current_project = self.current_project_path.name

        # Update the UI
        if self.main_window:
            QMetaObject.invokeMethod(
                self.main_window, 'update_project_display', Qt.QueuedConnection,
                Q_ARG(str, self.current_project)
            )
        if self.code_viewer:
            QMetaObject.invokeMethod(
                self.code_viewer, 'load_project', Qt.QueuedConnection,
                Q_ARG(str, project_path)
            )

        # Send a single confirmation message to the chat
        self.chat_message_received.emit(f"✅ Project loaded: {self.current_project}")

    def _on_workflow_started(self, prompt: str):
        self.logger.info(f"Workflow started: {prompt}")

    def _on_workflow_completed(self, result: dict):
        # FIX: The slot's job is to handle the signal, not re-emit it.
        # This breaks the infinite loop.
        self.logger.info(f"Workflow completed: {result.get('project_name', 'N/A')}")
        # self.workflow_completed.emit(result) # <-- REMOVED THIS LINE TO BREAK LOOP

    def _on_error_occurred(self, component: str, error_message: str):
        self.logger.error(f"Error in {component}: {error_message}")

    def _on_rag_status_changed(self, status_text: str, color: str):
        if self.main_window:
            QMetaObject.invokeMethod(
                self.main_window, 'update_rag_status_display', Qt.QueuedConnection,
                Q_ARG(str, status_text), Q_ARG(str, color)
            )

    def _update_performance_stats(self):
        pass

    def get_status(self) -> Dict[str, Any]:
        llm_models_list = self.llm_client.get_available_models() if self.llm_client else ["LLM Client not init"]
        rag_info = {
            "ready": False,
            "status_text": "RAG: Not Initialized",
            "available": RAG_MANAGER_AVAILABLE,
            "collections": {}
        }

        if RAG_MANAGER_AVAILABLE:
            if self.rag_manager and hasattr(self.rag_manager, 'is_ready'):
                rag_info["ready"] = self.rag_manager.is_ready
                rag_info["status_text"] = self.rag_manager.current_status
                if hasattr(self.rag_manager, 'get_collection_info'):
                    rag_info["collections"] = self.rag_manager.get_collection_info()
            else:
                rag_info["status_text"] = "RAG: Initialization Failed"
        else:
            rag_info["status_text"] = "RAG: Dependencies Missing"

        return {
            "ready": self.workflow_engine is not None,
            "error_aware": True,
            "llm_models": llm_models_list,
            "workspace": str(self.workspace_dir),
            "current_project": self.current_project,
            "current_session": self.current_session,
            "configuration": self.current_config,
            "rag": rag_info,
            "active_workflows": len(self.active_workflows),
            "last_error": self.last_error_context.get_error_summary() if self.last_error_context else None
        }

    def _handle_workflow_request(self, user_prompt: str):
        if not self.workflow_engine:
            self.error_occurred.emit("workflow", "Engine not available.")
            return
        self._open_terminal()
        asyncio.create_task(self.workflow_engine.execute_enhanced_workflow(user_prompt, []))

    def _handle_enhanced_workflow_request(self, user_prompt: str, conversation_history: List[Dict]):
        if not self.workflow_engine:
            self.error_occurred.emit("workflow", "Engine not available.")
            return

        # Check if this might be an error analysis request
        error_keywords = ['error', 'fix', 'debug', 'problem', 'issue', 'failed', 'exception']
        if any(keyword in user_prompt.lower() for keyword in error_keywords) and self.last_error_context:
            self._send_error_to_ava(user_prompt)
            return

        self._open_terminal()
        asyncio.create_task(self.workflow_engine.execute_enhanced_workflow(user_prompt, conversation_history))

    def _handle_sidebar_action(self, action: str):
        if action == "new_project":
            self.create_new_project_dialog()
        elif action == "load_project":
            self.load_existing_project_dialog()
        elif action == "open_terminal":
            self._open_terminal()
        elif action == "open_code_viewer":
            self._open_code_viewer()
        elif action == "run_project":
            asyncio.create_task(asyncio.to_thread(self.run_project_in_terminal))
        elif action == "analyze_last_error":
            if self.last_error_context:
                self._send_error_to_ava("Please analyze and fix the last error that occurred.")
            else:
                self.chat_message_received.emit("No recent errors to analyze.")

    def create_new_project_dialog(self):
        self.logger.info("Creating new project...")
        if not self.main_window:
            return

        project_name, ok = QInputDialog.getText(
            self.main_window, 'New Project', 'Enter project name:',
            text='my-ava-project'
        )
        if not (ok and project_name.strip()):
            return

        base_dir_str = QFileDialog.getExistingDirectory(
            self.main_window, 'Select Directory', str(self.workspace_dir)
        )
        if not base_dir_str:
            return

        project_path = Path(base_dir_str) / project_name.strip()
        if project_path.exists():
            reply = QMessageBox.question(
                self.main_window, 'Directory Exists',
                f'Directory "{project_name.strip()}" exists. Continue anyway?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        try:
            project_path.mkdir(parents=True, exist_ok=True)
            (project_path / ".gitignore").write_text(".venv/\n__pycache__/\n*.pyc\n", encoding='utf-8')
            (project_path / "main.py").write_text(f'print("Hello from {project_name.strip()}!")\n', encoding='utf-8')

            self.current_project_path = project_path
            self.current_project = project_path.name
            self.project_loaded.emit(str(project_path))
            self.logger.info(f"New project created: {project_path}")

        except Exception as e:
            self.logger.error(f"Failed to create project: {e}")
            QMessageBox.critical(self.main_window, "Error", f"Failed to create project: {e}")

    def load_existing_project_dialog(self):
        self.logger.info("Loading existing project...")
        if not self.main_window:
            return

        project_dir = QFileDialog.getExistingDirectory(
            self.main_window, 'Select Project Directory', str(self.workspace_dir)
        )
        if not project_dir:
            return

        # Let the workflow engine handle the logic and the project_loaded signal.
        if self.workflow_engine:
            self.chat_message_received.emit(f"🧠 Analyzing project '{Path(project_dir).name}'...")
            asyncio.create_task(self.workflow_engine.execute_analysis_workflow(project_dir))
        else:
            self.logger.error("Workflow engine not available for project analysis.")
            self.chat_message_received.emit("❌ Error: Workflow engine not available.")

    def cleanup(self):
        """Clean up resources before shutdown"""
        self.logger.info("Cleaning up AvA Application...")

        if self.performance_timer:
            self.performance_timer.stop()

        if self.rag_manager:
            try:
                # Assuming RAG manager has a cleanup method
                if hasattr(self.rag_manager, 'cleanup'):
                    self.rag_manager.cleanup()
            except Exception as e:
                self.logger.error(f"Error cleaning up RAG manager: {e}")

        self.logger.info("AvA Application cleanup completed.")