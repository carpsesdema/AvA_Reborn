# core/application.py
import asyncio
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from PySide6.QtCore import QObject, Signal, Slot, QTimer, Qt, QMetaObject, Q_ARG
from PySide6.QtWidgets import QFileDialog, QMessageBox

from core.enhanced_workflow_engine import EnhancedWorkflowEngine
from gui.code_viewer import CodeViewerWindow
from gui.main_window import AvAMainWindow
from gui.terminals import TerminalWindow
from utils.logger import get_logger

# Import RAGManager with error handling
try:
    from core.rag_manager import RAGManager

    RAG_MANAGER_AVAILABLE = True
except ImportError as e:
    RAG_MANAGER_AVAILABLE = False
    print(f"RAG Manager not available: {e}")

logger = get_logger(__name__)


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

        # Capture file states
        try:
            for file_path in Path(cwd).rglob("*.py"):
                self.file_states[str(file_path.relative_to(cwd))] = file_path.stat().st_mtime
        except Exception:
            pass

    def get_error_summary(self) -> str:
        """Get a summary of the error for Ava"""
        summary = f"Command: {self.last_command}\n"
        summary += f"Exit Code: {self.exit_code}\n"
        summary += f"Working Directory: {self.working_directory}\n"

        if self.stderr:
            summary += f"\nError Output:\n{self.stderr}\n"

        if self.stdout:
            summary += f"\nStandard Output:\n{self.stdout}\n"

        return summary


class AvAApplication(QObject):
    """Main application controller for AvA"""

    # Signals
    fully_initialized_signal = Signal()
    status_changed = Signal(dict)
    project_loaded = Signal(str)
    workflow_started = Signal(str)
    workflow_completed = Signal(dict)
    file_generated = Signal(str)
    error_occurred = Signal(str)
    chat_message_received = Signal(str)

    # Terminal signals for error awareness
    terminal_command_to_run = Signal(str)
    terminal_output = Signal(str)
    terminal_error = Signal(str)
    terminal_system_message = Signal(str)
    terminal_focus_requested = Signal()
    execution_error_detected = Signal(dict)
    error_analysis_requested = Signal(str, str)  # error_message, file_path

    def __init__(self):
        super().__init__()
        self.logger = logger
        self.logger.info("Initializing AvA Application...")

        # Core components
        self.main_window = None
        self.code_viewer = None
        self.terminal_window = None
        self.workflow_engine = None
        self.rag_manager = None

        # State
        self.current_project = "Default Project"
        self.current_project_path = None
        self.workspace_dir = Path.home() / "AvA_Projects"
        self.workspace_dir.mkdir(exist_ok=True)

        # Error tracking
        self.last_error_context = None
        self.error_history = []

        # Performance monitoring
        self.performance_timer = None
        self._last_memory_usage = 0

    async def initialize(self):
        """Async initialization of components"""
        try:
            # Initialize UI components (sync)
            self._initialize_ui()

            # Initialize async components
            await self._initialize_async_components()

            # Connect all components
            self._connect_components()

            # Start performance monitoring
            self._start_performance_monitoring()

            # Show main window
            if self.main_window:
                self.main_window.show()

            # Emit initialization complete
            self.fully_initialized_signal.emit()
            self.logger.info("AvA Application fully initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize application: {e}")
            raise

    def _initialize_ui(self):
        """Initialize UI components"""
        self.logger.info("Initializing UI components...")

        # Create main window
        self.main_window = AvAMainWindow()
        self.main_window.setWindowTitle("AvA - AI Development Assistant")

        # Create code viewer
        self.code_viewer = CodeViewerWindow()

        # Create terminal window
        self.terminal_window = TerminalWindow()

        self.logger.info("UI components initialized")

    async def _initialize_async_components(self):
        """Initialize async components"""
        self.logger.info("Initializing async components...")

        # Initialize RAG manager
        if RAG_MANAGER_AVAILABLE:
            try:
                self.rag_manager = RAGManager()
                # FIX: Call async_initialize instead of initialize
                await self.rag_manager.async_initialize()
                self.logger.info("RAG manager initialized successfully")
            except Exception as e:
                self.logger.warning(f"RAG manager initialization failed: {e}")
                self.rag_manager = None
        else:
            self.logger.warning("RAG manager not available - some features may be limited")
            self.rag_manager = None

        # Initialize workflow engine
        self.workflow_engine = EnhancedWorkflowEngine(rag_service=self.rag_manager)

        self.logger.info("Async components initialized")

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
        self.error_occurred.connect(self._on_error_occurred)

        if self.workflow_engine:
            self.workflow_engine.file_generated.connect(self._on_file_generated)
            # FIX: Don't forward the signal, just connect directly to the handler
            self.workflow_engine.project_loaded.connect(self._on_project_loaded)

        # FIX: Remove the duplicate connection
        # self.project_loaded.connect(self._on_project_loaded)

        self.logger.info("[OK] Components connected with error awareness.")

    def _start_performance_monitoring(self):
        """Start monitoring performance metrics"""
        self.performance_timer = QTimer()
        self.performance_timer.timeout.connect(self._update_performance_metrics)
        self.performance_timer.start(5000)  # Update every 5 seconds

    def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()

            metrics = {
                'memory_mb': memory_mb,
                'cpu_percent': cpu_percent,
                'timestamp': datetime.now().isoformat()
            }

            # Emit status update
            self.status_changed.emit({
                'type': 'performance',
                'metrics': metrics
            })

        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")

    @Slot(str)
    def _handle_workflow_request(self, prompt: str):
        """Handle workflow request from UI"""
        if not self.workflow_engine:
            self.logger.error("Workflow engine not initialized")
            self.chat_message_received.emit("❌ Error: Workflow engine not available")
            return

        # Create async task for workflow
        asyncio.create_task(self._execute_workflow(prompt))

    @Slot(str, dict)
    def _handle_enhanced_workflow_request(self, prompt: str, context: dict):
        """Handle enhanced workflow request with context"""
        if not self.workflow_engine:
            self.logger.error("Workflow engine not initialized")
            self.chat_message_received.emit("❌ Error: Workflow engine not available")
            return

        # Create async task for enhanced workflow
        asyncio.create_task(self._execute_enhanced_workflow(prompt, context))

    async def _execute_workflow(self, prompt: str):
        """Execute standard workflow"""
        try:
            self.workflow_started.emit(prompt)
            self.chat_message_received.emit(f"🚀 Starting workflow: {prompt}")

            result = await self.workflow_engine.execute_workflow(prompt, self.workspace_dir)

            # Workflow completed signal will be emitted by the engine

        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            self.error_occurred.emit(str(e))
            self.chat_message_received.emit(f"❌ Workflow failed: {str(e)}")

    async def _execute_enhanced_workflow(self, prompt: str, context: dict):
        """Execute enhanced workflow with context"""
        try:
            self.workflow_started.emit(prompt)
            self.chat_message_received.emit(f"🚀 Starting enhanced workflow: {prompt}")

            # Extract project path from context if available
            project_path = context.get('project_path', self.current_project_path)

            if not project_path:
                raise ValueError("No project path available for enhanced workflow")

            result = await self.workflow_engine.execute_workflow(
                prompt,
                project_path,
                use_existing_project=True
            )

            # Workflow completed signal will be emitted by the engine

        except Exception as e:
            self.logger.error(f"Enhanced workflow execution failed: {e}")
            self.error_occurred.emit(str(e))
            self.chat_message_received.emit(f"❌ Enhanced workflow failed: {str(e)}")

    @Slot(str)
    def _handle_sidebar_action(self, action: str):
        """Handle sidebar actions"""
        action_map = {
            'terminal': self._open_terminal,
            'code_viewer': self._open_code_viewer,
            'new_project': self.create_new_project_dialog,
            'load_project': self.load_existing_project_dialog,
        }

        handler = action_map.get(action)
        if handler:
            handler()
        else:
            self.logger.warning(f"Unknown sidebar action: {action}")

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
        self.logger.info(f"Workflow completed: {result}")

        # Update UI based on result
        success = result.get('success', False)
        if success:
            msg = f"✅ Workflow completed successfully!"
            if 'num_files' in result:
                msg += f" ({result['num_files']} files)"
        else:
            msg = f"❌ Workflow completed with errors"

        self.chat_message_received.emit(msg)

        # Update status
        self.status_changed.emit({
            'type': 'workflow_completed',
            'result': result
        })

    def _on_error_occurred(self, error_msg: str):
        self.logger.error(f"Error occurred: {error_msg}")
        self.chat_message_received.emit(f"❌ Error: {error_msg}")

    def get_status(self) -> Dict[str, Any]:
        """Get current application status"""
        status = {
            'initialized': True,
            'current_project': self.current_project,
            'workspace_dir': str(self.workspace_dir),
        }

        # Add workflow engine status
        if self.workflow_engine:
            status['workflow_engine'] = 'ready'

        # Add RAG status
        if self.rag_manager:
            status['rag'] = self.rag_manager.get_status() if hasattr(self.rag_manager, 'get_status') else 'initialized'

        return status

    def shutdown(self):
        """Shutdown the application gracefully"""
        self.logger.info("Shutting down AvA Application...")

        # Stop performance monitoring
        if self.performance_timer:
            self.performance_timer.stop()

        # Cleanup RAG manager
        if self.rag_manager:
            # Assuming cleanup method exists
            pass

        # Close windows
        if self.main_window:
            self.main_window.close()
        if self.code_viewer:
            self.code_viewer.close()
        if self.terminal_window:
            self.terminal_window.close()

        self.logger.info("AvA Application shutdown complete")

    # Project management methods
    def create_new_project_dialog(self):
        """Create a new project with dialog"""
        if not self.main_window:
            return

        from PySide6.QtWidgets import QInputDialog

        project_name, ok = QInputDialog.getText(
            self.main_window,
            "New Project",
            "Enter project name:"
        )

        if not ok or not project_name:
            return

        # Sanitize project name
        project_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        if not project_name:
            QMessageBox.warning(self.main_window, "Invalid Name", "Please enter a valid project name")
            return

        project_path = self.workspace_dir / project_name

        if project_path.exists():
            reply = QMessageBox.question(
                self.main_window,
                'Project Exists',
                f'Project "{project_name}" already exists. Continue anyway?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        try:
            project_path.mkdir(parents=True, exist_ok=True)
            (project_path / ".gitignore").write_text(".venv/\n__pycache__/\n*.pyc\n", encoding='utf-8')
            (project_path / "main.py").write_text(f'print("Hello from {project_name.strip()}!")\n', encoding='utf-8')

            self.current_project_path = project_path
            self.current_project = project_path.name
            # FIX: Only emit the application's project_loaded signal here
            self.project_loaded.emit(str(project_path))
            # Then manually call the handler to avoid duplicate handling
            self._on_project_loaded(str(project_path))
            self.logger.info(f"New project created: {project_path}")

        except Exception as e:
            self.logger.error(f"Failed to create project: {e}")
            QMessageBox.critical(self.main_window, "Error", f"Failed to create project: {e}")

    def load_existing_project_dialog(self):
        self.logger.info("Loading existing project...")
        if not self.main_window:
            return

        project_dir = QFileDialog.getExistingDirectory(
            self.main_window,
            'Select Project Directory',
            str(self.workspace_dir)
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

    # Error handling methods
    def _handle_execution_error(self, error_info: dict):
        """Handle execution errors and offer to send to Ava"""
        error_msg = error_info.get('error', 'Unknown error')
        file_path = error_info.get('file_path', '')

        # Store error context
        if not self.last_error_context:
            self.last_error_context = ErrorContext()

        # Update error context with captured info
        self.last_error_context.stderr = error_msg
        self.last_error_context.exit_code = error_info.get('return_code', 1)

        # Add to error history
        self.error_history.append(self.last_error_context)
        if len(self.error_history) > 10:  # Keep last 10 errors
            self.error_history.pop(0)

        # Show error in terminal
        self.terminal_error.emit(f"❌ Execution Error: {error_msg}")

        # Send to Ava for analysis
        self._send_error_to_ava(error_msg, file_path)

    def _send_error_to_ava(self, error_msg: str, file_path: str = ""):
        """Send error to Ava for analysis"""
        if file_path:
            prompt = f"I got this error when running {file_path}:\n\n{error_msg}\n\nCan you help fix it?"
        else:
            prompt = f"I got this error:\n\n{error_msg}\n\nCan you help fix it?"

        # Send to chat
        self.chat_message_received.emit(f"🔍 Analyzing error...")

        # Execute error fix workflow
        if self.workflow_engine and self.current_project_path:
            asyncio.create_task(
                self.workflow_engine.execute_workflow(
                    prompt,
                    self.current_project_path,
                    use_existing_project=True
                )
            )

    def run_project_in_terminal(self):
        """Run the current project in terminal with error capture"""
        if not self.current_project_path:
            self.terminal_system_message.emit("❌ No project loaded")
            return

        main_py = self.current_project_path / "main.py"
        if not main_py.exists():
            self.terminal_system_message.emit("❌ No main.py found in project")
            return

        try:
            # Show what we're running
            self.terminal_command_to_run.emit(f"python {main_py}")

            # Run with subprocess to capture output
            result = subprocess.run(
                [sys.executable, str(main_py)],
                cwd=str(self.current_project_path),
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )

            # Capture error context
            error_context = ErrorContext()
            error_context.capture_execution(f"python {main_py}", result, str(self.current_project_path))
            self.last_error_context = error_context

            # Show output
            if result.stdout:
                self.terminal_output.emit(result.stdout)

            if result.stderr:
                self.terminal_error.emit(result.stderr)
                # Detect if it's an actual error (non-zero return code)
                if result.returncode != 0:
                    self.execution_error_detected.emit({
                        'error': result.stderr,
                        'file_path': str(main_py),
                        'return_code': result.returncode
                    })

            if result.returncode == 0:
                self.terminal_system_message.emit("✅ Execution completed successfully")
            else:
                self.terminal_system_message.emit(f"❌ Execution failed with code {result.returncode}")

        except subprocess.TimeoutExpired:
            self.terminal_error.emit("❌ Execution timed out after 30 seconds")
        except Exception as e:
            self.terminal_error.emit(f"❌ Execution error: {str(e)}")
            self.execution_error_detected.emit({
                'error': str(e),
                'file_path': str(main_py)
            })