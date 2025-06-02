# core/application_fixed.py - PROPERLY WORKING APPLICATION

import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from PySide6.QtCore import QObject, Signal, QTimer
from PySide6.QtWidgets import QApplication

# Import the FIXED components
from gui.main_window import AvAMainWindow
from windows.terminal_window import LLMTerminalWindow
from windows.code_viewer import CodeViewerWindow
from core.workflow_engine import WorkflowEngine
from core.llm_client import LLMClient


class AvAApplication(QObject):
    """
    FIXED AvA Application - Proper chat and workflow separation
    """

    # Signals for status updates
    workflow_started = Signal(str)
    workflow_completed = Signal(dict)
    error_occurred = Signal(str, str)

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

        # Application state
        self.workspace_dir = Path("./workspace")
        self.workspace_dir.mkdir(exist_ok=True)

        self.current_project = "Default Project"
        self.current_session = "Main Chat"
        self.active_workflows = {}

    def _setup_logging(self):
        """Setup application logging"""
        log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "ava.log"),
                logging.StreamHandler()
            ]
        )

    def initialize(self):
        """Initialize all application components"""
        self.logger.info("🚀 Initializing AvA Application...")

        try:
            # Initialize core services
            self._initialize_core_services()

            # Initialize windows
            self._initialize_windows()

            # Connect components PROPERLY
            self._connect_components()

            # Setup window behaviors
            self._setup_window_behaviors()

            # Check system status
            status = self._check_system_status()
            self.logger.info(f"System status: {status}")

            self.logger.info("✅ AvA Application initialized successfully")

            # Show main window
            self.main_window.show()

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize AvA: {e}")
            self.error_occurred.emit("initialization", str(e))
            raise

    def _initialize_core_services(self):
        """Initialize core business logic services"""
        self.logger.info("Initializing core services...")

        # LLM Client
        self.llm_client = LLMClient()
        available_models = self.llm_client.get_available_models()
        self.logger.info(f"Available LLM models: {available_models}")

        if not available_models or available_models == ["No LLM services available"]:
            self.logger.warning("⚠️  No LLM services available!")

    def _initialize_windows(self):
        """Initialize all windows"""
        self.logger.info("Initializing windows...")

        # Main window with FIXED chat interface
        self.main_window = AvAMainWindow(ava_app=self)
        self.logger.info("✅ Main window initialized")

        # Terminal window (LLM workflow communication ONLY)
        self.terminal_window = LLMTerminalWindow()
        self.logger.info("✅ Terminal window initialized")

        # Code viewer (file browser/editor)
        self.code_viewer = CodeViewerWindow()
        self.logger.info("✅ Code viewer initialized")

        # FIXED Workflow engine (only triggers on actual build requests)
        self.workflow_engine = WorkflowEngine(
            self.llm_client,
            self.terminal_window,
            self.code_viewer
        )
        self.logger.info("✅ Workflow engine initialized")

    def _connect_components(self):
        """Connect all components with signals and slots - FIXED"""
        self.logger.info("Connecting components...")

        # FIXED: Main window workflow requests (only for actual build requests)
        self.main_window.workflow_requested.connect(self._handle_workflow_request)

        # Application signals
        self.workflow_started.connect(self._on_workflow_started)
        self.workflow_completed.connect(self._on_workflow_completed)
        self.error_occurred.connect(self._on_error_occurred)

        # Connect workflow engine signals to main window
        self.workflow_engine.workflow_started.connect(self.main_window.on_workflow_started)
        self.workflow_engine.workflow_completed.connect(self.main_window.on_workflow_completed)

        # Connect file generation to code viewer
        self.workflow_engine.file_generated.connect(self._on_file_generated)
        self.workflow_engine.project_loaded.connect(self._on_project_loaded)

    def _setup_window_behaviors(self):
        """Setup window behaviors and properties"""
        self.main_window.setWindowTitle(f"AvA - AI Development Assistant")
        self.terminal_window.setWindowTitle("AvA - LLM Workflow Terminal")
        self.code_viewer.setWindowTitle("AvA - Code Viewer")
        self._position_windows()

    def _position_windows(self):
        """Position windows in a logical layout"""
        screen = QApplication.primaryScreen().geometry()

        # Main window - center
        self.main_window.setGeometry(100, 50, 1400, 900)

        # Terminal - bottom right
        self.terminal_window.setGeometry(
            screen.width() - 850,
            screen.height() - 400,
            800, 350
        )

        # Code viewer - right side
        self.code_viewer.setGeometry(
            screen.width() - 1000,
            50,
            950, 700
        )

    def _check_system_status(self):
        """Check system status and dependencies"""
        status = {
            "llm_available": len(self.llm_client.get_available_models()) > 0,
            "workspace": self.workspace_dir.exists(),
            "windows_ready": all([
                self.main_window is not None,
                self.terminal_window is not None,
                self.code_viewer is not None,
                self.workflow_engine is not None
            ])
        }
        return status

    def _handle_workflow_request(self, user_prompt: str):
        """
        FIXED: Handle workflow execution requests
        Only processes legitimate build requests
        """
        self.logger.info(f"Workflow request received: {user_prompt[:100]}...")

        # Check if LLM is available
        if not self._check_system_status()["llm_available"]:
            error_msg = "No LLM services available. Please configure API keys."
            self.terminal_window.log(f"❌ Error: {error_msg}")
            self.error_occurred.emit("llm", error_msg)
            return

        # Open terminal to show workflow progress
        self._open_terminal()

        # Create workflow tracking
        workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.active_workflows[workflow_id] = {
            "prompt": user_prompt,
            "start_time": datetime.now(),
            "status": "running"
        }

        self.workflow_started.emit(user_prompt)

        try:
            # Pass to workflow engine (which has its own validation)
            self.workflow_engine.execute_workflow(user_prompt)
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            self.error_occurred.emit("workflow", str(e))

    def _open_terminal(self):
        """Open terminal window and bring to front"""
        self.terminal_window.show()
        self.terminal_window.raise_()
        self.terminal_window.activateWindow()

    def _open_code_viewer(self):
        """Open code viewer window and bring to front"""
        self.code_viewer.show()
        self.code_viewer.raise_()
        self.code_viewer.activateWindow()

    def _on_file_generated(self, file_path: str):
        """Handle file generation - auto-open in code viewer"""
        self.logger.info(f"File generated: {file_path}")
        if self.code_viewer:
            self.code_viewer.auto_open_file(file_path)

    def _on_project_loaded(self, project_path: str):
        """Handle project loading"""
        self.logger.info(f"Project loaded: {project_path}")
        self.current_project = Path(project_path).name
        # Auto-open code viewer
        self._open_code_viewer()

    def _on_workflow_started(self, prompt: str):
        """Handle workflow started event"""
        self.logger.info(f"✅ Workflow started: {prompt}")

    def _on_workflow_completed(self, result: dict):
        """Handle workflow completion"""
        self.logger.info(f"✅ Workflow completed: {result}")

        # Update active workflows
        for workflow_id, workflow in self.active_workflows.items():
            if workflow["status"] == "running":
                workflow["status"] = "completed"
                workflow["end_time"] = datetime.now()
                workflow["duration"] = (workflow["end_time"] - workflow["start_time"]).total_seconds()
                break

    def _on_error_occurred(self, component: str, error_message: str):
        """Handle error events"""
        self.logger.error(f"❌ Error in {component}: {error_message}")

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive application status"""
        return {
            "ready": True,
            "llm_models": self.llm_client.get_available_models() if self.llm_client else [],
            "workspace": str(self.workspace_dir),
            "current_project": self.current_project,
            "current_session": self.current_session,
            "windows": {
                "main": self.main_window.isVisible() if self.main_window else False,
                "terminal": self.terminal_window.isVisible() if self.terminal_window else False,
                "code_viewer": self.code_viewer.isVisible() if self.code_viewer else False
            },
            "active_workflows": len(self.active_workflows)
        }

    def shutdown(self):
        """Graceful application shutdown"""
        self.logger.info("Shutting down AvA Application...")

        if self.main_window:
            self.main_window.close()
        if self.terminal_window:
            self.terminal_window.close()
        if self.code_viewer:
            self.code_viewer.close()

        self.logger.info("AvA Application shutdown complete")


# FIXED main.py
def main():
    """FIXED Main entry point"""

    app = QApplication(sys.argv)
    app.setApplicationName("AvA")
    app.setApplicationDisplayName("AvA - AI Development Assistant")
    app.setApplicationVersion("2.0")

    try:
        # Create and initialize AvA application
        ava_app = AvAApplication()
        ava_app.initialize()

        print("🚀 AvA launched successfully!")
        status = ava_app.get_status()
        print(f"📊 Status: {status}")

        return app.exec()

    except Exception as e:
        print(f"❌ Failed to launch AvA: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())