# core/application.py - FIXED VERSION

import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from PySide6.QtCore import QObject, Signal, QTimer
from PySide6.QtWidgets import QApplication

from gui.main_window import AvAMainWindow
from windows.terminal_window import LLMTerminalWindow
from windows.code_viewer import CodeViewerWindow
from core.workflow_engine import WorkflowEngine
from core.llm_client import LLMClient


class AvAApplication(QObject):
    """
    AvA Application - Fixed
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

        self.current_project = None
        self.active_workflows = {}
        self.performance_metrics = {
            "workflows_completed": 0,
            "total_execution_time": 0.0,
            "error_count": 0,
            "start_time": datetime.now()
        }

        # Monitoring timer
        self.metrics_timer = QTimer()
        self.metrics_timer.timeout.connect(self._collect_metrics)
        self.metrics_timer.start(30000)

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
        self.logger.info("Initializing AvA Application...")

        try:
            # Initialize core services
            self._initialize_core_services()

            # Initialize windows
            self._initialize_windows()

            # Connect components
            self._connect_components()

            # Setup window behaviors
            self._setup_window_behaviors()

            # Check system status
            self._check_system_status()

            self.logger.info("AvA Application initialized successfully")

            # Show main window
            self.main_window.show()

        except Exception as e:
            self.logger.error(f"Failed to initialize AvA: {e}")
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
            self.logger.warning("No LLM services available!")

    def _initialize_windows(self):
        """Initialize all windows"""
        self.logger.info("Initializing windows...")

        # Main window with integrated chat and sidebar
        self.main_window = AvAMainWindow(ava_app=self)
        self.logger.info("Main window initialized")

        # Terminal window (streaming workflow)
        self.terminal_window = LLMTerminalWindow()
        self.logger.info("Terminal window initialized")

        # Code viewer (file browser/editor)
        self.code_viewer = CodeViewerWindow()
        self.logger.info("Code viewer initialized")

        # Workflow engine (coordinates AI workflow)
        self.workflow_engine = WorkflowEngine(
            self.llm_client,
            self.terminal_window,
            self.code_viewer
        )
        self.logger.info("Workflow engine initialized")

    def _connect_components(self):
        """Connect all components with signals and slots"""
        self.logger.info("Connecting components...")

        # Main window connections
        self.main_window.workflow_requested.connect(self._handle_workflow_request)

        # Application signals
        self.workflow_started.connect(self._on_workflow_started)
        self.workflow_completed.connect(self._on_workflow_completed)
        self.error_occurred.connect(self._on_error_occurred)

        # Connect signals to main window
        self.workflow_started.connect(self.main_window.on_workflow_started)
        self.workflow_completed.connect(self.main_window.on_workflow_completed)
        self.error_occurred.connect(self.main_window.on_error_occurred)

        # Connect code viewer to workflow for auto-opening generated files
        if hasattr(self.workflow_engine, 'file_generated'):
            self.workflow_engine.file_generated.connect(self.code_viewer.auto_open_file)

    def _setup_window_behaviors(self):
        """Setup window behaviors and properties"""
        self.main_window.setWindowTitle(f"AvA - AI Development Assistant")
        self.terminal_window.setWindowTitle("AvA - LLM Terminal")
        self.code_viewer.setWindowTitle("AvA - Code Viewer")
        self._position_windows()

    def _position_windows(self):
        """Position windows in a logical layout"""
        screen = QApplication.primaryScreen().geometry()
        self.main_window.setGeometry(50, 50, 1400, 900)
        self.terminal_window.setGeometry(300, screen.height() - 400, 800, 350)
        self.code_viewer.setGeometry(screen.width() - 1000, 50, 950, 700)

    def _check_system_status(self):
        """Check system status and dependencies"""
        status = {
            "llm_available": len(self.llm_client.get_available_models()) > 0,
            "workspace": self.workspace_dir.exists(),
            "windows_ready": all([
                self.main_window is not None,
                self.terminal_window is not None,
                self.code_viewer is not None
            ])
        }
        self.logger.info(f"System status: {status}")
        return status

    def _handle_workflow_request(self, user_prompt: str):
        """Handle workflow execution requests"""
        self.logger.info(f"Workflow requested: {user_prompt[:100]}...")

        if not self._check_system_status()["llm_available"]:
            error_msg = "No LLM services available. Please configure API keys."
            self.terminal_window.log(f"❌ Error: {error_msg}")
            self.error_occurred.emit("llm", error_msg)
            return

        workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.active_workflows[workflow_id] = {
            "prompt": user_prompt,
            "start_time": datetime.now(),
            "status": "running"
        }

        self.workflow_started.emit(user_prompt)
        self._open_terminal()

        try:
            self.workflow_engine.execute_workflow(user_prompt)
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            self.error_occurred.emit("workflow", str(e))

    def _open_terminal(self):
        """Open terminal window"""
        self.terminal_window.show()
        self.terminal_window.raise_()
        self.terminal_window.activateWindow()

    def _open_code_viewer(self):
        """Open code viewer window"""
        self.code_viewer.show()
        self.code_viewer.raise_()
        self.code_viewer.activateWindow()

    def _on_workflow_started(self, prompt: str):
        """Handle workflow started event"""
        self.logger.info(f"Workflow started: {prompt}")

    def _on_workflow_completed(self, result: dict):
        """Handle workflow completion"""
        self.logger.info(f"Workflow completed: {result}")
        self.performance_metrics["workflows_completed"] += 1

        active_workflows = list(self.active_workflows.keys())
        for workflow_id in active_workflows:
            workflow = self.active_workflows[workflow_id]
            if workflow["status"] == "running":
                workflow["status"] = "completed"
                workflow["end_time"] = datetime.now()
                workflow["duration"] = (workflow["end_time"] - workflow["start_time"]).total_seconds()
                break

    def _on_error_occurred(self, component: str, error_message: str):
        """Handle error events"""
        self.logger.error(f"Error in {component}: {error_message}")
        self.performance_metrics["error_count"] += 1

    def _collect_metrics(self):
        """Collect and log performance metrics"""
        uptime = (datetime.now() - self.performance_metrics["start_time"]).total_seconds()
        metrics = {
            "uptime_hours": uptime / 3600,
            "workflows_completed": self.performance_metrics["workflows_completed"],
            "error_count": self.performance_metrics["error_count"],
            "active_workflows": len([w for w in self.active_workflows.values() if w["status"] == "running"])
        }
        self.logger.debug(f"Performance metrics: {metrics}")

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive application status"""
        return {
            "ready": True,
            "llm_models": self.llm_client.get_available_models() if self.llm_client else [],
            "workspace": str(self.workspace_dir),
            "windows": {
                "main": self.main_window.isVisible() if self.main_window else False,
                "terminal": self.terminal_window.isVisible() if self.terminal_window else False,
                "code_viewer": self.code_viewer.isVisible() if self.code_viewer else False
            },
            "metrics": self.performance_metrics.copy(),
            "active_workflows": len(self.active_workflows)
        }

    def shutdown(self):
        """Graceful application shutdown"""
        self.logger.info("Shutting down AvA Application...")
        if self.metrics_timer:
            self.metrics_timer.stop()
        if self.main_window:
            self.main_window.close()
        if self.terminal_window:
            self.terminal_window.close()
        if self.code_viewer:
            self.code_viewer.close()
        self.logger.info("AvA Application shutdown complete")