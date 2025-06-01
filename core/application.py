import asyncio
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

from PySide6.QtCore import QObject, Signal, QTimer
import qasync

from core.config import ConfigManager
from core.llm_manager import LLMManager
from core.project_manager import ProjectManager
from core.workflow_engine import WorkflowEngine
from core.rag_service import RAGService
from core.plugin_manager import PluginManager


@dataclass
class WorkflowResult:
    """Professional workflow result with comprehensive metadata"""
    success: bool
    files_generated: List[str]
    execution_time: float
    tokens_used: Dict[str, int]
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]


class AvAApplication(QObject):
    """
    Professional core application managing all services and coordination.
    This is the heart of AvA - handles all business logic and orchestration.
    """

    # Professional signals with comprehensive data
    workflow_started = Signal(str, dict)  # (prompt, metadata)
    workflow_progress = Signal(str, int, str)  # (stage, progress_percent, details)
    workflow_completed = Signal(WorkflowResult)
    error_occurred = Signal(str, str, dict)  # (component, message, context)

    # Service status signals
    llm_status_changed = Signal(str, str, bool)  # (model_name, status, is_available)
    rag_status_changed = Signal(str, dict)  # (status, metadata)
    project_changed = Signal(str, dict)  # (project_path, project_info)

    def __init__(self, config: ConfigManager):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Core services
        self.llm_manager: Optional[LLMManager] = None
        self.project_manager: Optional[ProjectManager] = None
        self.workflow_engine: Optional[WorkflowEngine] = None
        self.rag_service: Optional[RAGService] = None
        self.plugin_manager: Optional[PluginManager] = None

        # State management
        self.current_project: Optional[str] = None
        self.active_workflows: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, Any] = {}

        # Professional monitoring
        self._setup_monitoring()

    async def initialize(self):
        """Initialize all application services with proper error handling"""

        self.logger.info("Initializing AvA Core Application")

        try:
            # 1. Initialize LLM Manager
            self.logger.info("Initializing LLM Manager...")
            self.llm_manager = LLMManager(self.config)
            await self.llm_manager.initialize()
            self.llm_manager.status_changed.connect(self.llm_status_changed.emit)

            # 2. Initialize Project Manager
            self.logger.info("Initializing Project Manager...")
            self.project_manager = ProjectManager(self.config)
            await self.project_manager.initialize()
            self.project_manager.project_changed.connect(self.project_changed.emit)

            # 3. Initialize RAG Service
            self.logger.info("Initializing RAG Service...")
            self.rag_service = RAGService(self.config)
            await self.rag_service.initialize()
            self.rag_service.status_changed.connect(self.rag_status_changed.emit)

            # 4. Initialize Workflow Engine
            self.logger.info("Initializing Workflow Engine...")
            self.workflow_engine = WorkflowEngine(
                llm_manager=self.llm_manager,
                rag_service=self.rag_service,
                project_manager=self.project_manager,
                config=self.config
            )
            await self.workflow_engine.initialize()
            self._connect_workflow_signals()

            # 5. Initialize Plugin Manager
            self.logger.info("Initializing Plugin Manager...")
            self.plugin_manager = PluginManager(self.config)
            await self.plugin_manager.initialize()

            # 6. Load default project
            await self._load_default_project()

            self.logger.info("AvA Core Application initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize AvA Core: {e}")
            self.error_occurred.emit("core", f"Initialization failed: {e}", {})
            raise

    def _setup_monitoring(self):
        """Setup professional monitoring and metrics collection"""

        # Performance monitoring timer
        self.metrics_timer = QTimer()
        self.metrics_timer.timeout.connect(self._collect_metrics)
        self.metrics_timer.start(30000)  # Collect metrics every 30 seconds

        # Initialize metrics
        self.performance_metrics = {
            "workflows_completed": 0,
            "total_tokens_used": 0,
            "average_workflow_time": 0.0,
            "error_count": 0,
            "uptime": datetime.now()
        }

    def _connect_workflow_signals(self):
        """Connect workflow engine signals to application signals"""

        if self.workflow_engine:
            self.workflow_engine.workflow_started.connect(
                lambda prompt, meta: self.workflow_started.emit(prompt, meta)
            )
            self.workflow_engine.workflow_progress.connect(
                lambda stage, progress, details: self.workflow_progress.emit(stage, progress, details)
            )
            self.workflow_engine.workflow_completed.connect(
                self._handle_workflow_completion
            )
            self.workflow_engine.error_occurred.connect(
                lambda comp, msg, ctx: self.error_occurred.emit(comp, msg, ctx)
            )

    async def _load_default_project(self):
        """Load the default project or create one if none exists"""

        try:
            if self.project_manager:
                projects = await self.project_manager.list_projects()

                if projects:
                    # Load the most recent project
                    default_project = max(projects, key=lambda p: p.get("last_modified", 0))
                    await self.set_current_project(default_project["path"])
                else:
                    # Create a default project
                    default_path = await self.project_manager.create_project(
                        name="Default Project",
                        description="Default AvA project"
                    )
                    await self.set_current_project(default_path)

        except Exception as e:
            self.logger.warning(f"Failed to load default project: {e}")

    @qasync.asyncSlot(str)
    async def execute_workflow(self, user_prompt: str) -> WorkflowResult:
        """
        Execute a complete development workflow from user prompt to finished code.
        Professional implementation with comprehensive error handling and monitoring.
        """

        workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()

        try:
            self.logger.info(f"Starting workflow {workflow_id}: {user_prompt[:100]}...")

            # Validate prerequisites
            if not self._validate_workflow_prerequisites():
                raise RuntimeError("Workflow prerequisites not met")

            # Create workflow context
            workflow_context = {
                "id": workflow_id,
                "prompt": user_prompt,
                "project_path": self.current_project,
                "timestamp": start_time.isoformat(),
                "config": self.config.get_workflow_config()
            }

            # Track active workflow
            self.active_workflows[workflow_id] = workflow_context

            # Execute workflow through engine
            result = await self.workflow_engine.execute(user_prompt, workflow_context)

            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()

            # Create professional result
            workflow_result = WorkflowResult(
                success=result["success"],
                files_generated=result.get("files", []),
                execution_time=execution_time,
                tokens_used=result.get("tokens_used", {}),
                errors=result.get("errors", []),
                warnings=result.get("warnings", []),
                metadata={
                    "workflow_id": workflow_id,
                    "project_path": self.current_project,
                    "model_info": await self.llm_manager.get_model_info()
                }
            )

            # Update metrics
            self._update_metrics(workflow_result)

            # Emit completion signal
            self.workflow_completed.emit(workflow_result)

            self.logger.info(f"Workflow {workflow_id} completed successfully in {execution_time:.2f}s")
            return workflow_result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_result = WorkflowResult(
                success=False,
                files_generated=[],
                execution_time=execution_time,
                tokens_used={},
                errors=[str(e)],
                warnings=[],
                metadata={"workflow_id": workflow_id, "error": True}
            )

            self.performance_metrics["error_count"] += 1
            self.error_occurred.emit("workflow", str(e), {"workflow_id": workflow_id})

            self.logger.error(f"Workflow {workflow_id} failed: {e}")
            return error_result

        finally:
            # Clean up active workflow tracking
            self.active_workflows.pop(workflow_id, None)

    def _validate_workflow_prerequisites(self) -> bool:
        """Validate that all prerequisites are met for workflow execution"""

        if not self.llm_manager or not self.llm_manager.is_ready():
            self.logger.error("LLM Manager not ready")
            return False

        if not self.workflow_engine or not self.workflow_engine.is_ready():
            self.logger.error("Workflow Engine not ready")
            return False

        if not self.current_project:
            self.logger.error("No active project")
            return False

        return True

    def _handle_workflow_completion(self, result: Dict[str, Any]):
        """Handle workflow completion from engine"""
        # This will be called by the workflow engine
        # Additional processing can be done here
        pass

    def _update_metrics(self, result: WorkflowResult):
        """Update performance metrics with workflow result"""

        self.performance_metrics["workflows_completed"] += 1

        # Update token usage
        for model, tokens in result.tokens_used.items():
            if model not in self.performance_metrics:
                self.performance_metrics[f"tokens_{model}"] = 0
            self.performance_metrics[f"tokens_{model}"] += tokens

        # Update average workflow time
        current_avg = self.performance_metrics["average_workflow_time"]
        count = self.performance_metrics["workflows_completed"]
        new_avg = (current_avg * (count - 1) + result.execution_time) / count
        self.performance_metrics["average_workflow_time"] = new_avg

    def _collect_metrics(self):
        """Collect and log performance metrics"""

        uptime = (datetime.now() - self.performance_metrics["uptime"]).total_seconds()

        metrics_summary = {
            "uptime_hours": uptime / 3600,
            "workflows_completed": self.performance_metrics["workflows_completed"],
            "average_workflow_time": self.performance_metrics["average_workflow_time"],
            "error_count": self.performance_metrics["error_count"],
            "active_workflows": len(self.active_workflows)
        }

        self.logger.debug(f"Performance metrics: {metrics_summary}")

    async def set_current_project(self, project_path: str):
        """Set the current active project"""

        try:
            if self.project_manager:
                project_info = await self.project_manager.load_project(project_path)
                self.current_project = project_path
                self.project_changed.emit(project_path, project_info)
                self.logger.info(f"Set current project to: {project_path}")

        except Exception as e:
            self.logger.error(f"Failed to set current project: {e}")
            self.error_occurred.emit("project", f"Failed to load project: {e}", {"path": project_path})

    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive application status"""

        status = {
            "ready": True,
            "current_project": self.current_project,
            "active_workflows": len(self.active_workflows),
            "metrics": self.performance_metrics.copy()
        }

        # Add service statuses
        if self.llm_manager:
            status["llm_status"] = await self.llm_manager.get_status()

        if self.rag_service:
            status["rag_status"] = await self.rag_service.get_status()

        if self.project_manager:
            status["project_status"] = await self.project_manager.get_status()

        return status

    async def shutdown(self):
        """Professional shutdown with cleanup"""

        self.logger.info("Shutting down AvA Core Application")

        try:
            # Stop metrics collection
            if hasattr(self, 'metrics_timer'):
                self.metrics_timer.stop()

            # Shutdown services in reverse order
            if self.plugin_manager:
                await self.plugin_manager.shutdown()

            if self.workflow_engine:
                await self.workflow_engine.shutdown()

            if self.rag_service:
                await self.rag_service.shutdown()

            if self.project_manager:
                await self.project_manager.shutdown()

            if self.llm_manager:
                await self.llm_manager.shutdown()

            self.logger.info("AvA Core Application shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.performance_metrics.copy()

    def get_active_workflows(self) -> Dict[str, Any]:
        """Get information about currently active workflows"""
        return self.active_workflows.copy()