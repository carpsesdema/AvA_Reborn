# core/enhanced_workflow_engine.py - V5.5 with Team Communication + ROBUST FIXES

import asyncio
import json
import logging
import traceback
import shutil
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from PySide6.QtCore import QObject, Signal

from core.project_analyzer import ProjectAnalyzer
from core.project_builder import ProjectBuilder
from core.ai_team_executor import AITeamExecutor
from core.project_finalizer import ProjectFinalizer
from core.workflow_services import ArchitectService, CoderService, ReviewerService
from core.project_state_manager import ProjectStateManager


class EnhancedWorkflowEngine(QObject):
    """
    🚀 V5.5 Workflow Engine: Now with AI team communication and collaborative learning.
    """
    workflow_started = Signal(str, str)
    workflow_completed = Signal(dict)
    workflow_progress = Signal(str, str)
    file_generated = Signal(str)
    project_loaded = Signal(str)
    detailed_log_event = Signal(str, str, str, str)
    task_progress = Signal(int, int)
    analysis_started = Signal(str)
    analysis_completed = Signal(str, dict)

    def __init__(self, llm_client, terminal_window, code_viewer, rag_manager=None):
        super().__init__()
        self.llm_client = llm_client
        self.streaming_terminal = terminal_window
        self.code_viewer = code_viewer
        self.rag_manager = rag_manager
        self.logger = logging.getLogger(__name__)

        self.project_state_manager: Optional[ProjectStateManager] = None
        self.current_tech_spec: Optional[dict] = None
        self.is_existing_project_loaded = False
        self.original_project_path: Optional[Path] = None
        self.active_working_path: Optional[Path] = None  # Tracks the current directory for modifications
        self.original_user_prompt: str = ""

        def service_log_emitter(agent_name: str, type_key: str, content: str, indent_level: int):
            self.detailed_log_event.emit(agent_name, type_key, content, str(indent_level))

        # Initialize services without project state initially
        self.architect_service = ArchitectService(self.llm_client, service_log_emitter, self.rag_manager)
        self.coder_service = CoderService(self.llm_client, service_log_emitter, self.rag_manager)
        self.reviewer_service = ReviewerService(self.llm_client, service_log_emitter, self.rag_manager)

        self._connect_terminal_signals()
        self.logger.info("✅ V5.5 'Team Communication' Workflow Engine initialized.")

    def _connect_terminal_signals(self):
        if self.streaming_terminal and hasattr(self.streaming_terminal, 'stream_log_rich'):
            try:
                self.detailed_log_event.connect(self.streaming_terminal.stream_log_rich)
                self.logger.info("Connected workflow engine signals to StreamingTerminal.")
            except Exception as e:
                self.logger.error(f"❌ Failed to connect terminal signals: {e}")
        else:
            self.logger.warning("No streaming terminal connected. AI logs will print to console.")
            self.detailed_log_event.connect(
                lambda agent, type_key, content, indent:
                self.logger.info(f"[{agent}:{type_key}] {'  ' * int(indent)}{content}")
            )

    def _setup_project_state_and_services(self, project_path: Path):
        """Initialize or update project state manager and connect it to AI services"""
        try:
            if not self.project_state_manager or str(self.project_state_manager.project_root) != str(project_path):
                self.detailed_log_event.emit("WorkflowEngine", "info",
                                             f"Setting up project state for: {project_path.name}", "1")
                self.project_state_manager = ProjectStateManager(project_path)
                self.architect_service.set_project_state(self.project_state_manager)
                self.coder_service.set_project_state(self.project_state_manager)
                self.reviewer_service.set_project_state(self.project_state_manager)
        except Exception as e:
            self.logger.error(f"Failed to setup project state: {e}", exc_info=True)
            self.detailed_log_event.emit("WorkflowEngine", "error", f"Project state setup failed: {e}", "1")

    def _read_gdd_context(self, project_path: Path, project_name: str) -> str:
        """Reads the GDD file content if it exists."""
        gdd_file_path = project_path / f"{project_name}_GDD.md"
        if gdd_file_path.exists():
            self.detailed_log_event.emit("WorkflowEngine", "info", "Found existing GDD, providing to Architect.", "1")
            try:
                return gdd_file_path.read_text(encoding='utf-8')
            except Exception as e:
                self.logger.error(f"Could not read GDD file: {e}")
                return f"Error reading GDD: {e}"
        return ""

    async def execute_analysis_workflow(self, project_path_str: str):
        self.logger.info(f"🚀 Starting Analysis workflow for: {project_path_str}...")
        self.analysis_started.emit(project_path_str)
        try:
            analyzer = ProjectAnalyzer(self.architect_service, self.detailed_log_event.emit, self.logger)
            tech_spec = await analyzer.analyze(project_path_str)

            if not tech_spec:
                raise Exception("Project analysis failed to produce a technical specification.")

            self.current_tech_spec = tech_spec
            self.is_existing_project_loaded = True

            # Set both original and active paths to the loaded project
            self.original_project_path = Path(project_path_str)
            self.active_working_path = Path(project_path_str)

            self.detailed_log_event.emit("WorkflowEngine", "success", "✅ Analysis complete! Ready for modification.",
                                         "0")
            self.analysis_completed.emit(project_path_str, self.current_tech_spec)
            self.project_loaded.emit(project_path_str)

        except Exception as e:
            self.logger.error(f"❌ Analysis Workflow failed: {e}", exc_info=True)
            self.detailed_log_event.emit("WorkflowEngine", "error", f"❌ Analysis Error: {str(e)}", "0")
            self.project_loaded.emit(project_path_str)

    async def execute_enhanced_workflow(self, user_prompt: str, conversation_context: List[Dict] = None):
        self.logger.info(f"🚀 Starting V5.5 workflow: {user_prompt[:100]}...")
        workflow_start_time = datetime.now()
        self.original_user_prompt = user_prompt
        gdd_context = ""

        try:
            # Determine if this is a modification and get necessary context
            if self.is_existing_project_loaded and self.active_working_path:
                self.workflow_started.emit("Project Modification", user_prompt[:60] + '...')
                project_name_for_gdd = self.current_tech_spec.get("project_name", self.active_working_path.name)
                # Read GDD from the active working path, which might be the dev branch
                gdd_context = self._read_gdd_context(self.active_working_path, project_name_for_gdd)
            else:
                self.workflow_started.emit("New Project", user_prompt[:60] + '...')

            # STAGE 1: Architect
            full_user_prompt = f"{user_prompt}\n\n--- GDD CONTEXT ---\n{gdd_context}"
            tech_spec = await self.architect_service.create_tech_spec(full_user_prompt, conversation_context)
            if not tech_spec or 'technical_specs' not in tech_spec:
                raise Exception("Architecture failed. Could not produce a valid tech spec.")

            project_name = tech_spec.get("project_name", "ai_project")

            # STAGE 2: Build Project Directory (with new simplified logic)
            builder = ProjectBuilder("./workspace", self.detailed_log_event.emit, self.logger)

            if self.is_existing_project_loaded:
                # Let the builder handle creating the dev branch if it doesn't exist
                self.active_working_path = builder.setup_project_directory(
                    project_name, is_modification=True, original_user_prompt=user_prompt,
                    original_project_path=self.original_project_path
                )
            else:
                # This is a new project from scratch
                self.active_working_path = builder.setup_project_directory(
                    project_name, is_modification=False, original_user_prompt=user_prompt
                )

            # This is a new project, so update state to reflect it's now "loaded"
            if not self.is_existing_project_loaded:
                self.is_existing_project_loaded = True
                self.original_project_path = self.active_working_path

            # STAGE 3: Execute AI Team
            self._setup_project_state_and_services(self.active_working_path)
            self.project_loaded.emit(str(self.active_working_path))

            executor = AITeamExecutor(self.coder_service, self.reviewer_service, self.detailed_log_event.emit,
                                      self.logger)
            executor.set_project_state(self.project_state_manager)
            ai_results = await executor.execute_workflow(tech_spec)

            # STAGE 4: Finalize Project
            finalizer = ProjectFinalizer(self.detailed_log_event.emit, self.logger)
            finalizer.finalize_project(self.active_working_path, ai_results.get("generated_files", {}), tech_spec,
                                       self.project_state_manager, user_prompt)

            # Announce completion
            elapsed_time = (datetime.now() - workflow_start_time).total_seconds()
            final_result = {
                "success": True,
                "project_name": project_name,
                "project_dir": str(self.active_working_path),
                "num_files": len(ai_results.get("generated_files", {})),
                "files_created": list(ai_results.get("generated_files", {}).keys()),
                "elapsed_time": elapsed_time,
            }
            self.workflow_completed.emit(final_result)
            return final_result

        except Exception as e:
            self.logger.error(f"❌ AI Workflow failed: {e}", exc_info=True)
            elapsed_time = (datetime.now() - workflow_start_time).total_seconds()
            self.workflow_progress.emit("error", f"Workflow failed: {str(e)}")
            self.detailed_log_event.emit("WorkflowEngine", "error", f"❌ Workflow Error: {str(e)}", "0")
            self.workflow_completed.emit({"success": False, "error": str(e), "elapsed_time": elapsed_time})
            raise