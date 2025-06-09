# core/enhanced_workflow_engine.py - V5.2 with Robust Logging

import asyncio
import json
import logging
import traceback
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from PySide6.QtCore import QObject, Signal

from core.workflow_services import ArchitectService, CoderService, ReviewerService
from core.project_state_manager import ProjectStateManager


class EnhancedWorkflowEngine(QObject):
    """
    🚀 V5.2 Workflow Engine: Now with robust logging to handle different
    terminal types and ensure stability.
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
        # This can be the StreamingTerminal or None
        self.streaming_terminal = terminal_window
        self.code_viewer = code_viewer
        self.rag_manager = rag_manager
        self.logger = logging.getLogger(__name__)

        self.project_state_manager: ProjectStateManager = None
        self.current_tech_spec: dict = None
        self.is_existing_project_loaded = False
        self.original_project_path: Optional[Path] = None

        def service_log_emitter(agent_name: str, type_key: str, content: str, indent_level: int):
            self.detailed_log_event.emit(agent_name, type_key, content, str(indent_level))

        self.architect_service = ArchitectService(self.llm_client, service_log_emitter, self.rag_manager)
        self.coder_service = CoderService(self.llm_client, service_log_emitter, self.rag_manager)
        self.reviewer_service = ReviewerService(self.llm_client, service_log_emitter, self.rag_manager)

        self._connect_terminal_signals()
        self.logger.info("✅ V5.2 'Robust Logging' Workflow Engine initialized.")

    def _connect_terminal_signals(self):
        # --- MODIFIED: Check if the passed terminal has the correct method ---
        if self.streaming_terminal and hasattr(self.streaming_terminal, 'stream_log_rich'):
            try:
                self.detailed_log_event.connect(self.streaming_terminal.stream_log_rich)
                self.logger.info("Connected workflow engine signals to StreamingTerminal.")
            except Exception as e:
                self.logger.error(f"❌ Failed to connect terminal signals: {e}")
        else:
            # If the right terminal isn't passed, just log to the console to avoid crashing.
            self.logger.warning("No streaming terminal connected. AI logs will print to console.")
            self.detailed_log_event.connect(
                lambda agent, type_key, content, indent:
                self.logger.info(f"[{agent}:{type_key}] {'  ' * int(indent)}{content}")
            )

    async def execute_analysis_workflow(self, project_path_str: str):
        self.logger.info(f"🚀 Starting Analysis workflow for: {project_path_str}...")
        self.analysis_started.emit(project_path_str)
        self.detailed_log_event.emit("WorkflowEngine", "stage_start",
                                     f"🚀 Initializing Analysis for '{Path(project_path_str).name}'...", "0")

        try:
            project_path = Path(project_path_str)
            self.project_state_manager = ProjectStateManager(project_path)
            self.detailed_log_event.emit("WorkflowEngine", "info",
                                         f"Project State Manager initialized. Scanned {len(self.project_state_manager.files)} files.",
                                         "1")

            tech_spec = await self.architect_service.analyze_and_create_spec_from_project(self.project_state_manager)
            if not tech_spec:
                raise Exception(
                    "Analysis failed. Architect could not produce a technical specification from the project files.")

            self.current_tech_spec = tech_spec
            self.is_existing_project_loaded = True
            self.original_project_path = project_path
            self.detailed_log_event.emit("WorkflowEngine", "success", "✅ Analysis complete! Technical spec created.",
                                         "0")

            self.analysis_completed.emit(project_path_str, self.current_tech_spec)
            self.project_loaded.emit(project_path_str)

        except Exception as e:
            self.logger.error(f"❌ Analysis Workflow failed: {e}", exc_info=True)
            self.detailed_log_event.emit("WorkflowEngine", "error", f"❌ Analysis Error: {str(e)}", "0")
            self.project_loaded.emit(project_path_str)

    async def execute_enhanced_workflow(self, user_prompt: str, conversation_context: List[Dict] = None):
        self.logger.info(f"🚀 Starting V5.2 workflow: {user_prompt[:100]}...")
        workflow_start_time = datetime.now()

        if self.is_existing_project_loaded:
            self.workflow_started.emit("Project Modification", user_prompt[:60] + '...')
            self.detailed_log_event.emit("WorkflowEngine", "stage_start",
                                         f"🚀 Initializing MODIFICATION workflow for '{self.original_project_path.name}'...",
                                         "0")
        else:
            self.workflow_started.emit("New Project", user_prompt[:60] + '...')
            self.detailed_log_event.emit("WorkflowEngine", "stage_start", "🚀 Initializing NEW PROJECT workflow...", "0")

        try:
            self.workflow_progress.emit("planning", "Architecting project...")

            if self.is_existing_project_loaded and self.current_tech_spec:
                self.detailed_log_event.emit("Architect", "thought", "Re-architecting based on new request...", 1)
                tech_spec = await self.architect_service.create_tech_spec(user_prompt, conversation_context)
            else:
                self.detailed_log_event.emit("Architect", "thought", "Creating new architecture from scratch...", 1)
                tech_spec = await self.architect_service.create_tech_spec(user_prompt, conversation_context)

            if not tech_spec or 'technical_specs' not in tech_spec:
                raise Exception("Architecture failed. Could not produce a valid Technical Specification Sheet.")

            project_name = tech_spec.get("project_name", "ai_project")

            if self.is_existing_project_loaded:
                project_dir_name = f"{self.original_project_path.name}_MOD_{datetime.now().strftime('%H%M%S')}"
                project_dir = Path("./workspace") / project_dir_name
                self.detailed_log_event.emit("WorkflowEngine", "file_op",
                                             f"Copying original project to '{project_dir}'...", "1")
                try:
                    shutil.copytree(self.original_project_path, project_dir, dirs_exist_ok=True,
                                    ignore=shutil.ignore_patterns('venv', '__pycache__', '.git'))
                except Exception as copy_error:
                    raise Exception(f"Failed to create a copy of the existing project: {copy_error}")
                self.detailed_log_event.emit("WorkflowEngine", "success", "Project copy complete.", "1")
            else:
                project_dir_name = f"{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                project_dir = Path("./workspace") / project_dir_name
                project_dir.mkdir(parents=True, exist_ok=True)
                self.detailed_log_event.emit("WorkflowEngine", "file_op",
                                             f"New project directory created: {project_dir}", "1")

            self.project_loaded.emit(str(project_dir))

            build_order = tech_spec.get("dependency_order", [])
            if not build_order:
                raise Exception("Architecture failed. The dependency order is empty.")

            self.workflow_progress.emit("generation", f"Building {len(build_order)} files...")
            self.detailed_log_event.emit("WorkflowEngine", "info", f"Determined build order: {', '.join(build_order)}",
                                         "1")

            knowledge_packets: Dict[str, Dict] = {}
            results = {"files_created": [], "project_dir": str(project_dir), "failed_files": []}

            for i, filename in enumerate(build_order):
                self.detailed_log_event.emit("WorkflowEngine", "stage_start",
                                             f"Processing file {i + 1}/{len(build_order)}: {filename}", "1")
                file_spec = tech_spec["technical_specs"].get(filename)
                if not file_spec: continue

                dependency_files = file_spec.get("dependencies", [])
                dependency_context = self._build_dependency_context(dependency_files, knowledge_packets)
                project_context = {"description": tech_spec.get("project_description", "")}

                generated_code = await self.coder_service.generate_file_from_spec(filename, file_spec, project_context,
                                                                                  dependency_context)

                review_data, review_passed = await self.reviewer_service.review_code(filename, generated_code,
                                                                                     project_context['description'])

                self.detailed_log_event.emit("WorkflowEngine", "info",
                                             f"Review for {filename} {'passed' if review_passed else 'failed'}. Writing file to disk.",
                                             "2")
                self._write_file(project_dir, filename, generated_code)

                knowledge_packets[filename] = {"spec": file_spec, "source_code": generated_code}
                results["files_created"].append(filename)

            self.workflow_progress.emit("finalization", "Finalizing project...")
            elapsed_time = (datetime.now() - workflow_start_time).total_seconds()
            final_result = await self._finalize_project(results, elapsed_time)

            self.workflow_progress.emit("complete", "Workflow complete!")
            self.workflow_completed.emit(final_result)
            return final_result

        except Exception as e:
            self.logger.error(f"❌ AI Workflow failed: {e}", exc_info=True)
            elapsed_time = (datetime.now() - workflow_start_time).total_seconds()
            self.workflow_progress.emit("error", f"Workflow failed: {str(e)}")
            self.detailed_log_event.emit("WorkflowEngine", "error", f"❌ Workflow Error: {str(e)}", "0")
            self.workflow_completed.emit({"success": False, "error": str(e), "elapsed_time": elapsed_time})
            raise

    def _build_dependency_context(self, dependency_files: List[str], knowledge_packets: Dict[str, Dict]) -> str:
        if not dependency_files: return "This file has no dependencies."
        context_str = ""
        for dep_file in dependency_files:
            if dep_file in knowledge_packets:
                packet = knowledge_packets[dep_file]
                context_str += f"\n\n--- CONTEXT FOR DEPENDENCY: {dep_file} ---\n"
                context_str += f"SPECIFICATION:\n```json\n{json.dumps(packet['spec'], indent=2)}\n```\n"
                context_str += f"FULL SOURCE CODE:\n```python\n{packet['source_code']}\n```\n"
            else:
                self.logger.warning(f"Dependency '{dep_file}' not found. Context incomplete.")
                context_str += f"\n\n--- NOTE: Context for '{dep_file}' was not available. ---\n"
        return context_str

    def _write_file(self, project_dir: Path, filename: str, content: str):
        file_path_obj = project_dir / filename
        file_path_obj.parent.mkdir(parents=True, exist_ok=True)
        file_path_obj.write_text(content, encoding='utf-8')
        self.detailed_log_event.emit("WorkflowEngine", "file_op", f"File written: {file_path_obj}", "2")
        self.file_generated.emit(str(file_path_obj))

    async def _finalize_project(self, results: Dict[str, Any], elapsed_time: float) -> Dict[str, Any]:
        project_dir = results.get("project_dir")
        if project_dir: self.project_loaded.emit(project_dir)
        final_result = {
            "success": len(results.get("failed_files", [])) == 0,
            "project_name": Path(project_dir).name if project_dir else "Unknown",
            "project_dir": project_dir,
            "num_files": len(results.get("files_created", [])),
            "files_created": results.get("files_created", []),
            "failed_files": results.get("failed_files", []),
            "elapsed_time": elapsed_time,
            "strategy": "V5.2 Rolling Context"
        }
        self.detailed_log_event.emit("WorkflowEngine", "success", "Project finalization complete.", "1")
        return final_result