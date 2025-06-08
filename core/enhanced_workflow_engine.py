# enhanced_workflow_engine.py - V5 "Rolling Context" Engine

import asyncio
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from PySide6.QtCore import QObject, Signal

# Import our V5 services
from core.workflow_services import ArchitectService, CoderService, ReviewerService


class EnhancedWorkflowEngine(QObject):
    """
    🚀 V5 Workflow Engine: Implements the "Rolling Context" system.
    Builds files sequentially, feeding the specs and source code of completed
    files into the context for the next file, ensuring perfect integration.
    """
    workflow_started = Signal(str)
    workflow_completed = Signal(dict)
    workflow_progress = Signal(str, str)
    file_generated = Signal(str)
    project_loaded = Signal(str)
    detailed_log_event = Signal(str, str, str, str)
    task_progress = Signal(int, int)

    def __init__(self, llm_client, terminal_window, code_viewer, rag_manager=None):
        super().__init__()
        self.llm_client = llm_client
        self.terminal_window = terminal_window
        self.code_viewer = code_viewer
        self.rag_manager = rag_manager
        self.logger = logging.getLogger(__name__)

        def service_log_emitter(agent_name: str, type_key: str, content: str, indent_level: int):
            self.detailed_log_event.emit(agent_name, type_key, content, str(indent_level))

        self.architect_service = ArchitectService(self.llm_client, service_log_emitter, self.rag_manager)
        self.coder_service = CoderService(self.llm_client, service_log_emitter, self.rag_manager)
        self.reviewer_service = ReviewerService(self.llm_client, service_log_emitter, self.rag_manager)

        self._connect_terminal_signals()
        self.logger.info("✅ V5 'Rolling Context' Workflow Engine initialized.")

    def _connect_terminal_signals(self):
        if not self.terminal_window: return
        try:
            self.workflow_progress.connect(self.terminal_window.update_workflow_progress)
            self.task_progress.connect(self.terminal_window.update_task_progress)
            self.detailed_log_event.connect(self.terminal_window.stream_log_rich)
        except Exception as e:
            self.logger.error(f"❌ Failed to connect terminal signals: {e}")

    async def execute_enhanced_workflow(self, user_prompt: str, conversation_context: List[Dict] = None):
        self.logger.info(f"🚀 Starting V5 workflow: {user_prompt[:100]}...")
        workflow_start_time = datetime.now()
        self.workflow_started.emit(user_prompt)
        self.detailed_log_event.emit("WorkflowEngine", "stage_start", "🚀 Initializing V5 'Rolling Context' workflow...",
                                     "0")

        try:
            # --- PHASE 1: ARCHITECTURE ---
            self.workflow_progress.emit("planning", "Phase 1: Architecting project technical specification...")
            self._update_task_progress(1, 4)
            tech_spec = await self.architect_service.create_tech_spec(user_prompt, conversation_context)
            if not tech_spec or 'technical_specs' not in tech_spec:
                raise Exception("Architecture failed. Could not produce a valid Technical Specification Sheet.")

            project_name = tech_spec.get("project_name", "ai_project")
            project_dir = Path("./workspace") / f"{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            project_dir.mkdir(parents=True, exist_ok=True)
            self.detailed_log_event.emit("WorkflowEngine", "file_op", f"Project directory created: {project_dir}", "1")

            # --- PHASE 2: SEQUENTIAL, CONTEXT-AWARE GENERATION ---
            build_order = tech_spec.get("dependency_order", [])
            if not build_order:
                raise Exception("Architecture failed. The dependency order is empty.")

            self.workflow_progress.emit("generation", f"Phase 2: Building {len(build_order)} files sequentially...")
            self._update_task_progress(2, 4)
            self.detailed_log_event.emit("WorkflowEngine", "info", f"Determined build order: {', '.join(build_order)}",
                                         "1")

            knowledge_packets: Dict[str, Dict] = {}
            results = {"files_created": [], "project_dir": str(project_dir), "failed_files": []}

            for i, filename in enumerate(build_order):
                self.detailed_log_event.emit("WorkflowEngine", "stage_start",
                                             f"Processing file {i + 1}/{len(build_order)}: {filename}", "1")

                file_spec = tech_spec["technical_specs"].get(filename)
                if not file_spec:
                    self.logger.error(f"Spec for {filename} not found. Skipping.")
                    results["failed_files"].append(filename)
                    continue

                # Assemble the "Knowledge Packet" for the dependencies
                dependency_files = file_spec.get("dependencies", [])
                dependency_context = self._build_dependency_context(dependency_files, knowledge_packets)

                project_context = {"description": tech_spec.get("project_description", "")}

                # Generate the full file code
                generated_code = await self.coder_service.generate_file_from_spec(filename, file_spec, project_context,
                                                                                  dependency_context)

                if "# FALLBACK" in generated_code:
                    self.detailed_log_event.emit("WorkflowEngine", "error",
                                                 f"Coder service returned a fallback for {filename}. Halting file processing.",
                                                 "2")
                    results["failed_files"].append(filename)
                    continue

                # Final quality check
                review_data, review_passed = await self.reviewer_service.review_code(filename, generated_code,
                                                                                     project_context['description'])

                if not review_passed:
                    self.detailed_log_event.emit("Reviewer", "warning",
                                                 f"⚠️ Final review for '{filename}' failed, but proceeding with generated code.",
                                                 "2")

                self._write_file(project_dir, filename, generated_code)

                # Create and store the knowledge packet for this completed file
                knowledge_packets[filename] = {
                    "spec": file_spec,
                    "source_code": generated_code
                }
                results["files_created"].append(filename)
                self.detailed_log_event.emit("WorkflowEngine", "success", f"✅ Completed processing for {filename}", "1")

            # --- PHASE 3: FINALIZATION ---
            self._update_task_progress(3, 4)
            self.workflow_progress.emit("finalization", "Finalizing project...")
            elapsed_time = (datetime.now() - workflow_start_time).total_seconds()
            final_result = await self._finalize_project(results, elapsed_time)
            self._update_task_progress(4, 4)

            self.workflow_progress.emit("complete", "AI workflow completed successfully")
            self.detailed_log_event.emit("WorkflowEngine", "success", "✅ AI workflow completed successfully!", "0")
            self.workflow_completed.emit(final_result)
            return final_result

        except Exception as e:
            self.logger.error(f"❌ AI Workflow failed: {e}", exc_info=True)
            elapsed_time = (datetime.now() - workflow_start_time).total_seconds()
            self.workflow_progress.emit("error", f"AI workflow failed: {str(e)}")
            self.detailed_log_event.emit("WorkflowEngine", "error", f"❌ Workflow Error: {str(e)}", "0")
            self.detailed_log_event.emit("WorkflowEngine", "debug", traceback.format_exc(), "1")
            self.workflow_completed.emit({"success": False, "error": str(e), "elapsed_time": elapsed_time})
            raise

    def _build_dependency_context(self, dependency_files: List[str], knowledge_packets: Dict[str, Dict]) -> str:
        """Constructs a context string from the knowledge packets of completed dependencies."""
        if not dependency_files:
            return "This file has no dependencies."

        context_str = ""
        for dep_file in dependency_files:
            if dep_file in knowledge_packets:
                packet = knowledge_packets[dep_file]
                context_str += f"\n\n--- CONTEXT FOR DEPENDENCY: {dep_file} ---\n"
                context_str += f"SPECIFICATION:\n```json\n{json.dumps(packet['spec'], indent=2)}\n```\n"
                context_str += f"FULL SOURCE CODE:\n```python\n{packet['source_code']}\n```\n"
            else:
                self.logger.warning(
                    f"Dependency '{dep_file}' was not found in knowledge packets. Context will be incomplete.")
                context_str += f"\n\n--- NOTE: Context for '{dep_file}' was not available. ---\n"

        return context_str

    def _write_file(self, project_dir: Path, filename: str, content: str):
        file_path_obj = project_dir / filename
        file_path_obj.parent.mkdir(parents=True, exist_ok=True)
        file_path_obj.write_text(content, encoding='utf-8')
        self.detailed_log_event.emit("WorkflowEngine", "file_op", f"File written: {file_path_obj}", "2")
        self.file_generated.emit(str(file_path_obj))

    def _update_task_progress(self, completed: int, total: int):
        self.task_progress.emit(completed, total)

    async def _finalize_project(self, results: Dict[str, Any], elapsed_time: float) -> Dict[str, Any]:
        project_dir = results.get("project_dir")
        if project_dir:
            self.project_loaded.emit(project_dir)
            self.detailed_log_event.emit("WorkflowEngine", "file_op", f"Project loaded into UI: {project_dir}", "1")

        final_result = {
            "success": len(results.get("failed_files", [])) == 0,
            "project_name": Path(project_dir).name if project_dir else "Unknown",
            "project_dir": project_dir,
            "num_files": len(results.get("files_created", [])),
            "files_created": results.get("files_created", []),
            "failed_files": results.get("failed_files", []),
            "elapsed_time": elapsed_time,
            "strategy": "V5 Rolling Context"
        }
        self.detailed_log_event.emit("WorkflowEngine", "success", "Project finalization complete.", "1")
        return final_result

    def execute_workflow(self, prompt: str):
        asyncio.create_task(self.execute_enhanced_workflow(prompt, []))