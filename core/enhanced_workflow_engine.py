# enhanced_workflow_engine.py - FINAL, STREAMLINED VERSION

import asyncio
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from PySide6.QtCore import QObject, Signal

# Import services from their new locations
from core.workflow_services import EnhancedPlannerService, EnhancedCoderService, EnhancedAssemblerService, \
    StructureService


class EnhancedWorkflowEngine(QObject):
    """
    🚀 Enhanced Workflow Engine with a self-healing, three-tiered recovery workflow.
    """
    workflow_started = Signal(str)
    workflow_completed = Signal(dict)
    workflow_progress = Signal(str, str)  # stage, description (overall stage)
    file_generated = Signal(str)
    project_loaded = Signal(str)

    detailed_log_event = Signal(str, str, str, str)
    task_progress = Signal(int, int)

    # Constants for the self-healing loop
    MAX_PATCH_ATTEMPTS = 3
    MAX_ESCALATIONS = 1  # Max times to re-plan a file

    def __init__(self, llm_client, terminal_window, code_viewer, rag_manager=None):
        super().__init__()
        self.llm_client = llm_client
        self.terminal_window = terminal_window
        self.code_viewer = code_viewer
        self.rag_manager = rag_manager
        self.logger = logging.getLogger(__name__)

        self.workflow_stats = {
            "cache_stats": {"cache_size": 0, "hit_rate": 0.0},
            "workflow_state": {"stage": "idle", "completed_tasks": 0, "total_tasks": 0}
        }

        def service_log_emitter(agent_name: str, type_key: str, content: str, indent_level: int):
            self.detailed_log_event.emit(agent_name, type_key, content, str(indent_level))

        self.structure_service = StructureService(self.llm_client, service_log_emitter, self.rag_manager)
        self.planner_service = EnhancedPlannerService(self.llm_client, service_log_emitter, self.rag_manager)
        self.coder_service = EnhancedCoderService(self.llm_client, service_log_emitter, self.rag_manager)
        self.assembler_service = EnhancedAssemblerService(self.llm_client, service_log_emitter, self.rag_manager)

        self._connect_terminal_signals()
        self.logger.info(
            "✅ Enhanced Workflow Engine initialized with a three-tiered self-healing loop.")

    def _connect_terminal_signals(self):
        if not self.terminal_window: return
        try:
            self.workflow_progress.connect(self.terminal_window.update_workflow_progress)
            self.task_progress.connect(self.terminal_window.update_task_progress)
            if hasattr(self.terminal_window, 'stream_log_rich'):
                self.detailed_log_event.connect(self.terminal_window.stream_log_rich)
            else:
                self.logger.error("Terminal does not have a suitable method to connect detailed_log_event.")
        except Exception as e:
            self.logger.error(f"❌ Failed to connect terminal signals: {e}")

    async def execute_enhanced_workflow(self, user_prompt: str, conversation_context: List[Dict] = None):
        self.logger.info(f"🚀 Starting enhanced workflow: {user_prompt[:100]}...")
        workflow_start_time = datetime.now()
        self.workflow_started.emit(user_prompt)
        self.detailed_log_event.emit("WorkflowEngine", "stage_start", "🚀 Initializing workflow", "0")

        try:
            self.workflow_progress.emit("planning", "Phase 1: Determining file structure...")
            self._update_task_progress(1, 5)
            high_level_plan = await self.structure_service.create_project_structure(user_prompt, conversation_context)
            if not high_level_plan or 'files' not in high_level_plan:
                raise Exception("Structuring failed. Could not determine a valid file structure.")

            project_name = high_level_plan.get("project_name", "ai_project")
            files_to_process = high_level_plan.get('files', [])
            total_files = len(files_to_process)

            project_dir = Path("./workspace") / f"{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            project_dir.mkdir(parents=True, exist_ok=True)
            self.detailed_log_event.emit("WorkflowEngine", "file_op", f"Project directory created: {project_dir}", "1")

            self.workflow_progress.emit("generation", f"Phase 2: Processing {total_files} files...")
            self._update_task_progress(2, 5)
            results = {"files_created": [], "project_dir": str(project_dir), "failed_files": []}

            processing_tasks = [
                self._process_single_file(file_info, high_level_plan, project_dir)
                for file_info in files_to_process
            ]
            files_processed_results = await asyncio.gather(*processing_tasks)

            # Categorize results
            for result_tuple in files_processed_results:
                if result_tuple and result_tuple[1]:  # (filename, success_bool)
                    results["files_created"].append(result_tuple[0])
                elif result_tuple:
                    results["failed_files"].append(result_tuple[0])

            self._update_task_progress(4, 5)

            self.workflow_progress.emit("finalization", "Finalizing project...")
            elapsed_time = (datetime.now() - workflow_start_time).total_seconds()
            final_result = await self._finalize_project(results, elapsed_time)
            self._update_task_progress(5, 5)

            self.workflow_progress.emit("complete", "AI workflow completed successfully")
            self.detailed_log_event.emit("WorkflowEngine", "success", "✅ AI workflow completed successfully!", "0")
            self.workflow_completed.emit(final_result)

            self.logger.info("✅ Conversational AI workflow completed successfully")
            return final_result

        except Exception as e:
            self.logger.error(f"❌ AI Workflow failed: {e}", exc_info=True)
            elapsed_time = (datetime.now() - workflow_start_time).total_seconds()
            self.workflow_progress.emit("error", f"AI workflow failed: {str(e)}")
            self.detailed_log_event.emit("WorkflowEngine", "error", f"❌ Workflow Error: {str(e)}", "0")
            self.detailed_log_event.emit("WorkflowEngine", "debug", traceback.format_exc(), "1")
            self.workflow_completed.emit({"success": False, "error": str(e), "elapsed_time": elapsed_time})
            raise

    async def _process_single_file(self, file_info: dict, high_level_plan: dict, project_dir: Path) -> Optional[
        tuple[str, bool]]:
        filename = file_info.get("filename")
        if not filename: return None

        self.detailed_log_event.emit("WorkflowEngine", "info", f"Starting pipeline for: {filename}", "1")

        escalation_count = 0
        last_failed_review = {}
        last_failed_code = ""

        while escalation_count <= self.MAX_ESCALATIONS:
            file_spec, other_files_context = await self._plan_file(
                filename, file_info, high_level_plan, escalation_count > 0, last_failed_review
            )
            if not file_spec:
                self.detailed_log_event.emit("WorkflowEngine", "error",
                                             f"Planning failed for {filename}. Aborting file.", "1")
                return filename, False

            project_context = {"description": high_level_plan.get("project_description", "")}
            task_results = await self.coder_service.execute_tasks_for_file(file_spec, project_context)

            final_code, review_passed, last_failed_review = await self._review_and_patch_loop(filename, task_results,
                                                                                              high_level_plan,
                                                                                              file_spec)

            if review_passed:
                self._write_file(project_dir, filename, final_code)
                return filename, True

            last_failed_code = final_code  # Store the last failed version
            escalation_count += 1
            if escalation_count > self.MAX_ESCALATIONS:
                self.detailed_log_event.emit("WorkflowEngine", "error",
                                             f"⛔️ CRITICAL FAILURE on {filename}. Max escalations reached.", "1")
                # --- TIER 3: THE ULTIMATE FIXER ---
                self.detailed_log_event.emit("WorkflowEngine", "fallback", f"Engaging Ultimate Fixer for {filename}...",
                                             1)

                ultimate_code = await self.assembler_service.create_ultimate_fix(
                    file_path=filename,
                    file_purpose=file_info.get("purpose", ""),
                    plan=high_level_plan,
                    last_failed_code=last_failed_code,
                    last_review_feedback=last_failed_review
                )
                if ultimate_code:
                    self.detailed_log_event.emit("WorkflowEngine", "success",
                                                 f"Ultimate Fixer completed {filename}. Writing file.", 1)
                    self._write_file(project_dir, filename, ultimate_code)
                    return filename, True
                else:
                    self.detailed_log_event.emit("WorkflowEngine", "error",
                                                 f"⛔️ Ultimate Fixer FAILED for {filename}. Halting work.", 1)
                    return filename, False
            else:
                self.detailed_log_event.emit("WorkflowEngine", "warning",
                                             f"Escalating {filename} back to Planner for re-evaluation.", "1")

    def _write_file(self, project_dir: Path, filename: str, content: str):
        """Helper to write file content and emit signals."""
        file_path_obj = project_dir / filename
        file_path_obj.parent.mkdir(parents=True, exist_ok=True)
        file_path_obj.write_text(content, encoding='utf-8')
        self.detailed_log_event.emit("WorkflowEngine", "file_op", f"File written: {file_path_obj}", "2")
        self.file_generated.emit(str(file_path_obj))

    async def _plan_file(self, filename: str, file_info: dict, high_level_plan: dict, is_escalation: bool,
                         last_failed_review: dict) -> tuple:
        """Handles both initial planning and re-planning."""
        file_purpose = file_info.get("purpose", "")
        project_description = high_level_plan.get("project_description", "")

        other_files_context_list = [
            f"- {f['filename']}: {f['purpose']}" for f in high_level_plan.get('files', []) if f['filename'] != filename
        ]
        other_files_context = "\n".join(
            other_files_context_list) if other_files_context_list else "This is a single-file project."

        if is_escalation:
            failure_summary = self.assembler_service._format_feedback_for_prompt(last_failed_review)
            replan_context = (f"\n\nRE-PLANNING CONTEXT for {filename}:\n"
                              f"A previous plan for this file resulted in code that could not be fixed after multiple attempts. "
                              f"The final unresolvable issues were:\n{failure_summary}\n\n"
                              f"Please create a new, more robust, and clearer plan to avoid these specific issues.")
            project_description += replan_context

        file_spec = await self.planner_service.create_detailed_file_plan(
            filename, file_purpose, project_description, other_files_context
        )
        if not file_spec or 'components' not in file_spec:
            return None, None

        file_spec['purpose'] = file_purpose
        file_spec['path'] = filename
        return file_spec, other_files_context

    async def _review_and_patch_loop(self, filename: str, task_results: List[dict], plan: dict, file_spec: dict) -> \
    tuple[str, bool, dict]:
        """Manages the assembly, review, and patching cycle."""
        patch_attempt = 0
        current_code = await self.assembler_service.assemble_code(filename, task_results, plan, file_spec)
        last_review_data = {}

        while patch_attempt < self.MAX_PATCH_ATTEMPTS:
            review_data, review_passed = await self.assembler_service.review_code(
                filename, current_code, plan, file_spec
            )
            last_review_data = review_data  # Keep track of the latest review

            if review_passed:
                return current_code, True, last_review_data

            patch_attempt += 1
            self.detailed_log_event.emit("Reviewer", "warning",
                                         f"⚠️ Review for '{filename}' needs patching (Attempt {patch_attempt}/{self.MAX_PATCH_ATTEMPTS}).",
                                         2)

            if patch_attempt >= self.MAX_PATCH_ATTEMPTS:
                self.detailed_log_event.emit("Reviewer", "error", f"❌ Maximum patch attempts reached for {filename}.",
                                             2)
                return current_code, False, last_review_data

            current_code = await self.assembler_service.patch_code(
                filename, current_code, review_data, plan, file_spec
            )

        return current_code, False, last_review_data

    def _update_task_progress(self, completed: int, total: int):
        self.workflow_stats["workflow_state"]["completed_tasks"] = completed
        self.workflow_stats["workflow_state"]["total_tasks"] = total
        self.task_progress.emit(completed, total)

    async def _finalize_project(self, results: Dict[str, Any], elapsed_time: float) -> Dict[str, Any]:
        project_dir = results.get("project_dir")
        if project_dir:
            self.project_loaded.emit(project_dir)
            self.detailed_log_event.emit("WorkflowEngine", "file_op", f"Project loaded into UI: {project_dir}", "1")

        final_result = {
            "success": True,
            "project_name": Path(project_dir).name if project_dir else "Unknown",
            "project_dir": project_dir,
            "file_count": len(results.get("files_created", [])),
            "files_created": results.get("files_created", []),
            "failed_files": results.get("failed_files", []),  # Include failed files in summary
            "elapsed_time": elapsed_time
        }
        self.detailed_log_event.emit("WorkflowEngine", "success", "Project finalization complete.", "1")
        return final_result

    def get_workflow_stats(self) -> Dict[str, Any]:
        return self.workflow_stats.copy()

    def execute_workflow(self, prompt: str):
        asyncio.create_task(self.execute_enhanced_workflow(prompt, []))