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
    🚀 Enhanced Workflow Engine with a streamlined, single-pass review and fix process.
    """
    workflow_started = Signal(str)
    workflow_completed = Signal(dict)
    workflow_progress = Signal(str, str)  # stage, description (overall stage)
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
            "✅ Enhanced Workflow Engine initialized with a streamlined review process.")

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

        # Step 1: Detailed planning for the file
        file_spec, other_files_context = await self._plan_file(filename, file_info, high_level_plan)
        if not file_spec:
            self.detailed_log_event.emit("WorkflowEngine", "error", f"Planning failed for {filename}. Aborting file.", "1")
            return filename, False

        # Step 2: Generate code based on the detailed plan
        project_context = {"description": high_level_plan.get("project_description", "")}
        task_results = await self.coder_service.execute_tasks_for_file(file_spec, project_context)

        # Step 3: Assemble the code snippets into a full file
        assembled_code = await self.assembler_service.assemble_code(filename, task_results, high_level_plan, file_spec)

        # Step 4: Perform a single, high-quality review
        review_data, review_passed = await self.assembler_service.review_code(filename, assembled_code, high_level_plan, file_spec)

        if review_passed:
            self.detailed_log_event.emit("Reviewer", "success", f"✅ Review for '{filename}' passed on first attempt!", "2")
            self._write_file(project_dir, filename, assembled_code)
            return filename, True
        else:
            self.detailed_log_event.emit("Reviewer", "warning", f"⚠️ Review for '{filename}' requires a one-shot fix.", "2")

            # Step 5: If review fails, perform a single, definitive fix (patch)
            final_code = await self.assembler_service.patch_code(filename, assembled_code, review_data, high_level_plan, file_spec)

            if final_code:
                # Step 6 (Optional but Recommended): A quick final review of the patched code
                final_review_data, final_review_passed = await self.assembler_service.review_code(
                    filename, final_code, high_level_plan, file_spec
                )
                if final_review_passed:
                    self.detailed_log_event.emit("Reviewer", "success", f"✅ Final version of '{filename}' passed review.", "2")
                    self._write_file(project_dir, filename, final_code)
                    return filename, True
                else:
                    self.detailed_log_event.emit("Reviewer", "error", f"⛔️ Patched code for '{filename}' still failed review. Writing last attempt.", "2")
                    # Even if it fails the second review, we write the "fixed" version as the best attempt.
                    self._write_file(project_dir, filename, final_code)
                    return filename, False # Mark as failed if the fix didn't pass review
            else:
                self.detailed_log_event.emit("Patcher", "error", f"⛔️ Patching failed for '{filename}'. Could not generate a fix.", "2")
                # Write the original assembled but failed code for debugging
                self._write_file(project_dir, filename, assembled_code)
                return filename, False

    def _write_file(self, project_dir: Path, filename: str, content: str):
        """Helper to write file content and emit signals."""
        file_path_obj = project_dir / filename
        file_path_obj.parent.mkdir(parents=True, exist_ok=True)
        file_path_obj.write_text(content, encoding='utf-8')
        self.detailed_log_event.emit("WorkflowEngine", "file_op", f"File written: {file_path_obj}", "2")
        self.file_generated.emit(str(file_path_obj))

    async def _plan_file(self, filename: str, file_info: dict, high_level_plan: dict) -> tuple:
        """Handles file planning."""
        file_purpose = file_info.get("purpose", "")
        project_description = high_level_plan.get("project_description", "")

        other_files_context_list = [
            f"- {f['filename']}: {f['purpose']}" for f in high_level_plan.get('files', []) if f['filename'] != filename
        ]
        other_files_context = "\n".join(other_files_context_list) if other_files_context_list else "This is a single-file project."

        file_spec = await self.planner_service.create_detailed_file_plan(
            filename, file_purpose, project_description, other_files_context
        )
        if not file_spec or 'components' not in file_spec:
            return None, None

        file_spec['purpose'] = file_purpose
        file_spec['path'] = filename
        return file_spec, other_files_context

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