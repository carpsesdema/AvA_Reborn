# enhanced_workflow_engine.py - COMPLETE WORKING FILE with Role-Based LLM

import asyncio
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from PySide6.QtCore import QObject, Signal

# Import services from their new locations
from core.workflow_services import EnhancedPlannerService, EnhancedCoderService, EnhancedAssemblerService, \
    StructureService


class EnhancedWorkflowEngine(QObject):
    """
    🚀 Enhanced Workflow Engine with Role-Based AI Code Generation
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

        # NEW: Instantiate the StructureService
        self.structure_service = StructureService(self.llm_client, service_log_emitter, self.rag_manager)
        self.planner_service = EnhancedPlannerService(self.llm_client, service_log_emitter, self.rag_manager)
        self.coder_service = EnhancedCoderService(self.llm_client, service_log_emitter, self.rag_manager)
        self.assembler_service = EnhancedAssemblerService(self.llm_client, service_log_emitter, self.rag_manager)

        self._connect_terminal_signals()
        self.logger.info(
            "✅ Enhanced Workflow Engine initialized with Structurer -> Planner -> Coder -> Assembler flow.")

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
        self.logger.info(f"🚀 Starting Conversational AI workflow: {user_prompt[:100]}...")
        workflow_start_time = datetime.now()
        self.workflow_started.emit(user_prompt)
        self.detailed_log_event.emit("WorkflowEngine", "stage_start", "🚀 Initializing Conversational AI workflow", "0")

        try:
            # --- Stage 1: High-Level Structuring ---
            self.workflow_progress.emit("planning", "Phase 1: Determining file structure...")
            self._update_task_progress(1, 5)

            high_level_plan = await self.structure_service.create_project_structure(user_prompt, conversation_context)
            if not high_level_plan or 'files' not in high_level_plan:
                raise Exception("Structuring failed. Could not determine a valid file structure.")

            project_name = high_level_plan.get("project_name", "ai_project")
            project_description = high_level_plan.get("project_description", "An AI-generated project.")
            files_to_process = high_level_plan.get('files', [])
            total_files = len(files_to_process)

            project_dir = Path("./workspace") / f"{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            project_dir.mkdir(parents=True, exist_ok=True)
            self.detailed_log_event.emit("WorkflowEngine", "file_op", f"Project directory created: {project_dir}", "1")

            # --- Stage 2: Detailed Planning, Coding, and Assembly (Iterative per File) ---
            self.workflow_progress.emit("generation", f"Phase 2: Processing {total_files} files...")
            self._update_task_progress(2, 5)

            completed_file_plans = {}
            results = {"files_created": [], "project_dir": str(project_dir)}
            project_context = {"description": project_description, "feedback_for_coder": None}

            for i, file_info in enumerate(files_to_process):
                filename = file_info["filename"]
                file_purpose = file_info["purpose"]
                self.detailed_log_event.emit("WorkflowEngine", "info",
                                             f"Processing file {i + 1}/{total_files}: {filename}", "1")

                other_files_context = json.dumps(
                    {k: {"purpose": v["purpose"]} for k, v in completed_file_plans.items()}, indent=2)

                file_spec = await self.planner_service.create_detailed_file_plan(filename, file_purpose,
                                                                                 project_description,
                                                                                 other_files_context)

                if not file_spec or 'components' not in file_spec:
                    self.detailed_log_event.emit("WorkflowEngine", "error",
                                                 f"Skipping file {filename} due to planning failure.", 1)
                    continue

                file_spec['purpose'] = file_purpose
                file_spec['path'] = filename
                completed_file_plans[filename] = file_spec

                max_attempts = 3
                for attempt in range(max_attempts):
                    self.detailed_log_event.emit("WorkflowEngine", "status",
                                                 f"Starting generation for '{filename}' (Attempt {attempt + 1}/{max_attempts})...",
                                                 1)

                    task_results = await self.coder_service.execute_tasks_for_file(file_spec, project_context)
                    assembled_code, review_ok, review_msg = await self.assembler_service.assemble_file(filename,
                                                                                                       task_results,
                                                                                                       high_level_plan,
                                                                                                       file_spec)
                    if review_ok:
                        self.detailed_log_event.emit("WorkflowEngine", "success", f"Code for '{filename}' approved.", 1)
                        break  # Exit the loop on success
                    else:
                        project_context[
                            "feedback_for_coder"] = f"Review failed. You MUST fix the following issues: {review_msg}"
                        self.detailed_log_event.emit("WorkflowEngine", "warning",
                                                     f"Review failed for '{filename}'. Retrying with feedback...", 1)

                else:  # This 'else' belongs to the 'for' loop, it runs if the loop completes without a 'break'
                    self.detailed_log_event.emit("WorkflowEngine", "error",
                                                 f"Halting workflow. Code for '{filename}' failed all review attempts.",
                                                 1)
                    raise Exception(
                        f"Failed to generate acceptable code for '{filename}' after {max_attempts} attempts.")

                file_path_obj = project_dir / filename
                file_path_obj.parent.mkdir(parents=True, exist_ok=True)
                file_path_obj.write_text(assembled_code, encoding='utf-8')
                self.detailed_log_event.emit("WorkflowEngine", "file_op", f"File written: {file_path_obj}", "2")
                self.file_generated.emit(str(file_path_obj))
                results["files_created"].append(filename)
                project_context["feedback_for_coder"] = None  # Reset feedback for next file

            self._update_task_progress(4, 5)

            # --- Stage 3: Finalization ---
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

    def _update_task_progress(self, completed: int, total: int):
        self.workflow_stats["workflow_state"]["completed_tasks"] = completed
        self.workflow_stats["workflow_state"]["total_tasks"] = total
        self.task_progress.emit(completed, total)

    async def _finalize_project(self, results: Dict[str, Any], elapsed_time: float) -> Dict[str, Any]:
        self.detailed_log_event.emit("WorkflowEngine", "thought", "Creating project summary...", "1")
        await asyncio.sleep(0.1)

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
            "elapsed_time": elapsed_time
        }
        self.detailed_log_event.emit("WorkflowEngine", "success", "Project finalization complete.", "1")
        return final_result

    def get_workflow_stats(self) -> Dict[str, Any]:
        return self.workflow_stats.copy()

    def execute_workflow(self, prompt: str):
        asyncio.create_task(self.execute_enhanced_workflow(prompt, []))