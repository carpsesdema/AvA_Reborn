# enhanced_workflow_engine.py - V6 "Surgical Modification" Engine

import asyncio
import json
import logging
import traceback
import shutil
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from PySide6.QtCore import QObject, Signal

from core.workflow_services import ArchitectService, CoderService, ReviewerService
from core.project_state_manager import ProjectStateManager


class EnhancedWorkflowEngine(QObject):
    workflow_started = Signal(str)
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
        self.terminal_window = terminal_window
        self.code_viewer = code_viewer
        self.rag_manager = rag_manager
        self.logger = logging.getLogger(__name__)

        self.project_state_manager: ProjectStateManager = None
        self.current_tech_spec: dict = None

        def service_log_emitter(agent_name: str, type_key: str, content: str, indent_level: int):
            self.detailed_log_event.emit(agent_name, type_key, content, str(indent_level))

        self.architect_service = ArchitectService(self.llm_client, service_log_emitter, self.rag_manager)
        self.coder_service = CoderService(self.llm_client, service_log_emitter, self.rag_manager)
        self.reviewer_service = ReviewerService(self.llm_client, service_log_emitter, self.rag_manager)

        self._connect_terminal_signals()
        self.logger.info("✅ V6 'Surgical Modification' Workflow Engine initialized.")

    def _connect_terminal_signals(self):
        if not self.terminal_window: return
        try:
            self.workflow_progress.connect(self.terminal_window.update_workflow_progress)
            self.task_progress.connect(self.terminal_window.update_task_progress)
            self.detailed_log_event.connect(self.terminal_window.stream_log_rich)
        except Exception as e:
            self.logger.error(f"❌ Failed to connect terminal signals: {e}")

    async def execute_analysis_workflow(self, project_path_str: str):
        self.logger.info(f"🚀 Starting Analysis workflow for: {project_path_str}...")
        self.analysis_started.emit(project_path_str)
        self.detailed_log_event.emit("WorkflowEngine", "stage_start",
                                     f"🚀 Initializing Analysis for '{Path(project_path_str).name}'...", "0")
        try:
            self.project_state_manager = ProjectStateManager(Path(project_path_str))
            self.detailed_log_event.emit("WorkflowEngine", "info", "Project State Manager initialized. Scanned files.",
                                         "1")
            tech_spec = await self.architect_service.analyze_and_create_spec_from_project(self.project_state_manager)
            if not tech_spec:
                raise Exception(
                    "Analysis failed. Architect could not produce a technical specification from the project files.")
            self.current_tech_spec = tech_spec
            self.detailed_log_event.emit("WorkflowEngine", "success", "✅ Analysis complete! Technical spec created.",
                                         "0")
            self.analysis_completed.emit(project_path_str, self.current_tech_spec)
            self.project_loaded.emit(project_path_str)
        except Exception as e:
            self.logger.error(f"❌ Analysis Workflow failed: {e}", exc_info=True)
            self.detailed_log_event.emit("WorkflowEngine", "error", f"❌ Analysis Error: {str(e)}", "0")
            self.project_loaded.emit(project_path_str)

    def _get_files_to_modify(self, user_prompt: str, all_files: List[str]) -> List[str]:
        """Identifies which files should be modified based on the user's prompt."""
        files_to_modify = []
        # A simple but effective heuristic: if a filename is mentioned, it's a target.
        for filename in all_files:
            if re.search(r'\b' + re.escape(filename) + r'\b', user_prompt, re.IGNORECASE):
                files_to_modify.append(filename)

        # If no files are explicitly mentioned, we might need a more advanced AI step.
        # For now, if none are mentioned, we'll assume we modify everything (the old behavior).
        if not files_to_modify:
            self.logger.warning("No specific files mentioned in modification prompt. Will attempt to regenerate all.")
            return all_files

        self.logger.info(f"Identified files to modify from prompt: {files_to_modify}")
        return files_to_modify

    async def execute_enhanced_workflow(self, user_prompt: str, conversation_context: List[Dict] = None):
        self.logger.info(f"🚀 Starting V6 workflow: {user_prompt[:100]}...")
        workflow_start_time = datetime.now()
        self.workflow_started.emit(user_prompt)
        try:
            is_modification = self.project_state_manager is not None and self.current_tech_spec is not None

            # --- PHASE 1: GET TECH SPEC ---
            tech_spec = self.current_tech_spec if is_modification else await self.architect_service.create_tech_spec(
                user_prompt, conversation_context)
            if not tech_spec or 'technical_specs' not in tech_spec:
                raise Exception("Architecture phase failed. No valid technical specification available.")

            # --- PHASE 2: SETUP PROJECT DIRECTORY ---
            project_name = tech_spec.get("project_name", "ai-project").replace(" ", "-")
            new_project_dir_name = f"{project_name}-mod-{datetime.now().strftime('%Y%m%d_%H%M%S')}" if is_modification else f"{project_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            project_dir = Path("./workspace") / new_project_dir_name
            project_dir.mkdir(parents=True, exist_ok=True)

            # --- NEW: WRITE REQUIREMENTS.TXT ---
            requirements = tech_spec.get("requirements", [])
            if requirements:
                req_content = "\n".join(requirements)
                self._write_file(project_dir, "requirements.txt", req_content)
                self.detailed_log_event.emit("WorkflowEngine", "file_op", "Generated requirements.txt", "1")

            # --- PHASE 3: FILE HANDLING (MODIFICATION-AWARE) ---
            build_order = tech_spec.get("dependency_order", [])
            files_in_spec = list(tech_spec.get("technical_specs", {}).keys())
            files_to_actively_modify = self._get_files_to_modify(user_prompt,
                                                                 files_in_spec) if is_modification else files_in_spec

            knowledge_packets = {}
            results = {"files_created": [], "project_dir": str(project_dir), "failed_files": []}

            self.detailed_log_event.emit("WorkflowEngine", "info",
                                         f"Files targeted for modification: {files_to_actively_modify}", "1")

            for filename in build_order:
                file_spec = tech_spec["technical_specs"].get(filename)
                if not file_spec: continue

                # Determine if we should generate this file or copy it
                if filename in files_to_actively_modify:
                    # GENERATE the modified file
                    self.detailed_log_event.emit("WorkflowEngine", "stage_start",
                                                 f"🧬 Generating modified file: {filename}", "1")
                    dependency_context = self._build_dependency_context(file_spec.get("dependencies", []),
                                                                        knowledge_packets)
                    project_context = {"description": tech_spec.get("project_description", "")}
                    generated_code = await self.coder_service.generate_file_from_spec(filename, file_spec,
                                                                                      project_context,
                                                                                      dependency_context)

                    if "# FALLBACK" in generated_code:
                        results["failed_files"].append(filename);
                        continue

                    self._write_file(project_dir, filename, generated_code)
                    knowledge_packets[filename] = {"spec": file_spec, "source_code": generated_code}
                elif is_modification:
                    # COPY the original, unchanged file
                    original_file_path = self.project_state_manager.project_root / filename
                    if original_file_path.exists():
                        destination_path = project_dir / filename
                        shutil.copy2(original_file_path, destination_path)
                        self.detailed_log_event.emit("WorkflowEngine", "file_op", f"Copied unchanged file: {filename}",
                                                     "1")
                        knowledge_packets[filename] = {"spec": file_spec, "source_code": original_file_path.read_text()}
                    else:
                        self.logger.warning(f"Wanted to copy {filename} but it doesn't exist in the source project.")
                # (The case for a new project is handled by the initial `if` block)

                results["files_created"].append(filename)

            # --- PHASE 4: FINALIZATION ---
            final_result = await self._finalize_project(results, (datetime.now() - workflow_start_time).total_seconds())
            self.workflow_completed.emit(final_result)
            return final_result
        except Exception as e:
            self.logger.error(f"❌ AI Workflow failed: {e}", exc_info=True)
            self.workflow_completed.emit({"success": False, "error": str(e),
                                          "elapsed_time": (datetime.now() - workflow_start_time).total_seconds()})
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
                self.logger.warning(
                    f"Dependency '{dep_file}' was not found in knowledge packets. Context will be incomplete.")
        return context_str

    def _write_file(self, project_dir: Path, filename: str, content: str):
        file_path_obj = project_dir / filename
        file_path_obj.parent.mkdir(parents=True, exist_ok=True)
        file_path_obj.write_text(content, encoding='utf-8')
        self.detailed_log_event.emit("WorkflowEngine", "file_op", f"File written/overwritten: {file_path_obj}", "2")
        self.file_generated.emit(str(file_path_obj))

    async def _finalize_project(self, results: Dict[str, Any], elapsed_time: float) -> Dict[str, Any]:
        project_dir = results.get("project_dir")
        if project_dir:
            self.project_loaded.emit(project_dir)
            self.detailed_log_event.emit("WorkflowEngine", "file_op", f"Project loaded into UI: {project_dir}", "1")
        return {"success": len(results.get("failed_files", [])) == 0,
                "project_name": Path(project_dir).name if project_dir else "Unknown", "project_dir": project_dir,
                "num_files": len(results.get("files_created", [])), "files_created": results.get("files_created", []),
                "failed_files": results.get("failed_files", []), "elapsed_time": elapsed_time,
                "strategy": "V6 Surgical Modification"}