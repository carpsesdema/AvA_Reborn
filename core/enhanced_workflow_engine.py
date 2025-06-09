# core/enhanced_workflow_engine.py - V5.4 with Robust GDD Creation

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

from core.workflow_services import ArchitectService, CoderService, ReviewerService
from core.project_state_manager import ProjectStateManager


class EnhancedWorkflowEngine(QObject):
    """
    🚀 V5.4 Workflow Engine: Now with robust GDD management for both new and loaded projects.
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

        self.project_state_manager: ProjectStateManager = None
        self.current_tech_spec: dict = None
        self.is_existing_project_loaded = False
        self.original_project_path: Optional[Path] = None
        self.original_user_prompt: str = ""

        def service_log_emitter(agent_name: str, type_key: str, content: str, indent_level: int):
            self.detailed_log_event.emit(agent_name, type_key, content, str(indent_level))

        self.architect_service = ArchitectService(self.llm_client, service_log_emitter, self.rag_manager)
        self.coder_service = CoderService(self.llm_client, service_log_emitter, self.rag_manager)
        self.reviewer_service = ReviewerService(self.llm_client, service_log_emitter, self.rag_manager)

        self._connect_terminal_signals()
        self.logger.info("✅ V5.4 'Robust GDD' Workflow Engine initialized.")

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

    def _create_initial_gdd(self, project_path: Path, project_name: str, initial_prompt: str):
        """Creates the initial Game Design Document for a new project."""
        gdd_file_path = project_path / f"{project_name}_GDD.md"
        if gdd_file_path.exists():
            self.logger.warning(f"GDD file already exists at {gdd_file_path}. Skipping creation.")
            return

        self.detailed_log_event.emit("WorkflowEngine", "file_op", f"Creating initial GDD: {gdd_file_path.name}", "1")
        gdd_template = f"""
# Game Design Document: {project_name}

## Project Vision
> {initial_prompt}

## Implemented Systems
_(This section will be populated as you build out the project.)_

---

## Development Log
- **{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**: Project initialized.
"""
        try:
            gdd_file_path.write_text(gdd_template.strip(), encoding='utf-8')
            self.file_generated.emit(str(gdd_file_path))
        except Exception as e:
            self.logger.error(f"❌ Failed to create initial GDD file: {e}")
            self.detailed_log_event.emit("WorkflowEngine", "error", f"Failed to create GDD: {e}", "1")

    def _update_gdd_log(self, project_path: Path, project_name: str, user_prompt: str, results: dict):
        """Appends a new entry to the GDD's development log."""
        gdd_file_path = project_path / f"{project_name}_GDD.md"
        if not gdd_file_path.exists():
            self.logger.error(f"Could not find GDD file to update at {gdd_file_path}")
            self.detailed_log_event.emit("WorkflowEngine", "error", f"GDD file not found for update.", "1")
            # If GDD is missing, create it with the log entry.
            vision = f"Project '{project_name}' was loaded. The original vision was not recorded."
            self._create_initial_gdd(project_path, project_name, vision)

        self.detailed_log_event.emit("WorkflowEngine", "file_op", f"Updating GDD log: {gdd_file_path.name}", "1")
        files_created = ", ".join(results.get("files_created", []))
        log_entry = f"""
- **{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**:
  - **Request**: "{user_prompt}"
  - **Result**: Successfully implemented changes.
  - **Files Modified/Created**: {files_created if files_created else "None specified."}
"""
        try:
            with gdd_file_path.open("a", encoding="utf-8") as f:
                f.write(log_entry)
        except Exception as e:
            self.logger.error(f"❌ Failed to update GDD log: {e}")
            self.detailed_log_event.emit("WorkflowEngine", "error", f"Failed to update GDD: {e}", "1")

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

            # --- NEW: Check for and create GDD on load ---
            project_name = tech_spec.get("project_name", project_path.name)
            project_description = tech_spec.get("project_description", "An existing project loaded into AvA.")
            self._create_initial_gdd(project_path, project_name, project_description)
            # --- END NEW ---

            self.detailed_log_event.emit("WorkflowEngine", "success", "✅ Analysis complete! Technical spec created.",
                                         "0")

            self.analysis_completed.emit(project_path_str, self.current_tech_spec)
            self.project_loaded.emit(project_path_str)

        except Exception as e:
            self.logger.error(f"❌ Analysis Workflow failed: {e}", exc_info=True)
            self.detailed_log_event.emit("WorkflowEngine", "error", f"❌ Analysis Error: {str(e)}", "0")
            self.project_loaded.emit(project_path_str)

    async def execute_enhanced_workflow(self, user_prompt: str, conversation_context: List[Dict] = None):
        self.logger.info(f"🚀 Starting V5.4 workflow: {user_prompt[:100]}...")
        workflow_start_time = datetime.now()
        self.original_user_prompt = user_prompt
        gdd_context = ""

        if self.is_existing_project_loaded:
            self.workflow_started.emit("Project Modification", user_prompt[:60] + '...')
            self.detailed_log_event.emit("WorkflowEngine", "stage_start",
                                         f"🚀 Initializing MODIFICATION workflow for '{self.original_project_path.name}'...",
                                         "0")
            project_name = self.current_tech_spec.get("project_name", self.original_project_path.name)
            gdd_context = self._read_gdd_context(self.original_project_path, project_name)

        else:
            self.workflow_started.emit("New Project", user_prompt[:60] + '...')
            self.detailed_log_event.emit("WorkflowEngine", "stage_start", "🚀 Initializing NEW PROJECT workflow...", "0")

        try:
            full_user_prompt = f"{user_prompt}\n\n--- GDD CONTEXT ---\n{gdd_context}"
            self.workflow_progress.emit("planning", "Architecting project...")
            tech_spec = await self.architect_service.create_tech_spec(full_user_prompt, conversation_context)

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
                                    ignore=shutil.ignore_patterns('venv', '__pycache__', '.git', '*_GDD.md'))
                except Exception as copy_error:
                    raise Exception(f"Failed to create a copy of the existing project: {copy_error}")
                self.detailed_log_event.emit("WorkflowEngine", "success", "Project copy complete.", "1")
            else:
                project_dir_name = f"{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                project_dir = Path("./workspace") / project_dir_name
                project_dir.mkdir(parents=True, exist_ok=True)
                self.detailed_log_event.emit("WorkflowEngine", "file_op",
                                             f"New project directory created: {project_dir}", "1")
                self._create_initial_gdd(project_dir, project_name, user_prompt)

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
                file_spec = tech_spec["technical_specs"].get(filename) or tech_spec["technical_specs"].get(
                    f"{filename}.py")

                if not file_spec:
                    self.detailed_log_event.emit("WorkflowEngine", "error",
                                                 f"Could not find spec for '{filename}'. Skipping.", 1)
                    continue

                dependency_files = file_spec.get("dependencies", [])
                dependency_context = self._build_dependency_context(dependency_files, knowledge_packets)
                project_context = {"description": tech_spec.get("project_description", "")}
                full_filename = filename if filename.endswith('.py') else f"{filename}.py"
                generated_code = await self.coder_service.generate_file_from_spec(full_filename, file_spec,
                                                                                  project_context,
                                                                                  dependency_context)
                review_data, review_passed = await self.reviewer_service.review_code(full_filename, generated_code,
                                                                                     project_context['description'])
                self.detailed_log_event.emit("WorkflowEngine", "info",
                                             f"Review for {full_filename} {'passed' if review_passed else 'failed'}. Writing file to disk.",
                                             "2")
                self._write_file(project_dir, full_filename, generated_code)
                knowledge_packets[full_filename] = {"spec": file_spec, "source_code": generated_code}
                results["files_created"].append(full_filename)

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
            lookup_key = dep_file if dep_file.endswith('.py') else f"{dep_file}.py"

            if lookup_key in knowledge_packets:
                packet = knowledge_packets[lookup_key]
                context_str += f"\n\n--- CONTEXT FOR DEPENDENCY: {lookup_key} ---\n"
                context_str += f"SPECIFICATION:\n```json\n{json.dumps(packet['spec'], indent=2)}\n```\n"
                context_str += f"FULL SOURCE CODE:\n```python\n{packet['source_code']}\n```\n"
            else:
                self.logger.warning(f"Dependency '{lookup_key}' not found. Context incomplete.")
                context_str += f"\n\n--- NOTE: Context for '{lookup_key}' was not available. ---\n"
        return context_str

    def _write_file(self, project_dir: Path, filename: str, content: str):
        file_path_obj = project_dir / filename
        file_path_obj.parent.mkdir(parents=True, exist_ok=True)
        file_path_obj.write_text(content, encoding='utf-8')
        self.detailed_log_event.emit("WorkflowEngine", "file_op", f"File written: {file_path_obj}", "2")
        self.file_generated.emit(str(file_path_obj))

    async def _finalize_project(self, results: Dict[str, Any], elapsed_time: float) -> Dict[str, Any]:
        project_dir_str = results.get("project_dir")
        final_result = {
            "success": len(results.get("failed_files", [])) == 0,
            "project_name": "Unknown",
            "project_dir": project_dir_str,
            "num_files": len(results.get("files_created", [])),
            "files_created": results.get("files_created", []),
            "failed_files": results.get("failed_files", []),
            "elapsed_time": elapsed_time,
            "strategy": "V5.4 Robust GDD"
        }

        if project_dir_str:
            self.project_loaded.emit(project_dir_str)
            project_path = Path(project_dir_str)
            raw_project_name = project_path.name
            final_result["project_name"] = raw_project_name

            base_project_name = raw_project_name
            if "_MOD_" in raw_project_name:
                base_project_name = raw_project_name.split("_MOD_")[0]
            else:
                try:
                    base_project_name = re.sub(r'_\d{8}_\d{6}$', '', raw_project_name)
                except Exception:
                    pass

            if final_result["success"]:
                self._update_gdd_log(project_path, base_project_name, self.original_user_prompt, results)

        self.detailed_log_event.emit("WorkflowEngine", "success", "Project finalization complete.", "1")
        return final_result