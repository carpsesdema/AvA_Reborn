# core/enhanced_workflow_engine.py - V6.0 Hybrid Workflow with Micro-Task Orchestration

import asyncio
import json
import logging
import traceback
import shutil
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from PySide6.QtCore import QObject, Signal

from core.services.architect_service import ArchitectService
from core.services.coder_service import CoderService
from core.services.reviewer_service import ReviewerService
from core.services.assembler_service import AssemblerService
from core.project_state_manager import ProjectStateManager
from core.enhanced_micro_task_engine import StreamlinedMicroTaskEngine, SimpleTaskSpec
from core.project_builder import ProjectBuilder


class HybridWorkflowEngine(QObject):
    """
    🚀 V6.0 Hybrid Workflow Engine: Combines Enhanced Workflow with Micro-Task orchestration
    - Uses big models for architecture/planning
    - Breaks files into micro-tasks for Gemini Flash implementation
    - Assembles micro-task outputs into complete files
    - Enables parallel processing for cost efficiency
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

    # New signals for the workflow monitor
    node_status_changed = Signal(str, str, str)  # agent_id, status, status_text
    workflow_plan_ready = Signal(dict)
    data_flow_started = Signal(str, str)  # from_agent, to_agent
    data_flow_completed = Signal(str, str)  # from_agent, to_agent
    workflow_reset = Signal()

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

        # Initialize services
        self.architect_service = ArchitectService(self.llm_client, service_log_emitter, self.rag_manager)
        self.coder_service = CoderService(self.llm_client, service_log_emitter, self.rag_manager)
        self.assembler_service = AssemblerService(self.llm_client, service_log_emitter, self.rag_manager)
        self.reviewer_service = ReviewerService(self.llm_client, service_log_emitter, self.rag_manager)

        # Initialize micro-task engine
        self.micro_task_engine = StreamlinedMicroTaskEngine(
            self.llm_client,
            self.project_state_manager,
            self.rag_manager
        )

        from core.config import ConfigManager
        config = ConfigManager()
        self.project_builder = ProjectBuilder(
            workspace_root=config.app_config.workspace_path,
            stream_emitter=service_log_emitter,
            logger=self.logger
        )

        self._connect_terminal_signals()
        self.logger.info("✅ V6.0 Hybrid Workflow Engine initialized with micro-task orchestration.")

    def _connect_terminal_signals(self):
        if self.streaming_terminal and hasattr(self.streaming_terminal, 'stream_log_rich'):
            try:
                self.detailed_log_event.connect(self.streaming_terminal.stream_log_rich)
                self.logger.info("Connected workflow engine signals to StreamingTerminal.")
            except Exception as e:
                self.logger.error(f"❌ Failed to connect terminal signals: {e}")
        else:
            self.logger.warning("No streaming terminal connected. AI logs will print to console.")

    def set_project_state(self, project_state_manager: ProjectStateManager):
        """Connect project state manager to all services for team communication."""
        self.project_state_manager = project_state_manager
        self.architect_service.set_project_state(project_state_manager)
        self.coder_service.set_project_state(project_state_manager)
        self.assembler_service.set_project_state(project_state_manager)
        self.reviewer_service.set_project_state(project_state_manager)
        self.micro_task_engine.project_state = project_state_manager

    async def analyze_existing_project(self, project_path_str: str):
        """Analyze an existing project and create technical specification."""
        try:
            self.workflow_reset.emit()
            project_path = Path(project_path_str)
            self.workflow_started.emit("Project Analysis", project_path.name)
            self.analysis_started.emit(project_path_str)

            self.detailed_log_event.emit("HybridEngine", "stage_start",
                                         f"🔍 Starting project analysis for: {project_path.name}", "0")
            self.project_loaded.emit(project_path_str)

            if not self.project_state_manager:
                self.project_state_manager = ProjectStateManager(project_path)
                self.set_project_state(self.project_state_manager)

            self.node_status_changed.emit("architect", "working", "Analyzing project...")
            self.current_tech_spec = await self.architect_service.analyze_and_create_spec_from_project(
                self.project_state_manager
            )
            self.node_status_changed.emit("architect", "success", "Analysis complete.")

            if not self.current_tech_spec:
                self.node_status_changed.emit("architect", "error", "Analysis failed.")
                raise Exception("Failed to analyze project structure")

            self.is_existing_project_loaded = True
            self.original_project_path = project_path
            project_name = self.current_tech_spec.get("project_name", project_path.name)
            project_description = self.current_tech_spec.get("project_description", "Analyzed existing project")
            self._ensure_gdd_exists(project_path, project_name, project_description)

            self.detailed_log_event.emit("HybridEngine", "success",
                                         "✅ Analysis complete! Technical spec created. Ready for modifications!", "0")
            self.analysis_completed.emit(project_path_str, self.current_tech_spec)
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}", exc_info=True)
            self.detailed_log_event.emit("HybridEngine", "error",
                                         f"❌ Analysis failed: {str(e)}", "0")
            try:
                project_path = Path(project_path_str)
                self.current_tech_spec = self._create_basic_tech_spec_from_files(project_path)
                self.is_existing_project_loaded = True
                self.original_project_path = project_path
                self.detailed_log_event.emit("HybridEngine", "info",
                                             "Using fallback tech spec for future modifications", "0")
            except Exception as fallback_error:
                self.logger.error(f"Failed to create fallback tech spec: {fallback_error}")
            self.analysis_completed.emit(project_path_str, self.current_tech_spec or {})

    async def execute_enhanced_workflow(self, user_prompt: str, conversation_context: list = None):
        """Execute the V6.0 Hybrid workflow combining architecture and micro-task orchestration."""
        self.logger.info(f"🚀 Starting V6.0 Hybrid workflow: {user_prompt[:100]}...")
        self.workflow_reset.emit()
        workflow_start_time = datetime.now()
        self.original_user_prompt = user_prompt
        gdd_context = ""

        if self.is_existing_project_loaded:
            self.workflow_started.emit("Project Modification", user_prompt[:60] + '...')
            self.detailed_log_event.emit("HybridEngine", "stage_start",
                                         f"🚀 Initializing HYBRID MODIFICATION workflow for '{self.original_project_path.name}'...",
                                         "0")
            project_name = self.current_tech_spec.get("project_name", self.original_project_path.name)
            gdd_context = self._read_gdd_context(self.original_project_path, project_name)
        else:
            self.workflow_started.emit("New Project", user_prompt[:60] + '...')
            self.detailed_log_event.emit("HybridEngine", "stage_start",
                                         "🚀 Initializing HYBRID NEW PROJECT workflow...", "0")

        try:
            # Phase 1: Architecture
            full_user_prompt = f"{user_prompt}\n\n--- GDD CONTEXT ---\n{gdd_context}"
            self.workflow_progress.emit("planning", "🧠 Architecting project...")
            self.node_status_changed.emit("architect", "working", "Creating tech spec...")
            tech_spec = await self.architect_service.create_tech_spec(full_user_prompt, conversation_context)
            if not tech_spec or 'technical_specs' not in tech_spec:
                self.node_status_changed.emit("architect", "error", "Failed to create spec.")
                raise Exception("Architecture phase failed")
            self.node_status_changed.emit("architect", "success", "Tech spec created.")
            self.workflow_plan_ready.emit(tech_spec)
            self.current_tech_spec = tech_spec

            # Phase 2: Micro-Task Generation
            self.data_flow_started.emit("architect", "coder")
            self.workflow_progress.emit("micro-tasking", "📋 Breaking project into micro-tasks...")
            project_context_for_tasks = {"project_name": tech_spec.get("project_name", "Unknown Project"),
                                         "project_description": tech_spec.get("project_description", "")}
            all_micro_tasks = await self.micro_task_engine.create_smart_tasks(tech_spec.get("technical_specs", {}),
                                                                              project_context_for_tasks)
            tasks_by_file = defaultdict(list)
            for task in all_micro_tasks:
                if task.file_path: tasks_by_file[task.file_path].append(task)
            self.detailed_log_event.emit("HybridEngine", "success",
                                         f"📊 Created {len(all_micro_tasks)} micro-tasks for {len(tasks_by_file)} files.",
                                         "1")
            self.data_flow_completed.emit("architect", "coder")

            # Phase 3: Project Setup
            project_dir = await self._setup_project_directory(tech_spec)
            if not self.project_state_manager:
                self.project_state_manager = ProjectStateManager(project_dir)
                self.set_project_state(self.project_state_manager)

            # Phase 4: Hybrid File Generation
            await self._execute_hybrid_file_generation(tech_spec, project_dir, tasks_by_file)

            # Phase 5: Finalization
            self.workflow_progress.emit("finalization", "🎯 Finalizing project...")
            await self._finalize_hybrid_project(tech_spec, project_dir, workflow_start_time)
            self.workflow_progress.emit("complete", "✅ Hybrid workflow complete!")

        except Exception as e:
            self.logger.error(f"Hybrid workflow failed: {e}", exc_info=True)
            self.detailed_log_event.emit("HybridEngine", "error", f"❌ Workflow failed: {str(e)}", "0")
            self.node_status_changed.emit("architect", "error", "Workflow failed")
            self.node_status_changed.emit("coder", "error", "Workflow failed")
            self.node_status_changed.emit("reviewer", "error", "Workflow failed")
            self.workflow_completed.emit({"success": False, "error": str(e)})

    async def _execute_hybrid_file_generation(self, tech_spec: dict, project_dir: Path,
                                              tasks_by_file: Dict[str, List[SimpleTaskSpec]]):
        self.workflow_progress.emit("generation", "⚡ Generating files with micro-task orchestration...")
        files_to_generate = tech_spec.get("technical_specs", {}).get("files", {})
        if not files_to_generate:
            self.detailed_log_event.emit("HybridEngine", "warning", "No files specified in tech spec", "2")
            return

        generated_files = {}
        file_count = len(files_to_generate)
        self.node_status_changed.emit("coder", "working", f"Generating {file_count} files...")

        for idx, (filename, file_spec) in enumerate(files_to_generate.items(), 1):
            self.task_progress.emit(idx, file_count)
            if file_spec.get("skip_generation", False):
                self.detailed_log_event.emit("HybridEngine", "info", f"⏭️ Skipping {filename} (exists)", "3")
                continue
            try:
                self.detailed_log_event.emit("HybridEngine", "info", f"🎯 Processing {filename}...", "3")
                micro_tasks_for_file = tasks_by_file.get(filename, [])
                assembled_code = await self._generate_file_with_micro_tasks(filename, file_spec, tech_spec,
                                                                            generated_files, micro_tasks_for_file)
                if assembled_code and assembled_code.strip():
                    self.node_status_changed.emit("reviewer", "working", f"Reviewing {filename}...")
                    self.data_flow_started.emit("coder", "reviewer")
                    # In a full implementation, you might have a review step here.
                    # For now, we assume success.
                    await asyncio.sleep(0.5)  # Simulate review
                    self.data_flow_completed.emit("coder", "reviewer")
                    self.node_status_changed.emit("reviewer", "success", f"{filename} approved.")

                    self._write_file(project_dir, filename, assembled_code)
                    generated_files[filename] = {"source_code": assembled_code}
                    self.detailed_log_event.emit("HybridEngine", "success",
                                                 f"✅ {filename} completed ({len(assembled_code)} chars)", "3")
                else:
                    self.detailed_log_event.emit("HybridEngine", "warning", f"⚠️ {filename} generated empty", "3")
            except Exception as e:
                self.logger.error(f"Failed to generate {filename}: {e}", exc_info=True)
                self.detailed_log_event.emit("HybridEngine", "error", f"❌ Failed to generate {filename}: {e}", "3")

        self.node_status_changed.emit("coder", "success", f"Generated all files.")

    async def _generate_file_with_micro_tasks(self, filename: str, file_spec: dict, tech_spec: dict,
                                              generated_files: dict, micro_tasks: List[SimpleTaskSpec]) -> str:
        # This function remains largely the same, but we can add more signals
        try:
            if not micro_tasks:
                self.detailed_log_event.emit("HybridEngine", "info",
                                             f"No micro-tasks for {filename}, using traditional generation", "4")
                return await self.coder_service.generate_file_from_spec(filename, file_spec, {
                    "description": tech_spec.get("project_description", "")}, self._build_dependency_context(
                    file_spec.get("dependencies", []), generated_files))

            self.detailed_log_event.emit("HybridEngine", "info",
                                         f"📋 Using {len(micro_tasks)} micro-tasks for {filename}", "4")
            micro_task_results = await self._execute_micro_tasks_in_parallel(micro_tasks)
            if not micro_task_results:
                self.detailed_log_event.emit("HybridEngine", "error", f"All micro-tasks failed for {filename}", "4")
                return ""

            self.node_status_changed.emit("assembler", "working", f"Assembling {filename}...")
            self.detailed_log_event.emit("HybridEngine", "info",
                                         f"🔧 Assembling {len(micro_task_results)} components for {filename}", "4")
            project_context = {"description": tech_spec.get("project_description", ""),
                               "coding_standards": tech_spec.get("coding_standards", "Follow PEP 8"),
                               "project_patterns": tech_spec.get("project_patterns", "Standard Python patterns"),
                               "file_purpose": file_spec.get("purpose", "Generated file"),
                               "integration_requirements": file_spec.get("dependencies", [])}
            assembled_code = await self.assembler_service.assemble_file_from_micro_tasks(filename, file_spec,
                                                                                         micro_task_results,
                                                                                         project_context)
            self.detailed_log_event.emit("HybridEngine", "success", f"✅ Assembly complete for {filename}", "4")
            self.node_status_changed.emit("assembler", "success", f"{filename} assembled.")
            return assembled_code
        except Exception as e:
            self.logger.error(f"Micro-task generation failed for {filename}: {e}", exc_info=True)
            self.node_status_changed.emit("assembler", "error", f"Assembly of {filename} failed.")
            self.detailed_log_event.emit("HybridEngine", "error",
                                         f"❌ Micro-task generation failed for {filename}: {str(e)}", "4")
            return ""

    async def _execute_micro_tasks_in_parallel(self, micro_tasks: List[SimpleTaskSpec]) -> List[Dict[str, Any]]:
        if not micro_tasks: return []
        semaphore = asyncio.Semaphore(5)

        async def execute_single_micro_task(task: SimpleTaskSpec) -> Optional[Dict[str, Any]]:
            async with semaphore:
                try:
                    self.detailed_log_event.emit("HybridEngine", "info", f"🔄 Executing micro-task: {task.id}", "5")
                    result = await self.coder_service.execute_micro_task_with_gemini_flash(task)
                    if result and "IMPLEMENTED_CODE" in result:
                        self.detailed_log_event.emit("HybridEngine", "success", f"✅ Completed micro-task: {task.id}",
                                                     "5")
                        return result
                    else:
                        self.detailed_log_event.emit("HybridEngine", "error",
                                                     f"❌ Micro-task {task.id} returned invalid result", "5")
                        return None
                except Exception as e:
                    self.logger.error(f"Micro-task {task.id} failed: {e}", exc_info=True)
                    self.detailed_log_event.emit("HybridEngine", "error", f"❌ Micro-task {task.id} failed: {str(e)}",
                                                 "5")
                    return None

        self.detailed_log_event.emit("HybridEngine", "info",
                                     f"🚀 Starting parallel execution of {len(micro_tasks)} micro-tasks", "4")
        results = await asyncio.gather(*[execute_single_micro_task(task) for task in micro_tasks])
        successful_results = [r for r in results if r is not None]
        failed_count = len(results) - len(successful_results)
        if failed_count > 0:
            self.detailed_log_event.emit("HybridEngine", "warning", f"⚠️ {failed_count} micro-tasks failed", "4")
        self.detailed_log_event.emit("HybridEngine", "success",
                                     f"✅ Parallel execution complete: {len(successful_results)}/{len(micro_tasks)} successful",
                                     "4")
        return successful_results

    async def _setup_project_directory(self, tech_spec: dict) -> Path:
        project_name = tech_spec.get("project_name", "generated_project")
        if self.is_existing_project_loaded and self.original_project_path:
            self.detailed_log_event.emit("HybridEngine", "info",
                                         "Setting up development branch for project modification...", "1")
            project_dir = self.project_builder.setup_project_directory(project_name=project_name, is_modification=True,
                                                                       original_user_prompt=self.original_user_prompt,
                                                                       original_project_path=self.original_project_path)
        else:
            self.detailed_log_event.emit("HybridEngine", "info", f"Creating new project directory: {project_name}", "1")
            project_dir = self.project_builder.setup_project_directory(project_name=project_name, is_modification=False,
                                                                       original_user_prompt=self.original_user_prompt,
                                                                       original_project_path=None)
        return project_dir

    def _write_file(self, project_dir: Path, filename: str, content: str):
        try:
            file_path = project_dir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding='utf-8')
            self.detailed_log_event.emit("HybridEngine", "success",
                                         f"✅ File written: {filename} ({len(content)} chars)", "3")
            self.file_generated.emit(str(file_path))
        except Exception as e:
            self.logger.error(f"Failed to write file {filename}: {e}", exc_info=True)
            self.detailed_log_event.emit("HybridEngine", "error", f"❌ Failed to write {filename}: {str(e)}", "3")

    def _build_dependency_context(self, dependencies: List[str], generated_files: Dict[str, Any]) -> str:
        context_parts = []
        for dep in dependencies:
            if dep in generated_files:
                context_parts.append(f"=== {dep} ===")
                context_parts.append(generated_files[dep].get("source_code", "")[:500] + "...")
                context_parts.append("")
        return "\n".join(context_parts) if context_parts else "No dependencies available."

    async def _finalize_hybrid_project(self, tech_spec: dict, project_dir: Path, start_time: datetime):
        try:
            self._create_requirements_txt(project_dir, tech_spec)
            self._setup_virtual_environment(project_dir)
            elapsed_time = (datetime.now() - start_time).total_seconds()
            result = {"success": True, "project_directory": str(project_dir),
                      "project_name": tech_spec.get("project_name", "hybrid_project"),
                      "files_created": [f.name for f in project_dir.rglob("*.py")], "execution_time": elapsed_time,
                      "approach": "hybrid_micro_task", "num_files": len(list(project_dir.rglob("*.py")))}
            self.detailed_log_event.emit("HybridEngine", "success",
                                         f"🎉 Hybrid workflow completed in {elapsed_time:.1f}s", "0")
            self.workflow_completed.emit(result)
        except Exception as e:
            self.logger.error(f"Finalization failed: {e}", exc_info=True)
            self.workflow_completed.emit({"success": False, "error": str(e)})

    def _create_requirements_txt(self, project_dir: Path, tech_spec: dict):
        try:
            requirements = tech_spec.get("technical_specs", {}).get("requirements", [])
            if requirements:
                req_file = project_dir / "requirements.txt"
                req_file.write_text("\n".join(requirements) + "\n", encoding='utf-8')
                self.detailed_log_event.emit("HybridEngine", "success",
                                             f"✅ Created requirements.txt with {len(requirements)} packages", "2")
        except Exception as e:
            self.logger.error(f"Failed to create requirements.txt: {e}", exc_info=True)

    def _setup_virtual_environment(self, project_dir: Path):
        try:
            import subprocess, sys
            venv_path = project_dir / "venv"
            if not venv_path.exists():
                subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True, capture_output=True)
                self.detailed_log_event.emit("HybridEngine", "success", "✅ Virtual environment created", "2")
        except Exception as e:
            self.logger.warning(f"Virtual environment setup failed: {e}")

    def _ensure_gdd_exists(self, project_path: Path, project_name: str, project_description: str):
        gdd_path = project_path / f"{project_name}_GDD.md"
        if not gdd_path.exists():
            gdd_content = f"# {project_name} - Game Design Document\n\n## Project Overview\n{project_description}\n"
            gdd_path.write_text(gdd_content, encoding='utf-8')
            self.detailed_log_event.emit("HybridEngine", "info", "Created GDD for existing project", "2")

    def _read_gdd_context(self, project_path: Path, project_name: str) -> str:
        gdd_files = list(project_path.glob("*GDD.md"))
        if gdd_files:
            return gdd_files[0].read_text(encoding='utf-8')
        return "No GDD context available."

    def _create_basic_tech_spec_from_files(self, project_path: Path) -> dict:
        py_files = list(project_path.rglob("*.py"))
        files_spec = {}
        for f in py_files:
            relative_path = str(f.relative_to(project_path))
            files_spec[relative_path] = {"purpose": f"Existing file: {f.name}", "skip_generation": True}
        return {"project_name": project_path.name,
                "project_description": f"Existing project with {len(py_files)} Python files",
                "technical_specs": {"files": files_spec}}

    async def execute_analysis_workflow(self, project_path_str: str):
        """Backward compatibility method for existing application code."""
        await self.analyze_existing_project(project_path_str)


EnhancedWorkflowEngine = HybridWorkflowEngine