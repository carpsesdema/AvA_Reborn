# core/enhanced_workflow_engine.py - V6.5 - Self-Correction Loop Integrated!

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict

from PySide6.QtCore import QObject, Signal

from core.services.architect_service import ArchitectService
from core.services.coder_service import CoderService
from core.services.reviewer_service import ReviewerService
from core.services.assembler_service import AssemblerService
from core.project_state_manager import ProjectStateManager
from core.enhanced_micro_task_engine import StreamlinedMicroTaskEngine, SimpleTaskSpec
from core.project_builder import ProjectBuilder
from core.domain_context_manager import DomainContextManager
# --- NEW IMPORT ---
from core.execution_engine import ExecutionEngine, ExecutionResult


class HybridWorkflowEngine(QObject):
    """
    🚀 V6.5 Hybrid Workflow Engine: Now with a self-correction loop!
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

    node_status_changed = Signal(str, str, str)
    workflow_plan_ready = Signal(dict)
    data_flow_started = Signal(str, str)
    data_flow_completed = Signal(str, str)
    workflow_reset = Signal()

    def __init__(self, llm_client, terminal_window, code_viewer, rag_manager=None):
        super().__init__()
        self.llm_client = llm_client
        self.streaming_terminal = terminal_window
        self.code_viewer = code_viewer
        self.rag_manager = rag_manager
        self.logger = logging.getLogger(__name__)

        self.project_state_manager: Optional[ProjectStateManager] = None
        self.domain_context_manager: Optional[DomainContextManager] = None
        self.current_tech_spec: dict = None
        self.is_existing_project_loaded = False
        self.original_project_path: Optional[Path] = None
        self.original_user_prompt: str = ""
        self.micro_task_engine: Optional[StreamlinedMicroTaskEngine] = None
        # --- NEW: Instantiate ExecutionEngine ---
        self.execution_engine: Optional[ExecutionEngine] = None

        def service_log_emitter(agent_name: str, type_key: str, content: str, indent_level: int):
            self.detailed_log_event.emit(agent_name, type_key, content, str(indent_level))

        self.architect_service = ArchitectService(self.llm_client, service_log_emitter, self.rag_manager)
        self.coder_service = CoderService(self.llm_client, service_log_emitter, self.rag_manager)
        self.assembler_service = AssemblerService(self.llm_client, service_log_emitter, self.rag_manager)
        self.reviewer_service = ReviewerService(self.llm_client, service_log_emitter, self.rag_manager)

        from core.config import ConfigManager
        config = ConfigManager()
        self.project_builder = ProjectBuilder(
            workspace_root=config.app_config.workspace_path,
            stream_emitter=service_log_emitter,
            logger=self.logger
        )
        self._connect_terminal_signals()

    def _connect_terminal_signals(self):
        if self.streaming_terminal and hasattr(self.streaming_terminal, 'stream_log_rich'):
            try:
                self.detailed_log_event.connect(self.streaming_terminal.stream_log_rich)
            except Exception as e:
                self.logger.error(f"❌ Failed to connect terminal signals: {e}")

    def set_project_state(self, project_state_manager: ProjectStateManager):
        """Connects the project state to all services and initializes dependent components."""
        self.project_state_manager = project_state_manager
        self.domain_context_manager = DomainContextManager(project_state_manager.project_root)

        self.architect_service.set_project_state(project_state_manager)
        self.coder_service.set_project_state(project_state_manager)
        self.assembler_service.set_project_state(project_state_manager)
        self.reviewer_service.set_project_state(project_state_manager)

        self.micro_task_engine = StreamlinedMicroTaskEngine(
            llm_client=self.llm_client,
            domain_context_manager=self.domain_context_manager
        )
        # --- NEW: Initialize ExecutionEngine with the project state ---
        self.execution_engine = ExecutionEngine(
            project_state_manager=project_state_manager,
            stream_emitter=lambda agent, type_key, content, level: self.detailed_log_event.emit(agent, type_key,
                                                                                                content, str(level))
        )
        self.logger.info("Project state correctly set for all AI services, including ExecutionEngine.")

    async def _prepare_and_set_domain_context(self):
        """Analyzes the project's domain context ONCE and stores it in the state manager."""
        if not self.domain_context_manager:
            return
        if self.project_state_manager and self.project_state_manager.domain_context is not None:
            self.detailed_log_event.emit("HybridEngine", "info", "Domain context already analyzed and cached.", "1")
            return

        self.detailed_log_event.emit("HybridEngine", "stage_start", "🔬 Analyzing project domain context...", "1")
        self.node_status_changed.emit("architect", "working", "Analyzing domain...")

        try:
            comprehensive_context = await self.domain_context_manager.get_comprehensive_context()
            if self.project_state_manager:
                self.project_state_manager.domain_context = comprehensive_context

            self.detailed_log_event.emit("HybridEngine", "success", "✅ Domain context analysis complete.", "1")
            self.node_status_changed.emit("architect", "success", "Domain analyzed.")
        except Exception as e:
            self.logger.error(f"Domain context analysis failed: {e}", exc_info=True)
            self.detailed_log_event.emit("HybridEngine", "error", f"❌ Domain analysis failed: {e}", "1")
            self.node_status_changed.emit("architect", "error", "Domain analysis failed.")

    async def analyze_existing_project(self, project_path_str: str):
        """Analyze an existing project and create technical specification."""
        try:
            self.workflow_reset.emit()
            project_path = Path(project_path_str)
            self.analysis_started.emit(project_path_str)
            self.detailed_log_event.emit("HybridEngine", "stage_start", f"🔍 Analyzing: {project_path.name}", "0")

            if not self.project_state_manager or self.project_state_manager.project_root != project_path:
                self.set_project_state(ProjectStateManager(str(project_path)))

            await self._prepare_and_set_domain_context()

            self.node_status_changed.emit("architect", "working", "Creating spec from code...")
            self.current_tech_spec = await self.architect_service.analyze_and_create_spec_from_project(
                self.project_state_manager)

            if not self.current_tech_spec:
                raise Exception("Failed to analyze project and create a specification.")

            self.node_status_changed.emit("architect", "success", "Analysis complete.")
            self.is_existing_project_loaded = True
            self.original_project_path = project_path

            self._ensure_gdd_exists(project_path, self.current_tech_spec.get("project_name", project_path.name),
                                    self.current_tech_spec.get("project_description", ""))
            self.detailed_log_event.emit("HybridEngine", "success", "✅ Analysis complete! Ready for modifications.",
                                         "0")
            self.analysis_completed.emit(project_path_str, self.current_tech_spec)
            self.project_loaded.emit(project_path_str)
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}", exc_info=True)
            self.detailed_log_event.emit("HybridEngine", "error", f"❌ Analysis failed: {str(e)}", "0")
            self.analysis_completed.emit(project_path_str, self.current_tech_spec or {})

    async def execute_enhanced_workflow(self, user_prompt: str, conversation_context: list = None):
        """Execute the V6.5 Hybrid workflow with self-correction."""
        self.logger.info(f"🚀 Starting V6.5 Hybrid workflow: {user_prompt[:100]}...")
        self.workflow_reset.emit()
        workflow_start_time = datetime.now()
        self.original_user_prompt = user_prompt

        try:
            if self.is_existing_project_loaded:
                project_name = self.original_project_path.name
                self.detailed_log_event.emit("HybridEngine", "stage_start",
                                             f"🚀 Initializing HYBRID MODIFICATION for '{project_name}'...", "0")
                await self._prepare_and_set_domain_context()
            else:
                self.detailed_log_event.emit("HybridEngine", "stage_start", "🚀 Initializing HYBRID NEW PROJECT...", "0")

            self.workflow_progress.emit("planning", "🧠 Architecting project...")
            self.node_status_changed.emit("architect", "working", "Creating tech spec...")
            tech_spec = await self.architect_service.create_tech_spec(user_prompt, conversation_context)
            if not tech_spec or 'technical_specs' not in tech_spec:
                raise Exception("Architecture phase failed")
            self.node_status_changed.emit("architect", "success", "Tech spec created.")
            self.workflow_plan_ready.emit(tech_spec)
            self.current_tech_spec = tech_spec

            self.data_flow_started.emit("architect", "coder")
            self.workflow_progress.emit("micro-tasking", "📋 Breaking into micro-tasks...")
            project_context_for_tasks = {"project_name": tech_spec.get("project_name", ""),
                                         "project_description": tech_spec.get("project_description", "")}

            files_to_plan = tech_spec.get("technical_specs", {}).get("files", {})
            all_micro_tasks = await self.micro_task_engine.create_smart_tasks(files_to_plan, project_context_for_tasks)

            tasks_by_file = defaultdict(list)
            for task in all_micro_tasks:
                if task.file_path: tasks_by_file[task.file_path].append(task)
            self.detailed_log_event.emit("HybridEngine", "success", f"📊 Created {len(all_micro_tasks)} micro-tasks.",
                                         "1")
            self.data_flow_completed.emit("architect", "coder")

            project_dir = await self._setup_project_directory(tech_spec)
            if not self.project_state_manager:
                self.set_project_state(ProjectStateManager(str(project_dir)))

            # --- MODIFIED: Call the new file generation method ---
            await self._execute_self_correcting_file_generation(tech_spec, project_dir, tasks_by_file)
            await self._finalize_hybrid_project(tech_spec, project_dir, workflow_start_time)

        except Exception as e:
            self.logger.error(f"Hybrid workflow failed: {e}", exc_info=True)
            self.detailed_log_event.emit("HybridEngine", "error", f"❌ Workflow failed: {str(e)}", "0")
            self.workflow_completed.emit({"success": False, "error": str(e)})

    # --- ENTIRELY NEW METHOD ---
    async def _execute_self_correcting_file_generation(self, tech_spec: dict, project_dir: Path,
                                                       tasks_by_file: Dict[str, List[SimpleTaskSpec]]):
        """Generates files using the new self-correction loop."""
        self.workflow_progress.emit("generation", "⚡ Generating & Validating Files...")
        files_to_generate = tech_spec.get("technical_specs", {}).get("files", {})
        if not files_to_generate: return

        generated_files_content = {}
        for idx, (filename, file_spec) in enumerate(files_to_generate.items(), 1):
            self.task_progress.emit(idx, len(files_to_generate))
            if file_spec.get("skip_generation", False): continue

            try:
                # Initial Code Generation
                self.node_status_changed.emit("coder", "working", f"Generating {filename}...")
                micro_tasks_for_file = tasks_by_file.get(filename, [])

                # --- Non-truncation change: Pass full generated_files_content ---
                assembled_code = await self._generate_file_with_micro_tasks(filename, file_spec, tech_spec,
                                                                            generated_files_content,
                                                                            micro_tasks_for_file)
                if not assembled_code:
                    self.logger.warning(f"No code was assembled for {filename}. Skipping.")
                    continue

                # Self-Correction Loop
                max_retries = 3
                for i in range(max_retries):
                    self.node_status_changed.emit("execution_engine", "working",
                                                  f"Validating {filename} (Attempt {i + 1})")
                    self.data_flow_started.emit("assembler", "execution_engine")
                    await asyncio.sleep(0.1)  # UI breath
                    validation_result = await self.execution_engine.validate_code(filename, assembled_code)
                    self.data_flow_completed.emit("assembler", "execution_engine")

                    if validation_result.result == ExecutionResult.SUCCESS:
                        self.node_status_changed.emit("execution_engine", "success", f"{filename} is valid.")
                        break  # Exit loop if successful
                    else:
                        self.node_status_changed.emit("execution_engine", "error", f"Validation failed")
                        self.detailed_log_event.emit("HybridEngine", "warning",
                                                     f"[{filename}] {validation_result.error}", "2")
                        if i == max_retries - 1:
                            self.logger.error(f"Exceeded max retries for {filename}. Using last generated code.")
                            self.detailed_log_event.emit("HybridEngine", "error",
                                                         f"Could not fix {filename} after {max_retries} attempts.", "1")
                            break

                        # Refinement Step
                        self.node_status_changed.emit("reviewer", "working", f"Refining {filename}...")
                        self.data_flow_started.emit("execution_engine", "reviewer")

                        instruction = f"The code for {filename} failed validation with this error: '{validation_result.error}'. Please fix the code and return only the complete, corrected code."
                        assembled_code = await self.reviewer_service.refine_code(assembled_code, instruction)

                        self.data_flow_completed.emit("execution_engine", "reviewer")
                        self.node_status_changed.emit("reviewer", "success", f"{filename} refined.")

                # Final step: write the (hopefully) corrected code and log for dependency context
                self._write_file(project_dir, filename, assembled_code)
                generated_files_content[filename] = assembled_code

            except Exception as e:
                self.logger.error(f"Failed to generate {filename}: {e}", exc_info=True)

    async def _generate_file_with_micro_tasks(self, filename: str, file_spec: dict, tech_spec: dict,
                                              generated_files: dict, micro_tasks: List[SimpleTaskSpec]) -> str:
        if not micro_tasks:
            # Use the full, untruncated dependency context for fallback
            dependency_context = self._build_dependency_context(file_spec.get("dependencies", []), generated_files)
            return await self.coder_service.generate_file_from_spec(filename, file_spec, {
                "description": tech_spec.get("project_description", "")}, dependency_context)

        micro_task_results = await self._execute_micro_tasks_in_parallel(micro_tasks)
        if not micro_task_results:
            self.logger.warning(f"Skipping assembly of {filename} as all its micro-tasks failed.")
            self.detailed_log_event.emit("Assembler", "warning", f"Skipping {filename}: no completed micro-tasks.", "2")
            return ""

        self.node_status_changed.emit("assembler", "working", f"Assembling {filename}...")
        self.data_flow_started.emit("coder", "assembler")
        project_context = {"description": tech_spec.get("project_description", ""),
                           "file_purpose": file_spec.get("purpose", "")}
        assembled_code = await self.assembler_service.assemble_file_from_micro_tasks(filename, file_spec,
                                                                                     micro_task_results,
                                                                                     project_context)
        self.data_flow_completed.emit("coder", "assembler")
        self.node_status_changed.emit("assembler", "success", f"{filename} assembled.")
        return assembled_code

    async def _execute_micro_tasks_in_parallel(self, micro_tasks: List[SimpleTaskSpec]) -> List[Dict[str, Any]]:
        if not micro_tasks: return []
        semaphore = asyncio.Semaphore(5)

        async def execute_task(task: SimpleTaskSpec):
            async with semaphore:
                try:
                    return await self.coder_service.execute_micro_task_with_gemini_flash(task)
                except Exception as e:
                    self.logger.error(f"Micro-task {task.id} failed permanently: {e}", exc_info=False)
                    self.detailed_log_event.emit("HybridEngine", "error", f"❌ Failed to execute micro-task {task.id}.",
                                                 "2")
                    return None

        results = await asyncio.gather(*[execute_task(task) for task in micro_tasks])
        return [res for res in results if res is not None]

    async def _setup_project_directory(self, tech_spec: dict) -> Path:
        project_name = tech_spec.get("project_name", "generated_project")
        if self.is_existing_project_loaded and self.original_project_path:
            return self.project_builder.setup_project_directory(project_name=project_name, is_modification=True,
                                                                original_user_prompt=self.original_user_prompt,
                                                                original_project_path=self.original_project_path)
        return self.project_builder.setup_project_directory(project_name=project_name, is_modification=False,
                                                            original_user_prompt=self.original_user_prompt,
                                                            original_project_path=None)

    def _write_file(self, project_dir: Path, filename: str, content: str):
        try:
            file_path = project_dir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding='utf-8')
            self.file_generated.emit(str(file_path))
        except Exception as e:
            self.logger.error(f"Failed to write file {filename}: {e}", exc_info=True)

    def _build_dependency_context(self, dependencies: List[str], generated_files: Dict[str, Any]) -> str:
        # --- NO TRUNCATION CHANGE ---
        # Provide the FULL source code of dependencies
        context = []
        for dep_filename in dependencies:
            if dep_filename in generated_files:
                context.append(f"--- DEPENDENCY: {dep_filename} ---\n\n```python\n{generated_files[dep_filename]}\n```")

        if not context:
            return "This file has no generated dependencies."

        return "\n\n".join(context)

    async def _finalize_hybrid_project(self, tech_spec: dict, project_dir: Path, start_time: datetime):
        self.workflow_progress.emit("finalization", "🎯 Finalizing project...")
        self._create_requirements_txt(project_dir, tech_spec)
        elapsed_time = (datetime.now() - start_time).total_seconds()
        result = {"success": True, "project_directory": str(project_dir), "execution_time": elapsed_time}
        self.workflow_progress.emit("complete", "✅ Hybrid workflow complete!")
        self.workflow_completed.emit(result)

    def _create_requirements_txt(self, project_dir: Path, tech_spec: dict):
        requirements = tech_spec.get("technical_specs", {}).get("requirements", [])
        if requirements: (project_dir / "requirements.txt").write_text("\n".join(requirements) + "\n")

    def _ensure_gdd_exists(self, project_path: Path, project_name: str, project_description: str):
        if not (project_path / f"{project_name}_GDD.md").exists():
            (project_path / f"{project_name}_GDD.md").write_text(f"# {project_name}\n\n{project_description}\n")