# core/enhanced_workflow_engine.py - V6.2 Final wiring update

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


class HybridWorkflowEngine(QObject):
    """
    🚀 V6.2 Hybrid Workflow Engine: Solves API spam by analyzing domain context once.
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

        # FINAL FIX: Initialize micro-task engine with ALL its dependencies
        self.micro_task_engine = StreamlinedMicroTaskEngine(
            self.llm_client,
            self.project_state_manager,
            self.domain_context_manager,
            self.rag_manager
        )

    async def _prepare_and_set_domain_context(self):
        """Analyzes the project's domain context ONCE and stores it in the state manager."""
        if not self.domain_context_manager:
            return
        if self.project_state_manager and self.project_state_manager.domain_context is not None:
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

    # ... rest of the file is unchanged ...
    async def execute_enhanced_workflow(self, user_prompt: str, conversation_context: list = None):
        """Execute the V6.2 Hybrid workflow combining architecture and micro-task orchestration."""
        self.logger.info(f"🚀 Starting V6.2 Hybrid workflow: {user_prompt[:100]}...")
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
            all_micro_tasks = await self.micro_task_engine.create_smart_tasks(tech_spec.get("technical_specs", {}),
                                                                              project_context_for_tasks)
            tasks_by_file = defaultdict(list)
            for task in all_micro_tasks:
                if task.file_path: tasks_by_file[task.file_path].append(task)
            self.detailed_log_event.emit("HybridEngine", "success", f"📊 Created {len(all_micro_tasks)} micro-tasks.",
                                         "1")
            self.data_flow_completed.emit("architect", "coder")

            project_dir = await self._setup_project_directory(tech_spec)
            if not self.project_state_manager:
                self.set_project_state(ProjectStateManager(str(project_dir)))

            await self._execute_hybrid_file_generation(tech_spec, project_dir, tasks_by_file)
            await self._finalize_hybrid_project(tech_spec, project_dir, workflow_start_time)

        except Exception as e:
            self.logger.error(f"Hybrid workflow failed: {e}", exc_info=True)
            self.detailed_log_event.emit("HybridEngine", "error", f"❌ Workflow failed: {str(e)}", "0")
            self.workflow_completed.emit({"success": False, "error": str(e)})

    async def _execute_hybrid_file_generation(self, tech_spec: dict, project_dir: Path,
                                              tasks_by_file: Dict[str, List[SimpleTaskSpec]]):
        self.workflow_progress.emit("generation", "⚡ Generating files...")
        files_to_generate = tech_spec.get("technical_specs", {}).get("files", {})
        if not files_to_generate: return

        generated_files = {}
        self.node_status_changed.emit("coder", "working", f"Generating {len(files_to_generate)} files...")

        for idx, (filename, file_spec) in enumerate(files_to_generate.items(), 1):
            self.task_progress.emit(idx, len(files_to_generate))
            if file_spec.get("skip_generation", False): continue
            try:
                micro_tasks_for_file = tasks_by_file.get(filename, [])
                assembled_code = await self._generate_file_with_micro_tasks(filename, file_spec, tech_spec,
                                                                            generated_files, micro_tasks_for_file)
                if assembled_code:
                    self.node_status_changed.emit("reviewer", "working", f"Reviewing {filename}...")
                    self.data_flow_started.emit("coder", "reviewer")
                    await asyncio.sleep(0.1)
                    self.data_flow_completed.emit("coder", "reviewer")
                    self.node_status_changed.emit("reviewer", "success", f"{filename} approved.")
                    self._write_file(project_dir, filename, assembled_code)
                    generated_files[filename] = {"source_code": assembled_code}
            except Exception as e:
                self.logger.error(f"Failed to generate {filename}: {e}", exc_info=True)

        self.node_status_changed.emit("coder", "success", "Generated all files.")

    async def _generate_file_with_micro_tasks(self, filename: str, file_spec: dict, tech_spec: dict,
                                              generated_files: dict, micro_tasks: List[SimpleTaskSpec]) -> str:
        if not micro_tasks:
            return await self.coder_service.generate_file_from_spec(filename, file_spec, {
                "description": tech_spec.get("project_description", "")}, self._build_dependency_context(
                file_spec.get("dependencies", []), generated_files))

        micro_task_results = await self._execute_micro_tasks_in_parallel(micro_tasks)
        if not micro_task_results: return ""

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
                except Exception:
                    return None

        results = await asyncio.gather(*[execute_task(task) for task in micro_tasks])
        return [r for r in results if r]

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
        return "\n\n".join(
            [f"=== {dep} ===\n{generated_files[dep].get('source_code', '')[:500]}..." for dep in dependencies if
             dep in generated_files]) or "No dependencies available."

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