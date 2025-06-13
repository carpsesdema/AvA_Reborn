# core/enhanced_workflow_engine.py - V6.8 - Staged Dependency-Aware Pipeline

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque
import subprocess
import sys

from PySide6.QtCore import QObject, Signal

from core.services.architect_service import ArchitectService
from core.services.coder_service import CoderService
from core.services.reviewer_service import ReviewerService
from core.services.assembler_service import AssemblerService
from core.project_state_manager import ProjectStateManager
from core.enhanced_micro_task_engine import SimpleTaskSpec
from core.project_builder import ProjectBuilder
from core.domain_context_manager import DomainContextManager
from core.execution_engine import ExecutionEngine, ExecutionResult


class HybridWorkflowEngine(QObject):
    """
    🚀 V6.8 Hybrid Workflow Engine: Staged, dependency-aware parallel pipeline.
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
        self.execution_engine: Optional[ExecutionEngine] = None

        def service_log_emitter(agent_name: str, type_key: str, content: str, indent_level: int):
            self.detailed_log_event.emit(agent_name, type_key, content, str(indent_level))

        self.architect_service = ArchitectService(self.llm_client, service_log_emitter, self.rag_manager)
        self.coder_service = CoderService(self.llm_client, service_log_emitter, self.rag_manager)
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
        self.reviewer_service.set_project_state(project_state_manager)

        self.execution_engine = ExecutionEngine(
            project_state_manager=project_state_manager,
            stream_emitter=lambda agent, type_key, content, level: self.detailed_log_event.emit(agent, type_key,
                                                                                                content, str(level))
        )
        self.logger.info("Project state correctly set for all AI services, including ExecutionEngine.")

    async def execute_enhanced_workflow(self, user_prompt: str, conversation_context: list = None):
        """Execute the V6.8 Hybrid workflow."""
        self.logger.info(f"🚀 Starting V6.8 Hybrid workflow: {user_prompt[:100]}...")
        self.workflow_reset.emit()
        workflow_start_time = datetime.now()
        self.original_user_prompt = user_prompt

        try:
            # --- ARCHITECTURE PHASE ---
            self.workflow_progress.emit("planning", "🧠 Architecting project...")
            self.node_status_changed.emit("architect", "working", "Creating tech spec...")

            # The Architect service now returns the full spec at once after streaming logs.
            tech_spec = await self.architect_service.create_tech_spec(user_prompt, conversation_context)
            if not tech_spec or not tech_spec.get("technical_specs", {}).get("files"):
                raise Exception("Architecture phase failed to produce any file specifications.")

            self.current_tech_spec = tech_spec
            self.node_status_changed.emit("architect", "success", "Tech spec created.")
            self.workflow_plan_ready.emit(tech_spec)

            # --- STAGED GENERATION PIPELINE ---
            project_dir, generated_files = await self.run_staged_generation_pipeline(tech_spec)

            await self._finalize_hybrid_project(self.current_tech_spec, project_dir, workflow_start_time,
                                                len(generated_files))

        except Exception as e:
            self.logger.error(f"Hybrid workflow failed: {e}", exc_info=True)
            self.detailed_log_event.emit("HybridEngine", "error", f"❌ Workflow failed: {str(e)}", "0")
            self.workflow_completed.emit({"success": False, "error": str(e)})

    def _build_dependency_graph(self, files_to_generate: Dict[str, Any]) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        """Builds a dependency graph from the tech spec."""
        dependencies = defaultdict(list)
        file_map = {}  # Maps key (e.g., 'main') to full filename (e.g., 'main.py')

        # Create a mapping from simple name (without ext) to full filename
        for full_filename in files_to_generate.keys():
            key = Path(full_filename).stem
            file_map[key] = full_filename

        for full_filename, spec in files_to_generate.items():
            key = Path(full_filename).stem
            # Ensure every file is a key in the dependency graph
            if key not in dependencies:
                dependencies[key] = []

            for dep_filename in spec.get("dependencies", []):
                dep_key = Path(dep_filename).stem
                # Only add dependency if it's a known file being generated
                if dep_key in file_map:
                    dependencies[key].append(dep_key)
        return dict(dependencies), file_map

    def _determine_generation_stages(self, dependencies: Dict[str, List[str]]) -> List[List[str]]:
        """Determines parallel generation stages using Kahn's algorithm for topological sorting."""
        if not dependencies: return []

        in_degree = {u: 0 for u in dependencies}
        for u in dependencies:
            for v in dependencies[u]:
                if v in in_degree: in_degree[v] += 1

        queue = deque([u for u, deg in in_degree.items() if deg == 0])
        stages = []
        count = 0
        while queue:
            current_stage = list(queue)
            stages.append(current_stage)

            for _ in range(len(current_stage)):
                u = queue.popleft()
                count += 1
                for v in dependencies.get(u, []):
                    if v in in_degree:
                        in_degree[v] -= 1
                        if in_degree[v] == 0:
                            queue.append(v)

        if count != len(dependencies):
            cycle_nodes = {node for node, deg in in_degree.items() if deg > 0}
            self.logger.error(
                f"A cycle was detected in the dependency graph involving nodes: {cycle_nodes}. Execution order may be incorrect.")
            if remaining_nodes := list(cycle_nodes):
                stages.append(remaining_nodes)
        return stages

    def _build_dependency_context(self, dependency_keys: List[str], knowledge_packets: Dict[str, str],
                                  project_description: str, file_map: Dict[str, str]) -> str:
        """Builds a rich context string including full source code of completed dependencies."""
        if not dependency_keys:
            return "This file has no project dependencies."

        context_str = f"You are building a file for the project '{project_description}'.\n"
        context_str += "This file must correctly import and use functions/classes from the following dependencies. The full source code for each completed dependency is provided for your reference:\n"

        for dep_key in dependency_keys:
            full_dep_filename = file_map.get(dep_key)
            if full_dep_filename and full_dep_filename in knowledge_packets:
                source_code = knowledge_packets[full_dep_filename]
                context_str += f"\n--- SOURCE CODE FOR DEPENDENCY: {full_dep_filename} ---\n"
                context_str += f"```python\n{source_code}\n```\n"
            else:
                context_str += f"\n--- NOTE: Source code for '{dep_key}' is not yet available. You may need to create placeholder imports. ---\n"
        return context_str

    async def run_staged_generation_pipeline(self, tech_spec: dict) -> Tuple[Path, Dict[str, str]]:
        """Runs the generation pipeline in stages based on file dependencies."""
        project_dir = await self._setup_project_directory(tech_spec)
        files_to_generate = tech_spec.get("technical_specs", {}).get("files", {})
        if not files_to_generate: return project_dir, {}

        dependencies, file_map = self._build_dependency_graph(files_to_generate)
        stages = self._determine_generation_stages(dependencies)

        knowledge_packets = {}  # Stores filename -> source_code

        for i, stage_keys in enumerate(stages):
            self.detailed_log_event.emit("HybridEngine", "stage_start",
                                         f"Executing Stage {i + 1}/{len(stages)} with {len(stage_keys)} file(s) in parallel...",
                                         "1")

            stage_tasks = []
            for key in stage_keys:
                full_filename = file_map.get(key)
                if not full_filename: continue

                file_spec = files_to_generate.get(full_filename)
                dep_keys = dependencies.get(key, [])

                task = self._generate_file_pipeline_task(full_filename, file_spec, tech_spec, knowledge_packets,
                                                         dep_keys, file_map)
                stage_tasks.append(task)

            # Wait for all tasks in the current stage to complete
            results = await asyncio.gather(*stage_tasks)
            for file_name, source_code in results:
                if source_code:
                    knowledge_packets[file_name] = source_code
                    self._write_file(project_dir, file_name, source_code)

        return project_dir, knowledge_packets

    async def _generate_file_pipeline_task(self, filename: str, file_spec: dict, tech_spec: dict,
                                           knowledge_packets: dict, dep_keys: List[str], file_map: dict) -> Tuple[
        str, Optional[str]]:
        """A complete, self-contained pipeline for generating a single file."""
        try:
            self.node_status_changed.emit("coder", "working", f"Coding {filename}...")

            dependency_context = self._build_dependency_context(dep_keys, knowledge_packets,
                                                                tech_spec.get("project_description", ""), file_map)
            project_context = {"description": tech_spec.get("project_description", "")}

            generated_code = await self.coder_service.generate_file_from_spec(filename, file_spec, project_context,
                                                                              dependency_context)

            max_retries = 3
            for i in range(max_retries):
                self.node_status_changed.emit("execution_engine", "working", f"Validating {filename} (Try {i + 1})")
                validation_result = await self.execution_engine.validate_code(filename, generated_code)

                if validation_result.result == ExecutionResult.SUCCESS:
                    self.node_status_changed.emit("execution_engine", "success", f"{filename} is valid.")
                    self.detailed_log_event.emit("HybridEngine", "success", f"✅ {filename} passed validation.", "2")
                    return filename, generated_code

                self.node_status_changed.emit("execution_engine", "error", f"{filename} invalid")
                self.detailed_log_event.emit("HybridEngine", "warning", f"[{filename}] {validation_result.clean_error}",
                                             "2")

                if i < max_retries - 1:
                    self.node_status_changed.emit("reviewer", "working", f"Refining {filename}...")
                    instruction = f"The code for {filename} failed validation. Please fix it. The error was: {validation_result.clean_error}"
                    generated_code = await self.reviewer_service.refine_code(generated_code, instruction)
                    self.node_status_changed.emit("reviewer", "success", f"{filename} refined.")
                else:
                    self.detailed_log_event.emit("HybridEngine", "error",
                                                 f"Could not fix {filename} after {max_retries} attempts. Using last version.",
                                                 "1")
                    return filename, generated_code  # Return the last attempt

            return filename, generated_code  # Should be unreachable, but as a fallback
        except Exception as e:
            self.logger.error(f"Task for generating {filename} failed: {e}", exc_info=True)
            self.detailed_log_event.emit("HybridEngine", "error", f"❌ Generation pipeline for {filename} failed.", "2")
            return filename, None  # Indicate failure

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

    async def _finalize_hybrid_project(self, tech_spec: dict, project_dir: Path, start_time: datetime, file_count: int):
        self.workflow_progress.emit("finalization", "🎯 Finalizing project...")
        if tech_spec: self._create_requirements_txt(project_dir, tech_spec)
        await self._setup_virtual_environment(project_dir)
        elapsed_time = (datetime.now() - start_time).total_seconds()
        result = {"success": True, "project_directory": str(project_dir), "execution_time": elapsed_time,
                  "file_count": file_count}
        self.workflow_progress.emit("complete", "✅ Hybrid workflow complete!")
        self.workflow_completed.emit(result)

    def _create_requirements_txt(self, project_dir: Path, tech_spec: dict):
        requirements = tech_spec.get("technical_specs", {}).get("requirements", [])
        if requirements: (project_dir / "requirements.txt").write_text("\n".join(set(requirements)) + "\n")

    async def _setup_virtual_environment(self, project_dir: Path):
        venv_path = project_dir / "venv"
        if venv_path.exists(): return

        self.detailed_log_event.emit("HybridEngine", "info", "Creating virtual environment...", "2")
        try:
            loop = asyncio.get_event_loop()
            cmd = [sys.executable, '-m', 'venv', venv_path.name]
            result = await loop.run_in_executor(None,
                                                lambda: subprocess.run(cmd, capture_output=True, text=True, check=False,
                                                                       cwd=str(project_dir)))
            if result.returncode == 0:
                self.detailed_log_event.emit("HybridEngine", "success", "Virtual environment created.", "3")
                req_file = project_dir / "requirements.txt"
                if req_file.exists() and req_file.read_text().strip(): await self._install_requirements(venv_path,
                                                                                                        req_file)
            else:
                self.detailed_log_event.emit("HybridEngine", "error",
                                             f"Failed to create venv: {result.stderr or result.stdout}", "3")
        except Exception as e:
            self.logger.error(f"Failed to create virtual environment: {e}", exc_info=True)

    async def _install_requirements(self, venv_path: Path, req_file: Path):
        self.detailed_log_event.emit("HybridEngine", "info", "Installing requirements...", "3")
        pip_path = venv_path / "Scripts" / "pip.exe" if sys.platform == "win32" else venv_path / "bin" / "pip"
        if not pip_path.exists():
            self.detailed_log_event.emit("HybridEngine", "error", f"Pip executable not found at {pip_path}", "3")
            return

        loop = asyncio.get_event_loop()
        cmd = [str(pip_path), 'install', '-r', str(req_file)]
        pip_result = await loop.run_in_executor(None,
                                                lambda: subprocess.run(cmd, capture_output=True, text=True, check=False,
                                                                       cwd=str(req_file.parent)))
        if pip_result.returncode == 0:
            self.detailed_log_event.emit("HybridEngine", "success", "Requirements installed successfully.", "3")
        else:
            self.detailed_log_event.emit("HybridEngine", "warning",
                                         f"Requirements installation had issues: {pip_result.stderr[:300]}", "3")