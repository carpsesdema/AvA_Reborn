# core/enhanced_workflow_engine.py - V6.0 Hybrid Workflow with Micro-Task Orchestration

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

from core.workflow_services import ArchitectService, CoderService, ReviewerService, AssemblerService
from core.project_state_manager import ProjectStateManager
from core.enhanced_micro_task_engine import StreamlinedMicroTaskEngine, SimpleTaskSpec


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

        # Connect all services to project state
        self.architect_service.set_project_state(project_state_manager)
        self.coder_service.set_project_state(project_state_manager)
        self.assembler_service.set_project_state(project_state_manager)
        self.reviewer_service.set_project_state(project_state_manager)

        # Update micro-task engine
        self.micro_task_engine.project_state = project_state_manager

    async def analyze_existing_project(self, project_path_str: str):
        """Analyze an existing project and create technical specification."""
        try:
            self.analysis_started.emit(project_path_str)
            project_path = Path(project_path_str)

            self.detailed_log_event.emit("HybridEngine", "stage_start",
                                         f"🔍 Starting project analysis for: {project_path.name}", "0")

            # Initialize project state manager for this project
            if not self.project_state_manager:
                self.project_state_manager = ProjectStateManager(project_path)
                self.set_project_state(self.project_state_manager)

            # Project is automatically scanned and loaded by ProjectStateManager constructor

            # Create tech spec from existing project
            self.current_tech_spec = await self.architect_service.analyze_and_create_spec_from_project(
                self.project_state_manager
            )

            if not self.current_tech_spec:
                raise Exception("Failed to analyze project structure")

            # Mark as existing project
            self.is_existing_project_loaded = True
            self.original_project_path = project_path

            # Extract project details
            project_name = self.current_tech_spec.get("project_name", project_path.name)
            project_description = self.current_tech_spec.get("project_description", "Analyzed existing project")

            # Ensure GDD exists
            self._ensure_gdd_exists(project_path, project_name, project_description)

            self.detailed_log_event.emit("HybridEngine", "success",
                                         "✅ Analysis complete! Technical spec created.", "0")

            self.analysis_completed.emit(project_path_str, self.current_tech_spec)
            self.project_loaded.emit(project_path_str)

        except Exception as e:
            self.logger.error(f"❌ Analysis failed: {e}", exc_info=True)
            self.detailed_log_event.emit("HybridEngine", "error", f"❌ Analysis Error: {str(e)}", "0")

            # Create fallback tech spec
            try:
                project_path = Path(project_path_str)
                self.current_tech_spec = self._create_basic_tech_spec_from_files(project_path)
                self.is_existing_project_loaded = True
                self.original_project_path = project_path
                self.detailed_log_event.emit("HybridEngine", "info",
                                             "Using fallback tech spec for future modifications", "0")
            except Exception as fallback_error:
                self.logger.error(f"Failed to create fallback tech spec: {fallback_error}")

            # Still emit project_loaded signal
            self.project_loaded.emit(project_path_str)

    async def execute_enhanced_workflow(self, user_prompt: str, conversation_context: List[Dict] = None):
        """Execute the hybrid workflow with micro-task orchestration."""
        self.logger.info(f"🚀 Starting V6.0 Hybrid workflow: {user_prompt[:100]}...")
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
            # Phase 1: Architecture (Big Model)
            full_user_prompt = f"{user_prompt}\n\n--- GDD CONTEXT ---\n{gdd_context}"
            self.workflow_progress.emit("planning", "🧠 Architecting project with big model...")

            tech_spec = await self.architect_service.create_tech_spec(full_user_prompt, conversation_context)

            if not tech_spec or 'technical_specs' not in tech_spec:
                raise Exception("Architecture phase failed - no technical specifications generated")

            self.current_tech_spec = tech_spec

            # Phase 2: Project Setup
            project_dir = await self._setup_project_directory(tech_spec)

            # Initialize project state if not already done
            if not self.project_state_manager:
                self.project_state_manager = ProjectStateManager(project_dir)
                self.set_project_state(self.project_state_manager)

            # Phase 3: Hybrid File Generation
            await self._execute_hybrid_file_generation(tech_spec, project_dir)

            # Phase 4: Finalization
            self.workflow_progress.emit("finalization", "🎯 Finalizing project...")
            await self._finalize_hybrid_project(tech_spec, project_dir, workflow_start_time)

            self.workflow_progress.emit("complete", "✅ Hybrid workflow complete!")

        except Exception as e:
            self.logger.error(f"❌ Hybrid Workflow failed: {e}", exc_info=True)
            self.detailed_log_event.emit("HybridEngine", "error", f"❌ Workflow Error: {str(e)}", "0")

            # Emit failure result
            self.workflow_completed.emit({
                "success": False,
                "error": str(e),
                "files_created": [],
                "project_directory": None
            })

    async def _execute_hybrid_file_generation(self, tech_spec: dict, project_dir: Path):
        """Execute the hybrid approach: micro-tasks + assembly for each file."""
        technical_specs = tech_spec.get("technical_specs", {})
        files_to_generate = technical_specs.get("files", {})

        if not files_to_generate:
            self.detailed_log_event.emit("HybridEngine", "warning", "No files specified for generation", "1")
            return

        self.detailed_log_event.emit("HybridEngine", "stage_start",
                                     f"🔄 Starting HYBRID generation for {len(files_to_generate)} files", "1")

        generated_files = {}

        for filename, file_spec in files_to_generate.items():
            if not file_spec or file_spec.get("skip_generation", False):
                self.detailed_log_event.emit("HybridEngine", "info",
                                             f"Skipping {filename} - marked for skip", "2")
                continue

            self.detailed_log_event.emit("HybridEngine", "info",
                                         f"🎯 Processing {filename} with hybrid approach", "2")

            try:
                # Step 1: Break file into micro-tasks (Big Model Planning)
                micro_tasks = await self._create_micro_tasks_for_file(filename, file_spec, tech_spec)

                if not micro_tasks:
                    self.detailed_log_event.emit("HybridEngine", "warning",
                                                 f"No micro-tasks generated for {filename}, using fallback", "3")
                    # Fallback to traditional approach
                    code = await self.coder_service.generate_file_from_spec(
                        filename, file_spec,
                        {"description": tech_spec.get("project_description", "")},
                        self._build_dependency_context(file_spec.get("dependencies", []), generated_files)
                    )
                    generated_files[filename] = {"spec": file_spec, "source_code": code}
                    self._write_file(project_dir, filename, code)
                    continue

                # Step 2: Execute micro-tasks in parallel (Gemini Flash)
                self.detailed_log_event.emit("HybridEngine", "info",
                                             f"🚀 Executing {len(micro_tasks)} micro-tasks in parallel", "3")

                micro_task_results = await self._execute_micro_tasks_parallel(micro_tasks)

                # Step 3: Assemble results (Medium Model)
                self.detailed_log_event.emit("HybridEngine", "info",
                                             f"🔧 Assembling {len(micro_task_results)} components", "3")

                assembled_code = await self._assemble_micro_task_results(
                    filename, file_spec, micro_task_results, tech_spec
                )

                # Step 4: Write file
                full_filename = filename if filename.endswith('.py') else f"{filename}.py"
                self._write_file(project_dir, full_filename, assembled_code)

                # Store for dependency context
                generated_files[filename] = {"spec": file_spec, "source_code": assembled_code}

                # Update project state
                if self.project_state_manager:
                    self.project_state_manager.add_file(
                        str(project_dir / full_filename),
                        assembled_code,
                        "hybrid_workflow",
                        f"Generated {full_filename} via hybrid micro-task approach"
                    )

                self.detailed_log_event.emit("HybridEngine", "success",
                                             f"✅ {filename} completed via hybrid approach", "2")

            except Exception as e:
                self.logger.error(f"❌ Hybrid generation failed for {filename}: {e}")
                self.detailed_log_event.emit("HybridEngine", "error",
                                             f"❌ Failed to generate {filename}: {str(e)}", "2")

    async def _create_micro_tasks_for_file(self, filename: str, file_spec: dict, tech_spec: dict) -> List[
        SimpleTaskSpec]:
        """Create micro-tasks for a file using the Planner's component breakdown."""
        try:
            components = file_spec.get("components", [])
            if not components:
                self.detailed_log_event.emit("HybridEngine", "warning",
                                             f"No components defined for {filename}", "4")
                return []

            micro_tasks = []

            for component in components:
                task_spec = SimpleTaskSpec(
                    id=component.get("task_id", f"{filename}_{len(micro_tasks)}"),
                    description=component.get("description", "Implement component"),
                    expected_lines=component.get("estimated_lines", 20),
                    context=f"File: {filename}, Component: {component.get('component_type', 'unknown')}",
                    exact_requirements=json.dumps(component, indent=2),
                    component_type=component.get("component_type", "function"),
                    file_path=filename
                )
                micro_tasks.append(task_spec)

            self.detailed_log_event.emit("HybridEngine", "success",
                                         f"Created {len(micro_tasks)} micro-tasks for {filename}", "4")
            return micro_tasks

        except Exception as e:
            self.logger.error(f"Failed to create micro-tasks for {filename}: {e}")
            return []

    async def _execute_micro_tasks_parallel(self, micro_tasks: List[SimpleTaskSpec]) -> List[Dict[str, Any]]:
        """Execute micro-tasks in parallel using Gemini Flash."""
        results = []

        # Create semaphore to limit concurrent tasks (avoid overwhelming the API)
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent tasks

        async def execute_single_task(task: SimpleTaskSpec) -> Dict[str, Any]:
            async with semaphore:
                try:
                    self.detailed_log_event.emit("HybridEngine", "info",
                                                 f"🔄 Executing micro-task: {task.id}", "5")

                    # Execute micro-task with Gemini Flash (fast/cheap model)
                    result = await self.coder_service.execute_micro_task_with_gemini_flash(task)

                    self.detailed_log_event.emit("HybridEngine", "success",
                                                 f"✅ Completed micro-task: {task.id}", "5")
                    return result

                except Exception as e:
                    self.logger.error(f"Micro-task {task.id} failed: {e}")
                    self.detailed_log_event.emit("HybridEngine", "error",
                                                 f"❌ Micro-task {task.id} failed: {str(e)}", "5")
                    return {"error": str(e), "task_id": task.id}

        # Execute all micro-tasks in parallel
        self.detailed_log_event.emit("HybridEngine", "info",
                                     f"🚀 Starting parallel execution of {len(micro_tasks)} micro-tasks", "4")

        results = await asyncio.gather(*[execute_single_task(task) for task in micro_tasks])

        # Filter out failed tasks
        successful_results = [r for r in results if "error" not in r]
        failed_count = len(results) - len(successful_results)

        if failed_count > 0:
            self.detailed_log_event.emit("HybridEngine", "warning",
                                         f"⚠️ {failed_count} micro-tasks failed", "4")

        self.detailed_log_event.emit("HybridEngine", "success",
                                     f"✅ Parallel execution complete: {len(successful_results)}/{len(micro_tasks)} successful",
                                     "4")

        return successful_results

    async def _assemble_micro_task_results(self, filename: str, file_spec: dict,
                                           micro_task_results: List[Dict[str, Any]], tech_spec: dict) -> str:
        """Assemble micro-task results into a complete file using the Assembler service."""
        try:
            self.detailed_log_event.emit("HybridEngine", "info",
                                         f"🔧 Assembling {len(micro_task_results)} components for {filename}", "4")

            # Prepare context for assembler
            project_context = {
                "description": tech_spec.get("project_description", ""),
                "coding_standards": tech_spec.get("coding_standards", "Follow PEP 8"),
                "project_patterns": tech_spec.get("project_patterns", "Standard Python patterns"),
                "file_purpose": file_spec.get("purpose", "Generated file"),
                "integration_requirements": file_spec.get("dependencies", [])
            }

            # Use assembler service to combine micro-task results
            assembled_code = await self.assembler_service.assemble_file_from_micro_tasks(
                filename, file_spec, micro_task_results, project_context
            )

            self.detailed_log_event.emit("HybridEngine", "success",
                                         f"✅ Assembly complete for {filename}", "4")

            return assembled_code

        except Exception as e:
            self.logger.error(f"Assembly failed for {filename}: {e}")
            self.detailed_log_event.emit("HybridEngine", "error",
                                         f"❌ Assembly failed for {filename}: {str(e)}", "4")

            # Fallback: concatenate code directly
            fallback_code = self._fallback_assembly(micro_task_results)
            return fallback_code

    def _fallback_assembly(self, micro_task_results: List[Dict[str, Any]]) -> str:
        """Fallback assembly method - simple concatenation with basic organization."""
        code_parts = []

        # Add imports first
        imports = set()
        for result in micro_task_results:
            code = result.get("IMPLEMENTED_CODE", "")
            for line in code.split('\n'):
                if line.strip().startswith(('import ', 'from ')):
                    imports.add(line.strip())

        if imports:
            code_parts.extend(sorted(imports))
            code_parts.append("")  # Empty line after imports

        # Add implementation code
        for result in micro_task_results:
            code = result.get("IMPLEMENTED_CODE", "")
            # Remove import lines since we handled them above
            filtered_lines = []
            for line in code.split('\n'):
                if not line.strip().startswith(('import ', 'from ')):
                    filtered_lines.append(line)

            if filtered_lines:
                code_parts.extend(filtered_lines)
                code_parts.append("")  # Empty line between components

        return '\n'.join(code_parts)

    async def _setup_project_directory(self, tech_spec: dict) -> Path:
        """Set up project directory structure."""
        project_name = tech_spec.get("project_name", "generated_project")

        if self.is_existing_project_loaded and self.original_project_path:
            # Use existing project directory
            project_dir = self.original_project_path
            self.detailed_log_event.emit("HybridEngine", "info",
                                         f"Using existing project directory: {project_dir}", "1")
        else:
            # Create new project directory
            from core.config import ConfigManager
            config = ConfigManager()
            workspace_path = Path(config.app_config.workspace_path)
            project_dir = workspace_path / project_name
            project_dir.mkdir(parents=True, exist_ok=True)
            self.detailed_log_event.emit("HybridEngine", "info",
                                         f"Created project directory: {project_dir}", "1")

        return project_dir

    def _write_file(self, project_dir: Path, filename: str, content: str):
        """Write a file to the project directory."""
        try:
            file_path = project_dir / filename
            file_path.write_text(content, encoding='utf-8')
            self.detailed_log_event.emit("HybridEngine", "success",
                                         f"✅ File written: {filename} ({len(content)} chars)", "3")
            self.file_generated.emit(str(file_path))
        except Exception as e:
            self.logger.error(f"Failed to write file {filename}: {e}")
            self.detailed_log_event.emit("HybridEngine", "error",
                                         f"❌ Failed to write {filename}: {str(e)}", "3")

    def _build_dependency_context(self, dependencies: List[str], generated_files: Dict[str, Any]) -> str:
        """Build dependency context for file generation."""
        context_parts = []

        for dep in dependencies:
            if dep in generated_files:
                context_parts.append(f"=== {dep} ===")
                context_parts.append(generated_files[dep].get("source_code", "")[:500] + "...")
                context_parts.append("")

        return "\n".join(context_parts) if context_parts else "No dependencies available."

    async def _finalize_hybrid_project(self, tech_spec: dict, project_dir: Path, start_time: datetime):
        """Finalize the hybrid project creation."""
        try:
            # Create requirements.txt
            self._create_requirements_txt(project_dir, tech_spec)

            # Set up virtual environment
            self._setup_virtual_environment(project_dir)

            # Calculate metrics
            elapsed_time = (datetime.now() - start_time).total_seconds()

            # Create final result
            result = {
                "success": True,
                "project_directory": str(project_dir),
                "project_name": tech_spec.get("project_name", "hybrid_project"),
                "files_created": [f.name for f in project_dir.glob("*.py")],
                "execution_time": elapsed_time,
                "approach": "hybrid_micro_task",
                "micro_tasks_executed": "parallel_processing_enabled"
            }

            self.detailed_log_event.emit("HybridEngine", "success",
                                         f"🎉 Hybrid workflow completed in {elapsed_time:.1f}s", "0")

            self.workflow_completed.emit(result)

        except Exception as e:
            self.logger.error(f"Finalization failed: {e}")
            self.workflow_completed.emit({"success": False, "error": str(e)})

    def _create_requirements_txt(self, project_dir: Path, tech_spec: dict):
        """Create requirements.txt file."""
        try:
            requirements = tech_spec.get("technical_specs", {}).get("requirements", [])
            if requirements:
                req_file = project_dir / "requirements.txt"
                req_file.write_text("\n".join(requirements) + "\n")
                self.detailed_log_event.emit("HybridEngine", "success",
                                             f"✅ Created requirements.txt with {len(requirements)} packages", "2")
        except Exception as e:
            self.logger.error(f"Failed to create requirements.txt: {e}")

    def _setup_virtual_environment(self, project_dir: Path):
        """Set up virtual environment for the project."""
        try:
            import subprocess
            venv_path = project_dir / "venv"
            if not venv_path.exists():
                subprocess.run(["python", "-m", "venv", str(venv_path)], check=True)
                self.detailed_log_event.emit("HybridEngine", "success",
                                             "✅ Virtual environment created", "2")
        except Exception as e:
            self.logger.warning(f"Virtual environment setup failed: {e}")

    def _ensure_gdd_exists(self, project_path: Path, project_name: str, project_description: str):
        """Ensure Game Design Document exists for existing projects."""
        gdd_path = project_path / f"{project_name}_GDD.md"
        if not gdd_path.exists():
            gdd_content = f"""# {project_name} - Game Design Document

## Project Overview
{project_description}

## Core Features
- To be documented based on existing codebase analysis

## Technical Implementation
- Analyzed existing project structure
- Generated technical specification for future modifications

---
*This GDD was auto-generated during project analysis*
"""
            gdd_path.write_text(gdd_content)
            self.detailed_log_event.emit("HybridEngine", "info", "Created GDD for existing project", "2")

    def _read_gdd_context(self, project_path: Path, project_name: str) -> str:
        """Read existing GDD context for project modifications."""
        gdd_path = project_path / f"{project_name}_GDD.md"
        if gdd_path.exists():
            return gdd_path.read_text()
        return "No GDD context available."

    def _create_basic_tech_spec_from_files(self, project_path: Path) -> dict:
        """Create a basic technical specification from existing files."""
        py_files = list(project_path.glob("*.py"))

        return {
            "project_name": project_path.name,
            "project_description": f"Existing project with {len(py_files)} Python files",
            "technical_specs": {
                "files": {
                    f.stem: {
                        "purpose": f"Existing file: {f.name}",
                        "skip_generation": True
                    } for f in py_files
                }
            }
        }

    # Backward compatibility methods
    async def execute_analysis_workflow(self, project_path_str: str):
        """Backward compatibility method for existing application code."""
        await self.analyze_existing_project(project_path_str)


# Backward compatibility alias
EnhancedWorkflowEngine = HybridWorkflowEngine