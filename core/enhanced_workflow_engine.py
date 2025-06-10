# core/enhanced_workflow_engine.py - V5.5 with Team Communication + ROBUST FIXES

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
    🚀 V5.5 Workflow Engine: Now with AI team communication and collaborative learning.
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

        # Initialize services without project state initially
        self.architect_service = ArchitectService(self.llm_client, service_log_emitter, self.rag_manager)
        self.coder_service = CoderService(self.llm_client, service_log_emitter, self.rag_manager)
        self.reviewer_service = ReviewerService(self.llm_client, service_log_emitter, self.rag_manager)

        self._connect_terminal_signals()
        self.logger.info("✅ V5.5 'Team Communication' Workflow Engine initialized.")

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

    def _setup_project_state_and_services(self, project_path: Path):
        """Initialize or update project state manager and connect it to AI services"""
        try:
            # Create or update project state manager
            if not self.project_state_manager or str(self.project_state_manager.project_root) != str(project_path):
                self.detailed_log_event.emit("WorkflowEngine", "info",
                                             f"Setting up project state for: {project_path.name}", "1")
                self.project_state_manager = ProjectStateManager(project_path)

                # Connect project state to all AI services for team communication
                self.architect_service.set_project_state(self.project_state_manager)
                self.coder_service.set_project_state(self.project_state_manager)
                self.reviewer_service.set_project_state(self.project_state_manager)

                self.detailed_log_event.emit("WorkflowEngine", "success",
                                             "Team communication system activated", "1")

                # Log team insight stats if available
                insights_count = len(self.project_state_manager.team_insights)
                if insights_count > 0:
                    self.detailed_log_event.emit("WorkflowEngine", "info",
                                                 f"Loaded {insights_count} team insights from previous work", "1")

        except Exception as e:
            self.logger.error(f"Failed to setup project state: {e}")
            self.detailed_log_event.emit("WorkflowEngine", "error",
                                         f"Project state setup failed: {e}", "1")

    ### MODIFIED ###
    def _ensure_gdd_exists(self, project_path: Path, project_name_raw: str, initial_prompt: str):
        """
        Ensures a Game Design Document (GDD) exists. If not, creates one.
        This is now the single point of truth for GDD creation.
        """
        # Sanitize project name for filename
        project_name_sanitized = "".join(c for c in project_name_raw.strip() if c.isalnum() or c in ('_', '-')).rstrip()
        if not project_name_sanitized:
            project_name_sanitized = "ava_project"

        gdd_file_path = project_path / f"{project_name_sanitized}_GDD.md"

        # Only create if it doesn't exist. This prevents overwriting user modifications.
        if gdd_file_path.exists():
            self.logger.info(f"GDD file already exists at {gdd_file_path}. Skipping creation.")
            return

        self.detailed_log_event.emit("WorkflowEngine", "file_op", f"GDD not found. Creating: {gdd_file_path.name}", "1")
        try:
            gdd_template = f"""# Game Design Document: {project_name_raw}

## 1. Project Vision
> Initial idea: {initial_prompt}

## 2. Core Gameplay Loop
_(To be defined as project evolves)_

## 3. Key Features
_(To be defined as project evolves)_

## 4. Implemented Systems
_(This section will be populated as you build out the project.)_

---

## Development Log
- **{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**: Project initialized or loaded into AvA.
"""
            gdd_file_path.write_text(gdd_template.strip(), encoding='utf-8')
            self.detailed_log_event.emit("WorkflowEngine", "success", f"GDD created successfully: {gdd_file_path.name}",
                                         "1")
        except Exception as e:
            self.logger.error(f"Failed to create GDD file: {e}")
            self.detailed_log_event.emit("WorkflowEngine", "error", f"Failed to create GDD: {e}", "1")

    ### MODIFIED ###
    def _update_gdd_log(self, project_path: Path, project_name: str, user_prompt: str, results: Dict[str, Any]):
        """Updates the project's GDD with the latest workflow results."""
        gdd_file_path = project_path / f"{project_name}_GDD.md"

        # GDD should exist by now, but we add a fallback just in case.
        if not gdd_file_path.exists():
            self.logger.error(f"Could not find GDD file to update at {gdd_file_path}. Recreating it.")
            vision = f"Project '{project_name}' was loaded. The original vision was not recorded."
            self._ensure_gdd_exists(project_path, project_name, vision)

        self.detailed_log_event.emit("WorkflowEngine", "file_op", f"Updating GDD log: {gdd_file_path.name}", "1")
        files_created = ", ".join(results.get("files_created", []))
        log_entry = f"""
---
- **{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**: Workflow Run
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

    def _create_basic_tech_spec_from_files(self, project_path: Path) -> dict:
        """Create a basic tech spec from existing files when analysis fails."""
        self.detailed_log_event.emit("WorkflowEngine", "info", "Creating fallback tech spec from existing files...",
                                     "1")

        python_files = list(project_path.rglob("*.py"))
        python_files = [f for f in python_files if '__pycache__' not in f.parts and '.venv' not in f.parts]

        file_specs = {}
        for py_file in python_files:
            rel_path = py_file.relative_to(project_path)
            file_name = str(rel_path).replace('\\', '/').replace('.py', '')
            file_specs[file_name] = {
                "purpose": f"Existing file: {rel_path}",
                "dependencies": [],
                "api_contract": {"description": f"Existing implementation in {rel_path}"}
            }

        return {
            "project_name": project_path.name,
            "project_description": f"Existing project '{project_path.name}' ready for modifications",
            "technical_specs": file_specs,
            "dependency_order": list(file_specs.keys()),
            "requirements": []  # Will be populated later if needed
        }

    def _create_requirements_txt(self, project_dir: Path, tech_spec: dict):
        """Create requirements.txt file from tech spec."""
        requirements = tech_spec.get("requirements", [])
        if requirements:
            req_file = project_dir / "requirements.txt"
            self.detailed_log_event.emit("WorkflowEngine", "file_op",
                                         f"Creating requirements.txt with {len(requirements)} packages", "1")
            try:
                req_content = "\n".join(requirements) + "\n"
                req_file.write_text(req_content, encoding='utf-8')
                self.detailed_log_event.emit("WorkflowEngine", "success", "requirements.txt created successfully", "1")
            except Exception as e:
                self.logger.error(f"Failed to create requirements.txt: {e}")
                self.detailed_log_event.emit("WorkflowEngine", "error", f"Failed to create requirements.txt: {e}", "1")

    def _setup_virtual_environment(self, project_dir: Path):
        """Set up virtual environment for the project."""
        import sys
        import subprocess

        venv_path = project_dir / "venv"
        if venv_path.exists():
            self.detailed_log_event.emit("WorkflowEngine", "info", "Virtual environment already exists", "1")
            return

        self.detailed_log_event.emit("WorkflowEngine", "info", "Creating virtual environment...", "1")
        try:
            # Create venv
            cmd = [sys.executable, '-m', 'venv', venv_path.name]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=str(project_dir))

            if result.returncode == 0:
                self.detailed_log_event.emit("WorkflowEngine", "success", "Virtual environment created successfully",
                                             "1")

                # Install requirements if they exist
                req_file = project_dir / "requirements.txt"
                if req_file.exists():
                    self.detailed_log_event.emit("WorkflowEngine", "info", "Installing requirements...", "1")

                    # Determine pip path
                    if sys.platform == "win32":
                        pip_path = venv_path / "Scripts" / "pip.exe"
                    else:
                        pip_path = venv_path / "bin" / "pip"

                    pip_result = subprocess.run([str(pip_path), 'install', '-r', str(req_file)],
                                                capture_output=True, text=True, cwd=str(project_dir))

                    if pip_result.returncode == 0:
                        self.detailed_log_event.emit("WorkflowEngine", "success", "Requirements installed successfully",
                                                     "1")
                    else:
                        self.detailed_log_event.emit("WorkflowEngine", "warning",
                                                     f"Requirements installation had issues: {pip_result.stderr[:200]}",
                                                     "1")
            else:
                self.detailed_log_event.emit("WorkflowEngine", "error",
                                             f"Failed to create venv: {result.stderr or result.stdout}",
                                             "1")

        except Exception as e:
            self.logger.error(f"Failed to create virtual environment: {e}")
            self.detailed_log_event.emit("WorkflowEngine", "error", f"Failed to create venv: {e}", "1")

    async def execute_analysis_workflow(self, project_path_str: str):
        self.logger.info(f"🚀 Starting Analysis workflow for: {project_path_str}...")
        self.analysis_started.emit(project_path_str)
        self.detailed_log_event.emit("WorkflowEngine", "stage_start",
                                     f"🚀 Initializing Analysis for '{Path(project_path_str).name}'...", "0")

        try:
            project_path = Path(project_path_str)

            # Validate project path exists
            if not project_path.exists():
                raise Exception(f"Project path does not exist: {project_path_str}")

            # Setup project state and team communication
            self._setup_project_state_and_services(project_path)

            # Check if project state manager was created successfully
            if not self.project_state_manager:
                raise Exception("Failed to initialize project state manager")

            files_count = len(self.project_state_manager.files)
            self.detailed_log_event.emit("WorkflowEngine", "info",
                                         f"Project State Manager initialized. Scanned {files_count} files.",
                                         "1")

            # If no files were found, that's important to know but not an error
            if files_count == 0:
                self.detailed_log_event.emit("WorkflowEngine", "warning",
                                             "No Python files found in project. Analysis will be limited.", "1")

            # Try to analyze the project - with better error handling and debugging
            self.detailed_log_event.emit("WorkflowEngine", "info", "Starting AI analysis of project structure...", "1")

            # Multiple attempts with different strategies
            tech_spec = None
            for attempt in range(3):
                try:
                    self.detailed_log_event.emit("WorkflowEngine", "info", f"Analysis attempt {attempt + 1}/3...", "1")
                    tech_spec = await self.architect_service.analyze_and_create_spec_from_project(
                        self.project_state_manager)

                    # Validate the tech spec thoroughly
                    if tech_spec and isinstance(tech_spec, dict) and tech_spec.get("technical_specs"):
                        self.detailed_log_event.emit("WorkflowEngine", "success", "✅ AI analysis successful!", "1")
                        break
                    else:
                        self.detailed_log_event.emit("WorkflowEngine", "warning",
                                                     f"Attempt {attempt + 1} returned incomplete analysis. Retrying...",
                                                     "1")
                        tech_spec = None

                except Exception as analysis_error:
                    self.logger.error(f"Analysis attempt {attempt + 1} failed: {analysis_error}", exc_info=True)
                    self.detailed_log_event.emit("WorkflowEngine", "warning",
                                                 f"Attempt {attempt + 1} failed: {analysis_error}", "1")
                    # Wait a bit between retries
                    if attempt < 2:
                        await asyncio.sleep(1)

            # If AI analysis failed, create fallback tech spec
            if not tech_spec or not isinstance(tech_spec, dict):
                self.detailed_log_event.emit("WorkflowEngine", "warning",
                                             "AI analysis failed. Creating fallback tech spec...", "1")
                tech_spec = self._create_basic_tech_spec_from_files(project_path)

            # Ensure we have a valid tech spec
            if not tech_spec:
                raise Exception("Failed to create any technical specification for the project")

            self.current_tech_spec = tech_spec
            self.is_existing_project_loaded = True
            self.original_project_path = project_path

            project_name = tech_spec.get("project_name", project_path.name)
            project_description = tech_spec.get("project_description", "An existing project loaded into AvA.")

            ### MODIFIED ###
            # Ensure GDD exists after loading an existing project
            self._ensure_gdd_exists(project_path, project_name, project_description)

            self.detailed_log_event.emit("WorkflowEngine", "success", "✅ Analysis complete! Technical spec created.",
                                         "0")

            self.analysis_completed.emit(project_path_str, self.current_tech_spec)
            self.project_loaded.emit(project_path_str)

        except Exception as e:
            self.logger.error(f"❌ Analysis Workflow failed: {e}", exc_info=True)
            self.detailed_log_event.emit("WorkflowEngine", "error", f"❌ Analysis Error: {str(e)}", "0")

            # CREATE FALLBACK TECH SPEC so modifications can still proceed
            try:
                project_path = Path(project_path_str)
                self.current_tech_spec = self._create_basic_tech_spec_from_files(project_path)
                self.is_existing_project_loaded = True
                self.original_project_path = project_path
                self.detailed_log_event.emit("WorkflowEngine", "info",
                                             "Using fallback tech spec for future modifications", "0")
            except Exception as fallback_error:
                self.logger.error(f"Failed to create fallback tech spec: {fallback_error}")

            # Still emit project_loaded signal so the UI doesn't hang
            self.project_loaded.emit(project_path_str)

    async def execute_enhanced_workflow(self, user_prompt: str, conversation_context: List[Dict] = None):
        self.logger.info(f"🚀 Starting V5.5 workflow: {user_prompt[:100]}...")
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
                                    ignore=shutil.ignore_patterns('venv', '__pycache__', '.git'))
                except Exception as copy_error:
                    raise Exception(f"Failed to create a copy of the existing project: {copy_error}")
                self.detailed_log_event.emit("WorkflowEngine", "success", "Project copy complete.", "1")
            else:
                project_dir_name = f"{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                project_dir = Path("./workspace") / project_dir_name
                project_dir.mkdir(parents=True, exist_ok=True)
                self.detailed_log_event.emit("WorkflowEngine", "file_op",
                                             f"New project directory created: {project_dir}", "1")

                ### MODIFIED ###
                # Ensure GDD exists for a new project
                self._ensure_gdd_exists(project_dir, project_name, user_prompt)

            # Setup team communication for the target project directory
            self._setup_project_state_and_services(project_dir)

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

                # Generate code with team communication
                generated_code = await self.coder_service.generate_file_from_spec(full_filename, file_spec,
                                                                                  project_context,
                                                                                  dependency_context)

                # --- REMOVED REVIEWER CALL ---
                # The Coder's output is now considered final for new files.
                self.detailed_log_event.emit("WorkflowEngine", "info",
                                             f"Coder finished {full_filename}. Writing file to disk.",
                                             "2")

                # Write file and record it in project state
                self._write_file(project_dir, full_filename, generated_code)

                # Add the file to project state for team communication
                if self.project_state_manager:
                    self.project_state_manager.add_file(str(project_dir / full_filename), generated_code,
                                                        "workflow_engine", f"Generated {full_filename}")

                knowledge_packets[full_filename] = {"spec": file_spec, "source_code": generated_code}
                results["files_created"].append(full_filename)

            # Create requirements.txt from tech spec
            self._create_requirements_txt(project_dir, tech_spec)

            # Set up virtual environment
            self._setup_virtual_environment(project_dir)

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
        """
        Builds a lean context string from dependencies, focusing on API contracts
        instead of full source code to keep prompts small and focused.
        """
        if not dependency_files:
            return "This file has no dependencies."

        context_str = "This file must correctly import and use functions/classes from the following dependencies according to their API contracts:\n"
        for dep_file in dependency_files:
            lookup_key = dep_file if dep_file.endswith('.py') else f"{dep_file}.py"

            if lookup_key in knowledge_packets:
                packet = knowledge_packets[lookup_key]
                # --- THIS IS THE KEY CHANGE ---
                # We ONLY provide the API contract, not the full source code.
                api_contract = packet.get('spec', {}).get('api_contract', {})
                context_str += f"\n--- DEPENDENCY: {lookup_key} ---\n"
                context_str += f"API CONTRACT:\n```json\n{json.dumps(api_contract, indent=2)}\n```\n"
                # --- FULL SOURCE CODE IS REMOVED ---
            else:
                self.logger.warning(f"Dependency '{lookup_key}' not found in knowledge packets. Context will be incomplete.")
                context_str += f"\n--- NOTE: API contract for '{lookup_key}' was not available. Assume standard imports. ---\n"

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
            "strategy": "V5.5 Team Communication"
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
                ### MODIFIED ###
                # The project name passed here should be the sanitized one for the filename
                sanitized_base_name = "".join(
                    c for c in base_project_name.strip() if c.isalnum() or c in ('_', '-')).rstrip()
                self._update_gdd_log(project_path, sanitized_base_name, self.original_user_prompt, results)

        # Log team insights summary
        if self.project_state_manager:
            insights_count = len(self.project_state_manager.team_insights)
            self.detailed_log_event.emit("WorkflowEngine", "info",
                                         f"Project completed with {insights_count} team insights stored", "1")

            # Save the project state to persist team insights
            try:
                self.project_state_manager.save_state()
                self.detailed_log_event.emit("WorkflowEngine", "success",
                                             "Team insights saved for future iterations", "1")
            except Exception as e:
                self.logger.error(f"Failed to save team insights: {e}")

        self.detailed_log_event.emit("WorkflowEngine", "success", "Project finalization complete.", "1")
        return final_result