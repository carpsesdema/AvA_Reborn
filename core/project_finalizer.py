# core/project_finalizer.py

import logging
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from core.project_state_manager import ProjectStateManager


class ProjectFinalizer:
    """
    Takes the in-memory results from the AITeamExecutor and writes them
    to disk, creating the final project artifacts like requirements.txt and venv.
    """

    def __init__(self, stream_emitter, logger: logging.Logger):
        self.stream_emitter = stream_emitter
        self.logger = logger
        self.logger.info("ProjectFinalizer initialized.")

    def finalize_project(self, project_dir: Path, generated_files: Dict[str, str], tech_spec: dict,
                         project_state: ProjectStateManager, original_user_prompt: str):
        """
        Writes all generated files and artifacts to the project directory.

        Args:
            project_dir: The target directory to write to.
            generated_files: A dictionary of filenames to content.
            tech_spec: The technical specification used for generation.
            project_state: The state manager for the current project.
            original_user_prompt: The initial user request for logging.
        """
        self.stream_emitter("ProjectFinalizer", "stage_start", f"Finalizing project at: {project_dir}", "1")

        # 1. Write generated files
        for filename, content in generated_files.items():
            self._write_file(project_dir, filename, content)
            project_state.add_file(str(project_dir / filename), content, "finalizer",
                                   f"Wrote generated file {filename}")

        # 2. Create requirements.txt
        self._create_requirements_txt(project_dir, tech_spec)

        # 3. Set up virtual environment
        self._setup_virtual_environment(project_dir)

        # 4. Update GDD log
        results = {"files_created": list(generated_files.keys())}
        self._update_gdd_log(project_dir, tech_spec.get("project_name", "ai_project"), original_user_prompt, results)

        # 5. Save the final state
        try:
            project_state.save_state()
            self.stream_emitter("ProjectFinalizer", "success", "Team insights saved for future iterations", "2")
        except Exception as e:
            self.logger.error(f"Failed to save project state: {e}", exc_info=True)

        self.stream_emitter("ProjectFinalizer", "success", "Project finalization complete.", "1")

    def _write_file(self, project_dir: Path, filename: str, content: str):
        file_path_obj = project_dir / filename
        file_path_obj.parent.mkdir(parents=True, exist_ok=True)
        file_path_obj.write_text(content, encoding='utf-8')
        self.stream_emitter("ProjectFinalizer", "file_op", f"File written: {file_path_obj}", "2")

    def _create_requirements_txt(self, project_dir: Path, tech_spec: dict):
        """Create requirements.txt file from tech spec."""
        requirements = tech_spec.get("requirements", [])
        if requirements:
            req_file = project_dir / "requirements.txt"
            self.stream_emitter("ProjectFinalizer", "file_op",
                                f"Creating requirements.txt with {len(requirements)} packages", "2")
            try:
                req_content = "\n".join(requirements) + "\n"
                req_file.write_text(req_content, encoding='utf-8')
            except Exception as e:
                self.logger.error(f"Failed to create requirements.txt: {e}", exc_info=True)

    def _setup_virtual_environment(self, project_dir: Path):
        """Set up virtual environment for the project."""
        venv_path = project_dir / "venv"
        if venv_path.exists():
            self.stream_emitter("ProjectFinalizer", "info", "Virtual environment already exists", "2")
            return

        self.stream_emitter("ProjectFinalizer", "info", "Creating virtual environment...", "2")
        try:
            cmd = [sys.executable, '-m', 'venv', venv_path.name]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=str(project_dir))

            if result.returncode == 0:
                self.stream_emitter("ProjectFinalizer", "success", "Virtual environment created", "3")
                req_file = project_dir / "requirements.txt"
                if req_file.exists():
                    self._install_requirements(venv_path, req_file)
            else:
                self.stream_emitter("ProjectFinalizer", "error",
                                    f"Failed to create venv: {result.stderr or result.stdout}", "3")

        except Exception as e:
            self.logger.error(f"Failed to create virtual environment: {e}", exc_info=True)

    def _install_requirements(self, venv_path: Path, req_file: Path):
        self.stream_emitter("ProjectFinalizer", "info", "Installing requirements...", "3")
        pip_path = venv_path / "Scripts" / "pip.exe" if sys.platform == "win32" else venv_path / "bin" / "pip"

        pip_result = subprocess.run([str(pip_path), 'install', '-r', str(req_file)], capture_output=True, text=True,
                                    cwd=str(req_file.parent))

        if pip_result.returncode == 0:
            self.stream_emitter("ProjectFinalizer", "success", "Requirements installed successfully", "3")
        else:
            self.stream_emitter("ProjectFinalizer", "warning",
                                f"Requirements installation had issues: {pip_result.stderr[:200]}", "3")

    def _update_gdd_log(self, project_path: Path, project_name: str, user_prompt: str, results: Dict[str, Any]):
        """Updates the project's GDD with the latest workflow results."""
        sanitized_name = "".join(c for c in project_name if c.isalnum() or c in ('_', '-')).rstrip() or "ava_project"
        gdd_file_path = project_path / f"{sanitized_name}_GDD.md"

        if not gdd_file_path.exists():
            self.logger.error(f"Could not find GDD file to update at {gdd_file_path}.")
            return

        self.stream_emitter("ProjectFinalizer", "file_op", f"Updating GDD log: {gdd_file_path.name}", "2")
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
            self.logger.error(f"❌ Failed to update GDD log: {e}", exc_info=True)