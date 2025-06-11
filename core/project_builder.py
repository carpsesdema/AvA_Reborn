# core/project_builder.py

import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional


class ProjectBuilder:
    """
    Handles all file system operations for setting up a new or modified
    project directory before AI code generation begins.
    """

    def __init__(self, workspace_root: str, stream_emitter, logger: logging.Logger):
        self.workspace_root = Path(workspace_root)
        self.stream_emitter = stream_emitter
        self.logger = logger
        self.workspace_root.mkdir(exist_ok=True, parents=True)
        self.logger.info("ProjectBuilder initialized.")

    def setup_project_directory(self, project_name: str, is_modification: bool, original_user_prompt: str,
                                original_project_path: Optional[Path] = None) -> Path:
        """
        Creates the project directory, copying existing files if necessary.

        Args:
            project_name: The name for the new project or modification.
            is_modification: Flag indicating if this is a modification of an existing project.
            original_user_prompt: The initial user request for GDD creation.
            original_project_path: Optional path to an existing project to copy from.

        Returns:
            The path to the newly created and prepared project directory.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if is_modification:
            if not original_project_path:
                raise ValueError("Original project path must be provided for modifications.")
            dir_name = f"{original_project_path.name}_MOD_{timestamp}"
            project_dir = self.workspace_root / dir_name

            self.stream_emitter("ProjectBuilder", "file_op", f"Copying original project to '{project_dir}'...", "1")
            try:
                shutil.copytree(original_project_path, project_dir, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns('venv', '__pycache__', '.git'))
                self.stream_emitter("ProjectBuilder", "success", "Project copy complete.", "1")
            except Exception as e:
                self.logger.error(f"Failed to copy project: {e}", exc_info=True)
                raise IOError(f"Failed to create a copy of the existing project: {e}")
        else:
            dir_name = f"{project_name}_{timestamp}"
            project_dir = self.workspace_root / dir_name
            project_dir.mkdir(parents=True, exist_ok=True)
            self.stream_emitter("ProjectBuilder", "file_op", f"New project directory created: {project_dir}", "1")

            # For new projects, ensure the GDD is created right away.
            self._ensure_gdd_exists(project_dir, project_name, original_user_prompt)

        return project_dir

    def _ensure_gdd_exists(self, project_path: Path, project_name_raw: str, initial_prompt: str):
        """Ensures a Game Design Document (GDD) exists. If not, creates one."""
        sanitized_name = "".join(
            c for c in project_name_raw if c.isalnum() or c in ('_', '-')).rstrip() or "ava_project"
        gdd_file_path = project_path / f"{sanitized_name}_GDD.md"

        if gdd_file_path.exists():
            return

        self.stream_emitter("ProjectBuilder", "file_op", f"GDD not found. Creating: {gdd_file_path.name}", "2")
        try:
            gdd_template = f"""# Game Design Document: {project_name_raw}

## 1. Project Vision
> Initial idea: {initial_prompt}

## 2. Core Loop
_(To be defined as project evolves)_

## 3. Key Features
_(To be defined as project evolves)_

## 4. Implemented Systems
_(This section will be populated as you build out the project.)_

---

## Development Log
- **{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**: Project initialized by AvA.
"""
            gdd_file_path.write_text(gdd_template.strip(), encoding='utf-8')
        except Exception as e:
            self.logger.error(f"Failed to create GDD file: {e}", exc_info=True)