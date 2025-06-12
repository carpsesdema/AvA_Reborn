# core/project_builder.py - V2 with Robust Sandbox Branching

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
        For modifications, it creates a stable '_dev' directory branch once.

        Args:
            project_name: The name for the new project or modification.
            is_modification: Flag indicating if this is a modification of an existing project.
            original_user_prompt: The initial user request for GDD creation.
            original_project_path: Optional path to an existing project to copy from.

        Returns:
            The path to the newly created and prepared project directory.
        """
        if is_modification:
            if not original_project_path:
                raise ValueError("Original project path must be provided for modifications.")

            # Create a predictable "development branch" directory name
            dev_dir_name = f"{original_project_path.name}_dev"
            project_dir = self.workspace_root / dev_dir_name

            # --- THIS IS THE KEY FIX ---
            # Only copy the original project if the dev directory doesn't already exist.
            # This ensures we create a complete sandbox on the first modification,
            # and then continue to work within that sandbox for subsequent changes.
            if not project_dir.exists():
                self.stream_emitter("ProjectBuilder", "file_op", f"Creating new development branch at '{project_dir}'...", "1")
                try:
                    # Use shutil.copytree to copy the entire project directory structure
                    shutil.copytree(
                        original_project_path,
                        project_dir,
                        dirs_exist_ok=False, # Be safe, we already checked for existence
                        # Ignore common development folders to keep the copy clean
                        ignore=shutil.ignore_patterns('venv', '.venv', '__pycache__', '.git', '.idea', '.vscode', '.pytest_cache', '*.db')
                    )
                    self.stream_emitter("ProjectBuilder", "success", "Branch created and populated successfully.", "1")
                except Exception as e:
                    self.logger.error(f"Failed to copy project to dev branch: {e}", exc_info=True)
                    raise IOError(f"Failed to create development branch: {e}")
            else:
                self.stream_emitter("ProjectBuilder", "info", f"Using existing development branch: '{project_dir}'", "1")

        else:  # This is a new project from scratch
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            dir_name = f"{project_name}_{timestamp}"
            project_dir = self.workspace_root / dir_name
            project_dir.mkdir(parents=True, exist_ok=True)
            self.stream_emitter("ProjectBuilder", "file_op", f"New project directory created: {project_dir}", "1")
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