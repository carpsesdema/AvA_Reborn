# core/project_analyzer.py

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from core.workflow_services import ArchitectService
from core.project_state_manager import ProjectStateManager


class ProjectAnalyzer:
    """
    Analyzes an existing project directory to understand its structure,
    dependencies, and create a technical specification representing its
    current state.
    """

    def __init__(self, architect_service: ArchitectService, stream_emitter, logger):
        self.architect_service = architect_service
        self.stream_emitter = stream_emitter
        self.logger = logger
        self.project_state_manager: ProjectStateManager = None

    async def analyze(self, project_path_str: str) -> Dict[str, Any]:
        """
        Performs a deep analysis of a project and returns its tech spec.
        """
        self.stream_emitter("ProjectAnalyzer", "stage_start", f"Analyzing project at: {project_path_str}", "1")
        try:
            project_path = Path(project_path_str)
            if not project_path.exists():
                raise FileNotFoundError(f"Project path does not exist: {project_path_str}")

            # This service is responsible for creating the state for its analysis scope.
            self.project_state_manager = ProjectStateManager(project_path)
            self.architect_service.set_project_state(self.project_state_manager)

            files_count = len(self.project_state_manager.files)
            self.stream_emitter("ProjectAnalyzer", "info", f"Found {files_count} files to analyze.", "2")

            tech_spec = await self.architect_service.analyze_and_create_spec_from_project(self.project_state_manager)

            if not tech_spec or not isinstance(tech_spec, dict):
                self.stream_emitter("ProjectAnalyzer", "warning", "AI analysis failed. Creating fallback tech spec.", "2")
                tech_spec = self._create_basic_tech_spec_from_files(project_path)

            project_name = tech_spec.get("project_name", project_path.name)
            project_description = tech_spec.get("project_description", "An existing project.")
            self._ensure_gdd_exists(project_path, project_name, project_description)

            self.stream_emitter("ProjectAnalyzer", "success", "Analysis complete. Technical spec created.", "1")
            return tech_spec

        except Exception as e:
            self.logger.error(f"Analysis failed: {e}", exc_info=True)
            self.stream_emitter("ProjectAnalyzer", "error", f"Analysis Error: {e}", "1")
            # Return a minimal spec on failure so the workflow can potentially continue
            return self._create_basic_tech_spec_from_files(Path(project_path_str))

    def _create_basic_tech_spec_from_files(self, project_path: Path) -> dict:
        """Create a basic tech spec from existing files when analysis fails."""
        self.stream_emitter("ProjectAnalyzer", "info", "Creating fallback tech spec from file structure...", "2")
        python_files = list(project_path.rglob("*.py"))
        file_specs = {
            str(f.relative_to(project_path)): {"purpose": f"Existing file: {f.name}"} for f in python_files
        }
        return {
            "project_name": project_path.name,
            "project_description": f"Existing project '{project_path.name}'",
            "technical_specs": file_specs,
            "dependency_order": list(file_specs.keys()),
            "requirements": []
        }

    def _ensure_gdd_exists(self, project_path: Path, project_name_raw: str, initial_prompt: str):
        """Ensures a Game Design Document (GDD) exists. If not, creates one."""
        sanitized_name = "".join(c for c in project_name_raw if c.isalnum() or c in ('_', '-')).rstrip() or "ava_project"
        gdd_file_path = project_path / f"{sanitized_name}_GDD.md"

        if gdd_file_path.exists():
            return

        self.stream_emitter("ProjectAnalyzer", "file_op", f"GDD not found. Creating: {gdd_file_path.name}", "2")
        try:
            gdd_template = f"# Game Design Document: {project_name_raw}\n\n## 1. Project Vision\n> Initial idea: {initial_prompt}\n"
            gdd_file_path.write_text(gdd_template, encoding='utf-8')
        except Exception as e:
            self.logger.error(f"Failed to create GDD file: {e}")