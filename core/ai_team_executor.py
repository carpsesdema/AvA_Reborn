# core/ai_team_executor.py

import json
import logging
from typing import Dict, List, Optional

from core.workflow_services import CoderService, ReviewerService
from core.project_state_manager import ProjectStateManager


class AITeamExecutor:
    """
    Executes the core AI generation loop (Architect -> Coder -> Reviewer)
    in-memory, without performing any file I/O.
    """

    def __init__(self, coder_service: CoderService, reviewer_service: ReviewerService, stream_emitter,
                 logger: logging.Logger):
        self.coder_service = coder_service
        self.reviewer_service = reviewer_service
        self.stream_emitter = stream_emitter
        self.logger = logger
        self.project_state_manager: Optional[ProjectStateManager] = None

    def set_project_state(self, project_state_manager: ProjectStateManager):
        """Connects the executor to the project's state for context."""
        self.project_state_manager = project_state_manager
        self.coder_service.set_project_state(project_state_manager)
        self.reviewer_service.set_project_state(project_state_manager)

    async def execute_workflow(self, tech_spec: dict) -> dict:
        """
        Runs the full AI team workflow to generate code.

        Args:
            tech_spec: The technical specification for the project.

        Returns:
            A dictionary containing the generated files and their content.
        """
        self.stream_emitter("AITeamExecutor", "stage_start", "AI team is starting the code generation phase.", "1")

        build_order = tech_spec.get("dependency_order", [])
        if not build_order:
            raise ValueError("Architecture failed. The dependency order is empty.")

        self.stream_emitter("AITeamExecutor", "info", f"Determined build order: {', '.join(build_order)}", "2")

        knowledge_packets: Dict[str, Dict] = {}
        generated_files: Dict[str, str] = {}

        for i, filename in enumerate(build_order):
            self.stream_emitter("AITeamExecutor", "stage_start",
                                f"Processing file {i + 1}/{len(build_order)}: {filename}", "2")

            file_spec = tech_spec["technical_specs"].get(filename) or tech_spec["technical_specs"].get(f"{filename}.py")

            if not file_spec:
                self.stream_emitter("AITeamExecutor", "error", f"Could not find spec for '{filename}'. Skipping.", "3")
                continue

            dependency_files = file_spec.get("dependencies", [])
            dependency_context = self._build_dependency_context(dependency_files, knowledge_packets)
            project_context = {"description": tech_spec.get("project_description", "")}
            full_filename = filename if filename.endswith('.py') else f"{filename}.py"

            # Generate code with team communication
            generated_code = await self.coder_service.generate_file_from_spec(
                full_filename, file_spec, project_context, dependency_context
            )

            self.stream_emitter("AITeamExecutor", "info", f"Coder finished {full_filename}. Storing in memory.", "3")

            # Store results in memory
            generated_files[full_filename] = generated_code
            knowledge_packets[full_filename] = {"spec": file_spec, "source_code": generated_code}

        self.stream_emitter("AITeamExecutor", "success", "AI team has completed code generation.", "1")
        return {"generated_files": generated_files, "final_tech_spec": tech_spec}

    def _build_dependency_context(self, dependency_files: List[str], knowledge_packets: Dict[str, Dict]) -> str:
        """
        Builds a lean context string from dependencies, focusing on API contracts.
        """
        if not dependency_files:
            return "This file has no dependencies."

        context_str = "This file must correctly import and use functions/classes from the following dependencies according to their API contracts:\n"
        for dep_file in dependency_files:
            lookup_key = dep_file if dep_file.endswith('.py') else f"{dep_file}.py"

            if lookup_key in knowledge_packets:
                packet = knowledge_packets[lookup_key]
                api_contract = packet.get('spec', {}).get('api_contract', {})
                context_str += f"\n--- DEPENDENCY: {lookup_key} ---\n"
                context_str += f"API CONTRACT:\n```json\n{json.dumps(api_contract, indent=2)}\n```\n"
            else:
                self.logger.warning(
                    f"Dependency '{lookup_key}' not found in knowledge packets. Context will be incomplete.")
                context_str += f"\n--- NOTE: API contract for '{lookup_key}' was not available. Assume standard imports. ---\n"

        return context_str