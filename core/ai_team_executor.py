# core/ai_team_executor.py

import asyncio
import json
import logging
from collections import defaultdict, deque
from typing import Dict, List, Optional

from core.workflow_services import CoderService, ReviewerService
from core.project_state_manager import ProjectStateManager


class AITeamExecutor:
    """
    Executes the core AI generation loop (Architect -> Coder -> Reviewer)
    in-memory, without performing any file I/O.
    Now with parallel code generation for maximum speed!
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
        Runs the full AI team workflow to generate code using parallel processing.

        Args:
            tech_spec: The technical specification for the project.

        Returns:
            A dictionary containing the generated files and their content.
        """
        self.stream_emitter("AITeamExecutor", "stage_start", "AI team is starting the code generation phase.", "1")

        # --- NEW: Parallel Execution Logic ---

        # 1. Build Dependency Graph
        dependencies, file_map = self._build_dependency_graph(tech_spec)

        # 2. Determine Generation Stages
        stages = self._determine_generation_stages(dependencies)

        self.stream_emitter("AITeamExecutor", "info", f"Planned {len(stages)} parallel generation stage(s).", "2")

        knowledge_packets: Dict[str, Dict] = {}
        generated_files: Dict[str, str] = {}

        # 3. Process Stages Concurrently
        for i, stage_files in enumerate(stages):
            self.stream_emitter("AITeamExecutor", "stage_start",
                                f"Executing Stage {i + 1}/{len(stages)} with {len(stage_files)} file(s) in parallel...",
                                "2")

            tasks = []
            for filename_key in stage_files:
                # Find the full filename (e.g., with .py) from the original tech spec
                full_filename = file_map.get(filename_key)
                if not full_filename:
                    self.logger.warning(f"Could not find full filename for key '{filename_key}'. Skipping.")
                    continue

                file_spec = tech_spec["technical_specs"].get(full_filename)
                if not file_spec:
                    self.stream_emitter("AITeamExecutor", "error",
                                        f"Could not find spec for '{full_filename}'. Skipping.", "3")
                    continue

                # Create a task for each file in the current stage
                task = self._generate_file_task(full_filename, file_spec, tech_spec, knowledge_packets)
                tasks.append(task)

            # Run all tasks for the current stage concurrently
            if tasks:
                stage_results = await asyncio.gather(*tasks)

                # Process results from the completed stage
                for result in stage_results:
                    if result:
                        fname, code, fspec = result
                        generated_files[fname] = code
                        knowledge_packets[fname] = {"spec": fspec, "source_code": code}

        self.stream_emitter("AITeamExecutor", "success", "AI team has completed code generation.", "1")
        return {"generated_files": generated_files, "final_tech_spec": tech_spec}

    def _build_dependency_graph(self, tech_spec: dict) -> tuple[dict, dict]:
        """Builds a dependency graph from the tech spec."""
        dependencies = defaultdict(list)
        file_map = {}  # Maps key (e.g., 'main') to full filename (e.g., 'main.py')
        all_file_keys = set()

        # First pass: collect all file keys and map them
        for full_filename in tech_spec.get("technical_specs", {}).keys():
            key = full_filename.split('.')[0]  # More robust than replace
            file_map[key] = full_filename
            all_file_keys.add(key)

        # Second pass: build dependencies, ensuring they are valid project files
        for full_filename, spec in tech_spec.get("technical_specs", {}).items():
            key = full_filename.split('.')[0]
            # Ensure the key itself is in the dependencies map
            dependencies[key] = []
            for dep in spec.get("dependencies", []):
                dep_key = dep.split('.')[0]
                # Only add dependency if it's a known file in the project
                if dep_key in all_file_keys:
                    dependencies[key].append(dep_key)

        return dict(dependencies), file_map

    def _determine_generation_stages(self, dependencies: dict) -> List[List[str]]:
        """Determines parallel generation stages using Kahn's algorithm for topological sorting."""
        in_degree = {u: 0 for u in dependencies}
        # Correctly calculate in-degrees
        for u in dependencies:
            for v in dependencies[u]:
                if v in in_degree:  # Ensure dependency is a node in our graph
                    in_degree[v] += 1

        # Queue for all nodes with in-degree 0
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

        # Check for cycles: if count doesn't match total nodes, there's a cycle.
        if count != len(dependencies):
            cycle_nodes = {node for node, deg in in_degree.items() if deg > 0}
            self.logger.error(
                f"A cycle was detected in the dependency graph involving nodes: {cycle_nodes}. Execution order may be incorrect.")
            # Add remaining nodes in a final stage to attempt recovery
            remaining_nodes = list(cycle_nodes)
            if remaining_nodes:
                stages.append(remaining_nodes)

        return stages

    async def _generate_file_task(self, full_filename: str, file_spec: dict, tech_spec: dict, knowledge_packets: dict):
        """Creates an awaitable task for generating a single file."""
        self.stream_emitter("AITeamExecutor", "info", f"Preparing task for {full_filename}", "3")

        dependency_files = file_spec.get("dependencies", [])
        dependency_context = self._build_dependency_context(dependency_files, knowledge_packets)
        project_context = {"description": tech_spec.get("project_description", "")}

        # Generate code with team communication
        generated_code = await self.coder_service.generate_file_from_spec(
            full_filename, file_spec, project_context, dependency_context
        )

        self.stream_emitter("AITeamExecutor", "info", f"Task for {full_filename} complete. Content generated.", "3")
        return full_filename, generated_code, file_spec

    def _build_dependency_context(self, dependency_files: List[str], knowledge_packets: Dict[str, Dict]) -> str:
        """
        Builds a lean context string from dependencies, focusing on API contracts.
        """
        if not dependency_files:
            return "This file has no dependencies."

        context_str = "This file must correctly import and use functions/classes from the following dependencies according to their API contracts:\n"
        for dep_file in dependency_files:
            # Look for the .py version first, then the base name
            lookup_key_py = f"{dep_file}.py" if not dep_file.endswith('.py') else dep_file

            if lookup_key_py in knowledge_packets:
                packet = knowledge_packets[lookup_key_py]
                api_contract = packet.get('spec', {}).get('api_contract', {})
                context_str += f"\n--- DEPENDENCY: {lookup_key_py} ---\n"
                context_str += f"API CONTRACT:\n```json\n{json.dumps(api_contract, indent=2)}\n```\n"
            else:
                self.logger.warning(
                    f"Dependency '{dep_file}' not found in knowledge packets. Context will be incomplete.")
                context_str += f"\n--- NOTE: API contract for '{dep_file}' was not available. Assume standard imports. ---\n"

        return context_str