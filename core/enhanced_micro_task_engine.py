# core/enhanced_micro_task_engine.py - Streamlined for RESULTS

import json
import re
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from core.llm_client import LLMRole


@dataclass
class SimpleTaskSpec:
    """Streamlined task specification - focused on results"""
    id: str
    description: str
    expected_lines: int  # We'll use a default for now or make it optional
    context: str
    exact_requirements: str
    # ADDED: To carry over the component type specified by the Planner
    component_type: Optional[str] = "unknown"
    file_path: Optional[str] = None  # To associate task with its file

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "expected_lines": self.expected_lines,
            "context": self.context,
            "exact_requirements": self.exact_requirements,
            "component_type": self.component_type,
            "file_path": self.file_path,
            "type": "code_chunk"  # Preserving original 'type' field for compatibility downstream
        }


class StreamlinedMicroTaskEngine:
    """
    🚀 Streamlined Micro-Task Engine - RESULTS FOCUSED
    Dynamically creates micro-tasks based on detailed Planner output.
    """

    def __init__(self, llm_client, project_state_manager, domain_context_manager):
        self.llm_client = llm_client
        self.project_state = project_state_manager
        self.domain_context = domain_context_manager

    async def create_smart_tasks(self,
                                 planner_json_plan: Dict[str, Any],
                                 project_overall_context: Dict[str, Any]  # e.g., project type, RAG context
                                 ) -> List[SimpleTaskSpec]:
        """
        Dynamically create SimpleTaskSpec objects based on the Planner's JSON output.
        The Planner's JSON is expected to have a detailed breakdown of components
        for each file, following the structure defined in the Planner's personality prompt.
        """
        all_micro_tasks: List[SimpleTaskSpec] = []
        project_name = planner_json_plan.get("project_name", "default_project")

        # The planner_json_plan structure is assumed to be:
        # {
        #   "project_name": "...",
        #   "files": {
        #     "file1.py": {
        #       "purpose": "...",
        #       "components": [
        #         { "task_id": "func1", "description": "...", "component_type": "function",
        #           "inputs": [...], "outputs": "...", "core_logic_steps": [...],
        #           "error_conditions_to_handle": [...], "interactions": [...], "critical_notes": "..." },
        #         // ... more components
        #       ]
        #     },
        #     "file2.py": { ... }
        #   }
        # }

        files_to_process = planner_json_plan.get("files", {})
        if not files_to_process:
            print("[ERROR] Planner provided no files in the plan. Using fallback task generation.")
            # Create a single fallback task if the planner's output is not as expected.
            # This needs a defined 'file_spec' which isn't directly available here.
            # For a true fallback, this engine might need more context or the calling workflow
            # should handle the empty plan from the Planner.
            # For now, let's assume the workflow handles an empty task list.
            return self._create_fallback_tasks_for_project(planner_json_plan)

        for file_path_str, file_spec in files_to_process.items():
            file_purpose = file_spec.get("purpose", f"Functionality for {file_path_str}")
            components = file_spec.get("components", [])

            if not components:
                # If Planner didn't break down components for a file, create one umbrella task.
                print(f"[INFO] No components detailed by Planner for {file_path_str}. Creating umbrella task.")
                umbrella_task = self._create_umbrella_task_for_file(
                    file_path_str,
                    file_purpose,
                    project_name,
                    file_spec.get("description", file_purpose)  # Use file_spec.description if available
                )
                all_micro_tasks.append(umbrella_task)
                continue

            for i, component_spec in enumerate(components):
                task_id = component_spec.get("task_id", f"comp_{i}")
                full_task_id = f"{Path(file_path_str).stem}_{task_id}"
                description = component_spec.get("description", "Implement component.")
                component_type = component_spec.get("component_type", "unknown")

                # Estimate lines or use a default. Planner ideally provides this.
                expected_lines = component_spec.get("expected_lines",
                                                    20 + len(component_spec.get("core_logic_steps", [])) * 3)

                context_str = f"File: {file_path_str} ({file_purpose}) within Project: {project_name}. Component: {description}"

                # Construct detailed exact_requirements from Planner's component spec
                req_parts = [f"COMPONENT TYPE: {component_type}"]

                if "inputs" in component_spec:
                    inputs_str = "\n  ".join(component_spec["inputs"]) if isinstance(component_spec["inputs"],
                                                                                     list) else str(
                        component_spec["inputs"])
                    req_parts.append(f"INPUTS:\n  {inputs_str}")

                if "outputs" in component_spec:
                    req_parts.append(f"OUTPUTS/SIDE EFFECTS:\n  {component_spec['outputs']}")

                if "core_logic_steps" in component_spec and component_spec["core_logic_steps"]:
                    steps_str = "\n  - ".join(component_spec["core_logic_steps"])
                    req_parts.append(f"CORE LOGIC STEPS:\n  - {steps_str}")

                if "error_conditions_to_handle" in component_spec and component_spec["error_conditions_to_handle"]:
                    errors_str = "\n  - ".join(component_spec["error_conditions_to_handle"])
                    req_parts.append(f"ERROR CONDITIONS TO HANDLE:\n  - {errors_str}")

                if "interactions" in component_spec and component_spec["interactions"]:
                    interactions_str = "\n  - ".join(component_spec["interactions"])
                    req_parts.append(f"INTERACTIONS WITH OTHER COMPONENTS:\n  - {interactions_str}")

                if "critical_notes" in component_spec and component_spec["critical_notes"]:
                    req_parts.append(f"CRITICAL NOTES:\n  {component_spec['critical_notes']}")

                exact_requirements_str = "\n\n".join(req_parts)

                micro_task = SimpleTaskSpec(
                    id=full_task_id,
                    description=description,
                    expected_lines=expected_lines,
                    context=context_str,
                    exact_requirements=exact_requirements_str,
                    component_type=component_type,
                    file_path=file_path_str
                )
                all_micro_tasks.append(micro_task)

        if not all_micro_tasks and files_to_process:  # If we iterated files but still no tasks (e.g. all files had no components)
            print(
                "[WARNING] No micro-tasks generated despite files being present in plan. This might indicate an issue with Planner's output format.")
            # Potentially add a single project-wide fallback task here if truly nothing was generated.
            # For now, an empty list will be returned, and the workflow engine needs to handle it.

        return all_micro_tasks

    def _create_umbrella_task_for_file(self, file_path_str: str, file_purpose: str, project_name: str,
                                       file_description: str) -> SimpleTaskSpec:
        """Creates a single task to implement an entire file if Planner provides no component breakdown."""
        task_id = f"{Path(file_path_str).stem}_implement_all"
        description = f"Fully implement {file_path_str} as per its purpose: {file_purpose}. Details: {file_description}"
        context_str = f"File: {file_path_str} ({file_purpose}) within Project: {project_name}."

        # General requirements for a full file implementation
        exact_requirements_str = f"""
        GOAL: Implement the entire functionality for the file '{file_path_str}'.
        PURPOSE: {file_purpose}
        DESCRIPTION FROM PLANNER: {file_description}

        EXPECTATIONS:
        - Write complete, professional, and production-ready Python code.
        - Include all necessary imports at the top of the file.
        - Define all required classes, functions, and variables.
        - Implement comprehensive error handling (try-except blocks, specific exceptions where appropriate).
        - Add clear docstrings for all public classes, functions, and methods.
        - Include comments for any non-obvious logic.
        - Ensure the code is modular and follows Python best practices (PEP 8).
        - If this is a main executable file, include an `if __name__ == \"__main__\":` block.
        - If this file interacts with other planned files, ensure interfaces are consistent (even if stubs for now).
        - CRITICAL: Adhere to any overarching architectural patterns, security guidelines (e.g., avoiding 'eval' on user input if applicable), or coding standards defined by the Planner for the project.
        """
        return SimpleTaskSpec(
            id=task_id,
            description=description,
            expected_lines=100,  # Default for a full file
            context=context_str,
            exact_requirements=exact_requirements_str.strip(),
            component_type="full_file",
            file_path=file_path_str
        )

    def _create_fallback_tasks_for_project(self, planner_json_plan: Dict[str, Any]) -> List[SimpleTaskSpec]:
        """Fallback if Planner's plan is unusable or completely empty."""
        project_name = planner_json_plan.get("project_name", "unspecified_project")
        project_description = planner_json_plan.get("description", "An AI-generated project.")

        print(f"[FALLBACK] Creating a single task for the entire project: {project_name}")

        # Try to find at least one file name if provided, otherwise default to main.py
        main_file_name = "main.py"
        files_data = planner_json_plan.get("files", {})
        if files_data and isinstance(files_data, dict):
            main_file_name = next(iter(files_data.keys()), "main.py")

        return [
            SimpleTaskSpec(
                id=f"{project_name}_full_project_implementation",
                description=f"Implement the complete project: {project_name}. Description: {project_description}",
                expected_lines=200,  # A wild guess for a whole project
                context=f"Project: {project_name}. Task: Implement all described functionality.",
                exact_requirements=f"""
                PROJECT GOAL: Create a fully functional application for '{project_name}'.
                PROJECT DESCRIPTION: {project_description}

                OVERALL REQUIREMENTS:
                - Implement all features as implied by the project description.
                - Create necessary file structures if not already defined (though aim for a single file if complexity allows for this fallback).
                - Ensure the application is runnable and produces meaningful output or interaction.
                - Write clean, robust, and well-documented Python code.
                - Adhere to Python best practices (PEP 8).
                - Implement error handling.
                - If it's a GUI application, ensure it's user-friendly. If CLI, ensure clear usage.
                - CRITICAL: Avoid unsafe practices like 'eval()' on raw user input.
                """,
                component_type="full_project_fallback",
                file_path=main_file_name  # Assign to a primary file
            )
        ]

    # Removed _detect_domain_fast, _create_gui_tasks, _create_api_tasks,
    # _create_cli_tasks, _create_generic_tasks as they are replaced by dynamic generation.
    # The _create_fallback_tasks (original one taking file_path, file_spec) is also replaced
    # by _create_umbrella_task_for_file and _create_fallback_tasks_for_project.