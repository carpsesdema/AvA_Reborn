# core/enhanced_micro_task_engine.py - Streamlined for RESULTS (with efficient context)

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
    expected_lines: int
    context: str
    exact_requirements: str
    component_type: Optional[str] = "unknown"
    file_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "expected_lines": self.expected_lines,
            "context": self.context,
            "exact_requirements": self.exact_requirements,
            "component_type": self.component_type,
            "file_path": self.file_path,
            "type": "code_chunk"
        }


class StreamlinedMicroTaskEngine:
    """
    🚀 Streamlined Micro-Task Engine - RESULTS FOCUSED
    Dynamically creates micro-tasks based on detailed Planner output.
    """

    def __init__(self, llm_client, domain_context_manager):
        """
        Initializes the engine.
        Note: The project_state_manager is no longer needed here as domain
        context is now handled at a higher level in the workflow.
        """
        self.llm_client = llm_client
        self.domain_context_manager = domain_context_manager

    async def create_smart_tasks(
            self,
            planner_json_plan: Dict[str, Any],
            project_overall_context: Dict[str, Any]
    ) -> List[SimpleTaskSpec]:
        """
        Dynamically create SimpleTaskSpec objects based on the Planner's JSON output.
        This version assumes domain context has been pre-analyzed and is available
        via the domain_context_manager.
        """
        all_micro_tasks: List[SimpleTaskSpec] = []
        project_name = project_overall_context.get("project_name", "default_project")

        files_to_process = planner_json_plan.get("files", {})
        if not files_to_process:
            return self._create_fallback_tasks_for_project(planner_json_plan)

        for file_path_str, file_spec in files_to_process.items():
            file_purpose = file_spec.get("purpose", f"Functionality for {file_path_str}")
            components = file_spec.get("components", [])

            # Get file-specific context (this should be fast if cached by the manager)
            file_specific_context_str = ""
            if self.domain_context_manager:
                try:
                    file_context = await self.domain_context_manager.get_context_for_file(file_path_str)
                    if file_context:
                        file_specific_context_str = json.dumps(file_context, default=str, indent=2)
                except Exception as e:
                    print(f"[WARNING] Could not get domain context for {file_path_str}: {e}")

            if not components:
                all_micro_tasks.append(self._create_umbrella_task_for_file(
                    file_path_str, file_purpose, project_name, file_spec.get("description", file_purpose)
                ))
                continue

            for i, component_spec in enumerate(components):
                task_id = component_spec.get("task_id", f"comp_{i}")
                full_task_id = f"{Path(file_path_str).stem}_{task_id}"
                description = component_spec.get("description", "Implement component.")
                component_type = component_spec.get("component_type", "unknown")
                expected_lines = component_spec.get("expected_lines",
                                                    20 + len(component_spec.get("core_logic_steps", [])) * 3)

                context_str = (f"File: {file_path_str} ({file_purpose}) within Project: {project_name}. "
                               f"Component: {description}\n\n")
                if file_specific_context_str:
                    context_str += f"--- FILE-SPECIFIC DOMAIN CONTEXT ---\n{file_specific_context_str}"

                req_parts = [f"COMPONENT TYPE: {component_type}"]
                if "inputs" in component_spec:
                    inputs_str = "\n  ".join(component_spec["inputs"]) if isinstance(component_spec["inputs"],
                                                                                     list) else str(
                        component_spec["inputs"])
                    req_parts.append(f"INPUTS:\n  {inputs_str}")
                if "outputs" in component_spec:
                    req_parts.append(f"OUTPUTS/SIDE EFFECTS:\n  {component_spec['outputs']}")
                if component_spec.get("core_logic_steps"):
                    steps_str = "\n  - ".join(component_spec["core_logic_steps"])
                    req_parts.append(f"CORE LOGIC STEPS:\n  - {steps_str}")
                if component_spec.get("error_conditions_to_handle"):
                    errors_str = "\n  - ".join(component_spec["error_conditions_to_handle"])
                    req_parts.append(f"ERROR CONDITIONS TO HANDLE:\n  - {errors_str}")
                if component_spec.get("interactions"):
                    interactions_str = "\n  - ".join(component_spec["interactions"])
                    req_parts.append(f"INTERACTIONS WITH OTHER COMPONENTS:\n  - {interactions_str}")
                if component_spec.get("critical_notes"):
                    req_parts.append(f"CRITICAL NOTES:\n  {component_spec['critical_notes']}")

                exact_requirements_str = "\n\n".join(req_parts)

                micro_task = SimpleTaskSpec(
                    id=full_task_id, description=description, expected_lines=expected_lines,
                    context=context_str, exact_requirements=exact_requirements_str,
                    component_type=component_type, file_path=file_path_str
                )
                all_micro_tasks.append(micro_task)

        return all_micro_tasks

    def _create_umbrella_task_for_file(self, file_path_str: str, file_purpose: str, project_name: str,
                                       file_description: str) -> SimpleTaskSpec:
        task_id = f"{Path(file_path_str).stem}_implement_all"
        description = f"Fully implement {file_path_str} as per its purpose: {file_purpose}. Details: {file_description}"
        context_str = f"File: {file_path_str} ({file_purpose}) within Project: {project_name}."
        exact_requirements_str = f"""
        GOAL: Implement the entire functionality for the file '{file_path_str}'.
        PURPOSE: {file_purpose}
        DESCRIPTION FROM PLANNER: {file_description}

        EXPECTATIONS:
        - Write complete, professional, and production-ready Python code.
        - Include all necessary imports at the top of the file.
        - Define all required classes, functions, and variables.
        - Implement comprehensive error handling.
        - Add clear docstrings for all public classes, functions, and methods.
        - Ensure the code is modular and follows Python best practices (PEP 8).
        - If this is a main executable file, include an `if __name__ == "__main__":` block.
        - Adhere to any overarching architectural patterns or coding standards defined by the Planner.
        """
        return SimpleTaskSpec(
            id=task_id, description=description, expected_lines=100,
            context=context_str, exact_requirements=exact_requirements_str.strip(),
            component_type="full_file", file_path=file_path_str
        )

    def _create_fallback_tasks_for_project(self, planner_json_plan: Dict[str, Any]) -> List[SimpleTaskSpec]:
        project_name = planner_json_plan.get("project_name", "unspecified_project")
        project_description = planner_json_plan.get("project_description", "An AI-generated project.")
        main_file_name = next(iter(planner_json_plan.get("files", {})), "main.py")

        return [
            SimpleTaskSpec(
                id=f"{project_name}_full_project_implementation",
                description=f"Implement the complete project: {project_name}. Description: {project_description}",
                expected_lines=200,
                context=f"Project: {project_name}. Task: Implement all described functionality.",
                exact_requirements=f"""
                PROJECT GOAL: Create a fully functional application for '{project_name}'.
                PROJECT DESCRIPTION: {project_description}

                OVERALL REQUIREMENTS:
                - Implement all features as implied by the project description.
                - Create necessary file structures if not already defined.
                - Ensure the application is runnable and produces meaningful output.
                - Write clean, robust, and well-documented Python code.
                """,
                component_type="full_project_fallback",
                file_path=main_file_name
            )
        ]