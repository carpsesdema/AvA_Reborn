# core/services/architect_service.py - V2 with Streaming Tech Specs!

import json
import textwrap
import logging
import asyncio
from typing import Dict, List, Any, AsyncGenerator

from core.llm_client import LLMRole
from core.project_state_manager import ProjectStateManager
from core.state_models import FileState
from .base_service import BaseAIService

# New prompt to first get the overall structure
PROJECT_STRUCTURE_PROMPT_TEMPLATE = textwrap.dedent("""
    You are the lead ARCHITECT AI. Your first task is to determine the overall file structure for a project based on a user's request.
    Focus ONLY on what files are needed and their dependencies. Do NOT generate component details yet.

    **USER REQUEST:** {user_prompt}
    **CONVERSATION CONTEXT:** {conversation_context}
    **GDD CONTEXT:** {gdd_context}
    **TEAM INSIGHTS & PATTERNS:** {team_context}
    **KNOWLEDGE BASE CONTEXT:** {rag_context}

    **INSTRUCTIONS:**
    1.  Analyze all provided context.
    2.  Determine a complete list of necessary files.
    3.  For each file, list its direct dependencies (other files in this project it will import).
    4.  Estimate the complexity of each file (low, medium, high).
    5.  List any external libraries required for the `requirements.txt` file.

    **CRITICAL:** Your response MUST be a single, valid JSON object with ONLY the following structure. Do not add explanations or markdown.
    {{
      "project_name": "string",
      "project_description": "string",
      "files": [
        {{
            "filename": "path/to/file.py",
            "purpose": "A concise, one-sentence purpose for this file.",
            "dependencies": ["dependency1.py", "dependency2.py"],
            "complexity": "low|medium|high"
        }}
      ],
      "requirements": ["package1", "package2"]
    }}
""")

# New prompt to generate the detailed spec for a SINGLE file
FILE_SPEC_PROMPT_TEMPLATE = textwrap.dedent("""
    You are a detail-oriented ARCHITECT AI. Your task is to create a detailed technical specification for a SINGLE file, based on the overall project goal.

    **PROJECT GOAL:** {project_description}
    **FILE TO SPEC:** `{filename}`
    **FILE's PURPOSE:** {file_purpose}
    **ALL PROJECT FILES & PURPOSES:**
    {all_files_summary}

    **INSTRUCTIONS:**
    Create a detailed component breakdown for ONLY the file `{filename}`.
    -   **task_id**: A unique snake_case identifier for the component (e.g., `render_world`, `player_input_handler`).
    -   **description**: A clear, concise explanation of what the component does.
    -   **component_type**: The type, e.g., "function", "class", "method", "variable_initialization".
    -   **core_logic_steps**: A high-level, step-by-step summary of its implementation logic.
    -   **error_conditions_to_handle**: Infer potential errors it handles or should handle.
    -   **interactions**: Describe how it interacts with other components, even those in other files.
    -   **critical_notes**: Note any non-obvious details regarding security, performance, or algorithms.

    **CRITICAL:** Your response MUST be a single, valid JSON object with ONLY the following structure for THIS ONE FILE. Do not add explanations.
    {{
        "purpose": "{file_purpose}",
        "dependencies": ["list", "of", "imported_project_files"],
        "components": [
            {{
              "task_id": "unique_id",
              "description": "component description",
              "component_type": "function|class|method",
              "core_logic_steps": ["step1", "step2"],
              "error_conditions_to_handle": ["error1", "error2"],
              "interactions": ["component1", "component2"],
              "critical_notes": "security/performance notes"
            }}
        ]
    }}
""")

FILE_ANALYSIS_PROMPT_TEMPLATE = textwrap.dedent("""
    You are a master software architect. Your task is to deeply analyze the provided Python file and reverse-engineer its complete technical specification into a JSON object.

    **FILE PATH:** {file_path}
    **FILE CONTENT:**
    ```python
    {file_content}
    ```

    **INSTRUCTIONS:**
    Deconstruct the file into its core components. For each component (class, function), you must infer and specify the following:
    1.  **purpose**: A concise description of the file's overall role.
    2.  **dependencies**: A list of all modules this file depends on (from imports).
    3.  **components**: A list of detailed component objects. For each component:
        -   **task_id**: Create a unique ID, like `{file_path}/component_name`.
        -   **description**: What the component does and its role.
        -   **component_type**: The type, e.g., "class", "function", "method".
        -   **core_logic_steps**: A high-level summary of its implementation.
        -   **error_conditions_to_handle**: Infer potential errors it handles or should handle (e.g., file not found, API errors, invalid input).
        -   **interactions**: Describe how it interacts with other components, even those in other files.
        -   **critical_notes**: Note any non-obvious details regarding security, performance, or algorithms.

    **CRITICAL:** Your response MUST be a single, valid JSON object with ONLY the following structure. Do not add explanations.
    {{
      "purpose": "A concise description of the file's role in the project.",
      "dependencies": ["list", "of", "imported_modules"],
      "components": [
        {{
          "task_id": "file_path/component_name",
          "description": "What this component does and its role in the file.",
          "component_type": "class|function",
          "core_logic_steps": ["High-level summary of implementation steps."],
          "error_conditions_to_handle": ["List of inferred error conditions."],
          "interactions": ["Description of interactions with other components."],
          "critical_notes": "Inferred notes on security, performance, etc."
        }}
      ]
    }}
""")


class ArchitectService(BaseAIService):
    """Architect Service refactored for streaming file-by-file specifications."""

    async def create_tech_spec_stream(self, user_prompt: str, conversation_context: List[Dict] = None) -> \
    AsyncGenerator[Dict[str, Any], None]:
        """
        Creates a technical specification by first planning the structure and then
        streaming the detailed specification for each file individually.
        """
        self.stream_emitter("Architect", "stage_start", "Phase 1: Planning project structure...", 0)

        # --- Stage 1: Get Overall Project Structure ---
        prompt_parts = user_prompt.split("\n\n--- GDD CONTEXT ---\n")
        actual_user_prompt = prompt_parts[0]
        gdd_context = prompt_parts[1] if len(prompt_parts) > 1 else "No GDD provided."

        team_context = self._get_team_context_string()
        rag_context = await self._get_intelligent_rag_context(actual_user_prompt)
        conversation_str = "\n".join([f"{msg.get('role', 'user')}: {msg.get('message', '')}" for msg in
                                      conversation_context[-3:]]) if conversation_context else ""

        structure_prompt = PROJECT_STRUCTURE_PROMPT_TEMPLATE.format(
            user_prompt=actual_user_prompt,
            conversation_context=conversation_str,
            gdd_context=gdd_context,
            team_context=team_context,
            rag_context=rag_context
        )

        structure_plan = await self._stream_and_collect_json(structure_prompt, LLMRole.ARCHITECT, "Architect")
        if not structure_plan or not structure_plan.get("files"):
            self.stream_emitter("Architect", "error", "Failed to determine project structure. Aborting.", 1)
            return

        # First, yield the overall project metadata
        project_metadata = {
            "project_name": structure_plan.get("project_name", "unnamed_project"),
            "project_description": structure_plan.get("project_description", "No description."),
            "requirements": structure_plan.get("requirements", [])
        }
        yield {"type": "metadata", "data": project_metadata}

        self.stream_emitter("Architect", "stage_start", "Phase 2: Generating detailed specs for each file...", 0)

        # --- Stage 2: Generate and Stream Detailed Spec for Each File ---
        files_to_spec = structure_plan.get("files", [])
        all_files_summary = "\n".join([f"- {f['filename']}: {f['purpose']}" for f in files_to_spec])

        for file_info in files_to_spec:
            filename = file_info["filename"]
            file_purpose = file_info["purpose"]
            self.stream_emitter("Architect", "info", f"Creating detailed spec for {filename}...", 1)

            spec_prompt = FILE_SPEC_PROMPT_TEMPLATE.format(
                project_description=project_metadata["project_description"],
                filename=filename,
                file_purpose=file_purpose,
                all_files_summary=all_files_summary
            )

            try:
                detailed_spec = await self._stream_and_collect_json(spec_prompt, LLMRole.ARCHITECT,
                                                                    f"Architect-{filename}")
                # Yield the file spec
                yield {"type": "file_spec", "filename": filename, "spec": detailed_spec}
                self.stream_emitter("Architect", "success", f"Specification for {filename} complete.", 2)
            except Exception as e:
                self.logger.error(f"Failed to generate spec for {filename}: {e}", exc_info=True)
                yield {"type": "error", "filename": filename, "error": str(e)}

    # This method is now used as a fallback or for simpler cases
    async def create_tech_spec(self, user_prompt: str, conversation_context: List[Dict] = None) -> dict:
        """Compatibility method that collects the stream into a single dictionary."""
        final_spec = {"technical_specs": {"files": {}}}
        async for item in self.create_tech_spec_stream(user_prompt, conversation_context):
            if item["type"] == "metadata":
                final_spec.update(item["data"])
                final_spec["technical_specs"]["requirements"] = item["data"].get("requirements", [])
            elif item["type"] == "file_spec":
                final_spec["technical_specs"]["files"][item["filename"]] = item["spec"]
            elif item["type"] == "error":
                self.logger.error(f"Error streaming spec for {item['filename']}: {item['error']}")
        return final_spec

    # The analysis methods remain the same as they already work file-by-file
    async def _analyze_file_in_parallel(self, file_state: FileState) -> Dict[str, Any]:
        """Analyzes a single file using a focused LLM call."""
        prompt = FILE_ANALYSIS_PROMPT_TEMPLATE.format(
            file_path=file_state.path,
            file_content=file_state.content
        )
        json_response = await self._stream_and_collect_json(prompt, LLMRole.CODER, f"Analyst-{file_state.path}")
        return {"file_path": file_state.path, "analysis": json_response}

    async def analyze_and_create_spec_from_project(self, project_state: ProjectStateManager) -> dict:
        """
        Analyzes an existing project by breaking it down into parallel,
        per-file analyses, and then synthesizes the results.
        """
        self.stream_emitter("Architect", "thought",
                            "Initiating high-speed parallel analysis of project files...", 1)

        files_to_analyze = list(project_state.files.values())
        if not files_to_analyze:
            self.stream_emitter("Architect", "warning", "No files found to analyze.", 1)
            return {}

        tasks = [self._analyze_file_in_parallel(file_state) for file_state in files_to_analyze]
        self.stream_emitter("Architect", "info", f"Dispatching {len(tasks)} parallel analysis tasks.", 2)
        analysis_results = await asyncio.gather(*tasks, return_exceptions=True)

        self.stream_emitter("Architect", "info", "Parallel analysis complete. Synthesizing final specification.", 1)
        final_spec = {"files": {}}
        successful_analyses = 0
        for result in analysis_results:
            if isinstance(result, Exception):
                self.logger.error(f"File analysis task failed: {result}")
                self.stream_emitter("Architect", "warning", f"A file analysis task failed: {result}", 2)
                continue

            file_path = result.get("file_path")
            analysis = result.get("analysis")
            if file_path and analysis:
                final_spec["files"][file_path] = analysis
                successful_analyses += 1

        if successful_analyses == 0:
            self.stream_emitter("Architect", "error", "All file analysis tasks failed. Cannot generate tech spec.", 1)
            return {}

        project_name = project_state.project_metadata.get("name", "Unknown Project")
        full_tech_spec = {
            "project_name": project_name,
            "project_description": f"An analysis of the existing project: {project_name}",
            "technical_specs": final_spec
        }

        self._contribute_team_insight(
            "architectural", "Architect",
            f"Completed parallel analysis of {successful_analyses}/{len(files_to_analyze)} files.", "medium"
        )

        self.stream_emitter("Architect", "success", "Project analysis complete. Technical spec created!", 1)
        return full_tech_spec