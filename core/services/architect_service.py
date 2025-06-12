# core/services/architect_service.py

import textwrap
import logging
import asyncio
from typing import Dict, List, Any

from core.llm_client import LLMRole
from core.project_state_manager import ProjectStateManager
from core.state_models import FileState
from .base_service import BaseAIService


ARCHITECT_PROMPT_TEMPLATE = textwrap.dedent("""
    You are the ARCHITECT AI. Create a comprehensive technical specification for this project.

    **USER REQUEST:** {user_prompt}
    **CONVERSATION CONTEXT:** {conversation_context}
    **GDD CONTEXT:** {gdd_context}

    **TEAM INSIGHTS & PATTERNS:**
    {team_context}

    **KNOWLEDGE BASE CONTEXT:**
    {rag_context}

    Create a detailed technical specification that includes:
    1. **Project Overview:** Clear description and objectives
    2. **File Structure:** Complete breakdown of required files with dependencies
    3. **Component Specifications:** For each file, provide detailed component breakdown including:
       - task_id: Unique identifier for each component
       - description: What the component does
       - component_type: function, class, method, etc.
       - core_logic_steps: Step-by-step implementation guide
       - error_conditions_to_handle: Specific error scenarios
       - interactions: How it connects with other components
       - critical_notes: Security, performance, or architectural constraints
    4. **Implementation Standards:** Coding patterns and best practices
    5. **Quality Criteria:** Review standards and testing requirements

    Your response MUST be a valid JSON object with this structure:
    {{
      "project_name": "string",
      "project_description": "string",
      "technical_specs": {{
        "files": {{
          "filename": {{
            "purpose": "string",
            "dependencies": ["list", "of", "files"],
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
        }},
        "requirements": ["package1", "package2"],
        "coding_standards": "PEP 8 compliance, type hints, docstrings",
        "project_patterns": "Architecture patterns to follow"
      }}
    }}

    Return ONLY the JSON object, no explanations or markdown formatting.
""")

# NEW: High-performance prompt for parallel file analysis
FILE_ANALYSIS_PROMPT_TEMPLATE = textwrap.dedent("""
    You are an expert code analyst. Analyze the provided Python file and create a JSON specification for it.

    **FILE PATH:** {file_path}
    **FILE CONTENT:**
    ```python
    {file_content}
    ```

    **INSTRUCTIONS:**
    1.  Determine the primary purpose of this file.
    2.  Identify all major components (classes, functions).
    3.  For each component, create a specification including its description, type, and core logic.
    4.  Infer the file's dependencies based on its import statements.

    **CRITICAL:** Your response MUST be a single, valid JSON object with ONLY the following structure:
    {{
      "purpose": "A concise description of the file's role in the project.",
      "dependencies": ["list", "of", "imported_modules"],
      "components": [
        {{
          "task_id": "{file_path}/component_name",
          "description": "What this component does.",
          "component_type": "class|function",
          "core_logic_steps": ["A high-level summary of the implementation steps."]
        }}
      ]
    }}

    Return ONLY the JSON object. Do not add explanations or markdown.
""")


class ArchitectService(BaseAIService):
    """Enhanced Architect Service for hybrid workflow planning."""

    async def create_tech_spec(self, user_prompt: str, conversation_context: List[Dict] = None) -> dict:
        """Create technical specification with detailed component breakdown."""
        self.stream_emitter("Architect", "thought",
                            "Phase 1: Creating comprehensive project architecture with micro-task breakdown...", 0)

        # Parse user prompt and GDD context
        prompt_parts = user_prompt.split("\n\n--- GDD CONTEXT ---\n")
        actual_user_prompt = prompt_parts[0]
        gdd_context = prompt_parts[1] if len(prompt_parts) > 1 else "No GDD provided."

        # Get team context and RAG context
        team_context = self._get_team_context_string()
        rag_context = await self._get_intelligent_rag_context(actual_user_prompt)

        # Build conversation context string
        conversation_str = ""
        if conversation_context:
            conversation_str = "\n".join([
                f"{msg.get('role', 'user')}: {msg.get('message', '')}"
                for msg in conversation_context[-3:]  # Last 3 messages for context
            ])

        # Create architecture prompt
        architecture_prompt = ARCHITECT_PROMPT_TEMPLATE.format(
            user_prompt=actual_user_prompt,
            conversation_context=conversation_str,
            gdd_context=gdd_context,
            team_context=team_context,
            rag_context=rag_context
        )

        self.stream_emitter("Architect", "info",
                            f"Sending architecture prompt ({len(architecture_prompt)} chars) to big model", 2)

        # Use big model for architecture
        tech_spec = await self._stream_and_collect_json(architecture_prompt, LLMRole.ARCHITECT, "Architect")

        if not tech_spec or not tech_spec.get("technical_specs"):
            self.stream_emitter("Architect", "error",
                                "Architecture failed. Could not produce valid technical specification.", 1)
            return {}

        # Contribute architectural insights
        self._contribute_team_insight(
            "architectural",
            "Architect",
            f"Created architecture for project: {tech_spec.get('project_name', 'Unknown')}",
            "high"
        )

        self.stream_emitter("Architect", "success", "Architecture complete with detailed component breakdown!", 1)
        return tech_spec

    async def _analyze_file_in_parallel(self, file_state: FileState) -> Dict[str, Any]:
        """Analyzes a single file using a focused LLM call."""
        prompt = FILE_ANALYSIS_PROMPT_TEMPLATE.format(
            file_path=file_state.path,
            file_content=file_state.content
        )
        # Use a faster model for this focused task
        json_response = await self._stream_and_collect_json(prompt, LLMRole.CODER, f"Analyst-{file_state.path}")
        # Return the analysis along with the file path for reassembly
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

        # Create a list of concurrent analysis tasks
        tasks = [self._analyze_file_in_parallel(file_state) for file_state in files_to_analyze]

        self.stream_emitter("Architect", "info", f"Dispatching {len(tasks)} parallel analysis tasks.", 2)

        # Execute all tasks concurrently and gather results
        analysis_results = await asyncio.gather(*tasks, return_exceptions=True)

        self.stream_emitter("Architect", "info", "Parallel analysis complete. Synthesizing final specification.", 1)

        # Synthesize the final technical specification
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

        # Construct the full tech_spec object expected by the system
        project_name = project_state.project_metadata.get("name", "Unknown Project")
        full_tech_spec = {
            "project_name": project_name,
            "project_description": f"An analysis of the existing project: {project_name}",
            "technical_specs": final_spec
            # requirements and other fields can be populated here if needed
        }

        self._contribute_team_insight(
            "architectural",
            "Architect",
            f"Completed parallel analysis of {successful_analyses}/{len(files_to_analyze)} files.",
            "medium"
        )

        self.stream_emitter("Architect", "success", "Project analysis complete. Technical spec created!", 1)
        return full_tech_spec