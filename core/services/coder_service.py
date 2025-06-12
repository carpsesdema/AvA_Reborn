# core/services/coder_service.py

import json
import re
import textwrap
from typing import Dict, Any

from core.llm_client import LLMRole
from core.enhanced_micro_task_engine import SimpleTaskSpec
from .base_service import BaseAIService


HYBRID_CODER_PROMPT_TEMPLATE = textwrap.dedent("""
    You are the CODER AI in the AvA hybrid system. Execute this micro-task with Gemini Flash efficiency.

    **MICRO-TASK SPECIFICATION:**
    {task_spec}

    **PROJECT CONTEXT:**
    {project_context}

    **TEAM PATTERNS & STANDARDS:**
    {team_context}

    **IMPLEMENTATION REQUIREMENTS:**
    1. Follow the exact specifications in the micro-task
    2. Implement robust error handling as specified
    3. Write clean, efficient code optimized for the component type
    4. Consider integration points with other components
    5. Follow security best practices (no eval, sanitize inputs)

    **CRITICAL OUTPUT REQUIREMENT:**
    You MUST respond with ONLY a valid JSON object in this exact structure:

    {{
      "IMPLEMENTED_CODE": "actual_python_code_here_as_string",
      "IMPLEMENTATION_NOTES": "technical_decisions_and_reasoning",
      "INTEGRATION_HINTS": "how_to_assemble_with_other_components",
      "EDGE_CASES_HANDLED": "list_of_edge_cases_addressed",
      "TESTING_CONSIDERATIONS": "unit_testing_guidance"
    }}

    Do not include any text, explanations, or markdown formatting. Return ONLY the JSON object.
""")


class CoderService(BaseAIService):
    """Enhanced Coder Service with hybrid workflow and model selection support."""

    async def execute_micro_task_with_gemini_flash(self, task: SimpleTaskSpec) -> Dict[str, Any]:
        """Execute a micro-task using Gemini Flash for cost efficiency."""
        try:
            self.stream_emitter("Coder", "info",
                                f"🚀 Executing micro-task {task.id} with Gemini Flash", 3)

            # Get team context and project patterns
            team_context = self._get_team_context_string(for_file=task.file_path)

            # Prepare task specification
            task_spec_json = {
                "id": task.id,
                "description": task.description,
                "component_type": task.component_type,
                "context": task.context,
                "requirements": task.exact_requirements,
                "expected_lines": task.expected_lines
            }

            # Build project context
            project_context = {
                "file_path": task.file_path,
                "component_type": task.component_type,
                "coding_standards": "PEP 8, type hints, comprehensive docstrings",
                "security_requirements": "No eval(), sanitize inputs, handle errors"
            }

            # Create hybrid coder prompt
            prompt = HYBRID_CODER_PROMPT_TEMPLATE.format(
                task_spec=json.dumps(task_spec_json, indent=2),
                project_context=json.dumps(project_context, indent=2),
                team_context=team_context
            )

            self.stream_emitter("Coder", "info",
                                f"Sending micro-task to Gemini Flash ({len(prompt)} chars)", 4)

            # Use Gemini Flash (fast, cost-effective model) for implementation
            result = await self._stream_and_collect_json(prompt, LLMRole.CODER, "Coder")

            if not result or "IMPLEMENTED_CODE" not in result:
                raise Exception("Micro-task execution failed - no code generated")

            # Validate JSON structure
            required_keys = ["IMPLEMENTED_CODE", "IMPLEMENTATION_NOTES", "INTEGRATION_HINTS",
                             "EDGE_CASES_HANDLED", "TESTING_CONSIDERATIONS"]

            for key in required_keys:
                if key not in result:
                    result[key] = f"Not provided for {key}"

            self.stream_emitter("Coder", "success",
                                f"✅ Micro-task {task.id} completed successfully", 3)

            # Contribute implementation insights
            self._contribute_team_insight(
                "implementation",
                "Coder",
                f"Implemented {task.component_type}: {task.description[:50]}...",
                "low",
                [task.file_path] if task.file_path else []
            )

            return result

        except Exception as e:
            self.logger.error(f"Micro-task execution failed for {task.id}: {e}")
            self.stream_emitter("Coder", "error",
                                f"❌ Micro-task {task.id} failed: {str(e)}", 3)
            raise

    async def generate_file_from_spec(self, file_path: str, file_spec: dict,
                                      project_context: dict, dependency_context: str) -> str:
        """Generate complete file from specification (fallback method)."""
        self.stream_emitter("Coder", "info", f"Generating {file_path} using traditional approach", 2)

        # Build comprehensive context
        team_context = self._get_team_context_string(for_file=file_path)
        rag_context = await self._get_intelligent_rag_context(f"{file_path} {file_spec.get('purpose', '')}")

        # Create traditional file generation prompt
        prompt = f"""
        Generate a complete Python file based on this specification.

        **FILE:** {file_path}
        **PURPOSE:** {file_spec.get('purpose', 'No purpose specified')}
        **DEPENDENCIES:** {dependency_context}

        **TEAM PATTERNS:**
        {team_context}

        **RAG CONTEXT:**
        {rag_context}

        **SPECIFICATION:**
        {json.dumps(file_spec, indent=2)}

        Generate complete, production-ready Python code following all specifications.
        Return ONLY the Python code, no explanations or markdown formatting.
        """

        # Use big model for complex file generation
        response = await self.llm_client.chat(prompt, role=LLMRole.CODER)

        # Clean code output
        cleaned_code = self._clean_code_output(response)

        self.stream_emitter("Coder", "success", f"✅ Generated {file_path} ({len(cleaned_code)} chars)", 2)

        return cleaned_code

    def _clean_code_output(self, code: str) -> str:
        """Remove markdown fences and clean code output."""
        # Handle markdown fences
        match = re.search(r"```(?:python|py)?\s*\n(.*?)\n\s*```", code, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Fallback for code without language specifier
        match = re.search(r"```\s*\n(.*?)\n\s*```", code, re.DOTALL)
        if match:
            return match.group(1).strip()

        return code.strip()