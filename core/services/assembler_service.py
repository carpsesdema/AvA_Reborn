# core/services/assembler_service.py

import json
import re
import textwrap
from typing import Dict, List, Any

from core.llm_client import LLMRole
from .base_service import BaseAIService

ASSEMBLER_PROMPT_TEMPLATE = textwrap.dedent("""
    You are the ASSEMBLER AI. Your task is to combine the provided micro-task code components into a single, complete, and professional Python file.

    **FILE TO ASSEMBLE:** {filename}
    **FILE SPECIFICATIONS:**
    {file_spec}

    **MICRO-TASK IMPLEMENTATIONS:**
    {micro_task_results}

    **PROJECT CONTEXT:**
    {project_context}

    **ASSEMBLY REQUIREMENTS:**
    1.  **Combine All Code:** Integrate the `IMPLEMENTED_CODE` from all micro-tasks into a single, cohesive file.
    2.  **Organize Imports:** Consolidate all imports at the top of the file, following PEP 8 standards (standard library, third-party, local modules). Remove any duplicate imports.
    3.  **Structure the File:**
        -   Place file-level constants and configurations right after the imports.
        -   Logically order classes and functions based on dependencies and the overall file specification.
    4.  **Add Documentation:**
        -   Create a professional, descriptive file-level docstring that explains the purpose of the file, based on the specifications.
        -   Ensure that all classes and functions from the micro-tasks have their docstrings correctly included.
    5.  **Ensure Quality and Polish:**
        -   The final output must be a single block of clean, production-ready Python code.
        -   Ensure consistent naming, style, and formatting throughout the file.
        -   Verify that the assembled code is complete and meets all requirements from the file specification.

    **CRITICAL OUTPUT FORMAT:**
    You MUST return ONLY the raw, complete Python code for the file `{filename}`.
    Do NOT include any explanations, introductory text, or markdown code fences like ```python ... ```.
    Your entire response should be start-to-finish Python code.
""")


class AssemblerService(BaseAIService):
    """New Assembler Service for combining micro-task outputs."""

    async def assemble_file_from_micro_tasks(self, filename: str, file_spec: dict,
                                             micro_task_results: List[Dict[str, Any]],
                                             project_context: dict) -> str:
        """Assemble micro-task results into a complete, professional file."""
        try:
            self.stream_emitter("Assembler", "info",
                                f"🔧 Assembling {len(micro_task_results)} micro-tasks for {filename}", 2)

            results_text = [f"=== Micro-Task {i + 1} ===\n{json.dumps(result, indent=2)}\n" for i, result in
                            enumerate(micro_task_results)]

            prompt = ASSEMBLER_PROMPT_TEMPLATE.format(
                filename=filename,
                file_spec=json.dumps(file_spec, indent=2),
                micro_task_results="\n".join(results_text),
                project_context=json.dumps(project_context, indent=2)
            )

            self.stream_emitter("Assembler", "info",
                                f"Sending assembly request ({len(prompt)} chars) to assembler model", 3)

            assembled_code = await self.llm_client.chat(prompt, role=LLMRole.ASSEMBLER)

            # --- FIX: Defensively clean the output to prevent markdown issues ---
            cleaned_code = self._clean_code_output(assembled_code)

            if not cleaned_code.strip():
                raise Exception("Assembly produced no code.")

            self.stream_emitter("Assembler", "success",
                                f"✅ Assembly complete for {filename} ({len(cleaned_code)} chars)", 2)

            self._contribute_team_insight(
                "assembly", "Assembler", f"Assembled {len(micro_task_results)} components into {filename}",
                "medium", [filename]
            )

            return cleaned_code

        except Exception as e:
            self.logger.error(f"Assembly failed for {filename}: {e}", exc_info=True)
            self.stream_emitter("Assembler", "error", f"❌ Assembly failed for {filename}: {str(e)}", 2)
            # Return an empty string or re-raise to indicate failure to the pipeline
            return ""

    def _clean_code_output(self, code: str) -> str:
        """
        Robustly strips markdown fences and other common LLM chatter from code output.
        """
        # Find the start of the first code block
        match = re.search(r"```(?:python|py)?\s*\n", code)
        if match:
            # If a marker is found, start the code from there
            code = code[match.end():]
            # Find the end of the last code block
            end_match = code.rfind("```")
            if end_match != -1:
                code = code[:end_match]

        return code.strip()