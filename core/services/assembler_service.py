# core/services/assembler_service.py

import json
import re
import textwrap
from typing import Dict, List, Any

from core.llm_client import LLMRole
from .base_service import BaseAIService


ASSEMBLER_PROMPT_TEMPLATE = textwrap.dedent("""
    You are the ASSEMBLER AI. Combine these micro-task implementations into a complete, professional Python file.

    **FILE TO ASSEMBLE:** {filename}
    **FILE SPECIFICATIONS:** {file_spec}

    **MICRO-TASK IMPLEMENTATIONS:**
    {micro_task_results}

    **PROJECT CONTEXT:**
    {project_context}

    **ASSEMBLY REQUIREMENTS:**
    1. **Code Organization:**
       - Organize imports (stdlib, third-party, local)
       - Place constants and configuration at top
       - Order classes and functions logically
       - Add proper file-level docstring

    2. **Integration:**
       - Extract IMPLEMENTED_CODE from each micro-task result
       - Ensure consistent naming and style
       - Add necessary glue code for component interaction
       - Verify all dependencies are imported
       - Remove duplicate imports

    3. **Quality Assurance:**
       - Follow project conventions and patterns
       - Add comprehensive documentation
       - Ensure logical flow and organization
       - Verify completeness against specifications

    4. **Professional Polish:**
       - Add file header with description
       - Ensure proper error handling throughout
       - Format according to PEP 8 standards
       - Create production-ready code

    Return ONLY the complete, assembled Python code. No explanations or formatting.
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

            # Prepare micro-task results for prompt
            results_text = []
            for i, result in enumerate(micro_task_results):
                results_text.append(f"=== Micro-Task {i + 1} ===")
                results_text.append(json.dumps(result, indent=2))
                results_text.append("")

            # Build assembly prompt
            prompt = ASSEMBLER_PROMPT_TEMPLATE.format(
                filename=filename,
                file_spec=json.dumps(file_spec, indent=2),
                micro_task_results="\n".join(results_text),
                project_context=json.dumps(project_context, indent=2)
            )

            self.stream_emitter("Assembler", "info",
                                f"Sending assembly request ({len(prompt)} chars) to medium model", 3)

            # Use medium model for assembly (balance of capability and cost)
            assembled_code = await self.llm_client.chat(prompt, role=LLMRole.ASSEMBLER)

            # Clean the assembled code
            cleaned_code = self._clean_code_output(assembled_code)

            if not cleaned_code or len(cleaned_code) < 50:
                raise Exception("Assembly produced insufficient code")

            self.stream_emitter("Assembler", "success",
                                f"✅ Assembly complete for {filename} ({len(cleaned_code)} chars)", 2)

            # Contribute assembly insights
            self._contribute_team_insight(
                "assembly",
                "Assembler",
                f"Assembled {len(micro_task_results)} components into {filename}",
                "medium",
                [filename]
            )

            return cleaned_code

        except Exception as e:
            self.logger.error(f"Assembly failed for {filename}: {e}")
            self.stream_emitter("Assembler", "error",
                                f"❌ Assembly failed for {filename}: {str(e)}", 2)
            raise

    def _clean_code_output(self, code: str) -> str:
        """Clean assembled code output."""
        # Remove markdown fences
        match = re.search(r"```(?:python|py)?\s*\n(.*?)\n\s*```", code, re.DOTALL)
        if match:
            return match.group(1).strip()

        return code.strip()