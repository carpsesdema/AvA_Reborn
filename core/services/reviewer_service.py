# core/services/reviewer_service.py

import json
import re
import textwrap

from core.llm_client import LLMRole
from .base_service import BaseAIService

REVIEWER_PROMPT_TEMPLATE = textwrap.dedent("""
    You are the REVIEWER AI. Conduct a comprehensive code review of this Python file.

    **FILE:** {file_path}
    **PROJECT DESCRIPTION:** {project_description}

    **TEAM QUALITY STANDARDS:**
    {team_context}

    **CODE TO REVIEW:**
    ```python
    {code}
    ```

    Perform a thorough review checking:
    1. Code quality and best practices
    2. Adherence to team standards and patterns
    3. Security considerations
    4. Performance implications
    5. Documentation completeness
    6. Error handling
    7. Testing considerations

    Your response MUST be a valid JSON object:
    {{
      "approved": true/false,
      "summary": "Brief overall assessment",
      "strengths": ["List of positive aspects"],
      "issues": ["List of concerns or improvements needed"],
      "suggestions": ["Specific improvement recommendations"],
      "confidence": 0.0-1.0
    }}

    Return ONLY the JSON object, no explanations or markdown formatting.
""")


def create_refinement_prompt(code: str, instruction: str) -> str:
    """Creates a standardized prompt for code refinement."""
    return textwrap.dedent(f"""
        You are a code refiner. Modify the given code based on this instruction.
        **IMPORTANT**: Return ONLY the complete, modified code. No explanations or markdown.

        INSTRUCTION: {instruction}

        CODE TO REFINE:
        ```python
        {code}
        ```

        COMPLETE REFINED CODE:
    """)


class ReviewerService(BaseAIService):
    """Enhanced Reviewer Service for code quality assurance."""

    async def review_file(self, file_path: str, code: str, project_description: str) -> dict:
        """Conduct comprehensive code review."""
        try:
            self.stream_emitter("Reviewer", "info", f"🔍 Reviewing {file_path}", 2)

            # Get team context for standards
            team_context = self._get_team_context_string(for_file=file_path)

            # Create review prompt
            prompt = REVIEWER_PROMPT_TEMPLATE.format(
                file_path=file_path,
                project_description=project_description,
                team_context=team_context,
                code=code
            )

            self.stream_emitter("Reviewer", "info",
                                f"Sending review request ({len(prompt)} chars) to reviewer model", 3)

            # Use big model for thorough review
            review_result = await self._stream_and_collect_json(prompt, LLMRole.REVIEWER, "Reviewer")

            if not review_result:
                return {
                    "approved": False,
                    "summary": "Review failed - could not analyze code",
                    "strengths": [],
                    "issues": ["Review process failed"],
                    "suggestions": ["Manual review required"],
                    "confidence": 0.0
                }

            self.stream_emitter("Reviewer", "success", f"✅ Review complete for {file_path}", 2)

            # Contribute review insights
            approval_status = "approved" if review_result.get("approved", False) else "needs_work"
            self._contribute_team_insight(
                "review",
                "Reviewer",
                f"Code review for {file_path}: {approval_status}",
                "medium",
                [file_path]
            )

            return review_result

        except Exception as e:
            self.logger.error(f"Review failed for {file_path}: {e}")
            self.stream_emitter("Reviewer", "error",
                                f"❌ Review failed for {file_path}: {str(e)}", 2)
            return {
                "approved": False,
                "summary": f"Review failed due to error: {str(e)}",
                "strengths": [],
                "issues": [f"Review error: {str(e)}"],
                "suggestions": ["Manual review required"],
                "confidence": 0.0
            }

    async def refine_code(self, code: str, instruction: str) -> str:
        """Refine code based on specific instruction."""
        try:
            self.stream_emitter("Reviewer", "info", f"🔧 Refining code based on: {instruction[:50]}...", 3)

            prompt = create_refinement_prompt(code, instruction)
            refined_code = await self.llm_client.chat(prompt, role=LLMRole.REVIEWER)

            # Clean the refined code
            cleaned_code = self._clean_code_output(refined_code)

            self.stream_emitter("Reviewer", "success", f"✅ Code refinement complete", 3)
            return cleaned_code

        except Exception as e:
            self.logger.error(f"Code refinement failed: {e}")
            self.stream_emitter("Reviewer", "error", f"❌ Refinement failed: {str(e)}", 3)
            return code  # Return original code if refinement fails

    def _clean_code_output(self, code: str) -> str:
        """Clean code output by removing markdown formatting."""
        # Remove markdown fences
        match = re.search(r"```(?:python|py)?\s*\n(.*?)\n\s*```", code, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Fallback for when the LLM just returns code without the python specifier
        match = re.search(r"```\s*\n(.*?)\n\s*```", code, re.DOTALL)
        if match:
            return match.group(1).strip()

        return code.strip()