# assembler.py (updated)

import asyncio
import json
from typing import List, Dict, Any
from utils.llm_client import LLMClient
from utils.file_manager import FileManager
from utils.logger import get_logger

logger = get_logger(__name__)


class Assembler:
    """
    Enhanced assembler that handles atomic micro tasks and creates cohesive files.
    Works with specialized code generation LLMs like CodeLlama or Qwen2.5-code.
    """

    def __init__(self, model_name: str = "qwen2.5-code"):
        self.client = LLMClient(model_name)
        self.file_manager = FileManager()

        # Templates for different micro-task types
        self.task_templates = {
            "function": self._function_template,
            "class": self._class_template,
            "import": self._import_template,
            "docstring": self._docstring_template,
            "test": self._test_template
        }

    async def generate_micro_task_code(self, task, context: str, rag_context: str = "") -> str:
        """
        Generate code for a single micro-task using specialized prompting.
        """

        # Get appropriate template for task type
        template_func = self.task_templates.get(task.task_type, self._generic_template)
        prompt = template_func(task, context, rag_context)

        try:
            response = await self.client.chat(
                system="You are a specialized Python code generator. Generate clean, professional code that follows best practices.",
                user=prompt
            )

            generated_code = response.choices[0].message.content

            # Clean up the generated code
            cleaned_code = self._clean_generated_code(generated_code)

            logger.debug(f"Generated code for task {task.id}: {len(cleaned_code)} chars")
            return cleaned_code

        except Exception as e:
            logger.error(f"Code generation failed for task {task.id}: {e}")
            raise

    async def assemble_file(self, file_path: str, code_snippets: List[str]) -> str:
        """
        Assemble multiple code snippets into a cohesive, properly formatted file.
        """

        # Prepare assembly prompt
        assembly_prompt = f"""
        Assemble the following code snippets into a complete, cohesive Python file for {file_path}.

        Requirements:
        - Organize imports at the top
        - Ensure proper class and function ordering
        - Add any missing docstrings
        - Follow PEP 8 formatting
        - Ensure all code works together properly
        - Remove any duplicate imports or definitions

        Code snippets to assemble:

        {self._format_snippets_for_assembly(code_snippets)}

        Return ONLY the complete, formatted Python file content.
        """

        try:
            response = await self.client.chat(
                system="You are a code assembler. Create clean, well-organized Python files from code snippets.",
                user=assembly_prompt
            )

            assembled_code = response.choices[0].message.content
            formatted_code = self._format_final_code(assembled_code)

            logger.info(f"Assembled file {file_path}: {len(formatted_code)} chars")
            return formatted_code

        except Exception as e:
            logger.error(f"File assembly failed for {file_path}: {e}")
            raise

    def _function_template(self, task, context: str, rag_context: str) -> str:
        """Template for generating functions"""
        return f"""
        Create a Python function based on this requirement:

        Task: {task.description}
        File: {task.file_path}

        Context from file:
        {context}

        RAG Context (best practices and examples):
        {rag_context}

        Generate ONLY the function definition with:
        - Proper type hints
        - Comprehensive docstring
        - Error handling if appropriate
        - Following Python best practices

        Return only the function code, no explanations.
        """

    def _class_template(self, task, context: str, rag_context: str) -> str:
        """Template for generating classes"""
        return f"""
        Create a Python class based on this requirement:

        Task: {task.description}
        File: {task.file_path}

        Context from file:
        {context}

        RAG Context (best practices and examples):
        {rag_context}

        Generate ONLY the class definition with:
        - Proper __init__ method with type hints
        - Class and method docstrings
        - Any required methods mentioned in the task
        - Following Python best practices and SOLID principles

        Return only the class code, no explanations.
        """

    def _import_template(self, task, context: str, rag_context: str) -> str:
        """Template for generating imports"""
        return f"""
        Generate the necessary import statements for:

        Task: {task.description}
        File: {task.file_path}

        Context from file:
        {context}

        Generate ONLY the import statements needed, organized by:
        1. Standard library imports
        2. Third-party imports  
        3. Local imports

        Return only the import statements, one per line.
        """

    def _docstring_template(self, task, context: str, rag_context: str) -> str:
        """Template for generating docstrings"""
        return f"""
        Generate a comprehensive docstring for:

        Task: {task.description}
        File: {task.file_path}

        Code context:
        {context}

        Generate a docstring following Google/NumPy style with:
        - Brief description
        - Args section (if applicable)
        - Returns section (if applicable) 
        - Raises section (if applicable)
        - Examples (if helpful)

        Return only the docstring content (without triple quotes).
        """

    def _test_template(self, task, context: str, rag_context: str) -> str:
        """Template for generating tests"""
        return f"""
        Generate unit tests for:

        Task: {task.description}
        File: {task.file_path}

        Code to test:
        {context}

        Generate pytest-style tests with:
        - Test class if appropriate
        - Multiple test methods covering different scenarios
        - Proper assertions
        - Mock objects if needed for external dependencies

        Return only the test code, no explanations.
        """

    def _generic_template(self, task, context: str, rag_context: str) -> str:
        """Generic template for unknown task types"""
        return f"""
        Generate Python code for:

        Task: {task.description}
        Type: {task.task_type}
        File: {task.file_path}

        Context:
        {context}

        RAG Context:
        {rag_context}

        Generate clean, professional Python code that fulfills the requirement.
        Include appropriate documentation and follow best practices.

        Return only the code, no explanations.
        """

    def _clean_generated_code(self, code: str) -> str:
        """Clean up generated code by removing markdown formatting, etc."""

        # Remove markdown code blocks
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]

        if code.endswith("```"):
            code = code[:-3]

        # Remove any leading/trailing whitespace
        code = code.strip()

        return code

    def _format_snippets_for_assembly(self, snippets: List[str]) -> str:
        """Format code snippets for the assembly prompt"""
        formatted = []
        for i, snippet in enumerate(snippets, 1):
            formatted.append(f"# Snippet {i}:\n{snippet}\n")
        return "\n".join(formatted)

    def _format_final_code(self, code: str) -> str:
        """Final formatting of assembled code"""

        # Clean up any markdown
        code = self._clean_generated_code(code)

        # Ensure proper line endings
        lines = code.split('\n')

        # Remove excessive blank lines
        cleaned_lines = []
        blank_count = 0

        for line in lines:
            if line.strip() == "":
                blank_count += 1
                if blank_count <= 2:  # Allow max 2 consecutive blank lines
                    cleaned_lines.append(line)
            else:
                blank_count = 0
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    async def review_and_fix_code(self, file_path: str, code: str, issues: List[str]) -> str:
        """
        Review and fix code based on identified issues.
        This can be called by the planner for iterative improvement.
        """

        review_prompt = f"""
        Review and fix the following Python code in {file_path}:

        Current code:
        {code}

        Issues to address:
        {chr(10).join(f"- {issue}" for issue in issues)}

        Fix these issues while maintaining the code's functionality.
        Return the complete corrected code.
        """

        try:
            response = await self.client.chat(
                system="You are a code reviewer and fixer. Improve code quality while preserving functionality.",
                user=review_prompt
            )

            fixed_code = self._format_final_code(response.choices[0].message.content)
            logger.info(f"Fixed code for {file_path}")
            return fixed_code

        except Exception as e:
            logger.error(f"Code review/fix failed for {file_path}: {e}")
            raise