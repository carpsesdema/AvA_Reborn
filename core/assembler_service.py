# core/assembler_service.py - Robust Assembler Service

import ast
import re
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path

from core.llm_client import LLMRole


class AssemblerService:
    """📄 Robust Assembly Service - Intelligent Code Integration with Review"""

    def __init__(self, llm_client, rag_manager=None):
        self.llm_client = llm_client
        self.rag_manager = rag_manager

    async def assemble_file(self, file_path: str, task_results: List[dict],
                            plan: dict, context_cache) -> Tuple[str, bool, str]:
        """
        Enhanced assembly with validation, conflict resolution, and mandatory review
        Returns: (assembled_code, review_approved, review_feedback)
        """

        # 1. Validate and clean task results
        validated_results = self._validate_task_results(task_results)
        if not validated_results:
            return self._create_empty_file_template(file_path, plan), False, "No valid task results to assemble"

        # 2. Order by dependencies
        ordered_results = self._resolve_task_dependencies(validated_results)

        # 3. Detect and resolve conflicts
        resolved_results = self._resolve_conflicts(ordered_results)

        # 4. Smart assembly
        assembled_code = await self._smart_assemble(file_path, resolved_results, plan, context_cache)

        # 5. Final validation and cleanup
        final_code = self._final_validation(assembled_code)

        # 6. MANDATORY REVIEW PROCESS
        review_approved, review_feedback = await self._mandatory_review(file_path, final_code, task_results, plan)

        return final_code, review_approved, review_feedback

    async def _mandatory_review(self, file_path: str, assembled_code: str,
                                original_tasks: List[dict], plan: dict) -> Tuple[bool, str]:
        """
        MANDATORY review process - code must pass review before approval
        """

        # Create comprehensive review prompt
        review_prompt = f"""
You are a Senior Code Reviewer. Thoroughly review this assembled Python file:

FILE: {file_path}
PROJECT: {plan.get('project_name', 'Project')}

ORIGINAL TASK COUNT: {len(original_tasks)}
ORIGINAL TASK DESCRIPTIONS:
{self._format_task_descriptions(original_tasks)}

ASSEMBLED CODE:
```python
{assembled_code}
```

REVIEW CRITERIA:
1. Syntax Correctness - Does the code parse without errors?
2. Task Completion - Are all original tasks properly implemented?
3. Code Quality - Is it clean, readable, and follows PEP 8?
4. Integration - Do all components work together harmoniously?
5. Best Practices - Are Python best practices followed?
6. Security - Are there any obvious security issues?
7. Performance - Any obvious performance problems?
8. Documentation - Are functions/classes properly documented?

RESPOND IN JSON FORMAT:
{{
    "approved": true/false,
    "overall_score": 1-10,
    "syntax_valid": true/false,
    "tasks_completed": true/false,
    "code_quality": 1-10,
    "integration_quality": 1-10,
    "issues": [
        "List any specific issues found"
    ],
    "suggestions": [
        "List improvement suggestions"
    ],
    "feedback": "Detailed review feedback",
    "requires_fixes": [
        "List critical fixes needed if not approved"
    ]
}}

Be thorough and constructive. Only approve code that meets professional standards.
"""

        try:
            # Get review from LLM
            response = await self.llm_client.stream_chat(review_prompt, LLMRole.REVIEWER)
            response_text = ''.join([chunk async for chunk in response])

            # Parse review result
            review_data = self._extract_json(response_text)

            approved = review_data.get("approved", False)
            feedback = review_data.get("feedback", "Review completed")

            # Additional programmatic checks
            syntax_valid = self._is_valid_python_syntax(assembled_code)
            if not syntax_valid:
                approved = False
                feedback += "\n❌ CRITICAL: Syntax errors detected - code will not execute"

            # Check for basic requirements
            if len(assembled_code.strip()) < 50:  # Suspiciously short
                approved = False
                feedback += "\n❌ CRITICAL: Code appears incomplete or too short"

            return approved, feedback

        except Exception as e:
            # If review fails, default to not approved
            return False, f"Review process failed: {e}. Code not approved pending manual review."

    def _format_task_descriptions(self, tasks: List[dict]) -> str:
        """Format task descriptions for review"""
        descriptions = []
        for i, task in enumerate(tasks, 1):
            task_info = task.get('task', task)
            desc = task_info.get('description', 'No description')
            task_type = task_info.get('type', 'general')
            descriptions.append(f"{i}. [{task_type}] {desc}")
        return "\n".join(descriptions)

    def _validate_task_results(self, task_results: List[dict]) -> List[dict]:
        """Validate and clean each micro-task result"""
        validated = []

        for result in task_results:
            code = result.get('code', '').strip()
            if not code:
                continue

            # Clean the code
            cleaned_code = self._clean_code(code)

            # Basic syntax check
            if self._is_valid_python_syntax(cleaned_code) or self._can_fix_syntax(cleaned_code):
                if not self._is_valid_python_syntax(cleaned_code):
                    cleaned_code = self._fix_common_syntax_issues(cleaned_code)

                validated.append({
                    'task': result['task'],
                    'code': cleaned_code,
                    'type': result['task'].get('type', 'general')
                })

        return validated

    def _resolve_task_dependencies(self, task_results: List[dict]) -> List[dict]:
        """Order tasks by logical dependencies"""
        # Categorize tasks
        imports = []
        constants = []
        classes = []
        functions = []
        main_logic = []
        other = []

        for result in task_results:
            task_type = result.get('type', 'general')
            code = result['code']

            if task_type == 'imports' or self._contains_imports(code):
                imports.append(result)
            elif task_type == 'constants' or self._contains_constants(code):
                constants.append(result)
            elif task_type == 'class' or self._contains_class_definitions(code):
                classes.append(result)
            elif task_type == 'function' or self._contains_function_definitions(code):
                functions.append(result)
            elif task_type == 'main' or self._contains_main_logic(code):
                main_logic.append(result)
            else:
                other.append(result)

        # Return in logical order
        return imports + constants + classes + functions + other + main_logic

    def _resolve_conflicts(self, task_results: List[dict]) -> List[dict]:
        """Detect and resolve naming conflicts"""
        function_names = set()
        class_names = set()
        resolved_results = []
        conflicts_detected = []

        for result in task_results:
            code = result['code']

            # Extract names from this code block
            functions = self._extract_function_names(code)
            classes = self._extract_class_names(code)

            # Check for conflicts
            has_conflict = False

            # Check function conflicts
            for func_name in functions:
                if func_name in function_names:
                    conflicts_detected.append(f"Duplicate function: {func_name}")
                    has_conflict = True
                else:
                    function_names.add(func_name)

            # Check class conflicts
            for class_name in classes:
                if class_name in class_names:
                    conflicts_detected.append(f"Duplicate class: {class_name}")
                    has_conflict = True
                else:
                    class_names.add(class_name)

            if not has_conflict:
                resolved_results.append(result)

        if conflicts_detected:
            # Log conflicts for review
            print(f"⚠️ Conflicts detected and resolved: {', '.join(conflicts_detected)}")

        return resolved_results

    async def _smart_assemble(self, file_path: str, task_results: List[dict],
                              plan: dict, context_cache) -> str:
        """Intelligent assembly with proper Python structure"""

        if not task_results:
            return self._create_empty_file_template(file_path, plan)

        # Separate imports from code
        all_imports = []
        code_sections = []

        for result in task_results:
            code = result['code']
            imports, clean_code = self._separate_imports_and_code(code)
            all_imports.extend(imports)

            if clean_code.strip():
                code_sections.append({
                    'description': result['task'].get('description', 'Code section'),
                    'code': clean_code,
                    'type': result.get('type', 'general')
                })

        # Organize imports
        organized_imports = self._organize_imports(all_imports)

        # Get assembly context
        assembly_context = await self._get_assembly_context(file_path, plan, context_cache)

        # Create structured prompt for LLM
        assembly_prompt = f"""
You are an expert Python developer. Assemble these code sections into a professional, production-ready Python file.

FILE: {file_path}
PROJECT: {plan.get('project_name', 'Project')}
DESCRIPTION: {plan.get('description', 'Generated project')}

ORGANIZED IMPORTS (use these exactly):
{organized_imports}

CODE SECTIONS TO INTEGRATE:
{self._format_code_sections(code_sections)}

CONTEXT: {assembly_context if assembly_context else "Standard Python best practices"}

ASSEMBLY REQUIREMENTS:
1. Start with a comprehensive module docstring
2. Use the organized imports exactly as provided
3. Maintain logical code flow and proper organization
4. Add appropriate spacing between major sections
5. Include proper error handling where needed
6. Add type hints for function parameters and returns
7. Include docstrings for all functions and classes
8. Follow PEP 8 standards strictly
9. Add if __name__ == "__main__": if this is a main/executable file
10. Ensure all code sections integrate seamlessly
11. Add brief comments for complex logic
12. Make the code production-ready and maintainable

IMPORTANT: Return ONLY the complete Python file code, no explanations or markdown.
"""

        # Get LLM to assemble
        response = await self.llm_client.stream_chat(assembly_prompt, LLMRole.ASSEMBLER)
        assembled_chunks = [chunk async for chunk in response]
        assembled_code = ''.join(assembled_chunks)

        return self._clean_code(assembled_code)

    def _organize_imports(self, imports: List[str]) -> str:
        """Organize imports according to PEP 8"""
        if not imports:
            return ""

        # Deduplicate and categorize
        import_set = set()
        future_imports = []
        standard_imports = []
        third_party_imports = []
        local_imports = []

        for imp in imports:
            imp = imp.strip()
            if not imp or imp in import_set:
                continue
            import_set.add(imp)

            if imp.startswith('from __future__') or imp.startswith('import __future__'):
                future_imports.append(imp)
            elif self._is_standard_library(imp):
                standard_imports.append(imp)
            elif imp.startswith('from .') or imp.startswith('import .'):
                local_imports.append(imp)
            else:
                third_party_imports.append(imp)

        # Sort each category
        future_imports.sort()
        standard_imports.sort()
        third_party_imports.sort()
        local_imports.sort()

        # Combine with proper spacing
        all_imports = []
        if future_imports:
            all_imports.extend(future_imports)
        if standard_imports:
            if all_imports:
                all_imports.append("")
            all_imports.extend(standard_imports)
        if third_party_imports:
            if all_imports:
                all_imports.append("")
            all_imports.extend(third_party_imports)
        if local_imports:
            if all_imports:
                all_imports.append("")
            all_imports.extend(local_imports)

        return "\n".join(all_imports)

    def _separate_imports_and_code(self, code: str) -> Tuple[List[str], str]:
        """Separate import statements from the rest of the code"""
        lines = code.split('\n')
        imports = []
        code_lines = []

        for line in lines:
            stripped = line.strip()
            if (stripped.startswith('import ') or
                    stripped.startswith('from ') or
                    (stripped.startswith('#') and 'import' in stripped and
                     ('from ' in stripped or 'import ' in stripped))):
                imports.append(line)
            else:
                code_lines.append(line)

        clean_code = '\n'.join(code_lines).strip()
        return imports, clean_code

    def _format_code_sections(self, code_sections: List[dict]) -> str:
        """Format code sections for the assembly prompt"""
        formatted = []
        for i, section in enumerate(code_sections, 1):
            formatted.append(f"## Section {i}: {section['description']}")
            formatted.append(f"Type: {section['type']}")
            formatted.append("```python")
            formatted.append(section['code'])
            formatted.append("```")
            formatted.append("")
        return "\n".join(formatted)

    def _is_standard_library(self, import_line: str) -> bool:
        """Check if import is from standard library"""
        standard_modules = {
            'os', 'sys', 'json', 'time', 'datetime', 'pathlib', 'typing',
            'collections', 'itertools', 'functools', 'operator', 're',
            'math', 'random', 'string', 'io', 'csv', 'urllib', 'http',
            'logging', 'argparse', 'configparser', 'sqlite3', 'threading',
            'asyncio', 'concurrent', 'multiprocessing', 'subprocess', 'ast',
            'inspect', 'copy', 'pickle', 'base64', 'hashlib', 'uuid'
        }

        # Extract module name from import statement
        if import_line.startswith('import '):
            module = import_line[7:].split('.')[0].split(' as ')[0].strip()
        elif import_line.startswith('from '):
            module = import_line[5:].split('.')[0].split(' import')[0].strip()
        else:
            return False

        return module in standard_modules

    def _contains_imports(self, code: str) -> bool:
        """Check if code contains import statements"""
        return bool(re.search(r'^(import|from)\s+\w+', code, re.MULTILINE))

    def _contains_constants(self, code: str) -> bool:
        """Check if code contains constant definitions"""
        return bool(re.search(r'^[A-Z_][A-Z0-9_]*\s*=', code, re.MULTILINE))

    def _contains_class_definitions(self, code: str) -> bool:
        """Check if code contains class definitions"""
        return bool(re.search(r'^class\s+\w+', code, re.MULTILINE))

    def _contains_function_definitions(self, code: str) -> bool:
        """Check if code contains function definitions"""
        return bool(re.search(r'^(def|async def)\s+\w+', code, re.MULTILINE))

    def _contains_main_logic(self, code: str) -> bool:
        """Check if code contains main execution logic"""
        return 'if __name__ == "__main__"' in code

    def _extract_function_names(self, code: str) -> List[str]:
        """Extract function names from code"""
        pattern = r'^(?:async\s+)?def\s+(\w+)'
        return re.findall(pattern, code, re.MULTILINE)

    def _extract_class_names(self, code: str) -> List[str]:
        """Extract class names from code"""
        pattern = r'^class\s+(\w+)'
        return re.findall(pattern, code, re.MULTILINE)

    def _is_valid_python_syntax(self, code: str) -> bool:
        """Check if code has valid Python syntax"""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _can_fix_syntax(self, code: str) -> bool:
        """Check if syntax errors can likely be fixed"""
        # Simple heuristics for fixable syntax issues
        common_fixable_issues = [
            'invalid character in identifier',
            'unexpected indent',
            'unindent does not match',
            'unexpected EOF while parsing'
        ]

        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            error_msg = str(e).lower()
            return any(issue in error_msg for issue in common_fixable_issues)

    def _fix_common_syntax_issues(self, code: str) -> str:
        """Fix common syntax issues in generated code"""
        # Remove common LLM artifacts
        code = re.sub(r'^```python\s*\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'\n```\s*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'^```\s*\n', '', code, flags=re.MULTILINE)

        # Fix indentation issues
        lines = code.split('\n')
        fixed_lines = []

        for line in lines:
            # Convert tabs to spaces
            line = line.expandtabs(4)
            # Remove trailing whitespace
            line = line.rstrip()
            fixed_lines.append(line)

        # Remove excessive blank lines
        cleaned_lines = []
        prev_blank = False
        for line in fixed_lines:
            if line.strip() == "":
                if not prev_blank:
                    cleaned_lines.append(line)
                prev_blank = True
            else:
                cleaned_lines.append(line)
                prev_blank = False

        return '\n'.join(cleaned_lines)

    def _final_validation(self, code: str) -> str:
        """Final validation and cleanup"""
        if not code.strip():
            return "# Empty file\npass"

        # Ensure proper line endings
        code = code.strip() + '\n'

        # Basic syntax check
        if not self._is_valid_python_syntax(code):
            # Try to fix
            fixed_code = self._fix_common_syntax_issues(code)
            if self._is_valid_python_syntax(fixed_code):
                code = fixed_code
            else:
                # Add pass statement if syntax is still broken
                code += "\n# Syntax validation failed - adding pass\npass\n"

        return code

    def _create_empty_file_template(self, file_path: str, plan: dict) -> str:
        """Create a basic file template when no tasks are provided"""
        project_name = plan.get('project_name', 'Project')
        description = plan.get('description', 'Generated file')

        return f'''"""
{file_path} - {description}

Generated for project: {project_name}
"""


def main():
    """Main function"""
    print("Hello from {file_path}")


if __name__ == "__main__":
    main()
'''

    async def _get_assembly_context(self, file_path: str, plan: dict, cache):
        """Get context for assembly from RAG"""
        if self.rag_manager and self.rag_manager.is_ready:
            context_query = f"python file organization {file_path} structure best practices"
            return self.rag_manager.get_context_for_code_generation(context_query, "python")
        return ""

    def _clean_code(self, code: str) -> str:
        """Extract and clean code from LLM response"""
        if not code:
            return ""

        # Remove markdown code blocks
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            parts = code.split("```")
            if len(parts) >= 3:
                code = parts[1]

        # Clean up common issues
        code = code.strip()

        # Remove any leading/trailing explanatory text
        lines = code.split('\n')
        clean_lines = []
        code_started = False

        for line in lines:
            # Skip explanatory text before code starts
            if not code_started and (line.strip().startswith('#') or
                                     line.strip().startswith('import') or
                                     line.strip().startswith('from') or
                                     line.strip().startswith('def') or
                                     line.strip().startswith('class') or
                                     line.strip().startswith('"""') or
                                     line.strip().startswith("'''")):
                code_started = True

            if code_started:
                clean_lines.append(line)

        return '\n'.join(clean_lines).strip()

    def _extract_json(self, text: str) -> dict:
        """Extract JSON from LLM response with robust parsing"""
        # Try multiple extraction methods
        methods = [
            self._extract_json_simple,
            self._extract_json_fuzzy,
            self._extract_json_fallback
        ]

        for method in methods:
            try:
                result = method(text)
                if result:
                    return result
            except:
                continue

        # Final fallback
        return {
            "approved": False,
            "feedback": "Could not parse review response",
            "overall_score": 1,
            "syntax_valid": False,
            "tasks_completed": False,
            "code_quality": 1,
            "integration_quality": 1,
            "issues": ["Review parsing failed"],
            "suggestions": [],
            "requires_fixes": ["Manual review needed"]
        }

    def _extract_json_simple(self, text: str) -> dict:
        """Simple JSON extraction"""
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = text[start:end]
            import json
            return json.loads(json_str)
        return None

    def _extract_json_fuzzy(self, text: str) -> dict:
        """Fuzzy JSON extraction - find largest valid JSON object"""
        import json
        for i in range(len(text)):
            if text[i] == '{':
                for j in range(len(text) - 1, i, -1):
                    if text[j] == '}':
                        try:
                            return json.loads(text[i:j + 1])
                        except:
                            continue
        return None

    def _extract_json_fallback(self, text: str) -> dict:
        """Fallback JSON extraction with repair attempts"""
        import json
        import re

        # Find JSON-like structure
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            try:
                return json.loads(json_str)
            except:
                # Try to fix common issues
                json_str = json_str.replace("'", '"')  # Single to double quotes
                json_str = re.sub(r'(\w+):', r'"\1":', json_str)  # Add quotes to keys
                try:
                    return json.loads(json_str)
                except:
                    pass
        return None