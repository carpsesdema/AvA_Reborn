import asyncio
import json
import hashlib  # Not used directly here, but good for context if services use it
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Callable, Any, Coroutine
from pathlib import Path
import re  # Added for JSON extraction

from core.llm_client import LLMRole


class EnhancedPlannerService:
    """🧠 Production-Ready Planning Service with Context Intelligence"""

    def __init__(self, llm_client, stream_emitter: Callable, rag_manager=None):
        self.llm_client = llm_client
        self.rag_manager = rag_manager
        self.stream_emitter = stream_emitter
        self.conversation_context = []

    def add_conversation_context(self, message: str, role: str = "user"):
        self.conversation_context.append({
            "role": role,
            "message": message,
            "timestamp": datetime.now()
        })
        if len(self.conversation_context) > 10:
            self.conversation_context.pop(0)

    async def create_project_plan(self, user_prompt: str, context_cache, full_conversation: List[Dict] = None) -> dict | \
                                                                                                                  dict[
                                                                                                                      Any, Any] | None:
        self.stream_emitter("Planner", "thought", "Okay, let's figure out the big picture for this project.", 0)
        self.stream_emitter("Planner", "thought_detail",
                            "First, I need to understand all the requirements from our chat.", 1)
        full_requirements = self._extract_full_requirements(user_prompt, full_conversation)
        self.stream_emitter("Planner", "info", f"Understood requirements: '{full_requirements[:150].strip()}...'", 1)

        self.stream_emitter("Planner", "thought_detail",
                            "Looking for any relevant code examples or patterns from the knowledge base...", 1)
        rag_context = await self._get_intelligent_rag_context(full_requirements)
        if rag_context and rag_context != "No RAG context available - use Python best practices" and rag_context != "No specific examples found - use standard Python GUI patterns":
            self.stream_emitter("Planner", "info", f"Found some helpful context (length: {len(rag_context)} chars).", 1)
        else:
            self.stream_emitter("Planner", "info",
                                "No specific examples found in knowledge base, proceeding with general best practices.",
                                1)

        self.stream_emitter("Planner", "thought_detail", "Determining the best project type...", 1)
        project_type = self._detect_project_type(full_requirements)
        self.stream_emitter("Planner", "info", f"This looks like a '{project_type}' project.", 1)

        plan_prompt = f"""
You are a Senior Software Architect creating a COMPLETE working project.

FULL REQUIREMENTS ANALYSIS:
{full_requirements}

PROJECT TYPE DETECTED: {project_type}

RELEVANT CODE EXAMPLES (if any):
{rag_context}

Create a comprehensive JSON plan for a FULLY FUNCTIONAL application.
The plan should include: "project_name", "description", "architecture_type", "main_requirements" (list), "files" (dictionary where keys are filenames and values are objects with "priority", "description", "size_estimate", "key_features" list), "dependencies" (list), and "execution_notes".

Example for a calculator:
{{
    "project_name": "PySideCalculator",
    "description": "A desktop calculator application using PySide6.",
    "architecture_type": "gui_calculator",
    "main_requirements": ["Basic arithmetic (+, -, *, /)", "Clear and Equals buttons", "Display for input/output"],
    "files": {{
        "main.py": {{ "priority": 1, "description": "Main application window, GUI layout, and event handling.", "size_estimate": "100-150 lines", "key_features": ["QMainWindow setup", "Button grid", "Display QLineEdit", "Signal/slot connections"] }},
        "calculator_logic.py": {{ "priority": 2, "description": "Core calculation logic, separated for clarity.", "size_estimate": "40-60 lines", "key_features": ["perform_calculation function", "Error handling for division by zero"] }}
    }},
    "dependencies": ["PySide6"],
    "execution_notes": "Run with 'python main.py'"
}}

CRITICAL: This must be a COMPLETE, WORKING application, not just stubs. Ensure the "files" dictionary values are objects with details as shown.
"""
        self.stream_emitter("Planner", "thought",
                            "Now, I'll ask the AI to draft the project structure and file breakdown...", 1)
        try:
            response_chunks = []
            stream_generator = None
            try:
                stream_generator = self.llm_client.stream_chat(plan_prompt, LLMRole.PLANNER)
                self.stream_emitter("Planner", "status", "Waiting for AI to generate the plan...", 2)
                idx = 0
                temp_plan_json = ""
                async for chunk in stream_generator:
                    response_chunks.append(chunk)
                    temp_plan_json += chunk
                    if idx % 5 == 0:
                        self.stream_emitter("Planner", "llm_chunk", f"Receiving plan details... (chunk {idx + 1})", 2)
                    idx += 1
                    if idx > 300:  # Safety break for very long streams
                        self.stream_emitter("Planner", "warning", "Plan generation is very long, might truncate.", 2)
                        break
                    await asyncio.sleep(0.005)  # Small sleep to allow UI to update if needed
            finally:
                if stream_generator and hasattr(stream_generator, 'aclose'):
                    try:
                        await stream_generator.aclose()
                    except Exception:
                        pass

            response_text = ''.join(response_chunks)
            self.stream_emitter("Planner", "thought_detail",
                                f"AI's draft plan received (raw length: {len(response_text)} chars). Now, let's make sense of it.",
                                1)
            plan = self._extract_json(response_text)

            if not isinstance(plan, dict) or 'files' not in plan or not isinstance(plan['files'], dict):
                self.stream_emitter("Planner", "error",
                                    "The AI's plan isn't structured correctly. I'll create a solid fallback plan.", 1)
                return self._create_intelligent_fallback_plan(full_requirements, project_type)

            self.stream_emitter("Planner", "success", "Project plan drafted and looks good!", 0)
            return plan

        except Exception as e:
            self.stream_emitter("Planner", "error",
                                f"Couldn't get a good plan from the AI ({e}). Using a reliable fallback.", 0)
            return self._create_intelligent_fallback_plan(full_requirements, project_type)

    def _extract_full_requirements(self, current_prompt: str, conversation: List[Dict] = None) -> str:
        # ... (no changes to internal logic, just ensure no direct stream_emitter calls from here) ...
        requirements = []
        if conversation:
            for msg in conversation[-5:]:
                if msg.get("role") == "user":
                    requirements.append(msg.get("message", ""))
        for ctx in self.conversation_context[-3:]:
            if ctx.get("role") == "user":
                requirements.append(ctx.get("message", ""))
        requirements.append(current_prompt)
        full_text = " ".join(req.strip() for req in requirements if req.strip())
        # Simple summary for logging, actual prompt for LLM will have full text
        return f"Combined user inputs: {full_text}"

    def _detect_project_type(self, requirements: str) -> str:
        # ... (no changes needed) ...
        req_lower = requirements.lower()
        if any(word in req_lower for word in
               ["calculator", "pyside", "gui", "interface", "desktop app"]): return "gui_calculator"
        if any(word in req_lower for word in ["web", "api", "flask", "fastapi", "django", "server"]): return "web_app"
        if any(word in req_lower for word in ["cli", "command", "terminal", "script"]): return "cli_app"
        return "desktop_app"  # Default if less specific

    async def _get_intelligent_rag_context(self, requirements: str) -> str:
        # ... (no direct stream_emitter, assumes RAG manager might log if verbose) ...
        if not self.rag_manager or not self.rag_manager.is_ready:
            return "No RAG context available - use Python best practices"
        project_type = self._detect_project_type(requirements)
        queries = []
        if project_type == "gui_calculator":
            queries = [
                "PySide6 calculator complete example", "Python GUI calculator arithmetic logic",
                "PySide6 QGridLayout button interface"
            ]
        elif project_type == "web_app":
            queries = ["Flask simple API example", "FastAPI route definition python"]
        # Add more queries as needed

        if not queries: queries.append(f"Python {project_type} best practices code examples")

        context_parts = []
        for i, query in enumerate(queries[:2]):  # Limit RAG queries for brevity in planning
            try:
                results = self.rag_manager.query_context(query, k=1)
                for result in results:
                    content = result.get('content', '')
                    if len(content) > 100:
                        context_parts.append(f"# Relevant Example ({query[:40]}...):\n{content[:400]}\n...\n")
            except Exception as e:
                pass  # Silently fail RAG for planning stage if needed, rely on LLM's general knowledge

        return "\n\n".join(
            context_parts) if context_parts else "No specific examples found - use standard Python patterns"

    async def create_micro_tasks(self, file_path: str, plan: dict, context_cache) -> list[dict] | list | None:
        self.stream_emitter("Planner", "thought",
                            f"Now, breaking down '{file_path}' into smaller, manageable coding tasks.", 1)
        files_info = plan.get('files', {})
        file_info = files_info.get(file_path, {})
        project_name = plan.get('project_name', 'project')
        project_type = plan.get('architecture_type', 'desktop_app')
        file_description = file_info.get('description', 'Application file')
        key_features_list = file_info.get('key_features', ['Core functionality'])
        key_features_str = ", ".join(key_features_list)

        self.stream_emitter("Planner", "thought_detail",
                            f"Checking knowledge base for patterns related to '{file_path}' ({project_type}, features: {key_features_str[:50]}...).",
                            2)
        file_context = await self._get_file_specific_context(file_path, project_type, key_features_list, context_cache)
        if file_context and not file_context.startswith("Standard"):
            self.stream_emitter("Planner", "info", f"Found relevant patterns for '{file_path}'.", 2)
        else:
            self.stream_emitter("Planner", "info", f"Using general best practices for '{file_path}'.", 2)

        task_prompt = f"""
You are a senior software architect. For the file '{file_path}' in a '{project_type}' project named '{project_name}', break down its implementation into 3-5 logical, detailed, and executable micro-tasks.
File Description: {file_description}
Key Features for this file: {key_features_str}

Consider this relevant context/examples if available:
{file_context}

Respond with a JSON list of tasks. Each task object should have: "id" (snake_case, e.g., "initialize_gui_elements"), "type" (e.g., "imports", "class_definition", "method_implementation", "ui_layout", "event_handler", "main_logic"), "description" (clear, natural language of what to implement for THIS task), "priority" (integer), "expected_lines" (estimated int), and "specific_requirements" (list of precise, actionable strings for the Coder).

Example for a calculator's main.py:
[
    {{ "id": "imports_and_setup", "type": "imports", "description": "Import PySide6 modules, sys, and any custom logic modules. Define the main application class structure.", "priority": 1, "expected_lines": 15, "specific_requirements": ["Import QApplication, QMainWindow, etc. from PySide6.QtWidgets.", "Import sys.", "Define 'CalculatorApp(QMainWindow):' with an __init__ method."] }},
    {{ "id": "gui_layout", "type": "ui_layout", "description": "Create the calculator's display (QLineEdit) and button grid (QGridLayout) within the main window.", "priority": 2, "expected_lines": 40, "specific_requirements": ["Instantiate QLineEdit for display, set it to read-only.", "Create QPushButton instances for numbers 0-9, operators (+, -, *, /), Clear, and Equals.", "Arrange buttons in a QGridLayout."] }},
    {{ "id": "event_handling", "type": "event_handler", "description": "Implement methods to handle button clicks: appending to display, performing calculations, and clearing.", "priority": 3, "expected_lines": 50, "specific_requirements": ["Method for number/operator clicks to update display text.", "Method for '=' click to evaluate expression from display (use a safe eval or custom parser from calculator_logic.py).", "Method for 'Clear' button to reset display and internal state."] }},
    {{ "id": "main_execution_block", "type": "main_logic", "description": "Standard Python main execution block to create QApplication, instantiate CalculatorApp, show window, and start event loop.", "priority": 4, "expected_lines": 10, "specific_requirements": ["if __name__ == '__main__':", "app = QApplication(sys.argv)", "window = CalculatorApp()", "window.show()", "sys.exit(app.exec())"] }}
]
Ensure each task is a logical, buildable unit. The sum of tasks should result in a complete, working file.
"""
        self.stream_emitter("Planner", "thought", f"Asking AI to detail micro-tasks for '{file_path}'...", 1)
        try:
            response_chunks = []
            stream_generator = None
            temp_task_json = ""
            try:
                stream_generator = self.llm_client.stream_chat(task_prompt, LLMRole.PLANNER)
                self.stream_emitter("Planner", "status", f"Waiting for AI to generate micro-tasks for '{file_path}'...",
                                    2)
                idx = 0
                async for chunk in stream_generator:
                    response_chunks.append(chunk)
                    temp_task_json += chunk
                    if idx % 5 == 0:
                        self.stream_emitter("Planner", "llm_chunk",
                                            f"Receiving task details for '{file_path}' (chunk {idx + 1})...", 2)
                    idx += 1
                    if idx > 300: break  # Safety break
                    await asyncio.sleep(0.005)
            finally:
                if stream_generator and hasattr(stream_generator, 'aclose'):
                    try:
                        await stream_generator.aclose()
                    except Exception:
                        pass

            response_text = ''.join(response_chunks)
            self.stream_emitter("Planner", "thought_detail",
                                f"AI's micro-task draft received (raw length: {len(response_text)}). Validating structure...",
                                2)
            tasks = self._extract_json(response_text)

            if not isinstance(tasks, list) or not tasks:  # Check if tasks is a non-empty list
                self.stream_emitter("Planner", "error",
                                    f"AI returned invalid micro-task structure for '{file_path}'. Using fallback.", 2)
                return self._create_fallback_tasks(file_path, project_type, file_description)

            for task_idx, task in enumerate(tasks):  # Ensure essential keys are present
                if not all(k in task for k in ["id", "description"]):
                    self.stream_emitter("Planner", "warning",
                                        f"Task {task_idx} for '{file_path}' is missing 'id' or 'description'. Attempting to use anyway.",
                                        2)
                task["file_path"] = file_path  # Ensure these are added
                task["project_type"] = project_type

            self.stream_emitter("Planner", "success", f"Micro-tasks for '{file_path}' are defined: {len(tasks)} tasks.",
                                1)
            return tasks

        except Exception as e:
            self.stream_emitter("Planner", "error",
                                f"Micro-task creation for '{file_path}' failed ({e}). Using fallback.", 1)
            return self._create_fallback_tasks(file_path, project_type, file_description)

    async def _get_file_specific_context(self, file_path: str, project_type: str, key_features: List[str],
                                         cache) -> str:
        # ... (no direct stream_emitter, RAG manager might log) ...
        if not self.rag_manager or not self.rag_manager.is_ready:
            return f"Standard {project_type} implementation patterns."

        # Create a more targeted query
        feature_query_part = ", ".join(key_features[:2])  # Use first 2 key features for query
        query = f"{project_type} file '{Path(file_path).name}' implementing {feature_query_part} code patterns and examples"

        try:
            results = self.rag_manager.query_context(query, k=1)
            if results and results[0].get('content'):
                return f"Found relevant example for '{Path(file_path).name}':\n{results[0]['content'][:500]}\n..."
        except Exception as e:
            pass  # Silently fail for this context retrieval
        return f"Standard {project_type} implementation patterns for file {Path(file_path).name}."

    def _create_intelligent_fallback_plan(self, requirements: str, project_type: str) -> dict:
        # ... (no changes needed, stream_emitter call already in invoking method) ...
        if project_type == "gui_calculator":
            return {
                "project_name": "pyside6_calculator_fb", "description": "Fallback: PySide6 Calculator",
                "architecture_type": "gui_calculator", "main_requirements": ["GUI", "Math ops"],
                "files": {"main.py": {"priority": 1, "description": "Complete calculator GUI and logic",
                                      "size_estimate": "120 lines",
                                      "key_features": ["GUI Layout", "Event Handling", "Calculation Logic"]}},
                "dependencies": ["PySide6"], "execution_notes": "Run main.py"
            }
        return {
            "project_name": "generated_app_fb", "description": requirements[:100],
            "architecture_type": project_type,
            "files": {
                "main.py": {"priority": 1, "description": "Main application file", "key_features": ["Core logic"]}},
            "dependencies": [], "execution_notes": "Run main.py"
        }

    def _create_fallback_tasks(self, file_path: str, project_type: str,
                               file_description: str = "Core file functionality") -> List[dict]:
        # ... (no changes needed, stream_emitter call already in invoking method) ...
        if project_type == "gui_calculator" and "main.py" in file_path.lower():  # More specific fallback for calculator main
            return [{"id": "complete_calculator_gui_logic_fb", "type": "complete_file",
                     "description": f"Implement the complete PySide6 calculator GUI, event handlers, and calculation logic in '{file_path}'.",
                     "file_path": file_path, "project_type": project_type,
                     "specific_requirements": ["Full GUI as described in plan.",
                                               "All arithmetic operations integrated.",
                                               "Error display for invalid operations."]}]
        return [{"id": f"implement_{Path(file_path).stem}_full_fb", "type": "complete_file",
                 "description": f"Implement all described functionality for '{file_path}': {file_description}",
                 "file_path": file_path, "project_type": project_type,
                 "specific_requirements": [f"Ensure '{file_path}' is fully functional based on project requirements."]}]

    def _extract_json(self, text: str):
        # ... (no changes needed, stream_emitter calls already in method) ...
        # Try to find JSON within markdown ```json ... ```
        match_md = re.search(r'```json\s*([\s\S]+?)\s*```', text)
        if match_md:
            json_text = match_md.group(1)
            try:
                return json.loads(json_text)
            except json.JSONDecodeError as e:
                self.stream_emitter("Planner", "warning",
                                    f"Found JSON in markdown, but failed to parse: {e}. Content: {json_text[:100]}...",
                                    3)
                # Fall through to try other methods

        # Try to find the largest JSON object or array
        # This regex tries to match balanced braces/brackets
        json_pattern = r'(\{[\s\S]*\}|\[[\s\S]*\])'

        # Find all potential JSON objects/arrays
        potential_jsons = []
        for match in re.finditer(json_pattern, text):
            try:
                # Test if it's valid JSON
                json.loads(match.group(0))
                potential_jsons.append(match.group(0))
            except json.JSONDecodeError:
                continue

        if potential_jsons:
            # Return the largest valid JSON found
            largest_json = max(potential_jsons, key=len)
            try:
                return json.loads(largest_json)
            except json.JSONDecodeError as e:  # Should not happen if logic above is correct
                self.stream_emitter("Planner", "error",
                                    f"Error parsing even the 'largest_json': {e}. Content: {largest_json[:100]}...", 3)

        self.stream_emitter("Planner", "error", f"Failed to extract any valid JSON. Raw text: {text[:200]}...", 3)
        return {}  # Return empty dict or list based on expectation if strict or raise error


class EnhancedCoderService:
    """⚙️ Production-Ready Coding Service with Intelligent Generation"""

    def __init__(self, llm_client, stream_emitter: Callable, rag_manager=None):
        self.llm_client = llm_client
        self.rag_manager = rag_manager
        self.stream_emitter = stream_emitter

    async def execute_micro_task(self, task: dict, file_context: str, context_cache) -> str | None:
        task_id = task.get('id', 'unknown_task')
        task_desc = task.get('description', 'N/A')
        self.stream_emitter("Coder", "thought", f"Alright, focusing on task '{task_id}': {task_desc[:60]}...", 1)

        self.stream_emitter("Coder", "thought_detail",
                            "Let me see if I have any good code snippets or patterns for this...", 2)
        task_examples = await self._get_task_specific_examples(task)
        if task_examples and not task_examples.startswith("Use Python best"):
            self.stream_emitter("Coder", "info", "Found some relevant examples to guide my coding.", 2)
        else:
            self.stream_emitter("Coder", "info", "No specific examples, I'll rely on my general coding knowledge.", 2)

        code_prompt = f"""
You are an expert Python developer. Your task is to generate a single, complete, and production-ready Python code snippet for the micro-task described below.

FILE CONTEXT: This snippet will be part of '{task.get('file_path', 'N/A')}' for a '{task.get('project_type', 'application')}' project.
TASK ID: {task_id}
TASK DESCRIPTION: {task_desc}
SPECIFIC REQUIREMENTS FOR THIS TASK:
{chr(10).join('- ' + req for req in task.get('specific_requirements', ['Implement the described functionality fully and correctly.']))}

RELEVANT EXAMPLES (use as inspiration for patterns, not direct copy):
{task_examples}

CRITICAL INSTRUCTIONS:
1.  Generate ONLY the Python code required to fulfill THIS SPECIFIC micro-task.
2.  If the task is about 'imports', provide only the import statements.
3.  If it's a 'class_definition', provide the class shell with `pass` in methods if their implementation is a separate task.
4.  If it's a 'method_implementation', provide the complete method body.
5.  Ensure the code is clean, efficient, well-commented (docstrings for public elements, comments for complex logic), and adheres to PEP 8.
6.  Make the code robust: include error handling (e.g., try-except blocks) where appropriate for the task.
7.  Do NOT generate a full file or surrounding code unless the task explicitly states "complete_file".
8.  Do NOT include placeholders like "# TODO" or "pass" if the task implies full implementation of that part. Use "pass" only if the task is to define a structure (like a class shell) and other tasks will fill it.

Generate ONLY the Python code snippet for this task:
"""
        self.stream_emitter("Coder", "thought",
                            f"Okay, I'm ready to write the code for '{task_id}'. Sending instructions to the AI code generator...",
                            1)
        try:
            raw_code_stream = []
            stream_generator = None
            accumulated_code = ""
            last_emit_time = 0

            try:
                stream_generator = self.llm_client.stream_chat(code_prompt, LLMRole.CODER)
                self.stream_emitter("Coder", "status", "AI is now generating code...", 2)

                async for chunk in stream_generator:
                    raw_code_stream.append(chunk)
                    accumulated_code += chunk

                    # Only emit code in meaningful chunks (lines or logical blocks)
                    current_time = asyncio.get_event_loop().time()
                    if ('\n' in chunk or len(accumulated_code) > 100 or
                            current_time - last_emit_time > 0.5):  # Every half second max

                        # Find complete lines to show
                        lines = accumulated_code.split('\n')
                        if len(lines) > 1:
                            # Show complete lines, keep the last incomplete one
                            complete_lines = '\n'.join(lines[:-1])
                            if complete_lines.strip():
                                self.stream_emitter("Coder", "code_chunk", complete_lines, 2)
                                accumulated_code = lines[-1]  # Keep the incomplete line
                                last_emit_time = current_time

                    await asyncio.sleep(0.001)

                # Emit any remaining code
                if accumulated_code.strip():
                    self.stream_emitter("Coder", "code_chunk", accumulated_code, 2)

            finally:
                if stream_generator and hasattr(stream_generator, 'aclose'):
                    try:
                        await stream_generator.aclose()
                    except Exception:
                        pass

            raw_code = ''.join(raw_code_stream)
            self.stream_emitter("Coder", "thought_detail",
                                f"AI finished generating code (raw length: {len(raw_code)}). Now cleaning it up.", 2)
            cleaned_code = self._clean_and_validate_code(raw_code, task)

            if not cleaned_code.strip() and not task.get('type',
                                                         '') == 'imports':  # Empty imports are fine if nothing to import
                self.stream_emitter("Coder", "warning",
                                    f"Task '{task_id}' resulted in empty code after cleaning. This might be an issue. Using fallback.",
                                    2)
                return self._create_emergency_fallback(task)

            self.stream_emitter("Coder", "success",
                                f"Code for task '{task_id}' is ready (final length: {len(cleaned_code)}).", 1)
            return cleaned_code

        except Exception as e:
            self.stream_emitter("Coder", "error",
                                f"Oops, something went wrong while coding task '{task_id}' ({e}). I'll put a placeholder.",
                                1)
            return self._create_emergency_fallback(task)

    async def _get_task_specific_examples(self, task: dict) -> str:
        # ... (no direct stream_emitter, RAG logs if any) ...
        if not self.rag_manager or not self.rag_manager.is_ready:
            return "Use Python best practices and standard patterns."

        task_type = task.get('type', 'general')
        project_type = task.get('project_type', 'application')
        file_path_str = task.get('file_path', 'unknown_file.py')
        task_desc = task.get('description', 'generic task')

        query = f"Python code example for {task_type} in a {project_type} context, specifically for '{task_desc[:50]}' in file '{Path(file_path_str).name}'"

        context_parts = []
        try:
            results = self.rag_manager.query_context(query, k=1)  # Get one best example
            if results and results[0].get('content'):
                content = results[0].get('content')
                context_parts.append(
                    f"Relevant Example Snippet (for task type '{task_type}'):\n```python\n{content[:400]}\n```\n...\n")
        except Exception as e:
            pass  # Silently ignore RAG errors for individual task examples

        return "\n\n".join(
            context_parts) if context_parts else f"Standard Python patterns for implementing a {task_type}."

    def _clean_and_validate_code(self, code: str, task: dict) -> str:
        # ... (no stream_emitter here, it's an internal helper) ...
        # More robust cleaning for various markdown code block styles
        if code.strip().startswith("```python"):
            code = code.split("```python", 1)[-1]
            if code.strip().endswith("```"):
                code = code.rsplit("```", 1)[0]
        elif code.strip().startswith("```"):
            code = code.split("```", 1)[-1]
            if code.strip().endswith("```"):
                code = code.rsplit("```", 1)[0]

        cleaned = code.strip()

        # If task type is 'imports' and it's empty, that's okay
        if task.get('type', '') == 'imports' and not cleaned:
            return ""

        # Basic validation: if not an import task and still empty, that's a problem.
        if not cleaned and task.get('type', '') != 'imports':
            # This warning will be seen if the calling method logs it
            # print(f"Warning: Cleaned code for task '{task.get('id','N/A')}' is empty.")
            return self._create_emergency_fallback(task)

        return cleaned

    def _create_emergency_fallback(self, task: dict) -> str:
        # ... (no stream_emitter, it's a fallback content generator) ...
        task_id = task.get('id', 'unknown_task')
        task_desc = task.get('description', 'Implement task')
        return f"# FALLBACK: Task '{task_id}' - {task_desc}\n# Please implement this part manually or re-run.\npass"

    async def execute_tasks_parallel(self, tasks: List[dict], file_context: str,
                                     context_cache, progress_callback=None) -> List[dict]:
        results = []
        if not isinstance(tasks, list):
            self.stream_emitter("Coder", "error", f"Invalid tasks format provided: {type(tasks)}", 0)
            return []

        total_tasks = len(tasks)
        self.stream_emitter("Coder", "status", f"Preparing to code {total_tasks} micro-tasks.", 0)

        for i, task in enumerate(tasks):
            task_id = task.get('id', f'task_{i + 1}')
            self.stream_emitter("Coder", "status", f"Starting work on task {i + 1}/{total_tasks}: '{task_id}'.", 0)

            # The progress_callback is for the overall workflow engine's view of task completion
            if progress_callback:
                progress_callback(f"Coding task {i + 1}/{total_tasks}: {task_id}")

            try:
                code = await self.execute_micro_task(task, file_context, context_cache)
                results.append({"task": task, "code": code})
                self.stream_emitter("Coder", "success", f"Finished coding task '{task_id}'.", 0)
            except Exception as e:
                self.stream_emitter("Coder", "error", f"Critical error during task '{task_id}': {e}", 0)
                results.append({"task": task, "code": f"# CRITICAL CODER ERROR for task {task_id}: {e}\npass"})

        self.stream_emitter("Coder", "status", f"All {total_tasks} micro-tasks for this file have been processed.", 0)
        return results


class EnhancedAssemblerService:
    """🔧 Production-Ready Code Assembly Service with Review"""

    def __init__(self, llm_client, stream_emitter: Callable, rag_manager=None):
        self.llm_client = llm_client
        self.rag_manager = rag_manager
        self.stream_emitter = stream_emitter

    async def assemble_file(self, file_path: str, task_results: List[dict], plan: dict, context_cache) -> Tuple[
        str, bool, str]:
        """Assemble code chunks into a complete file with AI review"""

        self.stream_emitter("Assembler", "thought", f"Time to put the pieces together for '{file_path}'...", 1)

        # Extract and validate code from task results
        code_chunks = []
        for result in task_results:
            task = result.get("task", {})
            code = result.get("code", "")
            task_id = task.get("id", "unknown")

            if code and code.strip():
                self.stream_emitter("Assembler", "info", f"Adding code from task '{task_id}' ({len(code)} chars)", 2)
                # Clean separator for better readability
                code_chunks.append(f"# {'-' * 20} {task_id} {'-' * 20}\n{code}\n")
            else:
                self.stream_emitter("Assembler", "warning", f"Task '{task_id}' produced no code - skipping", 2)

        if not code_chunks:
            self.stream_emitter("Assembler", "error", f"No code chunks to assemble for '{file_path}'!", 1)
            return self._create_emergency_file(file_path, plan), False, "No code chunks available for assembly"

        # Assemble the raw code
        self.stream_emitter("Assembler", "thought", "Combining all the code chunks into a cohesive file...", 1)
        raw_assembled = "\n".join(code_chunks)

        # Clean and organize the assembled code
        self.stream_emitter("Assembler", "thought_detail", "Cleaning up the assembled code and organizing structure...",
                            2)
        cleaned_code = self._clean_and_organize_code(raw_assembled, file_path)

        # AI Review Phase
        self.stream_emitter("Assembler", "thought", f"Now asking the AI Reviewer to check '{file_path}' for quality...",
                            1)
        review_approved, review_feedback = await self._ai_review_code(file_path, cleaned_code, plan)

        if review_approved:
            self.stream_emitter("Assembler", "success", f"✅ Assembly complete for '{file_path}' - Review: APPROVED", 1)
        else:
            self.stream_emitter("Assembler", "warning",
                                f"⚠️ Assembly complete for '{file_path}' - Review: NEEDS ATTENTION", 1)
            self.stream_emitter("Assembler", "info", f"Review feedback: {review_feedback[:100]}...", 2)

        return cleaned_code, review_approved, review_feedback

    def _clean_and_organize_code(self, raw_code: str, file_path: str) -> str:
        """Clean and organize the assembled code"""

        self.stream_emitter("Assembler", "thought_detail", "Organizing imports, removing duplicates, and formatting...",
                            2)

        # Split into lines for processing
        lines = raw_code.split('\n')

        # Organize sections
        imports = []
        docstring_lines = []
        class_defs = []
        function_defs = []
        main_block = []
        other_code = []

        current_section = "other"
        in_main_block = False

        for line in lines:
            stripped = line.strip()

            # Skip separator comments
            if stripped.startswith("# ---") and "---" in stripped:
                continue

            # Detect imports
            if stripped.startswith(('import ', 'from ')):
                if line not in imports:  # Avoid duplicates
                    imports.append(line)
                continue

            # Detect main block
            if 'if __name__ == "__main__"' in stripped or 'if __name__ == \'__main__\'' in stripped:
                in_main_block = True
                main_block.append(line)
                continue

            if in_main_block:
                main_block.append(line)
                continue

            # Detect class definitions
            if stripped.startswith('class '):
                current_section = "class"
                class_defs.append(line)
                continue

            # Detect function definitions
            if stripped.startswith('def '):
                current_section = "function"
                function_defs.append(line)
                continue

            # Add to current section
            if current_section == "class":
                class_defs.append(line)
            elif current_section == "function":
                function_defs.append(line)
            else:
                other_code.append(line)

        # Reconstruct the file in proper order
        final_lines = []

        # Add file header comment
        if file_path:
            final_lines.append(f'"""')
            final_lines.append(f'{file_path}')
            final_lines.append(f'Generated by AvA on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            final_lines.append(f'"""')
            final_lines.append('')

        # Add imports (sorted and deduplicated)
        if imports:
            unique_imports = list(dict.fromkeys(imports))  # Remove duplicates while preserving order
            final_lines.extend(unique_imports)
            final_lines.append('')

        # Add other code (constants, etc.)
        if other_code:
            final_lines.extend(other_code)
            final_lines.append('')

        # Add classes
        if class_defs:
            final_lines.extend(class_defs)
            final_lines.append('')

        # Add functions
        if function_defs:
            final_lines.extend(function_defs)
            final_lines.append('')

        # Add main block
        if main_block:
            final_lines.extend(main_block)

        return '\n'.join(final_lines)

    async def _ai_review_code(self, file_path: str, code: str, plan: dict) -> Tuple[bool, str]:
        """Get AI review of the assembled code"""

        self.stream_emitter("Assembler", "thought_detail", "Preparing detailed review prompt for the AI...", 2)

        review_prompt = f"""
You are a senior code reviewer examining this Python file for quality and correctness.

FILE: {file_path}
PROJECT CONTEXT: {plan.get('description', 'Python application')}

CODE TO REVIEW:
```python
{code}
```
Provide a thorough review focusing on:

Code functionality and correctness
Python best practices and PEP 8 compliance
Error handling and edge cases
Code organization and readability
Security considerations
Respond with JSON:
{{
"approved": true/false,
"overall_quality": "excellent|good|fair|poor",
"critical_issues": ["issue1", "issue2"],
"suggestions": ["suggestion1", "suggestion2"],
"summary": "Brief overall assessment"
}}
Be thorough but practical. Minor style issues shouldn't fail approval.
"""
        try:
            self.stream_emitter("Assembler", "status", "AI Reviewer is analyzing the code...", 2)

            # Get AI review using the reviewer role
            review_response = ""
            stream_generator = None
            try:
                stream_generator = self.llm_client.stream_chat(review_prompt, LLMRole.REVIEWER)
                response_chunks = []
                idx = 0
                async for chunk in stream_generator:
                    response_chunks.append(chunk)
                    if idx % 10 == 0:
                        self.stream_emitter("Assembler", "llm_chunk", f"Receiving review analysis... (chunk {idx + 1})",
                                            3)
                    idx += 1
                    if idx > 200: break  # Safety break for reviews
                    await asyncio.sleep(0.001)
                review_response = ''.join(response_chunks)
            finally:
                if stream_generator and hasattr(stream_generator, 'aclose'):
                    try:
                        await stream_generator.aclose()
                    except Exception:
                        pass

            self.stream_emitter("Assembler", "thought_detail",
                                f"AI review received ({len(review_response)} chars). Parsing results...", 2)

            # Parse the JSON response
            try:
                # Extract JSON from response
                start_idx = review_response.find('{')
                end_idx = review_response.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = review_response[start_idx:end_idx]
                    review_data = json.loads(json_str)

                    approved = review_data.get('approved', False)
                    summary = review_data.get('summary', 'AI review completed')
                    critical_issues = review_data.get('critical_issues', [])

                    if approved:
                        self.stream_emitter("Assembler", "success",
                                            f"AI Reviewer: Code quality is {review_data.get('overall_quality', 'good')}",
                                            2)
                    else:
                        self.stream_emitter("Assembler", "warning",
                                            f"AI Reviewer found {len(critical_issues)} critical issues", 2)
                        for issue in critical_issues[:3]:  # Show first 3 issues
                            self.stream_emitter("Assembler", "info", f"Issue: {issue}", 3)

                    return approved, summary

                else:
                    self.stream_emitter("Assembler", "warning", "Could not parse AI review JSON - using fallback", 2)

            except json.JSONDecodeError as e:
                self.stream_emitter("Assembler", "warning", f"AI review JSON parsing failed: {e}", 2)

        except Exception as e:
            self.stream_emitter("Assembler", "warning", f"AI review failed ({e}) - proceeding with basic validation", 2)

        # Fallback: Basic validation
        approved = len(code.strip()) > 50 and ('def ' in code or 'class ' in code)
        feedback = "Basic validation: " + ("Code structure looks reasonable" if approved else "Code seems incomplete")

        return approved, feedback

    def _create_emergency_file(self, file_path: str, plan: dict) -> str:
        """Create emergency fallback file when assembly fails"""

        self.stream_emitter("Assembler", "warning", f"Creating emergency fallback for '{file_path}'", 2)

        project_name = plan.get('project_name', 'Generated Project')
        description = plan.get('description', 'Python application')

        return f'''"""
{file_path}
Emergency fallback file for {project_name}
{description}
Generated by AvA on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
def main():
    """Main function - implement your logic here"""
    print("Emergency fallback - please implement the required functionality")
    # TODO: Add proper implementation
if __name__ == "__main__":
    main()
'''


PlannerService = EnhancedPlannerService
CoderService = EnhancedCoderService
AssemblerService = EnhancedAssemblerService


class WorkflowOrchestrator:
    """🎼 Enhanced Orchestrator with Conversation Intelligence"""

    def __init__(self, planner_service: EnhancedPlannerService, coder_service: EnhancedCoderService,
                 assembler_service: EnhancedAssemblerService,
                 terminal_emitter: Callable = None):
        self.planner = planner_service
        self.coder = coder_service
        self.assembler = assembler_service
        self.emit_stream_log = terminal_emitter if terminal_emitter else lambda agent, type_key, content, indent: print(
            f"[{agent}] {type_key} (indent {indent}): {content}")
        # Allow ContextCache to be properly imported or defined
        try:
            from core.enhanced_workflow_engine import ContextCache
        except ImportError:
            class ContextCache:  # Dummy if not found, services should handle None
                def __init__(self): self.cache = {}

                def get(self, key): return self.cache.get(key)

                def set(self, key, value): self.cache[key] = value
        self.context_cache = ContextCache()

    def set_conversation_context(self, conversation_history: List[Dict]):
        if hasattr(self.planner, 'add_conversation_context'):
            for msg in conversation_history:  # Only add user messages for planner's direct context
                if msg.get("role") == "user":
                    self.planner.add_conversation_context(msg.get("message", ""), msg.get("role", "user"))
        else:
            self.emit_stream_log("Orchestrator", "warning", "Planner service does not support conversation context.", 0)

    async def execute_workflow(self, user_prompt: str, output_dir: Path,
                               conversation_history: List[Dict] = None) -> dict:
        try:
            self.emit_stream_log("Orchestrator", "stage_start", "🚀 Kicking off the AvA Super-Workflow!", 0)
            if conversation_history and hasattr(self.planner, 'add_conversation_context'):
                self.set_conversation_context(conversation_history)  # This will pass only user messages
            self.emit_stream_log("Orchestrator", "thought", "Asking the Planner to draft the grand design...", 1)
            # Pass the full conversation history to the planner for its own analysis
            plan = await self.planner.create_project_plan(user_prompt, self.context_cache, conversation_history)
            if not isinstance(plan, dict) or 'files' not in plan or not isinstance(plan['files'], dict):
                self.emit_stream_log("Orchestrator", "error",
                                     "The Planner's blueprint seems incomplete. Can't proceed.", 0)
                return {"success": False, "error": "Planning phase failed: Received an invalid plan structure."}

            project_name = plan.get("project_name", "ava_super_project")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            project_dir = output_dir / f"{project_name}_{timestamp}"
            project_dir.mkdir(exist_ok=True, parents=True)
            self.emit_stream_log("Orchestrator", "file_op", f"Set up new project folder: {project_dir}", 1)

            files_to_process = plan.get("files", {})
            self.emit_stream_log("Orchestrator", "status",
                                 f"Planner has outlined {len(files_to_process)} files. Let's get to it!", 1)

            generated_files_details = []
            for file_idx, (file_path_str, file_info_dict) in enumerate(files_to_process.items()):
                self.emit_stream_log("Orchestrator", "stage_start",
                                     f" 집중! File {file_idx + 1}/{len(files_to_process)}: '{file_path_str}'",
                                     1)  # Korean for "Focus!"
                try:
                    result = await self._process_file_intelligently(file_path_str, file_info_dict, plan, project_dir)
                    generated_files_details.append(result)
                    if result.get("success"):
                        self.emit_stream_log("Orchestrator", "success",
                                             f"File '{file_path_str}' built successfully! Review: {'Approved' if result.get('review_approved') else 'Needs attention'}",
                                             2)
                    else:
                        self.emit_stream_log("Orchestrator", "warning",
                                             f"File '{file_path_str}' had some hiccups: {result.get('error', 'Unknown issue')}",
                                             2)
                except Exception as e:
                    self.emit_stream_log("Orchestrator", "error",
                                         f"Critical error while processing '{file_path_str}': {e}", 2)
                    generated_files_details.append({"success": False, "file_name": file_path_str, "error": str(e),
                                                    "file_path": str(project_dir / file_path_str)})
                self.emit_stream_log("Orchestrator", "stage_end", f"Done with file: '{file_path_str}'.", 1)

            successful_files = [f["file_name"] for f in generated_files_details if f.get("success")]
            self.emit_stream_log("Orchestrator", "stage_end",
                                 f"✅ Workflow complete! {len(successful_files)} files crafted.", 0)

            return {
                "success": True, "project_dir": str(project_dir),
                "files": successful_files, "project_name": project_name,
                "file_count": len(successful_files), "enhanced": True,
                "details": generated_files_details  # Include detailed results
            }
        except Exception as e:
            self.emit_stream_log("Orchestrator", "error", f"❌ The Super-Workflow hit a major snag: {e}", 0)
            # import traceback # Already imported at top level of engine
            # self.emit_stream_log("Orchestrator", "debug", traceback.format_exc(), 1)
            return {"success": False, "error": str(e)}

    async def _process_file_intelligently(self, file_path_str: str, file_info: dict,
                                          plan: dict, project_dir: Path) -> dict:
        self.emit_stream_log("Orchestrator", "thought", f"Planner is now detailing the work for '{file_path_str}'...",
                             2)
        tasks = await self.planner.create_micro_tasks(file_path_str, plan, self.context_cache)
        self.emit_stream_log("Orchestrator", "status",
                             f"Planner defined {len(tasks)} coding tasks for '{file_path_str}'.", 2)

        def coder_progress_callback(message_from_coder: str):
            # Coder's own stream_emitter will handle detailed logs. This is more for overall.
            self.emit_stream_log("Orchestrator", "info",
                                 f"Update from Coder on '{file_path_str}': {message_from_coder[:70]}...", 3)

        self.emit_stream_log("Orchestrator", "thought", f"Handing off tasks for '{file_path_str}' to the Coder.", 2)
        task_results = await self.coder.execute_tasks_parallel(
            tasks, f"File context for {file_path_str}", self.context_cache, coder_progress_callback
        )
        self.emit_stream_log("Orchestrator", "thought",
                             f"Coder finished. Assembler is now putting pieces of '{file_path_str}' together and reviewing...",
                             2)
        assembled_code, review_approved, review_feedback = await self.assembler.assemble_file(
            file_path_str, task_results, plan, self.context_cache
        )
        self.emit_stream_log("Orchestrator", "file_op", f"Saving final version of '{file_path_str}'...", 2)
        full_path_obj = project_dir / file_path_str
        full_path_obj.parent.mkdir(parents=True, exist_ok=True)
        full_path_obj.write_text(assembled_code, encoding='utf-8')

        return {
            "success": True, "file_path": str(full_path_obj), "file_name": file_path_str,
            "review_approved": review_approved, "review_feedback": review_feedback,
            "task_count": len(tasks), "assembled_length": len(assembled_code)
        }
