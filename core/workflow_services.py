import asyncio
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Callable, Any, Coroutine
from pathlib import Path
import re
import textwrap

from core.llm_client import LLMRole, EnhancedLLMClient

# --- Prompt Templates ---
# Prompts are dedented to allow for clean in-code formatting without sending leading whitespace to the LLM.

# NEW: High-level prompt for just the file structure
STRUCTURER_PROMPT_TEMPLATE = textwrap.dedent("""
    You are the STRUCTURER AI. Your single, focused task is to analyze a user's request and determine the necessary file structure for the project. You only think about what files are needed and what their high-level purpose is.

    USER REQUEST: "{full_requirements}"

    RELEVANT CONTEXT FROM KNOWLEDGE BASE:
    {rag_context}

    Your output MUST be a single, valid JSON object with the following structure:
    {{
      "project_name": "a-descriptive-snake-case-name",
      "project_description": "A one-sentence description of the application.",
      "files": [
        {{
          "filename": "main.py",
          "purpose": "The main entry point of the application."
        }},
        {{
          "filename": "another_file.py",
          "purpose": "A brief description of this file's role."
        }}
      ]
    }}

    **IMPORTANT: Your _entire_ output must be ONLY the JSON object. Do not include any text, conversational filler, or markdown formatting like ```json before or after the JSON. The system will fail if it receives anything other than the raw, valid JSON object.**
""")

# NEW: Detailed prompt for a single file's components
DETAILED_FILE_PLAN_PROMPT_TEMPLATE = textwrap.dedent("""
    You are the PLANNER AI, a master software architect. Your task is to create a detailed component-level plan for a SINGLE file within a larger project.

    OVERALL PROJECT DESCRIPTION: {project_description}
    FILE TO PLAN: `{file_path}`
    FILE'S PURPOSE: {file_purpose}

    CONTEXT OF OTHER PROJECT FILES PLANNED SO FAR:
    {other_files_context}

    Your output MUST be a single, valid JSON object containing a 'components' array. Each object in the array represents a single class, method, or function for the file `{file_path}`.

    For EACH component, you MUST provide the following fields: 'task_id', 'description', 'component_type', 'inputs', 'outputs', 'core_logic_steps', 'error_conditions_to_handle', 'interactions', and 'critical_notes'.

    **CRITICAL FOR GUIs**: For a Controller file, the 'interactions' field MUST detail the specific signal-to-slot connections. For a View, it must describe the UI elements. For a Model, it must describe the state it manages.

    **CRITICAL FOR CALCULATOR LOGIC**: For any calculator logic, you MUST design a stateful calculator. DO NOT create a plan that involves parsing a full mathematical expression string. Instead, the `core_logic_steps` for the model must detail a state-based approach with:
    1.  State variables (e.g., `current_display_value`, `first_operand`, `pending_operation`).
    2.  Methods to handle digit inputs (e.g., `process_digit_input`) that append to the current display value.
    3.  A method to handle operators (e.g., `process_operator_input`) that sets the pending operation and stores the first operand.
    4.  A method to trigger calculation (e.g., the equals button) which uses the stored state to perform a single operation.
    5.  A method to clear the state.
    This approach avoids the security risks and complexity of `eval()` and ad-hoc string parsing.

    Example 'components' array structure:
    {{
      "components": [
        {{
          "task_id": "ClassName_method_name",
          "description": "...",
          "component_type": "method_definition",
          "inputs": ["..."],
          "outputs": "...",
          "core_logic_steps": ["..."],
          "error_conditions_to_handle": ["..."],
          "interactions": ["..."],
          "critical_notes": "..."
        }}
      ]
    }}

**IMPORTANT: Your _entire_ output must be ONLY the JSON object. Do not include any text, conversational filler, or markdown formatting like ```json before or after the JSON. The system will fail if it receives anything other than the raw, valid JSON object.**
""")

# RE-ADDED a Coder prompt template
CODER_PROMPT_TEMPLATE = textwrap.dedent("""
    You are an expert Python developer. Your task is to generate a single, complete, and production-ready Python code snippet for the micro-task detailed in the SPECIFICATIONS below.

    PROJECT CONTEXT: {project_context_description}
    FILE CONTEXT: This snippet will be part of '{file_path}'
    {correction_feedback}
    SPECIFICATIONS:
    ```json
    {task_specs_json}
    ```

    CRITICAL INSTRUCTIONS:
    1.  Generate ONLY the Python code required to fulfill THIS SPECIFIC micro-task.
    2.  Adhere strictly to every detail in the 'core_logic_steps', 'interactions', and 'critical_notes' from the specifications.
    3.  If feedback is provided above, you MUST address it in your new code.
    4.  Ensure the code is clean, efficient, well-commented (docstrings for public elements, comments for complex logic), and follows PEP 8.
    5.  Do NOT generate a full file or surrounding code unless the task 'component_type' is 'full_file'.
    6.  If the task specifies "DO NOT use eval()", you MUST follow that rule.
    7.  **Your entire response MUST be ONLY the raw Python code. Do not include any explanations, introductory text, or markdown formatting like ```python.**

    Generate ONLY the required Python code snippet for this task:
""")

# RE-ADDED a Reviewer prompt template
REVIEWER_PROMPT_TEMPLATE = textwrap.dedent("""
    You are a senior code reviewer examining this Python file for quality, correctness, and adherence to the plan.

    FILE: {file_path}
    PROJECT CONTEXT: {project_description}

    CODE TO REVIEW:
    ```python
    {code}
    ```
    Provide a thorough review as a JSON object with keys: "approved" (boolean), "overall_quality" (string: "excellent", "good", "fair", "poor"), "critical_issues" (list of strings), "suggestions" (list of strings), and "summary" (string).
""")


class BaseAIService:
    def __init__(self, llm_client: EnhancedLLMClient, stream_emitter: Callable, rag_manager=None):
        self.llm_client = llm_client
        self.stream_emitter = stream_emitter
        self.rag_manager = rag_manager

    def _parse_json_from_response(self, text: str, agent_name: str) -> dict:
        self.stream_emitter(agent_name, "thought_detail",
                            f"Attempting to parse JSON from response (length: {len(text)})...", 3)
        text = text.strip()
        fence_pattern = r"```json\s*(\{[\s\S]*\})\s*```"
        match = re.search(fence_pattern, text, re.DOTALL)

        json_text = ""
        if match:
            json_text = match.group(1)
            self.stream_emitter(agent_name, "fallback", "Found JSON object inside markdown fences. Parsing that.", 3)
        else:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_text = text[start:end + 1]
                self.stream_emitter(agent_name, "fallback",
                                    "No markdown fences. Extracted content between first and last curly braces.", 3)
            else:
                self.stream_emitter(agent_name, "error", "Could not find any JSON-like structure.", 3)
                return {}

        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            self.stream_emitter(agent_name, "error", f"JSON parsing failed: {e}", 3)
            self.stream_emitter(agent_name, "debug", f"Failed to parse text: {json_text[:200]}...", 4)
            return {}

    async def _get_intelligent_rag_context(self, requirements: str) -> str:
        if not self.rag_manager or not self.rag_manager.is_ready: return "No RAG context available."
        req_lower = requirements.lower()
        query = f"PySide6 View-Controller-Model architecture example" if "pyside" in req_lower or "gui" in req_lower else f"Python code examples for a {req_lower}"
        try:
            results = self.rag_manager.query_context(query, k=2)
            return "\n\n".join([f"# Relevant Example:\n{r.get('content', '')[:500]}..." for r in results if
                                r.get('content')]) or "No specific examples found."
        except Exception:
            return "Could not query knowledge base."


class StructureService(BaseAIService):
    """**New Service** Determines the high-level file structure of the project."""

    async def create_project_structure(self, user_prompt: str, full_conversation: List[Dict] = None) -> dict:
        self.stream_emitter("Structurer", "thought", "Phase 1: Determining file structure...", 0)
        requirements = [msg.get("message", "") for msg in (full_conversation or []) if msg.get("role") == "user"]
        requirements.append(user_prompt)
        full_requirements = " ".join(req.strip() for req in requirements if req.strip())

        rag_context = await self._get_intelligent_rag_context(full_requirements)
        plan_prompt = STRUCTURER_PROMPT_TEMPLATE.format(full_requirements=full_requirements, rag_context=rag_context)

        response_text = await self.llm_client.chat(plan_prompt, LLMRole.STRUCTURER)
        plan = self._parse_json_from_response(response_text, "Structurer")

        if not plan or not plan.get("files"):
            self.stream_emitter("Structurer", "error",
                                "High-level planning failed. Could not determine file structure.", 1)
            return {}

        self.stream_emitter("Structurer", "success", "High-level plan created successfully!", 0)
        return plan


class EnhancedPlannerService(BaseAIService):
    """🧠 Plans the components for a single file."""

    async def create_detailed_file_plan(self, file_path: str, file_purpose: str, project_description: str,
                                        other_files_context: str) -> dict:
        self.stream_emitter("Planner", "thought", f"Phase 2: Creating detailed plan for `{file_path}`...", 1)

        plan_prompt = DETAILED_FILE_PLAN_PROMPT_TEMPLATE.format(
            project_description=project_description,
            file_path=file_path,
            file_purpose=file_purpose,
            other_files_context=other_files_context
        )

        response_text = await self.llm_client.chat(plan_prompt, LLMRole.PLANNER)
        file_plan = self._parse_json_from_response(response_text, "Planner")

        if not file_plan or not file_plan.get("components"):
            self.stream_emitter("Planner", "error", f"Detailed planning for `{file_path}` failed.", 2)
            return {
                "components": [{
                    "task_id": f"implement_{Path(file_path).stem}_full",
                    "description": f"Full implementation for {file_path}. Purpose: {file_purpose}",
                    "component_type": "full_file",
                    "core_logic_steps": ["Implement all necessary logic according to the file's purpose."]
                }]
            }

        self.stream_emitter("Planner", "success", f"Detailed plan for `{file_path}` is ready.", 1)
        return file_plan


class EnhancedCoderService(BaseAIService):
    """⚙️ Production-Ready Coding Service with Intelligent Generation"""

    async def execute_task(self, task: dict, project_context: dict) -> str:
        task_id, task_desc = task.get("task_id", "unknown_task"), task.get("description", "N/A")
        self.stream_emitter("Coder", "thought", f"Focusing on task '{task_id}': {task_desc[:60]}...", 2)

        feedback_text = ""
        if project_context.get("feedback_for_coder"):
            feedback_text = f"CORRECTION FEEDBACK FROM PREVIOUS ATTEMPT:\n{project_context['feedback_for_coder']}\n"
            self.stream_emitter("Coder", "warning", "Applying feedback from previous attempt.", 2)

        code_prompt = CODER_PROMPT_TEMPLATE.format(
            project_context_description=project_context.get('description', ''),
            file_path=task.get('file_path', ''),
            correction_feedback=feedback_text,
            task_specs_json=json.dumps(task, indent=2)
        )
        self.stream_emitter("Coder", "thought", f"Writing code for '{task_id}'...", 2)

        all_chunks = []
        try:
            # Use stream_chat to get real-time output
            async for chunk in self.llm_client.stream_chat(code_prompt, LLMRole.CODER):
                all_chunks.append(chunk)
                # Emit each chunk to the terminal for live streaming
                self.stream_emitter("Coder", "llm_chunk", chunk, 3)

            raw_code = "".join(all_chunks)
            cleaned_code = self._clean_and_validate_code(raw_code)
            self.stream_emitter("Coder", "success", f"Code for task '{task_id}' is ready.", 2)
            return cleaned_code
        except Exception as e:
            self.stream_emitter("Coder", "error", f"Error coding task '{task_id}' ({e}). Using placeholder.", 2)
            return f"# FALLBACK: Task '{task_id}' - {task_desc}\npass"

    def _clean_and_validate_code(self, code: str) -> str:
        if code.strip().startswith("```python"): code = code.split("```python", 1)[-1]
        if code.strip().endswith("```"): code = code.rsplit("```", 1)[0]
        return code.strip()

    async def execute_tasks_for_file(self, file_spec: dict, project_context: dict) -> List[dict]:
        file_path = file_spec.get("path", "unknown_file.py")
        component_tasks = file_spec.get("components", [])

        self.stream_emitter("Coder", "status", f"Preparing to code {len(component_tasks)} tasks for {file_path}.", 2)
        results = []
        for i, task in enumerate(component_tasks):
            task["file_path"] = file_path  # Ensure file_path is in the task spec
            self.stream_emitter("Coder", "status",
                                f"Starting task {i + 1}/{len(component_tasks)}: '{task.get('task_id', '')}'...", 2)
            code = await self.execute_task(task, project_context)
            results.append({"task": task, "code": code})
        return results


class EnhancedAssemblerService(BaseAIService):
    """🔧 Production-Ready Code Assembly Service with Review"""

    async def assemble_file(self, file_path: str, task_results: List[dict], plan: dict, file_spec: dict) -> Tuple[
        str, bool, str]:
        self.stream_emitter("Assembler", "thought", f"Assembling '{file_path}'...", 2)
        if not task_results:
            self.stream_emitter("Assembler", "error", f"No code to assemble for '{file_path}'!", 2)
            return f'# FALLBACK: No code generated for {file_path}', False, "No code chunks available"

        raw_assembled_code = "\n\n".join(res['code'] for res in task_results)

        file_purpose = "N/A"
        for f in plan.get('files', []):
            if f.get('filename') == file_path:
                file_purpose = f.get('purpose', 'N/A')
                break

        component_summary = "\n".join(
            [f"- {c.get('component_type', 'component')}: {c.get('description', 'N/A')}" for c in
             file_spec.get('components', [])])

        other_files_summary = "\n".join(
            [f"- {f.get('filename')}: {f.get('purpose')}" for f in plan.get('files', []) if
             f.get('filename') != file_path]
        )

        assembly_prompt = textwrap.dedent(f"""
            **Task: Assemble Code File**

            **File Path:** `{file_path}`
            **File Purpose:** {file_purpose}
            **Project File Structure Context:**
            {other_files_summary}

            **Input Code Snippets:**
            ---
            {raw_assembled_code}
            ---

            **Instructions:**
            1.  Combine the input snippets into a single, cohesive Python file.
            2.  Consolidate all imports at the top, following PEP 8.
            3.  **Use the correct module names for local imports based on the Project File Structure Context.** For example, if a file is `calculator_model.py`, the import should be `from calculator_model import ...`.
            4.  Add a professional file-level docstring summarizing the file's purpose.
            5.  Ensure all public classes and functions have docstrings.
            6.  Ensure consistent formatting (4-space indentation) and PEP 8 compliance.

            **CRITICAL: Your output MUST be ONLY the raw, complete Python code for the file `{file_path}`. Do not include any conversational text, explanations, or markdown formatting.**
        """)

        self.stream_emitter("Assembler", "thought", f"Asking AI Assembler to intelligently merge code...", 2)
        cleaned_code = await self.llm_client.chat(assembly_prompt, LLMRole.ASSEMBLER)

        self.stream_emitter("Assembler", "thought", f"Requesting AI review for '{file_path}'...", 2)
        review_approved, review_feedback = await self._ai_review_code(file_path, cleaned_code, plan)

        if review_approved:
            self.stream_emitter("Assembler", "success", f"✅ Assembly for '{file_path}' - Review: APPROVED", 2)
        else:
            self.stream_emitter("Assembler", "warning", f"⚠️ Assembly for '{file_path}' - Review: NEEDS ATTENTION", 2)
            self.stream_emitter("Assembler", "info", f"Feedback: {review_feedback[:100]}...", 3)

        return self.clean_llm_output(cleaned_code), review_approved, review_feedback

    def clean_llm_output(self, code: str) -> str:
        """Removes markdown fences from code output."""
        if code.strip().startswith("```python"):
            code = code.split("```python", 1)[-1]
        if code.strip().endswith("```"):
            code = code.rsplit("```", 1)[0]
        return code.strip()

    async def _ai_review_code(self, file_path: str, code: str, plan: dict) -> Tuple[bool, str]:
        self.stream_emitter("Assembler", "thought_detail", "Preparing review prompt...", 3)
        review_prompt = REVIEWER_PROMPT_TEMPLATE.format(file_path=file_path,
                                                        project_description=plan.get('project_description', ''),
                                                        code=code)
        try:
            review_response = await self.llm_client.chat(review_prompt, LLMRole.REVIEWER)
            review_data = self._parse_json_from_response(review_response, "Reviewer")
            approved = review_data.get('approved', False)
            summary = review_data.get('summary', 'AI review completed.')
            return approved, summary
        except Exception as e:
            self.stream_emitter("Assembler", "warning", f"AI review failed ({e}). Using basic validation.", 3)
            return len(code.strip()) > 10, "Basic validation: Code is not empty."