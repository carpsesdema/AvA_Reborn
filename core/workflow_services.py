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

# High-level prompt for just the file structure
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

# Detailed prompt for a single file's components
DETAILED_FILE_PLAN_PROMPT_TEMPLATE = textwrap.dedent("""
    You are the PLANNER AI, a master software architect. Your task is to create a detailed component-level plan for a SINGLE file within a larger project.

    OVERALL PROJECT DESCRIPTION: {project_description}
    FILE TO PLAN: `{file_path}`
    FILE'S PURPOSE: {file_purpose}

    CONTEXT OF OTHER PROJECT FILES PLANNED SO FAR:
    {other_files_context}

    RELEVANT EXAMPLES FROM KNOWLEDGE BASE:
    {rag_context}

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

# Coder prompt template
CODER_PROMPT_TEMPLATE = textwrap.dedent("""
    You are an expert Python developer. Your task is to generate a single, complete, and production-ready Python code snippet for the micro-task detailed in the SPECIFICATIONS below.

    PROJECT CONTEXT: {project_context_description}
    FILE CONTEXT: This snippet will be part of '{file_path}'
    RELEVANT EXAMPLES FROM KNOWLEDGE BASE:
    {rag_context}

    SPECIFICATIONS:
    ```json
    {task_specs_json}
    ```

    CRITICAL INSTRUCTIONS:
    1.  Generate ONLY the Python code required to fulfill THIS SPECIFIC micro-task.
    2.  Adhere strictly to every detail in the 'core_logic_steps', 'interactions', and 'critical_notes' from the specifications.
    3.  Ensure the code is clean, efficient, well-commented (docstrings for public elements, comments for complex logic), and follows PEP 8.
    4.  Do NOT generate a full file or surrounding code unless the task 'component_type' is 'full_file'.
    5.  If the task specifies "DO NOT use eval()", you MUST follow that rule.
    6.  **Your entire response MUST be ONLY the raw Python code. Do not include any explanations, introductory text, or markdown formatting like ```python.**

    Generate ONLY the required Python code snippet for this task:
""")

ASSEMBLY_PROMPT_TEMPLATE = textwrap.dedent("""
    You are the ASSEMBLER AI. Your job is to take a collection of Python code snippets and intelligently assemble them into a single, complete, and professional Python file.

    FILE PATH: `{file_path}`
    FILE PURPOSE: {file_purpose}

    PROJECT CONTEXT:
    {project_context_summary}

    CODE SNIPPETS TO ASSEMBLE:
    ---
    {raw_assembled_code}
    ---

    **CRITICAL INSTRUCTIONS:**
    1.  **Combine Snippets:** Logically arrange the provided code snippets into a cohesive file.
    2.  **Class Assembly:** When snippets include a class definition and its methods, you MUST place the method definitions (def method_name...) INSIDE the class block with correct indentation. Do not define methods outside the class they belong to.
    3.  **Manage Imports:** Consolidate all necessary `import` statements at the top of the file, removing duplicates. Follow PEP 8 import order (standard library, third-party, local modules).
    4.  **Local Imports:** Correctly formulate local imports based on the project context. For example, if a file is `calculator_model.py`, a local import should be `from calculator_model import ...`.
    5.  **Add Documentation:** Write a professional, file-level docstring that summarizes the file's purpose. Ensure all public classes and functions from the snippets retain their docstrings.
    6.  **Ensure Consistency:** Maintain consistent formatting (4-space indentation) and PEP 8 compliance throughout the entire file.

    **Your final output MUST be ONLY the raw, complete, and clean Python code for the file `{file_path}`. Do not include any conversational text, explanations, or markdown formatting.**
""")

# Reviewer prompt template
REVIEWER_PROMPT_TEMPLATE = textwrap.dedent("""
    You are a senior code reviewer examining this Python file for quality, correctness, and adherence to the plan.

    FILE: {file_path}
    PROJECT CONTEXT: {project_description}

    CODE TO REVIEW:
    ```python
    {code}
    ```
    Provide a thorough review as a JSON object with keys: "approved" (boolean), "overall_quality" (string: "excellent", "good", "fair", "poor"), "critical_issues" (list of strings), "suggestions" (list of strings), and "summary" (string). Your feedback must be specific and actionable.
""")

# NEW: Patcher prompt template
PATCHER_PROMPT_TEMPLATE = textwrap.dedent("""
    You are a senior developer tasked with fixing code based on a review. Your job is to take the original code, analyze the review feedback, and produce a new, corrected version of the code that resolves all the issues.

    ORIGINAL CODE FOR `{file_path}`:
    ```python
    {original_code}
    ```

    CODE REVIEW FEEDBACK (Issues to fix):
    ```
    {review_feedback}
    ```

    **CRITICAL INSTRUCTIONS:**
    1.  Carefully read every point in the code review feedback.
    2.  Modify the original code to address ALL critical issues and suggestions.
    3.  Do NOT add new features or change logic that was not mentioned in the feedback.
    4.  Preserve the overall structure and functionality of the original code.
    5.  Your output MUST be ONLY the full, corrected, and clean Python code for the file `{file_path}`. Do not include explanations, apologies, or markdown formatting.
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

    async def _stream_and_collect_json(self, prompt: str, role: LLMRole, agent_name: str) -> dict:
        """Helper to stream a response, show progress, and parse the final JSON."""
        self.stream_emitter(agent_name, "thought", f"Generating {role.value} response...", 2)

        all_chunks = []
        try:
            async for chunk in self.llm_client.stream_chat(prompt, role):
                all_chunks.append(chunk)
                # Stream raw chunks to terminal for live feedback
                self.stream_emitter(agent_name, "llm_chunk", chunk, 3)

            response_text = "".join(all_chunks)

            if not response_text.strip():
                self.stream_emitter(agent_name, "error", "LLM returned an empty response.", 2)
                return {}

            return self._parse_json_from_response(response_text, agent_name)

        except Exception as e:
            self.stream_emitter(agent_name, "error", f"Error during streaming/collection: {e}", 2)
            # Try to parse what we got so far
            if all_chunks:
                return self._parse_json_from_response("".join(all_chunks), agent_name)
            return {}

    async def _get_intelligent_rag_context(self, query: str, k: int = 2) -> str:
        """
        Generates a dynamic RAG query based on keywords in the user's request
        to provide more relevant context.
        """
        if not self.rag_manager or not self.rag_manager.is_ready:
            return "No RAG context available."

        self.stream_emitter("RAG", "thought", f"Generating context for: '{query}'", 4)
        lower_query = query.lower()
        gamedev_keywords = ["game", "player", "ursina", "pygame"]

        # Default general-purpose query
        dynamic_query = f"Python code example for {query}"

        # Check for specific libraries first, as they are more precise
        if "ursina" in lower_query:
            dynamic_query = f"Ursina engine example for {query}"
        elif "pygame" in lower_query:
            dynamic_query = f"Pygame sprite class for {query}"
        # Then check for other general gamedev keywords
        elif any(keyword in lower_query for keyword in gamedev_keywords):
            dynamic_query = f"Python game development example for {query}"

        self.stream_emitter("RAG", "info", f"Dynamic query: '{dynamic_query}'", 4)

        try:
            # This call is synchronous but is safe to call from an async method
            results = self.rag_manager.query_context(dynamic_query, k=k)
            if not results:
                return "No specific examples found in the knowledge base."

            # Format the results into a string for the prompt
            context = "\n\n---\n\n".join([
                f"Relevant Example from '{r.get('metadata', {}).get('filename', 'Unknown')}':\n```\n{r.get('content', '')[:700]}...\n```"
                for r in results if r.get('content')
            ])
            return context
        except Exception as e:
            self.stream_emitter("RAG", "error", f"Failed to query RAG: {e}", 4)
            return "Could not query knowledge base due to an error."


class StructureService(BaseAIService):
    """**New Service** Determines the high-level file structure of the project."""

    async def create_project_structure(self, user_prompt: str, full_conversation: List[Dict] = None) -> dict:
        self.stream_emitter("Structurer", "thought", "Phase 1: Determining file structure...", 0)
        requirements = [msg.get("message", "") for msg in (full_conversation or []) if msg.get("role") == "user"]
        requirements.append(user_prompt)
        full_requirements = " ".join(req.strip() for req in requirements if req.strip())

        rag_context = await self._get_intelligent_rag_context(full_requirements, k=1)
        plan_prompt = STRUCTURER_PROMPT_TEMPLATE.format(full_requirements=full_requirements, rag_context=rag_context)

        plan = await self._stream_and_collect_json(plan_prompt, LLMRole.STRUCTURER, "Structurer")

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

        rag_query = f"Component breakdown and architecture for a file like '{file_path}' with purpose: {file_purpose}"
        rag_context = await self._get_intelligent_rag_context(rag_query, k=2)

        plan_prompt = DETAILED_FILE_PLAN_PROMPT_TEMPLATE.format(
            project_description=project_description,
            file_path=file_path,
            file_purpose=file_purpose,
            other_files_context=other_files_context,
            rag_context=rag_context
        )

        file_plan = await self._stream_and_collect_json(plan_prompt, LLMRole.PLANNER, "Planner")

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

        rag_context = await self._get_intelligent_rag_context(task_desc, k=1)

        code_prompt = CODER_PROMPT_TEMPLATE.format(
            project_context_description=project_context.get('description', ''),
            file_path=task.get('file_path', ''),
            rag_context=rag_context,
            task_specs_json=json.dumps(task, indent=2)
        )
        self.stream_emitter("Coder", "thought", f"Writing code for '{task_id}'...", 2)

        all_chunks = []
        try:
            async for chunk in self.llm_client.stream_chat(code_prompt, LLMRole.CODER):
                all_chunks.append(chunk)
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
            task["file_path"] = file_path
            self.stream_emitter("Coder", "status",
                                f"Starting task {i + 1}/{len(component_tasks)}: '{task.get('task_id', '')}'...", 2)
            code = await self.execute_task(task, project_context)
            results.append({"task": task, "code": code})
        return results


class EnhancedAssemblerService(BaseAIService):
    """🔧 Assembles, Reviews, and Patches code."""

    def clean_llm_output(self, code: str) -> str:
        """Removes markdown fences from code output."""
        if code.strip().startswith("```python"):
            code = code.split("```python", 1)[-1]
        if code.strip().endswith("```"):
            code = code.rsplit("```", 1)[0]
        return code.strip()

    def _format_feedback_for_prompt(self, review_data: dict) -> str:
        """Helper to format JSON review feedback into a string for the Patcher."""
        summary = review_data.get('summary', 'No summary provided.')
        issues = review_data.get('critical_issues', [])
        suggestions = review_data.get('suggestions', [])

        # Ensure issues and suggestions are lists of strings
        def to_str_list(items):
            if not items:
                return []
            return [str(item) for item in items]

        issues_str = "\n- ".join(to_str_list(issues)) if issues else "None"
        suggestions_str = "\n- ".join(to_str_list(suggestions)) if suggestions else "None"

        return f"Review Summary: {summary}\n\nCritical Issues to Fix:\n- {issues_str}\n\nSuggestions to Apply:\n- {suggestions_str}"

    async def assemble_and_review_file(self, file_path: str, task_results: List[dict], plan: dict,
                                       file_spec: dict) -> str:
        self.stream_emitter("Assembler", "thought", f"Assembling '{file_path}'...", 2)
        if not task_results:
            self.stream_emitter("Assembler", "error", f"No code to assemble for '{file_path}'!", 2)
            return f'# FALLBACK: No code generated for {file_path}'

        # Stage 1: Initial Assembly
        raw_assembled_code = "\n\n".join(res['code'] for res in task_results)
        file_purpose = file_spec.get("purpose", "N/A")

        project_context_summary = f"Project: {plan.get('project_name', 'N/A')}\nDescription: {plan.get('project_description', 'N/A')}"

        assembly_prompt = ASSEMBLY_PROMPT_TEMPLATE.format(
            file_path=file_path, file_purpose=file_purpose,
            project_context_summary=project_context_summary, raw_assembled_code=raw_assembled_code
        )
        self.stream_emitter("Assembler", "thought", "Asking AI Assembler to intelligently merge code...", 2)
        assembled_code = await self.llm_client.chat(assembly_prompt, LLMRole.ASSEMBLER)
        cleaned_assembled_code = self.clean_llm_output(assembled_code)

        # Stage 2: AI Review
        self.stream_emitter("Reviewer", "thought", f"Requesting AI review for '{file_path}'...", 2)
        review_prompt = REVIEWER_PROMPT_TEMPLATE.format(
            file_path=file_path, project_description=plan.get('project_description', ''), code=cleaned_assembled_code
        )
        review_data = await self._stream_and_collect_json(review_prompt, LLMRole.REVIEWER, "Reviewer")

        if not review_data:
            self.stream_emitter("Reviewer", "error", "Review failed: Could not parse response. Accepting code as-is.",
                                2)
            return cleaned_assembled_code

        # Stage 3: Patching (if necessary)
        if review_data.get('approved'):
            self.stream_emitter("Reviewer", "success", f"✅ Review for '{file_path}': APPROVED", 2)
            return cleaned_assembled_code
        else:
            self.stream_emitter("Reviewer", "warning", f"⚠️ Review for '{file_path}': Needs patching.", 2)

            feedback_details = self._format_feedback_for_prompt(review_data)
            self.stream_emitter("Reviewer", "info", f"Feedback: {feedback_details}", 3)

            patch_prompt = PATCHER_PROMPT_TEMPLATE.format(
                file_path=file_path,
                original_code=cleaned_assembled_code,
                review_feedback=feedback_details
            )
            self.stream_emitter("Reviewer", "thought", "Applying patches...", 2)
            patched_code = await self.llm_client.chat(patch_prompt, LLMRole.REVIEWER)  # Use best model to patch
            self.stream_emitter("Reviewer", "success", "Patches applied. Final code generated.", 2)
            return self.clean_llm_output(patched_code)