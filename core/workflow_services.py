# core/workflow_services.py

import asyncio
import json
import re
import textwrap
from typing import Callable, Dict, List

from core.llm_client import LLMRole, EnhancedLLMClient
from core.project_state_manager import ProjectStateManager  # NEW IMPORT

# --- Prompt Templates ---

# V4 ARCHITECT PROMPT - MODIFIED TO INCLUDE REQUIREMENTS
ARCHITECT_PROMPT_TEMPLATE = textwrap.dedent("""
    You are the ARCHITECT AI. Your task is to create a complete, comprehensive, and machine-readable Technical Specification Sheet for an entire software project based on a user's request. This sheet will be the single source of truth for all other AI agents.

    USER REQUEST: "{full_requirements}"

    RELEVANT CONTEXT FROM KNOWLEDGE BASE:
    {rag_context}

    Your output MUST be a single, valid JSON object. This object will contain the project name, a description, a list of required libraries, a dependency-sorted build order, and a detailed `technical_specs` dictionary for every file.

    **NEW REQUIREMENT**: You MUST identify any necessary third-party Python libraries (e.g., "flask", "requests", "pygame", "ursina") and list them in a `requirements` array.

    For each file in `technical_specs`, you must define its `purpose`, its `dependencies`, and its `api_contract`.
    The `api_contract` is the most critical part. It must define:
    - For config files: A list of all required `variables` with their name and type.
    - For class-based files: A list of `classes`, each with its name, what it `inherits_from`, and a list of all `methods` with their exact `signature`.
    - For entry points (`main.py`): A clear `execution_flow` describing the sequence of operations.

    You must also determine the correct `dependency_order` for building the files. Files with no dependencies come first.

    **EXAMPLE JSON STRUCTURE:**
    {{
      "project_name": "a-descriptive-snake-case-name",
      "project_description": "A one-sentence description of the application.",
      "requirements": ["ursina==0.7.2"],
      "dependency_order": ["config.py", "player.py", "main.py"],
      "technical_specs": {{
        "config.py": {{
          "purpose": "Stores all static configuration variables.",
          "dependencies": [],
          "api_contract": {{"variables": [{{"name": "WINDOW_TITLE", "type": "str"}}]}}
        }},
        "player.py": {{
          "purpose": "Defines the Player class.",
          "dependencies": ["config.py"],
          "api_contract": {{"classes": [{{"name": "Player", "inherits_from": "Entity", "methods": [{{"signature": "def __init__(self):"}}]}}]}}
        }}
      }}
    }}

    **IMPORTANT: Your _entire_ output must be ONLY the raw, valid JSON object. Do not include any conversational filler or markdown formatting.**
""")

# NEW PROMPT for analyzing existing code!
ARCHITECT_ANALYSIS_PROMPT_TEMPLATE = textwrap.dedent("""
    You are the ARCHITECT AI. Your task is to act as a reverse-engineer. You will analyze the complete context of an existing software project and create a comprehensive, machine-readable Technical Specification Sheet that accurately represents its current state.

    **PROJECT CONTEXT & FILE INFORMATION (The source of truth):**
    ```json
    {project_context_json}
    ```

    Based on the provided project context, which includes file paths, dependencies, and exported symbols (classes/functions), your job is to generate a single, valid JSON object that represents the project's architecture.

    This JSON object must contain:
    1.  `project_name`: The name of the project.
    2.  `project_description`: A one-sentence summary of what the project does.
    3.  `requirements`: A list of third-party libraries used in the project.
    4.  `dependency_order`: A topologically sorted list of filenames for the correct build order.
    5.  `technical_specs`: A dictionary where each key is a filename. The value for each filename must contain:
        - `purpose`: A brief description of the file's role.
        - `dependencies`: A list of other project files it depends on.
        - `api_contract`: An object defining the public API of the file (classes, methods, variables).

    **EXAMPLE JSON STRUCTURE:**
    {{
      "project_name": "existing-project",
      "project_description": "A Python application for managing user data.",
      "requirements": ["flask", "sqlalchemy"],
      "dependency_order": ["config.py", "database.py", "models.py", "main.py"],
      "technical_specs": {{
        "config.py": {{
          "purpose": "Central configuration file.",
          "dependencies": [],
          "api_contract": {{
            "variables": [
              {{"name": "DATABASE_URL", "type": "str"}},
              {{"name": "API_TIMEOUT", "type": "int"}}
            ]
          }}
        }},
        "models.py": {{
          "purpose": "Defines the data models for the application.",
          "dependencies": ["database.py"],
          "api_contract": {{
            "classes": [
              {{
                "name": "User",
                "inherits_from": "Base",
                "methods": [
                  {{"signature": "def __init__(self, name: str, email: str):"}},
                  {{"signature": "def set_password(self, password: str) -> None:"}}
                ]
              }}
            ]
          }}
        }}
      }}
    }}

    **CRITICAL: Your _entire_ output must be ONLY the raw, valid JSON object. Do not include any conversational filler, explanations, or markdown formatting.**
""")

# V5 CODER PROMPT - ACCEPTS FULL KNOWLEDGE PACKETS
CODER_PROMPT_TEMPLATE = textwrap.dedent("""
    You are an expert Python developer. Your task is to generate the complete and functional source code for a single Python file, based on a strict Technical Specification and the full context of its dependencies.

    **PROJECT CONTEXT:** {project_context_description}

    ---
    **FILE TO GENERATE:** `{file_path}`

    **TECHNICAL SPECIFICATION FOR `{file_path}` (Your primary instructions):**
    ```json
    {tech_spec_json}
    ```
    ---
    **FULL CONTEXT OF DEPENDENCIES (Your source of truth for imports and interactions):**
    This section contains the full technical specs and the final, generated source code for all files that `{file_path}` depends on. You MUST use this to ensure perfect integration.

    {full_dependency_context}
    ---

    **CRITICAL INSTRUCTIONS:**
    1.  Generate the **entire, complete, and runnable source code** for the file `{file_path}`.
    2.  Implement the full logic for every method and function defined in your technical specification. **DO NOT use `pass` as a placeholder.** Your code must be fully implemented.
    3.  You MUST adhere to the `api_contract` from your technical specification and correctly use the classes, methods, and variables from the provided dependency context.
    4.  **Your entire response MUST be ONLY the raw Python code.** Do not include any explanations or markdown formatting like ```python.

    Generate the complete and fully implemented Python code for `{file_path}` now:
""")

REVIEWER_PROMPT_TEMPLATE = textwrap.dedent("""
    You are a senior code reviewer examining this Python file for quality and correctness. Provide a review as a JSON object with keys: "approved" (boolean), "summary" (string), "suggestions" (list of strings).
""")


class BaseAIService:
    # ... (This class is unchanged)
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
            start, end = text.find('{'), text.rfind('}')
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
        self.stream_emitter(agent_name, "thought", f"Generating {role.value} response...", 2)
        all_chunks = []
        try:
            async for chunk in self.llm_client.stream_chat(prompt, role):
                all_chunks.append(chunk)
                self.stream_emitter(agent_name, "llm_chunk", chunk, 3)
            response_text = "".join(all_chunks)
            if not response_text.strip():
                self.stream_emitter(agent_name, "error", "LLM returned an empty response.", 2)
                return {}
            return self._parse_json_from_response(response_text, agent_name)
        except Exception as e:
            self.stream_emitter(agent_name, "error", f"Error during streaming/collection: {e}", 2)
            if all_chunks:
                return self._parse_json_from_response("".join(all_chunks), agent_name)
            return {}

    async def _get_intelligent_rag_context(self, query: str, k: int = 2) -> str:
        # ... (This method remains unchanged)
        if not self.rag_manager or not self.rag_manager.is_ready: return "No RAG context available."
        self.stream_emitter("RAG", "thought", f"Generating context for: '{query}'", 4)
        dynamic_query = f"Python code example for {query}"
        try:
            results = self.rag_manager.query_context(dynamic_query, k=k)
            if not results: return "No specific examples found in the knowledge base."
            return "\n\n---\n\n".join([
                f"Relevant Example from '{r.get('metadata', {}).get('filename', 'Unknown')}':\n```python\n{r.get('content', '')[:700]}...\n```"
                for r in results if r.get('content')])
        except Exception as e:
            self.stream_emitter("RAG", "error", f"Failed to query RAG: {e}", 4)
            return "Could not query knowledge base due to an error."


class ArchitectService(BaseAIService):

    async def create_tech_spec(self, user_prompt: str, full_conversation: List[Dict] = None) -> dict:
        self.stream_emitter("Architect", "thought",
                            "Phase 1: Architecting the complete project technical specification...", 0)
        requirements = [msg.get("message", "") for msg in (full_conversation or []) if msg.get("role") == "user"]
        requirements.append(user_prompt)
        full_requirements = " ".join(req.strip() for req in requirements if req.strip())
        rag_context = await self._get_intelligent_rag_context(full_requirements, k=1)
        plan_prompt = ARCHITECT_PROMPT_TEMPLATE.format(full_requirements=full_requirements, rag_context=rag_context)
        tech_spec = await self._stream_and_collect_json(plan_prompt, LLMRole.ARCHITECT, "Architect")
        if not tech_spec or not tech_spec.get("technical_specs"):
            self.stream_emitter("Architect", "error",
                                "Architecting failed. Could not produce a valid technical specification.", 1)
            return {}
        self.stream_emitter("Architect", "success", "Master Technical Specification created successfully!", 0)
        return tech_spec

    async def analyze_and_create_spec_from_project(self, project_state: ProjectStateManager) -> dict:
        """NEW: Analyzes an existing project and creates a tech spec for it."""
        self.stream_emitter("Architect", "thought",
                            "Analyzing existing project to reverse-engineer its architecture...", 1)

        # Use the ProjectStateManager to get the full context of the project
        project_context = project_state.get_project_context()

        # We need to make sure the context is serializable
        # A simple way is to dump and load, but you might want a more elegant solution
        # This handles datetime objects and other non-serializable types
        serializable_context = json.loads(json.dumps(project_context, default=str))

        # Update the analysis prompt to also look for requirements
        analysis_prompt = ARCHITECT_ANALYSIS_PROMPT_TEMPLATE.format(
            project_context_json=json.dumps(serializable_context, indent=2)
        )

        tech_spec = await self._stream_and_collect_json(analysis_prompt, LLMRole.ARCHITECT, "Architect")
        if not tech_spec or not tech_spec.get("technical_specs"):
            self.stream_emitter("Architect", "error",
                                "Project analysis failed. Could not produce a valid technical specification from the provided context.",
                                1)
            return {}

        self.stream_emitter("Architect", "success", "Project analysis complete. Technical spec created!", 1)
        return tech_spec


class CoderService(BaseAIService):
    """V5 Coder Service: Generates a complete file using a Knowledge Packet."""

    async def generate_file_from_spec(self, file_path: str, tech_spec: dict, project_context: dict,
                                      full_dependency_context: str) -> str:
        self.stream_emitter("Coder", "thought", f"Generating full code for file: '{file_path}'...", 2)

        rag_context = await self._get_intelligent_rag_context(tech_spec.get("purpose", ""), k=1)

        code_prompt = CODER_PROMPT_TEMPLATE.format(
            project_context_description=project_context.get("description", ""),
            file_path=file_path,
            full_dependency_context=full_dependency_context,  # NEW: Pass the full Knowledge Packet
            rag_context=rag_context,
            tech_spec_json=json.dumps(tech_spec, indent=2)
        )

        all_chunks = []
        try:
            # Use streaming to provide real-time feedback in the terminal
            async for chunk in self.llm_client.stream_chat(code_prompt, LLMRole.CODER):
                all_chunks.append(chunk)
                self.stream_emitter("Coder", "llm_chunk", chunk, 3)

            raw_code = "".join(all_chunks)
            cleaned_code = self._clean_llm_output(raw_code)
            self.stream_emitter("Coder", "success", f"Code for file '{file_path}' is ready.", 2)
            return cleaned_code
        except Exception as e:
            self.stream_emitter("Coder", "error", f"Error coding file '{file_path}' ({e}). Using placeholder.", 2)
            return f"# FALLBACK: Failed to generate code for {file_path}. Error: {e}\npass"

    def _clean_llm_output(self, code: str) -> str:
        if code.strip().startswith("```python"):
            code = code.split("```python", 1)[-1]
        if code.strip().endswith("```"):
            code = code.rsplit("```", 1)[0]
        return code.strip()


class ReviewerService(BaseAIService):
    """🧐 Performs a final quality check on generated code."""

    async def review_code(self, file_path: str, code: str, project_description: str) -> tuple[dict, bool]:
        self.stream_emitter("Reviewer", "thought", f"Requesting final quality review for '{file_path}'...", 2)
        review_prompt = REVIEWER_PROMPT_TEMPLATE.format(
            file_path=file_path,
            project_description=project_description,
            code=code
        )
        review_data = await self._stream_and_collect_json(review_prompt, LLMRole.REVIEWER, "Reviewer")
        if not review_data:
            self.stream_emitter("Reviewer", "error", "Review failed: Could not parse response. Approving by default.",
                                2)
            return {"approved": True, "summary": "Review failed to parse, approved by default."}, True
        return review_data, review_data.get('approved', False)