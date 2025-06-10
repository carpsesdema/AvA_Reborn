# core/workflow_services.py - Enhanced with Granular Streaming Feedback

import asyncio
import json
import re
import textwrap
import ast
import time
from typing import Callable, Dict, List

from core.llm_client import LLMRole, EnhancedLLMClient
from core.project_state_manager import ProjectStateManager

# --- Enhanced Prompt Templates ---

ARCHITECT_PROMPT_TEMPLATE = textwrap.dedent("""
    You are the ARCHITECT AI. Your task is to create a complete, comprehensive, and machine-readable Technical Specification Sheet for an entire software project based on a user's request. This sheet will be the single source of truth for all other AI agents.

    **USER REQUEST**: "{full_requirements}"

    **PRIORITY CONTEXT (GDD - Game Design Document):**
    The following context is from the project's official design document. This is the MOST IMPORTANT source of truth for the project's high-level goals, implemented systems, and development history. You MUST use this to guide your architectural decisions.
    ---
    {gdd_context}
    ---

    **RELEVANT CONTEXT FROM KNOWLEDGE BASE (RAG):**
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
          "purpose": "Defines the Player class with movement and physics.",
          "dependencies": ["config.py"],
          "api_contract": {{
            "classes": [{{
              "name": "Player",
              "inherits_from": "Entity",
              "methods": [
                {{"signature": "__init__(self, **kwargs)", "purpose": "Initialize player"}},
                {{"signature": "input(self, key)", "purpose": "Handle input events"}},
                {{"signature": "update(self)", "purpose": "Update player state each frame"}}
              ]
            }}]
          }}
        }}
      }}
    }}

    Generate ONLY the JSON object. No explanations or markdown formatting.
""")

CODER_PROMPT_TEMPLATE = textwrap.dedent("""
    You are the CODER AI. Generate the complete, fully-functional Python code for the specified file.

    **PROJECT CONTEXT**: {project_context_description}

    **FILE TO GENERATE**: `{file_path}`

    **TECHNICAL SPECIFICATION**:
    ```json
    {tech_spec_json}
    ```

    **DEPENDENCY CONTEXT** (Code from files this file depends on):
    {full_dependency_context}

    **RELEVANT EXAMPLES** (RAG Context):
    {rag_context}

    **CRITICAL REQUIREMENTS**:
    1. Generate the **entire, complete, and runnable source code** for the file `{file_path}`.
    2. Implement the full logic for every method and function defined in your technical specification. **DO NOT use `pass` as a placeholder.** Your code must be fully implemented.
    3. You MUST adhere to the `api_contract` from your technical specification and correctly use the classes, methods, and variables from the provided dependency context.
    4. **Your entire response MUST be ONLY the raw Python code.** Do not include any explanations or markdown formatting like ```python.

    Generate the complete and fully implemented Python code for `{file_path}` now:
""")

REVIEWER_PROMPT_TEMPLATE = textwrap.dedent("""
    You are the SMART REVIEWER AI - the single comprehensive quality gate for all generated code. You perform deep analysis covering syntax, security, performance, style, best practices, and architectural consistency.

    **FILE TO REVIEW:** {file_path}
    **PROJECT DESCRIPTION:** {project_description}

    **CODE TO REVIEW:**
    ```python
    {code}
    ```

    **COMPREHENSIVE ANALYSIS REQUIRED:**

    1. **SYNTAX & CORRECTNESS**
       - Check for syntax errors, undefined variables, import issues
       - Verify all functions/classes are properly implemented (no bare `pass` statements)
       - Ensure proper indentation and Python grammar

    2. **SECURITY ANALYSIS**
       - Scan for dangerous functions (eval, exec, subprocess with shell=True)
       - Check for SQL injection vulnerabilities in query construction
       - Identify unsafe file operations or user input handling
       - Flag missing input validation and sanitization

    3. **PERFORMANCE & EFFICIENCY**
       - Identify inefficient algorithms or data structures
       - Check for memory leaks or unnecessary object creation
       - Flag blocking operations that should be async
       - Suggest optimization opportunities

    4. **STYLE & BEST PRACTICES**
       - PEP 8 compliance (line length, naming conventions, imports)
       - Proper docstring usage for classes and functions
       - Appropriate error handling with specific exceptions
       - Code readability and maintainability

    5. **ARCHITECTURAL CONSISTENCY**
       - Verify adherence to project patterns and conventions
       - Check proper separation of concerns
       - Ensure consistent API design and interfaces
       - Validate proper dependency management

    **OUTPUT FORMAT:**
    Return a JSON object with this exact structure:
    {{
        "approved": boolean,
        "quality_score": number (1-10),
        "summary": "Brief overall assessment",
        "issues": [
            {{
                "category": "security|performance|style|syntax|architecture",
                "severity": "critical|high|medium|low",
                "line": number or null,
                "message": "Specific issue description",
                "suggestion": "How to fix this issue"
            }}
        ],
        "suggestions": [
            "Overall improvement recommendations"
        ],
        "security_concerns": [
            "Any security-related issues found"
        ],
        "performance_notes": [
            "Performance optimization opportunities"
        ]
    }}

    **CRITICAL:** Return ONLY the JSON object. No explanations or markdown formatting.
""")


class BaseAIService:
    """Base service class with enhanced streaming capabilities"""

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
        if not self.rag_manager or not self.rag_manager.is_ready:
            return "No RAG context available."
        self.stream_emitter("RAG", "thought", f"Generating context for: '{query}'", 4)
        dynamic_query = f"Python code example for {query}"
        try:
            results = self.rag_manager.query_context(dynamic_query, k=k)
            if not results:
                return "No specific examples found in the knowledge base."
            return "\n\n---\n\n".join([
                f"Relevant Example from '{r.get('metadata', {}).get('filename', 'Unknown')}':\n```python\n{r.get('content', '')[:700]}...\n```"
                for r in results if r.get('content')])
        except Exception as e:
            self.stream_emitter("RAG", "error", f"Failed to query RAG: {e}", 4)
            return "Could not query knowledge base due to an error."


class ArchitectService(BaseAIService):
    """Enhanced Architect Service with better streaming feedback"""

    async def create_tech_spec(self, full_requirements: str, conversation_context: List = None) -> dict:
        # Use "Planner" instead of "Architect" to match terminal colors
        agent_name = "Planner"

        self.stream_emitter(agent_name, "thought", "Analyzing project requirements...", 1)

        # Enhanced RAG context retrieval
        rag_context = await self._get_intelligent_rag_context(full_requirements, k=3)

        self.stream_emitter(agent_name, "thought", "Building comprehensive technical specification...", 1)

        # Extract GDD context if available
        gdd_context = ""
        if "--- GDD CONTEXT ---" in full_requirements:
            parts = full_requirements.split("--- GDD CONTEXT ---")
            if len(parts) > 1:
                gdd_context = parts[1].strip()
                full_requirements = parts[0].strip()
                self.stream_emitter(agent_name, "info", f"Using GDD context ({len(gdd_context)} characters)", 2)

        arch_prompt = ARCHITECT_PROMPT_TEMPLATE.format(
            full_requirements=full_requirements,
            gdd_context=gdd_context,
            rag_context=rag_context
        )

        self.stream_emitter(agent_name, "thought", "Streaming architectural analysis...", 2)
        tech_spec = await self._stream_and_collect_json(arch_prompt, LLMRole.ARCHITECT, agent_name)

        if not tech_spec or 'technical_specs' not in tech_spec:
            self.stream_emitter(agent_name, "error", "Could not produce a valid technical specification.", 1)
            return {}

        # Enhanced success feedback
        file_count = len(tech_spec.get('technical_specs', {}))
        self.stream_emitter(agent_name, "success", f"Architecture complete! {file_count} files specified.", 1)
        return tech_spec

    async def analyze_and_create_spec_from_project(self, project_state: ProjectStateManager) -> dict:
        agent_name = "Planner"
        self.stream_emitter(agent_name, "thought", "Analyzing existing project architecture...", 1)

        project_context = project_state.get_project_context()
        serializable_context = json.loads(json.dumps(project_context, default=str))

        self.stream_emitter(agent_name, "info", f"Scanned {len(project_state.files)} project files", 2)

        # Use analysis prompt (would need to be defined)
        tech_spec = await self._stream_and_collect_json("", LLMRole.ARCHITECT, agent_name)

        if not tech_spec or not tech_spec.get("technical_specs"):
            self.stream_emitter(agent_name, "error", "Project analysis failed.", 1)
            return {}

        self.stream_emitter(agent_name, "success", "Project analysis complete!", 1)
        return tech_spec


class CoderService(BaseAIService):
    """Enhanced Coder Service with granular code generation streaming"""

    async def generate_file_from_spec(self, file_path: str, tech_spec: dict, project_context: dict,
                                      full_dependency_context: str) -> str:
        agent_name = "Coder"

        self.stream_emitter(agent_name, "thought", f"Starting code generation for '{file_path}'...", 2)

        # Enhanced pre-generation feedback
        purpose = tech_spec.get("purpose", "Unknown purpose")
        self.stream_emitter(agent_name, "info", f"Purpose: {purpose}", 3)

        dependencies = tech_spec.get("dependencies", [])
        if dependencies:
            self.stream_emitter(agent_name, "info", f"Dependencies: {', '.join(dependencies)}", 3)

        # Get RAG context
        rag_context = await self._get_intelligent_rag_context(purpose, k=1)

        # Build the prompt
        code_prompt = CODER_PROMPT_TEMPLATE.format(
            project_context_description=project_context.get("description", ""),
            file_path=file_path,
            full_dependency_context=full_dependency_context,
            rag_context=rag_context,
            tech_spec_json=json.dumps(tech_spec, indent=2)
        )

        self.stream_emitter(agent_name, "thought", "Generating code with streaming output...", 2)

        all_chunks = []
        chunk_count = 0
        last_progress_time = time.time()

        try:
            # Enhanced streaming with progress feedback
            async for chunk in self.llm_client.stream_chat(code_prompt, LLMRole.CODER):
                all_chunks.append(chunk)
                chunk_count += 1

                # Emit the chunk for real-time display
                self.stream_emitter(agent_name, "llm_chunk", chunk, 3)

                # Periodic progress updates
                current_time = time.time()
                if current_time - last_progress_time > 2.0:  # Every 2 seconds
                    estimated_chars = len("".join(all_chunks))
                    self.stream_emitter(agent_name, "debug", f"Generated {estimated_chars} characters...", 4)
                    last_progress_time = current_time

            raw_code = "".join(all_chunks)
            cleaned_code = self._clean_llm_output(raw_code)

            # Enhanced completion feedback
            lines_generated = len(cleaned_code.split('\n'))
            self.stream_emitter(agent_name, "success",
                                f"Code generation complete! {lines_generated} lines generated.", 2)

            return cleaned_code

        except Exception as e:
            self.stream_emitter(agent_name, "error", f"Code generation failed: {e}", 2)
            return f"# FALLBACK: Failed to generate code for {file_path}. Error: {e}\npass"

    def _clean_llm_output(self, code: str) -> str:
        """Clean LLM output with enhanced feedback"""
        original_length = len(code)

        if code.strip().startswith("```python"):
            code = code.split("```python", 1)[-1]
        if code.strip().endswith("```"):
            code = code.rsplit("```", 1)[0]

        cleaned = code.strip()

        # Log cleaning info if significant changes were made
        if len(cleaned) < original_length * 0.9:
            self.stream_emitter("Coder", "debug",
                                f"Cleaned output: {original_length} -> {len(cleaned)} chars", 4)

        return cleaned


class ReviewerService(BaseAIService):
    """Enhanced Reviewer Service with detailed analysis streaming"""

    async def review_code(self, file_path: str, code: str, project_description: str) -> tuple[dict, bool]:
        agent_name = "Reviewer"

        self.stream_emitter(agent_name, "thought", f"Starting comprehensive review of '{file_path}'...", 2)

        # Enhanced pre-review analysis
        lines_of_code = len(code.split('\n'))
        estimated_complexity = self._estimate_complexity(code)

        self.stream_emitter(agent_name, "info", f"Analyzing {lines_of_code} lines of code", 3)
        self.stream_emitter(agent_name, "info", f"Estimated complexity: {estimated_complexity}", 3)

        review_prompt = REVIEWER_PROMPT_TEMPLATE.format(
            file_path=file_path,
            project_description=project_description,
            code=code
        )

        self.stream_emitter(agent_name, "thought", "Performing deep code analysis...", 2)
        review_data = await self._stream_and_collect_json(review_prompt, LLMRole.REVIEWER, agent_name)

        if not review_data:
            self.stream_emitter(agent_name, "error", "Review analysis failed. Approving by default.", 2)
            return {"approved": True, "summary": "Review failed to parse, approved by default."}, True

        # Enhanced feedback processing
        approved = review_data.get('approved', False)
        quality_score = review_data.get('quality_score', 5.0)
        issues = review_data.get('issues', [])

        # Detailed issue analysis with streaming feedback
        if issues:
            critical_issues = [i for i in issues if i.get('severity') == 'critical']
            high_issues = [i for i in issues if i.get('severity') == 'high']
            medium_issues = [i for i in issues if i.get('severity') == 'medium']

            if critical_issues:
                self.stream_emitter(agent_name, "error", f"🚨 {len(critical_issues)} critical issues found", 2)
                for issue in critical_issues[:3]:  # Show first 3
                    self.stream_emitter(agent_name, "error", f"Critical: {issue.get('message', 'Unknown')}", 3)

            if high_issues:
                self.stream_emitter(agent_name, "warning", f"⚠️ {len(high_issues)} high-priority issues", 2)

            if medium_issues:
                self.stream_emitter(agent_name, "info", f"ℹ️ {len(medium_issues)} medium-priority issues", 2)

        # Security analysis
        security_concerns = review_data.get('security_concerns', [])
        if security_concerns:
            self.stream_emitter(agent_name, "security", f"🛡️ {len(security_concerns)} security concerns", 2)
            for concern in security_concerns[:2]:  # Show first 2
                self.stream_emitter(agent_name, "security", f"Security: {concern}", 3)

        # Performance analysis
        performance_notes = review_data.get('performance_notes', [])
        if performance_notes:
            self.stream_emitter(agent_name, "performance", f"⚡ Performance opportunities identified", 2)

        # Final verdict with enhanced feedback
        status_icon = "✅" if approved else "❌"
        status_text = "APPROVED" if approved else "NEEDS REVISION"

        self.stream_emitter(agent_name, "success" if approved else "warning",
                            f"{status_icon} Review complete: Quality {quality_score}/10 - {status_text}", 2)

        return review_data, approved

    def _estimate_complexity(self, code: str) -> str:
        """Estimate code complexity for better review context"""
        try:
            tree = ast.parse(code)

            classes = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
            functions = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
            imports = len([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))])

            if classes > 3 or functions > 10:
                return "High"
            elif classes > 1 or functions > 5:
                return "Medium"
            else:
                return "Low"
        except:
            return "Unknown"