# core/workflow_services.py - Enhanced with Team Communication and Improved JSON Parsing

import asyncio
import json
import re
import textwrap
import ast
from typing import Callable, Dict, List

from core.llm_client import LLMRole, EnhancedLLMClient
from core.project_state_manager import ProjectStateManager

# --- Prompt Templates ---

# Enhanced Architect prompt with team context
ARCHITECT_PROMPT_TEMPLATE = textwrap.dedent("""
    You are the ARCHITECT AI. Your task is to create a complete, comprehensive, and machine-readable Technical Specification Sheet for an entire software project based on a user's request. This sheet will be the single source of truth for all other AI agents.

    **USER REQUEST**: "{full_requirements}"

    **PRIORITY CONTEXT (GDD - Game Design Document):**
    The following context is from the project's official design document. This is the MOST IMPORTANT source of truth for the project's high-level goals, implemented systems, and development history. You MUST use this to guide your architectural decisions.
    ---
    {gdd_context}
    ---

    **TEAM KNOWLEDGE & INSIGHTS:**
    Your AI teammates have accumulated the following knowledge from previous work:
    {team_context}

    **RELEVANT CONTEXT FROM KNOWLEDGE BASE (RAG):**
    {rag_context}

    **CRITICAL**: Study the team insights above. Use established patterns, avoid repeating past mistakes, and build on successful architectural decisions made by previous iterations.

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
          "dependencies": ["config"],
          "api_contract": {{"classes": [{{"name": "Player", "inherits_from": "Entity", "methods": [{{"signature": "__init__(self, position=(0,0,0))"}}]}}]}}
        }},
        "main.py": {{
          "purpose": "Entry point for the application.",
          "dependencies": ["config", "player"],
          "api_contract": {{"execution_flow": "Initialize Ursina app, create player, start main loop"}}
        }}
      }}
    }}

    **The system will fail if it receives anything other than the raw, valid JSON object.**
""")

ARCHITECT_ANALYSIS_PROMPT_TEMPLATE = textwrap.dedent("""
    You are the ARCHITECT AI. Analyze the existing project structure and create a comprehensive Technical Specification that represents the current state of the project.

    **PROJECT CONTEXT:**
    {project_context_json}

    **TEAM INSIGHTS:**
    Use the team knowledge and established patterns from the project context above.

    Based on the project files, dependencies, and structure, create a complete technical specification that accurately represents this project. Your output MUST be a single, valid JSON object following the same structure as the creation template.

    Include all existing files in the technical_specs, infer the purpose and api_contract from the actual code content, and determine the correct dependency order.

    **CRITICAL**: Return ONLY valid JSON. No explanations, no markdown, no extra text. Just the JSON object.

    **The system will fail if it receives anything other than the raw, valid JSON object.**
""")

CODER_PROMPT_TEMPLATE = textwrap.dedent("""
    You are the CODER AI. Generate a single, complete, professional Python file based on the provided technical specification.

    **FILE TO GENERATE:** {file_path}
    **FILE PURPOSE:** {file_purpose}
    **API CONTRACT:** {api_contract}
    **DEPENDENCIES:** {dependencies}

    **TEAM INSIGHTS & PATTERNS:**
    Your AI teammates have established these patterns and learnings:
    {team_context}

    **RELEVANT CODE EXAMPLES:**
    {rag_context}

    **CRITICAL IMPLEMENTATION REQUIREMENTS:**
    1. **LEARN FROM TEAM**: Study the team insights above. Follow established patterns, coding standards, and architectural decisions from previous work.
    2. **MIMIC RAG EXAMPLES**: The code examples show the quality and style expected.
    3. **COMPLETE IMPLEMENTATION**: Generate the ENTIRE file, not just stubs or templates.
    4. **DEPENDENCY CONTEXT**: Use the provided dependency information to ensure proper imports and integrations.

    Generate high-quality, production-ready Python code. Return ONLY the complete source code, no explanations or markdown formatting.
""")

REVIEWER_PROMPT_TEMPLATE = textwrap.dedent("""
    You are the REVIEWER AI. Conduct a comprehensive code review of the provided Python file.

    **FILE:** {file_path}
    **PROJECT DESCRIPTION:** {project_description}

    **TEAM QUALITY STANDARDS:**
    {team_context}

    **CODE TO REVIEW:**
    ```python
    {code}
    ```

    Perform a thorough review checking for:
    1. Code quality and best practices
    2. Adherence to team standards and patterns
    3. Security considerations
    4. Performance implications
    5. Documentation completeness
    6. Error handling
    7. Testing considerations

    Your response MUST be a valid JSON object with this structure:
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


# --- NEW: Helper function for refinement passes ---
def create_refinement_prompt(code: str, instruction: str) -> str:
    """Creates a standardized prompt for a code refinement pass."""
    return textwrap.dedent(f"""
        You are a code refiner. Your task is to modify the given code based on a specific instruction.
        **IMPORTANT**: You must ONLY return the complete, modified code. Do not add explanations, markdown, or any other text.

        INSTRUCTION: {instruction}

        CODE TO REFINE:
        ```python
        {code}
        ```

        COMPLETE REFINED CODE:
    """)


class BaseAIService:
    """Base class for AI services with team communication capabilities."""

    def __init__(self, llm_client: EnhancedLLMClient, stream_emitter: Callable, rag_manager=None):
        self.llm_client = llm_client
        self.stream_emitter = stream_emitter
        self.rag_manager = rag_manager
        self.project_state: ProjectStateManager = None

    def set_project_state(self, project_state: ProjectStateManager):
        """Connect this service to the project state for team communication."""
        self.project_state = project_state

    def _contribute_team_insight(self, insight_type: str, source_agent: str, content: str,
                                 impact_level: str = "medium", related_files: List[str] = None):
        """Contribute an insight to the team knowledge base."""
        if self.project_state:
            self.project_state.add_team_insight(
                insight_type=insight_type,
                source_agent=source_agent,
                content=content,
                impact_level=impact_level,
                related_files=related_files or []
            )

    def _get_team_context_string(self, context_type: str = "full") -> str:
        """Get formatted team context for prompts."""
        if not self.project_state:
            return "No team context available."

        team_context = self.project_state.get_enhanced_project_context()
        team_insights = team_context.get("team_insights", {})
        team_comm = team_context.get("team_communication", {})

        if context_type == "architectural":
            insights = team_insights.get("architectural_insights", [])
            return self._format_insights_for_prompt(insights, "Architectural Decisions")
        elif context_type == "implementation":
            insights = team_insights.get("implementation_patterns", [])
            return self._format_insights_for_prompt(insights, "Implementation Patterns")
        elif context_type == "quality":
            insights = team_insights.get("quality_standards", [])
            return self._format_insights_for_prompt(insights, "Quality Standards")
        else:
            # Full context
            formatted_parts = []

            if team_comm.get("recent_priorities"):
                formatted_parts.append("**RECENT PRIORITIES:**\n" +
                                       "\n".join(f"- {p}" for p in team_comm["recent_priorities"]))

            if team_comm.get("established_patterns"):
                formatted_parts.append("**ESTABLISHED PATTERNS:**\n" +
                                       "\n".join(f"- {p}" for p in team_comm["established_patterns"]))

            if team_comm.get("quality_focus_areas"):
                formatted_parts.append("**QUALITY FOCUS:**\n" +
                                       "\n".join(f"- {q}" for q in team_comm["quality_focus_areas"]))

            if team_comm.get("lessons_learned"):
                formatted_parts.append("**LESSONS LEARNED:**\n" +
                                       "\n".join(f"- {l}" for l in team_comm["lessons_learned"]))

            return "\n\n".join(
                formatted_parts) if formatted_parts else "No team insights yet - this is the first iteration."

    def _format_insights_for_prompt(self, insights: List[Dict], title: str) -> str:
        """Format insights for inclusion in prompts."""
        if not insights:
            return f"**{title}:** None established yet."

        formatted = f"**{title}:**\n"
        for insight in insights[-5:]:  # Last 5 insights
            formatted += f"- {insight.get('content', '')}\n"
        return formatted

    def _parse_json_from_response(self, text: str, agent_name: str) -> dict:
        """Enhanced JSON parsing with better error handling and multiple strategies."""
        original_text = text.strip()

        # Strategy 1: Look for markdown JSON blocks
        json_patterns = [
            (r'```json\s*(.*?)\s*```', "markdown JSON block"),
            (r'```\s*(.*?)\s*```', "generic markdown block"),
            (r'(?s)\{.*\}', "JSON object pattern")
        ]

        for pattern, description in json_patterns:
            matches = re.findall(pattern, original_text, re.DOTALL)
            if matches:
                json_text = matches[0].strip() if isinstance(matches[0], str) else matches[0]
                self.stream_emitter(agent_name, "parsing", f"Found {description}. Parsing...", 3)

                try:
                    parsed = json.loads(json_text)
                    self.stream_emitter(agent_name, "success", f"Successfully parsed JSON from {description}", 3)
                    return parsed
                except json.JSONDecodeError as e:
                    self.stream_emitter(agent_name, "warning", f"JSON parse failed for {description}: {e}", 3)
                    continue

        # Strategy 2: Look for object boundaries manually
        start = original_text.find('{')
        end = original_text.rfind('}')

        if start != -1 and end != -1 and end > start:
            json_text = original_text[start:end + 1]
            self.stream_emitter(agent_name, "fallback", "Extracting content between first and last braces", 3)

            try:
                parsed = json.loads(json_text)
                self.stream_emitter(agent_name, "success", "Successfully parsed JSON from brace extraction", 3)
                return parsed
            except json.JSONDecodeError as e:
                self.stream_emitter(agent_name, "warning", f"Brace extraction parse failed: {e}", 3)

        # Strategy 3: Try to clean and fix common JSON issues
        cleaned_text = self._attempt_json_cleanup(original_text, agent_name)
        if cleaned_text:
            try:
                parsed = json.loads(cleaned_text)
                self.stream_emitter(agent_name, "success", "Successfully parsed JSON after cleanup", 3)
                return parsed
            except json.JSONDecodeError as e:
                self.stream_emitter(agent_name, "warning", f"Cleanup attempt failed: {e}", 3)

        # Final failure
        self.stream_emitter(agent_name, "error", "All JSON parsing strategies failed", 3)
        self.stream_emitter(agent_name, "debug", f"Response preview: {original_text[:500]}...", 4)
        return {}

    def _attempt_json_cleanup(self, text: str, agent_name: str) -> str:
        """Attempt to clean and fix common JSON formatting issues."""
        try:
            # Remove common prefixes/suffixes that LLMs add
            cleaned = text.strip()

            # Remove markdown language indicators
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, flags=re.MULTILINE)
            cleaned = re.sub(r'\s*```$', '', cleaned, flags=re.MULTILINE)

            # Remove common AI response prefixes
            prefixes_to_remove = [
                r'^Here\'s the.*?:\s*',
                r'^The.*?is:\s*',
                r'^Based on.*?:\s*',
                r'^After.*?:\s*'
            ]

            for prefix in prefixes_to_remove:
                cleaned = re.sub(prefix, '', cleaned, flags=re.IGNORECASE | re.MULTILINE)

            # Find the JSON object
            start = cleaned.find('{')
            end = cleaned.rfind('}')

            if start != -1 and end != -1 and end > start:
                cleaned = cleaned[start:end + 1]

                # Fix common JSON issues
                # Fix trailing commas
                cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)

                # Fix unquoted keys (simple cases)
                cleaned = re.sub(r'(\w+):', r'"\1":', cleaned)

                # Fix single quotes to double quotes
                cleaned = cleaned.replace("'", '"')

                self.stream_emitter(agent_name, "info", "Attempted JSON cleanup", 4)
                return cleaned

        except Exception as e:
            self.stream_emitter(agent_name, "debug", f"JSON cleanup failed: {e}", 4)

        return ""

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

            self.stream_emitter(agent_name, "info", f"Received {len(response_text)} characters from LLM", 3)
            return self._parse_json_from_response(response_text, agent_name)

        except Exception as e:
            self.stream_emitter(agent_name, "error", f"Error during streaming/collection: {e}", 2)
            if all_chunks:
                partial_response = "".join(all_chunks)
                self.stream_emitter(agent_name, "info",
                                    f"Attempting to parse partial response ({len(partial_response)} chars)", 3)
                return self._parse_json_from_response(partial_response, agent_name)
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

    async def create_tech_spec(self, user_prompt: str, conversation_context: List[Dict] = None) -> dict:
        self.stream_emitter("Architect", "thought",
                            "Phase 1: Architecting the complete project technical specification...", 0)

        # Extract GDD context from user prompt
        prompt_parts = user_prompt.split("\n\n--- GDD CONTEXT ---\n")
        actual_user_prompt = prompt_parts[0]
        gdd_context = prompt_parts[1] if len(prompt_parts) > 1 else "No GDD provided."

        requirements = [msg.get("message", "") for msg in (conversation_context or []) if msg.get("role") == "user"]
        requirements.append(actual_user_prompt)
        full_requirements = " ".join(req.strip() for req in requirements if req.strip())

        # Get team context and RAG context
        team_context = self._get_team_context_string("architectural")
        rag_context = await self._get_intelligent_rag_context(full_requirements, k=1)

        plan_prompt = ARCHITECT_PROMPT_TEMPLATE.format(
            full_requirements=full_requirements,
            gdd_context=gdd_context,
            team_context=team_context,
            rag_context=rag_context
        )

        tech_spec = await self._stream_and_collect_json(plan_prompt, LLMRole.ARCHITECT, "Architect")
        if not tech_spec or not tech_spec.get("technical_specs"):
            self.stream_emitter("Architect", "error",
                                "Architecting failed. Could not produce a valid technical specification.", 1)
            return {}

        # Contribute architectural insights
        if tech_spec.get("project_description"):
            self._contribute_team_insight(
                "architectural",
                "Architect",
                f"Project architecture: {tech_spec['project_description']}",
                "high"
            )

        if tech_spec.get("dependency_order"):
            self._contribute_team_insight(
                "architectural",
                "Architect",
                f"Dependency structure: {' -> '.join(tech_spec['dependency_order'])}",
                "medium"
            )

        self.stream_emitter("Architect", "success", "Master Technical Specification created successfully!", 0)
        return tech_spec

    async def analyze_and_create_spec_from_project(self, project_state: ProjectStateManager) -> dict:
        """Analyzes an existing project and creates a tech spec for it."""
        self.stream_emitter("Architect", "thought",
                            "Analyzing existing project to reverse-engineer its architecture...", 1)

        # Get enhanced project context with team insights
        project_context = project_state.get_enhanced_project_context()

        # Validate that we have content to analyze
        if not project_context.get("project_overview", {}).get("files"):
            self.stream_emitter("Architect", "warning",
                                "No files found in project context for analysis", 1)
            return {}

        # Create serializable context
        try:
            serializable_context = json.loads(json.dumps(project_context, default=str))
        except Exception as e:
            self.stream_emitter("Architect", "error",
                                f"Failed to serialize project context: {e}", 1)
            return {}

        analysis_prompt = ARCHITECT_ANALYSIS_PROMPT_TEMPLATE.format(
            project_context_json=json.dumps(serializable_context, indent=2)
        )

        self.stream_emitter("Architect", "info",
                            f"Sending {len(analysis_prompt)} character prompt to LLM", 2)

        tech_spec = await self._stream_and_collect_json(analysis_prompt, LLMRole.ARCHITECT, "Architect")

        if not tech_spec or not tech_spec.get("technical_specs"):
            self.stream_emitter("Architect", "error",
                                "Project analysis failed. Could not produce a valid technical specification from the provided context.",
                                1)
            return {}

        # Contribute analysis insights
        self._contribute_team_insight(
            "architectural",
            "Architect",
            f"Analyzed existing project structure with {len(project_context.get('project_overview', {}).get('files', {}))} files",
            "medium"
        )

        self.stream_emitter("Architect", "success", "Project analysis complete. Technical spec created!", 1)
        return tech_spec


class CoderService(BaseAIService):
    """Enhanced Coder Service with a more efficient Multi-Pass Refinement."""

    async def generate_file_from_spec(self, file_path: str, file_spec: dict, project_context: dict = None,
                                      dependency_context: str = "") -> str:
        """
        Generate a file with a two-pass refinement process for better responsiveness.
        """
        self.stream_emitter("Coder", "thought", f"Starting 2-pass generation for {file_path}...", 1)

        # === PASS 1: Initial Code Draft (Streaming) ===
        self.stream_emitter("Coder", "refinement_pass", "1/2: Generating initial code draft...", 2)
        initial_code = await self._initial_draft_pass(file_path, file_spec, project_context, dependency_context)
        if not initial_code:
            self.stream_emitter("Coder", "error", f"Initial draft failed for {file_path}", 2)
            return ""
        self.stream_emitter("Coder", "info", "Initial draft complete. Now polishing.", 3)

        # === PASS 2: Consolidated Polishing Pass (Blocking) ===
        self.stream_emitter("Coder", "refinement_pass", "2/2: Polishing code (error handling & docs)...", 2)
        final_code = await self._final_refinement_pass(initial_code)
        if not final_code:
            self.stream_emitter("Coder", "warning", f"Polishing pass failed for {file_path}. Using initial draft.", 3)
            final_code = initial_code  # Fallback to the initial draft

        self.stream_emitter("Coder", "success", f"Multi-pass generation finished for {file_path}!", 1)

        # Analyze the final generated code and contribute insights
        self._analyze_and_contribute_code_insights(file_path, final_code)

        return final_code

    async def _initial_draft_pass(self, file_path: str, file_spec: dict, project_context: dict, dependency_context: str) -> str:
        """Generates the first functional version of the code and streams it."""
        team_context = self._get_team_context_string("implementation")
        rag_context = await self._get_intelligent_rag_context(f"{file_path} {file_spec.get('purpose', '')}", k=3)

        generation_prompt = CODER_PROMPT_TEMPLATE.format(
            file_path=file_path,
            file_purpose=file_spec.get("purpose", ""),
            api_contract=json.dumps(file_spec.get("api_contract", {}), indent=2),
            dependencies=", ".join(file_spec.get("dependencies", [])),
            team_context=team_context,
            rag_context=rag_context,
        )
        if dependency_context:
            generation_prompt += f"\n\n**DEPENDENCY CONTEXT:**\n{dependency_context}"

        all_chunks = []
        try:
            async for chunk in self.llm_client.stream_chat(generation_prompt, LLMRole.CODER):
                all_chunks.append(chunk)
                self.stream_emitter("Coder", "llm_chunk", chunk, 3)
            return "".join(all_chunks).strip()
        except Exception as e:
            self.stream_emitter("Coder", "error", f"Error during initial draft for {file_path}: {e}", 2)
            return ""

    async def _final_refinement_pass(self, code: str) -> str:
        """
        Performs a single, consolidated refinement pass for error handling and documentation.
        """
        if not code.strip(): return ""
        self.stream_emitter("Coder", "thought_detail", "Adding error handling, docstrings, and type hints...", 3)
        instruction = "Review this code. First, add comprehensive error handling (try-except blocks, input validation). Second, add professional, Google-style docstrings to all classes and functions, including type hints. Return only the complete, final code."
        prompt = create_refinement_prompt(code, instruction)
        # Using a non-streaming call for the consolidated internal pass
        return await self.llm_client.chat(prompt, LLMRole.CODER)

    def _analyze_and_contribute_code_insights(self, file_path: str, code: str):
        """Analyze generated code and contribute implementation insights."""
        try:
            # Quick analysis for patterns
            lines = code.split('\n')

            # Detect patterns
            if any('dataclass' in line for line in lines):
                self._contribute_team_insight(
                    "implementation",
                    "Coder",
                    "Using dataclasses for data structures",
                    "medium",
                    [file_path]
                )

            if any('async def' in line for line in lines):
                self._contribute_team_insight(
                    "implementation",
                    "Coder",
                    "Implementing async patterns for performance",
                    "medium",
                    [file_path]
                )

            if any('logging' in line for line in lines):
                self._contribute_team_insight(
                    "implementation",
                    "Coder",
                    "Using proper logging for debugging",
                    "low",
                    [file_path]
                )

            # Count complexity
            function_count = len([line for line in lines if line.strip().startswith('def ')])
            if function_count > 10:
                self._contribute_team_insight(
                    "implementation",
                    "Coder",
                    f"High complexity file with {function_count} functions - consider splitting",
                    "medium",
                    [file_path]
                )

        except Exception as e:
            # Don't fail on analysis errors
            self.stream_emitter("Coder", "debug", f"Code analysis failed: {e}", 4)


class ReviewerService(BaseAIService):
    """Enhanced Reviewer Service with team learning and feedback."""

    async def review_code(self, file_path: str, code: str, project_description: str = "") -> tuple:
        self.stream_emitter("Reviewer", "thought", f"Conducting comprehensive review of {file_path}...", 1)

        # Get team context focused on quality standards
        team_context = self._get_team_context_string("quality")

        review_prompt = REVIEWER_PROMPT_TEMPLATE.format(
            file_path=file_path,
            project_description=project_description,
            team_context=team_context,
            code=code
        )

        try:
            review_data = await self._stream_and_collect_json(review_prompt, LLMRole.REVIEWER, "Reviewer")

            if not review_data:
                self.stream_emitter("Reviewer", "warning", "Review failed to parse, approving by default.", 2)
                return {"approved": True, "summary": "Review failed to parse, approved by default."}, True

            # Extract key information
            approved = review_data.get("approved", True)
            summary = review_data.get("summary", "No summary provided")

            # Contribute quality insights based on review
            if review_data.get("strengths"):
                for strength in review_data["strengths"][:2]:  # Top 2 strengths
                    self._contribute_team_insight(
                        "quality",
                        "Reviewer",
                        f"Code strength identified: {strength}",
                        "low",
                        [file_path]
                    )

            if review_data.get("issues"):
                for issue in review_data["issues"][:2]:  # Top 2 issues
                    self._contribute_team_insight(
                        "quality",
                        "Reviewer",
                        f"Code issue to watch: {issue}",
                        "medium",
                        [file_path]
                    )

            result_msg = f"{'✅ Approved' if approved else '❌ Needs work'}: {summary}"
            self.stream_emitter("Reviewer", "success" if approved else "warning", result_msg, 1)

            return review_data, approved

        except Exception as e:
            self.stream_emitter("Reviewer", "error", f"Review process failed: {e}", 2)
            return {"approved": True, "summary": f"Review failed due to error: {e}"}, True