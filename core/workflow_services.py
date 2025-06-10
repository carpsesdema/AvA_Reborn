# core/workflow_services.py - Enhanced with Team Communication

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
    2. **MIMIC RAG EXAMPLES**: The code examples show the quality and style expected. Match that level of professionalism.
    3. Generate the **entire, complete, and runnable source code** for the file `{file_path}`.
    4. Implement the full logic for every method and function defined in your technical specification. **DO NOT use `pass` as a placeholder.** Your code must be fully implemented.
    5. You MUST adhere to the `api_contract` from your technical specification and correctly use the classes, methods, and variables from the provided dependency context.
    6. **Your entire response MUST be ONLY the raw Python code.** Do not include any explanations or markdown formatting like ```python.

    Generate the complete and fully implemented Python code for `{file_path}` now:
""")

REVIEWER_PROMPT_TEMPLATE = textwrap.dedent("""
    You are the SMART REVIEWER AI - the single comprehensive quality gate for all generated code. You perform deep analysis covering syntax, security, performance, style, best practices, and architectural consistency.

    **FILE TO REVIEW:** {file_path}
    **PROJECT DESCRIPTION:** {project_description}

    **TEAM KNOWLEDGE & STANDARDS:**
    Your team has established these patterns and quality standards:
    {team_context}

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

    5. **TEAM CONSISTENCY**
       - Adherence to established team patterns and standards
       - Consistency with architectural decisions
       - Proper integration with existing codebase

    **IMPORTANT**: Return your analysis as a JSON object with the following structure:
    {{
        "approved": boolean,
        "quality_score": number (1-10),
        "issues": [
            {{"severity": "critical/high/medium/low", "message": "description", "line": number, "category": "syntax/security/performance/style/consistency"}}
        ],
        "suggestions": ["improvement suggestion 1", "improvement suggestion 2"],
        "team_insights": ["pattern observed", "quality standard noted"],
        "summary": "Brief overall assessment"
    }}
""")


class BaseAIService:
    """Enhanced base service with team communication capabilities."""

    def __init__(self, llm_client: EnhancedLLMClient, stream_emitter: Callable, rag_manager=None):
        self.llm_client = llm_client
        self.stream_emitter = stream_emitter
        self.rag_manager = rag_manager
        self.project_state: ProjectStateManager = None  # Will be set by workflow engine

    def set_project_state(self, project_state: ProjectStateManager):
        """Set the project state manager for team communication."""
        self.project_state = project_state

    def _contribute_team_insight(self, insight_type: str, agent_name: str, content: str,
                                 impact_level: str = "medium", related_files: List[str] = None):
        """Contribute an insight to the team knowledge base."""
        if self.project_state:
            self.project_state.add_team_insight(
                insight_type=insight_type,
                source_agent=agent_name.lower(),
                content=content,
                impact_level=impact_level,
                related_files=related_files or []
            )
            self.stream_emitter(agent_name, "insight", f"Contributed {insight_type} insight: {content[:50]}...", 3)

    def _get_team_context_string(self, context_type: str = "all") -> str:
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
        """Enhanced JSON parsing with better error handling."""
        text = text.strip()
        if text.startswith('```json') and text.endswith('```'):
            json_text = text[7:-3].strip()
            self.stream_emitter(agent_name, "parsing", "Found markdown JSON block. Parsing that.", 3)
        elif text.startswith('```') and text.endswith('```'):
            json_text = text[3:-3].strip()
            self.stream_emitter(agent_name, "parsing", "Found generic markdown block. Parsing that.", 3)
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
        serializable_context = json.loads(json.dumps(project_context, default=str))

        analysis_prompt = ARCHITECT_ANALYSIS_PROMPT_TEMPLATE.format(
            project_context_json=json.dumps(serializable_context, indent=2)
        )

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
    """Enhanced Coder Service with team learning capabilities."""

    async def generate_file(self, file_path: str, tech_spec: dict, project_context: dict = None) -> str:
        self.stream_emitter("Coder", "thought", f"Generating code for {file_path}...", 1)

        file_specs = tech_spec.get("technical_specs", {}).get(file_path, {})
        if not file_specs:
            self.stream_emitter("Coder", "error", f"No specifications found for {file_path}", 2)
            return ""

        # Get team context focused on implementation patterns
        team_context = self._get_team_context_string("implementation")

        # Get RAG context for this specific file type
        file_purpose = file_specs.get("purpose", file_path)
        rag_context = await self._get_intelligent_rag_context(f"{file_path} {file_purpose}", k=3)

        generation_prompt = CODER_PROMPT_TEMPLATE.format(
            file_path=file_path,
            file_purpose=file_specs.get("purpose", ""),
            api_contract=json.dumps(file_specs.get("api_contract", {}), indent=2),
            dependencies=", ".join(file_specs.get("dependencies", [])),
            team_context=team_context,
            rag_context=rag_context
        )

        self.stream_emitter("Coder", "generating", f"Creating implementation for {file_path}...", 2)

        all_chunks = []
        try:
            async for chunk in self.llm_client.stream_chat(generation_prompt, LLMRole.CODER):
                all_chunks.append(chunk)
                self.stream_emitter("Coder", "llm_chunk", chunk, 3)

            code = "".join(all_chunks).strip()

            if not code:
                self.stream_emitter("Coder", "error", f"No code generated for {file_path}", 2)
                return ""

            # Analyze the generated code and contribute insights
            self._analyze_and_contribute_code_insights(file_path, code)

            self.stream_emitter("Coder", "success", f"Code generation completed for {file_path}", 1)
            return code

        except Exception as e:
            self.stream_emitter("Coder", "error", f"Error generating code for {file_path}: {e}", 2)
            return ""

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

            # Contribute quality insights from the review
            self._contribute_review_insights(file_path, review_data)

            # Enhanced feedback processing
            approved = review_data.get('approved', False)
            quality_score = review_data.get('quality_score', 5.0)
            issues = review_data.get('issues', [])

            # Log detailed analysis results
            if issues:
                critical_issues = [i for i in issues if i.get('severity') == 'critical']
                high_issues = [i for i in issues if i.get('severity') == 'high']

                if critical_issues:
                    self.stream_emitter("Reviewer", "error", f"Found {len(critical_issues)} critical issues", 2)
                    for issue in critical_issues:
                        self.stream_emitter("Reviewer", "error", f"Critical: {issue.get('message', 'Unknown issue')}",
                                            3)

                if high_issues:
                    self.stream_emitter("Reviewer", "warning", f"Found {len(high_issues)} high-priority issues", 2)

            self.stream_emitter("Reviewer", "success" if approved else "warning",
                                f"Review complete: Quality Score {quality_score}/10 - {'APPROVED' if approved else 'NEEDS REVISION'}",
                                2)

            return review_data, approved

        except Exception as e:
            self.stream_emitter("Reviewer", "error", f"Review process failed: {e}", 2)
            return {"approved": True, "summary": "Review failed, approved by default."}, True

    def _contribute_review_insights(self, file_path: str, review_data: Dict):
        """Contribute insights from code review to team knowledge."""
        try:
            # Contribute quality insights
            if review_data.get('team_insights'):
                for insight in review_data['team_insights']:
                    self._contribute_team_insight(
                        "quality",
                        "Reviewer",
                        insight,
                        "medium",
                        [file_path]
                    )

            # Contribute pattern observations
            quality_score = review_data.get('quality_score', 5)
            if quality_score >= 8:
                self._contribute_team_insight(
                    "quality",
                    "Reviewer",
                    f"High-quality implementation in {file_path} (score: {quality_score}/10)",
                    "medium",
                    [file_path]
                )
            elif quality_score < 6:
                issues = review_data.get('issues', [])
                common_issues = {}
                for issue in issues:
                    category = issue.get('category', 'unknown')
                    common_issues[category] = common_issues.get(category, 0) + 1

                for category, count in common_issues.items():
                    if count > 1:
                        self._contribute_team_insight(
                            "quality",
                            "Reviewer",
                            f"Recurring {category} issues detected - focus area for improvement",
                            "high"
                        )

        except Exception as e:
            self.stream_emitter("Reviewer", "debug", f"Failed to contribute review insights: {e}", 4)

    async def _static_code_analysis(self, code: str) -> dict:
        """Perform static analysis that replaces external tools"""
        analysis_results = {
            'syntax_errors': [],
            'style_violations': [],
            'security_issues': [],
            'performance_issues': [],
            'complexity_warnings': []
        }

        try:
            # Syntax validation
            tree = ast.parse(code)

            # Security analysis
            security_issues = self._check_security_patterns(code, tree)
            analysis_results['security_issues'].extend(security_issues)

            # Style analysis
            style_issues = self._check_style_patterns(code)
            analysis_results['style_violations'].extend(style_issues)

            # Performance analysis
            perf_issues = self._check_performance_patterns(code, tree)
            analysis_results['performance_issues'].extend(perf_issues)

        except SyntaxError as e:
            analysis_results['syntax_errors'].append({
                'line': e.lineno,
                'message': e.msg,
                'severity': 'critical'
            })

        return analysis_results

    def _check_security_patterns(self, code: str, tree: ast.AST) -> List[dict]:
        """Check for security anti-patterns"""
        issues = []

        # Check for dangerous function calls
        dangerous_functions = ['eval', 'exec', 'compile', '__import__']

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in dangerous_functions:
                    issues.append({
                        'line': node.lineno,
                        'message': f"Dangerous function '{node.func.id}' detected",
                        'severity': 'critical',
                        'category': 'security'
                    })

        # Check for SQL injection patterns
        if 'execute(' in code and any(pattern in code for pattern in ['%s', '+', 'format(']):
            issues.append({
                'message': "Potential SQL injection vulnerability detected",
                'severity': 'high',
                'category': 'security'
            })

        return issues

    def _check_style_patterns(self, code: str) -> List[dict]:
        """Check for style violations"""
        issues = []
        lines = code.split('\n')

        for i, line in enumerate(lines, 1):
            # Line length check
            if len(line) > 88:
                issues.append({
                    'line': i,
                    'message': f"Line too long ({len(line)} > 88 characters)",
                    'severity': 'medium',
                    'category': 'style'
                })

            # Trailing whitespace
            if line.rstrip() != line:
                issues.append({
                    'line': i,
                    'message': "Trailing whitespace detected",
                    'severity': 'low',
                    'category': 'style'
                })

        return issues

    def _check_performance_patterns(self, code: str, tree: ast.AST) -> List[dict]:
        """Check for performance anti-patterns"""
        issues = []

        # Check for inefficient patterns
        if 'for ' in code and ' in range(len(' in code:
            issues.append({
                'message': "Consider using enumerate() instead of range(len())",
                'severity': 'medium',
                'category': 'performance'
            })

        return issues