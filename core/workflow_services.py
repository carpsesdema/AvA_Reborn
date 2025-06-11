# core/workflow_services.py - V6.0 with Hybrid Workflow Support

import json
import re
import textwrap
import logging
from typing import Dict, List, Any, Callable, Optional

from core.llm_client import EnhancedLLMClient, LLMRole
from core.project_state_manager import ProjectStateManager
from core.enhanced_micro_task_engine import SimpleTaskSpec

# Enhanced prompt templates for hybrid workflow
ARCHITECT_PROMPT_TEMPLATE = textwrap.dedent("""
    You are the ARCHITECT AI. Create a comprehensive technical specification for this project.

    **USER REQUEST:** {user_prompt}
    **CONVERSATION CONTEXT:** {conversation_context}
    **GDD CONTEXT:** {gdd_context}

    **TEAM INSIGHTS & PATTERNS:**
    {team_context}

    **KNOWLEDGE BASE CONTEXT:**
    {rag_context}

    Create a detailed technical specification that includes:
    1. **Project Overview:** Clear description and objectives
    2. **File Structure:** Complete breakdown of required files with dependencies
    3. **Component Specifications:** For each file, provide detailed component breakdown including:
       - task_id: Unique identifier for each component
       - description: What the component does
       - component_type: function, class, method, etc.
       - core_logic_steps: Step-by-step implementation guide
       - error_conditions_to_handle: Specific error scenarios
       - interactions: How it connects with other components
       - critical_notes: Security, performance, or architectural constraints
    4. **Implementation Standards:** Coding patterns and best practices
    5. **Quality Criteria:** Review standards and testing requirements

    Your response MUST be a valid JSON object with this structure:
    {{
      "project_name": "string",
      "project_description": "string", 
      "technical_specs": {{
        "files": {{
          "filename": {{
            "purpose": "string",
            "dependencies": ["list", "of", "files"],
            "components": [
              {{
                "task_id": "unique_id",
                "description": "component description",
                "component_type": "function|class|method",
                "core_logic_steps": ["step1", "step2"],
                "error_conditions_to_handle": ["error1", "error2"],
                "interactions": ["component1", "component2"],
                "critical_notes": "security/performance notes"
              }}
            ]
          }}
        }},
        "requirements": ["package1", "package2"],
        "coding_standards": "PEP 8 compliance, type hints, docstrings",
        "project_patterns": "Architecture patterns to follow"
      }}
    }}

    Return ONLY the JSON object, no explanations or markdown formatting.
""")

HYBRID_CODER_PROMPT_TEMPLATE = textwrap.dedent("""
    You are the CODER AI in the AvA hybrid system. Execute this micro-task with Gemini Flash efficiency.

    **MICRO-TASK SPECIFICATION:**
    {task_spec}

    **PROJECT CONTEXT:**
    {project_context}

    **TEAM PATTERNS & STANDARDS:**
    {team_context}

    **IMPLEMENTATION REQUIREMENTS:**
    1. Follow the exact specifications in the micro-task
    2. Implement robust error handling as specified
    3. Write clean, efficient code optimized for the component type
    4. Consider integration points with other components
    5. Follow security best practices (no eval, sanitize inputs)

    **CRITICAL OUTPUT REQUIREMENT:**
    You MUST respond with ONLY a valid JSON object in this exact structure:

    {{
      "IMPLEMENTED_CODE": "actual_python_code_here_as_string",
      "IMPLEMENTATION_NOTES": "technical_decisions_and_reasoning",
      "INTEGRATION_HINTS": "how_to_assemble_with_other_components", 
      "EDGE_CASES_HANDLED": "list_of_edge_cases_addressed",
      "TESTING_CONSIDERATIONS": "unit_testing_guidance"
    }}

    Do not include any text, explanations, or markdown formatting. Return ONLY the JSON object.
""")

ASSEMBLER_PROMPT_TEMPLATE = textwrap.dedent("""
    You are the ASSEMBLER AI. Combine these micro-task implementations into a complete, professional Python file.

    **FILE TO ASSEMBLE:** {filename}
    **FILE SPECIFICATIONS:** {file_spec}

    **MICRO-TASK IMPLEMENTATIONS:**
    {micro_task_results}

    **PROJECT CONTEXT:**
    {project_context}

    **ASSEMBLY REQUIREMENTS:**
    1. **Code Organization:**
       - Organize imports (stdlib, third-party, local)
       - Place constants and configuration at top
       - Order classes and functions logically
       - Add proper file-level docstring

    2. **Integration:**
       - Extract IMPLEMENTED_CODE from each micro-task result
       - Ensure consistent naming and style
       - Add necessary glue code for component interaction
       - Verify all dependencies are imported
       - Remove duplicate imports

    3. **Quality Assurance:**
       - Follow project conventions and patterns
       - Add comprehensive documentation
       - Ensure logical flow and organization
       - Verify completeness against specifications

    4. **Professional Polish:**
       - Add file header with description
       - Ensure proper error handling throughout
       - Format according to PEP 8 standards
       - Create production-ready code

    Return ONLY the complete, assembled Python code. No explanations or formatting.
""")

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


class BaseAIService:
    """Base class for AI services with team communication and model selection."""

    def __init__(self, llm_client: EnhancedLLMClient, stream_emitter: Callable, rag_manager=None):
        self.llm_client = llm_client
        self.stream_emitter = stream_emitter
        self.rag_manager = rag_manager
        self.project_state: ProjectStateManager = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def set_project_state(self, project_state: ProjectStateManager):
        """Connect this service to project state for team communication."""
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

        try:
            insights = self.project_state.get_team_insights(context_type)
            if not insights:
                return "No team insights available yet."

            context_parts = []
            for insight in insights[:10]:  # Limit to recent insights
                context_parts.append(
                    f"- {insight.get('insight_type', 'general').upper()}: {insight.get('content', '')}"
                )

            return "\n".join(context_parts)
        except Exception as e:
            self.logger.warning(f"Failed to get team context: {e}")
            return "Team context unavailable."

    def _parse_json_from_response(self, response_text: str, agent_name: str) -> dict:
        """Parse JSON from LLM response with enhanced error handling."""
        try:
            # Try direct JSON parsing first
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            # Try to extract JSON from markdown or other formatting
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass

            # Try to find JSON object in the text
            brace_start = response_text.find('{')
            brace_end = response_text.rfind('}')
            if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                try:
                    return json.loads(response_text[brace_start:brace_end + 1])
                except json.JSONDecodeError:
                    pass

            self.stream_emitter(agent_name, "error",
                                f"Failed to parse JSON response. Raw response: {response_text[:200]}...", 3)
            return {}

    async def _stream_and_collect_json(self, prompt: str, role: LLMRole, agent_name: str) -> dict:
        """Stream response and collect into JSON with error handling."""
        try:
            all_chunks = []

            async for chunk in self.llm_client.stream_chat(prompt, role=role):
                if chunk:
                    all_chunks.append(chunk)
                    self.stream_emitter(agent_name, "stream", chunk, 4)

            response_text = "".join(all_chunks).strip()

            if not response_text:
                self.stream_emitter(agent_name, "error", "Empty response from LLM", 2)
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
        """Get intelligent RAG context for the query."""
        if not self.rag_manager or not self.rag_manager.is_ready:
            return "No RAG context available."

        self.stream_emitter("RAG", "thought", f"Generating context for: '{query}'", 4)
        dynamic_query = f"Python code example for {query}"

        try:
            results = self.rag_manager.query_context(dynamic_query, k=k)
            self.stream_emitter("RAG", "success", "Context retrieval complete.", 4)

            if not results:
                return "No specific examples found in the knowledge base."

            return "\n\n---\n\n".join([
                f"Relevant Example from '{r.get('metadata', {}).get('filename', 'Unknown')}':\n```python\n{r.get('content', '')[:700]}...\n```"
                for r in results if r.get('content')
            ])
        except Exception as e:
            self.stream_emitter("RAG", "error", f"Failed to query RAG: {e}", 4)
            return "Could not query knowledge base due to an error."


class ArchitectService(BaseAIService):
    """Enhanced Architect Service for hybrid workflow planning."""

    async def create_tech_spec(self, user_prompt: str, conversation_context: List[Dict] = None) -> dict:
        """Create technical specification with detailed component breakdown."""
        self.stream_emitter("Architect", "thought",
                            "Phase 1: Creating comprehensive project architecture with micro-task breakdown...", 0)

        # Parse user prompt and GDD context
        prompt_parts = user_prompt.split("\n\n--- GDD CONTEXT ---\n")
        actual_user_prompt = prompt_parts[0]
        gdd_context = prompt_parts[1] if len(prompt_parts) > 1 else "No GDD provided."

        # Get team context and RAG context
        team_context = self._get_team_context_string()
        rag_context = await self._get_intelligent_rag_context(actual_user_prompt)

        # Build conversation context string
        conversation_str = ""
        if conversation_context:
            conversation_str = "\n".join([
                f"{msg.get('role', 'user')}: {msg.get('message', '')}"
                for msg in conversation_context[-3:]  # Last 3 messages for context
            ])

        # Create architecture prompt
        architecture_prompt = ARCHITECT_PROMPT_TEMPLATE.format(
            user_prompt=actual_user_prompt,
            conversation_context=conversation_str,
            gdd_context=gdd_context,
            team_context=team_context,
            rag_context=rag_context
        )

        self.stream_emitter("Architect", "info",
                            f"Sending architecture prompt ({len(architecture_prompt)} chars) to big model", 2)

        # Use big model for architecture
        tech_spec = await self._stream_and_collect_json(architecture_prompt, LLMRole.ARCHITECT, "Architect")

        if not tech_spec or not tech_spec.get("technical_specs"):
            self.stream_emitter("Architect", "error",
                                "Architecture failed. Could not produce valid technical specification.", 1)
            return {}

        # Contribute architectural insights
        self._contribute_team_insight(
            "architectural",
            "Architect",
            f"Created architecture for project: {tech_spec.get('project_name', 'Unknown')}",
            "high"
        )

        self.stream_emitter("Architect", "success", "Architecture complete with detailed component breakdown!", 1)
        return tech_spec

    async def analyze_and_create_spec_from_project(self, project_state: ProjectStateManager) -> dict:
        """Analyze existing project and create technical specification."""
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

        # Create analysis prompt using the template
        analysis_prompt = textwrap.dedent(f"""
            You are the ARCHITECT AI. Analyze the existing project structure and create a comprehensive Technical Specification that represents the current state of the project.

            **PROJECT CONTEXT:**
            {json.dumps(serializable_context, indent=2)}

            **TEAM INSIGHTS:**
            Use the team knowledge and established patterns from the project context above.

            Based on the project files, dependencies, and structure, create a complete technical specification that accurately represents this project. Your output MUST be a single, valid JSON object following the same structure as the creation template.

            Include all existing files in the technical_specs, infer the purpose and api_contract from the actual code content, and determine the correct dependency order.

            **CRITICAL**: Return ONLY valid JSON. No explanations, no markdown, no extra text. Just the JSON object.

            **The system will fail if it receives anything other than the raw, valid JSON object.**
        """)

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

        self.stream_emitter("Architect", "success", "Project analysis complete. Technical spec created!", 1)
        return tech_spec


class CoderService(BaseAIService):
    """Enhanced Coder Service with hybrid workflow and model selection support."""

    async def execute_micro_task_with_gemini_flash(self, task: SimpleTaskSpec) -> Dict[str, Any]:
        """Execute a micro-task using Gemini Flash for cost efficiency."""
        try:
            self.stream_emitter("Coder", "info",
                                f"🚀 Executing micro-task {task.id} with Gemini Flash", 3)

            # Get team context and project patterns
            team_context = self._get_team_context_string()

            # Prepare task specification
            task_spec_json = {
                "id": task.id,
                "description": task.description,
                "component_type": task.component_type,
                "context": task.context,
                "requirements": task.exact_requirements,
                "expected_lines": task.expected_lines
            }

            # Build project context
            project_context = {
                "file_path": task.file_path,
                "component_type": task.component_type,
                "coding_standards": "PEP 8, type hints, comprehensive docstrings",
                "security_requirements": "No eval(), sanitize inputs, handle errors"
            }

            # Create hybrid coder prompt
            prompt = HYBRID_CODER_PROMPT_TEMPLATE.format(
                task_spec=json.dumps(task_spec_json, indent=2),
                project_context=json.dumps(project_context, indent=2),
                team_context=team_context
            )

            self.stream_emitter("Coder", "info",
                                f"Sending micro-task to Gemini Flash ({len(prompt)} chars)", 4)

            # Use Gemini Flash (fast, cost-effective model) for implementation
            # Note: This assumes LLMRole.CODER is configured to use Gemini Flash
            result = await self._stream_and_collect_json(prompt, LLMRole.CODER, "Coder")

            if not result or "IMPLEMENTED_CODE" not in result:
                raise Exception("Micro-task execution failed - no code generated")

            # Validate JSON structure
            required_keys = ["IMPLEMENTED_CODE", "IMPLEMENTATION_NOTES", "INTEGRATION_HINTS",
                             "EDGE_CASES_HANDLED", "TESTING_CONSIDERATIONS"]

            for key in required_keys:
                if key not in result:
                    result[key] = f"Not provided for {key}"

            self.stream_emitter("Coder", "success",
                                f"✅ Micro-task {task.id} completed successfully", 3)

            # Contribute implementation insights
            self._contribute_team_insight(
                "implementation",
                "Coder",
                f"Implemented {task.component_type}: {task.description[:50]}...",
                "low",
                [task.file_path] if task.file_path else []
            )

            return result

        except Exception as e:
            self.logger.error(f"Micro-task execution failed for {task.id}: {e}")
            self.stream_emitter("Coder", "error",
                                f"❌ Micro-task {task.id} failed: {str(e)}", 3)
            raise

    async def generate_file_from_spec(self, file_path: str, file_spec: dict,
                                      project_context: dict, dependency_context: str) -> str:
        """Generate complete file from specification (fallback method)."""
        self.stream_emitter("Coder", "info", f"Generating {file_path} using traditional approach", 2)

        # Build comprehensive context
        team_context = self._get_team_context_string()
        rag_context = await self._get_intelligent_rag_context(f"{file_path} {file_spec.get('purpose', '')}")

        # Create traditional file generation prompt
        prompt = f"""
        Generate a complete Python file based on this specification.

        **FILE:** {file_path}
        **PURPOSE:** {file_spec.get('purpose', 'No purpose specified')}
        **DEPENDENCIES:** {dependency_context}

        **TEAM PATTERNS:**
        {team_context}

        **RAG CONTEXT:**
        {rag_context}

        **SPECIFICATION:**
        {json.dumps(file_spec, indent=2)}

        Generate complete, production-ready Python code following all specifications.
        Return ONLY the Python code, no explanations or markdown formatting.
        """

        # Use big model for complex file generation
        response = await self.llm_client.chat(prompt, role=LLMRole.CODER)

        # Clean code output
        cleaned_code = self._clean_code_output(response)

        self.stream_emitter("Coder", "success", f"✅ Generated {file_path} ({len(cleaned_code)} chars)", 2)

        return cleaned_code

    def _clean_code_output(self, code: str) -> str:
        """Remove markdown fences and clean code output."""
        # Handle markdown fences
        match = re.search(r"```(?:python|py)?\s*\n(.*?)\n\s*```", code, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Fallback for code without language specifier
        match = re.search(r"```\s*\n(.*?)\n\s*```", code, re.DOTALL)
        if match:
            return match.group(1).strip()

        return code.strip()


class AssemblerService(BaseAIService):
    """New Assembler Service for combining micro-task outputs."""

    async def assemble_file_from_micro_tasks(self, filename: str, file_spec: dict,
                                             micro_task_results: List[Dict[str, Any]],
                                             project_context: dict) -> str:
        """Assemble micro-task results into a complete, professional file."""
        try:
            self.stream_emitter("Assembler", "info",
                                f"🔧 Assembling {len(micro_task_results)} micro-tasks for {filename}", 2)

            # Prepare micro-task results for prompt
            results_text = []
            for i, result in enumerate(micro_task_results):
                results_text.append(f"=== Micro-Task {i + 1} ===")
                results_text.append(json.dumps(result, indent=2))
                results_text.append("")

            # Build assembly prompt
            prompt = ASSEMBLER_PROMPT_TEMPLATE.format(
                filename=filename,
                file_spec=json.dumps(file_spec, indent=2),
                micro_task_results="\n".join(results_text),
                project_context=json.dumps(project_context, indent=2)
            )

            self.stream_emitter("Assembler", "info",
                                f"Sending assembly request ({len(prompt)} chars) to medium model", 3)

            # Use medium model for assembly (balance of capability and cost)
            assembled_code = await self.llm_client.chat(prompt, role=LLMRole.ASSEMBLER)

            # Clean the assembled code
            cleaned_code = self._clean_code_output(assembled_code)

            if not cleaned_code or len(cleaned_code) < 50:
                raise Exception("Assembly produced insufficient code")

            self.stream_emitter("Assembler", "success",
                                f"✅ Assembly complete for {filename} ({len(cleaned_code)} chars)", 2)

            # Contribute assembly insights
            self._contribute_team_insight(
                "assembly",
                "Assembler",
                f"Assembled {len(micro_task_results)} components into {filename}",
                "medium",
                [filename]
            )

            return cleaned_code

        except Exception as e:
            self.logger.error(f"Assembly failed for {filename}: {e}")
            self.stream_emitter("Assembler", "error",
                                f"❌ Assembly failed for {filename}: {str(e)}", 2)
            raise

    def _clean_code_output(self, code: str) -> str:
        """Clean assembled code output."""
        # Remove markdown fences
        match = re.search(r"```(?:python|py)?\s*\n(.*?)\n\s*```", code, re.DOTALL)
        if match:
            return match.group(1).strip()

        return code.strip()


class ReviewerService(BaseAIService):
    """Enhanced Reviewer Service with team learning."""

    async def review_code(self, file_path: str, code: str, project_description: str = "") -> dict:
        """Review code and provide comprehensive feedback."""
        try:
            self.stream_emitter("Reviewer", "info", f"🧐 Reviewing {file_path}", 2)

            # Get team quality standards
            team_context = self._get_team_context_string("quality")

            # Build review prompt
            prompt = REVIEWER_PROMPT_TEMPLATE.format(
                file_path=file_path,
                project_description=project_description,
                team_context=team_context,
                code=code
            )

            self.stream_emitter("Reviewer", "info", "Conducting comprehensive code review...", 3)

            # Use big model for thorough review
            review_result = await self._stream_and_collect_json(prompt, LLMRole.REVIEWER, "Reviewer")

            if not review_result:
                raise Exception("Review failed to produce results")

            # Contribute review insights
            if review_result.get("approved"):
                self._contribute_team_insight(
                    "quality",
                    "Reviewer",
                    f"Code approved for {file_path}: {review_result.get('summary', '')}",
                    "medium",
                    [file_path]
                )
            else:
                self._contribute_team_insight(
                    "quality",
                    "Reviewer",
                    f"Code issues found in {file_path}: {', '.join(review_result.get('issues', []))}",
                    "high",
                    [file_path]
                )

            self.stream_emitter("Reviewer", "success",
                                f"✅ Review complete for {file_path} - {'Approved' if review_result.get('approved') else 'Needs work'}",
                                2)

            return review_result

        except Exception as e:
            self.logger.error(f"Review failed for {file_path}: {e}")
            self.stream_emitter("Reviewer", "error",
                                f"❌ Review failed for {file_path}: {str(e)}", 2)
            return {
                "approved": False,
                "summary": f"Review failed due to error: {str(e)}",
                "strengths": [],
                "issues": [f"Review process failed: {str(e)}"],
                "suggestions": ["Fix review process errors"],
                "confidence": 0.0
            }