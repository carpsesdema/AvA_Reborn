# core/services/base_service.py

import json
import re
import logging
import asyncio
from typing import Dict, List, Any, Callable, Optional

from core.llm_client import EnhancedLLMClient, LLMRole
from core.project_state_manager import ProjectStateManager


class BaseAIService:
    """Base class for AI services with team communication and model selection."""

    def __init__(self, llm_client: EnhancedLLMClient, stream_emitter: Callable, rag_manager=None):
        self.llm_client = llm_client
        self.stream_emitter = stream_emitter
        self.rag_manager = rag_manager
        self.project_state: Optional[ProjectStateManager] = None
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

    def _get_team_context_string(self, for_file: str = None) -> str:
        """
        Get a dynamically filtered context string including team insights
        and a summary of the analyzed domain context.
        """
        if not self.project_state:
            return "No project context available."

        try:
            agent_name = self.__class__.__name__
            context_data = self.project_state.get_enhanced_project_context(
                for_file=for_file,
                ai_role=agent_name
            )

            context_parts = []

            # --- NEW: Add Domain Context Summary ---
            if self.project_state.domain_context:
                context_parts.append("**Project Domain Analysis**")
                domain = self.project_state.domain_context
                db_summary = f"- Database: {len(domain.get('database_schema', {}).get('tables', []))} tables found."
                api_summary = f"- API: {len(domain.get('api_definition', {}).get('endpoints', []))} endpoints found."
                fw_summary = "- Frameworks: " + ", ".join(
                    [fw['name'] for fw in domain.get('frameworks', []) if fw.get('confidence', 0) > 0.3])
                context_parts.extend([db_summary, api_summary, fw_summary, "\n"])

            # Add Team Insights
            insights = context_data.get("team_insights", [])
            if insights:
                context_parts.append("**Relevant Team Insights**")
                for insight in insights:
                    related = f"(Related: {', '.join(insight['related_files'])})" if insight.get(
                        'related_files') else ""
                    context_parts.append(
                        f"- From {insight['source_agent']}: {insight['content']} {related}"
                    )

            return "\n".join(context_parts) if context_parts else "No relevant project context found."

        except Exception as e:
            self.logger.warning(f"Failed to get and format team context: {e}", exc_info=True)
            return "Team context is currently unavailable due to an internal error."

    def _parse_json_from_response(self, response_text: str, agent_name: str) -> dict:
        """A more robust JSON parser that handles markdown fences and other LLM quirks."""
        response_text = response_text.strip()

        # Pattern to find JSON within ```json ... ``` or ``` ... ```
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)

        if json_match:
            json_str = json_match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                self.stream_emitter(agent_name, "error", f"JSON parsing failed inside markdown: {e}", 3)
                self.logger.error(f"Failed to parse extracted JSON: {json_str[:500]}")
                # Fall through to the next method

        # If no markdown found or parsing failed, try finding the first '{' and last '}'
        brace_start = response_text.find('{')
        brace_end = response_text.rfind('}')

        if brace_start != -1 and brace_end > brace_start:
            json_candidate = response_text[brace_start:brace_end + 1]
            try:
                return json.loads(json_candidate)
            except json.JSONDecodeError as e:
                self.stream_emitter(agent_name, "error", f"JSON parsing failed on substring: {e}", 3)
                self.logger.error(f"Failed to parse substring JSON: {json_candidate[:500]}")
                # Fall through

        # Final attempt: try loading the whole string directly
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass  # Already logged implicitly or by other attempts

        self.stream_emitter(agent_name, "error", f"All JSON parsing methods failed. Preview: {response_text[:200]}...",
                            3)
        return {}

    async def _stream_and_collect_json(self, prompt: str, role: LLMRole, agent_name: str) -> dict:
        """Streams response and collects into a single JSON object."""
        all_chunks = []
        try:
            async for chunk in self.llm_client.stream_chat(prompt, role=role):
                all_chunks.append(chunk)
        except Exception as e:
            self.stream_emitter(agent_name, "error", f"LLM stream failed: {e}", 2)
            self.logger.error(f"LLM streaming call failed for {agent_name}: {e}", exc_info=True)
            return {}

        full_response = "".join(all_chunks)
        if not full_response:
            self.stream_emitter(agent_name, "error", "Received empty response from LLM.", 2)
            return {}

        return self._parse_json_from_response(full_response, agent_name)

    async def _get_intelligent_rag_context(self, query: str, k: int = 2) -> str:
        """Get intelligent RAG context for the query."""
        if not self.rag_manager or not self.rag_manager.is_ready:
            return "No RAG context available."

        self.stream_emitter("RAG", "thought", f"Generating context for: '{query}'", 4)
        dynamic_query = f"Python code example for {query}"

        try:
            # THE FIX: This is now an async call to prevent UI freezes.
            results = await self.rag_manager.query_context_async(dynamic_query, k=k)

            self.stream_emitter("RAG", "success", "Context retrieval complete.", 4)
            if not results: return "No specific examples found in the knowledge base."
            return "\n\n---\n\n".join([
                f"Relevant Example from '{r.get('metadata', {}).get('filename', 'Unknown')}':\n```python\n{r.get('content', '')[:700]}...\n```"
                for r in results if r.get('content')
            ])
        except Exception as e:
            self.stream_emitter("RAG", "error", f"Failed to query RAG: {e}", 4)
            self.logger.error(f"Error during RAG query: {e}", exc_info=True)
            return "Could not query knowledge base due to an error."