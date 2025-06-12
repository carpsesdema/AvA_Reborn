# core/services/base_service.py

import json
import re
import asyncio
import logging
from typing import Callable, Optional, List, Dict, Any

from core.llm_client import EnhancedLLMClient, LLMRole
from core.project_state_manager import ProjectStateManager
from core.rag_manager import RAGManager


class BaseAIService:
    """
    Base class for all AI agent services.
    Provides shared functionality for LLM interaction, state management,
    and structured logging.
    """

    def __init__(
            self,
            llm_client: EnhancedLLMClient,
            stream_emitter: Callable[[str, str, str, str], None],
            rag_manager: Optional[RAGManager] = None
    ):
        self.llm_client = llm_client
        self.stream_emitter = stream_emitter
        self.rag_manager = rag_manager
        self.project_state_manager: Optional[ProjectStateManager] = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def set_project_state(self, project_state_manager: ProjectStateManager):
        """Sets the project state for the service."""
        self.project_state_manager = project_state_manager

    def _get_team_context_string(self, for_file: Optional[str] = None) -> str:
        """
        Retrieves relevant team context (insights) from the project state.
        If for_file is provided, it prioritizes file-specific insights.
        """
        if not self.project_state_manager:
            return "No project state available."

        # CORRECTED LOGIC: Access the attribute directly and then filter.
        all_insights = self.project_state_manager.team_insights

        if for_file:
            # Filter for insights that are globally relevant (no related files)
            # or specifically related to the file in question.
            insights = [
                insight for insight in all_insights
                if not insight.related_files or for_file in insight.related_files
            ]
        else:
            insights = all_insights

        if not insights:
            return "No relevant team insights or patterns have been established yet."

        context_str = "### Established Team Insights & Patterns\n"
        # Sort by relevance score if available (assuming higher is better)
        try:
            insights.sort(key=lambda x: float(x.relevance_score), reverse=True)
        except (ValueError, TypeError):
            pass  # Ignore sorting if relevance_score is not a number

        for insight in insights:
            context_str += f"- **Type:** {insight.insight_type.capitalize()} | "
            context_str += f"**Source:** {insight.source_agent} | "
            context_str += f"**Relevance:** {insight.relevance_score}\n"
            context_str += f"  **Insight:** {insight.content}\n"
            if insight.related_files:
                context_str += f"  **Related Files:** {', '.join(insight.related_files)}\n"
        return context_str

    def _contribute_team_insight(
            self,
            insight_type: str,
            source_agent: str,
            content: str,
            relevance_score: str,
            related_files: Optional[List[str]] = None
    ):
        """Contributes a new insight to the project state."""
        if self.project_state_manager:
            self.project_state_manager.add_insight(
                insight_type, source_agent, content, relevance_score, related_files
            )

    async def _get_intelligent_rag_context(self, query: str) -> str:
        """
        Performs a RAG query to get relevant context from the knowledge base.
        """
        if not self.rag_manager:
            return "RAG system not available."
        try:
            results = await self.rag_manager.query(query, n_results=3)
            if not results or not results.get("documents"):
                return "No relevant information found in the knowledge base."

            context_str = "### Relevant Information from Knowledge Base\n"
            for doc in results["documents"][0]:
                context_str += f"- {doc}\n"
            return context_str
        except Exception as e:
            self.logger.error(f"RAG query failed: {e}", exc_info=True)
            return "Error retrieving information from the knowledge base."

    async def _stream_and_collect_json(self, prompt: str, role: LLMRole, agent_name: str) -> dict:
        """
        Streams from an LLM, collects the full response, and parses it as JSON.
        Now includes periodic progress updates to keep the UI responsive.
        """
        full_response_str = ""
        self.stream_emitter(agent_name, "info", f"Waiting for {role.value} model response...", 3)

        # --- Progress tracking ---
        last_update_time = asyncio.get_event_loop().time()

        try:
            async for chunk in self.llm_client.stream_chat(prompt, role=role):
                full_response_str += chunk
                current_time = asyncio.get_event_loop().time()

                # Emit a progress update every 0.3 seconds to prevent flooding the UI
                if (current_time - last_update_time) > 0.3:
                    self.stream_emitter(agent_name, "info", f"Receiving data... ({len(full_response_str)} chars)", 3)
                    last_update_time = current_time

            self.stream_emitter(agent_name, "success",
                                f"Response received ({len(full_response_str)} chars). Parsing...", 3)
            return self._parse_json_from_response(full_response_str)

        except Exception as e:
            self.logger.error(f"LLM streaming/parsing failed for {agent_name}: {e}", exc_info=True)
            self.stream_emitter(agent_name, "error", f"LLM streaming call failed for {agent_name}: {e}", 2)
            raise

    def _parse_json_from_response(self, response: str) -> Dict[str, Any]:
        """
        Robustly parses a JSON object from a string that might contain markdown fences or other text.
        """
        # Find the start and end of the JSON block
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Fallback for responses that are just the JSON object
            json_str = response.strip()
            if not (json_str.startswith('{') and json_str.endswith('}')):
                # If it's not a clear JSON object, try to find one within the text
                start = json_str.find('{')
                end = json_str.rfind('}') + 1
                if start != -1 and end != 0:
                    json_str = json_str[start:end]
                else:
                    raise json.JSONDecodeError("Could not find a JSON object in the response.", json_str, 0)

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode JSON: {e}\nResponse was:\n{json_str[:500]}...")
            raise