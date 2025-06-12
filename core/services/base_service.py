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

    def _get_team_context_string(self, for_file: str = None) -> str:
        """
        Get a dynamically filtered and formatted string of team context,
        tailored to the agent's role and the specific file being worked on.
        """
        if not self.project_state:
            return "No team context available."

        try:
            # Use the new intelligent context method
            agent_name = self.__class__.__name__
            context_data = self.project_state.get_enhanced_project_context(
                for_file=for_file,
                ai_role=agent_name
            )

            insights = context_data.get("team_insights", [])

            if not insights:
                return "No relevant team insights for this task."

            context_parts = ["**Relevant Team Insights & Learnings**"]
            for insight in insights:
                # Format each insight for maximum clarity in the prompt
                related = f"(Related files: {', '.join(insight['related_files'])})" if insight.get(
                    'related_files') else ""
                context_parts.append(
                    f"- **{insight['insight_type'].upper()}** "
                    f"from **{insight['source_agent']}** "
                    f"(Impact: {insight['impact_level']}): "
                    f"{insight['content']} {related}"
                )

            # Also include established patterns from the project context
            patterns = context_data.get("established_patterns", {})
            if patterns:
                context_parts.append("\n**Established Project Patterns**")
                for key, value in patterns.items():
                    if value:
                        # Assuming value is a dict of patterns, format it
                        if isinstance(value, dict):
                            for pat_id, pat_details in value.items():
                                if isinstance(pat_details, dict):
                                    context_parts.append(f"- {pat_details.get('description', pat_id)}")
                        else:
                            context_parts.append(f"- {key.replace('_', ' ').title()}: {value}")

            return "\n".join(context_parts)
        except Exception as e:
            self.logger.warning(f"Failed to get and format team context: {e}")
            # Log the full exception for debugging
            self.logger.exception("Exception details for team context failure:")
            return "Team context is currently unavailable due to an internal error."

    def _parse_json_from_response(self, response_text: str, agent_name: str) -> dict:
        """BULLETPROOF JSON parser that handles all edge cases."""
        response_text = response_text.strip()

        if not response_text:
            self.stream_emitter(agent_name, "error", "Empty response from LLM", 3)
            return {}

        # Method 1: Try direct JSON parsing
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            self.stream_emitter(agent_name, "warning", f"Direct JSON parse failed: {e}", 4)

        # Method 2: Extract from markdown code blocks
        json_patterns = [
            r'```(?:json)?\s*(\{.*?\})\s*```',  # ```json or ``` blocks
            r'```(?:json)?\s*(\[.*?\])\s*```',  # Arrays in blocks
        ]

        for pattern in json_patterns:
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue

        # Method 3: Find JSON object boundaries and try progressive parsing
        brace_start = response_text.find('{')
        if brace_start != -1:
            # Find the matching closing brace by counting
            brace_count = 0
            brace_end = -1

            for i in range(brace_start, len(response_text)):
                char = response_text[i]
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        brace_end = i
                        break

            if brace_end != -1:
                json_candidate = response_text[brace_start:brace_end + 1]
                try:
                    return json.loads(json_candidate)
                except json.JSONDecodeError:
                    # Method 4: Try to fix common JSON issues
                    fixed_json = self._attempt_json_repair(json_candidate, agent_name)
                    if fixed_json:
                        return fixed_json

        # Method 5: Look for key-value patterns and reconstruct JSON
        extracted_json = self._extract_json_from_text(response_text, agent_name)
        if extracted_json:
            return extracted_json

        # All methods failed
        self.stream_emitter(agent_name, "error",
                            f"All JSON parsing methods failed. Response preview: {response_text[:300]}...", 3)
        return {}

    def _attempt_json_repair(self, json_text: str, agent_name: str) -> dict:
        """Attempt to repair malformed JSON."""
        try:
            # Common fixes for JSON issues
            fixes = [
                # Fix unescaped newlines in strings
                lambda s: re.sub(r'(".*?)(\n)(.*?")', r'\1\\n\3', s, flags=re.DOTALL),
                # Fix unescaped quotes
                lambda s: re.sub(r'(".*?)"(.*?")', r'\1\"\2', s),
                # Fix trailing commas
                lambda s: re.sub(r',(\s*[}\]])', r'\1', s),
                # Fix missing commas between objects
                lambda s: re.sub(r'}\s*{', r'},{', s),
            ]

            current = json_text
            for fix in fixes:
                try:
                    current = fix(current)
                    parsed = json.loads(current)
                    self.stream_emitter(agent_name, "success", "JSON repair successful", 4)
                    return parsed
                except:
                    continue

        except Exception as e:
            self.stream_emitter(agent_name, "warning", f"JSON repair failed: {e}", 4)

        return {}

    def _extract_json_from_text(self, text: str, agent_name: str) -> dict:
        """Extract JSON structure from free text using pattern matching."""
        try:
            # Look for key patterns that indicate the required JSON structure
            required_keys = ["IMPLEMENTED_CODE", "IMPLEMENTATION_NOTES", "INTEGRATION_HINTS",
                             "EDGE_CASES_HANDLED", "TESTING_CONSIDERATIONS"]

            extracted = {}

            for key in required_keys:
                # Pattern to find "KEY": "value" or "KEY": [...] structures
                patterns = [
                    rf'"{key}":\s*"([^"]*(?:\\.[^"]*)*)"',  # String values
                    rf'"{key}":\s*\[(.*?)\]',  # Array values
                    rf'"{key}":\s*([^,}}\n]*)',  # Other values
                ]

                for pattern in patterns:
                    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                    if match:
                        value = match.group(1).strip()
                        # Clean up the value
                        if pattern.endswith('")'):  # String pattern
                            extracted[key] = value.replace('\\"', '"').replace('\\n', '\n')
                        elif pattern.endswith(')'):  # Array pattern
                            try:
                                # Try to parse as JSON array
                                extracted[key] = json.loads(f'[{value}]')
                            except:
                                # Fall back to splitting by comma
                                extracted[key] = [item.strip().strip('"') for item in value.split(',')]
                        else:
                            extracted[key] = value.strip('"')
                        break

                # If we couldn't extract this key, provide a default
                if key not in extracted:
                    extracted[key] = f"Could not extract {key} from response"

            if len(extracted) >= 3:  # If we got at least 3 keys, consider it successful
                self.stream_emitter(agent_name, "success", f"Extracted JSON structure with {len(extracted)} keys", 4)
                return extracted

        except Exception as e:
            self.stream_emitter(agent_name, "warning", f"JSON extraction failed: {e}", 4)

        return {}

    async def _stream_and_collect_json(self, prompt: str, role: LLMRole, agent_name: str) -> dict:
        """BULLETPROOF streaming and JSON collection with enhanced error handling."""
        all_chunks = []
        chunk_count = 0

        try:
            self.stream_emitter(agent_name, "info", f"Starting streaming for {agent_name}", 4)

            async for chunk in self.llm_client.stream_chat(prompt, role=role):
                if chunk:
                    all_chunks.append(chunk)
                    chunk_count += 1
                    # Only emit every 10th chunk to reduce noise
                    if chunk_count % 10 == 0:
                        self.stream_emitter(agent_name, "stream", f"[chunk {chunk_count}] {chunk[:50]}...", 4)

            response_text = "".join(all_chunks).strip()

            if not response_text:
                self.stream_emitter(agent_name, "error", "Empty response from LLM", 2)
                return {}

            self.stream_emitter(agent_name, "info",
                                f"Collected {len(response_text)} characters from {chunk_count} chunks", 3)

            # Use bulletproof JSON parsing
            parsed_json = self._parse_json_from_response(response_text, agent_name)

            if not parsed_json:
                # Last resort: create a minimal valid response
                self.stream_emitter(agent_name, "warning", "Creating fallback JSON response", 3)
                return {
                    "IMPLEMENTED_CODE": f"# Fallback response due to parsing failure\n# Original response length: {len(response_text)}\npass",
                    "IMPLEMENTATION_NOTES": "Generated fallback due to JSON parsing failure",
                    "INTEGRATION_HINTS": "Check logs for original response details",
                    "EDGE_CASES_HANDLED": ["JSON parsing failure handled"],
                    "TESTING_CONSIDERATIONS": "Verify implementation manually"
                }

            return parsed_json

        except Exception as e:
            self.stream_emitter(agent_name, "error", f"Critical error during streaming/collection: {e}", 2)

            # Try to salvage partial response
            if all_chunks:
                partial_response = "".join(all_chunks)
                self.stream_emitter(agent_name, "info",
                                    f"Attempting to parse partial response ({len(partial_response)} chars)", 3)

                salvaged = self._parse_json_from_response(partial_response, agent_name)
                if salvaged:
                    return salvaged

            # Complete fallback
            return {
                "IMPLEMENTED_CODE": f"# Critical error during generation: {str(e)}\npass",
                "IMPLEMENTATION_NOTES": f"Failed due to streaming error: {str(e)}",
                "INTEGRATION_HINTS": "Manual implementation required",
                "EDGE_CASES_HANDLED": ["Generation failure"],
                "TESTING_CONSIDERATIONS": "Implementation needs to be written manually"
            }

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