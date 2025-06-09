# core/llm_client.py - V4.7 FINAL - Complete file with user-defined defaults

import json
import os
from enum import Enum
from pathlib import Path
from typing import AsyncGenerator, Optional, Dict, Any
import aiohttp


class LLMRole(Enum):
    """The core AI specialist roles for the V4 workflow."""
    ARCHITECT = "architect"
    CODER = "coder"
    REVIEWER = "reviewer"
    CHAT = "chat"


class ModelConfig:
    """Configuration for a specific model"""

    def __init__(self, provider: str, model: str, api_key: str = None,
                 base_url: str = None, temperature: float = 0.7,
                 max_tokens: int = 4000, suitable_roles: list = None):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.suitable_roles = suitable_roles or [LLMRole.CHAT]


class EnhancedLLMClient:
    """
    Enhanced LLM client supporting multiple models for different AI roles.
    """

    def __init__(self):
        self.models: Dict[str, ModelConfig] = {}
        self.role_assignments: Dict[LLMRole, str] = {}
        self.personalities: Dict[LLMRole, str] = {}
        self._initialize_models()
        self._load_personalities_from_config()
        self._initialize_default_personalities()
        self._assign_roles()

    def _load_personalities_from_config(self):
        """Load personalities from personality_presets.json."""
        try:
            presets_file = Path("config") / "personality_presets.json"
            if presets_file.exists():
                with open(presets_file, 'r', encoding='utf-8') as f:
                    all_presets_data = json.load(f)
                for role_str, presets_list in all_presets_data.items():
                    try:
                        if role_str in ["planner", "structurer", "architect"]:
                            role_enum = LLMRole.ARCHITECT
                        elif role_str in ["assembler", "coder"]:
                            role_enum = LLMRole.CODER
                        else:
                            role_enum = LLMRole(role_str)

                        if presets_list:
                            user_preset = next((p for p in presets_list if p.get("author", "User") == "User"), None)
                            preset_to_load = user_preset or presets_list[0]
                            self.personalities[role_enum] = preset_to_load["personality"]
                    except (ValueError, KeyError):
                        continue
        except Exception as e:
            print(f"Error loading personalities from config: {e}")

    def _initialize_default_personalities(self):
        """Initialize default personalities for the new role structure."""
        default_personalities_map = {
            LLMRole.ARCHITECT: "You are the ARCHITECT AI, a master software architect. Your task is to create a complete, comprehensive, and machine-readable Technical Specification Sheet for an entire software project based on a user's request. This sheet will be the single source of truth for all other AI agents.",
            LLMRole.CODER: "You are an expert Python developer. Your task is to generate a single, complete, and production-ready Python file based on a strict Technical Specification and the full source code of its dependencies.",
            LLMRole.REVIEWER: "You are a senior code reviewer. Your primary goal is to ensure the generated code is of high quality, correct, and adheres to the technical specification. Provide a final 'approved' status and a brief summary.",
            LLMRole.CHAT: "You are AvA, a friendly and helpful AI development assistant."
        }
        for role_enum, personality_text in default_personalities_map.items():
            if role_enum not in self.personalities:
                self.personalities[role_enum] = personality_text

    def _initialize_models(self):
        """Initialize available models with their configurations"""
        if os.getenv("GEMINI_API_KEY"):
            self.models["gemini-2.5-flash-preview-05-20"] = ModelConfig(
                provider="gemini", model="gemini-2.5-flash-preview-05-20", api_key=os.getenv("GEMINI_API_KEY"),
                temperature=0.7, max_tokens=8000,
                suitable_roles=[LLMRole.CHAT]
            )
            self.models["gemini-2.5-pro-preview-06-05"] = ModelConfig(
                provider="gemini", model="gemini-2.5-pro-preview-06-05", api_key=os.getenv("GEMINI_API_KEY"),
                temperature=0.3, max_tokens=8000,
                suitable_roles=[LLMRole.ARCHITECT, LLMRole.REVIEWER]
            )
            print("✅ Gemini models loaded successfully (including Flash)")

        if os.getenv("DEEPSEEK_API_KEY"):
            deepseek_base_url = "https://api.deepseek.com"
            deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
            self.models["deepseek-reasoner"] = ModelConfig(
                provider="deepseek", model="deepseek-reasoner", api_key=deepseek_api_key,
                base_url=deepseek_base_url, temperature=0.1, max_tokens=32000,
                suitable_roles=[LLMRole.CODER]
            )
            print("✅ DeepSeek models loaded successfully")

    def _assign_roles(self):
        """Assign models to roles based on user's desired defaults."""
        self.role_assignments = {
            LLMRole.ARCHITECT: "gemini-2.5-pro-preview-06-05",
            LLMRole.CODER: "deepseek-reasoner",
            LLMRole.REVIEWER: "gemini-2.5-pro-preview-06-05",
            LLMRole.CHAT: "gemini-2.5-flash-preview-05-20"
        }

        print("🎯 Final Role Assignments (User Defaults):")
        for role, model_name in self.role_assignments.items():
            if model_name in self.models:
                provider = self.models[model_name].provider
                print(f"  {role.value.title()}: {provider}/{model_name}")
            else:
                print(f"  {role.value.title()}: ❌ WARNING: Model '{model_name}' not found!")

    def assign_role(self, role: LLMRole, model_name: str):
        """Manually assign a model to a specific role."""
        if model_name in self.models:
            self.role_assignments[role] = model_name
            print(f"✅ Assigned {model_name} to {role.value} role")
        else:
            print(f"❌ Model {model_name} not found in available models.")

    def get_role_model(self, role: LLMRole) -> Optional[ModelConfig]:
        """Get the model configuration assigned to a role."""
        model_name = self.role_assignments.get(role)
        if not model_name or model_name not in self.models:
            print(f"Warning: Model for role {role.value} ('{model_name}') not available. Check API keys.")
            return None
        return self.models.get(model_name)

    def get_role_assignments(self) -> Dict[str, str]:
        """Get current role assignments for display."""
        return {role_enum.value: model_name for role_enum, model_name in self.role_assignments.items()}

    async def chat(self, prompt: str, role: LLMRole = LLMRole.CHAT) -> str:
        model_config = self.get_role_model(role)
        personality = self.personalities.get(role, "")

        if not model_config:
            return self._fallback_response(prompt, role)
        try:
            provider_map = {
                "gemini": self._call_gemini,
                "anthropic": self._call_anthropic,
                "deepseek": self._call_deepseek
            }
            if model_config.provider in provider_map:
                return await provider_map[model_config.provider](prompt, model_config, personality)
            return self._fallback_response(prompt, role)
        except Exception as e:
            print(f"API call with {model_config.provider} for role {role.value} failed: {e}")
            return self._fallback_response(prompt, role, e)

    async def stream_chat(self, prompt: str, role: LLMRole = LLMRole.CHAT) -> AsyncGenerator[str, None]:
        model_config = self.get_role_model(role)
        personality = self.personalities.get(role, "")
        if not model_config:
            yield self._fallback_response(prompt, role)
            return
        stream_generator = None
        try:
            provider_map = {
                "gemini": self._stream_gemini,
                "anthropic": self._stream_anthropic,
                "deepseek": self._stream_deepseek
            }
            if model_config.provider in provider_map:
                stream_generator = provider_map[model_config.provider](prompt, model_config, personality)
                async for chunk in stream_generator:
                    yield chunk
            else:
                yield self._fallback_response(prompt, role)
        except Exception as e:
            print(f"Streaming API call for role {role.value} failed: {e}")
            yield self._fallback_response(prompt, role, e)
        finally:
            if stream_generator and hasattr(stream_generator, 'aclose'):
                await stream_generator.aclose()

    async def _call_gemini(self, prompt: str, config: ModelConfig, personality: str = "") -> str:
        final_prompt = f"{personality}\n\nUSER PROMPT:\n{prompt}" if personality else prompt
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.model}:generateContent?key={config.api_key}"
        payload = {"contents": [{"parts": [{"text": final_prompt}]}],
                   "generationConfig": {"temperature": config.temperature, "maxOutputTokens": config.max_tokens}}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers={"Content-Type": "application/json"}, json=payload) as response:
                response_json = await response.json()
                if response.status == 200 and "candidates" in response_json and response_json["candidates"]:
                    return response_json["candidates"][0]["content"]["parts"][0]["text"]
                raise Exception(f"Gemini API error {response.status}: {response_json}")

    async def _stream_gemini(self, prompt: str, config: ModelConfig, personality: str = "") -> AsyncGenerator[
        str, None]:
        final_prompt = f"{personality}\n\nUSER PROMPT:\n{prompt}" if personality else prompt
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.model}:streamGenerateContent?key={config.api_key}&alt=sse"
        payload = {"contents": [{"parts": [{"text": final_prompt}]}],
                   "generationConfig": {"temperature": config.temperature, "maxOutputTokens": config.max_tokens}}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    raise Exception(f"Gemini stream API error: {await response.text()}")
                async for line in response.content:
                    if line.startswith(b'data: '):
                        try:
                            data = json.loads(line[6:])
                            if data.get("candidates"):
                                yield data["candidates"][0]["content"]["parts"][0]["text"]
                        except json.JSONDecodeError:
                            print(f"Warning: Gemini stream JSON decode error: {line.decode('utf-8', 'ignore')}")

    async def _call_anthropic(self, prompt: str, config: ModelConfig, personality: str = "") -> str:
        url = config.base_url or "https://api.anthropic.com/v1/messages"
        headers = {"x-api-key": config.api_key, "Content-Type": "application/json", "anthropic-version": "2023-06-01"}
        payload = {"model": config.model, "max_tokens": config.max_tokens, "temperature": config.temperature,
                   "messages": [{"role": "user", "content": prompt}]}
        if personality: payload["system"] = personality
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                resp_json = await response.json()
                if response.status == 200: return resp_json["content"][0]["text"]
                raise Exception(f"Anthropic API error {response.status}: {resp_json}")

    async def _stream_anthropic(self, prompt: str, config: ModelConfig, personality: str = "") -> AsyncGenerator[
        str, None]:
        url = config.base_url or "https://api.anthropic.com/v1/messages"
        headers = {"x-api-key": config.api_key, "Content-Type": "application/json", "anthropic-version": "2023-06-01",
                   "Accept": "text/event-stream"}
        payload = {"model": config.model, "max_tokens": config.max_tokens, "temperature": config.temperature,
                   "messages": [{"role": "user", "content": prompt}], "stream": True}
        if personality: payload["system"] = personality
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    raise Exception(f"Anthropic stream API error: {await response.text()}")
                async for line in response.content:
                    if line.startswith(b"data: "):
                        try:
                            data = json.loads(line[6:])
                            if data.get("type") == "content_block_delta":
                                yield data["delta"]["text"]
                        except json.JSONDecodeError:
                            continue

    async def _call_deepseek(self, prompt: str, config: ModelConfig, personality: str = "") -> str:
        messages = [{"role": "system", "content": personality}] if personality else []
        messages.append({"role": "user", "content": prompt})
        url = f"{config.base_url}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}
        payload = {"model": config.model, "messages": messages, "temperature": config.temperature,
                   "max_tokens": config.max_tokens}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                resp_json = await response.json()
                if response.status == 200 and "choices" in resp_json:
                    message = resp_json["choices"][0]["message"]
                    return message.get('content', '')
                raise Exception(f"DeepSeek API error {response.status}: {resp_json}")

    async def _stream_deepseek(self, prompt: str, config: ModelConfig, personality: str = "") -> AsyncGenerator[
        str, None]:
        messages = [{"role": "system", "content": personality}] if personality else []
        messages.append({"role": "user", "content": prompt})
        url = f"{config.base_url}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}
        payload = {"model": config.model, "messages": messages, "temperature": config.temperature,
                   "max_tokens": config.max_tokens, "stream": True}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    raise Exception(f"DeepSeek stream API error: {await response.text()}")
                async for line in response.content:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith("data: "):
                        data_content = line_str[6:]
                        if data_content == "[DONE]": break
                        try:
                            chunk = json.loads(data_content)
                            delta = chunk["choices"][0].get("delta", {})
                            if delta.get("content"):
                                yield delta["content"]
                        except json.JSONDecodeError:
                            print(f"Warning: DeepSeek stream JSON decode error: {line_str}")

    def _fallback_response(self, prompt: str, role: LLMRole, error: Exception = None) -> str:
        error_msg = f"API call failed: {str(error)}" if error else "No model available."
        return f"# AvA Error: No {role.value} LLM available or {error_msg}\n# Request: {prompt[:100]}...\n# Check config and API keys."

    def get_available_models(self) -> list:
        available = []
        if not self.role_assignments:
            return ["No models assigned to roles yet."]

        for role_enum, model_name_key in self.role_assignments.items():
            if model_name_key and model_name_key in self.models:
                config = self.models[model_name_key]
                available.append(f"{role_enum.value.title()}: {config.provider}/{config.model}")
            else:
                available.append(f"{role_enum.value.title()}: Unassigned or '{model_name_key}' not found")

        if not available and self.models:
            return [f"Models loaded ({len(self.models)}) but not assigned."]
        elif not available:
            return ["No LLM services available or configured."]
        return available