# core/llm_client.py - V4 with consolidated Architect Role

import os
from pathlib import Path
import requests
import json
import asyncio
from typing import AsyncGenerator, Optional, Dict, Any
from enum import Enum


class LLMRole(Enum):
    """The core AI specialist roles for the V4 workflow."""
    ARCHITECT = "architect"  # Merged role for Structurer and Planner
    CODER = "coder"  # Generates code for a single file based on a spec
    REVIEWER = "reviewer"  # Performs a final quality check
    CHAT = "chat"  # Handles general user conversation


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
                        # Map old roles to new ones for compatibility
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
            self.models["gemini-2.5-pro-preview-06-05"] = ModelConfig(
                provider="gemini", model="gemini-2.5-pro-preview-06-05", api_key=os.getenv("GEMINI_API_KEY"),
                temperature=0.3, max_tokens=8000,
                suitable_roles=[LLMRole.ARCHITECT, LLMRole.REVIEWER, LLMRole.CHAT]
            )
            self.models["gemini-2.5-flash-preview-05-20"] = ModelConfig(
                provider="gemini", model="gemini-2.5-flash-preview-05-20", api_key=os.getenv("GEMINI_API_KEY"),
                temperature=0.2, max_tokens=8000,
                suitable_roles=[LLMRole.CODER, LLMRole.CHAT]
            )
        if os.getenv("ANTHROPIC_API_KEY"):
            self.models["claude-3-5-sonnet-20240620"] = ModelConfig(
                provider="anthropic", model="claude-3-5-sonnet-20240620", api_key=os.getenv("ANTHROPIC_API_KEY"),
                temperature=0.3, max_tokens=4000,
                suitable_roles=[LLMRole.ARCHITECT, LLMRole.REVIEWER, LLMRole.CHAT, LLMRole.CODER]
            )

        if os.getenv("OPENAI_API_KEY"):
            self.models["gpt-4o"] = ModelConfig(
                provider="openai", model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.3, max_tokens=4000,
                suitable_roles=[LLMRole.ARCHITECT, LLMRole.REVIEWER, LLMRole.CHAT, LLMRole.CODER]
            )
            self.models["gpt-4o-mini"] = ModelConfig(
                provider="openai", model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.2, max_tokens=16000,
                suitable_roles=[LLMRole.CODER, LLMRole.CHAT]
            )

        try:
            ollama_base_url = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
            response = requests.get(f"{ollama_base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                ollama_models_available = response.json().get("models", [])
                for model_info in ollama_models_available:
                    model_name = model_info["name"]
                    ollama_key = f"ollama-{model_name}"
                    current_suitable_roles = [LLMRole.CHAT]
                    temp = 0.3
                    if any(kw in model_name.lower() for kw in ["coder", "starcoder", "codellama"]):
                        current_suitable_roles.append(LLMRole.CODER)
                        temp = 0.1
                    if any(kw in model_name.lower() for kw in ["qwen", "llama", "mistral", "mixtral"]):
                        current_suitable_roles.extend([LLMRole.ARCHITECT, LLMRole.REVIEWER])
                    self.models[ollama_key] = ModelConfig(
                        provider="ollama", model=model_name, base_url=ollama_base_url,
                        temperature=temp, max_tokens=4000,
                        suitable_roles=list(set(current_suitable_roles))
                    )
        except requests.exceptions.RequestException:
            pass
        except Exception as e:
            print(f"Error during Ollama model discovery: {e}")

    def _assign_roles(self):
        """Assigns default models to the new, consolidated roles."""
        print("Assigning default models for consolidated roles...")

        default_assignments = {
            LLMRole.ARCHITECT: "gemini-2.5-pro-preview-06-05",
            LLMRole.CODER: "gemini-2.5-pro-preview-06-05",
            LLMRole.REVIEWER: "gemini-2.5-pro-preview-06-05",
            LLMRole.CHAT: "gemini-2.5-flash-preview-05-20"
        }

        self.role_assignments = {}
        available_model_keys = list(self.models.keys())

        for role, primary_model_key in default_assignments.items():
            if primary_model_key in self.models:
                self.role_assignments[role] = primary_model_key
            else:
                fallback_found = next((key for key in available_model_keys if role in self.models[key].suitable_roles),
                                      None)
                if fallback_found:
                    self.role_assignments[role] = fallback_found
                elif available_model_keys:
                    self.role_assignments[role] = available_model_keys[0]
                else:
                    self.role_assignments[role] = None

        self._apply_temperatures_from_presets()

    def _apply_temperatures_from_presets(self):
        try:
            presets_file = Path("config") / "personality_presets.json"
            if not presets_file.exists(): return

            with open(presets_file, 'r', encoding='utf-8') as f:
                all_presets_data = json.load(f)

            for role_str, presets_list in all_presets_data.items():
                try:
                    role_enum = LLMRole(role_str)
                    user_preset = next((p for p in presets_list if p.get("author", "User") == "User"), None)
                    preset_to_load = user_preset or (presets_list[0] if presets_list else None)

                    if preset_to_load and "temperature" in preset_to_load:
                        assigned_model_key = self.role_assignments.get(role_enum)
                        if assigned_model_key and assigned_model_key in self.models:
                            self.models[assigned_model_key].temperature = float(preset_to_load["temperature"])
                except (ValueError, KeyError):
                    continue
        except Exception as e:
            print(f"Note: Could not apply initial temperatures from presets: {e}")

    def get_role_model(self, role: LLMRole) -> Optional[ModelConfig]:
        """Get the assigned model for a specific role."""
        model_name = self.role_assignments.get(role)
        if model_name:
            return self.models.get(model_name)
        print(f"Warning: No model assigned for role {role.value}")
        # Fallback to chat model if the requested role is unassigned but chat is
        if role != LLMRole.CHAT:
            return self.get_role_model(LLMRole.CHAT)
        return None

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
                "gemini": self._call_gemini, "openai": self._call_openai,
                "anthropic": self._call_anthropic, "ollama": self._call_ollama,
                "deepseek": self._call_deepseek
            }
            if model_config.provider in provider_map:
                return await provider_map[model_config.provider](prompt, model_config, personality)
            else:
                return self._fallback_response(prompt, role)
        except Exception as e:
            print(f"API call with {model_config.provider} for role {role.value} failed: {e}")
            return self._fallback_response(prompt, role)

    async def stream_chat(self, prompt: str, role: LLMRole = LLMRole.CHAT) -> AsyncGenerator[str, None]:
        model_config = self.get_role_model(role)
        personality = self.personalities.get(role, "")

        if not model_config:
            yield self._fallback_response(prompt, role)
            return

        stream_generator = None
        try:
            provider_map = {
                "gemini": self._stream_gemini, "openai": self._stream_openai,
                "anthropic": self._stream_anthropic, "ollama": self._stream_ollama,
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
            yield self._fallback_response(prompt, role)
        finally:
            if stream_generator and hasattr(stream_generator, 'aclose'):
                await stream_generator.aclose()

    # --- Private API Call Methods ---
    # These methods remain unchanged, just collapse them for readability if you like.
    # _call_gemini, _stream_gemini, _call_openai, _stream_openai, etc.
    async def _call_gemini(self, prompt: str, config: ModelConfig, personality: str = "") -> str:
        import aiohttp
        final_prompt = f"{personality}\n\nUSER PROMPT:\n{prompt}" if personality else prompt
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.model}:generateContent?key={config.api_key}"
        payload = {"contents": [{"parts": [{"text": final_prompt}]}],
                   "generationConfig": {"temperature": config.temperature, "maxOutputTokens": config.max_tokens}}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers={"Content-Type": "application/json"}, json=payload) as response:
                response_json = await response.json()
                if response.status == 200 and "candidates" in response_json and response_json[
                    "candidates"] and "content" in response_json["candidates"][0] and "parts" in \
                        response_json["candidates"][0]["content"] and response_json["candidates"][0]["content"][
                    "parts"]:
                    return response_json["candidates"][0]["content"]["parts"][0]["text"]
                raise Exception(f"Gemini API error {response.status}: {response_json}")

    async def _stream_gemini(self, prompt: str, config: ModelConfig, personality: str = "") -> AsyncGenerator[
        str, None]:
        import aiohttp
        final_prompt = f"{personality}\n\nUSER PROMPT:\n{prompt}" if personality else prompt
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.model}:streamGenerateContent?key={config.api_key}&alt=sse"
        payload = {"contents": [{"parts": [{"text": final_prompt}]}],
                   "generationConfig": {"temperature": config.temperature, "maxOutputTokens": config.max_tokens}}

        session = None
        response = None
        try:
            session = aiohttp.ClientSession()
            response = await session.post(url, headers={"Content-Type": "application/json"}, json=payload)

            if response.status != 200:
                raise Exception(f"Gemini API stream error {response.status}: {await response.text()}")

            async for line in response.content:
                line_str = line.decode('utf-8').strip()
                if line_str.startswith("data: "):
                    try:
                        data = json.loads(line_str[6:])
                        if data.get("candidates") and data["candidates"][0].get("content", {}).get("parts"):
                            yield data["candidates"][0]["content"]["parts"][0]["text"]
                    except json.JSONDecodeError:
                        print(f"Warning: Gemini stream JSON decode error: {line_str}")
        except GeneratorExit:
            raise
        finally:
            if response:
                response.close()
            if session:
                await session.close()

    async def _call_openai(self, prompt: str, config: ModelConfig, personality: str = "") -> str:
        import aiohttp
        messages = [{"role": "system", "content": personality}] if personality else []
        messages.append({"role": "user", "content": prompt})
        url = config.base_url or "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}
        payload = {"model": config.model, "messages": messages, "temperature": config.temperature,
                   "max_tokens": config.max_tokens}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                resp_json = await response.json()
                if response.status == 200: return resp_json["choices"][0]["message"]["content"]
                raise Exception(f"OpenAI API error {response.status}: {resp_json}")

    async def _stream_openai(self, prompt: str, config: ModelConfig, personality: str = "") -> AsyncGenerator[
        str, None]:
        import aiohttp
        messages = [{"role": "system", "content": personality}] if personality else []
        messages.append({"role": "user", "content": prompt})
        url = config.base_url or "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}
        payload = {"model": config.model, "messages": messages, "temperature": config.temperature,
                   "max_tokens": config.max_tokens, "stream": True}

        session = None
        response = None
        try:
            session = aiohttp.ClientSession()
            response = await session.post(url, headers=headers, json=payload)

            if response.status != 200:
                raise Exception(f"OpenAI API stream error {response.status}: {await response.text()}")

            async for line in response.content:
                line_str = line.decode('utf-8').strip()
                if line_str.startswith("data: "):
                    data_content = line_str[6:]
                    if data_content == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_content)
                        if chunk["choices"][0].get("delta", {}).get("content"):
                            yield chunk["choices"][0]["delta"]["content"]
                    except json.JSONDecodeError:
                        print(f"Warning: OpenAI stream JSON decode error: {line_str}")
        except GeneratorExit:
            raise
        finally:
            if response:
                response.close()
            if session:
                await session.close()

    async def _call_anthropic(self, prompt: str, config: ModelConfig, personality: str = "") -> str:
        import aiohttp
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
        import aiohttp
        url = config.base_url or "https://api.anthropic.com/v1/messages"
        headers = {"x-api-key": config.api_key, "Content-Type": "application/json", "anthropic-version": "2023-06-01",
                   "Accept": "text/event-stream"}
        payload = {"model": config.model, "max_tokens": config.max_tokens, "temperature": config.temperature,
                   "messages": [{"role": "user", "content": prompt}], "stream": True}
        if personality: payload["system"] = personality

        session = None
        response = None
        try:
            session = aiohttp.ClientSession()
            response = await session.post(url, headers=headers, json=payload)

            if response.status != 200:
                raise Exception(f"Anthropic API stream error {response.status}: {await response.text()}")

            async for line in response.content:
                line_str = line.decode('utf-8').strip()
                if line_str.startswith("data: "):
                    try:
                        data = json.loads(line_str[6:])
                        if data.get("type") == "content_block_delta" and data["delta"]["type"] == "text_delta":
                            yield data["delta"]["text"]
                        elif data.get("type") == "message_stop":
                            break
                    except json.JSONDecodeError:
                        print(f"Warning: Anthropic stream JSON decode error: {line_str}")
        except GeneratorExit:
            raise
        finally:
            if response:
                response.close()
            if session:
                await session.close()

    async def _call_ollama(self, prompt: str, config: ModelConfig, personality: str = "") -> str:
        import aiohttp
        final_prompt = f"{personality}\n\nUSER PROMPT:\n{prompt}" if personality else prompt
        url = f"{config.base_url}/api/generate"
        payload = {"model": config.model, "prompt": final_prompt, "stream": False,
                   "options": {"temperature": config.temperature, "num_predict": config.max_tokens}}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                resp_json = await response.json()
                if response.status == 200: return resp_json["response"]
                raise Exception(f"Ollama API error {response.status}: {resp_json}")

    async def _stream_ollama(self, prompt: str, config: ModelConfig, personality: str = "") -> AsyncGenerator[
        str, None]:
        import aiohttp
        final_prompt = f"{personality}\n\nUSER PROMPT:\n{prompt}" if personality else prompt
        url = f"{config.base_url}/api/generate"
        payload = {"model": config.model, "prompt": final_prompt, "stream": True,
                   "options": {"temperature": config.temperature, "num_predict": config.max_tokens}}

        session = None
        response = None
        try:
            session = aiohttp.ClientSession()
            response = await session.post(url, json=payload)

            if response.status != 200:
                raise Exception(f"Ollama API stream error {response.status}: {await response.text()}")

            async for line in response.content:
                try:
                    if line:
                        data = json.loads(line.decode('utf-8'))
                        yield data.get("response", "")
                        if data.get("done"):
                            break
                except json.JSONDecodeError:
                    print(f"Warning: Ollama stream JSON decode error: {line.decode('utf-8', errors='ignore')}")
        except GeneratorExit:
            raise
        finally:
            if response:
                response.close()
            if session:
                await session.close()

    async def _call_deepseek(self, prompt: str, config: ModelConfig, personality: str = "") -> str:
        return await self._call_openai(prompt, config, personality)

    async def _stream_deepseek(self, prompt: str, config: ModelConfig, personality: str = "") -> AsyncGenerator[
        str, None]:
        async for chunk in self._stream_openai(prompt, config, personality):
            yield chunk

    def _fallback_response(self, prompt: str, role: LLMRole) -> str:
        return f"# AvA Error: No {role.value} LLM available or API call failed.\n# Request: {prompt[:100]}...\n# Check config and API keys."

    def get_available_models(self) -> list:
        """Get list of available models and their roles."""
        available = []
        if not self.role_assignments: return ["No models assigned to roles yet."]
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