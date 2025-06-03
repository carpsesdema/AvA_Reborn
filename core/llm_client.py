# core/llm_client.py - Enhanced Multi-Model LLM Client

import os
from pathlib import Path

import requests
import json
import asyncio
from typing import AsyncGenerator, Optional, Dict, Any
from enum import Enum


class LLMRole(Enum):
    """Different AI roles that require different models"""
    PLANNER = "planner"  # High-level planning, requires reasoning
    CODER = "coder"  # Code generation, can use specialized models
    ASSEMBLER = "assembler"  # Code assembly, similar to coder
    REVIEWER = "reviewer"  # Code review, requires reasoning
    CHAT = "chat"  # General chat, flexible model


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
    Enhanced LLM client supporting multiple models for different AI roles
    Optimizes cost by using appropriate models for each task
    Supports individual personalities for each AI role.
    """

    def __init__(self):
        self.models: Dict[str, ModelConfig] = {}
        self.role_assignments: Dict[LLMRole, str] = {}  # Keys are LLMRole Enum members
        self.personalities: Dict[LLMRole, str] = {}  # Keys are LLMRole Enum members
        self._initialize_models()
        self._load_personalities_from_config()  # Load from JSON first
        self._initialize_default_personalities()  # Then fill in missing with hardcoded
        self._assign_roles()

    def _load_personalities_from_config(self):
        """Load personalities from personality_presets.json for initial setup."""
        try:
            presets_file = Path("config") / "personality_presets.json"
            if presets_file.exists():
                with open(presets_file, 'r', encoding='utf-8') as f:
                    all_presets_data = json.load(f)

                for role_str, presets_list in all_presets_data.items():
                    try:
                        role_enum = LLMRole(role_str)  # Convert string role to enum
                        if presets_list:  # Take the first preset for this role as the default
                            # Prioritize user-created, then built-in if multiple exist
                            user_preset = next((p for p in presets_list if p.get("author", "User") == "User"), None)
                            if user_preset:
                                self.personalities[role_enum] = user_preset["personality"]
                                # Also set temperature for the model that will be assigned this role
                                # This is tricky as model isn't assigned yet. Store it temporarily or apply after role assignment.
                                print(
                                    f"Loaded personality for {role_enum.value} from user preset: {user_preset['name']}")
                                # We'll handle temperature application after model assignment
                            elif presets_list[0].get("author") == "AvA Built-in":
                                self.personalities[role_enum] = presets_list[0]["personality"]
                                print(
                                    f"Loaded personality for {role_enum.value} from built-in preset: {presets_list[0]['name']}")
                    except ValueError:
                        print(f"Warning: Unknown role '{role_str}' in personality_presets.json")
        except Exception as e:
            print(f"Error loading personalities from config: {e}")

    def _initialize_default_personalities(self):
        """Initialize default personalities if not loaded from config."""
        default_personalities_map = {
            LLMRole.PLANNER: "You are a senior software architect with 15+ years of experience. You think strategically, break down complex problems into clear steps, and always consider scalability and maintainability. You communicate clearly and provide detailed technical specifications.",
            LLMRole.CODER: "You are a coding specialist who writes clean, efficient, and well-documented code. You follow best practices, use proper error handling, and write code that is both functional and elegant. You focus on getting things done with high quality.",
            LLMRole.ASSEMBLER: "You are a meticulous code integrator who ensures all pieces work together seamlessly. You have an eye for detail, maintain consistent code style, and create professional, production-ready files with proper organization and documentation.",
            LLMRole.REVIEWER: "You are a detail-oriented code reviewer. Your primary goal is to ensure code quality, adherence to best practices, security, and performance. Provide constructive feedback and clear justifications for any issues found.",
            LLMRole.CHAT: "You are AvA, a friendly and helpful AI development assistant. Engage in natural conversation and guide users through their development tasks."
        }
        for role_enum, personality_text in default_personalities_map.items():
            if role_enum not in self.personalities:  # Only set if not already loaded from JSON
                self.personalities[role_enum] = personality_text
                print(f"Initialized hardcoded default personality for {role_enum.value}")

    def _initialize_models(self):
        """Initialize available models with their configurations"""
        # User-specified Gemini models
        if os.getenv("GEMINI_API_KEY"):
            self.models["gemini-2.5-pro-preview-05-06"] = ModelConfig(
                provider="gemini",
                model="gemini-2.5-pro-preview-05-06",
                api_key=os.getenv("GEMINI_API_KEY"),
                temperature=0.3,  # Default, can be overridden by presets/dialog
                max_tokens=8000,
                suitable_roles=[LLMRole.PLANNER, LLMRole.REVIEWER, LLMRole.CHAT]
            )
            self.models["gemini-2.5-flash-preview-05-20"] = ModelConfig(
                provider="gemini",
                model="gemini-2.5-flash-preview-05-20",
                api_key=os.getenv("GEMINI_API_KEY"),
                temperature=0.2,  # Default
                max_tokens=8000,
                suitable_roles=[LLMRole.ASSEMBLER, LLMRole.CODER, LLMRole.CHAT]
            )

        if os.getenv("GEMINI_API_KEY"):
            if "gemini-1.5-flash" not in self.models:
                self.models["gemini-1.5-flash"] = ModelConfig(
                    provider="gemini", model="gemini-1.5-flash", api_key=os.getenv("GEMINI_API_KEY"),
                    temperature=0.3, max_tokens=8000,
                    suitable_roles=[LLMRole.PLANNER, LLMRole.REVIEWER, LLMRole.CHAT, LLMRole.ASSEMBLER, LLMRole.CODER]
                )
            if "gemini-1.5-pro" not in self.models:
                self.models["gemini-1.5-pro"] = ModelConfig(
                    provider="gemini", model="gemini-1.5-pro", api_key=os.getenv("GEMINI_API_KEY"),
                    temperature=0.3, max_tokens=8000,
                    suitable_roles=[LLMRole.PLANNER, LLMRole.REVIEWER, LLMRole.CHAT]
                )

        if os.getenv("ANTHROPIC_API_KEY"):
            self.models["claude-3-5-sonnet-20240620"] = ModelConfig(
                provider="anthropic", model="claude-3-5-sonnet-20240620", api_key=os.getenv("ANTHROPIC_API_KEY"),
                temperature=0.3, max_tokens=4000,
                suitable_roles=[LLMRole.PLANNER, LLMRole.REVIEWER, LLMRole.CHAT, LLMRole.CODER]
                # Claude is good for coding too
            )

        if os.getenv("OPENAI_API_KEY"):
            self.models["gpt-4o"] = ModelConfig(
                provider="openai", model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.3, max_tokens=4000,
                suitable_roles=[LLMRole.PLANNER, LLMRole.REVIEWER, LLMRole.CHAT, LLMRole.CODER]
            )
            self.models["gpt-4o-mini"] = ModelConfig(
                provider="openai", model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.2, max_tokens=16000,
                suitable_roles=[LLMRole.CODER, LLMRole.ASSEMBLER, LLMRole.CHAT]
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
                    if "coder" in model_name.lower() or "starcoder" in model_name.lower() or "codellama" in model_name.lower():
                        current_suitable_roles.extend([LLMRole.CODER, LLMRole.ASSEMBLER])
                        temp = 0.1
                    elif any(kw in model_name.lower() for kw in ["qwen", "llama", "mistral", "mixtral"]):
                        current_suitable_roles.extend(
                            [LLMRole.PLANNER, LLMRole.REVIEWER, LLMRole.CODER, LLMRole.ASSEMBLER])
                    self.models[ollama_key] = ModelConfig(
                        provider="ollama", model=model_name, base_url=ollama_base_url,
                        temperature=temp, max_tokens=4000,
                        suitable_roles=list(set(current_suitable_roles))
                    )
            else:
                print(
                    f"Ollama API not responding as expected at {ollama_base_url}/api/tags. Status: {response.status_code}")
        except requests.exceptions.RequestException:
            print(
                f"Ollama not available or API not responding at {os.getenv('OLLAMA_API_BASE', 'http://localhost:11434')}/api/tags")
        except Exception as e:
            print(f"Error during Ollama model discovery: {e}")

        if os.getenv("DEEPSEEK_API_KEY"):
            self.models["deepseek-coder"] = ModelConfig(
                provider="deepseek", model="deepseek-coder", api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com/v1", temperature=0.1, max_tokens=16000,
                suitable_roles=[LLMRole.CODER, LLMRole.ASSEMBLER]
            )

        # After initializing models, apply temperatures from loaded personalities if they exist
        # This is a simplified approach. A more robust way would be to store default temps per role or model type.
        if hasattr(self, '_loaded_preset_temperatures'):  # A temporary dict to store temps from JSON
            for role_enum, temp_val in self._loaded_preset_temperatures.items():
                assigned_model_key = self.role_assignments.get(role_enum)
                if assigned_model_key and assigned_model_key in self.models:
                    self.models[assigned_model_key].temperature = temp_val
                    print(f"Applied preset temperature {temp_val} to {assigned_model_key} for role {role_enum.value}")
            del self._loaded_preset_temperatures  # Clean up temporary storage

    def _assign_roles(self):
        """Assign best available models to each role. Keys in role_assignments are LLMRole enums."""
        role_preferences = {
            LLMRole.PLANNER: ["gemini-2.5-pro-preview-05-06", "gemini-1.5-pro", "claude-3-5-sonnet-20240620", "gpt-4o",
                              "ollama-qwen2:72b-instruct-q4_K_M", "ollama-mixtral:8x7b-instruct-v0.1-q4_K_M"],
            LLMRole.CODER: ["ollama-qwen2.5-coder:14b", "deepseek-coder", "gpt-4o-mini",
                            "ollama-codellama:13b-instruct-q5_K_M", "gemini-2.5-flash-preview-05-20",
                            "claude-3-5-sonnet-20240620"],
            LLMRole.ASSEMBLER: ["gemini-2.5-flash-preview-05-20", "ollama-qwen2.5-coder:14b", "gpt-4o-mini",
                                "deepseek-coder", "claude-3-5-sonnet-20240620"],
            LLMRole.REVIEWER: ["gemini-2.5-pro-preview-05-06", "claude-3-5-sonnet-20240620", "gpt-4o", "gemini-1.5-pro",
                               "ollama-qwen2:72b-instruct-q4_K_M"],
            LLMRole.CHAT: ["gemini-2.5-pro-preview-05-06", "claude-3-5-sonnet-20240620", "gpt-4o", "gemini-1.5-flash",
                           "ollama-qwen2:14b-instruct-q5_K_M"]
        }

        for role_enum, preferences in role_preferences.items():  # role_enum is an LLMRole member
            assigned = False
            for model_key_preference in preferences:
                if model_key_preference in self.models:
                    config = self.models[model_key_preference]
                    if role_enum in config.suitable_roles:
                        self.role_assignments[role_enum] = model_key_preference
                        print(f"Assigned {model_key_preference} to role {role_enum.value}")
                        assigned = True
                        break
            if not assigned:
                print(f"Warning: No preferred model found for role {role_enum.value}. Attempting general fallback.")
                # Fallback logic remains the same
                available_model_keys = list(self.models.keys())
                if available_model_keys:
                    found_fallback = False
                    for model_key in available_model_keys:
                        if role_enum in self.models[model_key].suitable_roles:
                            self.role_assignments[role_enum] = model_key
                            print(f"Assigned suitable fallback model {model_key} to role {role_enum.value}")
                            found_fallback = True
                            break
                    if not found_fallback:
                        self.role_assignments[role_enum] = available_model_keys[0]
                        print(f"Assigned general fallback model {available_model_keys[0]} to role {role_enum.value}")
                else:
                    print(f"CRITICAL WARNING: No models available. Cannot assign to role {role_enum.value}.")

        # After roles are assigned, apply temperatures from any loaded presets
        # This needs careful handling if presets defined temps for roles, not specific models.
        # The `ModelConfigurationDialog` is the primary place to set temperatures per role/model.
        # Here, we can attempt to apply initial temperatures if they were loaded from `personality_presets.json`
        # based on the *role* the preset was for.

        try:
            presets_file = Path("config") / "personality_presets.json"
            if presets_file.exists():
                with open(presets_file, 'r', encoding='utf-8') as f:
                    all_presets_data = json.load(f)
                for role_str, presets_list in all_presets_data.items():
                    try:
                        role_enum = LLMRole(role_str)
                        assigned_model_key = self.role_assignments.get(role_enum)
                        if assigned_model_key and assigned_model_key in self.models and presets_list:
                            # Take the first preset's temperature for this role as the initial default
                            first_preset = presets_list[0]
                            if "temperature" in first_preset:
                                self.models[assigned_model_key].temperature = float(first_preset["temperature"])
                                print(
                                    f"Applied initial temperature {first_preset['temperature']} from '{first_preset['name']}' preset to {assigned_model_key} for role {role_enum.value}")
                    except ValueError:
                        pass  # Ignore unknown roles in preset file for this step
        except Exception as e:
            print(f"Note: Could not apply initial temperatures from presets during role assignment: {e}")

    def get_role_model(self, role: LLMRole) -> Optional[ModelConfig]:
        """Get the assigned model for a specific role (role is LLMRole enum member)"""
        model_name = self.role_assignments.get(role)
        if model_name:
            return self.models.get(model_name)
        print(f"Warning: No model assigned for role {role.value}")
        return None

    def get_role_assignments(self) -> Dict[str, str]:
        """Get current role assignments for display (keys are role strings)"""
        return {role_enum.value: model_name for role_enum, model_name in self.role_assignments.items()}

    async def chat(self, prompt: str, role: LLMRole = LLMRole.CHAT) -> str:
        model_config = self.get_role_model(role)  # role is LLMRole enum
        personality = self.personalities.get(role, "")  # role is LLMRole enum

        if not model_config:
            print(f"Error: No model configured for role {role.value}. Using fallback response.")
            return self._fallback_response(prompt, role)

        print(
            f"Role: {role.value} using Model: {model_config.provider}/{model_config.model} (Temp: {model_config.temperature}) with Personality: '{personality[:30]}...'")
        try:
            if model_config.provider == "gemini":
                return await self._call_gemini(prompt, model_config, personality)
            elif model_config.provider == "openai":
                return await self._call_openai(prompt, model_config, personality)
            elif model_config.provider == "anthropic":
                return await self._call_anthropic(prompt, model_config, personality)
            elif model_config.provider == "ollama":
                return await self._call_ollama(prompt, model_config, personality)
            elif model_config.provider == "deepseek":
                return await self._call_deepseek(prompt, model_config, personality)
            else:
                print(
                    f"Error: Unknown provider {model_config.provider} for role {role.value}")
                return self._fallback_response(
                    prompt, role)
        except Exception as e:
            print(f"API call with {model_config.provider}/{model_config.model} for role {role.value} failed: {e}")
            if role != LLMRole.CHAT:  # Try fallback
                chat_model_config = self.get_role_model(LLMRole.CHAT)
                if chat_model_config and chat_model_config.model != model_config.model:
                    try:
                        return await self.chat(prompt, LLMRole.CHAT)  # Recurse with CHAT role
                    except Exception as e_chat:
                        print(f"Fallback to CHAT model failed: {e_chat}")
            return self._fallback_response(prompt, role)

    async def stream_chat(self, prompt: str, role: LLMRole = LLMRole.CHAT) -> AsyncGenerator[str, None]:
        model_config = self.get_role_model(role)  # role is LLMRole enum
        personality = self.personalities.get(role, "")  # role is LLMRole enum

        if not model_config:
            print(f"Error: No model for role {role.value} for streaming. Fallback.")
            yield self._fallback_response(prompt, role)
            return

        print(
            f"Streaming - Role: {role.value} Model: {model_config.provider}/{model_config.model} (Temp: {model_config.temperature}) Pers: '{personality[:30]}...'")
        try:
            if model_config.provider == "gemini":
                stream_func = self._stream_gemini
            elif model_config.provider == "openai":
                stream_func = self._stream_openai
            elif model_config.provider == "anthropic":
                stream_func = self._stream_anthropic
            elif model_config.provider == "ollama":
                stream_func = self._stream_ollama
            elif model_config.provider == "deepseek":
                stream_func = self._stream_deepseek
            else:
                print(f"Error: Unknown provider {model_config.provider} for streaming.")
                yield self._fallback_response(
                    prompt, role); return

            async for chunk in stream_func(prompt, model_config, personality): yield chunk
        except Exception as e:
            print(f"Streaming API call for role {role.value} failed: {e}")
            yield self._fallback_response(prompt, role)

    async def _call_gemini(self, prompt: str, config: ModelConfig, personality: str = "") -> str:
        import aiohttp
        final_prompt = f"SYSTEM CONTEXT (Personality: {personality})\n\nUSER PROMPT:\n{prompt}" if personality else prompt
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
        final_prompt = f"SYSTEM CONTEXT (Personality: {personality})\n\nUSER PROMPT:\n{prompt}" if personality else prompt
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.model}:streamGenerateContent?key={config.api_key}&alt=sse"
        payload = {"contents": [{"parts": [{"text": final_prompt}]}],
                   "generationConfig": {"temperature": config.temperature, "maxOutputTokens": config.max_tokens}}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers={"Content-Type": "application/json"}, json=payload) as response:
                if response.status != 200: raise Exception(
                    f"Gemini API stream error {response.status}: {await response.text()}")
                async for line in response.content:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith("data: "):
                        try:
                            data = json.loads(line_str[6:])
                            if data.get("candidates") and data["candidates"][0].get("content", {}).get("parts"):
                                yield data["candidates"][0]["content"]["parts"][0]["text"]
                        except json.JSONDecodeError:
                            print(f"Warning: Gemini stream JSON decode error: {line_str}")

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
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200: raise Exception(
                    f"OpenAI API stream error {response.status}: {await response.text()}")
                async for line in response.content:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith("data: "):
                        data_content = line_str[6:]
                        if data_content == "[DONE]": break
                        try:
                            chunk = json.loads(data_content)
                            if chunk["choices"][0].get("delta", {}).get("content"): yield chunk["choices"][0]["delta"][
                                "content"]
                        except json.JSONDecodeError:
                            print(f"Warning: OpenAI stream JSON decode error: {line_str}")

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
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200: raise Exception(
                    f"Anthropic API stream error {response.status}: {await response.text()}")
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

    async def _call_ollama(self, prompt: str, config: ModelConfig, personality: str = "") -> str:
        import aiohttp
        final_prompt = f"SYSTEM PERSONALITY: {personality}\n\n{prompt}" if personality else prompt
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
        final_prompt = f"SYSTEM PERSONALITY: {personality}\n\n{prompt}" if personality else prompt
        url = f"{config.base_url}/api/generate"
        payload = {"model": config.model, "prompt": final_prompt, "stream": True,
                   "options": {"temperature": config.temperature, "num_predict": config.max_tokens}}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200: raise Exception(
                    f"Ollama API stream error {response.status}: {await response.text()}")
                async for line in response.content:
                    try:
                        if line: data = json.loads(line.decode('utf-8')); yield data.get("response", "");
                        if data.get("done"): break
                    except json.JSONDecodeError:
                        print(f"Warning: Ollama stream JSON decode error: {line.decode('utf-8', errors='ignore')}")

    async def _call_deepseek(self, prompt: str, config: ModelConfig, personality: str = "") -> str:
        import aiohttp
        messages = [{"role": "system", "content": personality}] if personality else []
        messages.append({"role": "user", "content": prompt})
        url = f"{config.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}
        payload = {"model": config.model, "messages": messages, "temperature": config.temperature,
                   "max_tokens": config.max_tokens}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                resp_json = await response.json()
                if response.status == 200: return resp_json["choices"][0]["message"]["content"]
                raise Exception(f"DeepSeek API error {response.status}: {resp_json}")

    async def _stream_deepseek(self, prompt: str, config: ModelConfig, personality: str = "") -> AsyncGenerator[
        str, None]:
        import aiohttp
        messages = [{"role": "system", "content": personality}] if personality else []
        messages.append({"role": "user", "content": prompt})
        url = f"{config.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}
        payload = {"model": config.model, "messages": messages, "temperature": config.temperature,
                   "max_tokens": config.max_tokens, "stream": True}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200: raise Exception(
                    f"DeepSeek API stream error {response.status}: {await response.text()}")
                async for line in response.content:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith("data: "):
                        data_content = line_str[6:]
                        if data_content == "[DONE]": break
                        try:
                            chunk = json.loads(data_content)
                            if chunk["choices"][0].get("delta", {}).get("content"): yield chunk["choices"][0]["delta"][
                                "content"]
                        except json.JSONDecodeError:
                            print(f"Warning: DeepSeek stream JSON decode error: {line_str}")

    def _fallback_response(self, prompt: str, role: LLMRole) -> str:
        return f"# AvA Error: No {role.value} LLM available or API call failed.\n# Request: {prompt[:100]}...\n# Check config and API keys."

    def get_available_models(self) -> list:
        """Get list of available models and their roles (keys are LLMRole enums)"""
        available = []
        if not self.role_assignments: return ["No models assigned to roles yet."]
        for role_enum, model_name_key in self.role_assignments.items():
            if model_name_key in self.models:
                config = self.models[model_name_key]
                available.append(f"{role_enum.value.title()}: {config.provider}/{config.model}")
            else:
                available.append(f"{role_enum.value.title()}: Unassigned or '{model_name_key}' not found")
        if not available and self.models:
            return [f"Models loaded ({len(self.models)}) but not assigned."]
        elif not available:
            return ["No LLM services available or configured."]
        return available


# For backward compatibility or simpler use cases
class LLMClient(EnhancedLLMClient):
    def __init__(self):
        super().__init__()

    def chat(self, prompt: str) -> str:  # Synchronous wrapper
        print("Warning: Synchronous LLMClient.chat() called. Prefer async.")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
        if loop.is_running():
            future = asyncio.run_coroutine_threadsafe(super().chat(prompt, LLMRole.CHAT), loop)
            return future.result(timeout=30)
        else:
            return loop.run_until_complete(super().chat(prompt, LLMRole.CHAT))

    async def stream_chat(self, prompt: str, role: LLMRole = LLMRole.CHAT) -> AsyncGenerator[str, None]:
        async for chunk in super().stream_chat(prompt, role): yield chunk