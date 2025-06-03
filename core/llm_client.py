# core/llm_client.py - Enhanced Multi-Model LLM Client

import os
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
        self.role_assignments: Dict[LLMRole, str] = {}
        self.personalities: Dict[LLMRole, str] = {}  # NEW: Store personalities
        self._initialize_models()
        self._assign_roles()
        self._initialize_default_personalities() # NEW: Initialize default personalities

    def _initialize_default_personalities(self):
        """Initialize default personalities if not set by config dialog"""
        # These would be overridden if a config dialog sets them
        default_personalities = {
            LLMRole.PLANNER: "You are a senior software architect with 15+ years of experience. You think strategically, break down complex problems into clear steps, and always consider scalability and maintainability. You communicate clearly and provide detailed technical specifications.",
            LLMRole.CODER: "You are a coding specialist who writes clean, efficient, and well-documented code. You follow best practices, use proper error handling, and write code that is both functional and elegant. You focus on getting things done with high quality.",
            LLMRole.ASSEMBLER: "You are a meticulous code integrator who ensures all pieces work together seamlessly. You have an eye for detail, maintain consistent code style, and create professional, production-ready files with proper organization and documentation.",
            LLMRole.REVIEWER: "You are a detail-oriented code reviewer. Your primary goal is to ensure code quality, adherence to best practices, security, and performance. Provide constructive feedback and clear justifications for any issues found.",
            LLMRole.CHAT: "You are AvA, a friendly and helpful AI development assistant. Engage in natural conversation and guide users through their development tasks."
        }
        for role, personality_text in default_personalities.items():
            if role not in self.personalities: # Only set if not already loaded (e.g., by config)
                self.personalities[role] = personality_text
                print(f"Initialized default personality for {role.value}")


    def _initialize_models(self):
        """Initialize available models with their configurations"""

        # User-specified Gemini models
        if os.getenv("GEMINI_API_KEY"):
            self.models["gemini-2.5-pro-preview-05-06"] = ModelConfig(
                provider="gemini",
                model="gemini-2.5-pro-preview-05-06",  # For Planner & Chat
                api_key=os.getenv("GEMINI_API_KEY"),
                temperature=0.1,
                max_tokens=8000,
                suitable_roles=[LLMRole.PLANNER, LLMRole.REVIEWER, LLMRole.CHAT]
            )
            self.models["gemini-2.5-flash-preview-05-20"] = ModelConfig(
                provider="gemini",
                model="gemini-2.5-flash-preview-05-20",  # For Assembler
                api_key=os.getenv("GEMINI_API_KEY"),
                temperature=0.1,
                max_tokens=8000,
                suitable_roles=[LLMRole.ASSEMBLER, LLMRole.CODER, LLMRole.CHAT]
            )

        # Existing Gemini models (can serve as fallbacks or for other roles)
        if os.getenv("GEMINI_API_KEY"):
            if "gemini-1.5-flash" not in self.models:
                self.models["gemini-1.5-flash"] = ModelConfig(
                    provider="gemini",
                    model="gemini-1.5-flash",
                    api_key=os.getenv("GEMINI_API_KEY"),
                    temperature=0.1,  # General purpose temperature
                    max_tokens=8000,
                    suitable_roles=[LLMRole.PLANNER, LLMRole.REVIEWER, LLMRole.CHAT, LLMRole.ASSEMBLER, LLMRole.CODER]
                )
            if "gemini-1.5-pro" not in self.models:  # Ensure not to overwrite a more specific "2.5-pro" if it was somehow keyed as "gemini-1.5-pro"
                self.models["gemini-1.5-pro"] = ModelConfig(
                    provider="gemini",
                    model="gemini-1.5-pro",
                    api_key=os.getenv("GEMINI_API_KEY"),
                    temperature=0.1,
                    max_tokens=8000,
                    suitable_roles=[LLMRole.PLANNER, LLMRole.REVIEWER, LLMRole.CHAT]
                )

        if os.getenv("ANTHROPIC_API_KEY"):
            # Assuming claude-3-5-sonnet-20240620 is a valid model identifier
            self.models["claude-3-5-sonnet-20240620"] = ModelConfig(
                provider="anthropic",
                model="claude-3-5-sonnet-20240620",
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                temperature=0.1,
                max_tokens=4000,
                suitable_roles=[LLMRole.PLANNER, LLMRole.REVIEWER, LLMRole.CHAT]
            )

        if os.getenv("OPENAI_API_KEY"):
            self.models["gpt-4o"] = ModelConfig(
                provider="openai",
                model="gpt-4o",
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.1,
                max_tokens=4000,  # GPT-4o's typical response length, can be higher
                suitable_roles=[LLMRole.PLANNER, LLMRole.REVIEWER, LLMRole.CHAT]
            )
            self.models["gpt-4o-mini"] = ModelConfig(
                provider="openai",
                model="gpt-4o-mini",
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.1,
                max_tokens=16000,  # Max output for gpt-4o-mini is 16k
                suitable_roles=[LLMRole.CODER, LLMRole.ASSEMBLER, LLMRole.CHAT]
            )

        # Ollama models
        try:
            ollama_base_url = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
            response = requests.get(f"{ollama_base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                ollama_models_available = response.json().get("models", [])
                for model_info in ollama_models_available:
                    model_name = model_info["name"]  # e.g., "qwen2.5-coder:14b"
                    ollama_key = f"ollama-{model_name}"  # e.g., "ollama-qwen2.5-coder:14b"

                    current_suitable_roles = [LLMRole.CHAT]
                    temp = 0.1  # Default temperature
                    if "coder" in model_name.lower() or "code" in model_name.lower():
                        current_suitable_roles.extend([LLMRole.CODER, LLMRole.ASSEMBLER])
                        temp = 0.05  # Lower temp for coding
                    elif "qwen" in model_name.lower() or "llama" in model_name.lower():  # General purpose models
                        current_suitable_roles.extend([LLMRole.PLANNER, LLMRole.CODER, LLMRole.ASSEMBLER])

                    self.models[ollama_key] = ModelConfig(
                        provider="ollama",
                        model=model_name,  # Pass the exact model name for Ollama API
                        base_url=ollama_base_url,
                        temperature=temp,
                        max_tokens=4000,  # Adjust as needed per model
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
                provider="deepseek",
                model="deepseek-coder",  # Or more specific version like "deepseek-coder-6.7b-instruct"
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com/v1",
                temperature=0.05,
                max_tokens=16000,  # deepseek-coder supports larger context
                suitable_roles=[LLMRole.CODER, LLMRole.ASSEMBLER]
            )

    def _assign_roles(self):
        """Assign best available models to each role"""

        role_preferences = {
            LLMRole.PLANNER: [
                "gemini-2.5-pro-preview-05-06",  # User specified
                "gemini-1.5-pro", "claude-3-5-sonnet-20240620", "gpt-4o",
                "gemini-1.5-flash", "ollama-qwen2.5:32b", "gpt-4o-mini"  # Fallbacks
            ],
            LLMRole.CODER: [
                "ollama-qwen2.5-coder:14b",  # User specified
                "ollama-qwen2.5-coder:32b", "deepseek-coder",
                "ollama-codellama:13b", "gpt-4o-mini",
                "gemini-2.5-flash-preview-05-20", "gemini-1.5-flash", "claude-3-5-sonnet-20240620"  # Fallbacks
            ],
            LLMRole.ASSEMBLER: [
                "gemini-2.5-flash-preview-05-20",  # User specified
                "ollama-qwen2.5-coder:14b", "deepseek-coder", "gpt-4o-mini",
                "gemini-1.5-flash", "claude-3-5-sonnet-20240620"  # Fallbacks
            ],
            LLMRole.REVIEWER: [
                "gemini-2.5-pro-preview-05-06",  # Can use the powerful planner model
                "claude-3-5-sonnet-20240620", "gpt-4o", "gemini-1.5-pro",
                "gemini-1.5-flash", "gpt-4o-mini"  # Fallbacks
            ],
            LLMRole.CHAT: [
                "gemini-2.5-pro-preview-05-06",  # User specified for Chat
                "gemini-1.5-flash", "gpt-4o-mini", "claude-3-5-sonnet-20240620",
                "ollama-qwen2.5:14b", "gpt-4o"  # Fallbacks
            ]
        }

        for role, preferences in role_preferences.items():
            for model_key_preference in preferences:
                # Check if the model_key_preference (e.g., "ollama-qwen2.5-coder:14b") exists in self.models
                if model_key_preference in self.models:
                    config = self.models[model_key_preference]
                    if role in config.suitable_roles:
                        self.role_assignments[role] = model_key_preference
                        print(f"Assigned {model_key_preference} to role {role.value}")
                        break
            if role not in self.role_assignments:
                print(f"Warning: No suitable model found for role {role.value} from preferences. Will try fallback.")

        # Fallback to any available model if a role is still not assigned
        available_model_keys = list(self.models.keys())
        for role_enum_member in LLMRole:  # Iterate through all defined roles
            if role_enum_member not in self.role_assignments and available_model_keys:
                # Try to find any model that lists this role as suitable
                found_fallback = False
                for model_key in available_model_keys:
                    if role_enum_member in self.models[model_key].suitable_roles:
                        self.role_assignments[role_enum_member] = model_key
                        print(f"Assigned fallback model {model_key} to role {role_enum_member.value}")
                        found_fallback = True
                        break
                if not found_fallback:  # If no specifically suitable model, assign the first available one
                    self.role_assignments[role_enum_member] = available_model_keys[0]
                    print(f"Assigned general fallback model {available_model_keys[0]} to role {role_enum_member.value}")
            elif role_enum_member not in self.role_assignments:
                print(
                    f"CRITICAL WARNING: No models available at all. Cannot assign any model to role {role_enum_member.value}.")

    def get_role_model(self, role: LLMRole) -> Optional[ModelConfig]:
        """Get the assigned model for a specific role"""
        model_name = self.role_assignments.get(role)
        if model_name:
            return self.models.get(model_name)
        print(f"Warning: No model assigned for role {role.value}")
        return None

    def get_role_assignments(self) -> Dict[str, str]:
        """Get current role assignments for display"""
        return {role.value: model_name for role, model_name in self.role_assignments.items()}

    async def chat(self, prompt: str, role: LLMRole = LLMRole.CHAT) -> str:
        model_config = self.get_role_model(role)
        personality = self.personalities.get(role, "") # NEW: Get personality

        if not model_config:
            print(f"Error: No model configured for role {role.value}. Using fallback response.")
            return self._fallback_response(prompt, role)

        print(f"Role: {role.value} using Model: {model_config.provider}/{model_config.model} with Personality: '{personality[:30]}...'")
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
                print(f"Error: Unknown provider {model_config.provider} for role {role.value}")
                return self._fallback_response(prompt, role)
        except Exception as e:
            print(f"API call with {model_config.provider}/{model_config.model} for role {role.value} failed: {e}")
            # Simple fallback: try the first available CHAT model if different
            if role != LLMRole.CHAT:
                print(f"Attempting fallback to CHAT model for role {role.value}")
                chat_model_config = self.get_role_model(LLMRole.CHAT)
                if chat_model_config and chat_model_config.model != model_config.model:  # Avoid re-calling the same failed model
                    try:
                        # Pass original role to get correct personality for fallback
                        return await self.chat(prompt, LLMRole.CHAT)
                    except Exception as e_chat:
                        print(f"Fallback to CHAT model also failed: {e_chat}")
            return self._fallback_response(prompt, role)

    async def stream_chat(self, prompt: str, role: LLMRole = LLMRole.CHAT) -> AsyncGenerator[str, None]:
        model_config = self.get_role_model(role)
        personality = self.personalities.get(role, "") # NEW: Get personality

        if not model_config:
            print(f"Error: No model configured for role {role.value} for streaming. Using fallback response.")
            yield self._fallback_response(prompt, role)
            return

        print(f"Streaming - Role: {role.value} using Model: {model_config.provider}/{model_config.model} with Personality: '{personality[:30]}...'")
        try:
            if model_config.provider == "gemini":
                async for chunk in self._stream_gemini(prompt, model_config, personality):
                    yield chunk
            elif model_config.provider == "openai":
                async for chunk in self._stream_openai(prompt, model_config, personality):
                    yield chunk
            elif model_config.provider == "anthropic":
                async for chunk in self._stream_anthropic(prompt, model_config, personality):
                    yield chunk
            elif model_config.provider == "ollama":
                async for chunk in self._stream_ollama(prompt, model_config, personality):
                    yield chunk
            elif model_config.provider == "deepseek":
                async for chunk in self._stream_deepseek(prompt, model_config, personality):
                    yield chunk
            else:
                print(f"Error: Unknown provider {model_config.provider} for role {role.value} for streaming.")
                yield self._fallback_response(prompt, role)
        except Exception as e:
            print(
                f"Streaming API call with {model_config.provider}/{model_config.model} for role {role.value} failed: {e}")
            yield self._fallback_response(prompt, role)

    async def _call_gemini(self, prompt: str, config: ModelConfig, personality: str = "") -> str:
        import aiohttp
        final_prompt = prompt
        if personality:
            final_prompt = f"SYSTEM CONTEXT (Personality: {personality})\n\nUSER PROMPT:\n{prompt}"

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.model}:generateContent?key={config.api_key}"
        payload = {
            "contents": [{"parts": [{"text": final_prompt}]}],
            "generationConfig": {
                "temperature": config.temperature,
                "maxOutputTokens": config.max_tokens
            }
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers={"Content-Type": "application/json"}, json=payload) as response:
                response_json = await response.json()
                if response.status == 200:
                    if ("candidates" in response_json and len(response_json["candidates"]) > 0 and
                            "content" in response_json["candidates"][0] and
                            "parts" in response_json["candidates"][0]["content"] and
                            len(response_json["candidates"][0]["content"]["parts"]) > 0):
                        return response_json["candidates"][0]["content"]["parts"][0]["text"]
                    else:
                        raise Exception(f"Gemini API response format error: {response_json}")
                raise Exception(f"Gemini API error: {response.status} - {response_json}")

    async def _stream_gemini(self, prompt: str, config: ModelConfig, personality: str = "") -> AsyncGenerator[str, None]:
        import aiohttp
        final_prompt = prompt
        if personality:
            final_prompt = f"SYSTEM CONTEXT (Personality: {personality})\n\nUSER PROMPT:\n{prompt}"

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.model}:streamGenerateContent?key={config.api_key}&alt=sse"
        payload = {
            "contents": [{"parts": [{"text": final_prompt}]}],
            "generationConfig": {
                "temperature": config.temperature,
                "maxOutputTokens": config.max_tokens
            }
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers={"Content-Type": "application/json"}, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Gemini API stream error: {response.status} - {error_text}")

                async for line in response.content:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith("data: "):
                        try:
                            data_json = json.loads(line_str[6:])
                            if ("candidates" in data_json and len(data_json["candidates"]) > 0 and
                                    "content" in data_json["candidates"][0] and
                                    "parts" in data_json["candidates"][0]["content"] and
                                    len(data_json["candidates"][0]["content"]["parts"]) > 0):
                                yield data_json["candidates"][0]["content"]["parts"][0]["text"]
                        except json.JSONDecodeError:
                            print(f"Warning: Could not decode JSON from Gemini stream: {line_str}")
                            continue

    async def _call_openai(self, prompt: str, config: ModelConfig, personality: str = "") -> str:
        import aiohttp
        messages = []
        if personality:
            messages.append({"role": "system", "content": personality})
        messages.append({"role": "user", "content": prompt})

        url = config.base_url or "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                response_json = await response.json()
                if response.status == 200:
                    return response_json["choices"][0]["message"]["content"]
                raise Exception(f"OpenAI API error: {response.status} - {response_json}")

    async def _stream_openai(self, prompt: str, config: ModelConfig, personality: str = "") -> AsyncGenerator[str, None]:
        import aiohttp
        messages = []
        if personality:
            messages.append({"role": "system", "content": personality})
        messages.append({"role": "user", "content": prompt})

        url = config.base_url or "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "stream": True
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"OpenAI API stream error: {response.status} - {error_text}")

                async for line in response.content:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith("data: "):
                        data_content = line_str[6:]
                        if data_content == "[DONE]":
                            break
                        try:
                            chunk_data = json.loads(data_content)
                            if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                                delta = chunk_data["choices"][0].get("delta", {})
                                if "content" in delta and delta["content"] is not None:
                                    yield delta["content"]
                        except json.JSONDecodeError:
                            print(f"Warning: Could not decode JSON from OpenAI stream: {line_str}")
                            continue

    async def _call_anthropic(self, prompt: str, config: ModelConfig, personality: str = "") -> str:
        import aiohttp
        url = config.base_url or "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": config.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        payload = {
            "model": config.model,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        if personality:
            payload["system"] = personality # Anthropic uses a dedicated system field

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                response_json = await response.json()
                if response.status == 200:
                    return response_json["content"][0]["text"]
                raise Exception(f"Anthropic API error: {response.status} - {response_json}")

    async def _stream_anthropic(self, prompt: str, config: ModelConfig, personality: str = "") -> AsyncGenerator[str, None]:
        import aiohttp
        url = config.base_url or "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": config.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "Accept": "text/event-stream"
        }
        payload = {
            "model": config.model,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True
        }
        if personality:
            payload["system"] = personality

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Anthropic API stream error: {response.status} - {error_text}")

                async for line in response.content:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith("event: message_delta"):
                        # The actual data comes in the next line starting with "data: "
                        continue
                    if line_str.startswith("data: "):
                        try:
                            data_json = json.loads(line_str[6:])
                            if data_json.get("type") == "content_block_delta":
                                if data_json["delta"]["type"] == "text_delta":
                                    yield data_json["delta"]["text"]
                            elif data_json.get("type") == "message_stop":
                                break
                        except json.JSONDecodeError:
                            print(f"Warning: Could not decode JSON from Anthropic stream: {line_str}")
                            continue

    async def _call_ollama(self, prompt: str, config: ModelConfig, personality: str = "") -> str:
        import aiohttp
        final_prompt = prompt
        if personality:
            final_prompt = f"SYSTEM PERSONALITY: {personality}\n\n{prompt}"

        url = f"{config.base_url}/api/generate"
        payload = {
            "model": config.model,
            "prompt": final_prompt,
            "stream": False,
            "options": {
                "temperature": config.temperature,
                "num_predict": config.max_tokens  # num_predict often used for max_tokens
            }
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                response_json = await response.json()
                if response.status == 200:
                    return response_json["response"]
                raise Exception(f"Ollama API error: {response.status} - {response_json}")

    async def _stream_ollama(self, prompt: str, config: ModelConfig, personality: str = "") -> AsyncGenerator[str, None]:
        import aiohttp
        final_prompt = prompt
        if personality:
            final_prompt = f"SYSTEM PERSONALITY: {personality}\n\n{prompt}"

        url = f"{config.base_url}/api/generate"
        payload = {
            "model": config.model,
            "prompt": final_prompt,
            "stream": True,
            "options": {
                "temperature": config.temperature,
                "num_predict": config.max_tokens
            }
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Ollama API stream error: {response.status} - {error_text}")

                async for line in response.content:
                    try:
                        if line:  # Ensure line is not empty
                            data = json.loads(line.decode('utf-8'))
                            if "response" in data:
                                yield data["response"]
                            if data.get("done", False):
                                break
                    except json.JSONDecodeError:
                        print(
                            f"Warning: Could not decode JSON from Ollama stream: {line.decode('utf-8', errors='ignore')}")
                        continue

    async def _call_deepseek(self, prompt: str, config: ModelConfig, personality: str = "") -> str:
        import aiohttp
        messages = []
        if personality:
            messages.append({"role": "system", "content": personality})
        messages.append({"role": "user", "content": prompt})

        url = f"{config.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                response_json = await response.json()
                if response.status == 200:
                    return response_json["choices"][0]["message"]["content"]
                raise Exception(f"DeepSeek API error: {response.status} - {response_json}")

    async def _stream_deepseek(self, prompt: str, config: ModelConfig, personality: str = "") -> AsyncGenerator[str, None]:
        import aiohttp
        messages = []
        if personality:
            messages.append({"role": "system", "content": personality})
        messages.append({"role": "user", "content": prompt})

        url = f"{config.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "stream": True
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"DeepSeek API stream error: {response.status} - {error_text}")

                async for line in response.content:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith("data: "):
                        data_content = line_str[6:]
                        if data_content == "[DONE]":
                            break
                        try:
                            chunk_data = json.loads(data_content)
                            if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                                delta = chunk_data["choices"][0].get("delta", {})
                                if "content" in delta and delta["content"] is not None:
                                    yield delta["content"]
                        except json.JSONDecodeError:
                            print(f"Warning: Could not decode JSON from DeepSeek stream: {line_str}")
                            continue

    def _fallback_response(self, prompt: str, role: LLMRole) -> str:
        return f"""
# AvA Error: No {role.value} LLM available or API call failed.

# Request: {prompt[:100]}...

# Ensure the intended model is configured and API keys are set:
# - GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, DEEPSEEK_API_KEY
# - For Ollama, ensure the service is running and models are pulled.

# Check logs for specific API errors.
"""

    def get_available_models(self) -> list:
        """Get list of available models and their roles"""
        available = []
        if not self.role_assignments:
            return ["No models assigned to roles yet. Check initialization."]
        for role, model_name_key in self.role_assignments.items():
            if model_name_key in self.models:
                config = self.models[model_name_key]
                available.append(f"{role.value.title()}: {config.provider}/{config.model}")
            else:
                available.append(f"{role.value.title()}: Unassigned or model '{model_name_key}' not found")

        if not available and self.models:
            return [f"Models loaded ({len(self.models)}) but not assigned to roles. Check _assign_roles logic."]
        elif not available:
            return ["No LLM services available or configured."]
        return available


# For backward compatibility or simpler use cases if needed elsewhere
class LLMClient(EnhancedLLMClient):
    def __init__(self):
        super().__init__()

    # Note: The synchronous 'chat' method is problematic in an async app.
    # It's better to use 'await EnhancedLLMClient.chat()' directly.
    # This is retained only if strictly needed for old synchronous code.
    def chat(self, prompt: str) -> str:
        print("Warning: Synchronous LLMClient.chat() called. Consider using async version.")
        # This is a simplified way to call the async method from sync code.
        # In a real Qt app, this could block the UI if not handled carefully.
        # It's better to refactor calling code to be async.
        try:
            # Try to get an existing loop if running in a context like qasync
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:  # No running event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            if loop.is_running():
                # This is tricky. If the loop is qasync's, submitting and waiting
                # can deadlock. If it's a different thread's loop, it's also complex.
                # The best approach is for the caller to be async.
                # For a simple non-GUI script context, this might work:
                future = asyncio.run_coroutine_threadsafe(super().chat(prompt, LLMRole.CHAT), loop)
                return future.result(timeout=30) # Blocking call with timeout
            else:
                return loop.run_until_complete(super().chat(prompt, LLMRole.CHAT))
        except Exception as e:
            print(f"Error running async chat in sync wrapper: {e}")
            return self._fallback_response(prompt, LLMRole.CHAT)

    async def stream_chat(self, prompt: str, role: LLMRole = LLMRole.CHAT) -> AsyncGenerator[str, None]:
        # Pass the role argument to the superclass method
        async for chunk in super().stream_chat(prompt, role):
            yield chunk