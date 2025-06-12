# core/llm_client.py - V5.1 - Integrated a wider range of local models

import json
import os
import asyncio
import random
from enum import Enum
from pathlib import Path
from typing import AsyncGenerator, Optional, Dict, Any, Callable, Awaitable
from functools import wraps
import aiohttp
import logging

logger = logging.getLogger(__name__)


# --- Corrected Retry Decorator for ASYNC GENERATORS ---
def retry_async_generator(retries=4, delay=2.0, backoff=2.0):
    """
    A decorator for retrying an async GENERATOR with exponential backoff.
    This is specifically designed for functions that use `async for` loops.
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay
            for i in range(retries):
                try:
                    # Get the generator object from the decorated function call
                    async_gen = func(*args, **kwargs)
                    # Yield from the generator
                    async for item in async_gen:
                        yield item
                    # If the loop completes without error, we are done.
                    return
                except Exception as e:
                    logger.warning(
                        f"API call for generator {func.__name__} failed (Attempt {i + 1}/{retries}). Error: {e}. Retrying in {current_delay:.2f}s...")
                    if i >= retries - 1:
                        logger.error(
                            f"API call for generator {func.__name__} failed after {retries} retries. Giving up.")
                        raise  # Re-raise the last exception
                    await asyncio.sleep(current_delay)
                    # Add jitter to avoid thundering herd problem
                    current_delay = (current_delay * backoff) + (random.uniform(0, 1))

        return wrapper

    return decorator


class LLMRole(Enum):
    """The core AI specialist roles for the V4 workflow."""
    ARCHITECT = "architect"
    CODER = "coder"
    ASSEMBLER = "assembler"
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
        self.config_dir = Path("config")
        self.config_dir.mkdir(exist_ok=True)
        self.assignments_file = self.config_dir / "role_assignments.json"

        self.role_assignments: Dict[LLMRole, str] = {}
        self.personalities: Dict[LLMRole, str] = {}
        self.role_temperatures: Dict[LLMRole, float] = {}
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
                        if role_str in ["planner", "structurer"]:
                            role_enum = LLMRole.ARCHITECT
                        elif role_str == "coder":
                            role_enum = LLMRole.CODER
                        else:
                            role_enum = LLMRole(role_str)

                        if presets_list:
                            user_presets = [p for p in presets_list if p.get("author") == "User"]
                            preset_to_load = user_presets[-1] if user_presets else presets_list[0]
                            self.personalities[role_enum] = preset_to_load["personality"]
                            self.role_temperatures[role_enum] = preset_to_load.get("temperature", 0.7)
                    except (ValueError, KeyError) as e:
                        logger.warning(f"Could not load personality for role '{role_str}': {e}")
                        continue
        except Exception as e:
            logger.error(f"Error loading personalities from config: {e}")

    def _initialize_default_personalities(self):
        """Initialize default personalities for the new role structure."""
        default_personalities_map = {
            LLMRole.ARCHITECT: "You are the ARCHITECT AI, a master software architect. Your task is to create a complete, comprehensive, and machine-readable Technical Specification Sheet for an entire software project based on a user's request. This sheet will be the single source of truth for all other AI agents.",
            LLMRole.CODER: "You are an expert Python developer. Your task is to generate a single, complete, and production-ready Python file based on a strict Technical Specification and the full source code of its dependencies.",
            LLMRole.ASSEMBLER: "You are the ASSEMBLER AI. Combine these micro-task implementations into a complete, professional Python file.",
            LLMRole.REVIEWER: "You are a senior code reviewer. Your primary goal is to ensure the generated code is of high quality, correct, and adheres to the technical specification. Provide a final 'approved' status and a brief summary.",
            LLMRole.CHAT: "You are AvA, a friendly and helpful AI development assistant."
        }
        for role_enum, personality_text in default_personalities_map.items():
            if role_enum not in self.personalities:
                self.personalities[role_enum] = personality_text

    def _initialize_models(self):
        """Initialize available models from environment variables."""
        # --- OpenAI ---
        if api_key := os.getenv("OPENAI_API_KEY"):
            base_url = os.getenv("OPENAI_API_BASE")
            self.models["gpt-4o"] = ModelConfig("openai", "gpt-4o", api_key, base_url, 0.5, 8000,
                                                [LLMRole.ARCHITECT, LLMRole.CODER, LLMRole.ASSEMBLER, LLMRole.REVIEWER,
                                                 LLMRole.CHAT])
            self.models["gpt-4o-mini"] = ModelConfig("openai", "gpt-4o-mini", api_key, base_url, 0.7, 16000,
                                                     [LLMRole.CHAT])
            logger.info("✅ OpenAI models loaded.")

        # --- Anthropic ---
        if api_key := os.getenv("ANTHROPIC_API_KEY"):
            base_url = os.getenv("ANTHROPIC_API_BASE")
            self.models["claude-3-5-sonnet-20240620"] = ModelConfig("anthropic", "claude-3-5-sonnet-20240620", api_key,
                                                                    base_url, 0.5, 8000,
                                                                    [LLMRole.ARCHITECT, LLMRole.CODER,
                                                                     LLMRole.ASSEMBLER, LLMRole.REVIEWER, LLMRole.CHAT])
            self.models["claude-3-opus-20240229"] = ModelConfig("anthropic", "claude-3-opus-20240229", api_key,
                                                                base_url, 0.5, 8000,
                                                                [LLMRole.ARCHITECT, LLMRole.REVIEWER])
            logger.info("✅ Anthropic models loaded.")

        # --- Google Gemini ---
        if api_key := os.getenv("GEMINI_API_KEY"):
            base_url = os.getenv("GOOGLE_API_BASE")
            self.models["gemini-2.5-pro-preview-06-05"] = ModelConfig("gemini", "gemini-2.5-pro-preview-06-05", api_key,
                                                                      base_url, 0.3, 8000,
                                                                      [LLMRole.ARCHITECT, LLMRole.REVIEWER])
            self.models["gemini-2.5-flash-preview-05-20"] = ModelConfig("gemini", "gemini-2.5-flash-preview-05-20",
                                                                        api_key, base_url, 0.7, 8000,
                                                                        [LLMRole.CODER, LLMRole.REVIEWER, LLMRole.CHAT])
            logger.info("✅ Google Gemini models loaded.")

        # --- DeepSeek ---
        if api_key := os.getenv("DEEPSEEK_API_KEY"):
            base_url = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
            self.models["deepseek-reasoner"] = ModelConfig("deepseek", "deepseek-reasoner", api_key, base_url, 0.1,
                                                           32000, [LLMRole.ARCHITECT, LLMRole.CODER, LLMRole.ASSEMBLER])
            self.models["deepseek-coder"] = ModelConfig("deepseek", "deepseek-coder", api_key, base_url, 0.1, 32000,
                                                        [LLMRole.CODER])
            self.models["deepseek-chat"] = ModelConfig("deepseek", "deepseek-chat", api_key, base_url, 0.7, 8000,
                                                       [LLMRole.CHAT])
            logger.info("✅ DeepSeek models loaded.")

        # --- Ollama (Local) ---
        ollama_base_url = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")

        # --- Good All-Rounders / Chat Models ---
        self.models["ollama_llama3"] = ModelConfig("ollama", "llama3", base_url=ollama_base_url,
                                                   temperature=0.7, suitable_roles=[LLMRole.CHAT, LLMRole.REVIEWER])
        self.models["ollama_qwen2.5-coder"] = ModelConfig("ollama", "qwen2.5-coder:latest", base_url=ollama_base_url,
                                                          temperature=0.6,
                                                          suitable_roles=[LLMRole.CHAT, LLMRole.ARCHITECT])

        # --- Specialist Coding Models (Now with general versions) ---
        self.models["ollama_codellama-13b"] = ModelConfig("ollama", "codellama:13b", base_url=ollama_base_url,
                                                          temperature=0.1, suitable_roles=[LLMRole.CODER])
        self.models["ollama_codellama-13b-instruct"] = ModelConfig("ollama", "codellama:13b-instruct",
                                                                   base_url=ollama_base_url,
                                                                   temperature=0.1, suitable_roles=[LLMRole.CODER])
        self.models["ollama_starcoder2-15b-instruct"] = ModelConfig("ollama", "starcoder2:15b-instruct",
                                                                    base_url=ollama_base_url,
                                                                    temperature=0.1, suitable_roles=[LLMRole.CODER])

        logger.info("✅ Ollama local models configured. Make sure your Ollama server is running!")

    def save_assignments(self):
        """Saves the current role assignments to the JSON file."""
        logger.info(f"Saving role assignments to {self.assignments_file}")
        assignments_to_save = {
            role.value: model_name for role, model_name in self.role_assignments.items()
        }
        with open(self.assignments_file, 'w', encoding='utf-8') as f:
            json.dump(assignments_to_save, f, indent=2)

    def _assign_roles(self):
        """
        Assigns models to roles. It first tries to load from the config file,
        then falls back to smart defaults.
        """
        if self.assignments_file.exists():
            logger.info(f"Loading role assignments from {self.assignments_file}")
            with open(self.assignments_file, 'r', encoding='utf-8') as f:
                try:
                    loaded_assignments = json.load(f)
                    for role_str, model_name in loaded_assignments.items():
                        if model_name in self.models:
                            self.role_assignments[LLMRole(role_str)] = model_name
                        else:
                            logger.warning(
                                f"Model '{model_name}' for role '{role_str}' from config is not available. Will re-assign.")
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Error loading role assignments file: {e}. Falling back to defaults.")

        preferences = {
            LLMRole.ARCHITECT: ["gemini-2.5-pro-preview-06-05", "deepseek-reasoner", "claude-3-opus-20240229",
                                "gpt-4o"],
            LLMRole.CODER: ["gemini-2.5-flash-preview-05-20", "deepseek-reasoner", "claude-3-5-sonnet-20240620",
                            "gpt-4o"],
            LLMRole.ASSEMBLER: ["deepseek-reasoner", "claude-3-5-sonnet-20240620", "gpt-4o"],
            LLMRole.REVIEWER: ["gemini-2.5-flash-preview-05-20", "gpt-4o-mini"],
            LLMRole.CHAT: ["deepseek-chat", "gemini-2.5-flash-preview-05-20", "gpt-4o-mini"]
        }

        for role, preferred_models in preferences.items():
            if role not in self.role_assignments:
                assigned_model = next((model_name for model_name in preferred_models if model_name in self.models),
                                      None)
                if not assigned_model:
                    assigned_model = next((model_key for model_key, model_config in self.models.items() if
                                           role in model_config.suitable_roles), None)
                if assigned_model:
                    self.role_assignments[role] = assigned_model

        available_models = list(self.models.keys())
        if not available_models:
            logger.error("❌ No models available. Please check your API keys in the .env file.")
            return

        for role in LLMRole:
            if role not in self.role_assignments:
                self.role_assignments[role] = available_models[0]
                logger.warning(f"Role '{role.value}' was unassigned, falling back to '{available_models[0]}'.")

        for role, model_name in self.role_assignments.items():
            if role not in self.role_temperatures and model_name and model_name in self.models:
                self.role_temperatures[role] = self.models[model_name].temperature
        if LLMRole.CODER in self.role_temperatures: self.role_temperatures[LLMRole.CODER] = 0.1

        logger.info("🎯 Final Role Assignments:")
        for role, model_name in self.role_assignments.items():
            if model_name and model_name in self.models:
                provider = self.models[model_name].provider
                temp = self.role_temperatures.get(role, "N/A")
                logger.info(f"  {role.value.title():<10}: {provider}/{self.models[model_name].model} (Temp: {temp})")
            else:
                logger.warning(f"  {role.value.title():<10}: ❌ Unassigned")

        self.save_assignments()

    def get_role_model(self, role: LLMRole) -> Optional[ModelConfig]:
        """Get the model configuration assigned to a role."""
        model_name = self.role_assignments.get(role)
        if not model_name or model_name not in self.models:
            logger.warning(f"Model for role {role.value} ('{model_name}') not available. Check API keys.")
            return None
        return self.models.get(model_name)

    async def chat(self, prompt: str, role: LLMRole = LLMRole.CHAT) -> str:
        """Standard, non-streaming chat call."""
        full_response = ""
        try:
            full_response = "".join([chunk async for chunk in self.stream_chat(prompt, role)])
            return full_response
        except Exception as e:
            logger.error(f"Chat failed for role {role.value}: {e}", exc_info=True)
            return f"# AvA Error: Chat failed for role {role.value}. See logs for details."

    @retry_async_generator()
    async def stream_chat(self, prompt: str, role: LLMRole = LLMRole.CHAT) -> AsyncGenerator[str, None]:
        """Streaming chat call with robust error handling and retry logic."""
        model_config = self.get_role_model(role)
        personality = self.personalities.get(role, "")
        if not model_config:
            logger.error(f"No model configured for role: {role.value}")
            yield self._fallback_response(prompt, role)
            return

        role_temp = self.role_temperatures.get(role, model_config.temperature)
        call_config = ModelConfig(
            provider=model_config.provider, model=model_config.model, api_key=model_config.api_key,
            base_url=model_config.base_url, temperature=role_temp, max_tokens=model_config.max_tokens
        )

        provider_map = {
            "gemini": self._stream_gemini, "anthropic": self._stream_anthropic,
            "deepseek": self._stream_deepseek, "openai": self._stream_openai,
            "ollama": self._stream_ollama
        }
        stream_func = provider_map.get(call_config.provider)

        if not stream_func:
            logger.error(f"Unsupported provider for streaming: {call_config.provider}")
            yield self._fallback_response(prompt, role)
            return

        async for chunk in stream_func(prompt, call_config, personality):
            yield chunk

    async def _stream_openai(self, prompt, config, personality):
        messages = [{"role": "system", "content": personality}] if personality else []
        messages.append({"role": "user", "content": prompt})
        url = f"{config.base_url or 'https://api.openai.com'}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}
        payload = {"model": config.model, "messages": messages, "temperature": config.temperature,
                   "max_tokens": config.max_tokens, "stream": True}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"OpenAI stream API error {response.status}: {error_text}")
                async for line in response.content:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith("data: "):
                        data_content = line_str[6:]
                        if data_content == "[DONE]": break
                        try:
                            chunk = json.loads(data_content)
                            if "error" in chunk:
                                raise Exception(f"OpenAI API error in stream: {chunk['error']}")
                            if content := chunk.get("choices", [{}])[0].get("delta", {}).get("content"):
                                yield content
                        except (json.JSONDecodeError, IndexError) as e:
                            logger.warning(f"Could not parse OpenAI stream chunk: {e} - Content: '{data_content}'")
                            continue

    async def _stream_gemini(self, prompt, config, personality):
        final_prompt = f"{personality}\n\nUSER PROMPT:\n{prompt}" if personality else prompt
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.model}:streamGenerateContent?key={config.api_key}&alt=sse"
        payload = {"contents": [{"parts": [{"text": final_prompt}]}],
                   "generationConfig": {"temperature": config.temperature, "maxOutputTokens": config.max_tokens}}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Gemini stream API error {response.status}: {error_text}")
                async for line in response.content:
                    if line.startswith(b'data: '):
                        try:
                            data = json.loads(line[6:])
                            if "error" in data:
                                raise Exception(f"Gemini API error in stream: {data['error']}")
                            if content := data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get(
                                    "text"):
                                yield content
                        except (json.JSONDecodeError, KeyError, IndexError) as e:
                            logger.warning(
                                f"Could not parse Gemini stream chunk: {e} - Content: '{line.decode('utf-8', 'ignore')}'")
                            continue

    async def _stream_anthropic(self, prompt, config, personality):
        url = config.base_url or "https://api.anthropic.com/v1/messages"
        headers = {"x-api-key": config.api_key, "Content-Type": "application/json", "anthropic-version": "2023-06-01",
                   "Accept": "text/event-stream"}
        payload = {"model": config.model, "max_tokens": config.max_tokens, "temperature": config.temperature,
                   "messages": [{"role": "user", "content": prompt}], "stream": True}
        if personality: payload["system"] = personality

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Anthropic stream API error {response.status}: {error_text}")
                async for line in response.content:
                    if line.startswith(b"data: "):
                        try:
                            data = json.loads(line[6:])
                            if data.get("type") == "error":
                                raise Exception(f"Anthropic API error in stream: {data['error']}")
                            if data.get("type") == "content_block_delta":
                                yield data["delta"]["text"]
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.warning(
                                f"Could not parse Anthropic stream chunk: {e} - Content: '{line.decode('utf-8', 'ignore')}'")
                            continue

    async def _stream_deepseek(self, prompt: str, config: ModelConfig, personality: str = ""):
        messages = [{"role": "system", "content": personality}] if personality else []
        messages.append({"role": "user", "content": prompt})
        url = f"{config.base_url}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}
        payload = {"model": config.model, "messages": messages, "temperature": config.temperature,
                   "max_tokens": config.max_tokens, "stream": True}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"DeepSeek stream API error {response.status}: {error_text}")
                async for line in response.content:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith("data: "):
                        data_content = line_str[6:]
                        if data_content == "[DONE]": break
                        try:
                            chunk = json.loads(data_content)
                            if "error" in chunk:
                                raise Exception(f"DeepSeek API error in stream: {chunk['error']}")
                            if content := chunk.get("choices", [{}])[0].get("delta", {}).get("content"):
                                yield content
                        except (json.JSONDecodeError, IndexError) as e:
                            logger.warning(f"Could not parse Deepseek stream chunk: {e} - Content: '{data_content}'")
                            continue

    async def _stream_ollama(self, prompt, config, personality):
        final_prompt = f"{personality}\n\nUSER PROMPT:\n{prompt}" if personality else prompt
        url = f"{config.base_url}/api/generate"
        payload = {"model": config.model, "prompt": final_prompt, "stream": True,
                   "options": {"temperature": config.temperature, "num_predict": config.max_tokens}}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error {response.status}: {error_text}")
                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line)
                            if data.get("error"):
                                raise Exception(f"Ollama error: {data['error']}")
                            if content := data.get("response"):
                                yield content
                        except json.JSONDecodeError as e:
                            logger.warning(
                                f"Could not parse Ollama stream chunk: {e} - Content: '{line.decode('utf-8', 'ignore')}'")
                            continue

    def _fallback_response(self, prompt: str, role: LLMRole, error: Exception = None) -> str:
        error_msg = f"API call failed: {str(error)}" if error else "No model available."
        return f"# AvA Error: No {role.value} LLM available or {error_msg}\n# Request: {prompt[:100]}...\n# Check config and API keys."

    def get_role_assignments(self) -> Dict[str, str]:
        """Get current role assignments for display."""
        return {role_enum.value: model_name for role_enum, model_name in self.role_assignments.items()}