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
    """

    def __init__(self):
        self.models: Dict[str, ModelConfig] = {}
        self.role_assignments: Dict[LLMRole, str] = {}
        self._initialize_models()
        self._assign_roles()

    def _initialize_models(self):
        """Initialize available models with their configurations"""

        # High-reasoning models (expensive but smart)
        if os.getenv("GEMINI_API_KEY"):
            self.models["gemini-flash"] = ModelConfig(
                provider="gemini",
                model="gemini-1.5-flash",
                api_key=os.getenv("GEMINI_API_KEY"),
                temperature=0.1,
                max_tokens=8000,
                suitable_roles=[LLMRole.PLANNER, LLMRole.REVIEWER, LLMRole.CHAT]
            )

            self.models["gemini-pro"] = ModelConfig(
                provider="gemini",
                model="gemini-1.5-pro",
                api_key=os.getenv("GEMINI_API_KEY"),
                temperature=0.1,
                max_tokens=8000,
                suitable_roles=[LLMRole.PLANNER, LLMRole.REVIEWER, LLMRole.CHAT]
            )

        if os.getenv("ANTHROPIC_API_KEY"):
            self.models["claude-sonnet"] = ModelConfig(
                provider="anthropic",
                model="claude-3-5-sonnet-20241022",
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
                max_tokens=4000,
                suitable_roles=[LLMRole.PLANNER, LLMRole.REVIEWER, LLMRole.CHAT]
            )

            self.models["gpt-4o-mini"] = ModelConfig(
                provider="openai",
                model="gpt-4o-mini",
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.1,
                max_tokens=4000,
                suitable_roles=[LLMRole.CODER, LLMRole.ASSEMBLER, LLMRole.CHAT]
            )

        # Specialized code models (cheaper, optimized for code)
        try:
            # Check if Ollama is available
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get("models", [])

                # Add available Ollama models
                for model in models:
                    model_name = model["name"]
                    if "coder" in model_name.lower() or "code" in model_name.lower():
                        self.models[f"ollama-{model_name}"] = ModelConfig(
                            provider="ollama",
                            model=model_name,
                            base_url="http://localhost:11434",
                            temperature=0.05,  # Lower temperature for code
                            max_tokens=4000,
                            suitable_roles=[LLMRole.CODER, LLMRole.ASSEMBLER]
                        )
                    elif "qwen" in model_name.lower():
                        self.models[f"ollama-{model_name}"] = ModelConfig(
                            provider="ollama",
                            model=model_name,
                            base_url="http://localhost:11434",
                            temperature=0.1,
                            max_tokens=4000,
                            suitable_roles=[LLMRole.PLANNER, LLMRole.CODER, LLMRole.ASSEMBLER, LLMRole.CHAT]
                        )
        except:
            pass  # Ollama not available

        # DeepSeek models (very cost-effective for code)
        if os.getenv("DEEPSEEK_API_KEY"):
            self.models["deepseek-coder"] = ModelConfig(
                provider="deepseek",
                model="deepseek-coder",
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com/v1",
                temperature=0.05,
                max_tokens=4000,
                suitable_roles=[LLMRole.CODER, LLMRole.ASSEMBLER]
            )

    def _assign_roles(self):
        """Assign best available models to each role"""

        # Priority order for each role
        role_preferences = {
            LLMRole.PLANNER: [
                "gemini-pro", "claude-sonnet", "gpt-4o",
                "gemini-flash", "ollama-qwen2.5:32b", "gpt-4o-mini"
            ],
            LLMRole.CODER: [
                "ollama-qwen2.5-coder:32b", "ollama-qwen2.5-coder:14b",
                "deepseek-coder", "ollama-codellama:13b",
                "gpt-4o-mini", "gemini-flash", "claude-sonnet"
            ],
            LLMRole.ASSEMBLER: [
                "ollama-qwen2.5-coder:14b", "deepseek-coder",
                "gpt-4o-mini", "gemini-flash", "claude-sonnet"
            ],
            LLMRole.REVIEWER: [
                "claude-sonnet", "gpt-4o", "gemini-pro",
                "gemini-flash", "gpt-4o-mini"
            ],
            LLMRole.CHAT: [
                "gemini-flash", "gpt-4o-mini", "claude-sonnet",
                "ollama-qwen2.5:14b", "gpt-4o"
            ]
        }

        # Assign first available model for each role
        for role, preferences in role_preferences.items():
            for model_name in preferences:
                if model_name in self.models:
                    config = self.models[model_name]
                    if role in config.suitable_roles:
                        self.role_assignments[role] = model_name
                        break

        # Fallback to any available model if role not assigned
        available_models = list(self.models.keys())
        for role in LLMRole:
            if role not in self.role_assignments and available_models:
                self.role_assignments[role] = available_models[0]

    def get_role_model(self, role: LLMRole) -> Optional[ModelConfig]:
        """Get the assigned model for a specific role"""
        model_name = self.role_assignments.get(role)
        if model_name:
            return self.models.get(model_name)
        return None

    def get_role_assignments(self) -> Dict[str, str]:
        """Get current role assignments for display"""
        return {role.value: model_name for role, model_name in self.role_assignments.items()}

    async def chat(self, prompt: str, role: LLMRole = LLMRole.CHAT) -> str:
        """
        Send prompt to appropriate LLM based on role
        """
        model_config = self.get_role_model(role)
        if not model_config:
            return self._fallback_response(prompt, role)

        try:
            if model_config.provider == "gemini":
                return await self._call_gemini(prompt, model_config)
            elif model_config.provider == "openai":
                return await self._call_openai(prompt, model_config)
            elif model_config.provider == "anthropic":
                return await self._call_anthropic(prompt, model_config)
            elif model_config.provider == "ollama":
                return await self._call_ollama(prompt, model_config)
            elif model_config.provider == "deepseek":
                return await self._call_deepseek(prompt, model_config)
        except Exception as e:
            print(f"Model {model_config.model} failed: {e}")
            # Try fallback to a different model
            return await self._try_fallback_models(prompt, role)

        return self._fallback_response(prompt, role)

    async def stream_chat(self, prompt: str, role: LLMRole = LLMRole.CHAT) -> AsyncGenerator[str, None]:
        """
        Stream response from appropriate LLM based on role
        """
        model_config = self.get_role_model(role)
        if not model_config:
            yield self._fallback_response(prompt, role)
            return

        try:
            if model_config.provider == "gemini":
                async for chunk in self._stream_gemini(prompt, model_config):
                    yield chunk
                return
            elif model_config.provider == "openai":
                async for chunk in self._stream_openai(prompt, model_config):
                    yield chunk
                return
            elif model_config.provider == "anthropic":
                async for chunk in self._stream_anthropic(prompt, model_config):
                    yield chunk
                return
            elif model_config.provider == "ollama":
                async for chunk in self._stream_ollama(prompt, model_config):
                    yield chunk
                return
            elif model_config.provider == "deepseek":
                async for chunk in self._stream_deepseek(prompt, model_config):
                    yield chunk
                return
        except Exception as e:
            print(f"Streaming from {model_config.model} failed: {e}")

        # Fallback
        yield self._fallback_response(prompt, role)

    async def _try_fallback_models(self, prompt: str, role: LLMRole) -> str:
        """Try other available models as fallback"""
        for model_name, config in self.models.items():
            if model_name != self.role_assignments.get(role):
                try:
                    if config.provider == "gemini":
                        return await self._call_gemini(prompt, config)
                    elif config.provider == "openai":
                        return await self._call_openai(prompt, config)
                    # Add other providers as needed
                except:
                    continue
        return self._fallback_response(prompt, role)

    async def _call_gemini(self, prompt: str, config: ModelConfig) -> str:
        """Call Gemini API"""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{config.model}:generateContent?key={config.api_key}",
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": config.temperature,
                            "maxOutputTokens": config.max_tokens
                        }
                    }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if ("candidates" in result and len(result["candidates"]) > 0 and
                            "content" in result["candidates"][0] and
                            "parts" in result["candidates"][0]["content"] and
                            len(result["candidates"][0]["content"]["parts"]) > 0):
                        return result["candidates"][0]["content"]["parts"][0]["text"]
                raise Exception(f"Gemini API error: {response.status}")

    async def _stream_gemini(self, prompt: str, config: ModelConfig) -> AsyncGenerator[str, None]:
        """Stream from Gemini (simulated)"""
        response = await self._call_gemini(prompt, config)
        if response:
            words = response.split()
            for i in range(0, len(words), 3):
                chunk = " ".join(words[i:i + 3]) + " "
                yield chunk
                await asyncio.sleep(0.05)

    async def _call_openai(self, prompt: str, config: ModelConfig) -> str:
        """Call OpenAI API"""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {config.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": config.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": config.temperature,
                        "max_tokens": config.max_tokens
                    }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                raise Exception(f"OpenAI API error: {response.status}")

    async def _stream_openai(self, prompt: str, config: ModelConfig) -> AsyncGenerator[str, None]:
        """Stream from OpenAI"""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {config.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": config.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": config.temperature,
                        "max_tokens": config.max_tokens,
                        "stream": True
                    }
            ) as response:
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk_data = json.loads(data)
                            if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                                delta = chunk_data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    yield delta["content"]
                        except json.JSONDecodeError:
                            continue

    async def _call_anthropic(self, prompt: str, config: ModelConfig) -> str:
        """Call Anthropic API"""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": config.api_key,
                        "Content-Type": "application/json",
                        "anthropic-version": "2023-06-01"
                    },
                    json={
                        "model": config.model,
                        "max_tokens": config.max_tokens,
                        "temperature": config.temperature,
                        "messages": [{"role": "user", "content": prompt}]
                    }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["content"][0]["text"]
                raise Exception(f"Anthropic API error: {response.status}")

    async def _stream_anthropic(self, prompt: str, config: ModelConfig) -> AsyncGenerator[str, None]:
        """Stream from Anthropic (simulated for now)"""
        response = await self._call_anthropic(prompt, config)
        if response:
            words = response.split()
            for i in range(0, len(words), 4):
                chunk = " ".join(words[i:i + 4]) + " "
                yield chunk
                await asyncio.sleep(0.03)

    async def _call_ollama(self, prompt: str, config: ModelConfig) -> str:
        """Call Ollama API"""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    f"{config.base_url}/api/generate",
                    json={
                        "model": config.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": config.temperature,
                            "num_predict": config.max_tokens
                        }
                    }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["response"]
                raise Exception(f"Ollama API error: {response.status}")

    async def _stream_ollama(self, prompt: str, config: ModelConfig) -> AsyncGenerator[str, None]:
        """Stream from Ollama"""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    f"{config.base_url}/api/generate",
                    json={
                        "model": config.model,
                        "prompt": prompt,
                        "stream": True,
                        "options": {
                            "temperature": config.temperature,
                            "num_predict": config.max_tokens
                        }
                    }
            ) as response:
                async for line in response.content:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if "response" in data:
                            yield data["response"]
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

    async def _call_deepseek(self, prompt: str, config: ModelConfig) -> str:
        """Call DeepSeek API"""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    f"{config.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {config.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": config.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": config.temperature,
                        "max_tokens": config.max_tokens
                    }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                raise Exception(f"DeepSeek API error: {response.status}")

    async def _stream_deepseek(self, prompt: str, config: ModelConfig) -> AsyncGenerator[str, None]:
        """Stream from DeepSeek"""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    f"{config.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {config.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": config.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": config.temperature,
                        "max_tokens": config.max_tokens,
                        "stream": True
                    }
            ) as response:
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk_data = json.loads(data)
                            if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                                delta = chunk_data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    yield delta["content"]
                        except json.JSONDecodeError:
                            continue

    def _fallback_response(self, prompt: str, role: LLMRole) -> str:
        """Fallback response when no LLM is available"""
        return f"""
# AvA Error: No {role.value} LLM available

# Request: {prompt[:100]}...

# To fix this, configure API keys for supported models:
# - Gemini: export GEMINI_API_KEY=your-key
# - OpenAI: export OPENAI_API_KEY=sk-your-key  
# - Anthropic: export ANTHROPIC_API_KEY=sk-ant-your-key
# - DeepSeek: export DEEPSEEK_API_KEY=sk-your-key

# Or install Ollama with code models:
# curl -fsSL https://ollama.ai/install.sh | sh
# ollama pull qwen2.5-coder:14b
# ollama pull codellama:13b

# Sample placeholder for your request:
def placeholder_function():
    '''Generated placeholder for: {prompt[:50]}...'''
    print("Replace this with actual implementation once LLM is configured")
    pass
"""

    def get_available_models(self) -> list:
        """Get list of available models and their roles"""
        available = []
        for role, model_name in self.role_assignments.items():
            if model_name in self.models:
                config = self.models[model_name]
                available.append(f"{role.value.title()}: {config.provider}/{config.model}")

        return available if available else ["No LLM services available"]

    def get_cost_estimate(self, role: LLMRole, token_count: int) -> str:
        """Estimate cost for using specific role (rough estimates)"""
        model_config = self.get_role_model(role)
        if not model_config:
            return "Unknown"

        # Rough cost estimates per 1K tokens (as of 2024)
        cost_per_1k = {
            "gemini-1.5-flash": 0.00015,
            "gemini-1.5-pro": 0.00125,
            "gpt-4o": 0.005,
            "gpt-4o-mini": 0.00015,
            "claude-3-5-sonnet": 0.003,
            "deepseek-coder": 0.00014,
            # Ollama models are free/local
        }

        cost = cost_per_1k.get(model_config.model, 0.0)
        if cost == 0.0:
            return "Free (Local)"

        estimated_cost = (token_count / 1000) * cost
        return f"${estimated_cost:.4f}"


# For backward compatibility
class LLMClient(EnhancedLLMClient):
    """Backward compatible wrapper"""

    def __init__(self):
        super().__init__()

    def chat(self, prompt: str) -> str:
        """Synchronous chat for backward compatibility"""
        # This would need to be updated to handle async properly
        # For now, it's a placeholder
        return asyncio.run(super().chat(prompt, LLMRole.CHAT))

    async def stream_chat(self, prompt: str) -> AsyncGenerator[str, None]:
        """Stream chat for backward compatibility"""
        async for chunk in super().stream_chat(prompt, LLMRole.CHAT):
            yield chunk