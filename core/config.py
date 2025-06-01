# core/config.py - Professional Configuration Management

import os
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from dotenv import load_dotenv


@dataclass
class LLMConfig:
    """LLM configuration with validation"""
    provider: str
    model: str
    api_key: str
    api_base: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4000
    timeout: int = 60


@dataclass
class AppConfig:
    """Main application configuration"""
    theme: str = "dark"
    auto_save: bool = True
    log_level: str = "INFO"
    rag_enabled: bool = True
    workspace_path: str = "./workspace"


class ConfigManager:
    """Professional configuration manager with validation and persistence"""

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path.home() / ".ava"
        self.config_file = self.config_dir / "config.json"
        self.secrets_file = self.config_dir / "secrets.json"

        # Configuration data
        self.llm_configs: Dict[str, LLMConfig] = {}
        self.app_config = AppConfig()

        # Ensure config directory exists
        self.config_dir.mkdir(exist_ok=True)

    async def load(self):
        """Load configuration from files and environment"""

        # Load .env file if it exists
        env_file = Path(".env")
        if env_file.exists():
            load_dotenv(env_file)

        # Load main config
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
                self.app_config = AppConfig(**config_data.get("app", {}))

        # Load LLM configurations
        await self._load_llm_configs()

        # Validate configuration
        self._validate_config()

    async def _load_llm_configs(self):
        """Load LLM configurations from environment and files"""

        # Define supported LLM providers
        providers = {
            "openai": {
                "chat_models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
                "code_models": ["gpt-4o", "gpt-4-turbo"]
            },
            "anthropic": {
                "chat_models": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229"],
                "code_models": ["claude-3-5-sonnet-20241022"]
            },
            "google": {
                "chat_models": ["gemini-2.0-flash-exp", "gemini-1.5-pro"],
                "code_models": ["gemini-2.0-flash-exp"]
            },
            "ollama": {
                "chat_models": ["llama3.2", "qwen2.5"],
                "code_models": ["qwen2.5-coder", "codellama"]
            }
        }

        # Load configurations from environment
        for provider, models in providers.items():
            api_key_var = f"{provider.upper()}_API_KEY"
            api_key = os.getenv(api_key_var)

            if api_key or provider == "ollama":  # Ollama doesn't need API key
                # Create configs for each model type
                for model_type, model_list in models.items():
                    for model in model_list:
                        config_key = f"{provider}_{model}"

                        self.llm_configs[config_key] = LLMConfig(
                            provider=provider,
                            model=model,
                            api_key=api_key or "",
                            api_base=self._get_api_base(provider),
                            temperature=0.7 if model_type == "chat" else 0.1,
                            max_tokens=4000,
                            timeout=60
                        )

    def _get_api_base(self, provider: str) -> Optional[str]:
        """Get API base URL for provider"""

        api_bases = {
            "openai": os.getenv("OPENAI_API_BASE"),
            "anthropic": os.getenv("ANTHROPIC_API_BASE"),
            "google": os.getenv("GOOGLE_API_BASE"),
            "ollama": os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
        }

        return api_bases.get(provider)

    def _validate_config(self):
        """Validate configuration and log warnings for missing items"""

        import logging
        logger = logging.getLogger(__name__)

        # Check for at least one working LLM config
        if not self.llm_configs:
            logger.warning("No LLM configurations found. Add API keys to .env file.")

        # Check workspace path
        workspace = Path(self.app_config.workspace_path)
        workspace.mkdir(parents=True, exist_ok=True)

    async def save(self):
        """Save configuration to files"""

        # Save app config
        config_data = {
            "app": asdict(self.app_config),
            "version": "2.0.0"
        }

        with open(self.config_file, 'w') as f:
            json.dump(config_data, f, indent=2)

    def get_llm_config(self, model_key: str) -> Optional[LLMConfig]:
        """Get LLM configuration by key"""
        return self.llm_configs.get(model_key)

    def get_available_models(self, model_type: str = "chat") -> Dict[str, str]:
        """Get available models for a specific type"""

        models = {}
        for key, config in self.llm_configs.items():
            if model_type == "chat" and config.temperature > 0.5:
                models[key] = f"{config.provider}/{config.model}"
            elif model_type == "code" and config.temperature <= 0.5:
                models[key] = f"{config.provider}/{config.model}"

        return models

    def get_workflow_config(self) -> Dict[str, Any]:
        """Get configuration for workflow execution"""

        return {
            "workspace_path": self.app_config.workspace_path,
            "auto_save": self.app_config.auto_save,
            "rag_enabled": self.app_config.rag_enabled
        }


# core/llm_manager.py - Professional LLM Manager

import asyncio
import logging
from typing import Dict, Any, Optional, AsyncGenerator
from datetime import datetime, timedelta

from PySide6.QtCore import QObject, Signal

# LLM SDK imports
try:
    from openai import AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from anthropic import AsyncAnthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai

    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

try:
    import httpx

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class LLMProvider:
    """Base class for LLM providers"""

    def __init__(self, config):
        self.config = config
        self.client = None
        self.last_request_time = None
        self.request_count = 0

    async def initialize(self):
        """Initialize the provider"""
        raise NotImplementedError

    async def chat(self, messages: list, **kwargs) -> str:
        """Send chat request"""
        raise NotImplementedError

    async def stream_chat(self, messages: list, **kwargs) -> AsyncGenerator[str, None]:
        """Stream chat response"""
        raise NotImplementedError

    def is_available(self) -> bool:
        """Check if provider is available"""
        return self.client is not None


class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation"""

    async def initialize(self):
        if not OPENAI_AVAILABLE:
            raise RuntimeError("OpenAI SDK not available")

        self.client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.api_base,
            timeout=self.config.timeout
        )

    async def chat(self, messages: list, **kwargs) -> str:
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **kwargs
        )
        self.request_count += 1
        return response.choices[0].message.content

    async def stream_chat(self, messages: list, **kwargs) -> AsyncGenerator[str, None]:
        async with self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True,
                **kwargs
        ) as response:
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content


class AnthropicProvider(LLMProvider):
    """Anthropic provider implementation"""

    async def initialize(self):
        if not ANTHROPIC_AVAILABLE:
            raise RuntimeError("Anthropic SDK not available")

        self.client = AsyncAnthropic(
            api_key=self.config.api_key,
            base_url=self.config.api_base,
            timeout=self.config.timeout
        )

    async def chat(self, messages: list, **kwargs) -> str:
        # Convert messages format for Anthropic
        system_message = None
        user_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                user_messages.append(msg)

        response = await self.client.messages.create(
            model=self.config.model,
            system=system_message,
            messages=user_messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **kwargs
        )
        self.request_count += 1
        return response.content[0].text


class GoogleProvider(LLMProvider):
    """Google Gemini provider implementation"""

    async def initialize(self):
        if not GOOGLE_AVAILABLE:
            raise RuntimeError("Google AI SDK not available")

        genai.configure(api_key=self.config.api_key)
        self.client = genai.GenerativeModel(self.config.model)

    async def chat(self, messages: list, **kwargs) -> str:
        # Convert messages to Gemini format
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

        response = await self.client.generate_content_async(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens
            )
        )
        self.request_count += 1
        return response.text


class OllamaProvider(LLMProvider):
    """Ollama local provider implementation"""

    async def initialize(self):
        if not OLLAMA_AVAILABLE:
            raise RuntimeError("httpx not available for Ollama")

        self.client = httpx.AsyncClient(
            base_url=self.config.api_base,
            timeout=self.config.timeout
        )

    async def chat(self, messages: list, **kwargs) -> str:
        response = await self.client.post("/api/chat", json={
            "model": self.config.model,
            "messages": messages,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens
            }
        })
        response.raise_for_status()
        result = response.json()
        self.request_count += 1
        return result["message"]["content"]


class LLMManager(QObject):
    """Professional LLM Manager with multiple provider support"""

    status_changed = Signal(str, str, bool)  # (model, status, available)

    def __init__(self, config_manager):
        super().__init__()
        self.config_manager = config_manager
        self.providers: Dict[str, LLMProvider] = {}
        self.logger = logging.getLogger(__name__)

        # Provider classes mapping
        self.provider_classes = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "google": GoogleProvider,
            "ollama": OllamaProvider
        }

    async def initialize(self):
        """Initialize all available LLM providers"""

        self.logger.info("Initializing LLM Manager")

        for model_key, llm_config in self.config_manager.llm_configs.items():
            try:
                provider_class = self.provider_classes.get(llm_config.provider)
                if not provider_class:
                    self.logger.warning(f"Unknown provider: {llm_config.provider}")
                    continue

                provider = provider_class(llm_config)
                await provider.initialize()

                self.providers[model_key] = provider
                self.status_changed.emit(model_key, "ready", True)
                self.logger.info(f"Initialized {model_key}")

            except Exception as e:
                self.logger.warning(f"Failed to initialize {model_key}: {e}")
                self.status_changed.emit(model_key, f"error: {e}", False)

    async def chat(self, model_key: str, messages: list, **kwargs) -> str:
        """Send chat request to specific model"""

        provider = self.providers.get(model_key)
        if not provider:
            raise ValueError(f"Model {model_key} not available")

        try:
            self.status_changed.emit(model_key, "working", True)
            result = await provider.chat(messages, **kwargs)
            self.status_changed.emit(model_key, "ready", True)
            return result

        except Exception as e:
            self.logger.error(f"Chat request failed for {model_key}: {e}")
            self.status_changed.emit(model_key, f"error: {e}", False)
            raise

    async def stream_chat(self, model_key: str, messages: list, **kwargs) -> AsyncGenerator[str, None]:
        """Stream chat response from specific model"""

        provider = self.providers.get(model_key)
        if not provider:
            raise ValueError(f"Model {model_key} not available")

        try:
            self.status_changed.emit(model_key, "streaming", True)

            if hasattr(provider, 'stream_chat'):
                async for chunk in provider.stream_chat(messages, **kwargs):
                    yield chunk
            else:
                # Fallback: return full response as single chunk
                result = await provider.chat(messages, **kwargs)
                yield result

            self.status_changed.emit(model_key, "ready", True)

        except Exception as e:
            self.logger.error(f"Stream chat failed for {model_key}: {e}")
            self.status_changed.emit(model_key, f"error: {e}", False)
            raise

    def is_ready(self) -> bool:
        """Check if at least one provider is ready"""
        return any(p.is_available() for p in self.providers.values())

    def get_available_models(self) -> Dict[str, bool]:
        """Get availability status of all models"""
        return {key: provider.is_available() for key, provider in self.providers.items()}

    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about all models"""

        info = {}
        for key, provider in self.providers.items():
            info[key] = {
                "provider": provider.config.provider,
                "model": provider.config.model,
                "available": provider.is_available(),
                "request_count": provider.request_count,
                "last_request": provider.last_request_time
            }

        return info

    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status information"""

        return {
            "total_providers": len(self.providers),
            "available_providers": len([p for p in self.providers.values() if p.is_available()]),
            "models": await self.get_model_info(),
            "ready": self.is_ready()
        }

    async def shutdown(self):
        """Shutdown all providers"""

        self.logger.info("Shutting down LLM Manager")

        for provider in self.providers.values():
            if hasattr(provider, 'client') and hasattr(provider.client, 'close'):
                try:
                    await provider.client.close()
                except Exception as e:
                    self.logger.warning(f"Error closing provider: {e}")

        self.providers.clear()