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
                "chat_models": [ "gemini-2.5-pro-preview-06-05"],
                "code_models": ["gemini-2.5-pro-preview-06-05"]
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