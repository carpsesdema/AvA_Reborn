import asyncio
from some_llm_sdk import AsyncLLM  # placeholder import
from utils.logger import get_logger

logger = get_logger(__name__)

class LLMClient:
    """
    Async wrapper for any supported LLM backend (Codellama, Qwen2.5, etc.).
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = AsyncLLM(model_name=model_name)

    async def chat(self, system: str, user: str):
        try:
            response = await self.client.chat(
                system_prompt=system,
                user_prompt=user,
                stream=True,
            )
            # For simplicity, accumulate full content (expand for streaming)
            full_text = ""
            async for chunk in response:
                full_text += chunk
                # Optionally update StreamingTerminal here
            return type("Resp", (), {"choices": [{"message": type("Msg", (), {"content": full_text})}]})
        except Exception as e:
            logger.error("LLM chat error: %s", e)
            raise