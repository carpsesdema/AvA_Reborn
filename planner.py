from core.llm_client import LLMClient
from utils.logger import get_logger

logger = get_logger(__name__)

class Planner:
    """
    Wraps a high-level planning LLM. Takes user prompt, returns micro-task breakdown documents.
    """
    def __init__(self, model_name: str = "planner-llm"):
        self.client = LLMClient(model_name)

    async def generate_plan(self, user_prompt: str) -> str:
        """Return a structured plan (YAML or JSON) listing micro tasks per file."""
        system_message = "You are a project planning assistant. Break down the requirements into atomic tasks per file."
        response = await self.client.chat(
            system=system_message,
            user=user_prompt,
        )
        plan_text = response.choices[0].message.content
        logger.info("Received plan from planner:%s", plan_text)
        return plan_text