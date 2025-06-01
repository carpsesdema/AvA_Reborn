import asyncio
import json
from utils.llm_client import LLMClient
from utils.file_manager import FileManager
from utils.logger import get_logger

logger = get_logger(__name__)

class Assembler:
    """
    Handles atomic micro tasks: calls code-generation LLM for each snippet, assembles into files.
    """
    def __init__(self, model_name: str = "codellama-13b"):
        self.client = LLMClient(model_name)
        self.file_manager = FileManager()

    async def process_tasks(self, tasks_json: str):
        """Given a JSON string of micro tasks, call LLM to generate each snippet and write files."""
        tasks = json.loads(tasks_json)
        for file_info in tasks["files"]:
            file_path = file_info["path"]
            snippets = file_info.get("snippets", [])
            assembled_code = """"  # start empty
            for snippet in snippets:
                prompt = f"// TASK: {snippet['description']}
// CONTEXT: Previously assembled code: {assembled_code}"
                response = await self.client.chat(system="Code generation assistant.", user=prompt)
                code_snippet = response.choices[0].message.content
                assembled_code += code_snippet + "\n"
            # Safely write assembled_code to disk
            self.file_manager.write_file(file_path, assembled_code)
            logger.info("Wrote file: %s", file_path)