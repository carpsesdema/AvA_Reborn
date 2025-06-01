import os
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)

class FileManager:
    """
    Handles safe file operations, sandboxing, and assembling code files.
    """
    def __init__(self, base_dir: str = '.'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def write_file(self, relative_path: str, content: str):
        filepath = self.base_dir / relative_path
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.debug("File written: %s", filepath)

    def read_file(self, relative_path: str) -> str:
        filepath = self.base_dir / relative_path
        if filepath.exists():
            return filepath.read_text(encoding='utf-8')
        else:
            logger.warning("Attempted to read missing file: %s", filepath)
            return ""