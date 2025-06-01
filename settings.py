import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # LLM API keys and endpoints (loaded from .env)
    PLANNER_LLM = os.getenv("PLANNER_LLM_MODEL", "planner-llm")
    CODER_LLM = os.getenv("CODER_LLM_MODEL", "codellama-13b")
    RAG_INDEX_PATH = os.getenv("RAG_INDEX_PATH", "./rag_index.db")
    CACHE_ENABLED = True
    # UI prefs
    DARK_MODE = True
    ACCENT_COLOR = "#0a84ff"  # electric blue