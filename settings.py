import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # LLM API keys and endpoints (loaded from .env)
    RAG_INDEX_PATH = os.getenv("RAG_INDEX_PATH", "./rag_index.db")
    CACHE_ENABLED = True
    # UI prefs
    DARK_MODE = True
    ACCENT_COLOR = "#0a84ff"  # electric blue