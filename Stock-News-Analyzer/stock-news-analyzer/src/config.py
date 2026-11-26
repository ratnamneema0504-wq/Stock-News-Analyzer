from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Load the project-root .env explicitly so it works regardless of the
# directory Streamlit is launched from. Never print or log values from it.
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "index"
CACHE_DIR = INDEX_DIR / "cache"
PROMPT_PATH = BASE_DIR / "prompts" / "system_prompt.txt"

DATA_FILE = DATA_DIR / os.getenv("DATA_FILE", "stock_market_news_extended_15000.csv")

# Logical names -> actual CSV columns.
COLUMN_MAP = {
    "headline": "news",
    "text": "body",
    "date": "date",
    "ticker": "company",
    "sector": "sector",
    "source": "source",
    "sentiment": "sentiment",
    "event_type": "topic",
    "open": None,
    "high": None,
    "low": None,
    "close": None,
    "volume": "volume",
    "price_before": "price_before",
    "price_after": "price_after",
    "price_change_percent": "price_change_percent",
    "price_direction": "price_direction",
    "impact": "impact",
}

# Columns the bundled dataset must contain.
EXPECTED_COLUMNS = [
    "news", "date", "year", "company", "sector", "source", "topic", "sentiment",
    "price_before", "price_after", "price_change_percent", "price_direction",
    "impact", "volume", "body",
]

TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", "8"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.30"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_MODEL = "llama3.2:3b"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
MAX_ARTICLE_CHARS = int(os.getenv("MAX_ARTICLE_CHARS", "6000"))

DISCLAIMER = "This is an analysis of the provided news data, not investment advice."
INSUFFICIENT_EVIDENCE = "The available evidence does not contain enough information to answer that."
AI_UNAVAILABLE = "AI analysis is currently unavailable. Please make sure Ollama is running and the configured local model is installed."


def get_model() -> str:
    return (os.getenv("OLLAMA_MODEL") or DEFAULT_MODEL).strip() or DEFAULT_MODEL


def ai_ready() -> bool:
    """Check whether the configured local Ollama model is reachable."""
    try:
        from .local_ai import is_available
        return is_available()
    except Exception:
        return False
