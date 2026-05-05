"""
Configuration for the novelty assessment pipeline.

All values are loaded from environment variables (via .env file or shell).
Copy .env_example to .env and fill in your API keys.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads .env from current working directory or project root

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# LLM API Configuration (Task 1: extraction, Task 3: verification)
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_API_ENDPOINT = os.getenv("LLM_API_ENDPOINT", "https://api.openai.com/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o")

# Output token limits
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "64000"))
LLM_PROVIDER_CAP = int(os.getenv("LLM_PROVIDER_CAP", "64000"))
EFFECTIVE_LLM_MAX_TOKENS = min(LLM_MAX_TOKENS, LLM_PROVIDER_CAP)

# Prompt size guard
LLM_MAX_PROMPT_CHARS = int(os.getenv("LLM_MAX_PROMPT_CHARS", "250000"))

# Max context characters for long text snippets
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "200000"))

# ---------------------------------------------------------------------------
# Semantic Scholar API Configuration (Task 2: related-works retrieval)
# ---------------------------------------------------------------------------
SEMANTIC_SCHOLAR_API_BASE = os.getenv(
    "SEMANTIC_SCHOLAR_API_BASE", "https://api.semanticscholar.org/graph/v1"
)
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None

# Multiple API keys for load balancing (comma-separated).
_ss_keys_raw = os.getenv("SEMANTIC_SCHOLAR_API_KEYS", "")
SEMANTIC_SCHOLAR_API_KEYS: list[str] = [
    k.strip() for k in _ss_keys_raw.split(",") if k.strip()
]
if not SEMANTIC_SCHOLAR_API_KEYS and SEMANTIC_SCHOLAR_API_KEY:
    SEMANTIC_SCHOLAR_API_KEYS = [SEMANTIC_SCHOLAR_API_KEY]

# Per-query retry attempts for Semantic Scholar
PHASE2_MAX_QUERY_ATTEMPTS = int(os.getenv("PHASE2_MAX_QUERY_ATTEMPTS", "8"))

# ---------------------------------------------------------------------------
# Retry & Timeout
# ---------------------------------------------------------------------------
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "120"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "30"))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", "5"))
