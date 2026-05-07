"""
Configuration for the novelty assessment pipeline.

AI model settings are imported from the centralized ai_config.py at repo root.
Only pipeline-specific and Semantic Scholar settings are defined here.

Usage:
    from config import LLM_PROVIDER, LLM_API_KEY, LLM_MODEL_NAME, ...
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads .env from current working directory or project root

# ── Import AI model settings from centralized config ──────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ai_config import (
    # Generic LLM provider (OpenAI-compatible)
    LLM_PROVIDER,
    LLM_API_KEY,
    LLM_API_ENDPOINT,
    LLM_MODEL_NAME,
    # Token limits
    EFFECTIVE_LLM_MAX_TOKENS,
    LLM_MAX_TOKENS,
    LLM_PROVIDER_CAP,
    LLM_MAX_PROMPT_CHARS,
    MAX_CONTEXT_CHARS,
    # Retry / timeout
    API_TIMEOUT,
    MAX_RETRIES,
    RETRY_DELAY,
    # Semantic Scholar
    SEMANTIC_SCHOLAR_API_BASE,
    SEMANTIC_SCHOLAR_API_KEY,
    SEMANTIC_SCHOLAR_API_KEYS,
    PHASE2_MAX_QUERY_ATTEMPTS,
)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
