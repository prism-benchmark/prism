"""
ai_config.py — Centralized AI/LLM configuration for the entire PRISM project.

Single source of truth for:
  - API keys (Gemini, OpenAI, Azure OpenAI, Mimo, HuggingFace, Semantic Scholar)
  - Model names and IDs for each provider
  - Default inference parameters (temperature, max tokens, top_p)
  - Provider base URLs

Both Aspects_benchmarking and LLM_reviewer import from here.

Usage:
    from ai_config import (
        # API keys
        GOOGLE_API_KEY, OPENAI_API_KEY, AZURE_OPENAI_API_KEY,
        MIMO_API_KEY, HF_TOKEN, SEMANTIC_SCHOLAR_API_KEYS,
        # Model names
        GEMINI_MODEL, GPT_MODEL, MIMO_MODEL, AZURE_OPENAI_DEPLOYMENT,
        # Inference defaults
        GEMINI_TEMP, GPT_TEMP, MIMO_TEMP, DEFAULT_MAX_OUTPUT_TOKENS,
        # Base URLs
        MIMO_BASE_URL, AZURE_OPENAI_ENDPOINT,
        # Provider config dicts (for UnifiedChatClient / LLMClient)
        get_provider_config,
    )

To override any value: set the corresponding environment variable in .env
at the repo root. See .env.example for the full list.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

# ── Load .env from repo root ──────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent
try:
    from dotenv import load_dotenv
    load_dotenv(_REPO_ROOT / ".env", override=False)
except ImportError:
    pass  # dotenv optional; fall back to os.environ


# ============================================================================
# API Keys
# ============================================================================

# Google / Gemini
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

# OpenAI (native)
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# Azure OpenAI
AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

# Xiaomi Mimo
MIMO_API_KEY: str = os.getenv("MIMO_API_KEY", "")
MIMO_BASE_URL: str = os.getenv("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")

# HuggingFace
HF_TOKEN: str = os.getenv("HF_TOKEN", "")

# Data root (shared dataset path)
DATA_ROOT: str = os.getenv("DATA_ROOT", "")

# Semantic Scholar (single key + multi-key load balancing)
SEMANTIC_SCHOLAR_API_KEY: Optional[str] = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None
_ss_keys_raw = os.getenv("SEMANTIC_SCHOLAR_API_KEYS", "")
SEMANTIC_SCHOLAR_API_KEYS: list[str] = [
    k.strip() for k in _ss_keys_raw.split(",") if k.strip()
]
if not SEMANTIC_SCHOLAR_API_KEYS and SEMANTIC_SCHOLAR_API_KEY:
    SEMANTIC_SCHOLAR_API_KEYS = [SEMANTIC_SCHOLAR_API_KEY]


# ============================================================================
# Model Names / IDs
# ============================================================================

# Gemini evaluator model
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

# OpenAI GPT evaluator model
GPT_MODEL: str = os.getenv("GPT_MODEL", "gpt-4o-mini")

# Mimo evaluator model
MIMO_MODEL: str = os.getenv("MIMO_MODEL", "mimo-v2.5-pro")

# Generic LLM provider / model (used by novelty verification and generic clients)
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gpt-4o")
LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
LLM_API_ENDPOINT: str = os.getenv("LLM_API_ENDPOINT", "https://api.openai.com/v1")

# Bosch Devmate (Gemini proxy)
GEMINI_DEVMATE_MODEL: str = os.getenv("GEMINI_DEVMATE_MODEL", os.getenv("DEVMATE_MODEL", "gemini-3-flash-preview"))
DEVMATE_API_KEY: str = os.getenv("DEVMATE_API_KEY", "")
DEVMATE_BASE_URL: str = os.getenv("DEVMATE_BASE_URL", "https://devmate.bosch.com/api/v3")
DEVMATE_PROXY: str = os.getenv("DEVMATE_PROXY", "http://rb-proxy-apac.bosch.com:8080")
DEVMATE_DISABLE_SSL_VERIFY: bool = os.getenv("DEVMATE_DISABLE_SSL_VERIFY", "true").strip().lower() in {"1", "true", "yes", "on"}


# ============================================================================
# Inference Defaults
# ============================================================================

# Temperatures
GEMINI_TEMP: float = float(os.getenv("GEMINI_TEMP", "0.0"))
GPT_TEMP: float = float(os.getenv("GPT_TEMP", "1.0"))
MIMO_TEMP: float = float(os.getenv("MIMO_TEMP", "0.0"))

# Max output tokens
DEFAULT_MAX_OUTPUT_TOKENS: int = int(os.getenv("DEFAULT_MAX_OUTPUT_TOKENS", "4096"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "64000"))
LLM_PROVIDER_CAP: int = int(os.getenv("LLM_PROVIDER_CAP", "64000"))
EFFECTIVE_LLM_MAX_TOKENS: int = min(LLM_MAX_TOKENS, LLM_PROVIDER_CAP)

# Prompt size guard
LLM_MAX_PROMPT_CHARS: int = int(os.getenv("LLM_MAX_PROMPT_CHARS", "250000"))
MAX_CONTEXT_CHARS: int = int(os.getenv("MAX_CONTEXT_CHARS", "200000"))

# Retry / timeout
API_TIMEOUT: int = int(os.getenv("API_TIMEOUT", "120"))
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "30"))
RETRY_DELAY: int = int(os.getenv("RETRY_DELAY", "5"))
SEMANTIC_SCHOLAR_API_BASE: str = os.getenv(
    "SEMANTIC_SCHOLAR_API_BASE", "https://api.semanticscholar.org/graph/v1"
)
PHASE2_MAX_QUERY_ATTEMPTS: int = int(os.getenv("PHASE2_MAX_QUERY_ATTEMPTS", "8"))


# ============================================================================
# Provider Config Helpers
# ============================================================================

def get_provider_config(provider: str = "gemini") -> Dict[str, Any]:
    """Return a config dict suitable for BaseClient.from_config() or UnifiedChatClient.

    Args:
        provider: one of 'gemini', 'openai', 'mimo', 'gemini-devmate',
                  'azure', 'llm' (generic OpenAI-compatible)

    Returns:
        dict with keys: api_key, base_url, default_model, default_temperature,
                        default_max_tokens
    """
    provider = provider.lower()

    if provider == "gemini":
        return {
            "api_key": GOOGLE_API_KEY,
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "default_model": GEMINI_MODEL,
            "default_temperature": GEMINI_TEMP,
            "default_max_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        }
    elif provider == "openai":
        return {
            "api_key": OPENAI_API_KEY,
            "base_url": None,  # OpenAI default
            "default_model": GPT_MODEL,
            "default_temperature": GPT_TEMP,
            "default_max_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        }
    elif provider == "mimo":
        return {
            "api_key": MIMO_API_KEY,
            "base_url": MIMO_BASE_URL,
            "default_model": MIMO_MODEL,
            "default_temperature": MIMO_TEMP,
            "default_max_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        }
    elif provider == "gemini-devmate":
        key = DEVMATE_API_KEY or GOOGLE_API_KEY
        return {
            "api_key": key,
            "base_url": DEVMATE_BASE_URL,
            "default_model": GEMINI_DEVMATE_MODEL,
            "default_temperature": GEMINI_TEMP,
            "default_max_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        }
    elif provider == "azure":
        return {
            "api_key": AZURE_OPENAI_API_KEY,
            "base_url": AZURE_OPENAI_ENDPOINT,
            "default_model": AZURE_OPENAI_DEPLOYMENT,
            "default_temperature": GPT_TEMP,
            "default_max_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        }
    elif provider == "llm":
        return {
            "api_key": LLM_API_KEY,
            "base_url": LLM_API_ENDPOINT,
            "default_model": LLM_MODEL_NAME,
            "default_temperature": 0.0,
            "default_max_tokens": EFFECTIVE_LLM_MAX_TOKENS,
        }
    else:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Supported: gemini, openai, mimo, gemini-devmate, azure, llm"
        )


def validate_env(
    require_gemini: bool = True,
    require_mimo: bool = False,
    require_openai: bool = False,
    require_azure: bool = False,
) -> None:
    """Raise EnvironmentError if required API keys are missing."""
    errors = []
    if require_gemini and not GOOGLE_API_KEY:
        errors.append("GOOGLE_API_KEY is not set")
    if require_mimo and not MIMO_API_KEY:
        errors.append("MIMO_API_KEY is not set")
    if require_openai and not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is not set")
    if require_azure and not AZURE_OPENAI_API_KEY:
        errors.append("AZURE_OPENAI_API_KEY is not set")
    if errors:
        raise EnvironmentError(
            "Missing required environment variables:\n"
            + "\n".join(f"  • {e}" for e in errors)
            + "\n\nPlease copy .env.example to .env and fill in the values."
        )
