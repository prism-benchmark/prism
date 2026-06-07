"""
ai_config.py — Thin facade over llm_client.py for backward compatibility.

All configuration lives in llm_config.yaml. All logic lives in llm_client.py.
This module re-exports everything and provides legacy constants for code that
imports from ai_config directly.

Quick start (new code — prefer this):
    from llm_client import PRISMLLMClient, get_aspect_config, list_enabled

Legacy (still works):
    from ai_config import (
        GOOGLE_API_KEY, OPENAI_API_KEY, GEMINI_MODEL, GPT_MODEL,
        get_provider_config, get_llm_client, validate_env,
    )
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
    pass

# ── Re-export everything from llm_client ──────────────────────────────────
from llm_client import (  # noqa: F401
    # Config loaders
    load_llm_config,
    reset_llm_config_cache,
    get_aspect_config,
    get_reviewer_config,
    get_provider_credentials,
    get_provider_defaults,
    get_referenced_api_providers,
    # Enable/disable & profiles
    is_enabled,
    list_all,
    list_enabled,
    list_disabled,
    get_profile_names,
    get_profile_items,
    resolve_items,
    # Client
    PRISMLLMClient,
)


# ── Helpers ───────────────────────────────────────────────────────────────


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _as_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _provider_name(provider: str) -> str:
    from llm_client import _normalize_provider_name

    return _normalize_provider_name(provider)


def _provider_credentials(provider: str) -> Dict[str, Any]:
    try:
        return get_provider_credentials(_provider_name(provider))
    except Exception:
        return {}


def _provider_defaults(provider: str) -> Dict[str, Any]:
    try:
        return get_provider_defaults(_provider_name(provider))
    except Exception:
        return {}


def _legacy_provider_config(provider: str) -> Dict[str, Any]:
    normalized = _provider_name(provider)
    creds = _provider_credentials(normalized)
    defaults = _provider_defaults(normalized)
    base_url = creds.get("base_url") or creds.get("endpoint")
    default_model = (
        defaults.get("default_model")
        or creds.get("deployment")
        or load_llm_config().get("defaults", {}).get("model")
        or ""
    )
    return {
        "api_key": creds.get("api_key", ""),
        "base_url": base_url,
        "default_model": default_model,
        "default_temperature": _as_float(defaults.get("default_temperature"), 0.0),
        "default_max_tokens": _as_int(defaults.get("default_max_tokens"), 4096),
    }


# ============================================================================
# Legacy API Keys (module-level constants)
# ============================================================================

# Google / Gemini
GOOGLE_API_KEY: str = str(_provider_credentials("gemini").get("api_key") or "")

# OpenAI (native)
OPENAI_API_KEY: str = str(_provider_credentials("openai").get("api_key") or "")

# Azure OpenAI
_AZURE_CREDS = _provider_credentials("azure")
AZURE_OPENAI_API_KEY: str = str(_AZURE_CREDS.get("api_key") or "")
AZURE_OPENAI_ENDPOINT: str = str(
    _AZURE_CREDS.get("endpoint") or _AZURE_CREDS.get("base_url") or ""
)
AZURE_OPENAI_DEPLOYMENT: str = str(
    _AZURE_CREDS.get("deployment")
    or _provider_defaults("azure").get("default_model")
    or "gpt-4o"
)

# Xiaomi Mimo
_MIMO_CREDS = _provider_credentials("mimo")
MIMO_API_KEY: str = str(_MIMO_CREDS.get("api_key") or "")
MIMO_BASE_URL: str = str(_MIMO_CREDS.get("base_url") or "")

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
# Legacy Model Names / IDs
# ============================================================================

GEMINI_MODEL: str = str(
    _provider_defaults("gemini").get("default_model") or "gemini-2.5-flash-lite"
)
GPT_MODEL: str = str(_provider_defaults("openai").get("default_model") or "gpt-4o-mini")
MIMO_MODEL: str = str(
    _provider_defaults("mimo").get("default_model") or "mimo-v2.5-pro"
)

# Generic LLM provider / model (used by novelty verification)
try:
    _NOVELTY_CFG = get_aspect_config("novelty")
except Exception:
    _NOVELTY_CFG = {}
LLM_PROVIDER: str = str(
    _NOVELTY_CFG.get("provider") or os.getenv("LLM_PROVIDER", "openai")
)
LLM_MODEL_NAME: str = str(
    _NOVELTY_CFG.get("model")
    or _provider_defaults(LLM_PROVIDER).get("default_model")
    or "gpt-4o"
)
_LLM_CREDS = _provider_credentials(LLM_PROVIDER)
LLM_API_KEY: str = str(_NOVELTY_CFG.get("api_key") or _LLM_CREDS.get("api_key") or "")
LLM_API_ENDPOINT: str = str(
    _NOVELTY_CFG.get("base_url")
    or _LLM_CREDS.get("base_url")
    or _LLM_CREDS.get("endpoint")
    or ""
)



# ============================================================================
# Legacy Inference Defaults
# ============================================================================

GEMINI_TEMP: float = _as_float(
    _provider_defaults("gemini").get("default_temperature"), 0.0
)
GPT_TEMP: float = _as_float(
    _provider_defaults("openai").get("default_temperature"), 1.0
)
MIMO_TEMP: float = _as_float(_provider_defaults("mimo").get("default_temperature"), 0.0)

DEFAULT_MAX_OUTPUT_TOKENS: int = _as_int(
    load_llm_config().get("defaults", {}).get("max_tokens"), 4096
)
LLM_MAX_TOKENS: int = _as_int(_NOVELTY_CFG.get("max_tokens"), DEFAULT_MAX_OUTPUT_TOKENS)
LLM_PROVIDER_CAP: int = int(os.getenv("LLM_PROVIDER_CAP", "64000"))
EFFECTIVE_LLM_MAX_TOKENS: int = min(LLM_MAX_TOKENS, LLM_PROVIDER_CAP)

LLM_MAX_PROMPT_CHARS: int = int(os.getenv("LLM_MAX_PROMPT_CHARS", "250000"))
MAX_CONTEXT_CHARS: int = int(os.getenv("MAX_CONTEXT_CHARS", "200000"))

API_TIMEOUT: int = _as_int(
    _NOVELTY_CFG.get("timeout") or load_llm_config().get("defaults", {}).get("timeout"),
    120,
)
MAX_RETRIES: int = _as_int(
    _NOVELTY_CFG.get("max_retries")
    or load_llm_config().get("defaults", {}).get("max_retries"),
    3,
)
RETRY_DELAY: int = _as_int(
    _NOVELTY_CFG.get("retry_delay")
    or load_llm_config().get("defaults", {}).get("retry_delay"),
    1,
)

SEMANTIC_SCHOLAR_API_BASE: str = os.getenv(
    "SEMANTIC_SCHOLAR_API_BASE", "https://api.semanticscholar.org/graph/v1"
)
PHASE2_MAX_QUERY_ATTEMPTS: int = int(os.getenv("PHASE2_MAX_QUERY_ATTEMPTS", "8"))


# ============================================================================
# Unified LLM Client Factory
# ============================================================================


def get_llm_client(
    name: str,
    step: Optional[str] = None,
    **overrides,
):
    """Get a PRISMLLMClient configured for an aspect or reviewer.

    Args:
        name: Aspect name or reviewer name.
        step: Optional step name for multi-step aspects.
        **overrides: Override any config field.

    Returns:
        PRISMLLMClient instance.
    """
    aspects = (load_llm_config().get("aspects") or {}).keys()
    if name in aspects:
        return PRISMLLMClient.for_aspect(name, step=step, **overrides)
    else:
        return PRISMLLMClient.for_reviewer(name, **overrides)


def get_provider_config(provider: str = "gemini") -> Dict[str, Any]:
    """Return a config dict for a provider (legacy interface).

    Args:
        provider: one of 'gemini', 'openai', 'mimo',
                  'azure', 'openrouter', 'llm'

    Returns:
        dict with keys: api_key, base_url, default_model, default_temperature,
                        default_max_tokens
    """
    provider = provider.lower()
    if provider == "llm":
        return {
            "api_key": LLM_API_KEY,
            "base_url": LLM_API_ENDPOINT,
            "default_model": LLM_MODEL_NAME,
            "default_temperature": _as_float(_NOVELTY_CFG.get("temperature"), 0.0),
            "default_max_tokens": EFFECTIVE_LLM_MAX_TOKENS,
        }

    normalized = _provider_name(provider)
    if normalized not in {"gemini", "openai", "mimo", "azure", "openrouter"}:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Supported: gemini, openai, mimo, azure, openrouter, llm"
        )
    cfg = _legacy_provider_config(normalized)
    return cfg


def validate_env(
    require_gemini: Optional[bool] = None,
    require_mimo: Optional[bool] = None,
    require_openai: Optional[bool] = None,
    require_azure: Optional[bool] = None,
    require_data_root: bool = False,
) -> None:
    """Raise EnvironmentError if required API keys are missing."""
    provider_env_vars = {
        "gemini": ("GOOGLE_API_KEY", GOOGLE_API_KEY),
        "openai": ("OPENAI_API_KEY", OPENAI_API_KEY),
        "azure": ("AZURE_OPENAI_API_KEY", AZURE_OPENAI_API_KEY),
        "mimo": ("MIMO_API_KEY", MIMO_API_KEY),
        "openrouter": ("LLM_API_KEY", LLM_API_KEY),
    }

    required_providers = set(get_referenced_api_providers())
    if require_gemini:
        required_providers.add("gemini")
    if require_mimo:
        required_providers.add("mimo")
    if require_openai:
        required_providers.add("openai")
    if require_azure:
        required_providers.add("azure")

    errors = []
    if require_data_root and not DATA_ROOT:
        errors.append("DATA_ROOT is not set")
    for provider in sorted(required_providers):
        env_name, value = provider_env_vars.get(provider, (None, None))
        if env_name and not value:
            errors.append(f"{env_name} is not set for provider '{provider}'")
    if errors:
        raise EnvironmentError(
            "Missing required environment variables:\n"
            + "\n".join(f"  - {e}" for e in errors)
            + "\n\nPlease copy .env.example to .env and fill in the values."
        )
