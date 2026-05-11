"""
env_loader.py — Thin re-export for backward compatibility.

All configuration lives in llm_config.yaml → llm_client.py → ai_config.py.
This module only adds conference path helpers specific to Aspects_benchmarking.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Import all AI config from centralized module ─────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from ai_config import (  # noqa: E402
    DATA_ROOT,
    GOOGLE_API_KEY,
    GEMINI_MODEL,
    MIMO_API_KEY,
    MIMO_MODEL,
    MIMO_BASE_URL,
    OPENAI_API_KEY,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT,
    validate_env as _ai_validate_env,
)

__all__ = [
    "DATA_ROOT", "conf_path", "paper_ids_file",
    "GOOGLE_API_KEY", "GEMINI_MODEL",
    "MIMO_API_KEY", "MIMO_MODEL", "MIMO_BASE_URL",
    "OPENAI_API_KEY",
    "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT",
    "validate_env",
]

# ---------------------------------------------------------------------------
# Conference path helpers
# ---------------------------------------------------------------------------

_CONF_DIRS = {
    "ICLR2024":    "ICLR2024",
    "ICLR2025":    "ICLR2025",
    "ICLR2026":    "ICLR2026",
    "ICML2025":    "ICML2025",
    "NeurIPS2025": "NeurIPS2025",
}


def conf_path(conference: str) -> str:
    """Return the absolute path to a conference data folder."""
    if not DATA_ROOT:
        raise EnvironmentError(
            "DATA_ROOT is not set. "
            "Copy .env.example to .env and set DATA_ROOT to your dataset path."
        )
    folder = _CONF_DIRS.get(conference, conference)
    return os.path.join(DATA_ROOT, folder)


def paper_ids_file(conference: str, subset: int = 50) -> str:
    """Return path to the paper_ids_{subset}_{conf_lower}.txt file."""
    base = conf_path(conference)
    if conference == "NeurIPS2025":
        fname = f"paper_ids_{subset}_neurips2025.txt"
    else:
        fname = f"paper_ids_{subset}_{conference.lower()}.txt"
    return os.path.join(base, fname)


def validate_env(require_gemini: bool | None = None, require_mimo: bool | None = None) -> None:
    """Raise EnvironmentError if required variables are missing."""
    _ai_validate_env(
        require_gemini=require_gemini,
        require_mimo=require_mimo,
        require_data_root=True,
    )
