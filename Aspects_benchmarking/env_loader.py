"""
env_loader.py
=============
Shared environment loader for the Aspects_benchmarking subsystem.

API keys and model settings are imported from the centralized
``ai_config.py`` at the repo root. Only DATA_ROOT and conference-path
helpers live here.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Import all AI config from centralized module ─────────────────────────
# ai_config.py loads .env from the repo root, so no need to load dotenv here.
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

# ── Re-export for backward compatibility ──────────────────────────────────
__all__ = [
    "DATA_ROOT", "conf_path", "paper_ids_file",
    "GOOGLE_API_KEY", "GEMINI_MODEL",
    "MIMO_API_KEY", "MIMO_MODEL", "MIMO_BASE_URL",
    "OPENAI_API_KEY",
    "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT",
    "validate_env",
]

# ---------------------------------------------------------------------------
# Core paths (DATA_ROOT is imported from ai_config above)
# ---------------------------------------------------------------------------

# Conference folder names (Neurlps2025 preserves original typo in dataset)
_CONF_DIRS = {
    "ICLR2024":    "ICLR2024",
    "ICLR2025":    "ICLR2025",
    "ICLR2026":    "ICLR2026",
    "ICML2025":    "ICML2025",
    "NeurIPS2025": "NeurIPS2025",   # note: dataset folder has this spelling
}


def conf_path(conference: str) -> str:
    """Return the absolute path to a conference data folder.

    Args:
        conference: one of ICLR2024 | ICLR2025 | ICLR2026 | ICML2025 | NeurIPS2025
    """
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
    # Handle NeurIPS special case
    if conference == "NeurIPS2025":
        fname = f"paper_ids_{subset}_neurips2025.txt"
    else:
        fname = f"paper_ids_{subset}_{conference.lower()}.txt"
    return os.path.join(base, fname)


# ---------------------------------------------------------------------------
# Backward-compatible validate_env (adds DATA_ROOT check)
# ---------------------------------------------------------------------------
def validate_env(require_gemini: bool = True, require_mimo: bool = False) -> None:
    """Raise EnvironmentError if required variables are missing."""
    errors = []
    if not DATA_ROOT:
        errors.append("DATA_ROOT is not set")
    # Delegate API key checks to centralized validator
    try:
        _ai_validate_env(
            require_gemini=require_gemini,
            require_mimo=require_mimo,
        )
    except EnvironmentError as e:
        errors.append(str(e))
    if errors:
        raise EnvironmentError(
            "Missing required environment variables:\n"
            + "\n".join(f"  • {e}" for e in errors)
            + "\n\nPlease copy .env.example to .env and fill in the values."
        )
