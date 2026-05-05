"""
env_loader.py
=============
Shared environment loader for all three evaluation aspects.
Reads from .env file at the repository root.

Usage in any script:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from env_loader import DATA_ROOT, GOOGLE_API_KEY, MIMO_API_KEY, conf_path
"""
from __future__ import annotations

import os
from pathlib import Path

# Load .env from repo root (works regardless of cwd)
try:
    from dotenv import load_dotenv
    _REPO_ROOT = Path(__file__).parent
    load_dotenv(_REPO_ROOT / ".env", override=False)
except ImportError:
    pass  # dotenv optional; fall back to os.environ only

# ---------------------------------------------------------------------------
# Core paths
# ---------------------------------------------------------------------------
DATA_ROOT: str = os.getenv("DATA_ROOT", "")

# Conference folder names (Neurlps2025 preserves original typo in dataset)
_CONF_DIRS = {
    "ICLR2024":    "ICLR2024",
    "ICLR2025":    "ICLR2025",
    "ICLR2026":    "ICLR2026",
    "ICML2025":    "ICML2025",
    "NeurIPS2025": "Neurlps2025",   # note: dataset folder has this spelling
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
    conf_lower = conference.lower().replace("neurips", "neurips")
    # Handle NeurIPS special case
    if conference == "NeurIPS2025":
        fname = f"paper_ids_{subset}_neurips2025.txt"
    else:
        fname = f"paper_ids_{subset}_{conference.lower()}.txt"
    return os.path.join(base, fname)


# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------
GOOGLE_API_KEY:         str = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL:           str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

MIMO_API_KEY:           str = os.getenv("MIMO_API_KEY", "")
MIMO_MODEL:             str = os.getenv("MIMO_MODEL", "mimo-v2.5-pro")
MIMO_BASE_URL:          str = os.getenv("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")

OPENAI_API_KEY:         str = os.getenv("OPENAI_API_KEY", "")

AZURE_OPENAI_API_KEY:   str = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT:  str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")


def validate_env(require_gemini: bool = True, require_mimo: bool = False) -> None:
    """Raise EnvironmentError if required variables are missing."""
    errors = []
    if not DATA_ROOT:
        errors.append("DATA_ROOT is not set")
    if require_gemini and not GOOGLE_API_KEY:
        errors.append("GOOGLE_API_KEY is not set")
    if require_mimo and not MIMO_API_KEY:
        errors.append("MIMO_API_KEY is not set")
    if errors:
        raise EnvironmentError(
            "Missing required environment variables:\n" +
            "\n".join(f"  • {e}" for e in errors) +
            "\n\nPlease copy .env.example to .env and fill in the values."
        )

