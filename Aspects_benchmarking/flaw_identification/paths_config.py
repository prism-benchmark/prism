"""
paths_config.py — Centralized path configuration for the Flaw Identification pipeline.
All paths are derived from DATA_ROOT in your root .env file.
"""
import os
import sys

# Load shared env (repo root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env_loader import (
    DATA_ROOT, conf_path,
    GOOGLE_API_KEY, GEMINI_MODEL,
    MIMO_API_KEY, MIMO_MODEL, MIMO_BASE_URL,
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT,
    OPENAI_API_KEY,
    validate_env,
)

# ---------------------------------------------------------------------------
# Conference data roots
# ---------------------------------------------------------------------------
def get_conf_data(conference: str) -> str:
    """Return the data directory for a conference."""
    return conf_path(conference)

# Convenience aliases used across main_cfi_*.py files
ICLR2024_DATA    = conf_path("ICLR2024")
ICLR2025_DATA    = conf_path("ICLR2025")
ICLR2026_DATA    = conf_path("ICLR2026")
ICML2025_DATA    = conf_path("ICML2025")
NEURIPS2025_DATA = conf_path("NeurIPS2025")


def reviewer_dir(conference: str, reviewer: str) -> str:
    """Return path to reviewer data folder.

    Args:
        conference: ICLR2024 | ICLR2025 | ICLR2026 | ICML2025 | NeurIPS2025
        reviewer:   sea | tree | reviewer2 | deepreview | cyclereview | human
    """
    base = conf_path(conference)
    # NeurIPS uses "neurlps" in folder names (original dataset spelling)
    conf_lower = conference.lower().replace("neurips2025", "neurlps2025")
    if reviewer == "human":
        return os.path.join(base, "human_reviews")
    return os.path.join(base, f"{reviewer}_{conf_lower}")


# ---------------------------------------------------------------------------
# Output dirs (relative to this module)
# ---------------------------------------------------------------------------
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(_MODULE_DIR, "output")

# ---------------------------------------------------------------------------
# API / Model settings
# ---------------------------------------------------------------------------
GEMINI_API_KEY = GOOGLE_API_KEY
GEMINI_TEMP    = 0.0
MIMO_TEMP      = 0.0

