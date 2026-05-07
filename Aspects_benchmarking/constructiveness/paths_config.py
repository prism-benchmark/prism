"""
paths_config.py — Centralized path configuration for the Constructiveness pipeline.
All paths are derived from DATA_ROOT in your root .env file.
AI model settings are imported from the centralized ai_config.py.
"""
import os
import sys

# Load shared env (repo root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env_loader import (
    DATA_ROOT, conf_path,
    GOOGLE_API_KEY, GEMINI_MODEL,
    MIMO_API_KEY, MIMO_MODEL, MIMO_BASE_URL,
    validate_env,
)

# ── Import AI model settings from centralized config ──────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ai_config import (
    GEMINI_TEMP,
    MIMO_TEMP,
)

# ---------------------------------------------------------------------------
# Reviewer sub-folder names inside each conference directory
# ---------------------------------------------------------------------------
REVIEWER_SUBDIRS = {
    "sea":         "sea_{conf_lower}",
    "tree":        "tree_{conf_lower}",
    "reviewer2":   "reviewer2_{conf_lower}",
    "deepreview":  "deepreview_{conf_lower}",
    "cyclereview": "cyclereview_{conf_lower}",
    "human":       "human_reviews",
}


def reviewer_dir(conference: str, reviewer: str) -> str:
    """Return path to reviewer data for a given conference.

    Args:
        conference: ICLR2024 | ICLR2025 | ICLR2026 | ICML2025 | NeurIPS2025
        reviewer:   sea | tree | reviewer2 | deepreview | cyclereview | human
    """
    template = REVIEWER_SUBDIRS.get(reviewer, f"{reviewer}_{{conf_lower}}")
    # NeurIPS uses "neurlps" in folder names (original dataset spelling)
    conf_lower = conference.lower().replace("neurips2025", "neurlps2025")
    subfolder = template.format(conf_lower=conf_lower)
    return os.path.join(conf_path(conference), subfolder)


# ---------------------------------------------------------------------------
# Convenience constants used by run_constructiveness*.py
# ---------------------------------------------------------------------------
CONFERENCES = ["ICLR2024", "ICLR2025", "ICLR2026", "ICML2025", "NeurIPS2025"]

# Output directory (relative to this module)
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(_MODULE_DIR, "output")

# ---------------------------------------------------------------------------
# API / Model settings (imported from ai_config via env_loader)
# ---------------------------------------------------------------------------
GEMINI_API_KEY = GOOGLE_API_KEY
