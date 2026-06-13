"""
paths_config.py — Centralized path configuration for the Flaw Identification pipeline.
All paths are derived from DATA_ROOT in your root .env file.
AI model settings are imported from the centralized ai_config.py.
"""
import os
import sys

# Load shared env (repo root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env_loader import (
    DATA_ROOT, conf_path, reviewer_dir,
    GOOGLE_API_KEY, GEMINI_MODEL,
    MIMO_API_KEY, MIMO_MODEL, MIMO_BASE_URL,
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT,
    OPENAI_API_KEY,
    validate_env,
)

# ── Import AI model settings from centralized config ──────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ai_config import (
    GEMINI_TEMP,
    MIMO_TEMP,
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

# ---------------------------------------------------------------------------
# Output dirs (relative to this module)
# ---------------------------------------------------------------------------
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(_MODULE_DIR, "output")

# ---------------------------------------------------------------------------
# API / Model settings (imported from ai_config via env_loader)
# ---------------------------------------------------------------------------
GEMINI_API_KEY = GOOGLE_API_KEY
