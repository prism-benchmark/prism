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
    DATA_ROOT, conf_path, reviewer_dir,
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
