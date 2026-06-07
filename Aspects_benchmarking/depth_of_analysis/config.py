"""
config.py — Centralized configuration for the Depth of Analysis pipeline.

All paths are derived from DATA_ROOT in your .env file.
AI model settings are imported from the centralized ai_config.py.

To add a new LLM source:
  1. Create a folder under DATA_ROOT/<CONFERENCE>/<source_name>/
  2. Add an entry to LLM_SOURCES below
  3. Run: python run_llm.py --source <name>
"""

import os
import sys

# ── Load shared env (repo root) ───────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from env_loader import (
    DATA_ROOT,
    conf_path,
    paper_ids_file,
    GOOGLE_API_KEY,
    GEMINI_MODEL,
    MIMO_API_KEY,
    MIMO_MODEL,
    MIMO_BASE_URL,
    OPENAI_API_KEY,
    validate_env,
)

# ── Import AI model settings from centralized config ──────────────────────
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ai_config import (
    GEMINI_TEMP,
    GPT_TEMP,
    GPT_MODEL,
    MIMO_TEMP,
)

# ── Project paths ─────────────────────────────────────────────────────────
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.environ.get("DOA_OUTPUT_ROOT") or os.path.join(PIPELINE_DIR, "output")
OUTPUT_METRICS_DIR = os.path.join(OUTPUT_ROOT, "metrics")

# Human review dirs per conference
HUMAN_DIRS = {
    "ICLR2024": os.path.join(conf_path("ICLR2024"), "human_reviews"),
    "ICLR2025": os.path.join(conf_path("ICLR2025"), "human_reviews"),
    "ICLR2026": os.path.join(conf_path("ICLR2026"), "human_reviews"),
    "ICML2025": os.path.join(conf_path("ICML2025"), "human_reviews"),
    "NeurIPS2025": os.path.join(conf_path("NeurIPS2025"), "human_reviews"),
}

# Default (ICLR 2026 — legacy compat)
HUMAN_DIR = HUMAN_DIRS["ICLR2026"]


# ── LLM Sources ───────────────────────────────────────────────────────────
def _p(conference: str, subfolder: str) -> str:
    return os.path.join(conf_path(conference), subfolder)


LLM_SOURCES = {
    # ── SEA ──────────────────────────────────────────────────────────────
    "sea_iclr2024": {"dir": _p("ICLR2024", "sea_iclr2024"), "format": "txt"},
    "sea_iclr2025": {"dir": _p("ICLR2025", "sea_iclr2025"), "format": "txt"},
    "sea_iclr2026": {"dir": _p("ICLR2026", "sea_iclr2026"), "format": "txt"},
    "sea_icml2025": {"dir": _p("ICML2025", "sea_icml2025"), "format": "txt"},
    "sea_neurlps2025": {"dir": _p("NeurIPS2025", "sea_neurlps2025"), "format": "txt"},
    # ── TreeReview ───────────────────────────────────────────────────────
    "tree_iclr2024": {"dir": _p("ICLR2024", "tree_iclr2024"), "format": "tree_json"},
    "tree_iclr2025": {"dir": _p("ICLR2025", "tree_iclr2025"), "format": "tree_json"},
    "tree_iclr2026": {"dir": _p("ICLR2026", "tree_iclr2026"), "format": "tree_json"},
    "tree_iclr2026_2": {
        "dir": _p("ICLR2026", "tree_iclr2026_2"),
        "format": "tree_json",
    },
    "tree_icml2025": {"dir": _p("ICML2025", "tree_icml2025"), "format": "tree_json"},
    "tree_icml2025_2": {
        "dir": _p("ICML2025", "tree_icml2025_2"),
        "format": "tree_json",
    },
    "tree_neurips2025": {
        "dir": _p("NeurIPS2025", "tree_neurips2025"),
        "format": "tree_json",
    },
    "tree_neurips2025_2": {
        "dir": _p("NeurIPS2025", "tree_neurips2025_2"),
        "format": "tree_json",
    },
    # ── Reviewer2 ────────────────────────────────────────────────────────
    "reviewer2_iclr2024": {
        "dir": _p("ICLR2024", "reviewer2_iclr2024"),
        "format": "reviewer2_txt",
    },
    "reviewer2_iclr2025": {
        "dir": _p("ICLR2025", "reviewer2_iclr2025"),
        "format": "reviewer2_txt",
    },
    "reviewer2_iclr2026": {
        "dir": _p("ICLR2026", "reviewer2_iclr2026"),
        "format": "reviewer2_txt",
    },
    "reviewer2_icml2025": {
        "dir": _p("ICML2025", "reviewer2_icml2025"),
        "format": "reviewer2_txt",
    },
    "reviewer2_neurips2025": {
        "dir": _p("NeurIPS2025", "reviewer2_neurips2025"),
        "format": "reviewer2_txt",
    },
    # ── DeepReview ───────────────────────────────────────────────────────
    "deepreview_iclr2024": {
        "dir": _p("ICLR2024", "deepreview_iclr2024"),
        "format": "deepreview_json",
    },
    "deepreview_iclr2025": {
        "dir": _p("ICLR2025", "deepreview_iclr2025"),
        "format": "deepreview_json",
    },
    "deepreview_iclr2026": {
        "dir": _p("ICLR2026", "deepreview_iclr2026"),
        "format": "deepreview_json",
    },
    "deepreview_icml2025": {
        "dir": _p("ICML2025", "deepreview_icml2025"),
        "format": "deepreview_json",
    },
    "deepreview_neurips2025": {
        "dir": _p("NeurIPS2025", "deepreview_neurips2025"),
        "format": "deepreview_json",
    },
    # ── CycleReview ──────────────────────────────────────────────────────
    "cyclereview_iclr2024": {
        "dir": _p("ICLR2024", "cyclereview_iclr2024"),
        "format": "cyclereview_json",
    },
    "cyclereview_iclr2025": {
        "dir": _p("ICLR2025", "cyclereview_iclr2025"),
        "format": "cyclereview_json",
    },
    "cyclereview_iclr2026": {
        "dir": _p("ICLR2026", "cyclereview_iclr2026"),
        "format": "cyclereview_json",
    },
    "cyclereview_icml2025": {
        "dir": _p("ICML2025", "cyclereview_icml2025"),
        "format": "cyclereview_json",
    },
    "cyclereview_neurlps2025": {
        "dir": _p("NeurIPS2025", "cyclereview_neurlps2025"),
        "format": "cyclereview_json",
    },
}


# ── Output helpers ────────────────────────────────────────────────────────
def get_llm_output_dir(source_name: str) -> str:
    return os.path.join(OUTPUT_ROOT, source_name)


# ── Paper IDs (50-paper subset per conference) ────────────────────────────
PAPER_IDS_50_BY_CONFERENCE = {
    conf: paper_ids_file(conf, 50)
    for conf in ["ICLR2024", "ICLR2025", "ICLR2026", "ICML2025", "NeurIPS2025"]
}
PAPER_IDS_50_FILE = PAPER_IDS_50_BY_CONFERENCE["ICLR2024"]

# ── Model settings (imported from ai_config) ─────────────────────────────
GEMINI_API_KEY = GOOGLE_API_KEY

GPT_API_KEY = OPENAI_API_KEY

# ── Output dirs for human (per conference) ────────────────────────────────
OUTPUT_HUMAN_DIR = os.path.join(OUTPUT_ROOT, "human_iclr2026")
