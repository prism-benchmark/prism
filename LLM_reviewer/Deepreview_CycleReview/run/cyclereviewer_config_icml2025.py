import os
import sys
from pathlib import Path as _Path

# Import centralized AI config
sys.path.insert(0, str(_Path(__file__).parent.parent.parent.parent))
from ai_config import HF_TOKEN as _HF_TOKEN, DATA_ROOT as _AI_DATA_ROOT  # noqa: E402

# ── MODEL ─────────────────────────────────────────────────────────────────────
MODEL_SIZE          = "8B"
GPU_ID              = "0"
USE_SEMANTIC_SEARCH = False    # Disable API semantic search
HF_TOKEN = _HF_TOKEN or os.getenv("HF_TOKEN", "YOUR_HF_TOKEN")
GPU_MEMORY_UTILIZATION = 0.85
MAX_MODEL_LEN       = 24000  # Model default: 24320, set to 24000 for safety margin

# ── DATA ──────────────────────────────────────────────────────────────────────
DATA_ROOT           = os.getenv("DATA_ROOT", _AI_DATA_ROOT or "/path/to/data")
PAPERS_FOLDER          = os.getenv("CYCLEREVIEWER_PAPERS_FOLDER", os.path.join(DATA_ROOT, "ICML2025", "grobid_fulltext"))
JSON_FOLDER         = os.getenv("CYCLEREVIEWER_JSON_FOLDER", os.path.join(DATA_ROOT, "ICML2025", "json"))

# ── PAPER SELECTION ───────────────────────────────────────────────────────────
# Set to None to process all papers, or provide a text file with one paper ID per line.
PAPER_IDS_FILE      = os.getenv("CYCLEREVIEWER_PAPER_IDS_FILE", os.path.join(DATA_ROOT, "ICML2025", "data_subset", "paper_ids.txt"))

# ── OUTPUT ────────────────────────────────────────────────────────────────────
OUTPUT_FOLDER       = os.getenv("CYCLEREVIEWER_OUTPUT_FOLDER", "outputs/cyclereview_icml2025")
SUMMARY_FILE        = os.getenv("CYCLEREVIEWER_SUMMARY_FILE", "outputs/summary_cyclereview_icml2025.json")
SKIP_COMPLETED      = True

# ── CACHE (use local model) ───────────────────────────────────────────────────
HF_HOME             = os.getenv("HF_HOME", "models")
