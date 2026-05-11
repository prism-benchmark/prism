import os
import sys
from pathlib import Path as _Path

# ── Import centralized AI config ─────────────────────────────────────────
sys.path.insert(0, str(_Path(__file__).parent.parent.parent.parent))
from ai_config import HF_TOKEN, DATA_ROOT as _AI_DATA_ROOT, get_reviewer_config as _get_reviewer_cfg  # noqa: E402

# ── Load per-reviewer config from llm_config.yaml ────────────────────────
_deepreview_cfg = {}
try:
    _deepreview_cfg = _get_reviewer_cfg("deepreview")
except Exception:
    pass

# ============================================================
# DeepReviewer Pipeline Configuration
# Per-reviewer overrides can be set in llm_config.yaml under `reviewers.deepreview`.
# ============================================================

# --- HuggingFace ---
HF_TOKEN = HF_TOKEN or os.getenv("HF_TOKEN", "YOUR_HF_TOKEN")

# --- Semantic Scholar ---
S2_API_KEY = os.getenv("S2_API_KEY", "YOUR_S2_API_KEY")

# --- GPU Assignment ---
DEEPREVIEWER_GPU      = "auto"

# --- Models ---
DEEPREVIEWER_SIZE      = _deepreview_cfg.get("model_size", "14B")
TENSOR_PARALLEL_SIZE   = int(os.getenv("DEEPREVIEWER_TENSOR_PARALLEL_SIZE", str(_deepreview_cfg.get("tensor_parallel_size", 2))))
GPU_MEMORY_UTILIZATION = float(os.getenv("DEEPREVIEWER_GPU_MEMORY_UTILIZATION", str(_deepreview_cfg.get("gpu_memory_utilization", 0.75))))
REVIEW_MODE            = _deepreview_cfg.get("review_mode", "Standard Mode")
REVIEWER_NUM           = int(os.getenv("DEEPREVIEWER_REVIEWER_NUM", str(_deepreview_cfg.get("reviewer_num", 1))))

# --- Dataset Folders ---
DATA_ROOT = os.getenv("DATA_ROOT", _AI_DATA_ROOT or "/path/to/data")
PAPERS_FOLDER  = os.getenv("DEEPREVIEWER_PAPERS_FOLDER", os.path.join(DATA_ROOT, "ICLR2026", "grobid_fulltext"))
JSON_FOLDER = os.getenv("DEEPREVIEWER_JSON_FOLDER", os.path.join(DATA_ROOT, "ICLR2026", "json"))

# --- Paper Selection ---
PAPER_IDS_FILE = os.getenv("DEEPREVIEWER_PAPER_IDS_FILE", os.path.join(DATA_ROOT, "ICLR2026", "data_subset", "paper_ids_200.txt"))

# --- Output ---
OUTPUT_FOLDER = os.getenv("DEEPREVIEWER_OUTPUT_FOLDER", "outputs/deepreview_iclr2026")
SUMMARY_FILE  = os.getenv("DEEPREVIEWER_SUMMARY_FILE", "outputs/summary_deepreview_iclr2026.json")

# --- Resuming ---
SKIP_COMPLETED = True

# --- Batch size ---
BATCH_SIZE = int(os.getenv("DEEPREVIEWER_BATCH_SIZE", str(_deepreview_cfg.get("batch_size", 8))))
