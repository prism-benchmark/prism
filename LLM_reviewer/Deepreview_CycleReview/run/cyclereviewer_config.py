import os
import sys
from pathlib import Path as _Path

# ── Import centralized AI config ─────────────────────────────────────────
sys.path.insert(0, str(_Path(__file__).parent.parent.parent.parent))
from ai_config import HF_TOKEN as _HF_TOKEN, DATA_ROOT as _AI_DATA_ROOT, get_reviewer_config as _get_reviewer_cfg  # noqa: E402

# ── Load per-reviewer config from llm_config.yaml ────────────────────────
_cyclereview_cfg = {}
try:
    _cyclereview_cfg = _get_reviewer_cfg("cyclereview")
except Exception:
    pass

# ── MODEL ─────────────────────────────────────────────────────────────────────
MODEL_SIZE          = _cyclereview_cfg.get("model_size", "8B")
GPU_ID              = os.getenv("CUDA_VISIBLE_DEVICES", "0")
USE_SEMANTIC_SEARCH = False
HF_TOKEN = _HF_TOKEN or os.getenv("HF_TOKEN", "YOUR_HF_TOKEN")
GPU_MEMORY_UTILIZATION = float(os.getenv("CYCLEREVIEWER_GPU_MEMORY_UTILIZATION", str(_cyclereview_cfg.get("gpu_memory_utilization", 0.90))))
MAX_MODEL_LEN       = int(os.getenv("CYCLEREVIEWER_MAX_MODEL_LEN", str(_cyclereview_cfg.get("max_model_len", 24000))))

# ── DATA ──────────────────────────────────────────────────────────────────────
DATA_ROOT           = os.getenv("DATA_ROOT", _AI_DATA_ROOT or "/path/to/data")
PAPERS_FOLDER       = os.getenv("CYCLEREVIEWER_PAPERS_FOLDER", os.path.join(DATA_ROOT, "ICLR2026", "grobid_fulltext"))
JSON_FOLDER         = os.getenv("CYCLEREVIEWER_JSON_FOLDER", os.path.join(DATA_ROOT, "ICLR2026", "json"))

# ── PAPER SELECTION ───────────────────────────────────────────────────────────
PAPER_IDS_FILE      = os.getenv("CYCLEREVIEWER_PAPER_IDS_FILE", os.path.join(DATA_ROOT, "ICLR2026", "data_subset", "paper_ids.txt"))

# ── OUTPUT ────────────────────────────────────────────────────────────────────
OUTPUT_FOLDER       = os.getenv("CYCLEREVIEWER_OUTPUT_FOLDER", "outputs/cyclereview_iclr2026")
SUMMARY_FILE        = os.getenv("CYCLEREVIEWER_SUMMARY_FILE", "outputs/summary_cyclereview_iclr2026.json")
SKIP_COMPLETED      = True

# ── CACHE ─────────────────────────────────────────────────────────────────────
HF_HOME             = os.getenv("HF_HOME", "models")
