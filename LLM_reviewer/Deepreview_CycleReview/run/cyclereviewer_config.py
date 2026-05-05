import os

# ── MODEL ─────────────────────────────────────────────────────────────────────
MODEL_SIZE          = "8B"
GPU_ID              = os.getenv("CUDA_VISIBLE_DEVICES", "0")
USE_SEMANTIC_SEARCH = False    # Disable API semantic search
HF_TOKEN = os.getenv("HF_TOKEN", "YOUR_HF_TOKEN")
GPU_MEMORY_UTILIZATION = 0.90
MAX_MODEL_LEN       = 24000

# ── DATA ──────────────────────────────────────────────────────────────────────
DATA_ROOT           = os.getenv("DATA_ROOT", "/path/to/data")
MMD_FOLDER          = os.getenv("CYCLEREVIEWER_MMD_FOLDER", os.path.join(DATA_ROOT, "ICLR2026", "grobid_fulltext"))
JSON_FOLDER         = os.getenv("CYCLEREVIEWER_JSON_FOLDER", os.path.join(DATA_ROOT, "ICLR2026", "json"))

# ── PAPER SELECTION ───────────────────────────────────────────────────────────
# Set to None to process all papers, or provide a text file with one paper ID per line.
PAPER_IDS_FILE      = os.getenv("CYCLEREVIEWER_PAPER_IDS_FILE", os.path.join(DATA_ROOT, "ICLR2026", "data_subset", "paper_ids.txt"))

# ── OUTPUT ────────────────────────────────────────────────────────────────────
OUTPUT_FOLDER       = os.getenv("CYCLEREVIEWER_OUTPUT_FOLDER", "outputs/cyclereview_iclr2026")
SUMMARY_FILE        = os.getenv("CYCLEREVIEWER_SUMMARY_FILE", "outputs/summary_cyclereview_iclr2026.json")
SKIP_COMPLETED      = True

# ── CACHE (use local model) ───────────────────────────────────────────────────
HF_HOME             = os.getenv("HF_HOME", "models")