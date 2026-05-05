import os

# ── MODEL ─────────────────────────────────────────────────────────────────────
MODEL_SIZE          = "8B"
GPU_ID              = "1"
USE_SEMANTIC_SEARCH = False    # Disable API semantic search
HF_TOKEN = os.getenv("HF_TOKEN", "YOUR_HF_TOKEN")
GPU_MEMORY_UTILIZATION = 0.85
MAX_MODEL_LEN       = 24000  # Model default: 24320, set to 24000 for safety margin

# ── DATA ──────────────────────────────────────────────────────────────────────
MMD_FOLDER          = "/mnt/duyna/review_assessment/data/NeurIPS2025/grobid_fulltext"
JSON_FOLDER         = "/mnt/duyna/review_assessment/data/NeurIPS2025/json"

# ── PAPER SELECTION ───────────────────────────────────────────────────────────
# Set to None to process all papers, or provide a text file with one paper ID per line.
PAPER_IDS_FILE      = "/mnt/duyna/review_assessment/data/NeurIPS2025/data_subset/paper_ids_error.txt"

# ── OUTPUT ────────────────────────────────────────────────────────────────────
OUTPUT_FOLDER       = "/mnt/duyna/review_assessment/Cyclereview_NEURIPS2025_output_fix_empty_review"
SUMMARY_FILE        = "/mnt/duyna/review_assessment/cyclereviewer_summary/summary_cycle_reviewer_neurips2025.json"
SKIP_COMPLETED      = True

# ── CACHE (use local model) ───────────────────────────────────────────────────
HF_HOME             = "/mnt/duyna/review_assessment/model"
