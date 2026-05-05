import os

# ============================================================
# DeepReviewer Pipeline Configuration - ICLR2024
# Edit ONLY this file to change settings
# ============================================================

# --- HuggingFace ---
HF_TOKEN = os.getenv("HF_TOKEN", "YOUR_HF_TOKEN")

# --- Semantic Scholar ---
# Semantic Scholar API enabled for enhanced paper search and retrieval
S2_API_KEY = os.getenv("S2_API_KEY", "YOUR_S2_API_KEY")

# --- GPU Assignment ---
# Use "auto" to pick the GPUs with the most free VRAM at runtime.
DEEPREVIEWER_GPU      = "auto"

# --- Models ---
DEEPREVIEWER_SIZE      = "14B"
TENSOR_PARALLEL_SIZE   = 2      # must match number of GPUs in DEEPREVIEWER_GPU
GPU_MEMORY_UTILIZATION = 0.9
REVIEW_MODE            = "Standard Mode"  # "Fast Mode", "Standard Mode", "Best Mode"
REVIEWER_NUM           = 1          # 3 reviewers + 1 meta review (auto)

# --- Dataset Folders ---
MMD_FOLDER  = "/mnt/duyna/review_assessment/data/ICLR2024/paper_nougat_mmd"
JSON_FOLDER = "/mnt/duyna/review_assessment/data/ICLR2024/Human_and_meta_reviews"

# --- Paper Selection ---
# Set to None to process all papers, or provide path to text file with paper IDs (one per line)
PAPER_IDS_FILE = "/mnt/duyna/review_assessment/data/ICLR2024/data_subset/paper_ids_error.txt"

# --- Output ---
OUTPUT_FOLDER = "/mnt/duyna/review_assessment/Deepreview_ICLR2024_output_fix_empty_review"
SUMMARY_FILE  = "/mnt/duyna/review_assessment/summary/summary2024iclrDEEP_fix_empty_review.json"

# --- Resuming ---
# If True, skip papers that already have a result file in OUTPUT_FOLDER
SKIP_COMPLETED = True

# --- Batch size ---
BATCH_SIZE = 8  # Xu ly 8 papers cung luc (co the tang len 16 neu GPU du)