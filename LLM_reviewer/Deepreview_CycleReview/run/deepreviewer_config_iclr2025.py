import os

# ============================================================
# DeepReviewer Pipeline Configuration - ICLR2025
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
DATA_ROOT = os.getenv("DATA_ROOT", "/path/to/data")
MMD_FOLDER  = os.getenv("DEEPREVIEWER_MMD_FOLDER", os.path.join(DATA_ROOT, "ICLR2025", "grobid_fulltext"))
JSON_FOLDER = os.getenv("DEEPREVIEWER_JSON_FOLDER", os.path.join(DATA_ROOT, "ICLR2025", "json"))

# --- Paper Selection ---
# Set to None to process all papers, or provide path to text file with paper IDs (one per line)
PAPER_IDS_FILE = os.getenv("DEEPREVIEWER_PAPER_IDS_FILE", os.path.join(DATA_ROOT, "ICLR2025", "data_subset", "paper_ids_200.txt"))

# --- Output ---
OUTPUT_FOLDER = os.getenv("DEEPREVIEWER_OUTPUT_FOLDER", "outputs/deepreview_iclr2025")
SUMMARY_FILE  = os.getenv("DEEPREVIEWER_SUMMARY_FILE", "outputs/summary_deepreview_iclr2025.json")

# --- Resuming ---
# If True, skip papers that already have a result file in OUTPUT_FOLDER
SKIP_COMPLETED = True

# --- Batch size ---
BATCH_SIZE = 8  # Xu ly 8 papers cung luc (co the tang len 16 neu GPU du)