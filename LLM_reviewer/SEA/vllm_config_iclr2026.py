"""
Configuration for vLLM Review Generation - ICLR2026
Edit this file to customize settings before running the pipeline
"""

# ============================================================
# Model Configuration
# ============================================================
MODEL_NAME = "Qwen/Qwen2-7B-Instruct"  # or "meta-llama/Llama-2-7b-chat-hf", etc.

# ============================================================
# Data Paths
# ============================================================
INPUT_DIR = "/mnt/duyna/review_assessment/data/ICLR2026/grobid_fulltext"
PAPER_IDS_FILE = "/mnt/duyna/review_assessment/data/ICLR2026/data_subset/paper_ids_200.txt"
OUTPUT_DIR = "/mnt/duyna/review_assessment/sea_output_iclr2026"

# ============================================================
# Inference Parameters
# ============================================================
BATCH_SIZE = 4                          # Increase for faster processing (requires more VRAM)
MAX_TOKENS = 8192                       # Maximum tokens per review

# Context Window Configuration
MAX_MODEL_LEN = 32000                  # Reduced to fit available KV cache on the current GPU setup
# This is the maximum context supported by the model's position embeddings

# Sampling parameters
TEMPERATURE = 0.7                       # Higher = more creative, Lower = more deterministic
TOP_P = 0.95                           # Nucleus sampling parameter

# ============================================================
# GPU Configuration
# ============================================================
TENSOR_PARALLEL_SIZE = 1               # Number of GPUs (set to 2+ for larger models)
GPU_MEMORY_UTILIZATION = 0.76          # GPU memory utilization (0.85-0.95 recommended)
CUDA_VISIBLE_DEVICES = "6"             # GPU IDs to use

# ============================================================
# Processing Options
# ============================================================
SKIP_COMPLETED = True                  # Skip papers already in output directory
VERBOSE = True                         # Print detailed logs
