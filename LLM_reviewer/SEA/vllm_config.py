"""
Configuration for vLLM Review Generation
Edit this file to customize settings before running the pipeline
"""

import os

# ============================================================
# Model Configuration
# ============================================================
MODEL_NAME = "Qwen/Qwen2-7B-Instruct"  # or "meta-llama/Llama-2-7b-chat-hf", etc.

# ============================================================
# Data Paths
# ============================================================
SEA_DATA_ROOT = os.getenv("SEA_DATA_ROOT", "/path/to/sea_data")
INPUT_DIR = os.getenv("SEA_INPUT_DIR", os.path.join(SEA_DATA_ROOT, "NeurIPS2025", "grobid_fulltext"))
PAPER_IDS_FILE = os.getenv("SEA_PAPER_IDS_FILE", os.path.join(SEA_DATA_ROOT, "NeurIPS2025", "data_subset", "paper_ids.txt"))
OUTPUT_DIR = os.getenv("SEA_OUTPUT_DIR", "outputs/sea_neurips2025")

# ============================================================
# Inference Parameters
# ============================================================
BATCH_SIZE = 8                          # Increase for faster processing (requires more VRAM)
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
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "0")  # GPU IDs to use

# ============================================================
# Processing Options
# ============================================================
SKIP_COMPLETED = True                  # Skip papers already in output directory
VERBOSE = True                         # Print detailed logs
