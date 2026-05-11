"""
Configuration for vLLM Review Generation (SEA)
AI model settings are imported from the centralized ai_config.py.
Per-reviewer overrides can be set in llm_config.yaml under `reviewers.sea`.
Edit this file to customize hardware/inference settings before running the pipeline.
"""

import os
import sys
from pathlib import Path as _Path

# ── Import centralized AI config ─────────────────────────────────────────
sys.path.insert(0, str(_Path(__file__).parent.parent.parent))
from ai_config import HF_TOKEN as _HF_TOKEN, DATA_ROOT as _AI_DATA_ROOT, get_reviewer_config as _get_reviewer_cfg  # noqa: E402

# ── Load per-reviewer config from llm_config.yaml ────────────────────────
_sea_cfg = {}
try:
    _sea_cfg = _get_reviewer_cfg("sea")
except Exception:
    pass

# ============================================================
# Model Configuration
# ============================================================
MODEL_NAME = _sea_cfg.get("model_path", "Qwen/Qwen2-7B-Instruct")

# ============================================================
# Data Paths
# ============================================================
SEA_DATA_ROOT = os.getenv("SEA_DATA_ROOT", _AI_DATA_ROOT or "/path/to/sea_data")
INPUT_DIR = os.getenv("SEA_INPUT_DIR", os.path.join(SEA_DATA_ROOT, "NeurIPS2025", "grobid_fulltext"))
PAPER_IDS_FILE = os.getenv("SEA_PAPER_IDS_FILE", os.path.join(SEA_DATA_ROOT, "NeurIPS2025", "data_subset", "paper_ids.txt"))
OUTPUT_DIR = os.getenv("SEA_OUTPUT_DIR", "outputs/sea_neurips2025")

# ============================================================
# Inference Parameters
# ============================================================
BATCH_SIZE = int(os.getenv("SEA_BATCH_SIZE", str(_sea_cfg.get("batch_size", 8))))
MAX_TOKENS = int(os.getenv("SEA_MAX_TOKENS", str(_sea_cfg.get("max_tokens", 8192))))

# Context Window Configuration
MAX_MODEL_LEN = int(os.getenv("SEA_MAX_MODEL_LEN", str(_sea_cfg.get("max_model_len", 32000))))

# Sampling parameters
TEMPERATURE = float(os.getenv("SEA_TEMPERATURE", str(_sea_cfg.get("temperature", 0.7))))
TOP_P = float(os.getenv("SEA_TOP_P", str(_sea_cfg.get("top_p", 0.95))))

# ============================================================
# GPU Configuration
# ============================================================
TENSOR_PARALLEL_SIZE = int(os.getenv("SEA_TENSOR_PARALLEL_SIZE", str(_sea_cfg.get("tensor_parallel_size", 1))))
GPU_MEMORY_UTILIZATION = float(os.getenv("SEA_GPU_MEMORY_UTILIZATION", str(_sea_cfg.get("gpu_memory_utilization", 0.76))))
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "0")

# ============================================================
# Processing Options
# ============================================================
SKIP_COMPLETED = True
VERBOSE = True
