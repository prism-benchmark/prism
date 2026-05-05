#!/usr/bin/env python3
"""
Simple wrapper script to run vLLM review generation with config file
Usage: python generate_reviews.py [--reconfigure]
"""

import os
import sys
import argparse
from pathlib import Path

# Add SEA directory to path
SEA_DIR = Path(__file__).parent
sys.path.insert(0, str(SEA_DIR))

def main():
    parser = argparse.ArgumentParser(
        description="Generate reviews using vLLM (quick runner)"
    )
    parser.add_argument(
        "--reconfigure",
        action="store_true",
        help="Show configuration before running"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show command without running"
    )
    
    args = parser.parse_args()
    
    # Import config
    try:
        import vllm_config as cfg
    except ImportError:
        print("Error: vllm_config.py not found!")
        print("Please run this script from the LLM_reviewer/SEA directory or add it to PYTHONPATH.")
        sys.exit(1)
    
    # Show configuration
    if args.reconfigure:
        print("=" * 60)
        print("Current Configuration")
        print("=" * 60)
        print(f"Model: {cfg.MODEL_NAME}")
        print(f"Input Dir: {cfg.INPUT_DIR}")
        print(f"Paper IDs: {cfg.PAPER_IDS_FILE}")
        print(f"Output Dir: {cfg.OUTPUT_DIR}")
        print(f"Batch Size: {cfg.BATCH_SIZE}")
        print(f"Max Tokens: {cfg.MAX_TOKENS}")
        print(f"Temperature: {cfg.TEMPERATURE}")
        print(f"Tensor Parallel Size: {cfg.TENSOR_PARALLEL_SIZE}")
        print(f"GPU Memory Util: {cfg.GPU_MEMORY_UTILIZATION}")
        print(f"CUDA Devices: {cfg.CUDA_VISIBLE_DEVICES}")
        print("=" * 60)
    
    # Build command
    cmd = [
        "python", "run_review_vllm.py",
        "--model", cfg.MODEL_NAME,
        "--input-dir", cfg.INPUT_DIR,
        "--paper-ids", cfg.PAPER_IDS_FILE,
        "--output-dir", cfg.OUTPUT_DIR,
        "--batch-size", str(cfg.BATCH_SIZE),
        "--max-tokens", str(cfg.MAX_TOKENS),
        "--temperature", str(cfg.TEMPERATURE),
        "--top-p", str(cfg.TOP_P),
        "--tensor-parallel-size", str(cfg.TENSOR_PARALLEL_SIZE),
        "--gpu-memory-utilization", str(cfg.GPU_MEMORY_UTILIZATION),
    ]
    
    if cfg.SKIP_COMPLETED:
        cmd.append("--skip-completed")
    
    if args.dry_run:
        print("\nCommand to execute:")
        print(" ".join(cmd))
        print("\nDry run complete. Remove --dry-run to execute.")
        return
    
    # Set CUDA devices
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CUDA_VISIBLE_DEVICES
    
    # Execute
    print(f"\n{'='*60}")
    print("Starting vLLM Review Generation")
    print(f"{'='*60}\n")
    
    os.execvp("python", cmd)


if __name__ == "__main__":
    main()
