#!/usr/bin/env python3
"""
SEA Review Generation Runner for ICLR2026
Simple wrapper script to run vLLM review generation with ICLR2026 config
Usage: python generate_reviews_iclr2026.py [--reconfigure] [--dry-run]
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
        description="Generate reviews using vLLM for ICLR2026 (quick runner)"
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
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of papers to process"
    )
    
    args = parser.parse_args()
    
    # Import config for ICLR2026
    try:
        import vllm_config_iclr2026 as cfg
    except ImportError:
        print("Error: vllm_config_iclr2026.py not found!")
        print("Please run this script from the LLM_reviewer/SEA directory or add it to PYTHONPATH.")
        sys.exit(1)
    
    # Show configuration
    if args.reconfigure:
        print("=" * 70)
        print("Current Configuration - ICLR2026")
        print("=" * 70)
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
        print("=" * 70)
    
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
    
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    
    if args.dry_run:
        print("\nCommand to execute:")
        print(" ".join(cmd))
        print("\nDry run complete. Remove --dry-run to execute.")
        return
    
    # Set CUDA devices
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CUDA_VISIBLE_DEVICES
    
    # Execute
    print(f"\n{'='*70}")
    print("Starting SEA vLLM Review Generation - ICLR2026")
    print(f"{'='*70}\n")
    
    os.execvp("python", cmd)


if __name__ == "__main__":
    main()
