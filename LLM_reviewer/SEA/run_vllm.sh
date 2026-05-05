#!/bin/bash
# Quick script to run vLLM review generation
# Usage: bash run_vllm.sh

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Import config
source <(python3 << 'EOF'
import vllm_config as cfg
print(f"export MODEL_NAME='{cfg.MODEL_NAME}'")
print(f"export INPUT_DIR='{cfg.INPUT_DIR}'")
print(f"export PAPER_IDS_FILE='{cfg.PAPER_IDS_FILE}'")
print(f"export OUTPUT_DIR='{cfg.OUTPUT_DIR}'")
print(f"export BATCH_SIZE={cfg.BATCH_SIZE}")
print(f"export MAX_TOKENS={cfg.MAX_TOKENS}")
print(f"export TEMPERATURE={cfg.TEMPERATURE}")
print(f"export TOP_P={cfg.TOP_P}")
print(f"export TENSOR_PARALLEL_SIZE={cfg.TENSOR_PARALLEL_SIZE}")
print(f"export GPU_MEMORY_UTILIZATION={cfg.GPU_MEMORY_UTILIZATION}")
print(f"export CUDA_VISIBLE_DEVICES='{cfg.CUDA_VISIBLE_DEVICES}'")
EOF
)

# Set GPU 
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"

echo "=========================================="
echo "vLLM Review Generation Pipeline"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Input: $INPUT_DIR"
echo "Paper IDs: $PAPER_IDS_FILE"
echo "Output: $OUTPUT_DIR"
echo "Batch Size: $BATCH_SIZE"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Run the pipeline
python3 run_review_vllm.py \
    --model "$MODEL_NAME" \
    --input-dir "$INPUT_DIR" \
    --paper-ids "$PAPER_IDS_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --skip-completed

echo "=========================================="
echo "Complete! Output saved to: $OUTPUT_DIR"
echo "=========================================="
