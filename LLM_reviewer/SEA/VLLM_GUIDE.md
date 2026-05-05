# vLLM Review Generation Guide

## Overview
This is an optimized review generation pipeline using **vLLM** for fast inference. vLLM is 10-20x faster than standard transformers inference and supports batch processing and tensor parallelism.

## Setup

### Install vLLM
```bash
pip install vllm
# or update requirements.txt:
# vllm>=0.3.0
```

### Configuration
Edit `vllm_config.py` to customize:
- **MODEL_NAME**: HuggingFace model (default: Qwen/Qwen2-7B-Instruct)
- **INPUT_DIR**: Path to text files (.mmd or .grobid.txt)
- **PAPER_IDS_FILE**: File with paper IDs to process
- **OUTPUT_DIR**: Output directory for reviews
- **BATCH_SIZE**: Increase for faster processing (needs more VRAM)
- **TENSOR_PARALLEL_SIZE**: Number of GPUs (1 for single GPU, 2+ for multi-GPU)
- **GPU_MEMORY_UTILIZATION**: 0.85-0.95 recommended
- **CUDA_VISIBLE_DEVICES**: GPU IDs to use

## Usage

### Quick Run (with Bash)
```bash
cd /home/duy.na/ongoing_projects/SEA
bash run_vllm.sh
```

### Advanced Run (Direct Python)
```bash
python run_review_vllm.py \
    --model "Qwen/Qwen2-7B-Instruct" \
    --input-dir "/mnt/duyna/review_assessment/data/ICLR2025/grobid_fulltext" \
    --paper-ids "/mnt/duyna/review_assessment/paper_ids_specific.txt" \
    --output-dir "/mnt/duyna/review_assessment/sea_output_new" \
    --batch-size 8 \
    --tensor-parallel-size 1 \
    --skip-completed
```

### Multi-GPU Usage
For faster processing with 2+ GPUs:

```bash
# Edit vllm_config.py:
TENSOR_PARALLEL_SIZE = 2  # or more
BATCH_SIZE = 16           # increase batch size

# Then run
bash run_vllm.sh
```

## Performance Tips

### 1. Increase Batch Size
- Single GPU: batch_size=4-8
- Multi-GPU (2x): batch_size=16-24
- Multi-GPU (4x): batch_size=32-64
- Monitor VRAM with `nvidia-smi`

### 2. Optimize Memory
```python
# Increase GPU utilization
GPU_MEMORY_UTILIZATION = 0.95

# Reduce max tokens if needed
MAX_TOKENS = 4096  # instead of 8192
```

### 3. Use Smaller Models
If running on single GPU with limited VRAM:
```python
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"      # ~15GB
# or
MODEL_NAME = "Qwen/Qwen2-7B-Instruct"             # ~15GB
```

### 4. Skip Already Processed
```bash
python run_review_vllm.py ... --skip-completed
```
This speeds up resuming interrupted runs.

## Speed Comparison

| Framework | Speed | Batch | Notes |
|-----------|-------|-------|-------|
| vLLM | 10-20x faster | Yes | Recommended |
| Transformers | 1x baseline | Limited | Original |
| DeepSpeed | ~5x faster | Yes | Complex setup |

## Troubleshooting

### CUDA Out of Memory
```python
# Option 1: Reduce batch size
BATCH_SIZE = 4

# Option 2: Reduce max tokens
MAX_TOKENS = 4096

# Option 3: Use smaller model
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

# Option 4: Reduce GPU utilization
GPU_MEMORY_UTILIZATION = 0.8
```

### Slow Processing
```python
# Option 1: Increase batch size (if VRAM allows)
BATCH_SIZE = 16

# Option 2: Use multi-GPU
TENSOR_PARALLEL_SIZE = 2

# Option 3: Use faster model
MODEL_NAME = "Qwen/Qwen2-7B-Instruct"  # Generally faster than Llama-2
```

### File Paths Not Found
- Check INPUT_DIR exists with `.mmd` files
- Verify PAPER_IDS_FILE is readable
- Create OUTPUT_DIR if it doesn't exist

## Output Format

Each review is saved as `{paper_id}.txt` in OUTPUT_DIR with format:
```
**Summary:**
[summary content]

**Strengths:**
- [strength 1]
- [strength 2]

**Weaknesses:**
- [weakness 1]
- [weakness 2]

**Questions:**
- [question 1]

**Soundness:**
[rating]

**Presentation:**
[rating]

**Contribution:**
[rating]

**Rating:**
[rating]

**Paper Decision:**
- Decision: Accept/Reject
- Reasons: [reasons]
```

## Example Commands

### Process specific papers (default config)
```bash
bash run_vllm.sh
```

### Process all papers in directory
```bash
python run_review_vllm.py \
    --model "Qwen/Qwen2-7B-Instruct" \
    --input-dir "/mnt/duyna/review_assessment/data/ICLR2025/grobid_fulltext" \
    --output-dir "/mnt/duyna/review_assessment/sea_output_new" \
    --no-skip-completed
```

### Resume interrupted run
```bash
bash run_vllm.sh  # Automatically skips completed papers
```

### Use different model
```bash
python run_review_vllm.py \
    --model "mistralai/Mistral-7B-Instruct-v0.1" \
    --input-dir "/mnt/duyna/review_assessment/data/ICLR2025/grobid_fulltext" \
    --output-dir "/mnt/duyna/review_assessment/sea_output_new"
```

## Model Recommendations

### For Quality (slower, more VRAM)
- `meta-llama/Llama-2-13b-chat-hf` (~26GB)
- `Qwen/Qwen2-72B-Instruct` (~144GB, multi-GPU only)

### For Balance (recommended)
- `Qwen/Qwen2-7B-Instruct` (~15GB) ✅
- `meta-llama/Llama-2-7b-chat-hf` (~15GB)

### For Speed (less VRAM, may see slight quality loss)
- `Qwen/Qwen2-1B-Instruct` (~2GB)
- `HuggingFaceH4/zephyr-7b-beta` (~15GB)

## Next Steps

1. Edit `vllm_config.py` with your settings
2. Run `bash run_vllm.sh`
3. Monitor GPU usage with `watch -n 1 nvidia-smi`
4. Check output reviews in OUTPUT_DIR
