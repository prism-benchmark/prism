# Configuration Updated ✅

## Changes Made

All vLLM scripts can be configured for the canonical PRISM `papers/` input directory:

### Updated Files

1. **vllm_config.py**
   - INPUT_DIR: `SEA_INPUT_DIR` or `/path/to/DATA_ROOT/ICLR2025/papers`
   - File format: `.grobid.txt`
   - PAPER_IDS_FILE: `SEA_PAPER_IDS_FILE` or `/path/to/data/ICLR2025/paper_ids.txt`
   - OUTPUT_DIR: `SEA_OUTPUT_DIR` or `outputs/sea_reviews`
   - BATCH_SIZE: 16 (increased for faster processing)

2. **run_review_vllm.py**
   - Updated `get_paper_files()` to auto-detect `.grobid.txt` files
   - Handles `.grobid.txt` and `.txt` formats
   - Default argument updated for new paths

3. **Documentation**
   - SETUP_COMPLETE.md: Updated paths and file format info
   - VLLM_GUIDE.md: Updated all examples with new paths

## Quick Start

```bash
cd LLM_reviewer/SEA
python generate_reviews.py
```

Or with bash:
```bash
bash run_vllm.sh
```

## Verification

✅ Input directory: 11,465 papers available
✅ Paper IDs file: 50 papers to process  
✅ Output directory: Ready for output

## Key Features

- ✅ Auto-detects `.grobid.txt` files
- ✅ Falls back to `.txt` if `.grobid.txt` not found
- ✅ Batch processing (16 papers at a time)
- ✅ GPU optimization for fast inference
- ✅ Resume support (skips completed papers)

## Performance Estimate

With batch_size=16 on single GPU:
- Estimated time: ~50 papers × ~2-3 min/paper = **1.5-2.5 hours total**
- With multi-GPU (2x): **~1 hour**
- With multi-GPU (4x): **~30 minutes**

To increase speed, edit vllm_config.py:
```python
BATCH_SIZE = 32              # Increase (needs more VRAM)
TENSOR_PARALLEL_SIZE = 2     # Use 2 GPUs
```

## Ready to Run! 🚀
