# TreeReview batch pipeline (multi-process, baseline-safe)

This package keeps the **TreeReview core logic** inside `treereview/` and adds only an outer batch layer.

## What stays unchanged
- `treereview/core.py`
- `treereview/agents/*`
- `treereview/prompts/*`
- TreeReview generation still uses **only** paper text (`.grobid.txt`).

## What the wrapper adds
- manifest-driven batch execution
- per-paper output folders
- `status.csv`
- per-paper `log.txt`
- `final_output.json` that attaches standardized human reviews without leaking them into generation prompts
- multi-process execution with Windows-safe `spawn`

## Directory structure

```text
project/
  treereview/
  phase_adapter.py
  batch_common.py
  run_one_paper.py
  run_batch_treereview.py
  example_manifest.csv
  requirements.txt
  sample_inputs/
    0A5o6dCKeK.grobid.txt
    0A5o6dCKeK.json
```

## Manifest schema

`manifest.csv` must contain:
- `paper_id`
- `paper_path`
- `reviews_json`
- `output_dir`

Paths may be absolute or relative to the manifest file.

## Single paper

```bash
python run_one_paper.py \
  --paper-id 0A5o6dCKeK \
  --paper-path sample_inputs/0A5o6dCKeK.grobid.txt \
  --reviews-json sample_inputs/0A5o6dCKeK.json \
  --output-dir outputs/0A5o6dCKeK
```

## Batch run

```bash
python run_batch_treereview.py \
  --manifest example_manifest.csv \
  --num-workers 2 \
  --ranker-device cpu
```

## Expected outputs per paper

```text
outputs/<paper_id>/
  final_output.json
  standardized_reviews.json
  checkpoint.json
  log.txt
```

`final_output.json` contains:
- `paper_id`
- `tree_review_result`
- `standardized_reviews`
- `metadata`

## Resume behavior

- If `final_output.json` already exists, that paper is skipped unless `--force-rerun` is used.
- If `checkpoint.json` exists but final output does not, TreeReview resumes through its own checkpoint logic.

## Research note

The wrapper never passes human reviews into TreeReview prompts. Human reviews remain a reference artifact for evaluation and downstream analysis, which preserves a clean baseline.

## Operational note for multiprocessing

The default TreeReview ranker is heavy (`meta-llama/Llama-3.1-8B-Instruct`). On a single GPU, starting many workers can cause OOM. Start with `--num-workers 1` or `2` and scale carefully.
