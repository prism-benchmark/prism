# SEA Baseline (Anonymous Artifact Version)

This directory contains the SEA-based reviewer generation baseline used in the
PRISM benchmark artifact. The content here is scoped to reproducible local
execution for benchmarking and intentionally excludes non-essential project
metadata.

## Scope

- Generate review drafts from paper text inputs.
- Run batched inference for benchmark splits.
- Export outputs in the format expected by downstream evaluation scripts.

## Quick Start

1. Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

2. Configure runtime settings:

- Edit `vllm_config.py` (or conference-specific variants).
- Set `SEA_DATA_ROOT`, `INPUT_DIR`, `PAPER_IDS_FILE`, and `OUTPUT_DIR`.
- Optionally set `CUDA_VISIBLE_DEVICES` in the environment.

3. Run generation:

```bash
python generate_reviews.py
```

For conference-specific runs, use the corresponding entrypoint such as
`generate_reviews_iclr2026.py`.

## Input/Output Contract

- **Input**: paper text files and optional paper-id subsets.
- **Output**: one generated review file per paper id under the configured
  output directory.

These outputs are consumed by scripts in `Aspects_benchmarking/`.

## Anonymity and Release Hygiene

- Do not add names, emails, institutions, personal URLs, or private hostnames.
- Keep secrets in environment variables or local `.env` files (never commit).
- Avoid machine-specific absolute paths in committed configs and docs.

