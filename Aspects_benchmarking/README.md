# PRISM Aspect Benchmarking

A unified, anonymized benchmark framework for evaluating the quality of AI-generated academic paper reviews across four complementary dimensions: **Depth of Analysis**, **Novelty Assessment**, **Flaw Identification & Prioritization**, and **Multi-Dimensional Constructiveness**.

The fastest path to running the repo is:

1. install dependencies,
2. copy `.env.example` to `.env`,
3. set `DATA_ROOT` and your API key(s),
4. choose the provider/model in `llm_config.yaml`,
5. run `python ../run.py` from `Aspects_benchmarking/`, or `python run.py` from the repo root.

---

## Table of Contents
1. [Repository Structure](#repository-structure)
2. [Dataset](#dataset)
3. [Metrics Overview](#metrics-overview)
4. [Conferences Evaluated](#conferences-evaluated)
5. [Reviewer Types Compared](#reviewer-types-compared)
6. [Setup](#setup)
7. [Running the Pipeline](#running-the-pipeline)
   - [Depth of Analysis](#depth-of-analysis)
   - [Constructiveness](#constructiveness)
   - [Flaw Identification](#flaw-identification)
8. [Release Notes](#release-notes)

---

## Repository Structure

```
Aspects_benchmarking/
├── .env.example                # copy to .env and fill in local paths/keys
├── env_loader.py               # shared env loader used by all aspects
├── requirements.txt
│
├── depth_of_analysis/          # DoA pipeline (ADU segmentation → grounding)
│   ├── config.py               # data paths + model settings (reads .env)
│   ├── evaluate.py             # core evaluation logic (Phase 1–3)
│   ├── evaluate_all.py         # batch evaluation across all conferences
│   ├── calculate_metrics.py    # R_Premise, Avg_GS, DoA_HM computation
│   ├── compare_human_llm.py    # Human vs LLM comparison analysis
│   ├── run_human.py            # any conference — human reviews
│   ├── run_human_icml2025.py   # ICML 2025 human reviews
│   ├── run_human_neurips2025.py    # NeurIPS 2025 human reviews
│   └── run_llm.py              # LLM reviews
│
├── constructiveness/           # multi-dimensional constructiveness pipeline
│   ├── paths_config.py         # conference paths (reads .env via env_loader)
│   ├── run_constructiveness.py # unified evaluator (provider from llm_config.yaml)
│   ├── compute_per_reviewer_metrics.py
│   ├── evaluate_results.py
│   └── src/
│       ├── evaluator.py        # ARC extraction + 5-dimension scoring
│       ├── metrics.py          # MCS, D1–D5 computation
│       ├── statistical.py      # inter-rater agreement, significance tests
│       └── utils.py
│
├── flaw_identification/        # CFI + CPS (nCPS prioritization) pipeline
│   ├── paths_config.py         # conference paths (reads .env via env_loader)
│   ├── main_cfi_iclr2024.py   # ICLR 2024
│   ├── main_cfi_iclr2025.py   # ICLR 2025
│   ├── main_cfi_iclr2026.py   # ICLR 2026
│   ├── main_cfi_icml2025.py   # ICML 2025
│   ├── main_cfi_neurips2025.py  # NeurIPS 2025
│   ├── compute_flaw_metrics.py
│   └── src/
│       ├── evaluator.py
│       ├── cfi_metrics.py      # Critical/Minor Recall
│       ├── cps_metrics.py      # nCPS (NDCG-style)
│       ├── unified_client.py
│       ├── gemini_client.py
│       ├── azure_openai_client.py
│       └── utils.py
│
└── figures/
    ├── combined_6metrics.pdf   # all 6 metrics (paper figure)
    └── combined_5metrics.pdf
```

---

## Dataset

The evaluation dataset is **not included** in this repository. Download it separately and set `DATA_ROOT` in your `.env` (see [Setup](#setup)).

For double-blind review, use an anonymized artifact host and avoid committing private storage URLs or local filesystem paths.

After downloading, the folder must have this layout:

```
DATA_ROOT/
├── ICLR2024/
│   ├── human_reviews/              # one {paper_id}.json per paper
│   ├── sea/                         # {paper_id}.txt
│   ├── tree/                        # {paper_id}_review.json
│   ├── reviewer2/                   # {paper_id}.txt
│   ├── deepreview/                  # {paper_id}.json
│   ├── cyclereview/                 # {paper_id}.json
│   ├── papers/                     # {paper_id}.grobid.txt  (full paper text)
│   ├── paper_ids_200_iclr2024.txt  # all 200 paper IDs
│   └── paper_ids_50_iclr2024.txt   # 50-paper robustness subset
│
├── ICLR2025/    (same sub-folder layout)
├── ICLR2026/    (same sub-folder layout)
├── ICML2025/    (same sub-folder layout)
└── NeurIPS2025/ (same sub-folder layout)
    ├── human_reviews/
    ├── sea/
    ├── tree/
    ├── reviewer2/
    ├── deepreview/
    ├── cyclereview/
    ├── papers/
    ├── paper_ids_200_neurips2025.txt
    └── paper_ids_50_neurips2025.txt
```

---

## Configuration

There are two config files to care about:

- `.env` stores local paths and credentials.
- `llm_config.yaml` selects the provider and model for each aspect or reviewer.

Minimum setup:

```bash
cp .env.example .env
```

Then edit `.env` with at least:

```bash
DATA_ROOT=/absolute/path/to/PRISM/Data/input
GOOGLE_API_KEY=...   # if you use Gemini
OPENAI_API_KEY=...    # if you use OpenAI
MIMO_API_KEY=...      # if you use Xiaomi Mimo
LLM_API_KEY=...       # if you use OpenRouter or another OpenAI-compatible provider
```

In `llm_config.yaml`, set the provider/model you actually want to run. Example:

```yaml
providers:
  together:
    api_key: ${TOGETHER_API_KEY}
    base_url: https://api.together.xyz/v1

aspects:
  constructiveness:
    provider: together
    model: meta-llama/Llama-3.3-70B-Instruct-Turbo
```

Any OpenAI-compatible client works as long as you add it under `providers` and point the aspect/reviewer config at that provider.

---

## Metrics Overview

| Dimension | Metric | Range | Description |
|---|---|---|---|
| **Depth of Analysis** | DoA\_HM | [0, 1] | Harmonic mean of R\_Premise and normalized Avg\_Grounding\_Score |
| **Constructiveness** | MCS | [0, 1] | Mean Constructiveness Score across 5 sub-dimensions (D1–D5) |
| **Flaw Identification** | Critical Recall | [0, 1] | Fraction of critical flaws identified |
| **Flaw Identification** | Minor Recall | [0, 1] | Fraction of minor flaws identified |
| **Prioritization** | nCPS | [0, 1] | NDCG-inspired score measuring critical-before-minor ordering |
| **Novelty Assessment** | NS | [0, 1] | Scored novelty assessment quality |

---

## Conferences Evaluated

| Conference | Papers |
|---|---|
| ICLR 2024 | 200 |
| ICLR 2025 | 200 |
| ICLR 2026 | 200 |
| ICML 2025 | 200 |
| NeurIPS 2025 | 200 |

---

## Reviewer Types Compared

| Type | Description |
|---|---|
| **Human** | Human expert reviewers |
| **SEA** | SEA LLM reviewer |
| **Tree** | TreeReview LLM reviewer |
| **Reviewer2** | Reviewer2 LLM reviewer |
| **DeepReview** | DeepReview LLM reviewer |
| **CycleReview** | CycleReview LLM reviewer |

---

## Quick Start (unified runner)

From the repo root:

```bash
pip install -r requirements.txt
python run.py --setup-data
python run.py --dry-run
python run.py --profile quick
```

Use `python run.py --list` to inspect the active aspects, reviewers, profiles, and provider/model selection.

---

## Setup (per-pipeline)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your `DATA_ROOT` and the API key for the provider you selected in `llm_config.yaml`.

### 3. Verify

```bash
python run.py --list
python run.py --dry-run
python run.py --workers 16 --limit 20
```

The unified runner evaluates papers concurrently (8 workers by default). Set
`PRISM_MAX_WORKERS` in `.env` for a persistent default, or pass `--workers N`
for one run. The runner maps this setting to each aspect's native concurrency
flag, including novelty's Task 1, Task 2, and Task 3 workers.

---

## Running the Pipeline (manual)

### Depth of Analysis

#### LLM Reviews — LLM evaluator

```bash
# Run a single source (conference is encoded in the source name)
python depth_of_analysis/run_llm.py --source sea_iclr2025
python depth_of_analysis/run_llm.py --source reviewer2_iclr2024
python depth_of_analysis/run_llm.py --source deepreview_iclr2026
python depth_of_analysis/run_llm.py --source tree_icml2025
python depth_of_analysis/run_llm.py --source cyclereview_neurips2025
```

Available `--source` identifiers (defined in
`depth_of_analysis/config.py`; these are not directory names):
`sea_iclr2024/2025/2026`, `sea_icml2025`, `sea_neurips2025`,
`tree_iclr2024/2025/2026`, `tree_icml2025`, `tree_neurips2025`,
`reviewer2_iclr2024/2025/2026`, `reviewer2_icml2025`, `reviewer2_neurips2025`,
`deepreview_iclr2024/2025/2026`, `deepreview_icml2025`, `deepreview_neurips2025`,
`cyclereview_iclr2024/2025/2026`, `cyclereview_icml2025`, `cyclereview_neurips2025`

The runner reads the active provider/model from `llm_config.yaml`. Use that
file to switch between Gemini, OpenAI, Mimo, OpenRouter, or any other
OpenAI-compatible client without changing the script.

#### Human Reviews

```bash
# ICLR 2026  (default run_human.py)
python depth_of_analysis/run_human.py

# ICML 2025
python depth_of_analysis/run_human_icml2025.py

# NeurIPS 2025
python depth_of_analysis/run_human_neurips2025.py

# Any conference (provider is read from llm_config.yaml)
python depth_of_analysis/run_human.py --conference ICLR2024
python depth_of_analysis/run_human.py --conference ICLR2025
python depth_of_analysis/run_human.py --conference ICLR2026
python depth_of_analysis/run_human.py --conference ICML2025
python depth_of_analysis/run_human.py --conference NeurIPS2025
```

#### Compute metrics table

```bash
python depth_of_analysis/evaluate_all.py --conference iclr2024
```

---

### Constructiveness

#### LLM Reviews — LLM evaluator

```bash
python constructiveness/run_constructiveness.py --mode reviewer2   --conf iclr2025
python constructiveness/run_constructiveness.py --mode sea         --conf iclr2026
python constructiveness/run_constructiveness.py --mode tree        --conf iclr2024
python constructiveness/run_constructiveness.py --mode deepreview  --conf iclr2025
python constructiveness/run_constructiveness.py --mode cyclereview --conf iclr2024
# Optional: override model
python constructiveness/run_constructiveness.py --mode reviewer2 --conf iclr2025 \
    --provider gemini --model gemini-2.5-flash
```

`--conf` choices: `iclr2024` | `iclr2025` | `iclr2026` | `icml2025` | `neurips2025`

#### Human Reviews — LLM evaluator

```bash
python constructiveness/run_constructiveness.py --mode human --conf iclr2024
python constructiveness/run_constructiveness.py --mode human --conf iclr2025
python constructiveness/run_constructiveness.py --mode human --conf iclr2026
python constructiveness/run_constructiveness.py --mode icml_human
python constructiveness/run_constructiveness.py --mode neurips_human
```

#### Aggregate & compare

```bash
python constructiveness/compute_per_reviewer_metrics.py
python constructiveness/evaluate_results.py
```

---

### Flaw Identification

Three modes per script:

| Mode | Action |
|---|---|
| `cfi_only` | Extract & validate flaws → save JSONL (recommended first run) |
| `all` | `cfi_only` + compute nCPS prioritization metrics |
| `cps_only` | Load cached JSONL, compute nCPS only |

#### LLM Reviews — LLM evaluator

```bash
# ICLR 2024
python flaw_identification/main_cfi_iclr2024.py --mode cfi_only --llm-type reviewer2
python flaw_identification/main_cfi_iclr2024.py --mode all      --llm-type deepreview
python flaw_identification/main_cfi_iclr2024.py --mode cps_only

# ICLR 2025 / 2026
python flaw_identification/main_cfi_iclr2025.py --mode cfi_only --llm-type sea
python flaw_identification/main_cfi_iclr2026.py --mode cfi_only --llm-type tree

# ICML 2025
python flaw_identification/main_cfi_icml2025.py --mode all --llm-type reviewer2

# NeurIPS 2025
python flaw_identification/main_cfi_neurips2025.py --mode cfi_only --llm-type deepreview
```

#### Human Reviews — LLM evaluator

```bash
python flaw_identification/main_cfi_iclr2024.py    --mode cfi_only --llm-type human
python flaw_identification/main_cfi_iclr2025.py    --mode cfi_only --llm-type human
python flaw_identification/main_cfi_iclr2026.py    --mode cfi_only --llm-type human
python flaw_identification/main_cfi_icml2025.py    --mode cfi_only --llm-type human
python flaw_identification/main_cfi_neurips2025.py --mode cfi_only --llm-type human
```

The runner reads the active provider/model from `llm_config.yaml`. Use that
file to switch between Gemini, OpenAI, Mimo, OpenRouter, or any other
OpenAI-compatible client without changing the script.

#### Aggregate & compare

```bash
python flaw_identification/compute_flaw_metrics.py
```

---

## Release Notes

This artifact is intended for anonymous review. Before distribution:

- Keep `.env`, generated outputs, caches, and cluster logs out of version control.
- Replace any private dataset URL with an anonymous artifact link.
- Add the final citation only after the anonymity period, or use the conference-provided anonymous citation format.
