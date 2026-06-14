```bash
# TLDR — reproduce all paper results in 4 commands:
pip install -r requirements.txt          # 1. install deps
python run.py --setup-data               # 2. download, extract, and configure data
cp .env.example .env                     # 3. only if .env was not created
python run.py                            # 4. run all evaluations
```

> See [CONFIG_GUIDE.md](CONFIG_GUIDE.md) for detailed setup, custom providers, and per-aspect configuration options. `llm_config.yaml` is the source of truth for provider/model selection.

---

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` — set `DATA_ROOT` and the API key for whichever provider you select in `llm_config.yaml`:

```bash
DATA_ROOT=/absolute/path/to/PRISM/Data/input
GOOGLE_API_KEY=AIza...
OPENAI_API_KEY=sk-...
MIMO_API_KEY=gsk_...
LLM_API_KEY=sk-or-openrouter-key...
```

`DATA_ROOT` uses one canonical layout for every pipeline:

```text
Data/input/
├── ICLR2024/
│   ├── papers/          # {paper_id}.txt
│   ├── human_reviews/   # {paper_id}.json
│   ├── sea/             # {paper_id}.txt
│   ├── reviewer2/       # {paper_id}.txt
│   ├── tree/            # {paper_id}_review.json
│   ├── deepreview/      # {paper_id}.json
│   └── cyclereview/     # {paper_id}.json
├── ICLR2025/
├── ICLR2026/
├── ICML2025/
└── NeurIPS2025/
```

Every conference uses the same reviewer subdirectory names. Paper-ID
manifests are optional; pipelines derive IDs from `papers/` when absent.

## 2. Download the data

Downloads `demo_data.zip` from Hugging Face, safely extracts it into
`Data/input/`, validates the canonical layout, and writes `DATA_ROOT` to the
root `.env`:

```bash
python run.py --setup-data
```

The source archive is:
[`anoyresearcher/prism_paper_data/demo_data.zip`](https://huggingface.co/datasets/anoyresearcher/prism_paper_data/blob/main/demo_data.zip).

Re-running setup uses an existing validated `Data/input/`. Use
`python Data/setup_aspect_benchmark.py --force` only when replacing it.

> No GPU needed — LLM reviews are pre-generated. You only need API access for the judge model.

## 3. Run the paper experiment

```bash
# Run all 4 evaluation aspects across all 5 conferences (default)
python run.py
```

This evaluates **Depth of Analysis**, **Novelty Assessment**, **Flaw Identification & Prioritization**, and **Constructiveness** for all reviewer types (Human, SEA, Reviewer2, TreeReview, DeepReview, CycleReview) across all conferences.

### Selective runs (faster iteration)

```bash
python run.py --list                                # see available aspects/reviewers
python run.py --profile aspects                     # evaluation only
python run.py --profile all_local                   # all aspects + local reviewers
python run.py --only depth_of_analysis              # single aspect
python run.py --conference iclr2024                 # single venue
python run.py --limit 10                            # first N papers (quick test)
python run.py --workers 16                          # evaluate 16 papers concurrently
python run.py --output-dir /path/to/results         # custom central output root
python run.py --dry-run                             # preview without executing
```

Evaluations run with 8 concurrent paper workers by default. Override this with
`--workers N` or set `PRISM_MAX_WORKERS=N` in `.env`. Use a smaller value when
your provider returns HTTP 429/rate-limit errors. Existing per-paper outputs are
used as checkpoints, so interrupted runs resume without recomputing completed
papers.

### Run with a different judge model or provider

Edit `llm_config.yaml` to switch any aspect to a different provider:

```yaml
aspects:
  constructiveness:
    provider: openai          # was: mimo
    model: gpt-4o
```

Or use any OpenAI-compatible API:

```yaml
providers:
  together:
    api_key: ${TOGETHER_API_KEY}
    base_url: https://api.together.xyz/v1

aspects:
  depth_of_analysis:
    provider: together
    model: meta-llama/Llama-3.3-70B-Instruct-Turbo
```

```bash
# .env
TOGETHER_API_KEY=tsk_...
```

The same pattern applies to any OpenAI-compatible client: add a provider entry once, then point the aspect or reviewer at that provider.

## 4. Results

Runs launched through `run.py` write all structured results under the central
`output/` directory. Use `--output-dir` to select a different root.

| Aspect | Output location | Key metrics |
|--------|----------------|-------------|
| Depth of Analysis | `output/depth_of_analysis/` | DoA, R_premise, S_depth |
| Constructiveness | `output/constructiveness/` | MCS, D1-D5 |
| Flaw ID & Prioritization | `output/flaw_identification/` | Critical/Minor Recall, nCPS |
| Novelty | `output/novelty/` | NS, SR, SSR |

Aggregated summary CSVs correspond to Table 1 in the paper.

## What gets evaluated

| Venue | Papers | Decisions sampled |
|-------|--------|-------------------|
| ICLR 2024 | 200 | Oral, Spotlight, Poster, Reject |
| ICLR 2025 | 200 | Oral, Spotlight, Poster, Reject |
| ICLR 2026 | 200 | Oral, Poster, Reject |
| ICML 2025 | 200 | Oral, Spotlight, Poster, Reject |
| NeurIPS 2025 | 200 | Oral, Spotlight, Poster, Reject |

| Reviewer system | Type |
|-----------------|------|
| Human | Expert peer reviewers |
| SEA | Supervised fine-tuning |
| Reviewer2 | Prompting-based |
| TreeReview | Prompting-based |
| DeepReview | Supervised fine-tuning |
| CycleReview | Supervised fine-tuning |

## Repository layout

```
PRISM/
├── run.py                       ← unified orchestrator (this is all you need)
├── llm_config.yaml              ← model/provider configuration
├── llm_client.py                ← unified LLM client
├── .env                         ← API keys & paths (gitignored)
├── Data/                        ← download scripts
├── Aspects_benchmarking/        ← evaluation pipelines (RQ1–RQ4)
│   ├── depth_of_analysis/
│   ├── novelty_vefification/
│   ├── flaw_identification/
│   └── constructiveness/
└── LLM_reviewer/                ← reviewer generation baselines (not needed for eval)
```

## Dependencies

The evaluation stack is lightweight:

```
google-generativeai  openai  httpx  numpy  scipy  matplotlib  python-dotenv
```

Heavier dependencies (vLLM, transformers) are only needed if re-generating LLM reviews — skip unless you're rerunning the generation baselines.
