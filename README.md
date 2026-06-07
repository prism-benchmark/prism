# PRISM: Peer Review Intelligence via Structured Multi-dimensional Assessment

Reproduce the paper's experiments in 4 steps.

---

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` — set your API keys (at minimum `GOOGLE_API_KEY` for the Gemini evaluator):

```bash
GOOGLE_API_KEY=AIza...
MIMO_API_KEY=gsk_...            # optional, for robustness checks
```

## 2. Download the data

Downloads papers, human reviews, and **pre-generated LLM reviews** (~4 GB):

```bash
python run.py --setup-data
```

This populates `Data/Final_LLM_Reviewer_Data/` with the full benchmark corpus (1,000 papers across ICLR 2024–2026, ICML 2025, NeurIPS 2025, with human + 5 LLM reviewer outputs).

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
python run.py --only depth_of_analysis              # single aspect
python run.py --conference iclr2024                 # single venue
python run.py --limit 10                            # first N papers (quick test)
python run.py --dry-run                             # preview without executing
```

### Run with a different judge model

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

## 4. Results

Each pipeline writes structured results to its `output/` directory:

| Aspect | Output location | Key metrics |
|--------|----------------|-------------|
| Depth of Analysis | `Aspects_benchmarking/depth_of_analysis/output/metrics/` | DoA, R_premise, S_depth |
| Constructiveness | `Aspects_benchmarking/constructiveness/output/` | MCS, D1–D5 |
| Flaw ID & Prioritization | `Aspects_benchmarking/flaw_identification/output_cfi_*/` | Critical/Minor Recall, nCPS |
| Novelty | `Aspects_benchmarking/novelty_vefification/output/` | NS, SR, SSR |

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
