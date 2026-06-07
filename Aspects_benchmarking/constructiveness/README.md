# Constructiveness Pipeline

Evaluates multi-dimensional constructiveness of reviews via a 2-phase pipeline:

1. **Phase 1 — ARC Extraction**: Extracts Actionable Review Comments
2. **Phase 2 — Dimension Scoring**: Scores each ARC on 5 dimensions [0, 2]

## Dimensions (D1–D5)

| Dim | Name | Description |
|---|---|---|
| D1 | Actionability | Does the comment suggest concrete improvements? |
| D2 | Specificity | Is the comment specific to the paper? |
| D3 | Justification | Is the comment backed by reasoning? |
| D4 | Solution Orientation | Does it propose solutions, not just problems? |
| D5 | Tone | Is the tone constructive and professional? |

**MCS** = mean(Σ D_k / 10) across all ARCs

## Run

```bash
# Provider/model is read from llm_config.yaml — switch evaluators there.
python run_constructiveness.py --mode reviewer2 --conf iclr2025
python run_constructiveness.py --mode human     --conf icml2025
python run_constructiveness.py --mode deepreview --conf neurips2025
python run_constructiveness.py --mode sea --conf iclr2024

# Analysis
python compute_per_reviewer_metrics.py    # Per-reviewer breakdown
```

## Modes

| Mode | Source |
|---|---|
| `human` | Human reviewer JSONs |
| `sea` | SEA `.txt` reviews |
| `tree` | TreeReview JSONs |
| `reviewer2` | Reviewer2 `.txt` reviews |
| `deepreview` | DeepReview JSONs |
| `cyclereview` | CycleReview JSONs |

