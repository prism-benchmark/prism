# Depth of Analysis (DoA) Pipeline

Evaluates the argumentative depth of reviews through a 3-phase pipeline:

1. **Phase 1 — ADU Segmentation**: Splits review text into Argumentative Discourse Units
2. **Phase 2 — Role & Aspect Classification**: Labels each ADU as `Premise` or `Non-Premise` with aspect tags
3. **Phase 3 — Grounding Score**: Scores each Premise as `0` (ungrounded), `1` (partially grounded), or `2` (fully grounded)

## Metrics

| Metric | Formula | Range |
|---|---|---|
| R_Premise | #Premises / #Total_ADUs | [0, 1] |
| Avg_GS | mean(grounding_score) for Premises | [0, 2] |
| DoA_HM | 2 × (R_Premise × Avg_GS/2) / (R_Premise + Avg_GS/2) | [0, 1] |

## Run

```bash
# Runners (provider/model read from llm_config.yaml)
python run_human.py          # human reviews, all ICLR conferences
python run_llm.py            # LLM reviews, ICLR (sea/tree/reviewer2/deepreview/cyclereview)
python run_human_icml2025.py # human reviews, ICML 2025
python run_human_neurips2025.py

# Analysis
python calculate_metrics.py              # compute DoA metrics from outputs
python compare_human_llm.py
```

## Config

Set `DATA_ROOT` in the repository root `.env`. Each conference directory must
contain `human_reviews/`, `papers/`, `sea/`, `tree/`, `reviewer2/`,
`deepreview/`, and `cyclereview/`. Optional 50-paper manifests use
`paper_ids_50_<conference>.txt`, for example
`paper_ids_50_neurips2025.txt`.

`config.py` derives all input paths from `DATA_ROOT`; `OUTPUT_DIR` controls
where result JSON files are written.
