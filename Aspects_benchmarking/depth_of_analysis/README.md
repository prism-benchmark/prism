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
# Gemini evaluator
python run_human.py          # human reviews, all ICLR conferences
python run_llm.py            # LLM reviews, ICLR (sea/tree/reviewer2/deepreview/cyclereview)
python run_human_icml2025.py # human reviews, ICML 2025
python run_human_neurips2025.py

# Mimo evaluator
python run_human_mimo.py
python run_llm_mimo.py
python run_human_icml2025_mimo.py
python run_human_neurips2025_mimo.py

# Analysis
python calculate_metrics.py              # compute DoA metrics from outputs
python compare_gemini_mimo_metrics.py --quiet --save_csv --plot_2
python compare_human_llm.py
```

## Config

Edit `config.py` to set:
- `DATA_ROOT` — path to `Final_LLM_Reviewer_Data/`
- `PAPER_IDS_50` — per-conference 50-paper ID files
- `OUTPUT_DIR` — where results JSON files are written

