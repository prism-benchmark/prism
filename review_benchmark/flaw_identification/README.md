# Flaw Identification & Prioritization Pipeline

Evaluates whether reviews correctly identify and prioritize paper flaws.

## Pipeline (2 steps LLM + metrics)

1. **Step 1 — Flaw Atomization**: Extracts micro-flaws from human and LLM reviews, groups them into a canonical issue bank (Critical / Minor)
2. **Step 2 — Validation**: Judges each flaw against paper content (valid / invalid / partially valid)
3. **Metrics**: Computes Critical Recall, Minor Recall, and nCPS

## Metrics

| Metric | Description |
|---|---|
| **Critical Recall** | Fraction of critical flaws from issue bank identified by reviewer |
| **Minor Recall** | Fraction of minor flaws from issue bank identified by reviewer |
| **nCPS** | Normalized Coverage-Priority Score — NDCG-style measure of whether critical flaws are ranked before minor flaws |

**nCPS formula**: `CPS = Σ w_i / log2(pos_i + 1)`, normalized by ideal `iCPS`  
where `w_i = 2` for Critical, `w_i = 1` for Minor

## Run

```bash
# ── Gemini evaluator ──────────────────────────────────────────────────
# Step 1+2 only (CFI metrics)
python main_cfi_iclr2024.py --mode cfi_only --llm-type reviewer2

# Step 1+2 + CPS
python main_cfi_iclr2024.py --mode all --llm-type deepreview

# CPS from cached results (fast, no re-extraction)
python main_cfi_iclr2024.py --mode cps_only --llm-type sea

# ── Mimo evaluator ────────────────────────────────────────────────────
python main_cfi_iclr2024_mimo.py --mode cfi_only --llm-type reviewer2
python main_cfi_iclr2024_mimo.py --mode cps_only --llm-type reviewer2

# ── Other conferences (same pattern) ─────────────────────────────────
python main_cfi_iclr2025.py --mode all --llm-type sea
python main_cfi_icml2025.py --mode cfi_only --llm-type tree
python main_cfi_neurips2025.py --mode all --llm-type cyclereview

# ── Aggregate analysis ────────────────────────────────────────────────
python compute_flaw_metrics.py              # Gemini aggregate
python compute_flaw_mimo_vs_gemini.py       # Gemini vs Mimo comparison + charts
```

## LLM Types

| Flag | Source folder |
|---|---|
| `sea` | `sea_{conf}/` |
| `tree` | `tree_{conf}/` |
| `reviewer2` | `reviewer2_{conf}/` |
| `deepreview` | `deepreview_{conf}/` |
| `cyclereview` | `cyclereview_{conf}/` |

## Environment Variables

```bash
GOOGLE_API_KEY=...          # Gemini evaluator
MIMO_API_KEY=...            # Mimo evaluator
AZURE_OPENAI_API_KEY=...    # Azure evaluator (optional)
```

