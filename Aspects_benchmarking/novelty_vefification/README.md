# Novelty Assessment Pipeline

An LLM-powered pipeline for **evidence-grounded novelty assessment** in scientific peer review.

In peer review, novelty denotes the degree to which a paper introduces new or non-trivial differences—such as ideas, methods, data, or perspectives—relative to existing knowledge. A genuine novelty judgment cannot be made in isolation: it requires situating the paper's claimed contributions within the retrievable prior literature. This pipeline operationalizes this principle by evaluating whether a reviewer's novelty comments are verifiably supported or refuted by prior work.

---

## Pipeline Overview

```
Paper + Review → [Extraction] → claims → [Retrieval] → candidates → [Verification] → verdicts
```

The pipeline proceeds in three stages:

| Stage | Function | Input | Output | Time | API |
|:-----:|----------|-------|--------|:----:|-----|
| **Extraction** | Extract claims and anchors | Paper text + Review text | Core task, contribution anchors, key terms, novelty claims C = {c₁, ..., cₙ} | ~15s | LLM |
| **Retrieval** | Retrieve related prior work | Extraction output | Candidate pool B = {b₁, ..., bₖ} (top-30) | ~5s | Semantic Scholar |
| **Verification** | Verify novelty claims | Extraction + Retrieval + Paper context | Evidence-support score s(cᵢ, bⱼ) ∈ {−2, −1, 0, +1, +2} | ~30s | LLM |

### Extraction (Task 1)

A constrained LLM extracts structured information from the paper and review:

- **From paper**: core task (1 sentence), 1–3 contribution anchors, key terms, must-have entities
- **From review**: verbatim novelty claims C = {c₁, ..., cₙ} with stance labels (`novel` / `somewhat_novel` / `not_novel`), confidence, and cited prior work

### Retrieval (Task 2)

Deterministic Semantic Scholar queries are constructed from the extracted anchors:

- Queries built from contribution claims and key terms
- Results deduplicated, filtered to prior publications, and diversified via Maximal Marginal Relevance (MMR)
- Produces a candidate pool B = {b₁, ..., bₖ} with metadata (title, abstract, year, venue)

### Verification (Task 3)

For each pair (cᵢ, bⱼ), an LLM judge compares the review claim against:

- **Paper context**: abstract + introduction of the paper being reviewed
- **Candidate prior work**: title + abstract of the related paper

The judge returns a discrete evidence-support score:

| Score | Label | Meaning |
|:-----:|-------|---------|
| +2 | SUPPORTED | Review claim aligns with and is supported by the related work evidence |
| +1 | OVERSTATED | Claim is somewhat supported but reviewer overstated the strength |
| 0 | AMBIGUOUS | Evidence is inconclusive or insufficient to judge |
| −1 | UNDERSTATED | Reviewer understated the novelty given the evidence |
| −2 | UNSUPPORTED | Evidence contradicts the reviewer claim or no supporting evidence found |

Per-claim verdicts are aggregated using a configurable policy (max / mean / weighted) to produce a final score per claim.

---

## Installation

```bash
git clone <repo-url>
cd novelty_vefification

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Configuration

All configuration is via environment variables. Create a `.env` file in the project root:

```bash
cp .env_example .env
# Edit .env with your API keys
```

### Required: LLM API

The pipeline requires an OpenAI-compatible LLM API for Extraction (Task 1) and Verification (Task 3):

```bash
export LLM_PROVIDER="openai"                              # openai | openrouter | azure
export LLM_API_KEY="your-api-key"
export LLM_API_ENDPOINT="https://api.openai.com/v1"       # or your provider's endpoint
export LLM_MODEL_NAME="gpt-4o"                            # model name
```

Examples of compatible providers:
- **OpenAI**: `https://api.openai.com/v1` with `gpt-4o`
- **OpenRouter**: `https://openrouter.ai/api/v1` with `anthropic/claude-sonnet-4.5`
- **vLLM local**: `http://localhost:8080/v1` with any served model
- **Google**: `https://generativelanguage.googleapis.com/v1beta/openai/` with `gemini-2.5-flash-lite`

### Optional: Semantic Scholar API

Retrieval (Task 2) uses the public Semantic Scholar API (no key needed, ~1 req/s). For higher throughput:

```bash
export SEMANTIC_SCHOLAR_API_KEYS="key1,key2,key3"
```

### Full Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | LLM provider (`openai`, `openrouter`, `azure`) |
| `LLM_API_KEY` | — | API key (required) |
| `LLM_API_ENDPOINT` | `https://api.openai.com/v1` | API endpoint URL |
| `LLM_MODEL_NAME` | `gpt-4o` | Model name |
| `LLM_MAX_TOKENS` | `64000` | Max output tokens per request |
| `SEMANTIC_SCHOLAR_API_KEYS` | — | Comma-separated API keys for higher rate limits |
| `API_TIMEOUT` | `120` | HTTP request timeout (seconds) |
| `MAX_RETRIES` | `30` | Max retry attempts per API call |
| `RETRY_DELAY` | `5` | Delay between retries (seconds) |

---

## Data Directory Structure

The top-level PRISM runner passes the normalized `DATA_ROOT` used by all
aspects:

```
DATA_ROOT/
├── ICLR2024/
│   ├── papers/                   # {paper_id}.txt or {paper_id}.grobid.txt
│   ├── human_reviews/            # {paper_id}.json
│   ├── sea/
│   ├── tree/
│   ├── reviewer2/
│   ├── deepreview/
│   └── cyclereview/
├── ICLR2025/
├── ICLR2026/
├── ICML2025/
└── NeurIPS2025/
```

The original Hugging Face artifact uses folders such as `ICLR_2024/json/`
and `ICLR_2024/grobid_fulltext/`. Run `python run.py --setup-data` or
`Data/map_hf_to_aspect_layout.py` before using that raw format.

Each normalized conference directory contains:
- `papers/` — paper full texts, one file per paper
- `human_reviews/` — peer-review JSON, matching paper IDs
- one plain directory per generated reviewer: `sea/`, `tree/`, `reviewer2/`,
  `deepreview/`, and `cyclereview/`

### Review JSON Format

Review JSONs should contain a `reviews` array with reviewer text:

```json
{
  "reviews": [
    {
      "Summary": "...",
      "Strengths": "...",
      "Weaknesses": "...",
      "Questions": "...",
      "Soundness": "...",
      "Presentation": "..."
    }
  ]
}
```

---

## Usage

### Run Full Pipeline — Single Paper

```bash
python scripts/run_pipeline.py \
  --paper path/to/paper.txt \
  --review path/to/review.txt \
  -o output/demo
```

Output:
```
output/demo/
├── task1_result.json   # Extraction: core task, contributions, novelty claims
├── task2_result.json   # Retrieval: candidate pool B = {b₁, ..., bₖ}
└── task3_result.json   # Verification: evidence-support scores per (cᵢ, bⱼ)
```

### Run Full Pipeline — Batch

```bash
# Auto-detects layout, runs all conferences
python scripts/run_pipeline.py \
  --data-root /path/to/paper_data \
  -o output/full_run

# Specific conferences, limit papers per conference
python scripts/run_pipeline.py \
  --data-root /path/to/paper_data \
  --conferences ICLR_2024 ICLR_2025 \
  --max-papers 50 \
  -o output/full_run

# Custom LLM (overrides .env)
python scripts/run_pipeline.py \
  --data-root /path/to/paper_data \
  --llm-provider openai --llm-model-name gpt-4o --llm-api-key "$OPENAI_API_KEY" \
  -o output/full_run
```

Output structure:
```
output/full_run/
├── _pipeline_summary.json
└── human/
    ├── ICLR_2024/
    │   ├── paper_id_1/
    │   │   ├── task1_result.json
    │   │   ├── task2_result.json
    │   │   └── task3_result.json
    │   └── paper_id_2/
    │       └── ...
    ├── ICLR_2025/
    │   └── ...
    ├── ICLR_2026/
    │   └── ...
    ├── ICML_2025/
    │   └── ...
    └── NeurIPS_2025/
        └── ...
```

### Run Individual Stages

```bash
# Extraction only
python scripts/run_task1.py --paper paper.txt --review review.txt -o task1.json

# Retrieval only (from Extraction output)
python scripts/run_task2.py --task1 task1.json -o task2.json

# Verification only (from Extraction + Retrieval output)
python scripts/run_task3.py --task1 task1.json --task2 task2.json --paper paper.txt -o task3.json
```

### Evaluate Results

```bash
# Summary statistics (no annotations needed)
python scripts/evaluate.py --run-dir output/full_run/human/ICLR_2024

# Compare two runs
python scripts/evaluate.py \
  --run-dir output/run_a --run-dir2 output/run_b \
  --names "Model-A" "Model-B" \
  -o output/eval_report.json

# Export as CSV
python scripts/evaluate.py --run-dir output/full_run/human/ICLR_2024 --format csv -o report.csv
```

### Visualize Results

```bash
# Generate publication-quality figures
python scripts/visualize.py --run-dir output/full_run/human/ICLR_2024 -o output/figures/
```

Generated figures:
- `fig_conference_distribution.png` — papers per conference
- `fig_stance_distribution.png` — novelty stance breakdown
- `fig_verdict_distribution.png` — evidence-support score distribution
- `fig_contribution_histogram.png` — contribution anchors per paper
- `fig_candidate_histogram.png` — candidate pool size per paper
- `fig_stance_by_conference.png` — stances grouped by conference
- `fig_verdict_by_conference.png` — verdicts grouped by conference
- `fig_pipeline_success.png` — pipeline success rate per conference

---

## Output Format Reference

### Extraction (`task1_result.json`)

```json
{
  "paper": {
    "core_task": "One-sentence description of the paper's main task",
    "contributions": ["Contribution claim 1", "Contribution claim 2"],
    "key_terms": ["term1", "term2"],
    "must_have_entities": ["Entity1", "Entity2"]
  },
  "review": {
    "novelty_claims": [
      {
        "claim_id": "C1",
        "text": "Verbatim review sentence about novelty",
        "stance": "not_novel",
        "confidence_lang": "high",
        "mentions_prior_work": true,
        "prior_work_strings": ["Smith et al. (2023)"]
      }
    ],
    "all_citations_raw": ["Smith et al. (2023)", "Zhang et al. (2023)"]
  }
}
```

### Retrieval (`task2_result.json`)

```json
{
  "candidate_pool_top30": [
    {
      "cand_id": "paper_hash_id",
      "title": "Related Paper Title",
      "abstract": "...",
      "year": 2023,
      "venue": "NeurIPS",
      "relevance_score": 0.85
    }
  ],
  "queries": ["query1", "query2"],
  "stats": { "total_candidates": 30 }
}
```

### Verification (`task3_result.json`)

```json
{
  "aggregated": [
    {
      "review_sentence_id": "C1",
      "text": "Review claim text",
      "classification": { "claim": 1, "proof": 0 },
      "evidence_results": [
        {
          "related_paper_id": "paper_hash_id",
          "score": 2,
          "label": "SUPPORTED",
          "explanation": "Short explanation of the verdict"
        }
      ],
      "final_score": 2.0,
      "best_evidence": ["paper_hash_id"],
      "best_evidence_policy": "max"
    }
  ],
  "pair_results": [...],
  "stats": {
    "review_sentences": 3,
    "related_works": 10,
    "pairs_attempted": 30,
    "pairs_completed": 30,
    "pairs_failed": 0
  }
}
```

---

## Command Reference

| Script | Description |
|--------|-------------|
| `scripts/run_pipeline.py` | Run full pipeline (Extraction → Retrieval → Verification) |
| `scripts/run_task1.py` | Run Extraction only (single paper) |
| `scripts/run_task2.py` | Run Retrieval only (from Extraction JSON) |
| `scripts/run_task3.py` | Run Verification only (from Extraction + Retrieval JSON) |
| `scripts/run_task1_batch.py` | Run Extraction in batch mode |
| `scripts/evaluate.py` | Evaluate pipeline results |
| `scripts/visualize.py` | Generate figures from pipeline results |
| `scripts/benchmark_evaluation.py` | Benchmark metrics with human annotations |
| `scripts/visualize_benchmarks.py` | Full benchmark visualization suite |

---

## Troubleshooting

| Issue | Symptom | Solution |
|-------|---------|----------|
| LLM API failure | `API Error` / `Invalid key` | Check `LLM_API_KEY`, `LLM_MODEL_NAME`, and quota |
| Semantic Scholar rate limit | `429 Too Many Requests` | Add multiple keys to `SEMANTIC_SCHOLAR_API_KEYS` |
| No papers found | `No papers found. Check --data-root` | Verify directory structure matches Format A or B |
| Empty verdicts | `note: no novelty claims` | Extraction produced 0 claims — check review text quality |
| Import errors | `ModuleNotFoundError` | Run from project root, `pip install -r requirements.txt` |

---

## Project Structure

```
.
├── task1_extractor.py               # Extraction: claim extraction (LLM)
├── task2_related_works.py           # Retrieval: related-works retrieval (Semantic Scholar)
├── task3_judge.py                   # Verification: novelty verification (LLM Judge)
├── config.py                        # Configuration (reads .env)
├── services/
│   ├── llm_client.py                # LLM API client
│   └── semantic_scholar_client.py   # Semantic Scholar API client
├── utils/
│   ├── text_cleaning.py             # Text preprocessing
│   └── paper_id.py                  # Canonical paper ID generation
├── scripts/                         # Entry-point scripts
│   ├── run_pipeline.py              # Full pipeline runner
│   ├── evaluate.py                  # Evaluation script
│   └── visualize.py                 # Visualization script
├── .env                             # API keys (not committed)
├── .env_example                     # Configuration template
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## Citation

```bibtex
@article{zhang2026opennovelty,
  title={OpenNovelty: An LLM-powered Agentic System for Verifiable Scholarly Novelty Assessment},
  author={Zhang, Ming and Tan, Kexin and Huang, Yueyuan and Shen, Yujiong and Ma, Chunchun and Ju, Li and Zhang, Xinran and Wang, Yuhui and Jing, Wenqing and Deng, Jingyi and others},
  journal={arXiv preprint arXiv:2601.01576},
  year={2026}
}
```

---

## License

Apache License 2.0. See [LICENSE](LICENSE).
