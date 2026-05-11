# PRISM

PRISM is an anonymous research artifact for evaluating LLM-generated peer reviews against human expert reviews. The repository contains two complementary parts:

- `Aspects_benchmarking/`: evaluation pipelines for Depth of Analysis, Novelty Assessment, Flaw Identification and Prioritization, and Multi-Dimensional Constructiveness.
- `LLM_reviewer/`: reviewer-generation baselines and adapters used to produce LLM review outputs.

This repository is prepared for double-blind review. Do not add author names, personal repository URLs, institution-specific paths, or local machine paths to code, documentation, or configuration files.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 1. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` and fill in your API keys (GOOGLE_API_KEY, OPENAI_API_KEY, etc.).
Set `DATA_ROOT` to the downloaded dataset root.

### 2. Choose models per aspect / reviewer

Edit `llm_config.yaml` at the repo root. This is the **single place** to configure
which LLM is used for each benchmarking aspect and each LLM reviewer:

```yaml
# Default (used when not overridden)
defaults:
  provider: openai
  model: gpt-4o-mini

# Per-aspect model selection (Aspects_benchmarking evaluators)
aspects:
  constructiveness:
    provider: devmate
    model: gemini-3-flash-preview
  flaw_identification:
    step1:
      provider: devmate
      model: gemini-3-flash-preview
    step2:
      provider: devmate
      model: gemini-3-flash-preview
  depth_of_analysis:
    provider: gemini
    model: gemini-2.5-flash-lite
  novelty:
    provider: openai
    model: gpt-4o

# Per-reviewer model settings (LLM_reviewer generation)
reviewers:
  sea:
    model_path: Qwen/Qwen2-7B-Instruct
  reviewer2:
    model_path: /path/to/Qwen3-14B
  treereview:
    provider: gemini
    model: gemini-2.0-flash
  deepreview:
    model_size: "14B"
  cyclereview:
    model_size: "8B"
```

### 3. Use the unified client in code

```python
from ai_config import get_llm_client

# Client for a specific benchmarking aspect (reads from llm_config.yaml)
client = get_llm_client("constructiveness")
result = client.generate_text("You are a judge.", "Score this review...")

# Client for a specific step
client = get_llm_client("flaw_identification", step="step1")

# Client for a specific reviewer
client = get_llm_client("treereview")

# Override model at call time
client = get_llm_client("novelty", model="gpt-4o")
```

## Data

The evaluation dataset is not stored in this repository. The expected dataset layout and per-dimension commands are documented in `Aspects_benchmarking/README.md`.

For anonymous review, distribute data through an anonymized artifact link and keep any private or non-anonymous storage locations out of this repository.

## Repository Layout

```text
PRISM/
├── ai_config.py            # centralized AI/LLM configuration (API keys, defaults)
├── llm_client.py           # unified LLM client (PRISMLLMClient)
├── llm_config.yaml         # per-aspect and per-reviewer model selection
├── .env.example            # API key template (copy to .env)
├── Aspects_benchmarking/   # metric pipelines and aggregation scripts
├── LLM_reviewer/           # LLM reviewer baselines and generation scripts
├── requirements.txt        # common evaluation dependencies
└── README.md
```

## Release Hygiene

Before sharing the artifact, run a final search for personal identifiers and local paths:

```bash
rg "(/home/|/mnt/|E:\\\\|github.com/|@)" .
```

Generated outputs, caches, virtual environments, `.env` files, and cluster logs are intentionally ignored by `.gitignore`.