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

cd Aspects_benchmarking
cp .env.example .env
```

Edit `Aspects_benchmarking/.env` and set `DATA_ROOT` to the downloaded dataset root. API keys are read only from environment variables or `.env`; no secrets should be committed.

## Data

The evaluation dataset is not stored in this repository. The expected dataset layout and per-dimension commands are documented in `Aspects_benchmarking/README.md`.

For anonymous review, distribute data through an anonymized artifact link and keep any private or non-anonymous storage locations out of this repository.

## Repository Layout

```text
PRISM/
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