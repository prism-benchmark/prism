---
language:
  - en
task_categories:
  - text-classification
  - text-generation
  - other
tags:
  - scientific-papers
  - peer-review
  - grobid
  - tei-xml
  - bibtex
  - openreview
  - machine-learning
  - croissant
pretty_name: "PRISM: A Multi-Dimensional Benchmark for Evaluating LLM Peer Reviewers"
license: other
---

# PRISM: A Multi-Dimensional Benchmark for Evaluating LLM Peer Reviewers

A large-scale benchmark built on **47,214 research papers** from five top machine learning conferences, including full text, peer reviews, editorial decisions, GROBID-parsed metadata, and bibliographic references.

PRISM evaluates LLM-based peer reviewers across five dimensions — **Validity**, **Helpfulness**, **Comprehensiveness**, **Specificity**, and **Faithfulness** — and supports tasks including **review generation**, **meta-review generation**, **acceptance prediction**, and **review score prediction**.

---

## Reviewer Quick Start: Aspect Benchmarking

Use this path when you want to reproduce the aspect experiments in `Aspects_benchmarking/`.

### 1. Download or unpack the benchmark data

If the archive is already present in `Data/`, run:

```bash
cd /path/to/PRISM
python3 Data/prepare_aspect_benchmark_data.py \
  --source Data/Final_LLM_Reviewer_Data_Sample.zip \
  --output-dir Data/Final_LLM_Reviewer_Data \
  --write-env \
  --force
```

For the full released artifact, replace `--source` with the downloaded zip path or extracted folder:

```bash
python3 Data/prepare_aspect_benchmark_data.py \
  --source /path/to/Final_LLM_Reviewer_Data.zip \
  --output-dir /path/to/Final_LLM_Reviewer_Data \
  --write-env
```

The preparation script:

- extracts or copies the dataset into the layout required by all aspect scripts;
- keeps the original `Neurlps2025` folder spelling used by the released data;
- creates `paper_ids_50_*` files from each conference's `paper_ids_200_*` file;
- validates reviewer folders for Human, SEA, TreeReview, Reviewer2, DeepReview, and CycleReview;
- optionally writes `Aspects_benchmarking/.env` with `DATA_ROOT=<prepared dataset path>`.

Use `--strict` if you want the script to fail when listed paper IDs do not have all expected files. The included sample archive contains one paper per conference but keeps 200-paper id lists, so non-strict validation reports warnings for the missing sample files.

### 2. Configure evaluator keys

Open `Aspects_benchmarking/.env` and add the keys needed for the evaluator you will run:

```dotenv
DATA_ROOT=/absolute/path/to/Final_LLM_Reviewer_Data
GOOGLE_API_KEY=
MIMO_API_KEY=
MIMO_BASE_URL=
```

### 3. Run an aspect experiment

```bash
cd Aspects_benchmarking
pip install -r requirements.txt

# Depth of Analysis, example LLM source
python3 depth_of_analysis/run_llm.py --source sea_iclr2026

# Constructiveness
python3 constructiveness/run_constructiveness.py

# Flaw Identification
python3 flaw_identification/main_cfi_iclr2026.py
```

All aspect pipelines read the same `DATA_ROOT`, so once `prepare_aspect_benchmark_data.py` succeeds the same dataset is reusable across Depth of Analysis, Constructiveness, Flaw Identification, and Novelty Assessment.

---

## Quick Start: SUBSET_1000

 We provide a **1,000-paper subset** with the same distribution as the full dataset — **200 papers per conference** — that use in my paper experiment

| Property | Value |
|---|---|
| **Papers** | 1,000 (200 × 5 conferences) |
| **Distribution** | Identical to full dataset (200 ICLR 2024, 200 ICLR 2025, 200 ICLR 2026, 200 ICML 2025, 200 NeurIPS 2025) |
| **Format** | `subset_1000.parquet` (30 KB) + `SUBSET_1000.zip` (4.5 GB) |
| **Contents** | All folder types: `json/`, `txt/`, `grobid_*`, `pdf/`, `review_*`, `paper_nougat_mmd/` |
| **Use Case** | Rapid prototyping, testing, and development without downloading full dataset |

**Download:**
```bash
python3 Data/download_data.py
```

**Load & unpack:**
```bash
# Load subset parquet only
python3 Data/load.py --subset --parquet-only

# Load and unpack the subset
python3 Data/load.py --subset

# Load and unpack a specific venue
python3 Data/load.py --venue ICLR_2025
```

**Our research team also provides the full dataset** with all 47,214 papers for complete analysis and model training.

---

## Dataset Summary

| Property | Value |
|---|---|
| **Total papers** | 47,214 |
| **Total reviews** | 186,090 |
| **Conferences** | ICLR 2024, ICLR 2025, ICLR 2026, ICML 2025, NeurIPS 2025 |
| **Source** | OpenReview (ICLR) + Conference proceedings (ICML, NeurIPS) |
| **Tabular format** | `papers.parquet` (1.8 GB, zstd compression) |
| **File-based format** | Directory structure per venue (zip archives) |

---

## Venue Statistics

### Paper Counts & Acceptance Rates

| Venue | Papers | Reviews | Reviews/Paper | Accepted | Rejected | Pending | Accept Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| **ICLR 2024** | 7,262 | 28,028 | 3.9 | 2,261 | 3,519 | 1,482 | 39.1% |
| **ICLR 2025** | 11,519 | 46,744 | 4.1 | 3,708 | 5,018 | 2,793 | 42.5% |
| **ICLR 2026** | 19,471 | 75,847 | 3.9 | 5,358 | 8,814 | 5,299 | 37.8% |
| **ICML 2025** | 3,422 | 13,102 | 3.8 | 3,260 | 162 | 0 | 95.3% |
| **NeurIPS 2025** | 5,540 | 22,369 | 4.0 | 5,286 | 254 | 0 | 95.4% |
| **Total** | **47,214** | **186,090** | **3.9** | **19,873** | **17,767** | **9,574** | — |

> *Accept Rate is calculated as `accepted / (accepted + rejected)`, excluding Pending papers (which have reviews but no final decision).*

> **⚠️ Important — Data Collection Bias:**
>
> **ICLR data** (2024, 2025, 2026) was collected from **OpenReview** and includes **both accepted and rejected** submissions, providing a representative sample of the full review process.
>
> **ICML 2025 and NeurIPS 2025** data was collected from **conference proceedings / accepted paper lists**, and therefore **heavily skews toward accepted papers**. The vast majority of rejected submissions are **not included**:
>
> | Venue | Real Submissions | Real Accepted | Real Rate | In Dataset | Dataset Rate |
> |---|---:|---:|---:|---:|---:|
> | NeurIPS 2025 | ~21,575 | ~5,290 | 24.5% | 5,540 | 95.4% |
> | ICML 2025 | ~12,107 | ~3,260 | 26.9% | 3,422 | 95.3% |
>
> The ~254 "rejected" papers in NeurIPS 2025 and ~162 in ICML 2025 likely represent workshop papers, withdrawn submissions, or scraping artifacts — **not** the main-track rejected papers.
>
> **Implication:** Any analysis of review quality, acceptance prediction, or reviewer behavior should **only use ICLR data** (which has balanced accept/reject coverage). NeurIPS and ICML data are suitable only for studying **accepted paper characteristics** and **bibliographic analysis**.

### Decision Distribution (ICLR Only — Representative Sample)

> The following distribution reflects **ICLR data only** (2024–2026), which is the only venue with balanced accept/reject coverage. NeurIPS 2025 and ICML 2025 are excluded from this table as their data is heavily biased toward accepted papers.
>
> "Pending" papers have initial reviews but no final decision (meta-review is "TBD") — they were scraped before decisions were announced and were never updated. Acceptance rates below are calculated excluding Pending papers.

| Decision | Papers | Share |
|---|---:|---:|
| Reject | 17,351 | 45.8% |
| Pending | 9,574 | 25.3% |
| Accept (poster) | 7,468 | 19.7% |
| Accept (spotlight) | 747 | 2.0% |
| Accept (oral) | 329 | 0.9% |
| Conditional Accept | 11 | <0.1% |

**ICML 2025 and NeurIPS 2025** (not representative): Accept 8,546 / Reject 416 — these numbers reflect only the small fraction of rejected papers that appeared in the proceedings scrape, not the full submission pool.

### Review Rating Distributions

**ICLR 2025** (n = 46,744 reviews, mean = 5.15, scale 1–10):

| Score | Count | Share |
|---:|---:|---:|
| 1 | 1,029 | 2.2% |
| 3 | 11,535 | 24.7% |
| 5 | 13,101 | 28.0% |
| 6 | 14,693 | 31.4% |
| 8 | 6,214 | 13.3% |
| 10 | 172 | 0.4% |

**ICLR 2026** (n = 75,847 reviews, mean = 4.21, scale 0–10):

| Score | Count | Share |
|---:|---:|---:|
| 0 | 1,319 | 1.7% |
| 2 | 19,864 | 26.2% |
| 4 | 29,759 | 39.2% |
| 6 | 19,707 | 26.0% |
| 8 | 5,010 | 6.6% |
| 10 | 188 | 0.2% |

**NeurIPS 2025** (n = 22,369 reviews, mean = 4.31, scale 1–6):

| Score | Count | Share |
|---:|---:|---:|
| 1 | 30 | 0.1% |
| 2 | 423 | 1.9% |
| 3 | 1,730 | 7.7% |
| 4 | 11,077 | 49.5% |
| 5 | 8,728 | 39.0% |
| 6 | 381 | 1.7% |

> **Note:** NeurIPS 2025 reviews are **biased toward accepted papers** (95.4% of the dataset). The distribution above reflects reviewer behavior on accepted papers only and is **not representative** of reviews on rejected submissions.
>
> **Note:** ICLR 2024 uses text-based rating fields (`Soundness`, `Presentation`, `Contribution` on a poor/fair/good/excellent scale) with no numeric `Rating` field, so aggregate rating statistics are not available. ICML 2025 uses `Overall Recommendation` instead of numeric ratings.

---

## Dataset Structure

### Directory Layout

```
paper_data/
├── papers.parquet              # Combined tabular dataset (all venues)
├── ICLR_2024/
│   ├── json/                   # Review data (JSON)
│   ├── txt/                    # Full text (extracted)
│   ├── grobid_metadata/        # GROBID metadata (JSON)
│   ├── grobid_bib/             # GROBID bibliography (JSON + BibTeX)
│   ├── grobid_tei/             # GROBID TEI XML
│   ├── grobid_fulltext/        # GROBID full text extraction
│   ├── pdf/                    # Original PDFs
│   ├── review_json/            # Raw review JSON (ICLR 2024 only)
│   ├── review_raw_txt/         # Raw review text (ICLR 2024 only)
│   ├── paper_nougat_mmd/       # Nougat Mathpix Markdown (ICLR 2024 only)
│   └── scraping_summary.json   # Scraping metadata
├── ICLR_2025/
│   ├── json/
│   ├── txt/
│   ├── grobid_metadata/
│   ├── grobid_bib/
│   ├── grobid_tei/
│   ├── grobid_fulltext/
│   ├── pdf/
│   └── scraping_summary.json
├── ICLR_2026/
│   └── ... (same structure as ICLR 2025)
├── ICML_2025/
│   └── ...
├── NeurIPS_2025/
│   └── ...
├── convert_to_parquet.py       # Conversion script
└── ICLR_2024.zip               # Zip archives for distribution
    ICLR_2025.zip
    ICLR_2026.zip
    ICML_2025.zip
    NeurIPS_2025.zip
```

### Folder Descriptions

| Folder | Description | File Format | File Count (total) |
|---|---|---|---:|
| `json/` | Peer reviews, decisions, meta-reviews, and structured review data (from OpenReview for ICLR; from proceedings for ICML/NeurIPS) | `.json` | 47,215 |
| `txt/` | Extracted full text of papers (plain text, from PDF conversion) | `.txt` | 47,215 |
| `grobid_metadata/` | GROBID-parsed metadata: title, authors, abstract, keywords, date | `.grobid.json` | 47,099 |
| `grobid_bib/` | GROBID-parsed bibliography: structured references (JSON) and BibTeX | `.grobid.json` + `.grobid.bib` | 94,103 |
| `grobid_tei/` | Full GROBID TEI XML output: structured document with sections, figures, tables, equations | `.grobid.tei.xml` | 47,155 |
| `grobid_fulltext/` | GROBID-extracted full text (cleaner than `txt/`, preserves section structure) | `.grobid.txt` | 47,084 |
| `pdf/` | Original PDF files (from OpenReview / conference proceedings, **not uploaded to Hugging Face**) | `.pdf` | 41,823 |
| `review_json/` | Raw review JSON (ICLR 2024 only) | `.json` | 7,262 |
| `review_raw_txt/` | Raw review text (ICLR 2024 only) | `.txt` | 7,262 |
| `paper_nougat_mmd/` | Nougat Mathpix Markdown output (ICLR 2024 only) | `.mmd` | 7,262 |

### File Sizes per Folder

| Folder | ICLR 2024 | ICLR 2025 | ICLR 2026 | ICML 2025 | NeurIPS 2025 | **Total** |
|---|---:|---:|---:|---:|---:|---:|
| `json/` | 127 MB | 188 MB | 341 MB | 68 MB | 94 MB | **818 MB** |
| `txt/` | 118 MB | 192 MB | 317 MB | 64 MB | 86 MB | **777 MB** |
| `grobid_metadata/` | 29 MB | 46 MB | 77 MB | 14 MB | 22 MB | **188 MB** |
| `grobid_bib/` | 360 MB | 630 MB | 1.1 GB | 175 MB | 326 MB | **2.6 GB** |
| `grobid_tei/` | 835 MB | 1.5 GB | 2.5 GB | 471 MB | 902 MB | **6.2 GB** |
| `grobid_fulltext/` | 350 MB | 599 MB | 1.1 GB | 200 MB | 410 MB | **2.7 GB** |
| `pdf/` | 35 GB | 67 GB | 122 GB | 14 GB | — | **238 GB** *(not uploaded)* |
| `paper_nougat_mmd/` | 439 MB | — | — | — | — | **439 MB** |
| `review_json/` | 94 MB | — | — | — | — | **94 MB** |
| `review_raw_txt/` | 259 MB | — | — | — | — | **259 MB** |
| **Venue Total** | **38 GB** | **70 GB** | **128 GB** | **15 GB** | **1.8 GB** | **253 GB** |

---

## Parquet Dataset (`papers.parquet`)

All structured data across five venues is consolidated into a single **Apache Parquet** file with **zstd** compression.

| Property | Value |
|---|---|
| File | `papers.parquet` |
| Size | **1.8 GB** (compressed from 12.2 GB of source text) |
| Compression ratio | **6.8×** |
| Rows | 47,214 |
| Columns | 31 |
| Engine | PyArrow |
| Compression | zstd |

### Schema

| Column | Type | Description |
|---|---|---|
| `paper_id` | string | OpenReview paper ID (e.g., `00ezkB2iZf`) |
| `venue` | string | Conference venue (`ICLR_2024`, `ICLR_2025`, `ICLR_2026`, `ICML_2025`, `NeurIPS_2025`) |
| `decision` | string | Editorial decision (`Accept (poster)`, `Reject`, `Pending`, etc.). **Note:** ICML/NeurIPS are mostly `Accept` variants — see [data collection bias](#venue-statistics). |
| `meta_review` | string | Meta-review text from the area chair |
| `num_reviews` | int64 | Number of peer reviews |
| `rating_avg` | float64 | Average reviewer rating (where available) |
| `rating_min` | int64 | Minimum reviewer rating |
| `rating_max` | int64 | Maximum reviewer rating |
| `confidence_avg` | float64 | Average reviewer confidence |
| `soundness_avg` | float64 | Average soundness score |
| `presentation_avg` | float64 | Average presentation score |
| `contribution_avg` | float64 | Average contribution score |
| `reviews_json` | string | Full reviews as JSON string (all review fields) |
| `keywords` | string | Paper keywords (JSON array) |
| `primary_area` | string | Primary subject area |
| `subject_areas` | string | Subject areas (JSON array) |
| `review_title` | string | Paper title from submission (available for all ICLR; may be empty for ICML/NeurIPS) |
| `review_abstract` | string | Abstract from submission (available for all ICLR; may be empty for ICML/NeurIPS) |
| `grobid_title` | string | Title extracted by GROBID |
| `grobid_abstract` | string | Abstract extracted by GROBID |
| `grobid_authors` | string | Authors extracted by GROBID (JSON array) |
| `grobid_keywords` | string | Keywords extracted by GROBID (JSON array) |
| `grobid_date` | string | Publication/acceptance date |
| `full_text` | string | Full paper text (from `txt/` folder) |
| `grobid_fulltext` | string | GROBID-extracted full text (from `grobid_fulltext/`) |
| `bibliography_json` | string | Bibliography as structured JSON |
| `bibliography_bib` | string | Bibliography in BibTeX format |
| `pdf_path` | string | Relative path to the original PDF file |
| `stat_num_reviews` | int64 | Number of reviews (from statistics field) |
| `stat_has_meta_review` | bool | Whether meta-review exists |
| `stat_has_decision` | bool | Whether decision exists |

### Usage

```python
import pandas as pd

# Load the full dataset
df = pd.read_parquet("papers.parquet")

# Filter by venue
iclr2025 = df[df["venue"] == "ICLR_2025"]

# Find top-rated accepted papers
accepted = df[df["decision"].str.contains("Accept", na=False)]
top = accepted.nlargest(10, "rating_avg")[
    ["paper_id", "venue", "grobid_title", "rating_avg", "decision"]
]

# Full-text search
rl_papers = df[df["full_text"].str.contains("reinforcement learning", case=False, na=False)]

# Parse structured reviews
import json
paper = df.iloc[0]
reviews = json.loads(paper["reviews_json"])
for r in reviews:
    print(f"Rating: {r.get('Rating')}, Summary: {r.get('Summary')[:100]}...")
```

```python
# Using DuckDB for SQL queries
import duckdb
con = duckdb.connect()
con.execute("CREATE TABLE papers AS SELECT * FROM read_parquet('papers.parquet')")

# ⚠️ Acceptance rate query — only valid for ICLR venues
# ICML/NeurIPS data is biased (mostly accepted papers from proceedings)
con.execute("""
    SELECT venue,
           COUNT(*) AS total,
           SUM(CASE WHEN decision LIKE '%Accept%' THEN 1 ELSE 0 END) AS accepted,
           ROUND(100.0 * accepted / total, 1) AS accept_rate
    FROM papers
    WHERE decision != 'Pending'
      AND venue LIKE 'ICLR%'   -- Only ICLR has balanced accept/reject data
    GROUP BY venue
    ORDER BY accept_rate DESC
""").fetchdf()
```

```python
# Using Polars (fastest)
import polars as pl
df = pl.read_parquet("papers.parquet")

# Find top-rated ICLR 2025 accepted papers (ICLR has balanced accept/reject data)
df.filter(
    (pl.col("venue") == "ICLR_2025") & (pl.col("decision").str.contains("Accept"))
).sort("rating_avg", descending=True).head(10)

# Topic analysis across all venues (works regardless of selection bias)
df.group_by("venue").agg(
    pl.col("grobid_abstract").str.contains("large language model").sum().alias("llm_papers"),
    pl.len().alias("total")
)
```

---

## Zip Archives (File-Based Format)

For direct access to individual files (TEI XML, BibTeX, full text), download the zip archives:

| Archive | Papers | Size (no PDFs) |
|---|---:|---:|
| `ICLR_2024.zip` | 7,262 | ~2.2 GB |
| `ICLR_2025.zip` | 11,519 | ~3.1 GB |
| `ICLR_2026.zip` | 19,471 | ~5.4 GB |
| `ICML_2025.zip` | 3,422 | ~990 MB |
| `NeurIPS_2025.zip` | 5,540 | ~1.8 GB |

### About PDFs

Original PDFs total **~238 GB** across all venues and are **not included** in the Hugging Face upload due to their size. If you need access to the PDF files, please contact the authors directly and we can share them separately.

The `pdf_path` column in `papers.parquet` references the original PDF location for each paper, so you can match papers to PDFs once obtained.

### Download from Hugging Face Hub

```python
from huggingface_hub import snapshot_download

local_dir = snapshot_download(repo_id="anoyresearcher/prism_paper_data", repo_type="dataset")
print(local_dir)
```

Then unzip the archives you need:

```bash
unzip -q ICLR_2025.zip -d ./extracted/
```

---

## File Format Details

### `json/` — Review Data

Each JSON file contains the complete peer review record for one paper:

```json
{
    "paper_id": "00ezkB2iZf",
    "Decision": "Reject",
    "Meta review": {
        "Metareview": "In this paper, the authors propose...",
        "Justification For Why Not Higher Score": "...",
        "Justification For Why Not Lower Score": "..."
    },
    "reviews": [
        {
            "Review ID": "TeO25XUwES",
            "Rating": "3",
            "Confidence": "4",
            "Summary": "...",
            "Soundness": "2",
            "Presentation": "2",
            "Contribution": "2",
            "Strengths": "...",
            "Weaknesses": "...",
            "Questions": "...",
            "Limitations": "..."
        }
    ],
    "keywords": ["robustness", "medical QA"],
    "primary_area": "Safety in Machine Learning",
    "subject_areas": [...],
    "title": "...",
    "abstract": "...",
    "statistics": {
        "num_reviews": 4,
        "has_meta_review": true,
        "has_decision": true
    }
}
```

> **Note:** Review field names and rating scales differ between venues. ICLR 2025/2026 use numeric ratings (1–10 or 0–10); ICLR 2024 uses text-based fields (Soundness/Presentation/Contribution: poor/fair/good/excellent) with an empty `Rating` field; ICML 2025 uses `Overall Recommendation`; NeurIPS uses 1–6 scale.

### `grobid_metadata/` — Structured Metadata

```json
{
    "title": "MEDFUZZ: EXPLORING THE ROBUSTNESS OF LARGE LANGUAGE MODELS...",
    "authors": ["Author A", "Author B"],
    "abstract": "Large language models (LLM) have achieved...",
    "keywords": [],
    "date": ""
}
```

### `grobid_bib/` — Bibliography

Two files per paper:
- **`.grobid.json`**: Structured JSON array of references
- **`.grobid.bib`**: BibTeX format

```json
[
    {
        "title": "Openbiollms: Advancing open-source large language models...",
        "authors": ["Author A", "Author B"],
        "year": "2024",
        "venue": ""
    }
]
```

### `grobid_tei/` — TEI XML

Full GROBID output in TEI (Text Encoding Initiative) XML format containing:
- Document structure (sections, paragraphs)
- Title, authors, abstract
- References and citations
- Figures, tables, equations (when parseable)

```xml
<TEI xmlns="http://www.tei-c.org/ns/1.0">
    <teiHeader>
        <fileDesc>
            <titleStmt><title>MEDFUZZ: ...</title></titleStmt>
            ...
        </fileDesc>
    </teiHeader>
    <text>
        <body>
            <div><head>INTRODUCTION</head><p>Cutting-edge large language models...</p></div>
            ...
        </body>
    </text>
</TEI>
```

### `txt/` and `grobid_fulltext/` — Full Text

Both contain the full paper text as plain text:
- **`txt/`**: Extracted from PDF (may contain OCR artifacts)
- **`grobid_fulltext/`**: Extracted by GROBID (typically cleaner, preserves section boundaries)

### `pdf/` — Original PDFs

Original PDF files downloaded from OpenReview / conference proceedings. Not included in zip archives (available separately). Referenced via the `pdf_path` column in `papers.parquet`.

---

## File Counts per Folder

| Folder | ICLR 2024 | ICLR 2025 | ICLR 2026 | ICML 2025 | NeurIPS 2025 |
|---|---:|---:|---:|---:|---:|
| `json/` | 7,262 | 11,520 | 19,471 | 3,422 | 5,540 |
| `txt/` | 7,262 | 11,520 | 19,471 | 3,422 | 5,540 |
| `grobid_metadata/` | 7,286 | 11,475 | 19,421 | 3,385 | 5,532 |
| `grobid_bib/` | 14,546 | 22,910 | 38,818 | 6,764 | 11,064 |
| `grobid_tei/` | 7,286 | 11,491 | 19,421 | 3,422 | 5,535 |
| `grobid_fulltext/` | 7,282 | 11,465 | 19,420 | 3,385 | 5,532 |
| `pdf/` | 7,304 | 11,629 | 19,468 | 3,422 | 0 |

> **Note:** `grobid_bib/` has ~2× the file count because each paper produces both `.grobid.json` and `.grobid.bib` files.

---

## Data Collection & Processing Pipeline

### Step 1 — Crawl from OpenReview / Conference Pages

Paper metadata, reviews, decisions, and PDF links are scraped from [OpenReview](https://openreview.net) for ICLR venues, and from conference proceedings pages for ICML and NeurIPS.

> **Scraping timeline:** For ICLR venues, data was scraped **during the review period** — after initial peer reviews were posted but before all meta-reviews and final decisions were announced. As a result, ~20–27% of ICLR papers have "Pending" as the decision (meta-review shows "TBD"). These papers still have complete reviews and full text; only the final decision and meta-review are missing.

**Outputs per paper:**
- `json/<paper_id>.json` — Structured review data: decision, meta-review, individual reviews (ratings, confidence, strengths, weaknesses, questions)
- `txt/<paper_id>.txt` — Full paper text extracted from the PDF
- `pdf/<paper_id>.pdf` — Original paper PDF
- `scraping_summary.json` — List of all processed paper IDs for the venue

### Step 2 — Process PDFs with GROBID

PDFs are processed through [GROBID](https://github.com/kermitt2/grobid) (a machine-learning-based document parser) to extract structured information from the papers.

**Outputs per paper:**
- `grobid_metadata/<paper_id>.grobid.json` — Title, authors, abstract, keywords, date
- `grobid_bib/<paper_id>.grobid.json` — Bibliography as structured JSON array
- `grobid_bib/<paper_id>.grobid.bib` — Bibliography in BibTeX format
- `grobid_tei/<paper_id>.grobid.tei.xml` — Full document in TEI XML (sections, figures, tables, equations)
- `grobid_fulltext/<paper_id>.grobid.txt` — Clean full-text extraction with section boundaries

### Step 3 — Consolidate to Parquet

All structured data is merged into a single `papers.parquet` file (1.8 GB) using `convert_to_parquet.py`. This combines review metadata, GROBID outputs, and full text into one queryable table.

```
OpenReview / Conference Pages
         │
         ▼
   ┌─────────────┐
   │  json/       │  Reviews, decisions, meta-reviews
   │  txt/        │  Full paper text
   │  pdf/        │  Original PDFs (not uploaded)
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │   GROBID     │  PDF parsing engine
   └──────┬──────┘
          │
          ▼
   ┌─────────────────┐
   │ grobid_metadata/ │  Title, authors, abstract
   │ grobid_bib/      │  Structured references
   │ grobid_tei/      │  Full TEI XML
   │ grobid_fulltext/  │  Clean text extraction
   └──────┬──────────┘
          │
          ▼
   ┌──────────────┐
   │ papers.parquet│  All venues merged (1.8 GB)
   └──────────────┘
```

### Venue Sources

| Venue | Source | Coverage |
|---|---|---|
| **ICLR 2024, 2025, 2026** | [OpenReview](https://openreview.net) | ✅ All submissions (accepted + rejected + pending) |
| **ICML 2025** | Conference proceedings page | ⚠️ Accepted papers only (~26.9% of submissions) |
| **NeurIPS 2025** | Conference proceedings page | ⚠️ Accepted papers only (~24.5% of submissions) |

### `scraping_summary.json`

Each venue folder contains a scraping summary with the list of collected paper IDs:

```json
{
    "total_processed": 11672,
    "total_failed": 0,
    "paper_ids": ["zz9jAssrwL", "zxg6601zoc", ...]
}
```

> **Note:** For ICLR venues, `paper_ids` includes all submissions (accepted + rejected + pending). For ICML and NeurIPS, `paper_ids` primarily contains accepted papers from the conference proceedings.

---

## Potential Use Cases

> **⚠️ Venue suitability varies by task.** See the [Data Collection Bias](#venue-statistics) section for details.

| Task | Suitable Venues | Reason |
|---|---|---|
| **Acceptance/Rejection Prediction** | ICLR 2024, 2025, 2026 only | Balanced accept/reject coverage required |
| **Peer Review Bias Analysis** | ICLR 2024, 2025, 2026 only | Need both accepted and rejected review data |
| **Reviewer Behavior Analysis** | ICLR 2024, 2025, 2026 only | Need reviews for both accepted and rejected papers |
| **Accepted Paper Characteristics** | All venues | Topic modeling, method trends, citation patterns |
| **Citation Network Analysis** | All venues | Bibliography data available for all papers |
| **NLP for Scientific Text** | All venues | Full text available for all papers |
| **Argument Mining** | ICLR 2024, 2025, 2026 only | Need reviews for both accepted and rejected papers |
| **Meta-Science / Research Trends** | All venues | Study topics, methods, impact across conferences |

### Detailed Use Cases

- **Peer Review Analysis** *(ICLR only)*: Predict review scores, detect bias, analyze reviewer behavior across accept/reject decisions
- **Paper Quality Prediction** *(ICLR only)*: Predict acceptance decisions from paper text and review scores
- **Citation Network Analysis** *(All venues)*: Build citation graphs from bibliography data
- **Meta-Science** *(All venues)*: Study trends in ML research, topic modeling, research impact
- **NLP for Scientific Text** *(All venues)*: Train/evaluate models on scientific document understanding
- **Argument Mining** *(ICLR only)*: Extract strengths, weaknesses, and arguments from reviews
- **Decision Prediction** *(ICLR only)*: Binary/multi-class classification of paper acceptance

---

## Known Limitations

- **GROBID parsing errors**: Some papers have incomplete or malformed GROBID output depending on PDF formatting
- **Rating scale differences**: ICLR 2025/2026 use numeric ratings (1–10 or 0–10), NeurIPS uses 1–6 scale, ICLR 2024 uses text-based fields (Soundness/Presentation/Contribution: poor/fair/good/excellent), and ICML 2025 uses `Overall Recommendation`. Cross-venue rating comparisons are not meaningful.
- **Incomplete coverage**: Not all papers have PDFs (NeurIPS 2025 PDFs not available); some GROBID outputs are missing
- **"Pending" decisions (ICLR 2024–2026)**: ~20–27% of ICLR papers show "Pending" as the decision, with meta-review text set to "TBD". This is because the data was scraped **during the review period** — after initial peer reviews were posted, but before the area chairs wrote meta-reviews and the final decisions were announced. These papers have complete reviews but no final decision. This affects ~1,482 ICLR 2024 papers, ~2,793 ICLR 2025 papers, and ~5,299 ICLR 2026 papers. They should be treated as **missing decisions**, not as withdrawn or rejected papers.
- **PDF quality**: Some PDFs contain scanned images or non-standard layouts that reduce extraction quality
- **ICML 2025 / NeurIPS 2025 selection bias (CRITICAL)**: These venues were collected from **conference proceedings, not OpenReview**. Only ~5,540 of ~21,575 NeurIPS submissions and ~3,422 of ~12,107 ICML submissions are in the dataset. The data is **not representative of the full submission pool** — rejected papers are almost entirely missing. This makes these venues **unsuitable for acceptance prediction, reviewer bias analysis, or rejection-related studies**. Only ICLR data (2024–2026) has balanced accept/reject coverage.

---

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{prism_2026,
    title     = {PRISM: A Multi-Dimensional Benchmark for Evaluating LLM Peer Reviewers},
    author    = {Anonymous},
    year      = {2026},
    note      = {Under review — author identity withheld for double-blind review}
}
```

---

## License

This dataset is provided for **research purposes only**. The data is sourced from [OpenReview](https://openreview.net) (ICLR venues) and conference proceedings pages (ICML, NeurIPS), and is subject to their respective Terms of Use. Users are responsible for complying with the original data sources' terms and conditions.

---

## Reproducing the Parquet Conversion

To regenerate `papers.parquet` from the raw data:

```bash
pip install pandas pyarrow
python3 convert_to_parquet.py
```

The script reads all venues, extracts structured fields from JSON/GROBID files, and writes a single Parquet file with zstd compression.
