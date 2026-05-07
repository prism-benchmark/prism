# Data Setup

This directory contains the setup scripts for preparing the aspect-benchmark dataset used by `Aspects_benchmarking/`.

## What It Does

Run the one-shot setup script to prepare the benchmark data from Hugging Face:

```bash
python3 Data/setup_aspect_benchmark.py --write-env
```

In order, the script:

1. Downloads `SUBSET_1000.zip` from Hugging Face. This archive contains the raw papers and human reviews.
2. Downloads `Final_LLM_Reviewer_Data_Sample.zip` from Hugging Face. This archive contains the LLM reviewer outputs.
3. Extracts both archives.
4. Maps `SUBSET_1000` into the aspect-benchmark layout:
   - `ICLR2024/`
   - `Neurlps2025/`
   - `human_reviews/`
   - `papers/{id}.grobid.txt`
   - `paper_ids_200_*.txt`
5. Overlays the LLM reviewer sample so these reviewer-output folders are populated:
   - `sea_*`
   - `tree_*`
   - `reviewer2_*`
   - `deepreview_*`
   - `cyclereview_*`
6. Generates `paper_ids_50_*.txt` subset files.
7. Writes `DATA_ROOT` into `Aspects_benchmarking/.env`.
8. Cleans up the downloaded zip files and intermediate extracted folders.

After setup, `Data/` should contain only:

```text
Data/
  Final_LLM_Reviewer_Data/
```

## Step-by-Step

From the repository root:

```bash
cd /path/to/PRISM
python3 Data/setup_aspect_benchmark.py --write-env
```

To rebuild from scratch and overwrite an existing prepared dataset:

```bash
python3 Data/setup_aspect_benchmark.py --write-env --force
```

To keep the downloaded zip files and extracted intermediate folders for debugging:

```bash
python3 Data/setup_aspect_benchmark.py --write-env --keep-downloads
```

## Output

The prepared dataset is written to:

```text
Data/Final_LLM_Reviewer_Data/
```

`Aspects_benchmarking/.env` is updated with:

```dotenv
DATA_ROOT=/absolute/path/to/Data/Final_LLM_Reviewer_Data
```

After that, the aspect benchmark scripts can read the dataset through `DATA_ROOT`.
