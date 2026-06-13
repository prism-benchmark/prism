# Data Setup

From the repository root:

```bash
python run.py --setup-data
```

This command:

1. Downloads `demo_data.zip` from
   `anoyresearcher/prism_paper_data` on Hugging Face.
2. Safely extracts the archive.
3. Installs the canonical dataset at `Data/input/`.
4. Validates every conference and reviewer directory.
5. Writes the absolute `DATA_ROOT` path to the root `.env`.
6. Removes the downloaded ZIP after successful setup.

The resulting directory is:

```text
Data/input/
├── ICLR2024/
├── ICLR2025/
├── ICLR2026/
├── ICML2025/
└── NeurIPS2025/
```

Each conference contains:

```text
papers/ human_reviews/ sea/ reviewer2/ tree/ deepreview/ cyclereview/
```

To replace existing data:

```bash
python Data/setup_aspect_benchmark.py --force
```

To install a local archive:

```bash
python Data/setup_aspect_benchmark.py --source /path/to/demo_data.zip --force
```
