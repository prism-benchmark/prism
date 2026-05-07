#!/usr/bin/env python3
"""End-to-end setup for the PRISM aspect-benchmark experiments.

This is the single entry point reviewers should use. It performs:

  1. Download `SUBSET_1000.zip` (raw papers + human reviews) from Hugging Face.
  2. Download `Final_LLM_Reviewer_Data_Sample.zip` (LLM reviewer outputs).
  3. Extract both archives.
  4. Map the SUBSET_1000 layout into Final_LLM_Reviewer_Data/ (the layout
     used by Aspects_benchmarking/) and overlay the LLM reviewer sample on
     top so the `sea_*`, `tree_*`, `reviewer2_*`, `deepreview_*`,
     `cyclereview_*` folders are populated.
  5. Generate `paper_ids_50_*` subset files.
  6. (Optionally) write DATA_ROOT into Aspects_benchmarking/.env.
  7. Clean up the downloaded zips and intermediate extractions, leaving
     `Data/` containing only the prepared `Final_LLM_Reviewer_Data/`.

Quick reviewer flow:

    python3 Data/setup_aspect_benchmark.py --write-env

After this finishes, fill in API keys in Aspects_benchmarking/.env and run
any of the aspect scripts documented in Aspects_benchmarking/README.md.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import download_data            # noqa: E402
import map_hf_to_aspect_layout  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download PRISM data and prepare it for Aspects_benchmarking."
    )
    parser.add_argument(
        "--subset-source",
        type=Path,
        default=None,
        help=(
            "Optional pre-downloaded SUBSET_1000.zip / extracted directory. "
            "When given, the SUBSET_1000 download step is skipped."
        ),
    )
    parser.add_argument(
        "--sample-source",
        type=Path,
        default=None,
        help=(
            "Optional pre-downloaded Final_LLM_Reviewer_Data_Sample.zip / "
            "extracted directory. When given, the sample download is skipped."
        ),
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Skip downloading/overlaying the LLM reviewer sample archive.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=THIS_DIR,
        help="Where to download the Hugging Face artifacts (default: Data/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=THIS_DIR / "Final_LLM_Reviewer_Data",
        help="Final DATA_ROOT directory (default: Data/Final_LLM_Reviewer_Data).",
    )
    parser.add_argument(
        "--materialize",
        choices=("symlink", "hardlink", "copy"),
        default="copy",
        help=(
            "How per-paper files are placed in DATA_ROOT (default: copy, so "
            "the prepared dataset survives cleanup of the download cache)."
        ),
    )
    parser.add_argument(
        "--max-papers-per-venue",
        type=int,
        default=200,
        help="Cap papers per venue (default: 200, matches the paper experiments).",
    )
    parser.add_argument(
        "--write-env",
        action="store_true",
        help="Write DATA_ROOT into Aspects_benchmarking/.env.",
    )
    parser.add_argument(
        "--keep-downloads",
        action="store_true",
        help=(
            "Keep the downloaded zips and extracted intermediates in --download-dir. "
            "By default they are deleted after the prepared DATA_ROOT is built."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite an existing prepared DATA_ROOT.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token (defaults to HF_TOKEN environment variable).",
    )
    return parser.parse_args()


def _download(filename: str, download_dir: Path, token: str | None, force: bool) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    print(f"== Downloading {filename} -> {download_dir}")
    return download_data.download_file(filename, download_dir, token, force)


def _cleanup(paths: list[Path]) -> None:
    for path in paths:
        if not path or not path.exists():
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink()
            print(f"Removed {path}")
        except OSError as exc:
            print(f"Could not remove {path}: {exc}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()

    cleanup_targets: list[Path] = []

    # --- Step 1: SUBSET_1000 (raw papers + human reviews) -----------------
    if args.subset_source is not None:
        subset_path = args.subset_source.expanduser().resolve()
        if not subset_path.exists():
            print(f"--subset-source not found: {subset_path}", file=sys.stderr)
            return 1
        print(f"== Using local SUBSET_1000 source: {subset_path}")
    else:
        subset_path = _download(
            download_data.SUBSET_ARCHIVE, args.download_dir, args.token, args.force
        )
        if not args.keep_downloads:
            cleanup_targets.append(subset_path)
            cleanup_targets.append(subset_path.with_suffix(""))  # extracted dir

    # --- Step 2: Final_LLM_Reviewer_Data_Sample (LLM reviewer outputs) ----
    sample_path: Path | None = None
    if not args.no_sample:
        if args.sample_source is not None:
            sample_path = args.sample_source.expanduser().resolve()
            if not sample_path.exists():
                print(f"--sample-source not found: {sample_path}", file=sys.stderr)
                return 1
            print(f"== Using local LLM-reviewer sample source: {sample_path}")
        else:
            sample_path = _download(
                download_data.SAMPLE_ARCHIVE, args.download_dir, args.token, args.force
            )
            if not args.keep_downloads:
                cleanup_targets.append(sample_path)
                cleanup_targets.append(sample_path.with_suffix(""))

    # --- Step 3 + 4: map subset, overlay sample ---------------------------
    print("== Mapping Hugging Face layout -> Final_LLM_Reviewer_Data/")
    sys.argv = [
        "map_hf_to_aspect_layout.py",
        "--source", str(subset_path),
        "--output-dir", str(output_dir),
        "--mode", args.materialize,
        "--max-papers-per-venue", str(args.max_papers_per_venue),
    ]
    if sample_path is not None:
        sys.argv += ["--overlay", str(sample_path)]
    if args.write_env:
        sys.argv.append("--write-env")
    if args.force:
        sys.argv.append("--force")

    rc = map_hf_to_aspect_layout.main()
    if rc != 0:
        return rc

    # --- Step 5: clean up the downloaded intermediates --------------------
    if cleanup_targets:
        if args.materialize == "symlink":
            print(
                "Skipping cleanup because --materialize=symlink keeps DATA_ROOT "
                "pointing at the extracted source files. Re-run with "
                "--materialize copy (default) or pass --keep-downloads to silence.",
                file=sys.stderr,
            )
        else:
            print("== Cleaning up downloaded intermediates")
            _cleanup(cleanup_targets)

    print()
    print(f"Prepared DATA_ROOT: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
