#!/usr/bin/env python3
"""Map the PRISM Hugging Face dataset into the Aspects_benchmarking layout.

The Hugging Face artifact (`anoyresearcher/prism_paper_data`, including the
`SUBSET_1000.zip`) ships with this directory shape:

    paper_data/
      ICLR_2024/
        json/                    # peer reviews + decisions
        txt/                     # PDF-extracted full text
        grobid_fulltext/         # GROBID full-text extraction .grobid.txt
        scraping_summary.json
      ICLR_2025/
      ICLR_2026/
      ICML_2025/
      NeurIPS_2025/

The aspect benchmarking scripts (`Aspects_benchmarking/*`) read DATA_ROOT in
this shape:

    Final_LLM_Reviewer_Data/
      ICLR2024/
        human_reviews/{paper_id}.json
        papers/{paper_id}.grobid.txt
        sea_iclr2024/  tree_iclr2024/  reviewer2_iclr2024/ ...
        paper_ids_200_iclr2024.txt
      ICLR2025/  ICLR2026/  ICML2025/
      Neurlps2025/                       # original spelling preserved

This module copies/links the Hugging Face files into that layout so the
reviewer can run every aspect script against the same DATA_ROOT.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Map Hugging Face venue folder -> aspect-benchmark conference folder.
VENUE_MAP = {
    "ICLR_2024":    "ICLR2024",
    "ICLR_2025":    "ICLR2025",
    "ICLR_2026":    "ICLR2026",
    "ICML_2025":    "ICML2025",
    "NeurIPS_2025": "Neurlps2025",  # dataset folder spelling
}

# Per-conference reviewer subfolders expected by the aspect scripts.
# These are placeholders; LLM-reviewer outputs are produced by LLM_reviewer/.
LLM_REVIEWER_SUFFIXES = {
    "ICLR2024":    ["sea_iclr2024",    "tree_iclr2024",    "reviewer2_iclr2024",
                    "deepreview_iclr2024",    "cyclereview_iclr2024"],
    "ICLR2025":    ["sea_iclr2025",    "tree_iclr2025",    "reviewer2_iclr2025",
                    "deepreview_iclr2025",    "cyclereview_iclr2025"],
    "ICLR2026":    ["sea_iclr2026",    "tree_iclr2026",    "reviewer2_iclr2026",
                    "deepreview_iclr2026",    "cyclereview_iclr2026"],
    "ICML2025":    ["sea_icml2025",    "tree_icml2025",    "reviewer2_icml2025",
                    "deepreview_icml2025",    "cyclereview_icml2025"],
    "Neurlps2025": ["sea_neurlps2025", "tree_neurips2025", "reviewer2_neurips2025",
                    "deepreview_neurips2025", "cyclereview_neurlps2025"],
}

# `paper_ids_200_*` filename per conference (the script then writes
# `paper_ids_50_*` from that list via prepare_aspect_benchmark_data.py).
ID_200_NAMES = {
    "ICLR2024":    "paper_ids_200_iclr2024.txt",
    "ICLR2025":    "paper_ids_200_iclr2025.txt",
    "ICLR2026":    "paper_ids_200_iclr2026.txt",
    "ICML2025":    "paper_ids_200_icml2025.txt",
    "Neurlps2025": "paper_ids_200_neurlps2025.txt",
}

ID_50_NAMES = {
    "ICLR2024":    ["paper_ids_50_iclr2024.txt"],
    "ICLR2025":    ["paper_ids_50_iclr2025.txt"],
    "ICLR2026":    ["paper_ids_50_iclr2026.txt"],
    "ICML2025":    ["paper_ids_50_icml2025.txt"],
    # env_loader looks up `paper_ids_50_neurips2025.txt`, the preparer
    # also writes the dataset-spelled `_neurlps2025` for compatibility.
    "Neurlps2025": ["paper_ids_50_neurips2025.txt", "paper_ids_50_neurlps2025.txt"],
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map the PRISM Hugging Face dataset (paper_data or SUBSET_1000) "
            "into the Final_LLM_Reviewer_Data layout used by Aspects_benchmarking."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help=(
            "Path to the Hugging Face dataset. Accepts a directory containing "
            "ICLR_2024/, ICLR_2025/, ... or a SUBSET_1000.zip / paper_data.zip "
            "archive that will be extracted in place."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "Final_LLM_Reviewer_Data",
        help="Destination DATA_ROOT directory (default: Data/Final_LLM_Reviewer_Data).",
    )
    parser.add_argument(
        "--mode",
        choices=("copy", "symlink", "hardlink"),
        default="symlink",
        help=(
            "How to materialize per-paper files: symlink (default, fast and "
            "small), hardlink (same partition only), or copy (portable)."
        ),
    )
    parser.add_argument(
        "--max-papers-per-venue",
        type=int,
        default=200,
        help="Cap the number of papers mapped per venue (default: 200).",
    )
    parser.add_argument(
        "--paper-id-subset-size",
        type=int,
        default=50,
        help="Subset size written to paper_ids_50_* files (default: 50).",
    )
    parser.add_argument(
        "--write-env",
        action="store_true",
        help="Write/refresh DATA_ROOT in Aspects_benchmarking/.env.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output directory.",
    )
    parser.add_argument(
        "--keep-extracted",
        action="store_true",
        help="When --source is a zip, keep the extracted folder next to it.",
    )
    parser.add_argument(
        "--overlay",
        type=Path,
        default=None,
        help=(
            "Optional Final_LLM_Reviewer_Data-shaped directory or zip "
            "(e.g. Final_LLM_Reviewer_Data_Sample.zip) to overlay on top "
            "of the mapped output. Used to drop in LLM reviewer outputs "
            "(sea_*, tree_*, reviewer2_*, deepreview_*, cyclereview_*)."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Source resolution (zip or directory)
# ---------------------------------------------------------------------------

def safe_extract(zip_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    target_root = target_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (target_root / member.filename).resolve()
            if target != target_root and target_root not in target.parents:
                raise ValueError(
                    f"Refusing to extract outside target: {member.filename!r}"
                )
        archive.extractall(target_root)


def find_dataset_root(path: Path) -> Path:
    """Return the directory that directly contains the venue folders."""
    if any((path / venue).is_dir() for venue in VENUE_MAP):
        return path
    children = [c for c in path.iterdir() if c.is_dir()]
    for child in children:
        if any((child / venue).is_dir() for venue in VENUE_MAP):
            return child
    raise FileNotFoundError(
        f"No venue folders ({', '.join(VENUE_MAP)}) found under {path}"
    )


def resolve_source(source: Path, keep_extracted: bool) -> Path:
    source = source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source does not exist: {source}")

    if source.is_file():
        if source.suffix.lower() != ".zip":
            raise ValueError(f"Only .zip archives are supported as files: {source}")
        extract_dir = source.with_suffix("")
        if not extract_dir.exists() or not any(extract_dir.iterdir()):
            print(f"Extracting {source.name} -> {extract_dir}")
            safe_extract(source, extract_dir)
        else:
            print(f"Using existing extracted directory: {extract_dir}")
        try:
            return find_dataset_root(extract_dir)
        finally:
            if not keep_extracted:
                # Caller may want to drop the extraction after mapping.
                pass

    if source.is_dir():
        return find_dataset_root(source)

    raise ValueError(f"Unsupported source: {source}")


# ---------------------------------------------------------------------------
# File materialization
# ---------------------------------------------------------------------------

def materialize(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)

    if mode == "symlink":
        try:
            os.symlink(src.resolve(), dst)
            return
        except OSError:
            pass  # fall through to copy on filesystems without symlink support
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def collect_paper_ids(venue_dir: Path) -> list[str]:
    """Return sorted paper IDs available in this Hugging Face venue folder.

    Prefers `scraping_summary.json` and falls back to listing `json/`.
    """
    summary = venue_dir / "scraping_summary.json"
    if summary.exists():
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
            ids = data.get("paper_ids") or []
            ids = [pid for pid in ids if (venue_dir / "json" / f"{pid}.json").exists()]
            if ids:
                return sorted(set(ids))
        except (OSError, json.JSONDecodeError):
            pass

    json_dir = venue_dir / "json"
    if not json_dir.is_dir():
        return []
    return sorted({p.stem for p in json_dir.glob("*.json")})


def find_paper_text(venue_dir: Path, paper_id: str) -> Path | None:
    """Pick the best paper text source for `papers/{paper_id}.grobid.txt`."""
    candidates = [
        venue_dir / "grobid_fulltext"  / f"{paper_id}.grobid.txt",
        venue_dir / "txt"              / f"{paper_id}.txt",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def map_venue(
    hf_root: Path,
    out_root: Path,
    hf_name: str,
    out_name: str,
    mode: str,
    max_papers: int,
) -> tuple[int, int]:
    venue_in  = hf_root / hf_name
    venue_out = out_root / out_name
    venue_out.mkdir(parents=True, exist_ok=True)

    paper_ids = collect_paper_ids(venue_in)
    if max_papers > 0:
        paper_ids = paper_ids[:max_papers]

    if not paper_ids:
        print(f"  [{out_name}] no paper IDs found in {venue_in}")
        return (0, 0)

    human_dir  = venue_out / "human_reviews"
    papers_dir = venue_out / "papers"
    human_dir.mkdir(exist_ok=True)
    papers_dir.mkdir(exist_ok=True)

    n_reviews = 0
    n_papers  = 0
    for pid in paper_ids:
        review_src = venue_in / "json" / f"{pid}.json"
        if review_src.exists():
            materialize(review_src, human_dir / f"{pid}.json", mode)
            n_reviews += 1

        text_src = find_paper_text(venue_in, pid)
        if text_src is not None:
            materialize(text_src, papers_dir / f"{pid}.grobid.txt", mode)
            n_papers += 1

    # Placeholder LLM reviewer dirs so prepare_aspect_benchmark_data.py and
    # the aspect scripts can detect the layout immediately.
    for sub in LLM_REVIEWER_SUFFIXES.get(out_name, []):
        (venue_out / sub).mkdir(exist_ok=True)

    # Write 200-paper id list so prepare_aspect_benchmark_data.py can derive
    # the 50-paper subset(s) used by robustness experiments.
    ids_file = venue_out / ID_200_NAMES[out_name]
    ids_file.write_text("\n".join(paper_ids) + "\n", encoding="utf-8")

    print(
        f"  [{out_name}] {len(paper_ids)} ids -> "
        f"{n_reviews} reviews, {n_papers} papers (mode={mode})"
    )
    return (n_reviews, n_papers)


# ---------------------------------------------------------------------------
# Subset id files + .env
# ---------------------------------------------------------------------------

def write_subset_files(out_root: Path, subset_size: int) -> None:
    for out_name, id200_name in ID_200_NAMES.items():
        venue_dir = out_root / out_name
        ids_file = venue_dir / id200_name
        if not ids_file.exists():
            continue
        ids = [ln.strip() for ln in ids_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        subset = ids[: min(subset_size, len(ids))]
        for fname in ID_50_NAMES[out_name]:
            (venue_dir / fname).write_text("\n".join(subset) + "\n", encoding="utf-8")


def overlay_sample(out_root: Path, overlay: Path) -> None:
    """Merge a Final_LLM_Reviewer_Data-shaped directory/zip onto out_root.

    Existing files are overwritten so the sample LLM reviewer outputs replace
    the empty placeholders created by `map_venue`.
    """
    overlay = overlay.expanduser().resolve()
    if not overlay.exists():
        raise FileNotFoundError(f"--overlay path does not exist: {overlay}")

    if overlay.is_file():
        if overlay.suffix.lower() != ".zip":
            raise ValueError(f"--overlay file must be a .zip: {overlay}")
        extract_dir = overlay.with_suffix("")
        if not extract_dir.exists() or not any(extract_dir.iterdir()):
            print(f"Extracting overlay {overlay.name} -> {extract_dir}")
            safe_extract(overlay, extract_dir)
        overlay_root = extract_dir
    else:
        overlay_root = overlay

    # Drill into wrapper dirs (e.g. Final_LLM_Reviewer_Data_Sample/Final_LLM_Reviewer_Data/)
    if not any((overlay_root / v).is_dir() for v in VENUE_MAP.values()):
        children = [c for c in overlay_root.iterdir() if c.is_dir()]
        for child in children:
            if any((child / v).is_dir() for v in VENUE_MAP.values()):
                overlay_root = child
                break

    print(f"Overlaying {overlay_root} -> {out_root}")
    copied = 0
    for venue in VENUE_MAP.values():
        src_venue = overlay_root / venue
        if not src_venue.is_dir():
            continue
        for item in src_venue.iterdir():
            dst = out_root / venue / item.name
            if item.is_dir():
                dst.mkdir(parents=True, exist_ok=True)
                for sub in item.rglob("*"):
                    if sub.is_file():
                        rel = sub.relative_to(item)
                        target = dst / rel
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(sub, target)
                        copied += 1
            elif item.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dst)
                copied += 1
    print(f"Overlay copied {copied} files")


def update_aspect_env(data_root: Path) -> None:
    env_path = REPO_ROOT / "Aspects_benchmarking" / ".env"
    env_path.parent.mkdir(parents=True, exist_ok=True)
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    else:
        example = env_path.with_name(".env.example")
        lines = example.read_text(encoding="utf-8").splitlines() if example.exists() else []

    new_line = f"DATA_ROOT={data_root.resolve()}"
    found = False
    for i, line in enumerate(lines):
        if line.startswith("DATA_ROOT="):
            lines[i] = new_line
            found = True
            break
    if not found:
        lines.insert(0, new_line)
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Updated {env_path.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    if args.paper_id_subset_size <= 0:
        raise ValueError("--paper-id-subset-size must be positive")

    out_root = args.output_dir.expanduser().resolve()
    if out_root.exists() and any(out_root.iterdir()):
        if not args.force:
            print(
                f"Output directory is not empty: {out_root}\n"
                "Pass --force to replace it, or choose a different --output-dir.",
                file=sys.stderr,
            )
            return 1
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    hf_root = resolve_source(args.source, args.keep_extracted)
    print(f"Hugging Face dataset root: {hf_root}")
    print(f"Aspect-benchmark output  : {out_root}")
    print(f"Materialization mode     : {args.mode}")

    total_reviews = total_papers = 0
    for hf_name, out_name in VENUE_MAP.items():
        if not (hf_root / hf_name).is_dir():
            print(f"  [skip] missing venue folder: {hf_name}")
            continue
        r, p = map_venue(
            hf_root, out_root, hf_name, out_name,
            args.mode, args.max_papers_per_venue,
        )
        total_reviews += r
        total_papers  += p

    write_subset_files(out_root, args.paper_id_subset_size)
    print(f"Mapped {total_reviews} reviews and {total_papers} papers across {len(VENUE_MAP)} venues.")

    if args.overlay is not None:
        overlay_sample(out_root, args.overlay)
        # Re-derive 50-paper subsets in case the overlay shipped its own 200-id list.
        write_subset_files(out_root, args.paper_id_subset_size)

    if args.write_env:
        update_aspect_env(out_root)

    print()
    print("Next:")
    print(f"  export DATA_ROOT={out_root}")
    print("  cd Aspects_benchmarking && pip install -r requirements.txt")
    print("  python depth_of_analysis/run_human_mimo.py --conference ICLR2025")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
