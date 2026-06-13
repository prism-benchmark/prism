#!/usr/bin/env python3
"""Prepare PRISM aspect-benchmark data for reviewer experiments.

The aspect benchmarking scripts all read one DATA_ROOT with this shape:

    DATA_ROOT/
      ICLR2024/
      ICLR2025/
      ICLR2026/
      ICML2025/
      NeurIPS2025/

This utility accepts either an extracted dataset directory or a zip archive,
copies/extracts it into that layout, creates the 50-paper id files expected by
robustness scripts, validates the reviewer subfolders, and can update
Aspects_benchmarking/.env with the prepared DATA_ROOT.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ASPECT_ENV = REPO_ROOT / "Aspects_benchmarking" / ".env"

CONFERENCES = {
    "ICLR2024": {
        "id_200": "paper_ids_200_iclr2024.txt",
        "id_50": ["paper_ids_50_iclr2024.txt"],
        "subdirs": [
            "human_reviews",
            "papers",
            "sea",
            "tree",
            "reviewer2",
            "deepreview",
            "cyclereview",
        ],
    },
    "ICLR2025": {
        "id_200": "paper_ids_200_iclr2025.txt",
        "id_50": ["paper_ids_50_iclr2025.txt"],
        "subdirs": [
            "human_reviews",
            "papers",
            "sea",
            "tree",
            "reviewer2",
            "deepreview",
            "cyclereview",
        ],
    },
    "ICLR2026": {
        "id_200": "paper_ids_200_iclr2026.txt",
        "id_50": ["paper_ids_50_iclr2026.txt"],
        "subdirs": [
            "human_reviews",
            "papers",
            "sea",
            "tree",
            "reviewer2",
            "deepreview",
            "cyclereview",
        ],
    },
    "ICML2025": {
        "id_200": "paper_ids_200_icml2025.txt",
        "id_50": ["paper_ids_50_icml2025.txt"],
        "subdirs": [
            "human_reviews",
            "papers",
            "sea",
            "tree",
            "reviewer2",
            "deepreview",
            "cyclereview",
        ],
    },
    "NeurIPS2025": {
        "id_200": "paper_ids_200_neurips2025.txt",
        "id_50": ["paper_ids_50_neurips2025.txt"],
        "subdirs": [
            "human_reviews",
            "papers",
            "sea",
            "tree",
            "reviewer2",
            "deepreview",
            "cyclereview",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare downloaded PRISM data for Aspects_benchmarking scripts."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parent / "Final_LLM_Reviewer_Data_Sample.zip",
        help="Dataset zip or extracted dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "Final_LLM_Reviewer_Data",
        help="Prepared DATA_ROOT directory used by aspect benchmark scripts.",
    )
    parser.add_argument(
        "--paper-id-subset-size",
        type=int,
        default=50,
        help="Number of ids to write to paper_ids_50_* files.",
    )
    parser.add_argument(
        "--write-env",
        action="store_true",
        help="Create/update Aspects_benchmarking/.env with DATA_ROOT.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output directory.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any configured paper id has missing files.",
    )
    return parser.parse_args()


def safe_extract(zip_path: Path, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target_root = target_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target_path = (target_root / member.filename).resolve()
            if target_path != target_root and target_root not in target_path.parents:
                raise ValueError(f"Refusing to extract outside target: {member.filename}")
        archive.extractall(target_root)

    roots = [p for p in target_dir.iterdir() if p.is_dir() and has_conference_dirs(p)]
    return roots[0] if len(roots) == 1 else target_dir


def has_conference_dirs(path: Path) -> bool:
    return any((path / conf).is_dir() for conf in CONFERENCES)


def resolve_source(source: Path, output_dir: Path, force: bool) -> Path:
    source = source.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source does not exist: {source}")

    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    elif output_dir.exists() and any(output_dir.iterdir()) and source.resolve() != output_dir:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Pass --force to replace it, or choose a different --output-dir."
        )

    if source.is_file():
        if source.suffix.lower() != ".zip":
            raise ValueError(f"Only .zip archives are supported as file sources: {source}")
        print(f"Extracting {source} -> {output_dir}")
        extracted_root = safe_extract(source, output_dir)
        if extracted_root != output_dir:
            move_children(extracted_root, output_dir)
            shutil.rmtree(extracted_root, ignore_errors=True)
        return output_dir

    if source.resolve() == output_dir.resolve():
        return output_dir

    print(f"Copying {source} -> {output_dir}")
    shutil.copytree(source, output_dir, dirs_exist_ok=True)
    return output_dir


def move_children(source: Path, target: Path) -> None:
    for child in source.iterdir():
        destination = target / child.name
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        shutil.move(str(child), str(destination))


def read_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def write_ids(path: Path, ids: list[str]) -> None:
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")


def create_subset_files(data_root: Path, subset_size: int) -> None:
    for conf, spec in CONFERENCES.items():
        conf_dir = data_root / conf
        ids_path = conf_dir / spec["id_200"]
        if not ids_path.exists():
            continue
        ids = read_ids(ids_path)
        subset_ids = ids[: min(subset_size, len(ids))]
        for filename in spec["id_50"]:
            out_path = conf_dir / filename
            write_ids(out_path, subset_ids)
            print(f"Wrote {out_path.relative_to(data_root)} ({len(subset_ids)} ids)")


def matching_file_count(directory: Path, paper_id: str) -> int:
    patterns = [
        f"{paper_id}.txt",
        f"{paper_id}.json",
        f"{paper_id}.grobid.txt",
        f"{paper_id}_review.json",
    ]
    return sum(1 for name in patterns if (directory / name).exists())


def validate(data_root: Path, strict: bool) -> int:
    errors: list[str] = []
    warnings: list[str] = []

    for conf, spec in CONFERENCES.items():
        conf_dir = data_root / conf
        if not conf_dir.is_dir():
            errors.append(f"Missing conference folder: {conf}")
            continue

        ids_path = conf_dir / spec["id_200"]
        ids = read_ids(ids_path) if ids_path.exists() else []
        if not ids:
            errors.append(f"Missing or empty id file: {conf}/{spec['id_200']}")

        for subdir in spec["subdirs"]:
            path = conf_dir / subdir
            if not path.is_dir():
                errors.append(f"Missing required folder: {conf}/{subdir}")
                continue
            if ids:
                available = sum(1 for paper_id in ids if matching_file_count(path, paper_id))
                if available == 0:
                    warnings.append(f"No files matching listed ids in {conf}/{subdir}")
                elif available < len(ids):
                    warnings.append(
                        f"{conf}/{subdir}: {available}/{len(ids)} listed ids have files"
                    )

    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)

    if errors or (strict and warnings):
        return 1
    return 0


def update_env(data_root: Path) -> None:
    ASPECT_ENV.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str]
    if ASPECT_ENV.exists():
        lines = ASPECT_ENV.read_text(encoding="utf-8").splitlines()
    else:
        example = ASPECT_ENV.with_name(".env.example")
        lines = example.read_text(encoding="utf-8").splitlines() if example.exists() else []

    new_line = f"DATA_ROOT={data_root.resolve()}"
    found = False
    for index, line in enumerate(lines):
        if line.startswith("DATA_ROOT="):
            lines[index] = new_line
            found = True
            break
    if not found:
        lines.insert(0, new_line)

    ASPECT_ENV.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Updated {ASPECT_ENV.relative_to(REPO_ROOT)}")


def main() -> int:
    args = parse_args()
    if args.paper_id_subset_size <= 0:
        raise ValueError("--paper-id-subset-size must be positive")

    data_root = resolve_source(args.source, args.output_dir, args.force)
    create_subset_files(data_root, args.paper_id_subset_size)
    status = validate(data_root, args.strict)

    if args.write_env:
        update_env(data_root)

    print(f"Prepared DATA_ROOT={data_root.resolve()}")
    if status == 0:
        print("Validation passed.")
    else:
        print("Validation failed.", file=sys.stderr)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
