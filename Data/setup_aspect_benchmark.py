#!/usr/bin/env python3
"""Download and install the canonical PRISM demo dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
DEFAULT_OUTPUT = THIS_DIR / "input"
CONFERENCES = ("ICLR2024", "ICLR2025", "ICLR2026", "ICML2025", "NeurIPS2025")
REVIEW_DIRS = (
    "human_reviews",
    "sea",
    "reviewer2",
    "tree",
    "deepreview",
    "cyclereview",
)

sys.path.insert(0, str(THIS_DIR))
import download_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download, extract, validate, and configure PRISM demo data."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Optional local demo_data.zip or extracted dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="DATA_ROOT destination (default: Data/input).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output directory and re-download the archive.",
    )
    parser.add_argument(
        "--keep-download",
        action="store_true",
        help="Keep Data/demo_data.zip after a successful setup.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token (default: HF_TOKEN).",
    )
    # Kept for compatibility with the existing `python run.py --setup-data` call.
    parser.add_argument("--write-env", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def safe_extract(archive_path: Path, target_dir: Path) -> None:
    target_root = target_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_path = (target_root / member.filename).resolve()
            if member_path != target_root and target_root not in member_path.parents:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        archive.extractall(target_root)


def find_data_root(root: Path) -> Path:
    # Some archives retain an older Data/input fixture alongside the current
    # dataset at input/. Prefer the current top-level layout when both exist.
    candidates = (root / "input", root / "Data" / "input", root)
    for candidate in candidates:
        if all((candidate / conference).is_dir() for conference in CONFERENCES):
            return candidate
    for candidate in root.rglob("input"):
        if all((candidate / conference).is_dir() for conference in CONFERENCES):
            return candidate
    raise ValueError(
        "Archive does not contain the expected Data/input/<conference> layout."
    )


def _file_ids(folder: Path) -> set[str]:
    ids: set[str] = set()
    for path in folder.iterdir():
        if not path.is_file():
            continue
        name = path.name
        if name.endswith("_review.json"):
            ids.add(name.removesuffix("_review.json"))
        elif name.endswith(".grobid.txt"):
            ids.add(name.removesuffix(".grobid.txt"))
        elif path.suffix in {".txt", ".json"}:
            ids.add(path.stem)
    return ids


def _nonempty_text_file(path: Path) -> bool:
    try:
        return bool(path.read_text(encoding="utf-8").strip())
    except (OSError, UnicodeError):
        return False


def _has_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _valid_review_json(path: Path, reviewer: str) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict):
        return False

    if reviewer == "human_reviews":
        reviews = data.get("reviews")
        return isinstance(reviews, list) and any(
            isinstance(review, dict) and any(_has_text(value) for value in review.values())
            for review in reviews
        )
    if reviewer == "tree":
        return _has_text(data.get("full_review"))
    if reviewer == "deepreview":
        generated = data.get("generated_review")
        if not isinstance(generated, list):
            return False
        return any(
            isinstance(item, dict)
            and isinstance(item.get("reviews"), list)
            and any(
                isinstance(review, dict) and _has_text(review.get("text"))
                for review in item["reviews"]
            )
            for item in generated
        )
    if reviewer == "cyclereview":
        generated = data.get("generated_review")
        if not isinstance(generated, dict):
            return False
        if _has_text(generated.get("content")):
            return True
        reviews = generated.get("reviews")
        return isinstance(reviews, list) and any(
            _has_text(review)
            or (isinstance(review, dict) and any(_has_text(v) for v in review.values()))
            for review in reviews
        )
    return False


def validate(data_root: Path) -> None:
    errors: list[str] = []
    for conference in CONFERENCES:
        conference_root = data_root / conference
        required = ("papers", *REVIEW_DIRS)
        for folder_name in required:
            folder = conference_root / folder_name
            if not folder.is_dir():
                errors.append(f"Missing directory: {folder}")

        papers_dir = conference_root / "papers"
        if not papers_dir.is_dir():
            continue
        paper_ids = _file_ids(papers_dir)
        if not paper_ids:
            errors.append(f"No paper files found in {papers_dir}")
            continue
        unusable_papers = [
            path.name
            for path in papers_dir.iterdir()
            if path.is_file() and not _nonempty_text_file(path)
        ]
        if unusable_papers:
            errors.append(
                f"{conference}/papers has {len(unusable_papers)} empty or unreadable file(s)"
            )
        for reviewer in REVIEW_DIRS:
            review_dir = conference_root / reviewer
            if not review_dir.is_dir():
                continue
            missing = paper_ids - _file_ids(review_dir)
            if missing:
                errors.append(
                    f"{conference}/{reviewer} is missing {len(missing)} paper(s)"
                )
            invalid = [
                path.name
                for path in review_dir.iterdir()
                if path.is_file()
                and (
                    not _nonempty_text_file(path)
                    if path.suffix == ".txt"
                    else not _valid_review_json(path, reviewer)
                )
            ]
            if invalid:
                errors.append(
                    f"{conference}/{reviewer} has {len(invalid)} unusable review file(s)"
                )

    if errors:
        raise ValueError("Dataset validation failed:\n  - " + "\n  - ".join(errors))


def update_root_env(data_root: Path) -> None:
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    else:
        example = REPO_ROOT / ".env.example"
        lines = example.read_text(encoding="utf-8").splitlines()

    setting = f"DATA_ROOT={data_root.resolve()}"
    for index, line in enumerate(lines):
        if line.startswith("DATA_ROOT="):
            lines[index] = setting
            break
    else:
        lines.insert(0, setting)
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Configured DATA_ROOT in {env_path}")


def install_source(source: Path, output_dir: Path, force: bool) -> None:
    with tempfile.TemporaryDirectory(prefix="prism-data-", dir=THIS_DIR) as temp:
        staging = Path(temp)
        if source.is_dir():
            extracted_root = source
        else:
            if source.suffix.lower() != ".zip":
                raise ValueError(f"Expected a .zip archive: {source}")
            safe_extract(source, staging)
            extracted_root = staging

        source_root = find_data_root(extracted_root)
        if output_dir.exists():
            if not force:
                validate(output_dir)
                print(f"Using existing validated dataset: {output_dir}")
                return
            shutil.rmtree(output_dir)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_root, output_dir)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()

    if output_dir.exists() and not args.force:
        try:
            validate(output_dir)
        except ValueError:
            print(
                f"Existing dataset is incomplete: {output_dir}\n"
                "Re-run with --force to replace it.",
                file=sys.stderr,
            )
            return 1
        update_root_env(output_dir)
        print(f"Dataset is ready: {output_dir}")
        return 0

    downloaded = False
    if args.source:
        source = args.source.expanduser().resolve()
        if not source.exists():
            print(f"Source not found: {source}", file=sys.stderr)
            return 1
    else:
        source = download_data.download_file(
            download_data.DEMO_ARCHIVE, THIS_DIR, args.token, args.force
        )
        downloaded = True

    try:
        install_source(source, output_dir, args.force)
        validate(output_dir)
        update_root_env(output_dir)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"Data setup failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if downloaded and not args.keep_download:
            source.unlink(missing_ok=True)

    print(f"Dataset is ready: {output_dir}")
    print("Run the pipeline with: python run.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
