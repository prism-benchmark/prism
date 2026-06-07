#!/usr/bin/env python3
"""Download the PRISM subset artifacts from Hugging Face.

The dataset README documents two subset files in
`anonymous/prism-benchmark-data`:

- `subset_1000.parquet`: lightweight metadata table.
- `SUBSET_1000.zip`: full 1,000-paper subset with file-based data.

This script intentionally downloads those files directly instead of using
`snapshot_download`, which would fetch the full PRISM dataset.
"""

from __future__ import annotations

import argparse
import os
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import quote

REPO_ID = "anonymous/prism-benchmark-data"
REPO_TYPE = "dataset"
REVISION = "main"
SUBSET_METADATA = "subset_1000.parquet"
SUBSET_ARCHIVE = "SUBSET_1000.zip"
SAMPLE_ARCHIVE = "Final_LLM_Reviewer_Data_Sample.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the PRISM SUBSET_1000 files from Hugging Face Hub."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory where subset files will be downloaded. Defaults to Data/.",
    )
    parser.add_argument(
        "--files",
        choices=("metadata", "archive", "both"),
        default="both",
        help=(
            "Subset files to download: metadata downloads subset_1000.parquet, "
            "archive downloads SUBSET_1000.zip, both downloads both files."
        ),
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract SUBSET_1000.zip after downloading it.",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=None,
        help="Directory for extracted files. Defaults to <output-dir>/SUBSET_1000.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even when Hugging Face has a cached copy.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token. Defaults to the HF_TOKEN environment variable.",
    )
    return parser.parse_args()


def download_file(
    filename: str, output_dir: Path, token: str | None, force: bool
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub is not installed; using direct HTTPS download.")
        return download_file_with_urllib(filename, output_dir, token, force)

    print(f"Downloading {filename} to {output_dir} ...")
    downloaded_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type=REPO_TYPE,
        local_dir=str(output_dir),
        token=token,
        force_download=force,
    )
    path = Path(downloaded_path)
    print(f"Saved {filename}: {path}")
    return path


def download_file_with_urllib(
    filename: str, output_dir: Path, token: str | None, force: bool
) -> Path:
    target_path = output_dir / filename
    if target_path.exists() and not force:
        print(f"Using existing file: {target_path}")
        return target_path

    encoded_repo_id = quote(REPO_ID, safe="/")
    encoded_filename = quote(filename)
    url = (
        f"https://huggingface.co/datasets/{encoded_repo_id}/resolve/"
        f"{REVISION}/{encoded_filename}"
    )
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(url, headers=headers)
    print(f"Downloading {filename} to {target_path} ...")
    try:
        with urllib.request.urlopen(request) as response:
            total_size = int(response.headers.get("Content-Length", "0"))
            downloaded = 0
            next_report = 0

            with tmp_path.open("wb") as output_file:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break

                    output_file.write(chunk)
                    downloaded += len(chunk)

                    if total_size and downloaded >= next_report:
                        percent = downloaded / total_size * 100
                        print(
                            f"Downloaded {downloaded / 1024 / 1024:.1f} MiB "
                            f"of {total_size / 1024 / 1024:.1f} MiB ({percent:.1f}%)"
                        )
                        next_report = downloaded + 100 * 1024 * 1024

        tmp_path.replace(target_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    print(f"Saved {filename}: {target_path}")
    return target_path


def extract_archive(archive_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    archive_root = archive_path.resolve().parent
    target_root = extract_dir.resolve()

    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target_path = (target_root / member.filename).resolve()
            if target_path != target_root and target_root not in target_path.parents:
                raise ValueError(
                    f"Refusing to extract {member.filename!r} outside {target_root}"
                )

        print(f"Extracting {archive_path.name} to {target_root} ...")
        archive.extractall(target_root)

    print(f"Extracted archive from {archive_root} to {target_root}")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()

    filenames: list[str] = []
    if args.files in {"metadata", "both"}:
        filenames.append(SUBSET_METADATA)
    if args.files in {"archive", "both"}:
        filenames.append(SUBSET_ARCHIVE)

    downloaded_paths = {
        filename: download_file(filename, output_dir, args.token, args.force)
        for filename in filenames
    }

    if args.extract:
        archive_path = downloaded_paths.get(SUBSET_ARCHIVE)
        if archive_path is None:
            archive_path = output_dir / SUBSET_ARCHIVE
            if not archive_path.exists():
                raise FileNotFoundError(
                    f"{SUBSET_ARCHIVE} was not downloaded and does not exist in {output_dir}"
                )

        extract_dir = (
            args.extract_dir.expanduser().resolve()
            if args.extract_dir is not None
            else output_dir / "SUBSET_1000"
        )
        extract_archive(archive_path, extract_dir)


if __name__ == "__main__":
    main()
