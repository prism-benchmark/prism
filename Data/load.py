"""Load and unpack the PRISM dataset."""
import zipfile
from pathlib import Path
from huggingface_hub import snapshot_download
import pandas as pd

REPO_ID = "anoyresearcher/prism_paper_data"


def load_parquet(subset: bool = False) -> pd.DataFrame:
    """Load papers.parquet (or subset_1000.parquet).

    Args:
        subset: If True, load the 1000-paper subset instead of the full dataset.

    Returns:
        DataFrame with all papers.
    """
    local_dir = snapshot_download(repo_id=REPO_ID, repo_type="dataset")
    filename = "subset_1000.parquet" if subset else "papers.parquet"
    path = Path(local_dir) / filename
    print(f"Loading {path}...")
    return pd.read_parquet(path)


def unpack_venue(venue: str, dest: str = "./extracted") -> Path:
    """Unpack a venue zip archive.

    Args:
        venue: Venue name (e.g. "ICLR_2025", "ICML_2025", "NeurIPS_2025").
        dest: Destination directory for extracted files.

    Returns:
        Path to the extracted venue directory.
    """
    local_dir = snapshot_download(repo_id=REPO_ID, repo_type="dataset")
    zip_path = Path(local_dir) / f"{venue}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Archive not found: {zip_path}")

    dest_path = Path(dest) / venue
    print(f"Unpacking {zip_path} -> {dest_path}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_path)
    print("Done.")
    return dest_path


def unpack_subset(dest: str = "./extracted") -> Path:
    """Unpack the 1000-paper subset zip.

    Args:
        dest: Destination directory for extracted files.

    Returns:
        Path to the extracted subset directory.
    """
    local_dir = snapshot_download(repo_id=REPO_ID, repo_type="dataset")
    zip_path = Path(local_dir) / "SUBSET_1000.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Archive not found: {zip_path}")

    dest_path = Path(dest) / "SUBSET_1000"
    print(f"Unpacking {zip_path} -> {dest_path}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_path)
    print("Done.")
    return dest_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and unpack PRISM dataset")
    parser.add_argument("--subset", action="store_true", help="Use 1000-paper subset")
    parser.add_argument("--parquet-only", action="store_true", help="Only load parquet, skip unpack")
    parser.add_argument("--venue", help="Unpack specific venue (e.g. ICLR_2025)")
    parser.add_argument("--dest", default="./extracted", help="Extraction directory")
    args = parser.parse_args()

    df = load_parquet(subset=args.subset)
    print(f"Loaded {len(df)} papers, {len(df.columns)} columns")

    if not args.parquet_only:
        if args.venue:
            unpack_venue(args.venue, args.dest)
        elif args.subset:
            unpack_subset(args.dest)
        else:
            for venue in ["ICLR_2024", "ICLR_2025", "ICLR_2026", "ICML_2025", "NeurIPS_2025"]:
                try:
                    unpack_venue(venue, args.dest)
                except FileNotFoundError:
                    print(f"Skipping {venue} (archive not found)")
