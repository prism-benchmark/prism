import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional


REQUIRED_MANIFEST_COLUMNS = ["paper_id", "mmd_path", "reviews_json", "output_dir"]
FINAL_OUTPUT_NAME = "final_output.json"
STANDARDIZED_REVIEWS_NAME = "standardized_reviews.json"
LOG_NAME = "log.txt"
DEFAULT_CHECKPOINT_NAME = "checkpoint.json"


@dataclass(frozen=True)
class ManifestRecord:
    paper_id: str
    mmd_path: str
    reviews_json: str
    output_dir: str


@dataclass(frozen=True)
class RuntimeConfig:
    max_depth: int = 4
    retrieval_top_k: int = 3
    chunk_size: int = 1024
    ranker_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    ranker_device: str = "cuda"
    force_rerun: bool = False
    save_standardized_reviews: bool = True


@dataclass(frozen=True)
class BatchConfig(RuntimeConfig):
    num_workers: int = 1


@dataclass
class PaperRunResult:
    paper_id: str
    status: str
    output_dir: str
    final_output_path: str
    checkpoint_path: str
    log_path: str
    standardized_reviews_path: Optional[str] = None
    error_message: str = ""
    started_at: str = ""
    finished_at: str = ""
    duration_seconds: float = 0.0

    def to_status_row(self) -> Dict[str, str]:
        return {
            "paper_id": self.paper_id,
            "status": self.status,
            "output_dir": self.output_dir,
            "final_output_path": self.final_output_path,
            "checkpoint_path": self.checkpoint_path,
            "log_path": self.log_path,
            "standardized_reviews_path": self.standardized_reviews_path or "",
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": f"{self.duration_seconds:.3f}",
            "error_message": self.error_message,
        }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_manifest(manifest_path: str) -> List[ManifestRecord]:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Manifest CSV is missing a header row.")
        missing = [col for col in REQUIRED_MANIFEST_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"Manifest missing required columns: {missing}")
        records: List[ManifestRecord] = []
        for row in reader:
            if not any((row.get(col) or "").strip() for col in REQUIRED_MANIFEST_COLUMNS):
                continue
            records.append(
                ManifestRecord(
                    paper_id=(row["paper_id"] or "").strip(),
                    mmd_path=(row["mmd_path"] or "").strip(),
                    reviews_json=(row["reviews_json"] or "").strip(),
                    output_dir=(row["output_dir"] or "").strip(),
                )
            )
        return records


def resolve_manifest_record(record: ManifestRecord, manifest_path: str) -> ManifestRecord:
    manifest_dir = Path(manifest_path).resolve().parent

    def _resolve(value: str) -> str:
        path = Path(value)
        if not path.is_absolute():
            path = manifest_dir / path
        return str(path.resolve())

    return ManifestRecord(
        paper_id=record.paper_id,
        mmd_path=_resolve(record.mmd_path),
        reviews_json=_resolve(record.reviews_json),
        output_dir=_resolve(record.output_dir),
    )


def load_existing_status(status_csv_path: str) -> Dict[str, Dict[str, str]]:
    path = Path(status_csv_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {row["paper_id"]: row for row in reader if row.get("paper_id")}


def write_status_csv(status_csv_path: str, rows: Iterable[Dict[str, str]]) -> None:
    path = Path(status_csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fieldnames = [
        "paper_id",
        "status",
        "output_dir",
        "final_output_path",
        "checkpoint_path",
        "log_path",
        "standardized_reviews_path",
        "started_at",
        "finished_at",
        "duration_seconds",
        "error_message",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_jsonl(path: str, payload: Dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def final_output_exists(output_dir: str) -> bool:
    return (Path(output_dir) / FINAL_OUTPUT_NAME).exists()


def ensure_output_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
