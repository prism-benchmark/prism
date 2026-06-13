"""
run_constructiveness.py — Separate evaluation runner for Constructiveness pipeline.

Evaluates Human reviews and SEA (LLM) reviews independently,
saving results into separate subfolders.

Output layout:
    output/human/all_results.jsonl        — one record per paper (all human reviewers, ICLR2024)
    output/sea/all_results.jsonl          — one record per paper (SEA review, ICLR2024)
    output/icml2025/human/all_results_lite.jsonl — ICML2025 human reviews (200 papers)

Usage examples:
    python run_constructiveness.py --mode human
    python run_constructiveness.py --mode sea
    python run_constructiveness.py --mode both
    python run_constructiveness.py --mode reviewer2 --conf iclr2024
    python run_constructiveness.py --mode reviewer2 --conf iclr2025 --provider gemini --model gemini-2.5-flash
    python run_constructiveness.py --mode tree --conf iclr2024
    python run_constructiveness.py --mode cyclereview --conf iclr2024
    python run_constructiveness.py --mode icml_human
    python run_constructiveness.py --mode neurips_human
    python run_constructiveness.py --mode sea   --limit 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

append_lock = threading.Lock()
print_lock = threading.Lock()

# ── Resolve paths ─────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_FI = os.path.normpath(os.path.join(_HERE, "..", "flaw_identification"))
sys.path.insert(0, _FI)

from dotenv import load_dotenv

_REPO_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
for _p in [
    os.path.join(_REPO_ROOT, ".env"),
    os.path.join(_HERE, ".env"),
    os.path.join(_FI, ".env"),
]:
    if os.path.exists(_p):
        load_dotenv(_p, override=False)

from paths_config import conf_path as _conf_path, reviewer_dir as _reviewer_dir

from src.evaluator import ConstructivenessEvaluator
from src.metrics import compute_review_metrics
from src.utils import (
    format_human_review_full,
    format_human_review_full_icml,
    format_human_review_full_neurips,
    get_cyclereview_pairs_from_ids,
    get_paper_pairs,
    get_paper_pairs_from_ids,
    load_cyclereview_first_text,
    load_cyclereview_metadata,
    load_deepreview_text,
    load_human_meta_json,
    load_llm_txt,
    load_paper_metadata,
    load_paper_metadata_icml,
    load_paper_metadata_neurips,
    load_reviewer2_txt,
    load_tree_review_text,
)

# ── Path constants ─────────────────────────────────────────────────────────────
_DATA = os.path.normpath(os.path.join(_HERE, "..", "data"))
# _SEA_ROOT    = os.path.join(_DATA, "iclr2024", "sea")
_SEA_ROOT = _conf_path("ICLR2024")
HUMAN_FOLDER = _reviewer_dir("ICLR2024", "human")
SEA_FOLDER = _reviewer_dir("ICLR2024", "sea")
REVIEWER2_FOLDER = _reviewer_dir("ICLR2024", "reviewer2")
DEEPREVIEW_FOLDER = _reviewer_dir("ICLR2024", "deepreview")
# TREE_FOLDER       = os.path.join(_DATA, "tree_iclr2024_full")    # _review.json files
TREE_FOLDER = _reviewer_dir("ICLR2024", "tree")
OUTPUT_ROOT = os.environ.get("CONSTRUCTIVENESS_OUTPUT_ROOT") or os.path.join(
    _HERE, "output"
)

HUMAN_OUTPUT = os.path.join(OUTPUT_ROOT, "iclr2026", "human", "all_results_lite.jsonl")
SEA_OUTPUT = os.path.join(OUTPUT_ROOT, "neurips2025", "sea", "all_results_lite.jsonl")
REVIEWER2_OUTPUT = os.path.join(
    OUTPUT_ROOT, "iclr2024", "reviewer2", "all_results_lite.jsonl"
)
DEEPREVIEW_OUTPUT = os.path.join(
    OUTPUT_ROOT, "neurips2025", "deepreview_neurips2025", "all_results_lite.jsonl"
)
TREE_OUTPUT = os.path.join(OUTPUT_ROOT, "iclr2025", "tree", "all_results_lite.jsonl")

# ── ICML 2025 ──────────────────────────────────────────────────────────────────
_ICML2025_ROOT = _conf_path("ICML2025")
ICML2025_HUMAN_FOLDER = os.path.join(_ICML2025_ROOT, "human_reviews")  # .json files
ICML2025_PAPER_IDS = os.path.join(_ICML2025_ROOT, "paper_ids_200_icml2025.txt")
ICML2025_HUMAN_OUTPUT = os.path.join(
    OUTPUT_ROOT, "icml2025", "human", "all_results_lite.jsonl"
)

# ── NeurIPS 2025 ───────────────────────────────────────────────────────────────
_NEURIPS2025_ROOT = _conf_path("NeurIPS2025")
NEURIPS2025_HUMAN_FOLDER = os.path.join(
    _NEURIPS2025_ROOT, "human_reviews"
)  # .json files
NEURIPS2025_PAPER_IDS = os.path.join(_NEURIPS2025_ROOT, "paper_ids_200_neurips2025.txt")
NEURIPS2025_HUMAN_OUTPUT = os.path.join(
    OUTPUT_ROOT, "neurips2025", "human", "all_results_lite.jsonl"
)

# ── Per-conference root paths ──────────────────────────────────────────────────
_ICLR2024_ROOT = _conf_path("ICLR2024")
_ICLR2025_ROOT = _conf_path("ICLR2025")
_ICLR2026_ROOT = _conf_path("ICLR2026")

# ── Tree review — per-conference ───────────────────────────────────────────────
TREE_FOLDERS: dict[str, str] = {
    conf.lower(): _reviewer_dir(conf, "tree")
    for conf in ("ICLR2024", "ICLR2025", "ICLR2026", "ICML2025", "NeurIPS2025")
}

TREE_HUMAN_FOLDERS: dict[str, str] = {
    "iclr2024": os.path.join(_ICLR2024_ROOT, "human_reviews"),
    "iclr2025": os.path.join(_ICLR2025_ROOT, "human_reviews"),
    "iclr2026": os.path.join(_ICLR2026_ROOT, "human_reviews"),
    "icml2025": os.path.join(_ICML2025_ROOT, "human_reviews"),
    "neurips2025": os.path.join(_NEURIPS2025_ROOT, "human_reviews"),
}

TREE_PAPER_IDS: dict[str, str] = {
    "iclr2024": os.path.join(_ICLR2024_ROOT, "paper_ids_200_iclr2024.txt"),
    "iclr2025": os.path.join(_ICLR2025_ROOT, "paper_ids_200_iclr2025.txt"),
    "iclr2026": os.path.join(_ICLR2026_ROOT, "paper_ids_200_iclr2026.txt"),
    "icml2025": os.path.join(_ICML2025_ROOT, "paper_ids_200_icml2025.txt"),
    "neurips2025": os.path.join(_NEURIPS2025_ROOT, "paper_ids_200_neurips2025.txt"),
}

TREE_OUTPUTS: dict[str, str] = {
    "iclr2024": os.path.join(
        OUTPUT_ROOT, "iclr2024", "tree_2", "all_results_lite.jsonl"
    ),
    "iclr2025": os.path.join(OUTPUT_ROOT, "iclr2025", "tree", "all_results_lite.jsonl"),
    "iclr2026": os.path.join(
        OUTPUT_ROOT, "iclr2026", "tree_2", "all_results_lite.jsonl"
    ),
    "icml2025": os.path.join(
        OUTPUT_ROOT, "icml2025", "tree_2", "all_results_lite.jsonl"
    ),
    "neurips2025": os.path.join(
        OUTPUT_ROOT, "neurips2025", "tree_2", "all_results_lite.jsonl"
    ),
}

# ── Reviewer2 — per-conference ────────────────────────────────────────────────
REVIEWER2_FOLDERS: dict[str, str] = {
    conf.lower(): _reviewer_dir(conf, "reviewer2")
    for conf in ("ICLR2024", "ICLR2025", "ICLR2026", "ICML2025", "NeurIPS2025")
}

REVIEWER2_HUMAN_FOLDERS: dict[str, str] = {
    "iclr2024": os.path.join(_ICLR2024_ROOT, "human_reviews"),
    "iclr2025": os.path.join(_ICLR2025_ROOT, "human_reviews"),
    "iclr2026": os.path.join(_ICLR2026_ROOT, "human_reviews"),
    "icml2025": os.path.join(_ICML2025_ROOT, "human_reviews"),
    "neurips2025": os.path.join(_NEURIPS2025_ROOT, "human_reviews"),
}

REVIEWER2_PAPER_IDS: dict[str, str] = {
    "iclr2024": os.path.join(_ICLR2024_ROOT, "paper_ids_200_iclr2024.txt"),
    "iclr2025": os.path.join(_ICLR2025_ROOT, "paper_ids_200_iclr2025.txt"),
    "iclr2026": os.path.join(_ICLR2026_ROOT, "paper_ids_200_iclr2026.txt"),
    "icml2025": os.path.join(_ICML2025_ROOT, "paper_ids_200_icml2025.txt"),
    "neurips2025": os.path.join(_NEURIPS2025_ROOT, "paper_ids_200_neurips2025.txt"),
}

REVIEWER2_OUTPUTS: dict[str, str] = {
    "iclr2024": os.path.join(
        OUTPUT_ROOT, "iclr2024", "reviewer2", "all_results_lite.jsonl"
    ),
    "iclr2025": os.path.join(
        OUTPUT_ROOT, "iclr2025", "reviewer2", "all_results_lite.jsonl"
    ),
    "iclr2026": os.path.join(
        OUTPUT_ROOT, "iclr2026", "reviewer2", "all_results_lite.jsonl"
    ),
    "icml2025": os.path.join(
        OUTPUT_ROOT, "icml2025", "reviewer2", "all_results_lite.jsonl"
    ),
    "neurips2025": os.path.join(
        OUTPUT_ROOT, "neurips2025", "reviewer2", "all_results_lite.jsonl"
    ),
}

# ── CycleReview — per-conference ───────────────────────────────────────────────
CYCLEREVIEW_FOLDERS: dict[str, str] = {
    conf.lower(): _reviewer_dir(conf, "cyclereview")
    for conf in ("ICLR2024", "ICLR2025", "ICLR2026", "ICML2025", "NeurIPS2025")
}

CYCLEREVIEW_PAPER_IDS: dict[str, str] = {
    "iclr2024": os.path.join(_ICLR2024_ROOT, "paper_ids_200_iclr2024.txt"),
    "iclr2025": os.path.join(_ICLR2025_ROOT, "paper_ids_200_iclr2025.txt"),
    "iclr2026": os.path.join(_ICLR2026_ROOT, "paper_ids_200_iclr2026.txt"),
    "icml2025": os.path.join(_ICML2025_ROOT, "paper_ids_200_icml2025.txt"),
    "neurips2025": os.path.join(_NEURIPS2025_ROOT, "paper_ids_200_neurips2025.txt"),
}

CYCLEREVIEW_OUTPUTS: dict[str, str] = {
    "iclr2024": os.path.join(
        OUTPUT_ROOT, "iclr2024", "cyclereview", "all_results_lite.jsonl"
    ),
    "iclr2025": os.path.join(
        OUTPUT_ROOT, "iclr2025", "cyclereview", "all_results_lite.jsonl"
    ),
    "iclr2026": os.path.join(
        OUTPUT_ROOT, "iclr2026", "cyclereview", "all_results_lite.jsonl"
    ),
    "icml2025": os.path.join(
        OUTPUT_ROOT, "icml2025", "cyclereview", "all_results_lite.jsonl"
    ),
    "neurips2025": os.path.join(
        OUTPUT_ROOT, "neurips2025", "cyclereview", "all_results_lite.jsonl"
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Constructiveness evaluation — human and SEA reviews separately.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=[
            "human",
            "sea",
            "reviewer2",
            "deepreview",
            "tree",
            "both",
            "icml_human",
            "neurips_human",
            "cyclereview",
        ],
        default="both",
        help=(
            "Which reviews to evaluate:\n"
            "  human         — human peer-reviews only (ICLR)\n"
            "  sea           — SEA (LLM) reviews only\n"
            "  reviewer2     — reviewer2 LLM reviews (use --conf for specific conference)\n"
            "  deepreview    — DeepReview LLM reviews only (reviewer_id=1)\n"
            "  tree          — Tree review LLM reviews only (use --conf)\n"
            "  both          — human + sea (default)\n"
            "  icml_human    — ICML2025 human peer-reviews (200 papers)\n"
            "  neurips_human — NeurIPS2025 human peer-reviews (200 papers)\n"
            "  cyclereview   — CycleReview LLM reviews (use --conf)"
        ),
    )
    p.add_argument(
        "--conf",
        choices=["iclr2024", "iclr2025", "iclr2026", "icml2025", "neurips2025"],
        default="iclr2024",
        help=(
            "Conference to evaluate when --mode reviewer2, cyclereview or tree:\n"
            "  iclr2024 / iclr2025 / iclr2026 / icml2025 / neurips2025\n"
            "(default: iclr2024)"
        ),
    )
    p.add_argument(
        "--provider",
        choices=["gemini", "azure"],
        default=None,
        help="LLM provider (default: gemini).",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Override model/deployment name for the provider.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N papers (for quick tests).",
    )
    p.add_argument(
        "--with-paper",
        action="store_true",
        default=False,
        help="Include paper text as context (higher accuracy, more tokens).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, sequential).",
    )
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def discover_pairs() -> list[tuple[str, str, str]]:
    """Discover all (paper_id, human_path, sea_path) pairs from the data folders."""
    pairs = get_paper_pairs(HUMAN_FOLDER, SEA_FOLDER)
    pairs.sort(key=lambda t: t[0])  # deterministic order
    return pairs


def discover_reviewer2_pairs(
    reviewer2_folder: str,
    human_folder: str,
    paper_ids_file: str | None = None,
) -> list[tuple[str, str, str]]:
    """Discover all (paper_id, human_path, reviewer2_path) pairs.

    Matches reviewer2 .txt files against human .json files.
    If paper_ids_file is given, only papers listed there are included.
    Returns only papers that have both files.
    """
    import glob

    # Load allowed paper IDs (if provided)
    allowed: set[str] | None = None
    if paper_ids_file and os.path.exists(paper_ids_file):
        with open(paper_ids_file, "r", encoding="utf-8") as f:
            allowed = {ln.strip() for ln in f if ln.strip()}

    r2_files = glob.glob(os.path.join(reviewer2_folder, "*.txt"))
    pairs = []
    for r2_path in r2_files:
        basename = os.path.basename(r2_path)
        paper_id = os.path.splitext(basename)[0]

        if allowed is not None and paper_id not in allowed:
            continue

        h_path = os.path.join(human_folder, f"{paper_id}.json")
        if os.path.exists(h_path):
            pairs.append((paper_id, h_path, r2_path))
        else:
            print(f"  [WARNING] Missing human JSON for reviewer2 paper {paper_id}")
    pairs.sort(key=lambda t: t[0])
    return pairs


def discover_deepreview_pairs() -> list[tuple[str, str, str]]:
    """Discover all (paper_id, human_path, deepreview_path) pairs.

    Matches deepreview_iclr2024 .json files against human .json files.
    Only papers that exist in both datasets are returned.
    """
    import glob

    dr_files = glob.glob(os.path.join(DEEPREVIEW_FOLDER, "*.json"))
    pairs = []
    for dr_path in dr_files:
        basename = os.path.basename(dr_path)
        paper_id = os.path.splitext(basename)[0]
        h_path = os.path.join(HUMAN_FOLDER, f"{paper_id}.json")
        if os.path.exists(h_path):
            pairs.append((paper_id, h_path, dr_path))
    pairs.sort(key=lambda t: t[0])
    return pairs


def discover_tree_pairs(
    tree_folder: str,
    human_folder: str,
    paper_ids_file: str | None = None,
) -> list[tuple[str, str, str]]:
    """Discover all (paper_id, human_path, tree_path) pairs.

    Matches *_review.json files in tree_folder against human .json files.
    If paper_ids_file is given, only papers listed there are included.
    """
    import glob

    # Load allowed paper IDs (if provided)
    allowed: set[str] | None = None
    if paper_ids_file and os.path.exists(paper_ids_file):
        with open(paper_ids_file, "r", encoding="utf-8") as f:
            allowed = {ln.strip() for ln in f if ln.strip()}

    tree_files = glob.glob(os.path.join(tree_folder, "*_review.json"))
    pairs = []
    for tree_path in tree_files:
        basename = os.path.basename(tree_path)
        paper_id = basename.replace("_review.json", "")
        if allowed is not None and paper_id not in allowed:
            continue
        h_path = os.path.join(human_folder, f"{paper_id}.json")
        if os.path.exists(h_path):
            pairs.append((paper_id, h_path, tree_path))
        else:
            print(f"  [WARNING] Missing human JSON for tree paper {paper_id}")
    pairs.sort(key=lambda t: t[0])
    return pairs


def load_processed_ids(jsonl_path: str) -> set[str]:
    """Read a results JSONL and return the set of already-processed paper_ids."""
    processed: set[str] = set()
    if not os.path.exists(jsonl_path):
        return processed
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                pid = rec.get("paper_id")
                if pid:
                    processed.add(pid)
            except json.JSONDecodeError:
                continue
    return processed


def append_record(jsonl_path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with append_lock:
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _pct(done: int, total: int) -> str:
    if total == 0:
        return "0%"
    return f"{done / total * 100:.1f}%"


def _build_evaluator(args: argparse.Namespace) -> ConstructivenessEvaluator:
    provider = args.provider
    model = args.model
    api_key = None
    if provider is not None:
        if provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
        else:
            api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    return ConstructivenessEvaluator(
        provider=provider,
        api_key=api_key,
        model=model,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Human evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def process_human_paper(
    paper_id: str,
    h_path: str,
    evaluator: ConstructivenessEvaluator,
    with_paper: bool = False,
    paper_text: Optional[str] = None,
) -> dict:
    """Score every human reviewer for one paper. Returns a single record."""
    human_data = load_human_meta_json(h_path)
    metadata = load_paper_metadata(human_data)

    human_list = (
        human_data.get("reviews", []) if isinstance(human_data, dict) else human_data
    )

    reviewer_results = []
    for idx, review_obj in enumerate(human_list):
        reviewer_id = f"Human_{idx + 1}"
        review_text = format_human_review_full(review_obj)

        if not review_text.strip():
            reviewer_results.append(
                {
                    "reviewer_id": reviewer_id,
                    "status": "empty_input",
                    "atomic_comments": [],
                    "metrics": None,
                }
            )
            continue

        scored = evaluator.score_review(review_text, reviewer_id, paper_text)
        metrics = compute_review_metrics(scored["atomic_comments"])
        reviewer_results.append(
            {
                "reviewer_id": reviewer_id,
                "status": scored.get("status", "unknown"),
                "atomic_comments": scored["atomic_comments"],
                "metrics": metrics,
            }
        )

    return {
        "paper_id": paper_id,
        "metadata": metadata,
        "reviewers": reviewer_results,
    }


def run_human(
    pairs: list[tuple[str, str, str]],
    evaluator: ConstructivenessEvaluator,
    with_paper: bool = False,
    workers: int = 1,
) -> None:
    processed = load_processed_ids(HUMAN_OUTPUT)
    todo = [(pid, hp, sp) for pid, hp, sp in pairs if pid not in processed]

    total = len(pairs)
    done_pre = len(processed)
    todo_n = len(todo)

    print(f"\n{'=' * 65}")
    print(f"  [HUMAN] {done_pre}/{total} already done — {todo_n} remaining")
    print(f"{'=' * 65}")

    if not todo:
        print("  Nothing to do — all papers already processed.")
        return

    success, errors = 0, 0

    def process_one(item):
        pid, h_path, _ = item
        try:
            record = process_human_paper(pid, h_path, evaluator, with_paper)
            append_record(HUMAN_OUTPUT, record)
            n_rev = len(record["reviewers"])
            n_ok = sum(1 for r in record["reviewers"] if r["status"] == "success")
            with print_lock:
                print(f"  [{pid}] → Saved {n_ok}/{n_rev} reviewers OK")
            return True
        except Exception as exc:
            with print_lock:
                print(f"  [ERROR] {pid}: {type(exc).__name__}: {exc}")
            return False

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_one, item): item for item in todo}
            for future in tqdm(as_completed(futures), total=len(futures), desc="[HUMAN] Progress", unit="paper"):
                if future.result():
                    success += 1
                else:
                    errors += 1
    else:
        for i, item in enumerate(todo, 1):
            pid, h_path, _ = item
            print(
                f"\n  [{i}/{todo_n}] Paper: {pid}  (total progress: {done_pre + i}/{total}, {_pct(done_pre + i, total)})"
            )
            if process_one(item):
                success += 1
            else:
                errors += 1

    print(f"\n{'=' * 65}")
    print(f"  [HUMAN] Done — {success} success, {errors} errors")
    print(f"  Results: {HUMAN_OUTPUT}")
    print(f"{'=' * 65}")


# ═══════════════════════════════════════════════════════════════════════════════
# SEA evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def process_sea_paper(
    paper_id: str,
    h_path: str,
    s_path: str,
    evaluator: ConstructivenessEvaluator,
    with_paper: bool = False,
    paper_text: Optional[str] = None,
) -> dict:
    """Score the SEA review for one paper. Returns a single record."""
    human_data = load_human_meta_json(h_path)
    metadata = load_paper_metadata(human_data)
    sea_text = load_llm_txt(s_path)

    scored = evaluator.score_review(sea_text, "SEA_Reviewer", paper_text)
    metrics = compute_review_metrics(scored["atomic_comments"])

    return {
        "paper_id": paper_id,
        "metadata": metadata,
        "reviewer_id": "SEA_Reviewer",
        "status": scored.get("status", "unknown"),
        "atomic_comments": scored["atomic_comments"],
        "metrics": metrics,
    }


def run_sea(
    pairs: list[tuple[str, str, str]],
    evaluator: ConstructivenessEvaluator,
    with_paper: bool = False,
    workers: int = 1,
) -> None:
    processed = load_processed_ids(SEA_OUTPUT)
    todo = [(pid, hp, sp) for pid, hp, sp in pairs if pid not in processed]

    total = len(pairs)
    done_pre = len(processed)
    todo_n = len(todo)

    print(f"\n{'=' * 65}")
    print(f"  [SEA]   {done_pre}/{total} already done — {todo_n} remaining")
    print(f"{'=' * 65}")

    if not todo:
        print("  Nothing to do — all papers already processed.")
        return

    success, errors = 0, 0

    def process_one(item):
        pid, h_path, s_path = item
        try:
            record = process_sea_paper(pid, h_path, s_path, evaluator, with_paper)
            append_record(SEA_OUTPUT, record)
            n_arcs = len(record.get("atomic_comments", []))
            with print_lock:
                print(f"  [{pid}] → Saved (status={record['status']}, n_arcs={n_arcs})")
            return True
        except Exception as exc:
            with print_lock:
                print(f"  [ERROR] {pid}: {type(exc).__name__}: {exc}")
            return False

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_one, item): item for item in todo}
            for future in tqdm(as_completed(futures), total=len(futures), desc="[SEA] Progress", unit="paper"):
                if future.result():
                    success += 1
                else:
                    errors += 1
    else:
        for i, item in enumerate(todo, 1):
            pid, h_path, s_path = item
            print(
                f"\n  [{i}/{todo_n}] Paper: {pid}  (total progress: {done_pre + i}/{total}, {_pct(done_pre + i, total)})"
            )
            if process_one(item):
                success += 1
            else:
                errors += 1

    print(f"\n{'=' * 65}")
    print(f"  [SEA]   Done — {success} success, {errors} errors")
    print(f"  Results: {SEA_OUTPUT}")
    print(f"{'=' * 65}")


# ═══════════════════════════════════════════════════════════════════════════════
# Reviewer2 evaluation  (reviewer2_iclr2024 LLM reviews)
# ═══════════════════════════════════════════════════════════════════════════════


def process_reviewer2_paper(
    paper_id: str,
    h_path: str,
    r2_path: str,
    evaluator: ConstructivenessEvaluator,
    with_paper: bool = False,
    paper_text: Optional[str] = None,
) -> dict:
    """Score the reviewer2 LLM review for one paper. Returns a single record."""
    human_data = load_human_meta_json(h_path)
    metadata = load_paper_metadata(human_data)
    review_text = load_reviewer2_txt(r2_path)

    scored = evaluator.score_review(review_text, "Reviewer2_LLM", paper_text)
    metrics = compute_review_metrics(scored["atomic_comments"])

    return {
        "paper_id": paper_id,
        "metadata": metadata,
        "reviewer_id": "Reviewer2_LLM",
        "status": scored.get("status", "unknown"),
        "atomic_comments": scored["atomic_comments"],
        "metrics": metrics,
    }


def run_reviewer2(
    pairs: list[tuple[str, str, str]],
    evaluator: ConstructivenessEvaluator,
    output_path: str,
    with_paper: bool = False,
    workers: int = 1,
) -> None:
    """Evaluate all reviewer2 reviews and save to output_path."""
    processed = load_processed_ids(output_path)
    todo = [(pid, hp, rp) for pid, hp, rp in pairs if pid not in processed]

    total = len(pairs)
    done_pre = len(processed)
    todo_n = len(todo)

    print(f"\n{'=' * 65}")
    print(f"  [REVIEWER2] {done_pre}/{total} already done — {todo_n} remaining")
    print(f"{'=' * 65}")

    if not todo:
        print("  Nothing to do — all papers already processed.")
        return

    success, errors = 0, 0

    def process_one(item):
        pid, h_path, r2_path = item
        try:
            record = process_reviewer2_paper(
                pid, h_path, r2_path, evaluator, with_paper
            )
            append_record(output_path, record)
            n_arcs = len(record.get("atomic_comments", []))
            with print_lock:
                print(f"  [{pid}] → Saved (status={record['status']}, n_arcs={n_arcs})")
            return True
        except Exception as exc:
            with print_lock:
                print(f"  [ERROR] {pid}: {type(exc).__name__}: {exc}")
            return False

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_one, item): item for item in todo}
            for future in tqdm(as_completed(futures), total=len(futures), desc="[REVIEWER2] Progress", unit="paper"):
                if future.result():
                    success += 1
                else:
                    errors += 1
    else:
        for i, item in enumerate(todo, 1):
            pid, h_path, r2_path = item
            print(
                f"\n  [{i}/{todo_n}] Paper: {pid}  "
                f"(total progress: {done_pre + i}/{total}, {_pct(done_pre + i, total)})"
            )
            if process_one(item):
                success += 1
            else:
                errors += 1

    print(f"\n{'=' * 65}")
    print(f"  [REVIEWER2] Done — {success} success, {errors} errors")
    print(f"  Results: {output_path}")
    print(f"{'=' * 65}")


# ═══════════════════════════════════════════════════════════════════════════════
# DeepReview evaluation  (deepreview_iclr2024, reviewer_id=1 only)
# ═══════════════════════════════════════════════════════════════════════════════


def process_deepreview_paper(
    paper_id: str,
    h_path: str,
    dr_path: str,
    evaluator: ConstructivenessEvaluator,
    with_paper: bool = False,
    paper_text: Optional[str] = None,
) -> dict:
    """Score the DeepReview LLM review (reviewer_id=1) for one paper."""
    human_data = load_human_meta_json(h_path)
    metadata = load_paper_metadata(human_data)
    review_text = load_deepreview_text(dr_path, reviewer_id=1)

    if not review_text.strip():
        raise ValueError(f"No review text found for reviewer_id=1 in {dr_path}")

    scored = evaluator.score_review(review_text, "DeepReview_LLM", paper_text)
    metrics = compute_review_metrics(scored["atomic_comments"])

    return {
        "paper_id": paper_id,
        "metadata": metadata,
        "reviewer_id": "DeepReview_LLM",
        "status": scored.get("status", "unknown"),
        "atomic_comments": scored["atomic_comments"],
        "metrics": metrics,
    }


def run_deepreview(
    pairs: list[tuple[str, str, str]],
    evaluator: ConstructivenessEvaluator,
    with_paper: bool = False,
    workers: int = 1,
) -> None:
    """Evaluate all DeepReview reviews and save to DEEPREVIEW_OUTPUT."""
    processed = load_processed_ids(DEEPREVIEW_OUTPUT)
    todo = [(pid, hp, dp) for pid, hp, dp in pairs if pid not in processed]

    total = len(pairs)
    done_pre = len(processed)
    todo_n = len(todo)

    print(f"\n{'=' * 65}")
    print(f"  [DEEPREVIEW] {done_pre}/{total} already done — {todo_n} remaining")
    print(f"{'=' * 65}")

    if not todo:
        print("  Nothing to do — all papers already processed.")
        return

    success, errors = 0, 0

    def process_one(item):
        pid, h_path, dr_path = item
        try:
            record = process_deepreview_paper(
                pid, h_path, dr_path, evaluator, with_paper
            )
            append_record(DEEPREVIEW_OUTPUT, record)
            n_arcs = len(record.get("atomic_comments", []))
            with print_lock:
                print(f"  [{pid}] → Saved (status={record['status']}, n_arcs={n_arcs})")
            return True
        except Exception as exc:
            with print_lock:
                print(f"  [ERROR] {pid}: {type(exc).__name__}: {exc}")
            return False

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_one, item): item for item in todo}
            for future in tqdm(as_completed(futures), total=len(futures), desc="[DEEPREVIEW] Progress", unit="paper"):
                if future.result():
                    success += 1
                else:
                    errors += 1
    else:
        for i, item in enumerate(todo, 1):
            pid, h_path, dr_path = item
            print(
                f"\n  [{i}/{todo_n}] Paper: {pid}  "
                f"(total progress: {done_pre + i}/{total}, {_pct(done_pre + i, total)})"
            )
            if process_one(item):
                success += 1
            else:
                errors += 1

    print(f"\n{'=' * 65}")
    print(f"  [DEEPREVIEW] Done — {success} success, {errors} errors")
    print(f"  Results: {DEEPREVIEW_OUTPUT}")
    print(f"{'=' * 65}")


# ═══════════════════════════════════════════════════════════════════════════════
# Tree review evaluation  (tree_iclr2024_full)
# ═══════════════════════════════════════════════════════════════════════════════


def process_tree_paper(
    paper_id: str,
    h_path: str,
    tree_path: str,
    evaluator: ConstructivenessEvaluator,
    conf: str = "iclr2024",
    with_paper: bool = False,
    paper_text: Optional[str] = None,
) -> dict:
    """Score the Tree review for one paper."""
    human_data = load_human_meta_json(h_path)

    # Choose metadata loader based on conference
    if conf == "icml2025":
        metadata = load_paper_metadata_icml(human_data)
    elif conf == "neurips2025":
        metadata = load_paper_metadata_neurips(human_data)
    else:
        metadata = load_paper_metadata(human_data)

    review_text = load_tree_review_text(tree_path)

    if not review_text.strip():
        raise ValueError(f"No review text found in {tree_path}")

    scored = evaluator.score_review(review_text, "Tree_LLM", paper_text)
    metrics = compute_review_metrics(scored["atomic_comments"])

    return {
        "paper_id": paper_id,
        "metadata": metadata,
        "reviewer_id": "Tree_LLM",
        "status": scored.get("status", "unknown"),
        "atomic_comments": scored["atomic_comments"],
        "metrics": metrics,
    }


def run_tree(
    pairs: list[tuple[str, str, str]],
    evaluator: ConstructivenessEvaluator,
    output_path: str,
    conf: str = "iclr2024",
    with_paper: bool = False,
    workers: int = 1,
) -> None:
    """Evaluate all Tree reviews and save to output_path."""
    processed = load_processed_ids(output_path)
    todo = [(pid, hp, tp) for pid, hp, tp in pairs if pid not in processed]

    total = len(pairs)
    done_pre = len(processed)
    todo_n = len(todo)

    print(f"\n{'=' * 65}")
    print(
        f"  [TREE-{conf.upper()}] {done_pre}/{total} already done — {todo_n} remaining"
    )
    print(f"{'=' * 65}")

    if not todo:
        print("  Nothing to do — all papers already processed.")
        return

    success, errors = 0, 0

    def process_one(item):
        pid, h_path, tree_path = item
        try:
            record = process_tree_paper(
                pid, h_path, tree_path, evaluator, conf, with_paper
            )
            append_record(output_path, record)
            n_arcs = len(record.get("atomic_comments", []))
            with print_lock:
                print(f"  [{pid}] → Saved (status={record['status']}, n_arcs={n_arcs})")
            return True
        except Exception as exc:
            with print_lock:
                print(f"  [ERROR] {pid}: {type(exc).__name__}: {exc}")
            return False

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_one, item): item for item in todo}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"[TREE-{conf.upper()}] Progress", unit="paper"):
                if future.result():
                    success += 1
                else:
                    errors += 1
    else:
        for i, item in enumerate(todo, 1):
            pid, h_path, tree_path = item
            print(
                f"\n  [{i}/{todo_n}] Paper: {pid}  "
                f"(total progress: {done_pre + i}/{total}, {_pct(done_pre + i, total)})"
            )
            if process_one(item):
                success += 1
            else:
                errors += 1

    print(f"\n{'=' * 65}")
    print(f"  [TREE-{conf.upper()}] Done — {success} success, {errors} errors")
    print(f"  Results: {output_path}")
    print(f"{'=' * 65}")


# ═══════════════════════════════════════════════════════════════════════════════
# Summary / stats helpers
# ═══════════════════════════════════════════════════════════════════════════════


def print_progress_summary(total: int = 0) -> None:
    """Print a brief summary of how far along each mode is."""
    human_done = len(load_processed_ids(HUMAN_OUTPUT))
    sea_done = len(load_processed_ids(SEA_OUTPUT))
    reviewer2_done = len(load_processed_ids(REVIEWER2_OUTPUT))
    deepreview_done = len(load_processed_ids(DEEPREVIEW_OUTPUT))
    tree_done = len(load_processed_ids(TREE_OUTPUT))

    print("\n── Current progress ─────────────────────────────────────────")
    print(f"  Total papers    : {total}")
    print(f"  Human done      : {human_done}      ({_pct(human_done, total)} of total)")
    print(f"  SEA done        : {sea_done}         ({_pct(sea_done, total)} of total)")
    print(
        f"  Reviewer2 done  : {reviewer2_done}   ({_pct(reviewer2_done, total)} of total)"
    )
    print(
        f"  DeepReview done : {deepreview_done}  ({_pct(deepreview_done, total)} of total)"
    )
    print(f"  Tree done       : {tree_done}        ({_pct(tree_done, total)} of total)")
    print("─────────────────────────────────────────────────────────────\n")


# ═══════════════════════════════════════════════════════════════════════════════
# ICML 2025 Human evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def process_icml_human_paper(
    paper_id: str,
    h_path: str,
    evaluator: ConstructivenessEvaluator,
    with_paper: bool = False,
    paper_text: Optional[str] = None,
) -> dict:
    """Score every human reviewer for one ICML2025 paper. Returns a single record."""
    human_data = load_human_meta_json(h_path)
    metadata = load_paper_metadata_icml(human_data)

    human_list = human_data.get("reviews", [])

    reviewer_results = []
    for idx, review_obj in enumerate(human_list):
        reviewer_id = f"Human_{idx + 1}"
        review_text = format_human_review_full_icml(review_obj)

        if not review_text.strip():
            reviewer_results.append(
                {
                    "reviewer_id": reviewer_id,
                    "status": "empty_input",
                    "atomic_comments": [],
                    "metrics": None,
                }
            )
            continue

        scored = evaluator.score_review(review_text, reviewer_id, paper_text)
        metrics = compute_review_metrics(scored["atomic_comments"])
        reviewer_results.append(
            {
                "reviewer_id": reviewer_id,
                "status": scored.get("status", "unknown"),
                "atomic_comments": scored["atomic_comments"],
                "metrics": metrics,
            }
        )

    return {
        "paper_id": paper_id,
        "metadata": metadata,
        "reviewers": reviewer_results,
    }


def run_icml_human(
    pairs: list[tuple[str, str]],
    evaluator: ConstructivenessEvaluator,
    with_paper: bool = False,
    workers: int = 1,
) -> None:
    """Evaluate all ICML2025 human reviews and save to ICML2025_HUMAN_OUTPUT."""
    processed = load_processed_ids(ICML2025_HUMAN_OUTPUT)
    todo = [(pid, hp) for pid, hp in pairs if pid not in processed]

    total = len(pairs)
    done_pre = len(processed)
    todo_n = len(todo)

    print(f"\n{'=' * 65}")
    print(f"  [ICML2025-HUMAN] {done_pre}/{total} already done — {todo_n} remaining")
    print(f"{'=' * 65}")

    if not todo:
        print("  Nothing to do — all papers already processed.")
        return

    success, errors = 0, 0

    def process_one(item):
        pid, h_path = item
        try:
            record = process_icml_human_paper(pid, h_path, evaluator, with_paper)
            append_record(ICML2025_HUMAN_OUTPUT, record)
            n_rev = len(record["reviewers"])
            n_ok = sum(1 for r in record["reviewers"] if r["status"] == "success")
            with print_lock:
                print(f"  [{pid}] → Saved {n_ok}/{n_rev} reviewers OK")
            return True
        except Exception as exc:
            with print_lock:
                print(f"  [ERROR] {pid}: {type(exc).__name__}: {exc}")
            return False

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_one, item): item for item in todo}
            for future in tqdm(as_completed(futures), total=len(futures), desc="[ICML2025-HUMAN] Progress", unit="paper"):
                if future.result():
                    success += 1
                else:
                    errors += 1
    else:
        for i, item in enumerate(todo, 1):
            pid, h_path = item
            print(
                f"\n  [{i}/{todo_n}] Paper: {pid}  (total progress: {done_pre + i}/{total}, {_pct(done_pre + i, total)})"
            )
            if process_one(item):
                success += 1
            else:
                errors += 1

    print(f"\n{'=' * 65}")
    print(f"  [ICML2025-HUMAN] Done — {success} success, {errors} errors")
    print(f"  Results: {ICML2025_HUMAN_OUTPUT}")
    print(f"{'=' * 65}")


# ═══════════════════════════════════════════════════════════════════════════════
# NeurIPS 2025 Human evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def process_neurips_human_paper(
    paper_id: str,
    h_path: str,
    evaluator: ConstructivenessEvaluator,
    with_paper: bool = False,
    paper_text: Optional[str] = None,
) -> dict:
    """Score every human reviewer for one NeurIPS 2025 paper. Returns a single record."""
    human_data = load_human_meta_json(h_path)
    metadata = load_paper_metadata_neurips(human_data)

    human_list = human_data.get("reviews", [])

    reviewer_results = []
    for idx, review_obj in enumerate(human_list):
        reviewer_id = f"Human_{idx + 1}"
        review_text = format_human_review_full_neurips(review_obj)

        if not review_text.strip():
            reviewer_results.append(
                {
                    "reviewer_id": reviewer_id,
                    "status": "empty_input",
                    "atomic_comments": [],
                    "metrics": None,
                }
            )
            continue

        scored = evaluator.score_review(review_text, reviewer_id, paper_text)
        metrics = compute_review_metrics(scored["atomic_comments"])
        reviewer_results.append(
            {
                "reviewer_id": reviewer_id,
                "status": scored.get("status", "unknown"),
                "atomic_comments": scored["atomic_comments"],
                "metrics": metrics,
            }
        )

    return {
        "paper_id": paper_id,
        "metadata": metadata,
        "reviewers": reviewer_results,
    }


def run_neurips_human(
    pairs: list[tuple[str, str]],
    evaluator: ConstructivenessEvaluator,
    with_paper: bool = False,
    workers: int = 1,
) -> None:
    """Evaluate all NeurIPS 2025 human reviews and save to NEURIPS2025_HUMAN_OUTPUT."""
    processed = load_processed_ids(NEURIPS2025_HUMAN_OUTPUT)
    todo = [(pid, hp) for pid, hp in pairs if pid not in processed]

    total = len(pairs)
    done_pre = len(processed)
    todo_n = len(todo)

    print(f"\n{'=' * 65}")
    print(f"  [NEURIPS2025-HUMAN] {done_pre}/{total} already done — {todo_n} remaining")
    print(f"{'=' * 65}")

    if not todo:
        print("  Nothing to do — all papers already processed.")
        return

    success, errors = 0, 0

    def process_one(item):
        pid, h_path = item
        try:
            record = process_neurips_human_paper(pid, h_path, evaluator, with_paper)
            append_record(NEURIPS2025_HUMAN_OUTPUT, record)
            n_rev = len(record["reviewers"])
            n_ok = sum(1 for r in record["reviewers"] if r["status"] == "success")
            with print_lock:
                print(f"  [{pid}] → Saved {n_ok}/{n_rev} reviewers OK")
            return True
        except Exception as exc:
            with print_lock:
                print(f"  [ERROR] {pid}: {type(exc).__name__}: {exc}")
            return False

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_one, item): item for item in todo}
            for future in tqdm(as_completed(futures), total=len(futures), desc="[NEURIPS2025-HUMAN] Progress", unit="paper"):
                if future.result():
                    success += 1
                else:
                    errors += 1
    else:
        for i, item in enumerate(todo, 1):
            pid, h_path = item
            print(
                f"\n  [{i}/{todo_n}] Paper: {pid}  (total progress: {done_pre + i}/{total}, {_pct(done_pre + i, total)})"
            )
            if process_one(item):
                success += 1
            else:
                errors += 1

    print(f"\n{'=' * 65}")
    print(f"  [NEURIPS2025-HUMAN] Done — {success} success, {errors} errors")
    print(f"  Results: {NEURIPS2025_HUMAN_OUTPUT}")
    print(f"{'=' * 65}")


# ═══════════════════════════════════════════════════════════════════════════════
# CycleReview evaluation  (CycleReviewer-ML, first reviewer only)
# ═══════════════════════════════════════════════════════════════════════════════


def process_cyclereview_paper(
    paper_id: str,
    cr_path: str,
    evaluator: ConstructivenessEvaluator,
    with_paper: bool = False,
    paper_text: Optional[str] = None,
) -> dict:
    """Score the CycleReview first reviewer for one paper. Returns a single record."""
    metadata = load_cyclereview_metadata(cr_path)
    review_text = load_cyclereview_first_text(cr_path)

    if not review_text.strip():
        raise ValueError(f"No review text found for first reviewer in {cr_path}")

    scored = evaluator.score_review(review_text, "CycleReview_LLM", paper_text)
    metrics = compute_review_metrics(scored["atomic_comments"])

    return {
        "paper_id": paper_id,
        "metadata": metadata,
        "reviewer_id": "CycleReview_LLM",
        "status": scored.get("status", "unknown"),
        "atomic_comments": scored["atomic_comments"],
        "metrics": metrics,
    }


def run_cyclereview(
    pairs: list[tuple[str, str]],
    evaluator: ConstructivenessEvaluator,
    output_path: str,
    conf: str,
    with_paper: bool = False,
    workers: int = 1,
) -> None:
    """Evaluate all CycleReview reviews for a conference and save to output_path."""
    processed = load_processed_ids(output_path)
    todo = [(pid, cp) for pid, cp in pairs if pid not in processed]

    total = len(pairs)
    done_pre = len(processed)
    todo_n = len(todo)

    print(f"\n{'=' * 65}")
    print(
        f"  [CYCLEREVIEW-{conf.upper()}] {done_pre}/{total} already done — {todo_n} remaining"
    )
    print(f"{'=' * 65}")

    if not todo:
        print("  Nothing to do — all papers already processed.")
        return

    success, errors = 0, 0

    def process_one(item):
        pid, cr_path = item
        try:
            record = process_cyclereview_paper(pid, cr_path, evaluator, with_paper)
            append_record(output_path, record)
            n_arcs = len(record.get("atomic_comments", []))
            with print_lock:
                print(f"  [{pid}] → Saved (status={record['status']}, n_arcs={n_arcs})")
            return True
        except Exception as exc:
            with print_lock:
                print(f"  [ERROR] {pid}: {type(exc).__name__}: {exc}")
            return False

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_one, item): item for item in todo}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"[CYCLEREVIEW-{conf.upper()}] Progress", unit="paper"):
                if future.result():
                    success += 1
                else:
                    errors += 1
    else:
        for i, item in enumerate(todo, 1):
            pid, cr_path = item
            print(
                f"\n  [{i}/{todo_n}] Paper: {pid}  "
                f"(total progress: {done_pre + i}/{total}, {_pct(done_pre + i, total)})"
            )
            if process_one(item):
                success += 1
            else:
                errors += 1

    print(f"\n{'=' * 65}")
    print(f"  [CYCLEREVIEW-{conf.upper()}] Done — {success} success, {errors} errors")
    print(f"  Results: {output_path}")
    print(f"{'=' * 65}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    global HUMAN_FOLDER, SEA_FOLDER, REVIEWER2_FOLDER, DEEPREVIEW_FOLDER, TREE_FOLDER
    args = parse_args()
    conference = {
        "iclr2024": "ICLR2024",
        "iclr2025": "ICLR2025",
        "iclr2026": "ICLR2026",
        "icml2025": "ICML2025",
        "neurips2025": "NeurIPS2025",
    }[args.conf]
    HUMAN_FOLDER = _reviewer_dir(conference, "human")
    SEA_FOLDER = _reviewer_dir(conference, "sea")
    REVIEWER2_FOLDER = _reviewer_dir(conference, "reviewer2")
    DEEPREVIEW_FOLDER = _reviewer_dir(conference, "deepreview")
    TREE_FOLDER = _reviewer_dir(conference, "tree")

    pairs: list[tuple[str, str, str]] = []
    r2_pairs: list[tuple[str, str, str]] = []
    dr_pairs: list[tuple[str, str, str]] = []
    tr_pairs: list[tuple[str, str, str]] = []
    icml_pairs: list[tuple[str, str]] = []
    neurips_pairs: list[tuple[str, str]] = []
    cr_pairs: list[tuple[str, str]] = []

    # --- discover paper pairs ---
    if args.mode == "icml_human":
        print(f"\n[INFO] Discovering ICML2025 human papers from:")
        print(f"       human     → {ICML2025_HUMAN_FOLDER}")
        print(f"       paper IDs → {ICML2025_PAPER_IDS}")
        icml_pairs = get_paper_pairs_from_ids(ICML2025_HUMAN_FOLDER, ICML2025_PAPER_IDS)
        total = len(icml_pairs)
        print(f"[INFO] {total} ICML2025 papers found.")
        if args.limit:
            icml_pairs = icml_pairs[: args.limit]
            print(
                f"[INFO] --limit {args.limit} applied → {len(icml_pairs)} papers to process."
            )

    elif args.mode == "neurips_human":
        print(f"\n[INFO] Discovering NeurIPS2025 human papers from:")
        print(f"       human     → {NEURIPS2025_HUMAN_FOLDER}")
        print(f"       paper IDs → {NEURIPS2025_PAPER_IDS}")
        neurips_pairs = get_paper_pairs_from_ids(
            NEURIPS2025_HUMAN_FOLDER, NEURIPS2025_PAPER_IDS
        )
        total = len(neurips_pairs)
        print(f"[INFO] {total} NeurIPS2025 papers found.")
        if args.limit:
            neurips_pairs = neurips_pairs[: args.limit]
            print(
                f"[INFO] --limit {args.limit} applied → {len(neurips_pairs)} papers to process."
            )

    elif args.mode == "reviewer2":
        conf = args.conf
        r2_folder = REVIEWER2_FOLDERS[conf]
        human_folder = REVIEWER2_HUMAN_FOLDERS[conf]
        ids_file = REVIEWER2_PAPER_IDS[conf]
        print(f"\n[INFO] Discovering reviewer2 papers for {conf.upper()} from:")
        print(f"       folder    → {r2_folder}")
        print(f"       human     → {human_folder}")
        print(f"       paper IDs → {ids_file}")
        r2_pairs = discover_reviewer2_pairs(r2_folder, human_folder, ids_file)
        total = len(r2_pairs)
        print(f"[INFO] {total} matched reviewer2 pairs found.")
        if args.limit:
            r2_pairs = r2_pairs[: args.limit]
            print(
                f"[INFO] --limit {args.limit} applied → {len(r2_pairs)} papers to process."
            )

    elif args.mode == "deepreview":
        print(f"\n[INFO] Discovering deepreview papers from:")
        print(f"       deepreview → {DEEPREVIEW_FOLDER}")
        print(f"       human      → {HUMAN_FOLDER}")
        dr_pairs = discover_deepreview_pairs()
        total = len(dr_pairs)
        print(f"[INFO] {total} matched deepreview pairs found.")
        if args.limit:
            dr_pairs = dr_pairs[: args.limit]
            print(
                f"[INFO] --limit {args.limit} applied → {len(dr_pairs)} papers to process."
            )

    elif args.mode == "tree":
        conf = args.conf
        tree_folder = TREE_FOLDERS[conf]
        human_folder = TREE_HUMAN_FOLDERS[conf]
        ids_file = TREE_PAPER_IDS[conf]
        print(f"\n[INFO] Discovering tree review papers from:")
        print(f"       tree      → {tree_folder}")
        print(f"       human     → {human_folder}")
        print(f"       paper IDs → {ids_file}")
        tr_pairs = discover_tree_pairs(tree_folder, human_folder, ids_file)
        total = len(tr_pairs)
        print(f"[INFO] {total} matched tree review pairs found.")
        if args.limit:
            tr_pairs = tr_pairs[: args.limit]
            print(
                f"[INFO] --limit {args.limit} applied → {len(tr_pairs)} papers to process."
            )

    elif args.mode == "cyclereview":
        conf = args.conf
        cr_folder = CYCLEREVIEW_FOLDERS[conf]
        cr_ids = CYCLEREVIEW_PAPER_IDS[conf]
        print(f"\n[INFO] Discovering CycleReview papers for {conf.upper()} from:")
        print(f"       folder    → {cr_folder}")
        print(f"       paper IDs → {cr_ids}")
        cr_pairs = get_cyclereview_pairs_from_ids(cr_folder, cr_ids)
        total = len(cr_pairs)
        print(f"[INFO] {total} CycleReview papers found.")
        if args.limit:
            cr_pairs = cr_pairs[: args.limit]
            print(
                f"[INFO] --limit {args.limit} applied → {len(cr_pairs)} papers to process."
            )

    else:
        # human / sea / both — discover standard pairs
        print(f"\n[INFO] Discovering papers from:")
        print(f"       human → {HUMAN_FOLDER}")
        print(f"       sea   → {SEA_FOLDER}")
        pairs = discover_pairs()
        total = len(pairs)
        print(f"[INFO] {total} matched paper pairs found (human .json + sea .txt).")
        if args.limit:
            pairs = pairs[: args.limit]
            print(
                f"[INFO] --limit {args.limit} applied → {len(pairs)} papers to process."
            )

    print_progress_summary(total)

    # --- initialise evaluator ---
    print(f"[INFO] Initialising evaluator (provider={args.provider})...")
    try:
        evaluator = _build_evaluator(args)
    except Exception as exc:
        print(f"[FATAL] Could not init evaluator: {exc}")
        sys.exit(1)
    print("[INFO] Evaluator ready!\n")

    # Read environment variable override if set
    env_workers = os.getenv("PRISM_MAX_WORKERS")
    if env_workers:
        try:
            args.workers = int(env_workers)
        except ValueError:
            pass

    # --- run requested modes ---
    t0 = time.time()

    if args.mode in ("human", "both"):
        run_human(pairs, evaluator, args.with_paper, args.workers)

    if args.mode in ("sea", "both"):
        run_sea(pairs, evaluator, args.with_paper, args.workers)

    if args.mode == "reviewer2":
        conf = args.conf
        output_path = REVIEWER2_OUTPUTS[conf]
        run_reviewer2(r2_pairs, evaluator, output_path, args.with_paper, args.workers)

    if args.mode == "deepreview":
        run_deepreview(dr_pairs, evaluator, args.with_paper, args.workers)

    if args.mode == "tree":
        conf = args.conf
        output_path = TREE_OUTPUTS[conf]
        run_tree(tr_pairs, evaluator, output_path, conf, args.with_paper, args.workers)

    if args.mode == "icml_human":
        run_icml_human(icml_pairs, evaluator, args.with_paper, args.workers)

    if args.mode == "neurips_human":
        run_neurips_human(neurips_pairs, evaluator, args.with_paper, args.workers)

    if args.mode == "cyclereview":
        conf = args.conf
        output_path = CYCLEREVIEW_OUTPUTS[conf]
        run_cyclereview(cr_pairs, evaluator, output_path, conf, args.with_paper, args.workers)

    elapsed = time.time() - t0
    print(f"\n[INFO] Total time elapsed: {elapsed / 60:.1f} min")
    print_progress_summary(total)


if __name__ == "__main__":
    main()
