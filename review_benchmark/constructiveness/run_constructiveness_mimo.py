"""
run_constructiveness_mimo.py — Constructiveness evaluation runner using Mimo v2.5 Pro.

Same logic as run_constructiveness.py but:
  - Default provider: mimo (model: mimo-v2.5-pro)
  - Uses paper_ids_50 files (50 papers per conference) instead of paper_ids_200
  - Output goes to output/<conf>/mimo/<mode>/all_results_lite.jsonl

Required env var:
  MIMO_API_KEY  – API key from platform.xiaomimimo.com
  MIMO_BASE_URL – (optional) defaults to https://api.xiaomimimo.com/v1

Usage examples:
    python run_constructiveness_mimo.py --mode reviewer2 --conf icml2025
    python run_constructiveness_mimo.py --mode reviewer2 --conf neurips2025
    python run_constructiveness_mimo.py --mode reviewer2 --conf iclr2024
    python run_constructiveness_mimo.py --mode reviewer2 --conf iclr2025
    python run_constructiveness_mimo.py --mode reviewer2 --conf iclr2026
    python run_constructiveness_mimo.py --mode sea --conf iclr2025
    python run_constructiveness_mimo.py --mode sea --conf icml2025
    python run_constructiveness_mimo.py --mode sea --conf neurips2025
    python run_constructiveness_mimo.py --mode deepreview --conf iclr2025
    python run_constructiveness_mimo.py --mode deepreview --conf icml2025
    python run_constructiveness_mimo.py --mode deepreview --conf neurips2025
    python run_constructiveness_mimo.py --mode tree --conf iclr2025
    python run_constructiveness_mimo.py --mode cyclereview --conf iclr2024
    python run_constructiveness_mimo.py --mode icml_human
    python run_constructiveness_mimo.py --mode neurips_human
    python run_constructiveness_mimo.py --mode human --conf iclr2024
    python run_constructiveness_mimo.py --mode reviewer2 --conf icml2025 --limit 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

# ── Resolve paths ─────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_FI   = os.path.normpath(os.path.join(_HERE, "..", "flaw_identification"))
sys.path.insert(0, _FI)

from dotenv import load_dotenv

_REPO_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
for _p in [os.path.join(_REPO_ROOT, ".env"), os.path.join(_HERE, ".env"), os.path.join(_FI, ".env")]:
    if os.path.exists(_p):
        load_dotenv(_p, override=False)

from paths_config import conf_path as _conf_path

from src.evaluator import ConstructivenessEvaluator
from src.metrics import compute_review_metrics
from src.utils import (
    format_human_review_full,
    format_human_review_full_icml,
    format_human_review_full_neurips,
    get_cyclereview_pairs_from_ids,
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
_ICLR2024_ROOT    = _conf_path("ICLR2024")
_ICLR2025_ROOT    = _conf_path("ICLR2025")
_ICLR2026_ROOT    = _conf_path("ICLR2026")
_ICML2025_ROOT    = _conf_path("ICML2025")
_NEURIPS2025_ROOT = _conf_path("NeurIPS2025")

OUTPUT_ROOT = os.path.join(_HERE, "output")

# ── Paper IDs — 50-paper subsets ───────────────────────────────────────────────
PAPER_IDS_50: dict[str, str] = {
    "iclr2024":    os.path.join(_ICLR2024_ROOT,   "paper_ids_50_iclr2024.txt"),
    "iclr2025":    os.path.join(_ICLR2025_ROOT,   "paper_ids_50_iclr2025.txt"),
    "iclr2026":    os.path.join(_ICLR2026_ROOT,   "paper_ids_50_iclr2026.txt"),
    "icml2025":    os.path.join(_ICML2025_ROOT,   "paper_ids_50_icml2025.txt"),
    "neurips2025": os.path.join(_NEURIPS2025_ROOT, "paper_ids_50_neurips2025.txt"),
}

# ── Human review folders ───────────────────────────────────────────────────────
HUMAN_FOLDERS: dict[str, str] = {
    "iclr2024":    os.path.join(_ICLR2024_ROOT,   "human_reviews"),
    "iclr2025":    os.path.join(_ICLR2025_ROOT,   "human_reviews"),
    "iclr2026":    os.path.join(_ICLR2026_ROOT,   "human_reviews"),
    "icml2025":    os.path.join(_ICML2025_ROOT,   "human_reviews"),
    "neurips2025": os.path.join(_NEURIPS2025_ROOT, "human_reviews"),
}

# ── Tree review folders ────────────────────────────────────────────────────────
TREE_FOLDERS: dict[str, str] = {
    "iclr2024":    os.path.join(_ICLR2024_ROOT,   "tree_iclr2024"),
    "iclr2025":    os.path.join(_ICLR2025_ROOT,   "tree_iclr2025"),
    "iclr2026":    os.path.join(_ICLR2026_ROOT,   "tree_iclr2026"),
    "icml2025":    os.path.join(_ICML2025_ROOT,   "tree_icml2025"),
    "neurips2025": os.path.join(_NEURIPS2025_ROOT, "tree_neurips2025"),
}

# ── Reviewer2 folders ─────────────────────────────────────────────────────────
REVIEWER2_FOLDERS: dict[str, str] = {
    "iclr2024":    os.path.join(_ICLR2024_ROOT,   "reviewer2_iclr2024"),
    "iclr2025":    os.path.join(_ICLR2025_ROOT,   "reviewer2_iclr2025"),
    "iclr2026":    os.path.join(_ICLR2026_ROOT,   "reviewer2_iclr2026"),
    "icml2025":    os.path.join(_ICML2025_ROOT,   "reviewer2_icml2025"),
    "neurips2025": os.path.join(_NEURIPS2025_ROOT, "reviewer2_neurips2025"),
}

# ── CycleReview folders ───────────────────────────────────────────────────────
CYCLEREVIEW_FOLDERS: dict[str, str] = {
    "iclr2024":    os.path.join(_ICLR2024_ROOT,   "cyclereview_iclr2024"),
    "iclr2025":    os.path.join(_ICLR2025_ROOT,   "cyclereview_iclr2025"),
    "iclr2026":    os.path.join(_ICLR2026_ROOT,   "cyclereview_iclr2026"),
    "icml2025":    os.path.join(_ICML2025_ROOT,   "cyclereview_icml2025"),
    "neurips2025": os.path.join(_NEURIPS2025_ROOT, "cyclereview_neurlps2025"),
}

# ── SEA folders ───────────────────────────────────────────────────────────────
SEA_FOLDERS: dict[str, str] = {
    "iclr2024":    os.path.join(_ICLR2024_ROOT,   "sea_iclr2024"),
    "iclr2025":    os.path.join(_ICLR2025_ROOT,   "sea_iclr2025"),
    "iclr2026":    os.path.join(_ICLR2026_ROOT,   "sea_iclr2026"),
    "icml2025":    os.path.join(_ICML2025_ROOT,   "sea_icml2025"),
    "neurips2025": os.path.join(_NEURIPS2025_ROOT, "sea_neurlps2025"),
}

# ── DeepReview folders ────────────────────────────────────────────────────────
DEEPREVIEW_FOLDERS: dict[str, str] = {
    "iclr2024":    os.path.join(_ICLR2024_ROOT,   "deepreview_iclr2024"),
    "iclr2025":    os.path.join(_ICLR2025_ROOT,   "deepreview_iclr2025"),
    "iclr2026":    os.path.join(_ICLR2026_ROOT,   "deepreview_iclr2026"),
    "icml2025":    os.path.join(_ICML2025_ROOT,   "deepreview_icml2025"),
    "neurips2025": os.path.join(_NEURIPS2025_ROOT, "deepreview_neurips2025"),
}

# ── Output paths (mimo subfolder) ─────────────────────────────────────────────
def _out(conf: str, mode: str) -> str:
    return os.path.join(OUTPUT_ROOT, conf, "mimo", mode, "all_results_lite.jsonl")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Constructiveness evaluation — Mimo v2.5 Pro, 50-paper subsets.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=["human", "sea", "deepreview", "reviewer2", "tree", "cyclereview",
                 "icml_human", "neurips_human"],
        default="reviewer2",
        help=(
            "Which reviews to evaluate:\n"
            "  human         — human peer-reviews (use --conf)\n"
            "  sea           — SEA LLM reviews (use --conf)\n"
            "  deepreview    — DeepReview LLM reviews (use --conf)\n"
            "  reviewer2     — reviewer2 LLM reviews (use --conf)\n"
            "  tree          — Tree review LLM reviews (use --conf)\n"
            "  cyclereview   — CycleReview LLM reviews (use --conf)\n"
            "  icml_human    — ICML2025 human peer-reviews (50 papers)\n"
            "  neurips_human — NeurIPS2025 human peer-reviews (50 papers)\n"
        ),
    )
    p.add_argument(
        "--conf",
        choices=["iclr2024", "iclr2025", "iclr2026", "icml2025", "neurips2025"],
        default="iclr2025",
        help="Conference to evaluate (default: iclr2025).",
    )
    p.add_argument(
        "--provider", choices=["mimo", "gemini", "azure"], default="mimo",
        help="LLM provider (default: mimo).",
    )
    p.add_argument(
        "--model", default=None,
        help="Override model name (default: mimo-v2.5-pro).",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N papers (for quick tests).",
    )
    p.add_argument(
        "--with-paper", action="store_true", default=False,
        help="Include paper text as context.",
    )
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_processed_ids(jsonl_path: str) -> set[str]:
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
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _pct(done: int, total: int) -> str:
    if total == 0:
        return "0%"
    return f"{done/total*100:.1f}%"


def _build_evaluator(args: argparse.Namespace) -> ConstructivenessEvaluator:
    if args.provider == "mimo":
        api_key = os.getenv("MIMO_API_KEY", "")
    elif args.provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
    else:
        api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    return ConstructivenessEvaluator(
        provider=args.provider,
        api_key=api_key,
        model=args.model or ("mimo-v2.5-pro" if args.provider == "mimo" else None),
    )


def discover_pairs_from_ids(
    folder: str,
    paper_ids_file: str,
) -> list[tuple[str, str]]:
    """Return list of (paper_id, json_path) for papers listed in IDs file."""
    import glob
    allowed: set[str] = set()
    if os.path.exists(paper_ids_file):
        with open(paper_ids_file, "r", encoding="utf-8") as f:
            allowed = {ln.strip() for ln in f if ln.strip()}
    pairs = []
    for path in glob.glob(os.path.join(folder, "*.json")):
        pid = os.path.splitext(os.path.basename(path))[0]
        if pid in allowed:
            pairs.append((pid, path))
    pairs.sort(key=lambda t: t[0])
    return pairs


def discover_reviewer2_pairs(
    r2_folder: str,
    human_folder: str,
    paper_ids_file: str,
) -> list[tuple[str, str, str]]:
    import glob
    allowed: set[str] = set()
    if os.path.exists(paper_ids_file):
        with open(paper_ids_file, "r", encoding="utf-8") as f:
            allowed = {ln.strip() for ln in f if ln.strip()}
    pairs = []
    for r2_path in glob.glob(os.path.join(r2_folder, "*.txt")):
        pid = os.path.splitext(os.path.basename(r2_path))[0]
        if pid not in allowed:
            continue
        h_path = os.path.join(human_folder, f"{pid}.json")
        if os.path.exists(h_path):
            pairs.append((pid, h_path, r2_path))
        else:
            print(f"  [WARNING] Missing human JSON for reviewer2 paper {pid}")
    pairs.sort(key=lambda t: t[0])
    return pairs


def discover_sea_pairs(
    sea_folder: str,
    human_folder: str,
    paper_ids_file: str,
) -> list[tuple[str, str, str]]:
    """Return (paper_id, human_json_path, sea_txt_path) for the 50-paper subset."""
    import glob
    allowed: set[str] = set()
    if os.path.exists(paper_ids_file):
        with open(paper_ids_file, "r", encoding="utf-8") as f:
            allowed = {ln.strip() for ln in f if ln.strip()}
    pairs = []
    for sea_path in glob.glob(os.path.join(sea_folder, "*.txt")):
        pid = os.path.splitext(os.path.basename(sea_path))[0]
        if pid not in allowed:
            continue
        h_path = os.path.join(human_folder, f"{pid}.json")
        if os.path.exists(h_path):
            pairs.append((pid, h_path, sea_path))
        else:
            print(f"  [WARNING] Missing human JSON for SEA paper {pid}")
    pairs.sort(key=lambda t: t[0])
    return pairs


def discover_deepreview_pairs(
    dr_folder: str,
    human_folder: str,
    paper_ids_file: str,
) -> list[tuple[str, str, str]]:
    """Return (paper_id, human_json_path, deepreview_json_path) for the 50-paper subset."""
    import glob
    allowed: set[str] = set()
    if os.path.exists(paper_ids_file):
        with open(paper_ids_file, "r", encoding="utf-8") as f:
            allowed = {ln.strip() for ln in f if ln.strip()}
    pairs = []
    for dr_path in glob.glob(os.path.join(dr_folder, "*.json")):
        pid = os.path.splitext(os.path.basename(dr_path))[0]
        if pid not in allowed:
            continue
        h_path = os.path.join(human_folder, f"{pid}.json")
        if os.path.exists(h_path):
            pairs.append((pid, h_path, dr_path))
        else:
            print(f"  [WARNING] Missing human JSON for DeepReview paper {pid}")
    pairs.sort(key=lambda t: t[0])
    return pairs


def discover_tree_pairs(    tree_folder: str,
    human_folder: str,
    paper_ids_file: str,
) -> list[tuple[str, str, str]]:
    import glob
    allowed: set[str] = set()
    if os.path.exists(paper_ids_file):
        with open(paper_ids_file, "r", encoding="utf-8") as f:
            allowed = {ln.strip() for ln in f if ln.strip()}
    pairs = []
    for tree_path in glob.glob(os.path.join(tree_folder, "*_review.json")):
        pid = os.path.basename(tree_path).replace("_review.json", "")
        if pid not in allowed:
            continue
        h_path = os.path.join(human_folder, f"{pid}.json")
        if os.path.exists(h_path):
            pairs.append((pid, h_path, tree_path))
        else:
            print(f"  [WARNING] Missing human JSON for tree paper {pid}")
    pairs.sort(key=lambda t: t[0])
    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
# Metadata helper: choose loader by conference
# ═══════════════════════════════════════════════════════════════════════════════

def _load_metadata(human_data: dict, conf: str) -> dict:
    if conf == "icml2025":
        return load_paper_metadata_icml(human_data)
    elif conf == "neurips2025":
        return load_paper_metadata_neurips(human_data)
    else:
        return load_paper_metadata(human_data)


def _format_human_review(review_obj: dict, conf: str) -> str:
    if conf == "icml2025":
        return format_human_review_full_icml(review_obj)
    elif conf == "neurips2025":
        return format_human_review_full_neurips(review_obj)
    else:
        return format_human_review_full(review_obj)


# ═══════════════════════════════════════════════════════════════════════════════
# Human evaluation (ICLR conferences)
# ═══════════════════════════════════════════════════════════════════════════════

def run_human(
    conf: str,
    evaluator: ConstructivenessEvaluator,
    limit: Optional[int],
    with_paper: bool = False,
) -> None:
    human_folder = HUMAN_FOLDERS[conf]
    ids_file     = PAPER_IDS_50[conf]
    output_path  = _out(conf, "human")

    pairs = discover_pairs_from_ids(human_folder, ids_file)
    processed = load_processed_ids(output_path)
    todo = [(pid, hp) for pid, hp in pairs if pid not in processed]
    if limit:
        todo = todo[:limit]

    total    = len(pairs)
    done_pre = len(processed)
    todo_n   = len(todo)

    print(f"\n{'='*65}")
    print(f"  [HUMAN-{conf.upper()}] {done_pre}/{total} already done — {todo_n} remaining")
    print(f"{'='*65}")

    if not todo:
        print("  Nothing to do — all papers already processed.")
        return

    success, errors = 0, 0
    for i, (pid, h_path) in enumerate(todo, 1):
        print(f"\n  [{i}/{todo_n}] Paper: {pid}  ({_pct(done_pre+i, total)})")
        try:
            human_data = load_human_meta_json(h_path)
            metadata   = _load_metadata(human_data, conf)
            human_list = human_data.get("reviews", [])

            reviewer_results = []
            for idx, review_obj in enumerate(human_list):
                reviewer_id = f"Human_{idx + 1}"
                review_text = _format_human_review(review_obj, conf)
                if not review_text.strip():
                    reviewer_results.append({
                        "reviewer_id": reviewer_id,
                        "status": "empty_input",
                        "atomic_comments": [],
                        "metrics": None,
                    })
                    continue
                scored  = evaluator.score_review(review_text, reviewer_id)
                metrics = compute_review_metrics(scored["atomic_comments"])
                reviewer_results.append({
                    "reviewer_id": reviewer_id,
                    "status": scored.get("status", "unknown"),
                    "atomic_comments": scored["atomic_comments"],
                    "metrics": metrics,
                })

            record = {"paper_id": pid, "metadata": metadata, "reviewers": reviewer_results}
            append_record(output_path, record)
            success += 1
            n_ok = sum(1 for r in reviewer_results if r["status"] == "success")
            print(f"  → Saved {n_ok}/{len(reviewer_results)} reviewers OK")
        except Exception as exc:
            errors += 1
            print(f"  [ERROR] {pid}: {type(exc).__name__}: {exc}")

    print(f"\n{'='*65}")
    print(f"  [HUMAN-{conf.upper()}] Done — {success} success, {errors} errors")
    print(f"  Results: {output_path}")
    print(f"{'='*65}")


# ═══════════════════════════════════════════════════════════════════════════════
# ICML / NeurIPS human — dedicated shortcuts
# ═══════════════════════════════════════════════════════════════════════════════

def run_icml_human(evaluator: ConstructivenessEvaluator, limit: Optional[int], with_paper: bool = False) -> None:
    run_human("icml2025", evaluator, limit, with_paper)


def run_neurips_human(evaluator: ConstructivenessEvaluator, limit: Optional[int], with_paper: bool = False) -> None:
    run_human("neurips2025", evaluator, limit, with_paper)


# ═══════════════════════════════════════════════════════════════════════════════
# Reviewer2 evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def run_reviewer2(
    conf: str,
    evaluator: ConstructivenessEvaluator,
    limit: Optional[int],
    with_paper: bool = False,
) -> None:
    r2_folder    = REVIEWER2_FOLDERS[conf]
    human_folder = HUMAN_FOLDERS[conf]
    ids_file     = PAPER_IDS_50[conf]
    output_path  = _out(conf, "reviewer2")

    pairs = discover_reviewer2_pairs(r2_folder, human_folder, ids_file)
    processed = load_processed_ids(output_path)
    todo = [(pid, hp, rp) for pid, hp, rp in pairs if pid not in processed]
    if limit:
        todo = todo[:limit]

    total    = len(pairs)
    done_pre = len(processed)
    todo_n   = len(todo)

    print(f"\n{'='*65}")
    print(f"  [REVIEWER2-{conf.upper()}] {done_pre}/{total} already done — {todo_n} remaining")
    print(f"{'='*65}")

    if not todo:
        print("  Nothing to do — all papers already processed.")
        return

    success, errors = 0, 0
    for i, (pid, h_path, r2_path) in enumerate(todo, 1):
        print(f"\n  [{i}/{todo_n}] Paper: {pid}  ({_pct(done_pre+i, total)})")
        try:
            human_data  = load_human_meta_json(h_path)
            metadata    = _load_metadata(human_data, conf)
            review_text = load_reviewer2_txt(r2_path)

            scored  = evaluator.score_review(review_text, "Reviewer2_LLM")
            metrics = compute_review_metrics(scored["atomic_comments"])

            record = {
                "paper_id":        pid,
                "metadata":        metadata,
                "reviewer_id":     "Reviewer2_LLM",
                "status":          scored.get("status", "unknown"),
                "atomic_comments": scored["atomic_comments"],
                "metrics":         metrics,
            }
            append_record(output_path, record)
            success += 1
            print(f"  → Saved (status={record['status']}, n_arcs={len(record['atomic_comments'])})")
        except Exception as exc:
            errors += 1
            print(f"  [ERROR] {pid}: {type(exc).__name__}: {exc}")

    print(f"\n{'='*65}")
    print(f"  [REVIEWER2-{conf.upper()}] Done — {success} success, {errors} errors")
    print(f"  Results: {output_path}")
    print(f"{'='*65}")


# ═══════════════════════════════════════════════════════════════════════════════
# Tree evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def run_tree(
    conf: str,
    evaluator: ConstructivenessEvaluator,
    limit: Optional[int],
    with_paper: bool = False,
) -> None:
    tree_folder  = TREE_FOLDERS[conf]
    human_folder = HUMAN_FOLDERS[conf]
    ids_file     = PAPER_IDS_50[conf]
    output_path  = _out(conf, "tree")

    pairs = discover_tree_pairs(tree_folder, human_folder, ids_file)
    processed = load_processed_ids(output_path)
    todo = [(pid, hp, tp) for pid, hp, tp in pairs if pid not in processed]
    if limit:
        todo = todo[:limit]

    total    = len(pairs)
    done_pre = len(processed)
    todo_n   = len(todo)

    print(f"\n{'='*65}")
    print(f"  [TREE-{conf.upper()}] {done_pre}/{total} already done — {todo_n} remaining")
    print(f"{'='*65}")

    if not todo:
        print("  Nothing to do — all papers already processed.")
        return

    success, errors = 0, 0
    for i, (pid, h_path, tree_path) in enumerate(todo, 1):
        print(f"\n  [{i}/{todo_n}] Paper: {pid}  ({_pct(done_pre+i, total)})")
        try:
            human_data  = load_human_meta_json(h_path)
            metadata    = _load_metadata(human_data, conf)
            review_text = load_tree_review_text(tree_path)

            if not review_text.strip():
                raise ValueError(f"No review text found in {tree_path}")

            scored  = evaluator.score_review(review_text, "Tree_LLM")
            metrics = compute_review_metrics(scored["atomic_comments"])

            record = {
                "paper_id":        pid,
                "metadata":        metadata,
                "reviewer_id":     "Tree_LLM",
                "status":          scored.get("status", "unknown"),
                "atomic_comments": scored["atomic_comments"],
                "metrics":         metrics,
            }
            append_record(output_path, record)
            success += 1
            print(f"  → Saved (status={record['status']}, n_arcs={len(record['atomic_comments'])})")
        except Exception as exc:
            errors += 1
            print(f"  [ERROR] {pid}: {type(exc).__name__}: {exc}")

    print(f"\n{'='*65}")
    print(f"  [TREE-{conf.upper()}] Done — {success} success, {errors} errors")
    print(f"  Results: {output_path}")
    print(f"{'='*65}")


# ═══════════════════════════════════════════════════════════════════════════════
# SEA evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def run_sea(
    conf: str,
    evaluator: ConstructivenessEvaluator,
    limit: Optional[int],
    with_paper: bool = False,
) -> None:
    sea_folder   = SEA_FOLDERS[conf]
    human_folder = HUMAN_FOLDERS[conf]
    ids_file     = PAPER_IDS_50[conf]
    output_path  = _out(conf, "sea")

    pairs = discover_sea_pairs(sea_folder, human_folder, ids_file)
    processed = load_processed_ids(output_path)
    todo = [(pid, hp, sp) for pid, hp, sp in pairs if pid not in processed]
    if limit:
        todo = todo[:limit]

    total    = len(pairs)
    done_pre = len(processed)
    todo_n   = len(todo)

    print(f"\n{'='*65}")
    print(f"  [SEA-{conf.upper()}] {done_pre}/{total} already done — {todo_n} remaining")
    print(f"{'='*65}")

    if not todo:
        print("  Nothing to do — all papers already processed.")
        return

    success, errors = 0, 0
    for i, (pid, h_path, s_path) in enumerate(todo, 1):
        print(f"\n  [{i}/{todo_n}] Paper: {pid}  ({_pct(done_pre+i, total)})")
        try:
            human_data  = load_human_meta_json(h_path)
            metadata    = _load_metadata(human_data, conf)
            review_text = load_llm_txt(s_path)

            scored  = evaluator.score_review(review_text, "SEA_Reviewer")
            metrics = compute_review_metrics(scored["atomic_comments"])

            record = {
                "paper_id":        pid,
                "metadata":        metadata,
                "reviewer_id":     "SEA_Reviewer",
                "status":          scored.get("status", "unknown"),
                "atomic_comments": scored["atomic_comments"],
                "metrics":         metrics,
            }
            append_record(output_path, record)
            success += 1
            print(f"  → Saved (status={record['status']}, n_arcs={len(record['atomic_comments'])})")
        except Exception as exc:
            errors += 1
            print(f"  [ERROR] {pid}: {type(exc).__name__}: {exc}")

    print(f"\n{'='*65}")
    print(f"  [SEA-{conf.upper()}] Done — {success} success, {errors} errors")
    print(f"  Results: {output_path}")
    print(f"{'='*65}")


# ═══════════════════════════════════════════════════════════════════════════════
# DeepReview evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def run_deepreview(
    conf: str,
    evaluator: ConstructivenessEvaluator,
    limit: Optional[int],
    with_paper: bool = False,
) -> None:
    dr_folder    = DEEPREVIEW_FOLDERS[conf]
    human_folder = HUMAN_FOLDERS[conf]
    ids_file     = PAPER_IDS_50[conf]
    output_path  = _out(conf, "deepreview")

    pairs = discover_deepreview_pairs(dr_folder, human_folder, ids_file)
    processed = load_processed_ids(output_path)
    todo = [(pid, hp, dp) for pid, hp, dp in pairs if pid not in processed]
    if limit:
        todo = todo[:limit]

    total    = len(pairs)
    done_pre = len(processed)
    todo_n   = len(todo)

    print(f"\n{'='*65}")
    print(f"  [DEEPREVIEW-{conf.upper()}] {done_pre}/{total} already done — {todo_n} remaining")
    print(f"{'='*65}")

    if not todo:
        print("  Nothing to do — all papers already processed.")
        return

    success, errors = 0, 0
    for i, (pid, h_path, dr_path) in enumerate(todo, 1):
        print(f"\n  [{i}/{todo_n}] Paper: {pid}  ({_pct(done_pre+i, total)})")
        try:
            human_data  = load_human_meta_json(h_path)
            metadata    = _load_metadata(human_data, conf)
            review_text = load_deepreview_text(dr_path, reviewer_id=1)

            if not review_text.strip():
                raise ValueError(f"No review text found for reviewer_id=1 in {dr_path}")

            scored  = evaluator.score_review(review_text, "DeepReview_LLM")
            metrics = compute_review_metrics(scored["atomic_comments"])

            record = {
                "paper_id":        pid,
                "metadata":        metadata,
                "reviewer_id":     "DeepReview_LLM",
                "status":          scored.get("status", "unknown"),
                "atomic_comments": scored["atomic_comments"],
                "metrics":         metrics,
            }
            append_record(output_path, record)
            success += 1
            print(f"  → Saved (status={record['status']}, n_arcs={len(record['atomic_comments'])})")
        except Exception as exc:
            errors += 1
            print(f"  [ERROR] {pid}: {type(exc).__name__}: {exc}")

    print(f"\n{'='*65}")
    print(f"  [DEEPREVIEW-{conf.upper()}] Done — {success} success, {errors} errors")
    print(f"  Results: {output_path}")
    print(f"{'='*65}")


# ═══════════════════════════════════════════════════════════════════════════════
# CycleReview evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def run_cyclereview(
    conf: str,
    evaluator: ConstructivenessEvaluator,
    limit: Optional[int],
    with_paper: bool = False,
) -> None:
    cr_folder   = CYCLEREVIEW_FOLDERS[conf]
    ids_file    = PAPER_IDS_50[conf]
    output_path = _out(conf, "cyclereview")

    pairs = get_cyclereview_pairs_from_ids(cr_folder, ids_file)
    processed = load_processed_ids(output_path)
    todo = [(pid, cp) for pid, cp in pairs if pid not in processed]
    if limit:
        todo = todo[:limit]

    total    = len(pairs)
    done_pre = len(processed)
    todo_n   = len(todo)

    print(f"\n{'='*65}")
    print(f"  [CYCLEREVIEW-{conf.upper()}] {done_pre}/{total} already done — {todo_n} remaining")
    print(f"{'='*65}")

    if not todo:
        print("  Nothing to do — all papers already processed.")
        return

    success, errors = 0, 0
    for i, (pid, cr_path) in enumerate(todo, 1):
        print(f"\n  [{i}/{todo_n}] Paper: {pid}  ({_pct(done_pre+i, total)})")
        try:
            metadata    = load_cyclereview_metadata(cr_path)
            review_text = load_cyclereview_first_text(cr_path)

            if not review_text.strip():
                raise ValueError(f"No review text found for first reviewer in {cr_path}")

            scored  = evaluator.score_review(review_text, "CycleReview_LLM")
            metrics = compute_review_metrics(scored["atomic_comments"])

            record = {
                "paper_id":        pid,
                "metadata":        metadata,
                "reviewer_id":     "CycleReview_LLM",
                "status":          scored.get("status", "unknown"),
                "atomic_comments": scored["atomic_comments"],
                "metrics":         metrics,
            }
            append_record(output_path, record)
            success += 1
            print(f"  → Saved (status={record['status']}, n_arcs={len(record['atomic_comments'])})")
        except Exception as exc:
            errors += 1
            print(f"  [ERROR] {pid}: {type(exc).__name__}: {exc}")

    print(f"\n{'='*65}")
    print(f"  [CYCLEREVIEW-{conf.upper()}] Done — {success} success, {errors} errors")
    print(f"  Results: {output_path}")
    print(f"{'='*65}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    print(f"[INFO] Mode: {args.mode} | Conf: {args.conf} | Provider: {args.provider}")
    print(f"[INFO] Paper IDs file: {PAPER_IDS_50.get(args.conf, 'N/A')}")
    print(f"[INFO] Initialising evaluator (provider={args.provider})...")

    try:
        evaluator = _build_evaluator(args)
    except Exception as exc:
        print(f"[FATAL] Could not init evaluator: {exc}")
        sys.exit(1)
    print("[INFO] Evaluator ready!\n")

    t0 = time.time()

    if args.mode == "human":
        run_human(args.conf, evaluator, args.limit, args.with_paper)

    elif args.mode == "sea":
        run_sea(args.conf, evaluator, args.limit, args.with_paper)

    elif args.mode == "deepreview":
        run_deepreview(args.conf, evaluator, args.limit, args.with_paper)

    elif args.mode == "icml_human":
        run_icml_human(evaluator, args.limit, args.with_paper)

    elif args.mode == "neurips_human":
        run_neurips_human(evaluator, args.limit, args.with_paper)

    elif args.mode == "reviewer2":
        run_reviewer2(args.conf, evaluator, args.limit, args.with_paper)

    elif args.mode == "tree":
        run_tree(args.conf, evaluator, args.limit, args.with_paper)

    elif args.mode == "cyclereview":
        run_cyclereview(args.conf, evaluator, args.limit, args.with_paper)

    elapsed = time.time() - t0
    print(f"\n[INFO] Total time elapsed: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()









