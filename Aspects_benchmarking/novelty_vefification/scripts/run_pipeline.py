#!/usr/bin/env python3
"""
Run the full novelty assessment pipeline (Task 1 → Task 2 → Task 3).

Supports two modes:
  1. Single paper:  --paper paper.txt --review review.txt
  2. Batch (data directory with conference structure):  --data-root /path/to/data

Output structure:
  output/<review_type>/<conference>/<paper_id>/
      task1_result.json   (extraction)
      task2_result.json   (related works)
      task3_result.json   (novelty judgment)

Usage:
  # Single paper
  python scripts/run_pipeline.py --paper paper.txt --review review.txt -o output/demo

  # Batch: all conferences, human reviews
  python scripts/run_pipeline.py --data-root /path/to/data

  # Batch: specific conferences, multiple review types, share task2
  python scripts/run_pipeline.py --data-root /path/to/data \\
      --conferences ICLR_2024 ICML_2025 \\
      --review-types human sea \\
      --share-task2

  # With custom LLM
  python scripts/run_pipeline.py --data-root /path/to/data \\
      --llm-provider openai --llm-model-name gpt-4o
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Conference definitions for the canonical DATA_ROOT layout.
# ---------------------------------------------------------------------------
CONFERENCES = [
    {"name": "ICLR_2024",    "folder": "ICLR2024",    "paper_dir": "papers", "paper_ext": ".txt", "year": 2024},
    {"name": "ICLR_2025",    "folder": "ICLR2025",    "paper_dir": "papers", "paper_ext": ".txt", "year": 2025},
    {"name": "ICLR_2026",    "folder": "ICLR2026",    "paper_dir": "papers", "paper_ext": ".txt", "year": 2026},
    {"name": "ICML_2025",    "folder": "ICML2025",    "paper_dir": "papers", "paper_ext": ".txt", "year": 2025},
    {"name": "NeurIPS_2025", "folder": "NeurIPS2025", "paper_dir": "papers", "paper_ext": ".txt", "year": 2025},
]

REVIEW_TYPE_DIRS = {
    "human": "human_reviews",
    "sea": "sea_reviews",
    "tree": "tree_reviews",
    "cyclereview": "cyclereview_reviews",
    "deepreview": "deepreview_reviews",
    "reviewer2": "reviewer2_reviews",
}

# Task 2 rate limiting
_task2_lock = threading.Lock()
_task2_last_call = 0.0
_TASK2_MIN_INTERVAL = 1.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("pipeline")


def resolve_review_dir(conf_root: Path, conf: dict, review_type: str) -> Optional[Path]:
    """Resolve the review directory for a given conference/review type."""
    conf_name = conf["name"].lower()
    candidates = []
    if review_type in REVIEW_TYPE_DIRS:
        candidates.append(conf_root / REVIEW_TYPE_DIRS[review_type])
    candidates.append(conf_root / review_type)
    candidates.append(conf_root / f"{review_type}_reviews")
    candidates.extend(sorted(p for p in conf_root.glob(f"{review_type}_*") if p.is_dir()))
    if "review_dir" in conf:
        candidates.append(conf_root / conf["review_dir"])
    seen = set()
    for c in candidates:
        key = str(c)
        if key not in seen and c.exists() and c.is_dir():
            seen.add(key)
            return c
    return None


def discover_papers(data_root: Path, conf: dict, review_type: str = "human") -> List[dict]:
    """Return list of paper entries for a conference."""
    conf_root = data_root / conf.get("folder", conf["name"])
    paper_dir = conf_root / conf["paper_dir"]
    review_dir = resolve_review_dir(conf_root, conf, review_type)
    if not paper_dir.exists() or review_dir is None:
        return []
    papers = []
    review_files = list(review_dir.glob("*.json")) + list(review_dir.glob("*.txt"))
    for review_file in sorted(review_files):
        stem = review_file.stem
        candidate_ids = [stem]
        if stem.endswith("_review"):
            candidate_ids.append(stem[:-7])
        if stem.endswith("-review"):
            candidate_ids.append(stem[:-7])
        for paper_id in candidate_ids:
            paper_file = paper_dir / (paper_id + conf["paper_ext"])
            if not paper_file.exists():
                paper_file = paper_dir / (paper_id + ".txt")
            if not paper_file.exists():
                paper_file = paper_dir / (paper_id + ".grobid.txt")
            if paper_file.exists():
                papers.append({
                    "paper_id": paper_id,
                    "conference": conf["name"],
                    "year": conf["year"],
                    "paper_path": paper_file,
                    "review_path": review_file,
                    "review_type": review_type,
                })
                break
    return papers


def _merge_text_parts(parts: List[str]) -> str:
    cleaned = [p.strip() for p in parts if isinstance(p, str) and p.strip()]
    return "\n\n".join(cleaned)


def _reviews_to_text(reviews: Any) -> str:
    if not isinstance(reviews, list) or not reviews:
        return ""
    if all(isinstance(r, str) for r in reviews):
        return _merge_text_parts(reviews)
    parts: List[str] = []
    if all(isinstance(r, dict) for r in reviews):
        for i, rev in enumerate(reviews):
            text = rev.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
                continue
            sections = []
            for key in ("Summary", "Strengths", "Weaknesses", "Questions", "Limitations",
                         "Soundness", "Presentation", "Contribution",
                         "summary", "strengths", "weaknesses", "questions", "limitations"):
                val = rev.get(key)
                if isinstance(val, str) and val.strip():
                    sections.append(f"## {key.capitalize()}\n{val.strip()}")
            if sections:
                reviewer_id = rev.get("reviewer_id") or i + 1
                rating = rev.get("Rating") or rev.get("rating") or "N/A"
                parts.append(f"### Reviewer {reviewer_id} (Rating: {rating})\n" + "\n".join(sections))
    return _merge_text_parts(parts)


def _generated_review_to_text(generated_review: Any) -> str:
    if isinstance(generated_review, str):
        return generated_review.strip()
    if isinstance(generated_review, dict):
        parts: List[str] = []
        parsed_reviews = _reviews_to_text(generated_review.get("reviews"))
        if parsed_reviews:
            parts.append(parsed_reviews)
            for key in ("meta_review", "decision"):
                val = generated_review.get(key)
                if isinstance(val, str) and val.strip():
                    parts.append(val.strip())
            return _merge_text_parts(parts)
        for key in ("content", "raw_text", "review", "text", "meta_review", "decision"):
            val = generated_review.get(key)
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())
        return _merge_text_parts(parts)
    if isinstance(generated_review, list):
        return _merge_text_parts([_generated_review_to_text(item) for item in generated_review])
    return ""


def load_review_text(review_path: Path) -> str:
    if review_path.suffix.lower() == ".txt":
        return review_path.read_text(encoding="utf-8")
    with open(review_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    full_review = data.get("full_review")
    if isinstance(full_review, str) and full_review.strip():
        return full_review.strip()
    generated_review_text = _generated_review_to_text(data.get("generated_review"))
    if generated_review_text:
        return generated_review_text
    reviews_text = _reviews_to_text(data.get("reviews"))
    if reviews_text:
        return reviews_text
    for key in ("review", "content", "text"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return json.dumps(data, ensure_ascii=False)


def load_paper_text(paper_path: Path) -> str:
    return paper_path.read_text(encoding="utf-8")


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Task runners
# ---------------------------------------------------------------------------

def run_task1_single(entry: dict, output_dir: Path, log: logging.Logger, skip_existing: bool) -> dict:
    from task1_extractor import extract_task1
    paper_id = entry["paper_id"]
    conference = entry["conference"]
    review_type = entry.get("review_type", "human")
    paper_out = output_dir / review_type / conference / paper_id
    task1_path = paper_out / "task1_result.json"
    if skip_existing and task1_path.exists():
        log.info(f"[{conference}/{paper_id}] Task 1 SKIP (exists)")
        return {"paper_id": paper_id, "conference": conference, "ok": True, "error": None,
                "result": json.loads(task1_path.read_text(encoding="utf-8"))}
    try:
        log.info(f"[{conference}/{paper_id}] Task 1 START")
        paper_text = load_paper_text(entry["paper_path"])
        review_text = load_review_text(entry["review_path"])
        task1 = extract_task1(paper_text=paper_text, review_text=review_text, logger=log)
        save_json(task1_path, task1)
        log.info(f"[{conference}/{paper_id}] Task 1 OK")
        return {"paper_id": paper_id, "conference": conference, "ok": True, "error": None, "result": task1}
    except Exception as e:
        log.error(f"[{conference}/{paper_id}] Task 1 FAILED: {e}")
        return {"paper_id": paper_id, "conference": conference, "ok": False, "error": f"Task1: {e}", "result": None}


def _rate_limit_task2() -> None:
    global _task2_last_call
    with _task2_lock:
        now = time.time()
        elapsed = now - _task2_last_call
        if elapsed < _TASK2_MIN_INTERVAL:
            time.sleep(_TASK2_MIN_INTERVAL - elapsed)
        _task2_last_call = time.time()


def run_task2_single(entry: dict, task1_result: dict, output_dir: Path, log: logging.Logger, skip_existing: bool) -> dict:
    from task2_related_works import retrieve_related_works
    paper_id = entry["paper_id"]
    conference = entry["conference"]
    review_type = entry.get("review_type", "human")
    paper_out = output_dir / review_type / conference / paper_id
    task2_path = paper_out / "task2_result.json"
    if skip_existing and task2_path.exists():
        log.info(f"[{conference}/{paper_id}] Task 2 SKIP (exists)")
        return {"paper_id": paper_id, "conference": conference, "ok": True, "error": None,
                "result": json.loads(task2_path.read_text(encoding="utf-8"))}
    try:
        _rate_limit_task2()
        log.info(f"[{conference}/{paper_id}] Task 2 START")
        task2 = retrieve_related_works(task1_result, paper_year=entry.get("year"), mode="per_contribution", logger=log)
        save_json(task2_path, task2)
        log.info(f"[{conference}/{paper_id}] Task 2 OK")
        return {"paper_id": paper_id, "conference": conference, "ok": True, "error": None, "result": task2}
    except Exception as e:
        log.error(f"[{conference}/{paper_id}] Task 2 FAILED: {e}")
        return {"paper_id": paper_id, "conference": conference, "ok": False, "error": f"Task2: {e}", "result": None}


def run_task3_single(entry: dict, task1_result: dict, task2_result: dict, output_dir: Path, log: logging.Logger, skip_existing: bool) -> dict:
    from task3_judge import extract_abstract_intro_from_text, run_task3_verification
    paper_id = entry["paper_id"]
    conference = entry["conference"]
    review_type = entry.get("review_type", "human")
    paper_out = output_dir / review_type / conference / paper_id
    task3_path = paper_out / "task3_result.json"
    if skip_existing and task3_path.exists():
        log.info(f"[{conference}/{paper_id}] Task 3 SKIP (exists)")
        return {"paper_id": paper_id, "conference": conference, "ok": True, "error": None}
    try:
        log.info(f"[{conference}/{paper_id}] Task 3 START")
        paper_text = load_paper_text(entry["paper_path"])
        paper_context = extract_abstract_intro_from_text(paper_text, max_chars=12000)
        review = task1_result.get("review") or {}
        claims = review.get("novelty_claims") or []
        review_sentences = []
        for idx, claim in enumerate(claims):
            if not isinstance(claim, dict):
                continue
            text = (claim.get("text") or "").strip()
            if not text:
                continue
            sid = claim.get("claim_id") or f"S_{idx + 1:03d}"
            review_sentences.append({"review_sentence_id": str(sid), "text": text})
        related_works = []
        for key in ("candidate_pool_top30", "candidate_pool_topN", "candidate_pool", "candidates"):
            val = task2_result.get(key)
            if isinstance(val, list):
                related_works = val
                break
        if not review_sentences:
            log.warning(f"[{conference}/{paper_id}] No novelty claims for Task 3")
            save_json(task3_path, {"note": "no novelty claims", "verdicts": []})
        else:
            task3 = run_task3_verification(review_sentences=review_sentences, paper_context=paper_context, related_works=related_works, logger=log)
            save_json(task3_path, task3)
            log.info(f"[{conference}/{paper_id}] Task 3 OK")
        return {"paper_id": paper_id, "conference": conference, "ok": True, "error": None}
    except Exception as e:
        log.error(f"[{conference}/{paper_id}] Task 3 FAILED: {e}")
        return {"paper_id": paper_id, "conference": conference, "ok": False, "error": f"Task3: {e}"}


# ---------------------------------------------------------------------------
# Single-paper mode
# ---------------------------------------------------------------------------

def run_single_paper(paper_path: Path, review_path: Path, output_dir: Path, log: logging.Logger, paper_title: Optional[str] = None) -> None:
    """Run the full pipeline on a single paper+review pair."""
    from task1_extractor import extract_task1
    from task2_related_works import retrieve_related_works
    from task3_judge import extract_abstract_intro_from_text, run_task3_verification

    paper_text = load_paper_text(paper_path)
    review_text = load_review_text(review_path)
    paper_id = paper_path.stem

    # Task 1
    log.info("=" * 60)
    log.info("TASK 1: Extracting claims and structure")
    log.info("=" * 60)
    task1 = extract_task1(paper_text=paper_text, review_text=review_text, logger=log)
    save_json(output_dir / "task1_result.json", task1)
    log.info("✓ Task 1 saved to %s", output_dir / "task1_result.json")

    # Task 2
    log.info("=" * 60)
    log.info("TASK 2: Retrieving related works")
    log.info("=" * 60)
    task2 = retrieve_related_works(task1, mode="per_contribution", logger=log)
    save_json(output_dir / "task2_result.json", task2)
    candidates = task2.get("candidate_pool_top30", [])
    log.info("✓ Task 2 saved to %s (%d candidates)", output_dir / "task2_result.json", len(candidates))

    # Task 3
    log.info("=" * 60)
    log.info("TASK 3: Verifying novelty claims")
    log.info("=" * 60)
    paper_context = extract_abstract_intro_from_text(paper_text, max_chars=12000)
    review_data = task1.get("review") or {}
    claims = review_data.get("novelty_claims") or []
    review_sentences = []
    for idx, claim in enumerate(claims):
        if not isinstance(claim, dict):
            continue
        text = (claim.get("text") or "").strip()
        if not text:
            continue
        sid = claim.get("claim_id") or f"S_{idx + 1:03d}"
        review_sentences.append({"review_sentence_id": str(sid), "text": text})

    if not review_sentences:
        log.warning("No novelty claims found — skipping Task 3")
        save_json(output_dir / "task3_result.json", {"note": "no novelty claims", "verdicts": []})
    else:
        task3 = run_task3_verification(
            review_sentences=review_sentences,
            paper_context=paper_context,
            related_works=candidates,
            logger=log,
        )
        save_json(output_dir / "task3_result.json", task3)
        log.info("✓ Task 3 saved to %s", output_dir / "task3_result.json")

    # Summary
    n_claims = len(claims)
    n_candidates = len(candidates)
    verdicts = (task3 if not review_sentences else task3).get("verdicts", [])
    log.info("=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info(f"  Paper:       {paper_id}")
    log.info(f"  Claims:      {n_claims}")
    log.info(f"  Candidates:  {n_candidates}")
    log.info(f"  Verdicts:    {len(verdicts)}")
    log.info(f"  Output:      {output_dir}")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

def run_batch(data_root: Path, output_dir: Path, conferences: Optional[List[str]],
              review_types: List[str], share_task2: bool, max_workers: int,
              task2_workers: int, max_papers: Optional[int], skip_existing: bool,
              log: logging.Logger) -> None:
    """Run the full pipeline on a batch of papers from a data directory."""
    selected_confs = [c["name"] for c in CONFERENCES] if not conferences else conferences
    all_papers = []
    for review_type in review_types:
        for conf in CONFERENCES:
            if conf["name"] not in selected_confs:
                continue
            papers = discover_papers(data_root, conf, review_type=review_type)
            if max_papers:
                papers = papers[:max_papers]
            log.info(f"{conf['name']} ({review_type}): {len(papers)} papers")
            all_papers.extend(papers)

    log.info(f"Total papers to process: {len(all_papers)}")
    if not all_papers:
        log.error("No papers found. Check --data-root path.")
        sys.exit(1)

    t0 = time.time()
    entry_map = {(e["conference"], e["paper_id"], e.get("review_type", "human")): e for e in all_papers}

    # Phase 1: Task 1
    log.info("=" * 60)
    log.info(f"PHASE 1: Extracting claims — {len(all_papers)} papers, {max_workers} workers")
    log.info("=" * 60)
    task1_results: Dict[tuple, dict] = {}
    task1_ok = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_task1_single, entry, output_dir, log, skip_existing): entry for entry in all_papers}
        for future in as_completed(futures):
            entry = futures[future]
            key = (entry["conference"], entry["paper_id"], entry.get("review_type", "human"))
            res = future.result()
            task1_results[key] = res
            if res["ok"]:
                task1_ok += 1
    t1 = time.time()
    log.info(f"PHASE 1 COMPLETE: {task1_ok}/{len(all_papers)} succeeded in {t1 - t0:.0f}s")

    task1_passed = [e for e in all_papers if task1_results[(e["conference"], e["paper_id"], e.get("review_type", "human"))]["ok"]]

    # Phase 2: Task 2
    log.info("=" * 60)
    if share_task2:
        primary_rt = review_types[0]
        task2_entries = [e for e in task1_passed if e.get("review_type", "human") == primary_rt]
        log.info(f"PHASE 2: Retrieving related works (--share-task2) — {len(task2_entries)} papers")
    else:
        task2_entries = task1_passed
        log.info(f"PHASE 2: Retrieving related works — {len(task2_entries)} papers")
    log.info("=" * 60)
    task2_results: Dict[tuple, dict] = {}
    task2_ok = 0
    with ThreadPoolExecutor(max_workers=task2_workers) as executor:
        futures = {}
        for entry in task2_entries:
            key = (entry["conference"], entry["paper_id"], entry.get("review_type", "human"))
            t1_res = task1_results[key]["result"]
            futures[executor.submit(run_task2_single, entry, t1_res, output_dir, log, skip_existing)] = entry
        for future in as_completed(futures):
            entry = futures[future]
            key = (entry["conference"], entry["paper_id"], entry.get("review_type", "human"))
            res = future.result()
            task2_results[key] = res
            if res["ok"]:
                task2_ok += 1
    if share_task2:
        primary_rt = review_types[0]
        for entry in task1_passed:
            rt = entry.get("review_type", "human")
            if rt != primary_rt:
                pk = (entry["conference"], entry["paper_id"], primary_rt)
                if pk in task2_results:
                    task2_results[(entry["conference"], entry["paper_id"], rt)] = task2_results[pk]
    t2 = time.time()
    log.info(f"PHASE 2 COMPLETE: {task2_ok}/{len(task1_passed)} succeeded in {t2 - t1:.0f}s")

    task2_passed = [e for e in task1_passed if task2_results.get((e["conference"], e["paper_id"], e.get("review_type", "human")), {}).get("ok")]

    # Phase 3: Task 3
    log.info("=" * 60)
    log.info(f"PHASE 3: Judging novelty — {len(task2_passed)} papers, {max_workers} workers")
    log.info("=" * 60)
    task3_ok = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for entry in task2_passed:
            key = (entry["conference"], entry["paper_id"], entry.get("review_type", "human"))
            t1_res = task1_results[key]["result"]
            t2_res = task2_results[key]["result"]
            futures[executor.submit(run_task3_single, entry, t1_res, t2_res, output_dir, log, skip_existing)] = entry
        for future in as_completed(futures):
            res = future.result()
            if res["ok"]:
                task3_ok += 1
    t3 = time.time()
    log.info(f"PHASE 3 COMPLETE: {task3_ok}/{len(task2_passed)} succeeded in {t3 - t2:.0f}s")

    # Summary
    elapsed = time.time() - t0
    task2_passed_keys = {(e["conference"], e["paper_id"], e.get("review_type", "human")) for e in task2_passed}
    results = []
    for entry in all_papers:
        key = (entry["conference"], entry["paper_id"], entry.get("review_type", "human"))
        t1r = task1_results.get(key, {})
        t2r = task2_results.get(key, {})
        rt = entry.get("review_type", "human")
        task3_path = output_dir / rt / entry["conference"] / entry["paper_id"] / "task3_result.json"
        results.append({
            "paper_id": entry["paper_id"],
            "conference": entry["conference"],
            "review_type": rt,
            "task1_ok": t1r.get("ok", False),
            "task2_ok": t2r.get("ok", False),
            "task3_ok": key in task2_passed_keys and task3_path.exists(),
            "error": t1r.get("error") or t2r.get("error"),
        })
    ok1 = sum(1 for r in results if r["task1_ok"])
    ok2 = sum(1 for r in results if r["task2_ok"])
    ok3 = sum(1 for r in results if r["task3_ok"])
    full_ok = sum(1 for r in results if r["task1_ok"] and r["task2_ok"] and r["task3_ok"])
    failed = [r for r in results if r.get("error")]
    summary = {
        "total_papers": len(all_papers),
        "task1_success": ok1,
        "task2_success": ok2,
        "task3_success": ok3,
        "full_pipeline_success": full_ok,
        "failed": len(failed),
        "elapsed_seconds": round(elapsed, 1),
        "failures": failed,
    }
    save_json(output_dir / "_pipeline_summary.json", summary)
    log.info("=" * 60)
    log.info(f"PIPELINE COMPLETE in {elapsed:.0f}s")
    log.info(f"  Task 1:  {ok1}/{len(all_papers)}")
    log.info(f"  Task 2:  {ok2}/{len(all_papers)}")
    log.info(f"  Task 3:  {ok3}/{len(all_papers)}")
    log.info(f"  Full OK: {full_ok}/{len(all_papers)}")
    log.info(f"  Failed:  {len(failed)}")
    log.info("=" * 60)
    if failed:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full novelty assessment pipeline (Task 1 → 2 → 3).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single paper
  python scripts/run_pipeline.py --paper paper.txt --review review.txt -o output/demo

  # Batch: all conferences
  python scripts/run_pipeline.py --data-root /path/to/data

  # Batch: specific conferences, multiple review types
  python scripts/run_pipeline.py --data-root /path/to/data \\
      --conferences ICLR_2024 ICML_2025 --review-types human sea --share-task2
        """,
    )

    # Mode: single paper or batch
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--paper", type=Path, help="Path to paper text file (single-paper mode)")
    mode.add_argument("--data-root", type=Path, help="Root of conference data directory (batch mode)")

    # Single-paper options
    parser.add_argument("--review", type=Path, help="Path to review text file (required with --paper)")
    parser.add_argument("--title", type=str, help="Paper title (optional, single-paper mode)")

    # Batch options
    parser.add_argument("--conferences", nargs="*", default=None, help="Run only specific conferences")
    parser.add_argument("--review-types", nargs="*", default=["human"], help="Review types to process (default: human)")
    parser.add_argument("--share-task2", action="store_true", help="Reuse Task 2 across review types")
    parser.add_argument("--no-skip", action="store_true", help="Re-run even if outputs exist")
    parser.add_argument("--max-workers", type=int, default=20, help="Concurrent workers for Task 1/3 (default: 20)")
    parser.add_argument("--task2-workers", type=int, default=1, help="Concurrent workers for Task 2 (default: 1)")
    parser.add_argument("--max-papers", type=int, default=None, help="Max papers per conference (default: all)")

    # Output
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("output/pipeline_results"), help="Output directory")

    # LLM options
    parser.add_argument("--llm-provider", type=str, help="LLM provider")
    parser.add_argument("--llm-api-key", type=str, help="LLM API key")
    parser.add_argument("--llm-api-endpoint", type=str, help="LLM API endpoint URL")
    parser.add_argument("--llm-model-name", type=str, help="LLM model name")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if args.paper and not args.review:
        parser.error("--paper requires --review")

    # Set env vars
    if args.llm_provider:
        os.environ["LLM_PROVIDER"] = args.llm_provider
    if args.llm_api_key:
        os.environ["LLM_API_KEY"] = args.llm_api_key
    if args.llm_api_endpoint:
        os.environ["LLM_API_ENDPOINT"] = args.llm_api_endpoint
    if args.llm_model_name:
        os.environ["LLM_MODEL_NAME"] = args.llm_model_name

    log = setup_logging(verbose=args.verbose)

    if args.paper:
        # Single-paper mode
        if not args.paper.exists():
            log.error(f"Paper file not found: {args.paper}")
            sys.exit(1)
        if not args.review.exists():
            log.error(f"Review file not found: {args.review}")
            sys.exit(1)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        run_single_paper(args.paper, args.review, args.output_dir, log, paper_title=args.title)
    else:
        # Batch mode
        log.info(f"Using conferences: {[c['name'] for c in CONFERENCES]}")
        run_batch(
            data_root=args.data_root,
            output_dir=args.output_dir,
            conferences=args.conferences,
            review_types=args.review_types,
            share_task2=args.share_task2,
            max_workers=args.max_workers,
            task2_workers=args.task2_workers,
            max_papers=args.max_papers,
            skip_existing=not args.no_skip,
            log=log,
        )


if __name__ == "__main__":
    main()
