"""
Flaw-Identification Pipeline – ICLR 2024 entry point.

Data paths  : <DATA_ROOT>/ICLR2024/
Default LLM : Google Gemini 2.5 Flash Lite  (google-genai library)

Three modes
-----------
  cfi_only  – run Step 1 (flaw extraction) + Step 2 (validation) + CFI recall metrics
              → outputs to output_cfi_iclr2024/all_papers_results.jsonl
  all       – same as cfi_only, then also compute CPS coverage metrics
  cps_only  – skip Step 1/2, read a cached CFI JSONL, compute CPS only
              → default cache: output_cfi_iclr2024/all_papers_results.jsonl

Example commands
----------------
  # Run CFI only (recommended first pass):
  python main_cfi_iclr2024.py --mode cfi_only

  # Run full pipeline (CFI + CPS):
  python main_cfi_iclr2024.py --mode all

  # Run CPS only from a saved CFI result:
  python main_cfi_iclr2024.py --mode cps_only

  # Override model:
  python main_cfi_iclr2024.py --mode cfi_only --model gemini-2.5-flash
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Literal

# ---------------------------------------------------------------------------
# Make sure the parent directory is on the path so `src.*` imports work when
# running from the flaw_identification folder.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from src.cfi_metrics import DecoupledMetricsCalculator
from src.cps_metrics import SEVERITY_MAP
from src.evaluator import (
    ReviewEvaluatorPipeline,
    build_cps_issue_bank,
    process_single_review_for_cps,
)
from src.azure_openai_client import AzureOpenAIConfigError
from src.utils import (
    extract_from_cyclereview_json,
    extract_from_deepreview_json,
    extract_from_reviewer2_txt,
    format_human_review_text,
    get_paper_pairs,
    get_paper_pairs_cyclereview,
    get_paper_pairs_deepreview,
    get_paper_pairs_reviewer2,
    get_paper_pairs_tree,
    load_human_meta_json,
    load_llm_txt,
    load_paper_content,
    load_tree_review_from_path,
)

# ---------------------------------------------------------------------------
# Path configuration  –  all data lives under ICLR2024_DATA
# ---------------------------------------------------------------------------
from paths_config import ICLR2024_DATA
HUMAN_FOLDER         = os.path.join(ICLR2024_DATA, "human_reviews")
SEA_FOLDER           = os.path.join(ICLR2024_DATA, "sea_iclr2024")
TREE_FOLDER          = os.path.join(ICLR2024_DATA, "tree_iclr2024_2")
REVIEWER2_FOLDER     = os.path.join(ICLR2024_DATA, "reviewer2_iclr2024")
DEEPREVIEW_FOLDER    = os.path.join(ICLR2024_DATA, "deepreview_iclr2024")
CYCLEREVIEW_FOLDER   = os.path.join(ICLR2024_DATA, "cyclereview_iclr2024")
PAPERS_FOLDER        = os.path.join(ICLR2024_DATA, "papers")
PAPER_IDS_FILE       = os.path.join(ICLR2024_DATA, "paper_ids_200_iclr2024.txt")
OUTPUT_DIR           = os.path.join(_HERE, "output_cfi_iclr2024_2")

VALID_MODES     = {"all", "cfi_only", "cps_only"}
VALID_PROVIDERS = {"gemini", "openai", "azure"}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run flaw-identification pipeline on ICLR 2024 data.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=sorted(VALID_MODES),
        default="cfi_only",
        help=(
            "Pipeline mode:\n"
            "  cfi_only  – Step 1 + Step 2 + CFI recall metrics (default)\n"
            "  all       – cfi_only + CPS coverage metrics\n"
            "  cps_only  – CPS from cached CFI JSONL (no paper text needed)"
        ),
    )
    parser.add_argument(
        "--provider",
        choices=sorted(VALID_PROVIDERS),
        default=None,
        help="LLM provider  (default: gemini)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Override model name for the chosen provider.\n"
            "Gemini default : gemini-2.5-flash-lite\n"
            "OpenAI default : gpt-4o-mini"
        ),
    )
    parser.add_argument(
        "--cfi-cache",
        default=None,
        help=(
            "Path to a cached CFI JSONL file (only used in cps_only mode).\n"
            f"Default: {OUTPUT_DIR}/all_papers_results.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Override output directory  (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--llm-type",
        choices=["sea", "tree", "reviewer2", "deepreview", "cyclereview"],
        default="sea",
        help=(
            "LLM reviewer source:\n"
            "  sea          – SEA reviews from sea_iclr2024/ (default)\n"
            "  tree         – Tree reviews from tree_iclr2024_full/\n"
            "  reviewer2    – Reviewer2 reviews from reviewer2_iclr2024/\n"
            "  deepreview   – DeepReview JSON from deepreview_iclr2024/\n"
            "  cyclereview  – CycleReview JSON from cyclereview_iclr2024/"
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------

def _build_pipeline(provider: str | None, model: str | None) -> ReviewEvaluatorPipeline:
    return ReviewEvaluatorPipeline(provider=provider, model=model)


# ---------------------------------------------------------------------------
# Core processing: full pipeline (all / cfi_only)
# ---------------------------------------------------------------------------

def process_single_paper(
    paper_id: str,
    h_path: str,
    llm_path: str,
    pipeline: ReviewEvaluatorPipeline,
    mode: Literal["all", "cfi_only"],
    llm_type: str = "sea",
) -> dict | None:
    print(f"\n--- Processing Paper ID: {paper_id} ---")

    # Load paper content (.grobid.txt)
    paper_content = load_paper_content(paper_id, PAPERS_FOLDER)
    if not paper_content:
        print(f"  [SKIP] No paper file for {paper_id}")
        return None

    # Load human reviews
    human_data = load_human_meta_json(h_path)
    human_reviews_dict: dict[str, str] = {}
    human_list = human_data.get("reviews", []) if isinstance(human_data, dict) else human_data
    for idx, review_obj in enumerate(human_list):
        text = format_human_review_text(review_obj)
        if text.strip():
            human_reviews_dict[f"Human_{idx + 1}"] = text
    human_ids = list(human_reviews_dict.keys())

    # Load LLM review (SEA, Tree, Reviewer2, DeepReview, or CycleReview)
    if llm_type == "tree":
        llm_review_text = load_tree_review_from_path(llm_path)
        if not llm_review_text:
            print(f"  [SKIP] Empty tree review for {paper_id}")
            return None
    elif llm_type == "reviewer2":
        raw_r2 = load_llm_txt(llm_path)
        llm_review_text = extract_from_reviewer2_txt(raw_r2)
    elif llm_type == "deepreview":
        llm_review_text = extract_from_deepreview_json(llm_path)
        if not llm_review_text:
            print(f"  [SKIP] Empty deepreview for {paper_id}")
            return None
    elif llm_type == "cyclereview":
        llm_review_text = extract_from_cyclereview_json(llm_path)
        if not llm_review_text:
            print(f"  [SKIP] Empty cyclereview for {paper_id}")
            return None
    else:
        llm_review_text = load_llm_txt(llm_path)

    # --- Step 1: Flaw extraction ---
    print(">> Step 1: Extracting micro-flaws...")
    try:
        step1_flaws = pipeline.step1_atomize_and_group(human_reviews_dict, llm_review_text)
    except Exception as exc:
        print(f"[ERROR] Processing {paper_id}: Step 1 failed: {exc}")
        return None

    # --- Step 2: Flaw validation ---
    print(">> Step 2: Validating flaws against paper...")
    try:
        step2_evals = pipeline.step2_judge_flaws(paper_content, step1_flaws)
    except Exception as exc:
        print(f"[ERROR] Processing {paper_id}: Step 2 failed: {exc}")
        return None

    # --- Step 3: Metrics ---
    print(">> Step 3: Computing metrics...")
    report: dict = {}

    total_reviewers = len(human_ids) + 1  # humans + LLM
    cfi_calc = DecoupledMetricsCalculator(
        micro_flaws_json=step1_flaws,
        evaluations_json=step2_evals,
        total_reviewers_count=total_reviewers,
    )
    report["cfi"] = cfi_calc.generate_final_report(human_ids)

    if mode == "all":
        report["cps"] = _compute_cps_for_paper(
            pipeline, step1_flaws, step2_evals, human_reviews_dict, llm_review_text,
        )

    report["mode"] = mode

    return {
        "paper_id":       paper_id,
        "micro_flaws":    step1_flaws,
        "evaluations":    step2_evals,
        "metrics_report": report,
    }


# ---------------------------------------------------------------------------
# CPS-only  (from cached CFI JSONL)
# ---------------------------------------------------------------------------

def process_cps_from_cache(
    cached_record: dict,
    pipeline: ReviewEvaluatorPipeline,
    llm_type: str = "sea",
) -> dict | None:
    paper_id     = cached_record["paper_id"]
    step1_flaws  = cached_record["micro_flaws"]
    step2_evals  = cached_record["evaluations"]
    print(f"\n--- CPS from cache: {paper_id} ---")

    h_path = os.path.join(HUMAN_FOLDER, f"{paper_id}.json")
    if llm_type == "tree":
        llm_path = os.path.join(TREE_FOLDER, f"{paper_id}_review.json")
    elif llm_type == "reviewer2":
        llm_path = os.path.join(REVIEWER2_FOLDER, f"{paper_id}.txt")
    elif llm_type == "deepreview":
        llm_path = os.path.join(DEEPREVIEW_FOLDER, f"{paper_id}.json")
    elif llm_type == "cyclereview":
        llm_path = os.path.join(CYCLEREVIEW_FOLDER, f"{paper_id}.json")
    else:
        llm_path = os.path.join(SEA_FOLDER, f"{paper_id}.txt")

    if not os.path.exists(h_path):
        print(f"  [WARNING] Missing human review: {h_path}")
        return None
    if not os.path.exists(llm_path):
        print(f"  [WARNING] Missing {llm_type} review: {llm_path}")
        return None

    human_data = load_human_meta_json(h_path)
    human_reviews_dict: dict[str, str] = {}
    human_list = human_data.get("reviews", []) if isinstance(human_data, dict) else human_data
    for idx, review_obj in enumerate(human_list):
        text = format_human_review_text(review_obj)
        if text.strip():
            human_reviews_dict[f"Human_{idx + 1}"] = text

    if llm_type == "tree":
        llm_review_text = load_tree_review_from_path(llm_path)
        if not llm_review_text:
            print(f"  [WARNING] Empty tree review for {paper_id}")
            return None
    elif llm_type == "reviewer2":
        raw_r2 = load_llm_txt(llm_path)
        llm_review_text = extract_from_reviewer2_txt(raw_r2)
    elif llm_type == "deepreview":
        llm_review_text = extract_from_deepreview_json(llm_path)
        if not llm_review_text:
            print(f"  [WARNING] Empty deepreview for {paper_id}")
            return None
    elif llm_type == "cyclereview":
        llm_review_text = extract_from_cyclereview_json(llm_path)
        if not llm_review_text:
            print(f"  [WARNING] Empty cyclereview for {paper_id}")
            return None
    else:
        llm_review_text = load_llm_txt(llm_path)

    print(">> Computing CPS metrics (using cached Step 1+2 outputs)...")
    cps_report = _compute_cps_for_paper(
        pipeline, step1_flaws, step2_evals, human_reviews_dict, llm_review_text,
    )

    report: dict = {"cps": cps_report, "mode": "cps_only"}
    cached_metrics = cached_record.get("metrics_report", {})
    if "cfi" in cached_metrics:
        report["cfi"] = cached_metrics["cfi"]

    return {
        "paper_id":       paper_id,
        "micro_flaws":    step1_flaws,
        "evaluations":    step2_evals,
        "metrics_report": report,
    }


# ---------------------------------------------------------------------------
# Shared CPS computation
# ---------------------------------------------------------------------------

def _compute_cps_for_paper(
    pipeline: ReviewEvaluatorPipeline,
    step1_flaws: dict,
    step2_evals: dict,
    human_reviews_dict: dict[str, str],
    llm_review_text: str,
) -> dict:
    issue_bank = build_cps_issue_bank(step1_flaws, step2_evals)
    cps_rows: list[dict] = []

    for reviewer_id, review_text in human_reviews_dict.items():
        result = process_single_review_for_cps(
            pipeline=pipeline,
            review_text=review_text,
            reviewer_id=reviewer_id,
            issue_bank=issue_bank,
        )
        cps_rows.append(result["metrics"])

    llm_result = process_single_review_for_cps(
        pipeline=pipeline,
        review_text=llm_review_text,
        reviewer_id="LLM_Reviewer",
        issue_bank=issue_bank,
    )
    cps_rows.append(llm_result["metrics"])

    raw_values = [r["Raw_CPS"] for r in cps_rows]
    cps_min   = min(raw_values) if raw_values else 0.0
    cps_max   = max(raw_values) if raw_values else 0.0
    cps_range = cps_max - cps_min
    for row in cps_rows:
        if cps_range > 0:
            row["CPS_norm"] = round((row["Raw_CPS"] - cps_min) / cps_range, 4)
        else:
            row["CPS_norm"] = 1.0 if row["Raw_CPS"] > 0 else 0.0

    cps_rows.sort(key=lambda r: (r["nCPS"], r["Raw_CPS"]), reverse=True)
    return {
        "Severity_Map":       SEVERITY_MAP,
        "Canonical_Issue_Bank": issue_bank,
        "Reviewer_Rankings":  cps_rows,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_cfi_cache(jsonl_path: str) -> list[dict]:
    records: list[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "paper_id" in data and "micro_flaws" in data and "evaluations" in data:
                    records.append(data)
            except json.JSONDecodeError:
                continue
    return records


def _load_target_paper_ids() -> set[str]:
    if not os.path.exists(PAPER_IDS_FILE):
        raise FileNotFoundError(f"Paper-IDs file not found: {PAPER_IDS_FILE}")
    ids: set[str] = set()
    with open(PAPER_IDS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            pid = line.strip()
            if pid and not pid.startswith("#"):
                ids.add(pid)
    if not ids:
        raise ValueError(f"No paper IDs found in: {PAPER_IDS_FILE}")
    return ids


def _load_processed_ids(jsonl_path: str) -> set[str]:
    """Read already-processed paper IDs from an existing output JSONL."""
    done: set[str] = set()
    if not os.path.exists(jsonl_path):
        return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "paper_id" in data:
                    done.add(data["paper_id"])
            except json.JSONDecodeError:
                continue
    return done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    mode   = args.mode
    llm_type = args.llm_type

    # Separate output dirs for each llm-type
    if args.output_dir:
        output_dir = args.output_dir
    elif llm_type == "tree":
        output_dir = OUTPUT_DIR + "_tree"
    elif llm_type == "reviewer2":
        output_dir = OUTPUT_DIR + "_reviewer2"
    elif llm_type == "deepreview":
        output_dir = OUTPUT_DIR + "_deepreview"
    elif llm_type == "cyclereview":
        output_dir = OUTPUT_DIR + "_cyclereview"
    else:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Mode      : {mode}")
    print(f"[INFO] LLM type  : {llm_type}")
    print(f"[INFO] Provider  : {args.provider}")
    print(f"[INFO] Model     : {args.model or '(default for provider)'}")
    print(f"[INFO] Output    : {output_dir}")

    # Load target paper IDs
    try:
        target_ids = _load_target_paper_ids()
    except (FileNotFoundError, ValueError) as exc:
        print(f"[FATAL] {exc}")
        return
    print(f"[INFO] {len(target_ids)} target paper IDs loaded from {PAPER_IDS_FILE}")

    # Build LLM pipeline
    print(f"\n[INFO] Initialising pipeline (provider={args.provider})...")
    try:
        pipeline = _build_pipeline(args.provider, args.model)
    except (AzureOpenAIConfigError, RuntimeError) as exc:
        print(f"[FATAL] Pipeline init failed: {exc}")
        return
    print("[INFO] Pipeline ready!\n")

    # -----------------------------------------------------------------------
    # Mode: cps_only
    # -----------------------------------------------------------------------
    if mode == "cps_only":
        default_cache = os.path.join(output_dir, "all_papers_results.jsonl")
        cache_path = args.cfi_cache or default_cache
        if not os.path.exists(cache_path):
            print(f"[FATAL] CFI cache not found: {cache_path}")
            print("  Run --mode cfi_only first to generate the cache.")
            return

        cached_records = _load_cfi_cache(cache_path)
        cached_records = [r for r in cached_records if r["paper_id"] in target_ids]
        print(f"[INFO] {len(cached_records)} cached papers will be processed for CPS.")

        out_path = os.path.join(output_dir, "cps_results.jsonl")
        done_ids = _load_processed_ids(out_path)
        print(f"[INFO] {len(done_ids)} already done — {len(cached_records) - len(done_ids)} remaining\n")

        for record in cached_records:
            pid = record["paper_id"]
            if pid in done_ids:
                print(f"[SKIP] {pid}")
                continue
            try:
                result = process_cps_from_cache(record, pipeline, llm_type=llm_type)
                if result:
                    with open(out_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    done_ids.add(pid)
            except Exception as exc:
                print(f"[ERROR] CPS {pid}: {exc}")

        print(f"\n[OK] CPS done! Results → {out_path}")
        return

    # -----------------------------------------------------------------------
    # Mode: cfi_only / all
    # -----------------------------------------------------------------------
    if llm_type == "tree":
        if not os.path.isdir(TREE_FOLDER):
            print(f"[FATAL] Tree review folder not found: {TREE_FOLDER}")
            return
        paper_pairs = get_paper_pairs_tree(HUMAN_FOLDER, TREE_FOLDER)
    elif llm_type == "reviewer2":
        if not os.path.isdir(REVIEWER2_FOLDER):
            print(f"[FATAL] Reviewer2 folder not found: {REVIEWER2_FOLDER}")
            return
        paper_pairs = get_paper_pairs_reviewer2(HUMAN_FOLDER, REVIEWER2_FOLDER)
    elif llm_type == "deepreview":
        if not os.path.isdir(DEEPREVIEW_FOLDER):
            print(f"[FATAL] DeepReview folder not found: {DEEPREVIEW_FOLDER}")
            return
        paper_pairs = get_paper_pairs_deepreview(HUMAN_FOLDER, DEEPREVIEW_FOLDER)
    elif llm_type == "cyclereview":
        if not os.path.isdir(CYCLEREVIEW_FOLDER):
            print(f"[FATAL] CycleReview folder not found: {CYCLEREVIEW_FOLDER}")
            return
        paper_pairs = get_paper_pairs_cyclereview(HUMAN_FOLDER, CYCLEREVIEW_FOLDER)
    else:
        paper_pairs = get_paper_pairs(HUMAN_FOLDER, SEA_FOLDER)

    paper_pairs = [(pid, h, llm) for pid, h, llm in paper_pairs if pid in target_ids]
    print(f"[INFO] {len(paper_pairs)} papers matched target IDs.")

    out_path = os.path.join(output_dir, "all_papers_results.jsonl")
    done_ids = _load_processed_ids(out_path)
    remaining = [p for p in paper_pairs if p[0] not in done_ids]
    total     = len(paper_pairs)
    print(f"[INFO] {len(done_ids)} already done — {len(remaining)} remaining\n")

    for i, (paper_id, h_path, llm_path) in enumerate(remaining, start=1):
        current_done = len(done_ids)
        pct = (current_done / total * 100) if total else 0
        print(f"=================================================================")
        print(f"  [{i}/{len(remaining)}] Paper: {paper_id}  (total progress: {current_done + 1}/{total}, {pct:.1f}%)")
        print(f"=================================================================")

        try:
            result = process_single_paper(
                paper_id=paper_id,
                h_path=h_path,
                llm_path=llm_path,
                pipeline=pipeline,
                mode=mode,
                llm_type=llm_type,
            )
            if result:
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                done_ids.add(paper_id)
                print(f"  [SAVED] {paper_id}")
        except Exception as exc:
            print(f"[ERROR] Processing {paper_id}: {exc}")

    print(f"\n[OK] Pipeline completed! Results → {out_path}")
    print(f"     Total processed : {len(done_ids)}/{total}")


if __name__ == "__main__":
    main()

