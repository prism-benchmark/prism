"""
Flaw-Identification Pipeline – ICLR 2026 entry point.

Data paths  : <DATA_ROOT>/ICLR2026/
Default LLM : Google Gemini 2.5 Flash Lite  (google-genai library)

Key differences from ICLR 2024:
  - Paper files are .txt  (plain-text extraction)
  - Human reviews include a 'Limitations' field → included in review text
  - SEA reviews have extra sections (Soundness, Rating, Paper Decision)
    → only Summary / Weaknesses / Questions are passed to the model

Three modes
-----------
  cfi_only  – Step 1 (flaw extraction) + Step 2 (validation) + CFI metrics
              → outputs to output_cfi_iclr2026/all_papers_results.jsonl
  all       – cfi_only + CPS coverage metrics
  cps_only  – CPS from a saved CFI JSONL (no paper text needed)
              → default cache: output_cfi_iclr2026/all_papers_results.jsonl

Example commands
----------------
  python main_cfi_iclr2026.py --mode cfi_only
  python main_cfi_iclr2026.py --mode all
  python main_cfi_iclr2026.py --mode cps_only
  python main_cfi_iclr2026.py --mode cfi_only --model gemini-2.5-flash
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Literal

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
    extract_sea_relevant_sections,
    format_human_review_text_extended,
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
# Path configuration
# ---------------------------------------------------------------------------
from paths_config import ICLR2026_DATA
HUMAN_FOLDER       = os.path.join(ICLR2026_DATA, "human_reviews")
SEA_FOLDER         = os.path.join(ICLR2026_DATA, "sea_iclr2026")
TREE_FOLDER        = os.path.join(ICLR2026_DATA, "tree_iclr2026_2")
REVIEWER2_FOLDER   = os.path.join(ICLR2026_DATA, "reviewer2_iclr2026")
DEEPREVIEW_FOLDER  = os.path.join(ICLR2026_DATA, "deepreview_iclr2026")
CYCLEREVIEW_FOLDER = os.path.join(ICLR2026_DATA, "cyclereview_iclr2026")
PAPERS_FOLDER      = os.path.join(ICLR2026_DATA, "papers")
PAPER_IDS_FILE     = os.path.join(ICLR2026_DATA, "paper_ids_200_iclr2026.txt")
OUTPUT_DIR         = os.path.join(_HERE, "output_cfi_iclr2026_2")

VALID_MODES     = {"all", "cfi_only", "cps_only"}
VALID_PROVIDERS = {"gemini", "openai", "azure"}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run flaw-identification pipeline on ICLR 2026 data.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--mode", choices=sorted(VALID_MODES), default="cfi_only",
        help="Pipeline mode (default: cfi_only)")
    parser.add_argument("--provider", choices=sorted(VALID_PROVIDERS), default=None,
        help="LLM provider (default: gemini)")
    parser.add_argument("--model", default=None,
        help="Override model name. Gemini default: gemini-2.5-flash-lite")
    parser.add_argument("--cfi-cache", default=None,
        help=f"CFI cache JSONL for cps_only mode.\nDefault: {OUTPUT_DIR}/all_papers_results.jsonl")
    parser.add_argument("--output-dir", default=None,
        help=f"Override output directory (default: {OUTPUT_DIR})")
    parser.add_argument(
        "--llm-type",
        choices=["sea", "tree", "reviewer2", "deepreview", "cyclereview"],
        default="sea",
        help=(
            "LLM reviewer source:\n"
            "  sea         – SEA reviews from sea_iclr2026/ (default)\n"
            "  tree        – Tree reviews from tree_iclr2026/\n"
            "  reviewer2   – Reviewer2 reviews from reviewer2_iclr2026/\n"
            "  deepreview  – DeepReview JSON from deepreview_iclr2026/\n"
            "  cyclereview – CycleReview JSON from cyclereview_iclr2026/"
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------

def _build_pipeline(provider: str | None, model: str | None) -> ReviewEvaluatorPipeline:
    return ReviewEvaluatorPipeline(provider=provider, model=model)


# ---------------------------------------------------------------------------
# Core processing (all / cfi_only)
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

    # Load paper content (tries .grobid.txt first, then .txt)
    paper_content = load_paper_content(paper_id, PAPERS_FOLDER)
    if not paper_content:
        print(f"  [SKIP] No paper file for {paper_id}")
        return None

    # Load human reviews — include Limitations field for ICLR 2026
    human_data = load_human_meta_json(h_path)
    human_reviews_dict: dict[str, str] = {}
    human_list = human_data.get("reviews", []) if isinstance(human_data, dict) else human_data
    for idx, review_obj in enumerate(human_list):
        text = format_human_review_text_extended(review_obj)
        if text.strip():
            human_reviews_dict[f"Human_{idx + 1}"] = text
    human_ids = list(human_reviews_dict.keys())

    # Load LLM review
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
        raw_sea = load_llm_txt(llm_path)
        llm_review_text = extract_sea_relevant_sections(raw_sea)

    # Step 1
    print(">> Step 1: Extracting micro-flaws...")
    try:
        step1_flaws = pipeline.step1_atomize_and_group(human_reviews_dict, llm_review_text)
    except Exception as exc:
        print(f"[ERROR] {paper_id}: Step 1 failed: {exc}")
        return None

    # Step 2
    print(">> Step 2: Validating flaws against paper...")
    try:
        step2_evals = pipeline.step2_judge_flaws(paper_content, step1_flaws)
    except Exception as exc:
        print(f"[ERROR] {paper_id}: Step 2 failed: {exc}")
        return None

    # Metrics
    print(">> Step 3: Computing metrics...")
    report: dict = {}
    cfi_calc = DecoupledMetricsCalculator(
        micro_flaws_json=step1_flaws,
        evaluations_json=step2_evals,
        total_reviewers_count=len(human_ids) + 1,
    )
    report["cfi"] = cfi_calc.generate_final_report(human_ids)
    if mode == "all":
        report["cps"] = _compute_cps(pipeline, step1_flaws, step2_evals,
                                     human_reviews_dict, llm_review_text)
    report["mode"] = mode

    return {
        "paper_id":       paper_id,
        "micro_flaws":    step1_flaws,
        "evaluations":    step2_evals,
        "metrics_report": report,
    }


# ---------------------------------------------------------------------------
# CPS-only (from cached CFI JSONL)
# ---------------------------------------------------------------------------

def process_cps_from_cache(cached_record: dict,
                           pipeline: ReviewEvaluatorPipeline,
                           llm_type: str = "sea") -> dict | None:
    paper_id    = cached_record["paper_id"]
    step1_flaws = cached_record["micro_flaws"]
    step2_evals = cached_record["evaluations"]
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

    if not os.path.exists(h_path) or not os.path.exists(llm_path):
        print(f"  [WARNING] Missing review files for {paper_id}")
        return None

    human_data = load_human_meta_json(h_path)
    human_reviews_dict: dict[str, str] = {}
    human_list = human_data.get("reviews", []) if isinstance(human_data, dict) else human_data
    for idx, review_obj in enumerate(human_list):
        text = format_human_review_text_extended(review_obj)
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
        raw_sea = load_llm_txt(llm_path)
        llm_review_text = extract_sea_relevant_sections(raw_sea)

    print(">> Computing CPS (cached Step 1+2)...")
    cps_report = _compute_cps(pipeline, step1_flaws, step2_evals,
                               human_reviews_dict, llm_review_text)
    report: dict = {"cps": cps_report, "mode": "cps_only"}
    if "cfi" in cached_record.get("metrics_report", {}):
        report["cfi"] = cached_record["metrics_report"]["cfi"]

    return {"paper_id": paper_id, "micro_flaws": step1_flaws,
            "evaluations": step2_evals, "metrics_report": report}


# ---------------------------------------------------------------------------
# Shared CPS computation
# ---------------------------------------------------------------------------

def _compute_cps(pipeline, step1_flaws, step2_evals,
                 human_reviews_dict, llm_review_text) -> dict:
    issue_bank = build_cps_issue_bank(step1_flaws, step2_evals)
    rows: list[dict] = []
    for rid, rtext in human_reviews_dict.items():
        rows.append(process_single_review_for_cps(
            pipeline=pipeline, review_text=rtext, reviewer_id=rid,
            issue_bank=issue_bank)["metrics"])
    rows.append(process_single_review_for_cps(
        pipeline=pipeline, review_text=llm_review_text,
        reviewer_id="LLM_Reviewer", issue_bank=issue_bank)["metrics"])

    raw = [r["Raw_CPS"] for r in rows]
    mn, mx = min(raw), max(raw)
    rng = mx - mn
    for r in rows:
        r["CPS_norm"] = round((r["Raw_CPS"] - mn) / rng, 4) if rng > 0 else (1.0 if r["Raw_CPS"] > 0 else 0.0)
    rows.sort(key=lambda r: (r["nCPS"], r["Raw_CPS"]), reverse=True)
    return {"Severity_Map": SEVERITY_MAP, "Canonical_Issue_Bank": issue_bank,
            "Reviewer_Rankings": rows}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_cfi_cache(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if "paper_id" in d and "micro_flaws" in d and "evaluations" in d:
                    records.append(d)
            except json.JSONDecodeError:
                continue
    return records


def _load_target_ids() -> set[str]:
    if not os.path.exists(PAPER_IDS_FILE):
        raise FileNotFoundError(f"Paper-IDs file not found: {PAPER_IDS_FILE}")
    ids: set[str] = set()
    with open(PAPER_IDS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            pid = line.strip()
            if pid and not pid.startswith("#"):
                ids.add(pid)
    if not ids:
        raise ValueError(f"No IDs in {PAPER_IDS_FILE}")
    return ids


def _load_done_ids(jsonl_path: str) -> set[str]:
    done: set[str] = set()
    if not os.path.exists(jsonl_path):
        return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if "paper_id" in d:
                    done.add(d["paper_id"])
            except json.JSONDecodeError:
                continue
    return done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    mode = args.mode
    llm_type = args.llm_type

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

    try:
        target_ids = _load_target_ids()
    except (FileNotFoundError, ValueError) as exc:
        print(f"[FATAL] {exc}"); return
    print(f"[INFO] {len(target_ids)} target IDs from {PAPER_IDS_FILE}")

    print(f"\n[INFO] Initialising pipeline (provider={args.provider})...")
    try:
        pipeline = _build_pipeline(args.provider, args.model)
    except (AzureOpenAIConfigError, RuntimeError) as exc:
        print(f"[FATAL] {exc}"); return
    print("[INFO] Pipeline ready!\n")

    # ── cps_only ──────────────────────────────────────────────────────────
    if mode == "cps_only":
        cache_path = args.cfi_cache or os.path.join(output_dir, "all_papers_results.jsonl")
        if not os.path.exists(cache_path):
            print(f"[FATAL] CFI cache not found: {cache_path}")
            print("  Run --mode cfi_only first."); return
        cached = [r for r in _load_cfi_cache(cache_path) if r["paper_id"] in target_ids]
        out_path = os.path.join(output_dir, "cps_results.jsonl")
        done = _load_done_ids(out_path)
        print(f"[INFO] {len(cached)} cached papers, {len(done)} already done")
        for rec in cached:
            pid = rec["paper_id"]
            if pid in done:
                print(f"[SKIP] {pid}"); continue
            try:
                result = process_cps_from_cache(rec, pipeline, llm_type=llm_type)
                if result:
                    with open(out_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    done.add(pid)
            except Exception as exc:
                print(f"[ERROR] CPS {pid}: {exc}")
        print(f"\n[OK] CPS done! → {out_path}")
        return

    # ── cfi_only / all ────────────────────────────────────────────────────
    if llm_type == "tree":
        if not os.path.isdir(TREE_FOLDER):
            print(f"[FATAL] Tree review folder not found: {TREE_FOLDER}"); return
        paper_pairs = get_paper_pairs_tree(HUMAN_FOLDER, TREE_FOLDER)
    elif llm_type == "reviewer2":
        if not os.path.isdir(REVIEWER2_FOLDER):
            print(f"[FATAL] Reviewer2 folder not found: {REVIEWER2_FOLDER}"); return
        paper_pairs = get_paper_pairs_reviewer2(HUMAN_FOLDER, REVIEWER2_FOLDER)
    elif llm_type == "deepreview":
        if not os.path.isdir(DEEPREVIEW_FOLDER):
            print(f"[FATAL] DeepReview folder not found: {DEEPREVIEW_FOLDER}"); return
        paper_pairs = get_paper_pairs_deepreview(HUMAN_FOLDER, DEEPREVIEW_FOLDER)
    elif llm_type == "cyclereview":
        if not os.path.isdir(CYCLEREVIEW_FOLDER):
            print(f"[FATAL] CycleReview folder not found: {CYCLEREVIEW_FOLDER}"); return
        paper_pairs = get_paper_pairs_cyclereview(HUMAN_FOLDER, CYCLEREVIEW_FOLDER)
    else:
        paper_pairs = get_paper_pairs(HUMAN_FOLDER, SEA_FOLDER)

    paper_pairs = [(pid, h, llm) for pid, h, llm in paper_pairs if pid in target_ids]
    print(f"[INFO] {len(paper_pairs)} papers matched target IDs.")

    out_path = os.path.join(output_dir, "all_papers_results.jsonl")
    done = _load_done_ids(out_path)
    remaining = [p for p in paper_pairs if p[0] not in done]
    total = len(paper_pairs)
    print(f"[INFO] {len(done)} already done — {len(remaining)} remaining\n")

    for i, (paper_id, h_path, llm_path) in enumerate(remaining, start=1):
        pct = (len(done) / total * 100) if total else 0
        print(f"=================================================================")
        print(f"  [{i}/{len(remaining)}] Paper: {paper_id}  (progress: {len(done)+1}/{total}, {pct:.1f}%)")
        print(f"=================================================================")
        try:
            result = process_single_paper(paper_id, h_path, llm_path, pipeline, mode, llm_type=llm_type)
            if result:
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                done.add(paper_id)
                print(f"  [SAVED] {paper_id}")
        except Exception as exc:
            print(f"[ERROR] {paper_id}: {exc}")

    print(f"\n[OK] Pipeline done! → {out_path}")
    print(f"     Total: {len(done)}/{total}")


if __name__ == "__main__":
    main()

