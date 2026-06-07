"""
Multi-Dimensional Constructiveness Evaluation (MDCE) — main entry point.

Two modes:
  evaluate  — Run LLM-as-Judge to atomize reviews and score constructiveness (D1–D5)
  analyze   — Run statistical analysis (Wilcoxon, Cliff's delta, bootstrap CI) from cached results
"""

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_FI = os.path.normpath(os.path.join(_HERE, "..", "flaw_identification"))
sys.path.insert(0, _FI)

from dotenv import find_dotenv, load_dotenv

_env_paths = [
    os.path.join(_HERE, ".env"),
    os.path.join(_FI, ".env"),
]
for p in _env_paths:
    if os.path.exists(p):
        load_dotenv(p, override=False)

from src.evaluator import ConstructivenessEvaluator
from src.metrics import compute_paper_comparison, compute_review_metrics
from src.utils import (
    format_human_review_full,
    get_paper_pairs,
    load_human_meta_json,
    load_llm_txt,
    load_paper_grobid,
    load_paper_metadata,
)

# ── Path configuration ───────────────────────────────────────────────────
_DATA = os.path.normpath(os.path.join(_HERE, "..", "data"))

HUMAN_FOLDER = os.path.join(_DATA, "Human_and_meta_reviews")
SEA_FOLDER = os.path.join(_DATA, "SEA_reviews")
PAPERS_FOLDER = os.path.join(_DATA, "grobid_fulltext")
PAPER_IDS_PATH = os.path.join(_DATA, "data_subset", "paper_ids.txt")
OUTPUT_DIR = os.path.join(_HERE, "output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Dimensional Constructiveness Evaluation pipeline.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "azure"],
        default="gemini",
        help=(
            "LLM provider for constructiveness scoring.\n"
            "  gemini          — Gemini endpoint (default)\n"
            "  azure           — Azure OpenAI"
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "For gemini, defaults to GEMINI_MODEL or GOOGLE_MODEL from environment."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["evaluate", "analyze"],
        default="evaluate",
        help=(
            "Pipeline mode:\n"
            "  evaluate  — LLM scoring of review constructiveness\n"
            "  analyze   — statistical analysis from cached results"
        ),
    )
    parser.add_argument(
        "--with-paper",
        action="store_true",
        default=False,
        help="Include paper text as context for the LLM judge (more accurate D2/D3, higher cost).",
    )
    parser.add_argument(
        "--cache",
        default=None,
        help="Path to cached results JSONL (required for --mode=analyze).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N papers (for testing).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Output directory (default: {OUTPUT_DIR}).",
    )
    return parser.parse_args()


def load_target_paper_ids(filepath: str) -> set[str]:
    """Load the paper IDs that should be processed in evaluate mode."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Paper ID subset file not found: {filepath}")

    paper_ids: set[str] = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            paper_id = line.strip()
            if paper_id:
                paper_ids.add(paper_id)
    return paper_ids


# ── Evaluate mode ─────────────────────────────────────────────────────────

def process_single_paper(
    paper_id: str,
    h_path: str,
    llm_path: str,
    evaluator: ConstructivenessEvaluator,
    with_paper: bool = False,
) -> dict | None:
    print(f"\n{'='*60}")
    print(f"  Paper: {paper_id}")
    print(f"{'='*60}")

    human_data = load_human_meta_json(h_path)
    metadata = load_paper_metadata(human_data)

    paper_text = None
    if with_paper:
        paper_text = load_paper_grobid(paper_id, PAPERS_FOLDER)

    human_reviews: dict[str, str] = {}
    human_list = human_data.get("reviews", []) if isinstance(human_data, dict) else human_data
    for idx, review_obj in enumerate(human_list):
        formatted = format_human_review_full(review_obj)
        if formatted.strip():
            human_reviews[f"Human_{idx + 1}"] = formatted

    llm_review_text = load_llm_txt(llm_path)

    all_reviewer_results: list[dict] = []
    human_metrics_list: list[dict] = []

    for reviewer_id, review_text in human_reviews.items():
        scored = evaluator.score_review(review_text, reviewer_id, paper_text)
        status = scored.get("status", "unknown")
        metrics = compute_review_metrics(scored["atomic_comments"])
        all_reviewer_results.append({
            "reviewer_id": reviewer_id,
            "status": status,
            "atomic_comments": scored["atomic_comments"],
            "metrics": metrics,
        })
        human_metrics_list.append({"status": status, **metrics})

    llm_scored = evaluator.score_review(llm_review_text, "LLM_Reviewer", paper_text)
    llm_status = llm_scored.get("status", "unknown")
    llm_metrics = compute_review_metrics(llm_scored["atomic_comments"])
    all_reviewer_results.append({
        "reviewer_id": "LLM_Reviewer",
        "status": llm_status,
        "atomic_comments": llm_scored["atomic_comments"],
        "metrics": llm_metrics,
    })

    valid_human_metrics = [
        m for m in human_metrics_list if m.get("status") == "success"
    ]
    valid_llm_metrics = llm_metrics if llm_status == "success" else None

    comparison = compute_paper_comparison(valid_human_metrics, valid_llm_metrics)

    _print_paper_summary(paper_id, comparison)

    return {
        "paper_id": paper_id,
        "metadata": metadata,
        "reviewers": all_reviewer_results,
        "comparison": comparison,
    }


def _print_paper_summary(paper_id: str, comparison: dict) -> None:
    if not comparison.get("comparison_valid"):
        n_valid_h = comparison.get("n_human_reviewers_valid", 0)
        llm_ok = comparison.get("llm_valid", False)
        print(
            f"  [SKIP] Comparison invalid — "
            f"valid humans: {n_valid_h}, LLM valid: {llm_ok}"
        )
        return

    h_mcs = comparison.get("human_avg_MCS", 0) or 0
    l_mcs = comparison.get("llm_MCS", 0) or 0
    delta = comparison.get("delta_MCS", 0) or 0
    winner = "LLM" if delta > 0 else "Human" if delta < 0 else "Tie"

    n_h = comparison.get("n_human_reviewers_valid", 0)
    print(
        f"  Humans ({n_h} valid): MCS={h_mcs:.4f} | LLM: MCS={l_mcs:.4f} | "
        f"Delta={delta:+.4f} ({winner})"
    )

    dim_labels = ["D1:Act", "D2:Spec", "D3:Just", "D4:Sol", "D5:Tone"]
    dim_keys = [
        "D1_actionability_mean", "D2_specificity_mean",
        "D3_justification_mean", "D4_solution_mean", "D5_tone_mean",
    ]
    parts = []
    for label, key in zip(dim_labels, dim_keys):
        d = comparison.get(f"delta_{key}")
        parts.append(f"{label}={d:+.2f}" if d is not None else f"{label}=N/A")
    print(f"  Deltas: {' | '.join(parts)}")


def run_evaluate(args: argparse.Namespace) -> None:
    output_dir = args.output_dir or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    try:
        target_paper_ids = load_target_paper_ids(PAPER_IDS_PATH)
    except FileNotFoundError as exc:
        print(f"[FATAL] {exc}")
        return

    all_paper_pairs = get_paper_pairs(HUMAN_FOLDER, SEA_FOLDER)
    paper_pairs = [
        (paper_id, h_path, llm_path)
        for paper_id, h_path, llm_path in all_paper_pairs
        if paper_id in target_paper_ids
    ]

    if args.limit:
        paper_pairs = paper_pairs[: args.limit]
    print(
        f"[INFO] Loaded {len(target_paper_ids)} target paper IDs from: {PAPER_IDS_PATH}"
    )
    print(
        f"[INFO] Found {len(paper_pairs)} matching papers to process "
        f"(from {len(all_paper_pairs)} available human/LLM pairs)."
    )

    print(f"[INFO] Initialising ConstructivenessEvaluator (provider={args.provider})...")
    try:
        evaluator = ConstructivenessEvaluator(
            provider=args.provider,
            api_key=(
                os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
                if args.provider == "gemini"
                else os.getenv("AZURE_OPENAI_API_KEY", "")
            ),
            model=args.model,
        )
    except Exception as exc:
        print(f"[FATAL] Evaluator init failed: {exc}")
        return
    print("[INFO] Evaluator ready!")

    jsonl_path = os.path.join(output_dir, "all_results.jsonl")

    processed_ids: set[str] = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "paper_id" in data:
                        processed_ids.add(data["paper_id"])
                except json.JSONDecodeError:
                    continue
        print(f"[INFO] Resuming: {len(processed_ids)} papers already processed.")

    success_count = 0
    error_count = 0

    for paper_id, h_path, llm_path in paper_pairs:
        if paper_id in processed_ids:
            continue

        try:
            result = process_single_paper(
                paper_id, h_path, llm_path, evaluator, args.with_paper,
            )
            if result:
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                processed_ids.add(paper_id)
                success_count += 1
        except Exception as exc:
            print(f"[ERROR] {paper_id}: {type(exc).__name__}: {exc}")
            error_count += 1
            continue

    print(f"\n{'='*60}")
    print(f"  DONE — {success_count} succeeded, {error_count} failed")
    print(f"  Results: {jsonl_path}")
    print(f"{'='*60}")


# ── Analyze mode ──────────────────────────────────────────────────────────

def run_analyze(args: argparse.Namespace) -> None:
    from src.statistical import run_full_analysis, run_subgroup_analysis

    cache_path = args.cache
    if not cache_path:
        default_path = os.path.join(OUTPUT_DIR, "all_results.jsonl")
        if os.path.exists(default_path):
            cache_path = default_path
        else:
            print("[FATAL] --cache path required for analyze mode.")
            return

    if not os.path.exists(cache_path):
        print(f"[FATAL] Cache file not found: {cache_path}")
        return

    records = []
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"[INFO] Loaded {len(records)} paper records from: {cache_path}")

    # Primary analysis
    print("\n" + "=" * 60)
    print("  PRIMARY ANALYSIS: Human vs. LLM Constructiveness")
    print("=" * 60)

    analysis = run_full_analysis(records)
    _print_analysis_table(analysis)

    # Subgroup analysis
    print("\n" + "=" * 60)
    print("  SUBGROUP ANALYSIS")
    print("=" * 60)

    subgroup = run_subgroup_analysis(records)
    _print_subgroup_summary(subgroup)

    # Save results
    output_dir = args.output_dir or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    analysis_path = os.path.join(output_dir, "statistical_analysis.json")
    full_output = {
        "primary_analysis": analysis,
        "subgroup_analysis": subgroup,
    }
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Full analysis saved to: {analysis_path}")


def _print_analysis_table(analysis: dict) -> None:
    print(f"\nN = {analysis['n_papers']} papers\n")

    header = f"{'Metric':<28} {'Human':>8} {'LLM':>8} {'Delta':>8} {'p-val':>10} {'adj-p':>10} {'Cliff-d':>8} {'Effect':>10} {'Sig?':>5}"
    print(header)
    print("-" * len(header))

    for key in [
        "MCS", "AR", "SD", "CD",
        "D1_actionability_mean", "D2_specificity_mean",
        "D3_justification_mean", "D4_solution_mean", "D5_tone_mean",
    ]:
        data = analysis["per_metric"].get(key, {})
        if data.get("skipped"):
            print(f"{key:<28} {'SKIPPED':>8}")
            continue

        h = data.get("human_mean", 0)
        l = data.get("llm_mean", 0)
        d = data.get("mean_diff", 0)
        p = data.get("wilcoxon_p", 1)
        ap = data.get("adjusted_p", 1)
        cd = data.get("cliffs_delta", 0)
        mag = data.get("cliffs_delta_magnitude", "?")
        sig = "YES" if data.get("significant_after_correction") else "no"

        print(f"{key:<28} {h:>8.4f} {l:>8.4f} {d:>+8.4f} {p:>10.6f} {ap:>10.6f} {cd:>+8.4f} {mag:>10} {sig:>5}")

    summary = analysis.get("summary", {})
    print(f"\nLLM wins:   {', '.join(summary.get('llm_significantly_better', [])) or 'none'}")
    print(f"Human wins: {', '.join(summary.get('human_significantly_better', [])) or 'none'}")
    print(f"No diff:    {', '.join(summary.get('no_significant_difference', [])) or 'none'}")


def _print_subgroup_summary(subgroup: dict) -> None:
    if "by_decision" in subgroup:
        for group_name in ("accept", "reject"):
            data = subgroup["by_decision"].get(group_name, {})
            n = data.get("n_papers", 0)
            mcs_data = data.get("MCS", {})
            if not mcs_data or mcs_data.get("skipped"):
                continue
            print(f"\n  {group_name.upper()} papers (n={n}): "
                  f"Human MCS={mcs_data.get('human_mean', 0):.4f}, "
                  f"LLM MCS={mcs_data.get('llm_mean', 0):.4f}, "
                  f"Cliff's d={mcs_data.get('cliffs_delta', 0):+.4f} ({mcs_data.get('magnitude', '?')})")

    if "by_quality_tier" in subgroup:
        for tier_name in ("top_25", "middle_50", "bottom_25"):
            data = subgroup["by_quality_tier"].get(tier_name)
            if not data:
                continue
            n = data.get("n_papers", 0)
            mcs_data = data.get("MCS", {})
            if not mcs_data or mcs_data.get("skipped"):
                continue
            print(f"  {tier_name.upper()} (n={n}): "
                  f"Human MCS={mcs_data.get('human_mean', 0):.4f}, "
                  f"LLM MCS={mcs_data.get('llm_mean', 0):.4f}, "
                  f"Cliff's d={mcs_data.get('cliffs_delta', 0):+.4f} ({mcs_data.get('magnitude', '?')})")


# ── Entry point ───────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.mode == "evaluate":
        run_evaluate(args)
    elif args.mode == "analyze":
        run_analyze(args)
    else:
        print(f"[FATAL] Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
