#!/usr/bin/env python3
"""
Compare novelty pipeline results between two LLMs:
- Mimo 2.5 Pro (subset_50): 50 papers/conference × 5 conferences = 250 papers
- Gemini 2.5 Flash Lite (full_conf_results): 200 papers/conference × 5 conferences = 1000 papers

Comparison is on the overlapping 250 papers (50 per conference), across 6 methods.
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import csv

# Paths
BASE = Path(os.getenv("NOVELTY_OUTPUT_ROOT", str(Path(__file__).resolve().parents[1] / "output")))
MIMO_BASE = BASE / "subset_50"
GEMINI_BASE = BASE / "full_conf_results"
OUTPUT_DIR = BASE / "llm_comparison_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Conference mapping (subset_50 uses underscores, full_conf_results doesn't)
CONF_MAP = {
    "ICLR_2024": "ICLR2024",
    "ICLR_2025": "ICLR2025",
    "ICLR_2026": "ICLR2026",
    "ICML_2025": "ICML2025",
    "NeurIPS_2025": "NeurIPS2025",
}

METHODS = ["human", "sea", "deepreview", "reviewer2", "cyclereview", "tree"]


def load_task1(path):
    """Extract Phase 1 (extraction) metrics from task1_result.json."""
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception as e:
        return {"error": str(e)}

    paper = d.get("paper", {})
    review = d.get("review", {})

    claims = paper.get("contributions", [])
    key_terms = paper.get("key_terms", [])
    must_have = paper.get("must_have_entities", [])

    # Review sentences
    review_text = review.get("core_review", "") if isinstance(review, dict) else ""
    review_claims = review.get("claims", []) if isinstance(review, dict) else []

    return {
        "claim_count": len(claims),
        "key_term_count": len(key_terms),
        "must_have_count": len(must_have),
        "has_core_task": bool(paper.get("core_task")),
        "review_claim_count": len(review_claims) if isinstance(review_claims, list) else 0,
        "error": None,
    }


def load_task2(path):
    """Extract Phase 2 (retrieval) metrics from task2_result.json."""
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception as e:
        return {"error": str(e)}

    candidates = d.get("candidate_pool_top30", [])
    stats = d.get("stats", {})

    return {
        "candidate_count": len(candidates),
        "queries_count": len(d.get("queries", [])),
        "error": None,
    }


def load_task3(path):
    """Extract Phase 3 (verification) metrics from task3_result.json."""
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception as e:
        return {"error": str(e)}

    aggregated = d.get("aggregated", [])
    stats = d.get("stats", {})
    errors = d.get("errors", [])
    coverage = stats.get("coverage", {})

    # Extract claim-level data
    scores = []
    labels = []
    for item in aggregated:
        score = item.get("final_score", None)
        if score is not None:
            scores.append(score)
        cls = item.get("classification", {})
        if cls.get("claim", 0) == 1:
            if cls.get("proof", 0) == 1:
                labels.append("SUPPORTED")
            else:
                labels.append("OVERSTATED")

    # Compute paper-level score (mean of claim scores)
    paper_score = sum(scores) / len(scores) if scores else None

    # Label distribution
    label_dist = {}
    for l in labels:
        label_dist[l] = label_dist.get(l, 0) + 1

    return {
        "paper_score": paper_score,
        "claim_count_phase3": len(aggregated),
        "total_score": sum(scores) if scores else 0,
        "review_sentences": stats.get("review_sentences", 0),
        "related_works": stats.get("related_works", 0),
        "pairs_attempted": stats.get("pairs_attempted", 0),
        "pairs_completed": stats.get("pairs_completed", 0),
        "pairs_failed": stats.get("pairs_failed", 0),
        "coverage_rate": coverage.get("claim_success_coverage_rate", 0),
        "evidence_coverage_rate": coverage.get("evidence_coverage_rate", 0),
        "decisive_coverage_rate": coverage.get("decisive_coverage_rate", 0),
        "avg_evidence_per_claim": coverage.get("avg_evidence_per_claim", 0),
        "label_dist": label_dist,
        "labels": labels,
        "scores": scores,
        "error_count": len(errors),
        "error": None,
    }


def collect_paper_ids(conference_s50):
    """Get overlapping paper IDs between subset_50 and full_conf_results for a conference."""
    conference_fc = CONF_MAP[conference_s50]
    s50_dir = MIMO_BASE / "human" / conference_s50
    fc_dir = GEMINI_BASE / "human" / conference_fc

    s50_ids = set(p.name for p in s50_dir.iterdir() if p.is_dir()) if s50_dir.exists() else set()
    fc_ids = set(p.name for p in fc_dir.iterdir() if p.is_dir()) if fc_dir.exists() else set()

    return sorted(s50_ids & fc_ids)


def main():
    print("=" * 80)
    print("LLM Comparison: Mimo 2.5 Pro vs Gemini 2.5 Flash Lite")
    print("Comparing on overlapping 50 papers per conference × 5 conferences")
    print("=" * 80)

    all_records = []
    summary_by_method = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    summary_by_conf = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    summary_by_llm = defaultdict(lambda: defaultdict(list))

    for conf_s50, conf_fc in CONF_MAP.items():
        paper_ids = collect_paper_ids(conf_s50)
        print(f"\n{conf_s50}: {len(paper_ids)} overlapping papers")

        for paper_id in paper_ids:
            for method in METHODS:
                # Mimo paths
                mimo_t1 = MIMO_BASE / method / conf_s50 / paper_id / "task1_result.json"
                mimo_t2 = MIMO_BASE / method / conf_s50 / paper_id / "task2_result.json"
                mimo_t3 = MIMO_BASE / method / conf_s50 / paper_id / "task3_result.json"

                # Gemini paths
                gemini_t1 = GEMINI_BASE / method / conf_fc / paper_id / "task1_result.json"
                gemini_t2 = GEMINI_BASE / method / conf_fc / paper_id / "task2_result.json"
                gemini_t3 = GEMINI_BASE / method / conf_fc / paper_id / "task3_result.json"

                # Load metrics
                m1 = load_task1(mimo_t1) if mimo_t1.exists() else {"error": "missing"}
                m2 = load_task2(mimo_t2) if mimo_t2.exists() else {"error": "missing"}
                m3 = load_task3(mimo_t3) if mimo_t3.exists() else {"error": "missing"}

                g1 = load_task1(gemini_t1) if gemini_t1.exists() else {"error": "missing"}
                g2 = load_task2(gemini_t2) if gemini_t2.exists() else {"error": "missing"}
                g3 = load_task3(gemini_t3) if gemini_t3.exists() else {"error": "missing"}

                record = {
                    "conference": conf_s50,
                    "paper_id": paper_id,
                    "method": method,
                    # Phase 1 - Mimo
                    "mimo_claim_count": m1.get("claim_count"),
                    "mimo_key_terms": m1.get("key_term_count"),
                    "mimo_has_core_task": m1.get("has_core_task"),
                    # Phase 1 - Gemini
                    "gemini_claim_count": g1.get("claim_count"),
                    "gemini_key_terms": g1.get("key_term_count"),
                    "gemini_has_core_task": g1.get("has_core_task"),
                    # Phase 2 - Mimo
                    "mimo_candidates": m2.get("candidate_count"),
                    # Phase 2 - Gemini
                    "gemini_candidates": g2.get("candidate_count"),
                    # Phase 3 - Mimo
                    "mimo_paper_score": m3.get("paper_score"),
                    "mimo_review_sentences": m3.get("review_sentences"),
                    "mimo_related_works": m3.get("related_works"),
                    "mimo_pairs_attempted": m3.get("pairs_attempted"),
                    "mimo_pairs_completed": m3.get("pairs_completed"),
                    "mimo_coverage_rate": m3.get("coverage_rate"),
                    "mimo_evidence_coverage": m3.get("evidence_coverage_rate"),
                    "mimo_decisive_coverage": m3.get("decisive_coverage_rate"),
                    "mimo_avg_evidence": m3.get("avg_evidence_per_claim"),
                    "mimo_claim_count_p3": m3.get("claim_count_phase3"),
                    # Phase 3 - Gemini
                    "gemini_paper_score": g3.get("paper_score"),
                    "gemini_review_sentences": g3.get("review_sentences"),
                    "gemini_related_works": g3.get("related_works"),
                    "gemini_pairs_attempted": g3.get("pairs_attempted"),
                    "gemini_pairs_completed": g3.get("pairs_completed"),
                    "gemini_coverage_rate": g3.get("coverage_rate"),
                    "gemini_evidence_coverage": g3.get("evidence_coverage_rate"),
                    "gemini_decisive_coverage": g3.get("decisive_coverage_rate"),
                    "gemini_avg_evidence": g3.get("avg_evidence_per_claim"),
                    "gemini_claim_count_p3": g3.get("claim_count_phase3"),
                    # Errors
                    "mimo_t1_error": m1.get("error"),
                    "mimo_t2_error": m2.get("error"),
                    "mimo_t3_error": m3.get("error"),
                    "gemini_t1_error": g1.get("error"),
                    "gemini_t2_error": g2.get("error"),
                    "gemini_t3_error": g3.get("error"),
                }
                all_records.append(record)

                # Accumulate for summaries
                for metric in ["claim_count", "paper_score", "coverage_rate",
                               "evidence_coverage", "decisive_coverage", "avg_evidence",
                               "review_sentences", "related_works", "pairs_attempted",
                               "pairs_completed", "candidates"]:
                    mimo_val = record.get(f"mimo_{metric}")
                    gemini_val = record.get(f"gemini_{metric}")
                    if mimo_val is not None:
                        summary_by_method[method]["mimo"][metric].append(mimo_val)
                        summary_by_conf[conf_s50]["mimo"][metric].append(mimo_val)
                        summary_by_llm["mimo"][metric].append(mimo_val)
                    if gemini_val is not None:
                        summary_by_method[method]["gemini"][metric].append(gemini_val)
                        summary_by_conf[conf_s50]["gemini"][metric].append(gemini_val)
                        summary_by_llm["gemini"][metric].append(gemini_val)

    # Save detailed CSV
    csv_path = OUTPUT_DIR / "llm_comparison_detailed.csv"
    if all_records:
        fieldnames = all_records[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)
        print(f"\nSaved {len(all_records)} records to {csv_path}")

    # Save summary JSON
    summary = {
        "by_method": {m: {llm: {metric: {"mean": sum(vals)/len(vals), "std": (sum((x-sum(vals)/len(vals))**2 for x in vals)/len(vals))**0.5, "n": len(vals)} if vals else None
                            for metric, vals in metrics.items()}
                         for llm, metrics in llm_data.items()}
                     for m, llm_data in summary_by_method.items()},
        "by_conference": {c: {llm: {metric: {"mean": sum(vals)/len(vals), "std": (sum((x-sum(vals)/len(vals))**2 for x in vals)/len(vals))**0.5, "n": len(vals)} if vals else None
                                   for metric, vals in metrics.items()}
                             for llm, metrics in llm_data.items()}
                         for c, llm_data in summary_by_conf.items()},
        "overall": {llm: {metric: {"mean": sum(vals)/len(vals), "std": (sum((x-sum(vals)/len(vals))**2 for x in vals)/len(vals))**0.5, "n": len(vals)} if vals else None
                         for metric, vals in metrics.items()}
                   for llm, metrics in summary_by_llm.items()},
        "total_records": len(all_records),
        "overlapping_papers_per_conference": {conf: len(collect_paper_ids(conf)) for conf in CONF_MAP},
    }

    summary_path = OUTPUT_DIR / "llm_comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")

    # Print overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    metrics_to_report = [
        ("claim_count", "Phase 1: Claim Count"),
        ("paper_score", "Phase 3: Paper Score"),
        ("coverage_rate", "Phase 3: Claim Coverage Rate"),
        ("evidence_coverage", "Phase 3: Evidence Coverage Rate"),
        ("decisive_coverage", "Phase 3: Decisive Coverage Rate"),
        ("avg_evidence", "Phase 3: Avg Evidence per Claim"),
        ("review_sentences", "Phase 3: Review Sentences"),
        ("related_works", "Phase 3: Related Works Found"),
        ("pairs_attempted", "Phase 3: Pairs Attempted"),
        ("pairs_completed", "Phase 3: Pairs Completed"),
        ("candidates", "Phase 2: Candidate Count"),
    ]

    print(f"\n{'Metric':<40} {'Mimo (mean±std)':>20} {'Gemini (mean±std)':>20} {'N':>6}")
    print("-" * 90)
    for metric_key, metric_name in metrics_to_report:
        mimo = summary_by_llm["mimo"].get(metric_key, [])
        gemini = summary_by_llm["gemini"].get(metric_key, [])
        if mimo and gemini:
            m_mean = sum(mimo)/len(mimo)
            m_std = (sum((x-m_mean)**2 for x in mimo)/len(mimo))**0.5
            g_mean = sum(gemini)/len(gemini)
            g_std = (sum((x-g_mean)**2 for x in gemini)/len(gemini))**0.5
            print(f"{metric_name:<40} {m_mean:>8.3f} ±{m_std:>6.3f}   {g_mean:>8.3f} ±{g_std:>6.3f}   {min(len(mimo), len(gemini)):>5}")

    # Print by method
    print("\n" + "=" * 80)
    print("BY METHOD: Paper Score (Phase 3)")
    print("=" * 80)
    print(f"\n{'Method':<20} {'Mimo (mean±std)':>20} {'Gemini (mean±std)':>20} {'Diff':>10} {'N':>6}")
    print("-" * 80)
    for method in METHODS:
        mimo_scores = summary_by_method[method]["mimo"].get("paper_score", [])
        gemini_scores = summary_by_method[method]["gemini"].get("paper_score", [])
        if mimo_scores and gemini_scores:
            m_mean = sum(mimo_scores)/len(mimo_scores)
            m_std = (sum((x-m_mean)**2 for x in mimo_scores)/len(mimo_scores))**0.5
            g_mean = sum(gemini_scores)/len(gemini_scores)
            g_std = (sum((x-g_mean)**2 for x in gemini_scores)/len(gemini_scores))**0.5
            diff = g_mean - m_mean
            print(f"{method:<20} {m_mean:>8.3f} ±{m_std:>6.3f}   {g_mean:>8.3f} ±{g_std:>6.3f}   {diff:>+8.3f}   {min(len(mimo_scores), len(gemini_scores)):>5}")

    # Print by conference
    print("\n" + "=" * 80)
    print("BY CONFERENCE: Paper Score (Phase 3)")
    print("=" * 80)
    print(f"\n{'Conference':<20} {'Mimo (mean±std)':>20} {'Gemini (mean±std)':>20} {'Diff':>10} {'N':>6}")
    print("-" * 80)
    for conf in CONF_MAP:
        mimo_scores = summary_by_conf[conf]["mimo"].get("paper_score", [])
        gemini_scores = summary_by_conf[conf]["gemini"].get("paper_score", [])
        if mimo_scores and gemini_scores:
            m_mean = sum(mimo_scores)/len(mimo_scores)
            m_std = (sum((x-m_mean)**2 for x in mimo_scores)/len(mimo_scores))**0.5
            g_mean = sum(gemini_scores)/len(gemini_scores)
            g_std = (sum((x-g_mean)**2 for x in gemini_scores)/len(gemini_scores))**0.5
            diff = g_mean - m_mean
            print(f"{conf:<20} {m_mean:>8.3f} ±{m_std:>6.3f}   {g_mean:>8.3f} ±{g_std:>6.3f}   {diff:>+8.3f}   {min(len(mimo_scores), len(gemini_scores)):>5}")

    return all_records, summary


if __name__ == "__main__":
    records, summary = main()
