#!/usr/bin/env python3
"""
Aggregation policy sensitivity analysis (ablation).

Recomputes final_scores from existing pair_results in task3_result.json
under all 4 aggregation policies: max, mean, weighted, top3_relevance.
Reports NS, SR, SSR for each policy.

Usage:
  python scripts/ablation_aggregation.py --dir output/iclr2024
  python scripts/ablation_aggregation.py --dir output/iclr2024 --output output/ablation_aggregation.json
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from task3_judge import _aggregate_scores

POLICIES = ["max", "mean", "weighted", "top3_relevance"]

log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Metric helpers
# ------------------------------------------------------------------

def _normalize_score(mean_score: float) -> float:
    """NS = (mean_score + 2) / 4, clipped to [0, 1]."""
    return max(0.0, min(1.0, (mean_score + 2.0) / 4.0))


def _support_rate(scores: List[float], threshold: float = 1.0) -> float:
    """Fraction of scores >= threshold."""
    if not scores:
        return 0.0
    return sum(1 for s in scores if s >= threshold) / len(scores)


# ------------------------------------------------------------------
# Core recomputation
# ------------------------------------------------------------------

def recompute_aggregation(task3_data: dict, policy: str) -> dict:
    """Recompute final_scores for every review sentence under *policy*.

    Returns dict with: final_scores, mean_score, max_score, ns, sr, ssr.
    """
    pair_results: List[Dict[str, Any]] = task3_data.get("pair_results", [])
    related_works: List[Dict[str, Any]] = task3_data.get("related_works", [])

    # Build relevance lookup: related_paper_id -> relevance_score
    relevance_map: Dict[str, Optional[float]] = {}
    for rw in related_works:
        rid = rw.get("related_paper_id")
        if rid is not None:
            relevance_map[rid] = rw.get("relevance_score")

    # Group pair_results by review_sentence_id
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for pr in pair_results:
        sid = pr.get("review_sentence_id")
        if sid is not None:
            groups[sid].append(pr)

    final_scores: List[float] = []
    for sid, results in groups.items():
        scores = [
            r["score"]
            for r in results
            if isinstance(r.get("score"), (int, float))
        ]
        if not scores:
            continue

        agg_kwargs: Dict[str, Any] = {"policy": policy}
        if policy == "top3_relevance":
            agg_kwargs["relevance_scores"] = [
                relevance_map.get(r.get("related_paper_id"))
                for r in results
                if isinstance(r.get("score"), (int, float))
            ]

        final_score = _aggregate_scores(scores, **agg_kwargs)
        final_scores.append(final_score)

    if not final_scores:
        return {
            "final_scores": [],
            "mean_score": 0.0,
            "max_score": 0.0,
            "ns": 0.0,
            "sr": 0.0,
            "ssr": 0.0,
        }

    mean_score = sum(final_scores) / len(final_scores)
    max_score = max(final_scores)
    return {
        "final_scores": final_scores,
        "mean_score": round(mean_score, 4),
        "max_score": round(max_score, 4),
        "ns": round(_normalize_score(mean_score), 4),
        "sr": round(_support_rate(final_scores, threshold=1.0), 4),
        "ssr": round(_support_rate(final_scores, threshold=2.0), 4),
    }


# ------------------------------------------------------------------
# Ablation runner
# ------------------------------------------------------------------

def run_ablation(data_dir: Path, max_papers: Optional[int] = None) -> dict:
    """Iterate over paper dirs, recompute under all policies, aggregate."""
    paper_dirs = sorted(
        p for p in data_dir.iterdir()
        if p.is_dir() and (p / "task3_result.json").exists()
    )
    if max_papers is not None:
        paper_dirs = paper_dirs[:max_papers]

    per_paper: List[Dict[str, Any]] = []
    # policy -> list of per-paper metric values
    policy_ns: Dict[str, List[float]] = defaultdict(list)
    policy_sr: Dict[str, List[float]] = defaultdict(list)
    policy_ssr: Dict[str, List[float]] = defaultdict(list)
    policy_mean: Dict[str, List[float]] = defaultdict(list)

    for pd in paper_dirs:
        task3_path = pd / "task3_result.json"
        try:
            task3_data = json.loads(task3_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Skipping %s: %s", pd.name, exc)
            continue

        paper_entry: Dict[str, Any] = {"paper_id": pd.name, "policies": {}}
        for policy in POLICIES:
            result = recompute_aggregation(task3_data, policy)
            paper_entry["policies"][policy] = {
                k: v for k, v in result.items() if k != "final_scores"
            }
            policy_ns[policy].append(result["ns"])
            policy_sr[policy].append(result["sr"])
            policy_ssr[policy].append(result["ssr"])
            policy_mean[policy].append(result["mean_score"])

        per_paper.append(paper_entry)

    n_papers = len(per_paper)
    policies_summary: Dict[str, Dict[str, float]] = {}
    for policy in POLICIES:
        if n_papers == 0:
            policies_summary[policy] = {
                "avg_ns": 0.0, "avg_sr": 0.0, "avg_ssr": 0.0, "avg_mean_score": 0.0,
            }
        else:
            policies_summary[policy] = {
                "avg_ns": round(sum(policy_ns[policy]) / n_papers, 4),
                "avg_sr": round(sum(policy_sr[policy]) / n_papers, 4),
                "avg_ssr": round(sum(policy_ssr[policy]) / n_papers, 4),
                "avg_mean_score": round(sum(policy_mean[policy]) / n_papers, 4),
            }

    return {
        "n_papers": n_papers,
        "policies": policies_summary,
        "per_paper": per_paper,
    }


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregation policy sensitivity analysis (ablation).",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="Pipeline output directory containing paper subdirs with task3_result.json",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("ablation_aggregation.json"),
        help="Output JSON path (default: ablation_aggregation.json)",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Limit number of papers (for testing)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not args.dir.is_dir():
        log.error("Directory does not exist: %s", args.dir)
        sys.exit(1)

    summary = run_ablation(args.dir, max_papers=args.max_papers)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("✓ Ablation results saved to: %s", args.output)

    # Print summary table
    print(f"\n{'Policy':<18} {'avg_NS':>8} {'avg_SR':>8} {'avg_SSR':>8} {'avg_mean':>10}")
    print("-" * 56)
    for policy in POLICIES:
        s = summary["policies"][policy]
        print(
            f"{policy:<18} {s['avg_ns']:>8.4f} {s['avg_sr']:>8.4f} "
            f"{s['avg_ssr']:>8.4f} {s['avg_mean_score']:>10.4f}"
        )
    print(f"\nTotal papers: {summary['n_papers']}")


if __name__ == "__main__":
    main()
