#!/usr/bin/env python3
"""
Evaluate novelty assessment pipeline results.

Scans a pipeline output directory for task1/2/3 results and computes
summary statistics. If human_annotations.json files exist, computes
benchmark metrics (Phase 1 P/R/F1, Phase 2 Recall@K/MRR, Phase 3 F1).

Usage:
  # Evaluate a pipeline run (summary stats)
  python scripts/evaluate.py --run-dir output/pipeline_results/human

  # Evaluate with human annotations (benchmark metrics)
  python scripts/evaluate.py --run-dir output/iclr2024/human

  # Compare two runs
  python scripts/evaluate.py --run-dir output/run_a --run-dir2 output/run_b --names "Model-A" "Model-B"

  # Output as CSV
  python scripts/evaluate.py --run-dir output/results --format csv -o report.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def discover_papers(run_dir: Path) -> List[Path]:
    """Find paper directories containing task results."""
    papers = []
    # Direct structure: run_dir/<conference>/<paper_id>/
    for conf_dir in sorted(run_dir.iterdir()):
        if not conf_dir.is_dir() or conf_dir.name.startswith("_"):
            continue
        for paper_dir in sorted(conf_dir.iterdir()):
            if not paper_dir.is_dir():
                continue
            has_task = any(
                (paper_dir / f"task{i}_result.json").exists()
                for i in (1, 2, 3)
            )
            if has_task:
                papers.append(paper_dir)

    # Flat structure: run_dir/<paper_id>/
    if not papers:
        for paper_dir in sorted(run_dir.iterdir()):
            if not paper_dir.is_dir() or paper_dir.name.startswith("_"):
                continue
            has_task = any(
                (paper_dir / f"task{i}_result.json").exists()
                for i in (1, 2, 3)
            )
            if has_task:
                papers.append(paper_dir)

    return papers


# ---------------------------------------------------------------------------
# Summary statistics (no annotations needed)
# ---------------------------------------------------------------------------

def compute_summary(papers: List[Path], run_dir: Path) -> Dict[str, Any]:
    """Compute summary statistics from pipeline output."""
    n_total = len(papers)
    n_task1 = 0
    n_task2 = 0
    n_task3 = 0
    n_full = 0

    claim_counts = []
    candidate_counts = []
    verdict_labels = Counter()
    stance_labels = Counter()
    conferences = Counter()

    for paper_dir in papers:
        conf = paper_dir.parent.name if paper_dir.parent != run_dir else "unknown"
        conferences[conf] += 1

        t1 = load_json(paper_dir / "task1_result.json")
        t2 = load_json(paper_dir / "task2_result.json")
        t3 = load_json(paper_dir / "task3_result.json")

        has_t1 = t1 is not None
        has_t2 = t2 is not None
        has_t3 = t3 is not None

        if has_t1:
            n_task1 += 1
            paper = t1.get("paper", {})
            review = t1.get("review", {})
            contributions = paper.get("contributions", [])
            claim_counts.append(len(contributions))
            claims = review.get("novelty_claims", [])
            for claim in claims:
                if isinstance(claim, dict):
                    stance = claim.get("stance", "unknown")
                    stance_labels[stance] += 1

        if has_t2:
            n_task2 += 1
            pool = t2.get("candidate_pool_top30") or t2.get("candidates") or []
            candidate_counts.append(len(pool))

        if has_t3:
            n_task3 += 1
            # Support both formats: 'verdicts' and 'aggregated'
            verdicts = t3.get("verdicts") or t3.get("aggregated") or []
            for v in verdicts:
                if isinstance(v, dict):
                    label = v.get("verdict") or v.get("label")
                    if not label:
                        # Derive from final_score
                        score = v.get("final_score")
                        if score is not None:
                            label = {2: "SUPPORTED", 1: "OVERSTATED", 0: "AMBIGUOUS", -1: "UNDERSTATED", -2: "UNSUPPORTED"}.get(int(score), "unknown")
                        else:
                            label = "unknown"
                    verdict_labels[label] += 1

        if has_t1 and has_t2 and has_t3:
            n_full += 1

    summary = {
        "total_papers": n_total,
        "task1_success": n_task1,
        "task2_success": n_task2,
        "task3_success": n_task3,
        "full_pipeline_success": n_full,
        "conferences": dict(conferences.most_common()),
    }

    if claim_counts:
        summary["contributions_per_paper"] = {
            "mean": round(sum(claim_counts) / len(claim_counts), 2),
            "min": min(claim_counts),
            "max": max(claim_counts),
            "total": sum(claim_counts),
        }
    if candidate_counts:
        summary["candidates_per_paper"] = {
            "mean": round(sum(candidate_counts) / len(candidate_counts), 2),
            "min": min(candidate_counts),
            "max": max(candidate_counts),
        }
    if stance_labels:
        summary["stance_distribution"] = dict(stance_labels.most_common())
    if verdict_labels:
        summary["verdict_distribution"] = dict(verdict_labels.most_common())

    return summary


def print_summary(summary: Dict[str, Any], name: str = "") -> None:
    """Print a human-readable summary."""
    prefix = f"[{name}] " if name else ""
    print(f"\n{'=' * 60}")
    print(f"{prefix}Pipeline Evaluation Summary")
    print(f"{'=' * 60}")
    print(f"  Papers:          {summary['total_papers']}")
    print(f"  Task 1 (extract): {summary['task1_success']}/{summary['total_papers']}")
    print(f"  Task 2 (retrieve): {summary['task2_success']}/{summary['total_papers']}")
    print(f"  Task 3 (judge):   {summary['task3_success']}/{summary['total_papers']}")
    print(f"  Full pipeline:    {summary['full_pipeline_success']}/{summary['total_papers']}")

    if "conferences" in summary:
        print(f"\n  Conferences:")
        for conf, count in summary["conferences"].items():
            print(f"    {conf}: {count} papers")

    if "contributions_per_paper" in summary:
        cp = summary["contributions_per_paper"]
        print(f"\n  Contributions/paper: mean={cp['mean']}, min={cp['min']}, max={cp['max']}")

    if "candidates_per_paper" in summary:
        cp = summary["candidates_per_paper"]
        print(f"  Candidates/paper:    mean={cp['mean']}, min={cp['min']}, max={cp['max']}")

    if "stance_distribution" in summary:
        print(f"\n  Stance distribution:")
        for stance, count in summary["stance_distribution"].items():
            print(f"    {stance}: {count}")

    if "verdict_distribution" in summary:
        print(f"\n  Verdict distribution:")
        for verdict, count in summary["verdict_distribution"].items():
            print(f"    {verdict}: {count}")

    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Benchmark evaluation (requires annotations)
# ---------------------------------------------------------------------------

def run_benchmark_evaluation(
    run_dir: Path,
    gt_dir: Optional[Path] = None,
    phases: List[int] = [1, 2, 3],
    match_threshold: float = 0.5,
    papers: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Run benchmark evaluation if annotations are available."""
    try:
        # Import from benchmark_evaluation.py
        sys.path.insert(0, str(Path(__file__).parent))
        from benchmark_evaluation import evaluate_paper, aggregate_results

        gt_dir = gt_dir or run_dir

        if papers is None:
            papers = sorted(
                d.name for d in gt_dir.iterdir()
                if d.is_dir() and (d / "human_annotations.json").exists()
            )

        if not papers:
            return None

        per_paper = []
        for pid in papers:
            paper_dir = run_dir / pid
            if not paper_dir.is_dir():
                continue
            result = evaluate_paper(paper_dir, gt_dir, phases, match_threshold)
            if result is not None:
                per_paper.append(result)

        if not per_paper:
            return None

        report = aggregate_results(per_paper, phases)
        report["run_dir"] = str(run_dir)
        report["gt_dir"] = str(gt_dir)
        report["phases_evaluated"] = phases
        report["per_paper"] = per_paper
        return report

    except ImportError:
        return None
    except Exception as e:
        print(f"  [WARN] Benchmark evaluation failed: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate novelty assessment pipeline results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--run-dir", type=Path, required=True,
        help="Pipeline output directory (e.g. output/pipeline_results/human)",
    )
    parser.add_argument(
        "--gt-dir", type=Path, default=None,
        help="Directory with human_annotations.json (default: same as --run-dir)",
    )
    parser.add_argument(
        "--run-dir2", type=Path, default=None,
        help="Optional second run directory for comparison",
    )
    parser.add_argument(
        "--names", nargs="*", default=None,
        help="Display names for runs (e.g. 'Model-A' 'Model-B')",
    )
    parser.add_argument(
        "--phases", type=int, nargs="+", default=[1, 2, 3],
        help="Phases to evaluate with annotations (default: 1 2 3)",
    )
    parser.add_argument(
        "--match-threshold", type=float, default=0.5,
        help="ROUGE-L F1 threshold for Phase 1 (default: 0.5)",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output file path (default: stdout + auto-named file)",
    )
    parser.add_argument(
        "--format", choices=["json", "csv", "both"], default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--papers", nargs="*", default=None,
        help="Evaluate only these paper IDs",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"Error: {run_dir} is not a directory.", file=sys.stderr)
        return 1

    # Discover papers
    papers = discover_papers(run_dir)
    if not papers:
        print(f"No papers with task results found in {run_dir}", file=sys.stderr)
        return 1

    # Summary statistics
    summary = compute_summary(papers, run_dir)
    name1 = (args.names or [run_dir.name])[0]
    print_summary(summary, name1)

    # Benchmark evaluation (if annotations exist)
    benchmark = run_benchmark_evaluation(
        run_dir, args.gt_dir, args.phases, args.match_threshold, args.papers
    )
    if benchmark:
        print(f"  Benchmark metrics computed for {len(benchmark.get('per_paper', []))} papers")
        summary["benchmark"] = benchmark

    # Optional comparison with second run
    if args.run_dir2:
        run_dir2 = args.run_dir2.resolve()
        if run_dir2.is_dir():
            papers2 = discover_papers(run_dir2)
            if papers2:
                summary2 = compute_summary(papers2, run_dir2)
                name2 = args.names[1] if args.names and len(args.names) > 1 else run_dir2.name
                print_summary(summary2, name2)

                benchmark2 = run_benchmark_evaluation(
                    run_dir2, args.gt_dir, args.phases, args.match_threshold, args.papers
                )
                if benchmark2:
                    summary2["benchmark"] = benchmark2

                summary = {
                    "run1": {"name": name1, "dir": str(run_dir), **summary},
                    "run2": {"name": name2, "dir": str(run_dir2), **summary2},
                }

    # Output
    out_path = args.output
    if out_path is None:
        out_path = Path("evaluation_report.json")

    if args.format in ("json", "both"):
        p = out_path.with_suffix(".json") if args.format == "both" else (
            out_path if out_path.suffix == ".json" else out_path.with_suffix(".json")
        )
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Report saved to: {p}")

    if args.format in ("csv", "both"):
        p = out_path.with_suffix(".csv") if args.format == "both" else (
            out_path if out_path.suffix == ".csv" else out_path.with_suffix(".csv")
        )
        p.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(summary, p)
        print(f"CSV saved to: {p}")

    return 0


def _write_csv(summary: Dict[str, Any], path: Path) -> None:
    """Write summary as CSV."""
    import csv
    rows = []

    def flatten(d: Dict, prefix: str = "") -> None:
        for k, v in d.items():
            key = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict):
                flatten(v, key + ".")
            elif isinstance(v, list):
                rows.append((key, str(v)))
            else:
                rows.append((key, v))

    if "run1" in summary:
        for run_key in ("run1", "run2"):
            run = summary[run_key]
            name = run.get("name", run_key)
            for k, v in run.items():
                if k in ("name", "dir"):
                    continue
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        rows.append((f"{name}.{k}.{k2}", v2))
                else:
                    rows.append((f"{name}.{k}", v))
    else:
        flatten(summary)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for metric, value in rows:
            writer.writerow([metric, value])


if __name__ == "__main__":
    sys.exit(main())
