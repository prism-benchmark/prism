#!/usr/bin/env python3
"""
Generate publication-quality figures from novelty assessment results.

Supports two modes:
  1. Pipeline output (task1/2/3_result.json) — generates summary figures
  2. Benchmark results (from multi_model_benchmark.py) — delegates to visualize_benchmarks.py

Usage:
  # From pipeline output
  python scripts/visualize.py --run-dir output/pipeline_results/human -o figures/

  # From benchmark results (full benchmark suite)
  python scripts/visualize.py --bench-dir output/full_conf_results/benchmarks/all_conferences -o figures/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


STANCE_COLORS = {
    "not_novel": "#F44336",
    "somewhat_novel": "#FF9800",
    "novel": "#4CAF50",
    "unclear": "#9E9E9E",
    "unknown": "#BDBDBD",
}

VERDICT_COLORS = {
    "SUPPORTED": "#4CAF50",
    "OVERSTATED": "#FF9800",
    "AMBIGUOUS": "#9E9E9E",
    "UNDERSTATED": "#2196F3",
    "UNSUPPORTED": "#F44336",
}


# ---------------------------------------------------------------------------
# Data loading
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
    for conf_dir in sorted(run_dir.iterdir()):
        if not conf_dir.is_dir() or conf_dir.name.startswith("_"):
            continue
        for paper_dir in sorted(conf_dir.iterdir()):
            if not paper_dir.is_dir():
                continue
            if any((paper_dir / f"task{i}_result.json").exists() for i in (1, 2, 3)):
                papers.append(paper_dir)
    if not papers:
        for paper_dir in sorted(run_dir.iterdir()):
            if not paper_dir.is_dir() or paper_dir.name.startswith("_"):
                continue
            if any((paper_dir / f"task{i}_result.json").exists() for i in (1, 2, 3)):
                papers.append(paper_dir)
    return papers


def load_pipeline_data(run_dir: Path) -> Dict[str, Any]:
    """Load all pipeline data into a structured dict."""
    papers = discover_papers(run_dir)
    data = {
        "papers": [],
        "conferences": Counter(),
        "stance_counts": Counter(),
        "verdict_counts": Counter(),
        "contribution_counts": [],
        "candidate_counts": [],
        "claim_counts": [],
    }

    for paper_dir in papers:
        conf = paper_dir.parent.name if paper_dir.parent != run_dir else "unknown"
        paper_id = paper_dir.name
        data["conferences"][conf] += 1

        t1 = load_json(paper_dir / "task1_result.json")
        t2 = load_json(paper_dir / "task2_result.json")
        t3 = load_json(paper_dir / "task3_result.json")

        entry = {
            "paper_id": paper_id,
            "conference": conf,
            "task1": t1,
            "task2": t2,
            "task3": t3,
        }
        data["papers"].append(entry)

        if t1:
            paper = t1.get("paper", {})
            review = t1.get("review", {})
            contributions = paper.get("contributions", [])
            data["contribution_counts"].append(len(contributions))
            claims = review.get("novelty_claims", [])
            data["claim_counts"].append(len(claims))
            for claim in claims:
                if isinstance(claim, dict):
                    data["stance_counts"][claim.get("stance", "unknown")] += 1

        if t2:
            pool = t2.get("candidate_pool_top30") or t2.get("candidates") or []
            data["candidate_counts"].append(len(pool))

        if t3:
            verdicts = t3.get("verdicts") or t3.get("aggregated") or []
            for v in verdicts:
                if isinstance(v, dict):
                    label = v.get("verdict") or v.get("label")
                    if not label:
                        score = v.get("final_score")
                        if score is not None:
                            label = {2: "SUPPORTED", 1: "OVERSTATED", 0: "AMBIGUOUS", -1: "UNDERSTATED", -2: "UNSUPPORTED"}.get(int(score), "unknown")
                        else:
                            label = "unknown"
                    data["verdict_counts"][label] += 1

    return data


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------

def fig_conference_distribution(data: Dict, out_dir: Path):
    """Bar chart of papers per conference."""
    confs = data["conferences"]
    if not confs:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(confs.keys())
    counts = list(confs.values())
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    bars = ax.bar(names, counts, color=colors, edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, fmt="%d", padding=3)
    ax.set_ylabel("Number of Papers")
    ax.set_title("Papers per Conference")
    ax.set_ylim(0, max(counts) * 1.15)
    fig.savefig(out_dir / "fig_conference_distribution.png")
    plt.close(fig)


def fig_stance_distribution(data: Dict, out_dir: Path):
    """Pie/bar chart of novelty stance distribution."""
    stances = data["stance_counts"]
    if not stances:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    labels = list(stances.keys())
    counts = list(stances.values())
    colors = [STANCE_COLORS.get(l, "#BDBDBD") for l in labels]
    bars = ax1.barh(labels, counts, color=colors, edgecolor="white")
    ax1.bar_label(bars, fmt="%d", padding=3)
    ax1.set_xlabel("Count")
    ax1.set_title("Novelty Stance Distribution")
    ax1.invert_yaxis()

    # Pie chart
    ax2.pie(counts, labels=labels, colors=colors, autopct="%1.1f%%",
            startangle=90, textprops={"fontsize": 10})
    ax2.set_title("Stance Proportions")

    fig.tight_layout()
    fig.savefig(out_dir / "fig_stance_distribution.png")
    plt.close(fig)


def fig_verdict_distribution(data: Dict, out_dir: Path):
    """Bar chart of Task 3 verdict distribution."""
    verdicts = data["verdict_counts"]
    if not verdicts:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(verdicts.keys())
    counts = list(verdicts.values())
    colors = [VERDICT_COLORS.get(l, "#BDBDBD") for l in labels]
    bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, fmt="%d", padding=3)
    ax.set_ylabel("Count")
    ax.set_title("Task 3 Verdict Distribution")
    ax.set_ylim(0, max(counts) * 1.15)
    plt.xticks(rotation=15, ha="right")
    fig.savefig(out_dir / "fig_verdict_distribution.png")
    plt.close(fig)


def fig_contribution_histogram(data: Dict, out_dir: Path):
    """Histogram of contributions per paper."""
    counts = data["contribution_counts"]
    if not counts:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = range(0, max(counts) + 2)
    ax.hist(counts, bins=bins, color="#2196F3", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(counts), color="#F44336", linestyle="--", label=f"Mean = {np.mean(counts):.1f}")
    ax.set_xlabel("Contributions per Paper")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Extracted Contributions")
    ax.legend()
    fig.savefig(out_dir / "fig_contribution_histogram.png")
    plt.close(fig)


def fig_candidate_histogram(data: Dict, out_dir: Path):
    """Histogram of related work candidates per paper."""
    counts = data["candidate_counts"]
    if not counts:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = range(0, max(counts) + 2, 2)
    ax.hist(counts, bins=bins, color="#4CAF50", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(counts), color="#F44336", linestyle="--", label=f"Mean = {np.mean(counts):.1f}")
    ax.set_xlabel("Candidates per Paper")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Related Work Candidates (Task 2)")
    ax.legend()
    fig.savefig(out_dir / "fig_candidate_histogram.png")
    plt.close(fig)


def fig_stance_by_conference(data: Dict, out_dir: Path):
    """Grouped bar chart of stances per conference."""
    conf_stance = defaultdict(Counter)
    for paper in data["papers"]:
        t1 = paper.get("task1")
        if not t1:
            continue
        claims = t1.get("review", {}).get("novelty_claims", [])
        for claim in claims:
            if isinstance(claim, dict):
                conf_stance[paper["conference"]][claim.get("stance", "unknown")] += 1

    if not conf_stance:
        return

    confs = sorted(conf_stance.keys())
    all_stances = sorted(set(s for c in conf_stance.values() for s in c))
    if not all_stances:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(confs))
    width = 0.8 / len(all_stances)

    for i, stance in enumerate(all_stances):
        vals = [conf_stance[c].get(stance, 0) for c in confs]
        offset = (i - len(all_stances) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=stance,
               color=STANCE_COLORS.get(stance, "#BDBDBD"), edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(confs, rotation=15, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Novelty Stance by Conference")
    ax.legend(title="Stance", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_stance_by_conference.png")
    plt.close(fig)


def fig_verdict_by_conference(data: Dict, out_dir: Path):
    """Grouped bar chart of verdicts per conference."""
    conf_verdict = defaultdict(Counter)
    for paper in data["papers"]:
        t3 = paper.get("task3")
        if not t3:
            continue
        verdicts = t3.get("verdicts") or t3.get("aggregated") or []
        for v in verdicts:
            if isinstance(v, dict):
                label = v.get("verdict") or v.get("label")
                if not label:
                    score = v.get("final_score")
                    if score is not None:
                        label = {2: "SUPPORTED", 1: "OVERSTATED", 0: "AMBIGUOUS", -1: "UNDERSTATED", -2: "UNSUPPORTED"}.get(int(score), "unknown")
                    else:
                        label = "unknown"
                conf_verdict[paper["conference"]][label] += 1

    if not conf_verdict:
        return

    confs = sorted(conf_verdict.keys())
    all_verdicts = sorted(set(v for c in conf_verdict.values() for v in c))
    if not all_verdicts:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(confs))
    width = 0.8 / len(all_verdicts)

    for i, verdict in enumerate(all_verdicts):
        vals = [conf_verdict[c].get(verdict, 0) for c in confs]
        offset = (i - len(all_verdicts) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=verdict,
               color=VERDICT_COLORS.get(verdict, "#BDBDBD"), edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(confs, rotation=15, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Task 3 Verdicts by Conference")
    ax.legend(title="Verdict", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_verdict_by_conference.png")
    plt.close(fig)


def fig_pipeline_success(data: Dict, out_dir: Path):
    """Stacked bar chart of pipeline success per conference."""
    conf_success = defaultdict(lambda: {"t1": 0, "t2": 0, "t3": 0, "full": 0, "total": 0})
    for paper in data["papers"]:
        conf = paper["conference"]
        conf_success[conf]["total"] += 1
        if paper.get("task1"):
            conf_success[conf]["t1"] += 1
        if paper.get("task2"):
            conf_success[conf]["t2"] += 1
        if paper.get("task3"):
            conf_success[conf]["t3"] += 1
        if paper.get("task1") and paper.get("task2") and paper.get("task3"):
            conf_success[conf]["full"] += 1

    if not conf_success:
        return

    confs = sorted(conf_success.keys())
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(confs))
    width = 0.2
    ax.bar(x - 1.5*width, [conf_success[c]["t1"] for c in confs], width, label="Task 1", color="#2196F3")
    ax.bar(x - 0.5*width, [conf_success[c]["t2"] for c in confs], width, label="Task 2", color="#4CAF50")
    ax.bar(x + 0.5*width, [conf_success[c]["t3"] for c in confs], width, label="Task 3", color="#FF9800")
    ax.bar(x + 1.5*width, [conf_success[c]["full"] for c in confs], width, label="Full", color="#9C27B0")

    ax.set_xticks(x)
    ax.set_xticklabels(confs, rotation=15, ha="right")
    ax.set_ylabel("Papers")
    ax.set_title("Pipeline Success by Conference")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig_pipeline_success.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_FIGURES = [
    ("fig_conference_distribution", fig_conference_distribution),
    ("fig_stance_distribution", fig_stance_distribution),
    ("fig_verdict_distribution", fig_verdict_distribution),
    ("fig_contribution_histogram", fig_contribution_histogram),
    ("fig_candidate_histogram", fig_candidate_histogram),
    ("fig_stance_by_conference", fig_stance_by_conference),
    ("fig_verdict_by_conference", fig_verdict_by_conference),
    ("fig_pipeline_success", fig_pipeline_success),
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate figures from novelty assessment results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--run-dir", type=Path,
        help="Pipeline output directory with task1/2/3 results",
    )
    mode.add_argument(
        "--bench-dir", type=Path,
        help="Benchmark results directory (delegates to visualize_benchmarks.py)",
    )
    parser.add_argument(
        "-o", "--out-dir", type=Path, default=Path("figures"),
        help="Output directory for figures (default: figures/)",
    )
    parser.add_argument(
        "--figures", nargs="*", default=None,
        help="Generate only specific figures (e.g. stance_distribution verdict_distribution)",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.bench_dir:
        # Delegate to visualize_benchmarks.py
        import subprocess
        cmd = [
            sys.executable, str(Path(__file__).parent / "visualize_benchmarks.py"),
            "--bench-dir", str(args.bench_dir),
            "--out-dir", str(args.out_dir),
        ]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        return result.returncode

    # Pipeline mode
    setup_style()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"Error: {run_dir} is not a directory.", file=sys.stderr)
        return 1

    print(f"Loading pipeline data from {run_dir}...", file=sys.stderr)
    data = load_pipeline_data(run_dir)
    n_papers = len(data["papers"])
    print(f"  Loaded {n_papers} papers from {len(data['conferences'])} conferences", file=sys.stderr)

    if n_papers == 0:
        print("No papers found.", file=sys.stderr)
        return 1

    # Filter figures if specified
    figures_to_gen = ALL_FIGURES
    if args.figures:
        names = set(args.figures)
        figures_to_gen = [(n, f) for n, f in ALL_FIGURES if any(k in n for k in names)]

    for name, func in figures_to_gen:
        print(f"  Generating {name}...", file=sys.stderr)
        try:
            func(data, args.out_dir)
        except Exception as e:
            print(f"  [ERROR] {name}: {e}", file=sys.stderr)

    print(f"\n✓ Figures saved to: {args.out_dir}/", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
