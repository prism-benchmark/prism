#!/usr/bin/env python3
"""
Generate publication-quality novelty figures for the main paper's
Novelty Assessment section.

Figure (a): Novelty stance distribution per reviewer as a stacked horizontal
            bar chart.
Figure (b): Task 3 score curves stratified by Task 1 stance.

Outputs (default base path: docs/paper/images/novelty_detailed_analysis.pdf):
- docs/paper/images/novelty_detailed_analysis_a.pdf
- docs/paper/images/novelty_detailed_analysis_b.pdf
(plus PNG companions)
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
# Reset ALL rcParams to defaults first to avoid dark-theme contamination
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFERENCES = ["ICLR2024", "ICLR2025", "ICLR2026", "ICML2025", "NeurIPS2025"]
MODELS = ["human", "sea", "deepreview", "reviewer2", "cyclereview", "tree"]
MODEL_LABELS = ["Human", "SEA", "DeepReview", "Reviewer2", "CycleReview", "TreeReview"]

STANCE_KEYS = ["novel", "somewhat_novel", "not_novel", "unclear"]
STANCE_LABELS = ["Novel", "Somewhat novel", "Not novel", "Unclear"]
STANCE_COLORS = ["#35C89A", "#F29A45", "#F53B3A", "#6F8DA6"]

ALL_LABELS = ["SUPPORTED", "OVERSTATED", "AMBIGUOUS", "UNDERSTATED", "UNSUPPORTED"]
SCORE_BY_LABEL = {
    "UNSUPPORTED": -2,
    "UNDERSTATED": -1,
    "AMBIGUOUS": 0,
    "OVERSTATED": 1,
    "SUPPORTED": 2,
}
SCORE_LABEL_ORDER = [
    "UNSUPPORTED",
    "UNDERSTATED",
    "AMBIGUOUS",
    "OVERSTATED",
    "SUPPORTED",
]
MODEL_COLORS = {
    "human": "#C83349",
    "sea": "#D99A2B",
    "deepreview": "#5B7FA8",
    "reviewer2": "#8A6FB0",
    "cyclereview": "#3E8B6A",
    "tree": "#444444",
}
MODEL_LINESTYLES = {
    "human": "-",
    "sea": "--",
    "deepreview": "-.",
    "reviewer2": ":",
    "cyclereview": "-",
    "tree": "-",
}

PART_B_COLUMNS = [
    ("Task 1", "Not novel", "not_novel"),
    ("Task 1", "Somewhat novel", "somewhat_novel"),
    ("Task 1", "Novel", "novel"),
    ("Task 1", "Unclear", "unclear"),
    ("Task 3", "Supported", "SUPPORTED"),
    ("Task 3", "Overstated", "OVERSTATED"),
    ("Task 3", "Ambiguous", "AMBIGUOUS"),
    ("Task 3", "Understated", "UNDERSTATED"),
    ("Task 3", "Unsupported", "UNSUPPORTED"),
]


def load_benchmarks(bench_dir: Path) -> Dict[str, Dict]:
    """Load expected benchmark JSON files, warning on omissions.

    Tries multiple naming conventions:
      - benchmark_{conf}_6models.json (legacy "balanced" outputs)
      - benchmark_{conf}.json         (full-conference outputs)
    """
    data: Dict[str, Dict] = {}
    for conf in CONFERENCES:
        candidates = [
            bench_dir / f"benchmark_{conf}_6models.json",
            bench_dir / f"benchmark_{conf}.json",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            print(
                f"Warning: missing benchmark file for {conf}; tried: "
                + ", ".join(str(p) for p in candidates),
                file=sys.stderr,
            )
            continue

        with path.open(encoding="utf-8") as f:
            data[conf] = json.load(f)

    return data


def _format_claim_count(total_claims: float) -> str:
    if total_claims >= 1000:
        return f"{total_claims / 1000:.1f}k"
    return f"{int(total_claims):,}"


def _aggregate_task1_task3_model_details(
    benchmarks: Dict[str, Dict],
) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    """Aggregate per-model Task 1 stance and Task 3 label distributions.

    Task 1 benchmark files store per-conference percentages plus total claim
    counts, so we weight by claim volume. Task 3 stores label counts directly.
    """
    details: Dict[str, Dict[str, float]] = {}
    mat = np.zeros((len(MODELS), len(PART_B_COLUMNS)), dtype=float)

    for row_idx, model in enumerate(MODELS):
        task1_counts = {stance: 0.0 for stance in STANCE_KEYS}
        task1_total = 0.0
        task3_counts = {label: 0.0 for label in ALL_LABELS}

        for conf in CONFERENCES:
            bench = benchmarks.get(conf, {})
            stance_data = bench.get("phase1_extraction", {}).get(
                "stance_distributions", {}
            )
            dist = stance_data.get("per_model", {}).get(model, {})
            n_claims = float(
                stance_data.get("total_claims_per_model", {}).get(model, 0.0)
            )
            if dist and n_claims > 0:
                for stance in STANCE_KEYS:
                    task1_counts[stance] += n_claims * float(dist.get(stance, 0.0))
                task1_total += n_claims

            label_dist = (
                bench.get("phase3_verification", {})
                .get("label_distributions", {})
                .get(model, {})
            )
            for label in ALL_LABELS:
                value = label_dist.get(label, 0.0)
                if isinstance(value, (int, float)):
                    task3_counts[label] += float(value)

        task3_total = sum(task3_counts.values())
        details[model] = {
            "task1_total_claims": task1_total,
            "task3_total_labels": task3_total,
        }

        for col_idx, (task, _, key) in enumerate(PART_B_COLUMNS):
            if task == "Task 1":
                pct = (
                    100.0 * task1_counts[key] / task1_total if task1_total > 0 else 0.0
                )
                details[model][f"task1_{key}_pct"] = pct
                details[model][f"task1_{key}_count"] = task1_counts[key]
            else:
                pct = (
                    100.0 * task3_counts[key] / task3_total if task3_total > 0 else 0.0
                )
                details[model][f"task3_{key.lower()}_pct"] = pct
                details[model][f"task3_{key.lower()}_count"] = task3_counts[key]
            mat[row_idx, col_idx] = pct

    return mat, details


def _normalize_stance(value: Any) -> str:
    stance = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if stance in {"not_novel", "non_novel", "notnovel"}:
        return "not_novel"
    if stance in {"somewhat_novel", "partially_novel", "partial_novel", "somewhat"}:
        return "somewhat_novel"
    if stance == "novel":
        return "novel"
    return "unclear"


def _score_from_task3_item(item: Dict[str, Any]) -> Optional[int]:
    score = item.get("final_score", item.get("score"))
    if isinstance(score, (int, float)):
        return int(max(-2, min(2, round(float(score)))))

    label = (
        str(
            item.get("label")
            or item.get("final_label")
            or item.get("verification_label")
            or ""
        )
        .strip()
        .upper()
    )
    if label in SCORE_BY_LABEL:
        return SCORE_BY_LABEL[label]
    return None


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _task1_stances_by_id(task1: Dict[str, Any]) -> Dict[str, str]:
    claims = (task1.get("review") or {}).get("novelty_claims") or []
    stances: Dict[str, str] = {}
    for idx, claim in enumerate(claims):
        if not isinstance(claim, dict):
            continue
        if not str(claim.get("text") or "").strip():
            continue
        claim_id = str(claim.get("claim_id") or f"S_{idx + 1:03d}")
        stances[claim_id] = _normalize_stance(claim.get("stance"))
    return stances


def _task3_scores_by_id(task3: Dict[str, Any]) -> Dict[str, int]:
    scores: Dict[str, int] = {}
    for item in task3.get("aggregated") or []:
        if not isinstance(item, dict):
            continue
        claim_id = str(item.get("review_sentence_id") or item.get("claim_id") or "")
        score = _score_from_task3_item(item)
        if claim_id and score is not None:
            scores[claim_id] = score
    return scores


def _aggregate_raw_stance_score_counts(
    benchmarks: Dict[str, Dict],
    raw_root: Optional[Path] = None,
) -> Tuple[Dict[str, Dict[str, Dict[int, float]]], int]:
    """Build exact Task 1 stance x Task 3 score counts from raw run dirs."""
    counts: Dict[str, Dict[str, Dict[int, float]]] = {
        model: {stance: defaultdict(float) for stance in STANCE_KEYS}
        for model in MODELS
    }
    total = 0

    for bench in benchmarks.values():
        models = bench.get("models") or MODELS
        run_dirs = bench.get("run_dirs") or []
        for model, run_dir_raw in zip(models, run_dirs):
            if model not in MODELS:
                continue
            run_dir = _resolve_run_dir(Path(str(run_dir_raw)), model, raw_root)
            if not run_dir.is_dir():
                continue

            for paper_dir in run_dir.iterdir():
                if not paper_dir.is_dir():
                    continue
                task1 = _read_json(paper_dir / "task1_result.json")
                task3 = _read_json(paper_dir / "task3_result.json")
                if not task1 or not task3:
                    continue

                stances = _task1_stances_by_id(task1)
                scores = _task3_scores_by_id(task3)
                for claim_id, score in scores.items():
                    stance = stances.get(claim_id)
                    if stance not in STANCE_KEYS:
                        continue
                    counts[model][stance][score] += 1.0
                    total += 1

    return counts, total


def _resolve_run_dir(run_dir: Path, model: str, raw_root: Optional[Path]) -> Path:
    """Resolve benchmark run_dirs, optionally remapping through a local raw root."""
    if run_dir.is_dir():
        return run_dir
    if raw_root is None:
        return run_dir

    conf = run_dir.name
    candidates = [
        raw_root / model / conf,
        raw_root / "full_conf_results" / model / conf,
        raw_root / "output" / "full_conf_results" / model / conf,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return run_dir


def _task3_curve_for_stance_counts(
    counts: Dict[str, Dict[str, Dict[int, float]]],
    model: str,
    stance: str,
    x: np.ndarray,
) -> np.ndarray:
    """Smooth the exact score histogram for one model within one Task 1 stance."""
    score_counts = counts.get(model, {}).get(stance, {})
    total = float(sum(score_counts.values()))
    if total <= 0:
        return np.zeros_like(x)

    y = np.zeros_like(x, dtype=float)
    bandwidth = 0.22
    norm = bandwidth * np.sqrt(2.0 * np.pi)
    for score, count in score_counts.items():
        y += (count / total) * np.exp(-0.5 * ((x - score) / bandwidth) ** 2) / norm
    return y


def _write_part_b_details_csv(path: Path, details: Dict[str, Dict[str, float]]) -> None:
    """Write the numerical values used in panel (b) for easier comparison."""
    fieldnames = ["model", "model_label", "task1_total_claims", "task3_total_labels"]
    for task, _, key in PART_B_COLUMNS:
        prefix = "task1" if task == "Task 1" else "task3"
        metric = key if task == "Task 1" else key.lower()
        fieldnames.extend([f"{prefix}_{metric}_pct", f"{prefix}_{metric}_count"])

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model, label in zip(MODELS, MODEL_LABELS):
            row = {"model": model, "model_label": label}
            row.update(details.get(model, {}))
            for key, value in list(row.items()):
                if isinstance(value, float):
                    if key.endswith("_count") or key in {
                        "task1_total_claims",
                        "task3_total_labels",
                    }:
                        row[key] = int(round(value))
                    else:
                        row[key] = round(value, 4)
            writer.writerow(row)


def create_figure(
    benchmarks: Dict[str, Dict],
    out_path: Path,
    raw_root: Optional[Path] = None,
):
    """Create two standalone figures: panel (a) and panel (b)."""

    if not benchmarks:
        raise ValueError(
            "No benchmark files were loaded; cannot create novelty detail figure."
        )

    # Force clean white academic style
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 11.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 6.0,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_a = out_path.with_name(f"{out_path.stem}_a{out_path.suffix}")
    out_b = out_path.with_name(f"{out_path.stem}_b{out_path.suffix}")

    # ==================================================================
    # Figure A: Novelty Stance Distribution per Reviewer
    # ==================================================================
    fig_a, ax_a = plt.subplots(figsize=(4.7, 4.9))
    fig_a.patch.set_facecolor("white")

    stance_props = {}
    total_claims = {model: 0.0 for model in MODELS}
    for model in MODELS:
        stance_counts = {s: 0.0 for s in STANCE_KEYS}
        for conf in CONFERENCES:
            stance_data = (
                benchmarks.get(conf, {})
                .get("phase1_extraction", {})
                .get("stance_distributions", {})
            )
            dist = stance_data.get("per_model", {}).get(model, {})
            n_claims = float(
                stance_data.get("total_claims_per_model", {}).get(model, 0)
            )
            if dist and n_claims > 0:
                for s in STANCE_KEYS:
                    stance_counts[s] += float(dist.get(s, 0.0)) * n_claims
                total_claims[model] += n_claims
        if total_claims[model] > 0:
            stance_props[model] = {
                s: 100.0 * stance_counts[s] / total_claims[model] for s in STANCE_KEYS
            }

    y = np.arange(len(MODELS))
    left = np.zeros(len(MODELS))

    for stance, label, color in zip(STANCE_KEYS, STANCE_LABELS, STANCE_COLORS):
        values = np.array(
            [stance_props.get(model, {}).get(stance, 0.0) for model in MODELS]
        )
        ax_a.barh(
            y,
            values,
            left=left,
            height=0.66,
            label=label,
            color=color,
            alpha=0.92,
            edgecolor="white",
            linewidth=0.55,
        )
        for i, (value, offset) in enumerate(zip(values, left)):
            if value >= 8.0:
                ax_a.text(
                    offset + value / 2,
                    i,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8.25,
                    color="white",
                    fontweight="bold",
                )
        left += values

    ax_a.set_yticks(y)
    ax_a.set_yticklabels(MODEL_LABELS)
    ax_a.get_yticklabels()[0].set_fontweight("bold")
    ax_a.xaxis.set_label_position("top")
    ax_a.tick_params(axis="both", which="major", width=0.9, length=5)
    ax_a.tick_params(axis="x", which="minor", width=0.8, length=3)
    ax_a.legend(
        handles=ax_a.get_legend_handles_labels()[0],
        labels=ax_a.get_legend_handles_labels()[1],
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.13),
        fontsize=11.25,
        frameon=False,
        handlelength=1.1,
        columnspacing=0.9,
        handletextpad=0.4,
        borderaxespad=0.0,
    )
    ax_a.set_xlim(0, 100)
    ax_a.xaxis.set_major_locator(mticker.MultipleLocator(25))
    ax_a.xaxis.set_minor_locator(mticker.MultipleLocator(12.5))
    ax_a.grid(axis="x", alpha=0.24, linewidth=0.35, color="#777777")
    ax_a.set_axisbelow(True)
    ax_a.invert_yaxis()
    fig_a.subplots_adjust(bottom=0.20, top=0.86, left=0.22, right=0.98)

    fig_a.savefig(str(out_a), bbox_inches="tight", dpi=300, facecolor="white")
    out_a_png = out_a.with_suffix(".png")
    fig_a.savefig(str(out_a_png), bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig_a)

    # ==================================================================
    # Figure B: Task 3 score distributions by Task 1 stance
    # ==================================================================

    _, part_b_details = _aggregate_task1_task3_model_details(benchmarks)
    if not part_b_details:
        raise ValueError("No Task 1 / Task 3 per-model detail data could be loaded.")

    stance_score_counts, stance_score_total = _aggregate_raw_stance_score_counts(
        benchmarks,
        raw_root=raw_root,
    )

    fig_b = plt.figure(figsize=(9.6, 5.8))
    fig_b.patch.set_facecolor("white")
    font_scale_b = 1.5
    grid_b = fig_b.add_gridspec(2, 2, wspace=0.30, hspace=0.50)
    ax_b_grid = [
        fig_b.add_subplot(grid_b[0, 0]),
        fig_b.add_subplot(grid_b[0, 1]),
        fig_b.add_subplot(grid_b[1, 0]),
        fig_b.add_subplot(grid_b[1, 1]),
    ]

    stance_titles = ["Not novel", "Somewhat novel", "Novel", "Unclear"]
    handles: list = []
    labels: list = []

    SCORE_DIVERGING_COLORS = {
        -2: "#B5263A",
        -1: "#E37A87",
        0: "#BFBFBF",
        1: "#7FB28A",
        2: "#2E7D4F",
    }
    SCORE_LABELS_DIV = {
        -2: "Unsupported",
        -1: "Understated",
        0: "Ambiguous",
        1: "Overstated",
        2: "Supported",
    }
    EXPECTED_ZONE = {
        "not_novel": (-1.00, 0.00),
        "somewhat_novel": (-0.40, +0.40),
        "novel": (0.00, +1.00),
        "unclear": None,
    }

    if stance_score_total <= 0:
        print(
            "Warning: no claim-level task1_result.json/task3_result.json pairs found; "
            "figure (b) cannot draw stance-conditioned Task 3 distributions.",
            file=sys.stderr,
        )
        for ax, title in zip(ax_b_grid, stance_titles):
            ax.set_title(
                title,
                fontsize=10.0 * font_scale_b,
                fontweight="bold",
                pad=3.5,
            )
            ax.axis("off")
        ax_b_grid[0].text(
            0.0,
            0.72,
            "Claim-level Task 1 x Task 3 data is unavailable.",
            transform=ax_b_grid[0].transAxes,
            ha="left",
            va="center",
            fontsize=9.0 * font_scale_b,
            fontweight="bold",
        )
        ax_b_grid[0].text(
            0.0,
            0.48,
            "The benchmark summaries only contain separate Task 1 stance and\n"
            "Task 3 score marginals, so a stance-conditioned distribution\n"
            "cannot be computed without task1_result.json/task3_result.json.",
            transform=ax_b_grid[0].transAxes,
            ha="left",
            va="center",
            fontsize=7.5 * font_scale_b,
            linespacing=1.35,
        )
    else:
        y_pos = np.arange(len(MODELS))

        for ax, stance, title in zip(ax_b_grid, STANCE_KEYS, stance_titles):
            zone = EXPECTED_ZONE.get(stance)
            if zone is not None:
                ax.axvspan(zone[0], zone[1], color="#9CC4A8", alpha=0.10, zorder=0)

            for i, model in enumerate(MODELS):
                score_counts = stance_score_counts.get(model, {}).get(stance, {})
                n = float(sum(score_counts.values()))
                if n <= 0:
                    ax.text(
                        0.0,
                        i,
                        "no data",
                        ha="center",
                        va="center",
                        fontsize=6.0 * font_scale_b,
                        color="#888",
                        style="italic",
                    )
                    continue

                left_acc = 0.0
                for s in (-1, -2):
                    p = score_counts.get(s, 0.0) / n
                    if p > 0:
                        ax.barh(
                            i,
                            -p,
                            left=left_acc,
                            height=0.66,
                            color=SCORE_DIVERGING_COLORS[s],
                            edgecolor="white",
                            linewidth=0.45,
                            zorder=2,
                        )
                        if p >= 0.10:
                            ax.text(
                                left_acc - p / 2,
                                i,
                                f"{int(round(p * 100))}",
                                ha="center",
                                va="center",
                                fontsize=5.6 * font_scale_b,
                                color="white",
                                fontweight="bold",
                                zorder=3,
                            )
                        left_acc -= p

                right_acc = 0.0
                for s in (0, 1, 2):
                    p = score_counts.get(s, 0.0) / n
                    if p > 0:
                        ax.barh(
                            i,
                            p,
                            left=right_acc,
                            height=0.66,
                            color=SCORE_DIVERGING_COLORS[s],
                            edgecolor="white",
                            linewidth=0.45,
                            zorder=2,
                        )
                        if p >= 0.10:
                            text_color = "white" if s != 0 else "#222"
                            ax.text(
                                right_acc + p / 2,
                                i,
                                f"{int(round(p * 100))}",
                                ha="center",
                                va="center",
                                fontsize=5.6 * font_scale_b,
                                color=text_color,
                                fontweight="bold",
                                zorder=3,
                            )
                        right_acc += p

                mean_score = (
                    sum(s * score_counts.get(s, 0.0) for s in (-2, -1, 0, 1, 2)) / n
                )
                ax.scatter(
                    mean_score / 2.0,
                    i,
                    s=26,
                    color="white",
                    edgecolor="black",
                    linewidth=0.9,
                    zorder=6,
                )


            ax.axvline(0.0, color="black", linewidth=0.85, zorder=4)
            ax.set_title(
                title,
                fontsize=10.0 * font_scale_b,
                fontweight="bold",
                pad=3.5,
            )
            ax.set_xlim(-0.55, 1.05)
            ax.set_ylim(-0.6, len(MODELS) - 0.4)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(MODEL_LABELS, fontsize=7.5 * font_scale_b)
            ax.get_yticklabels()[0].set_fontweight("bold")
            ax.set_xticks([-0.5, 0.0, 0.5, 1.0])
            ax.set_xticklabels(
                ["50", "0", "50", "100"],
                fontsize=7.0 * font_scale_b,
            )
            ax.tick_params(axis="both", which="major", width=0.9, length=4)
            ax.grid(axis="x", alpha=0.18, linewidth=0.35, color="#777777")
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(0.9)
            ax.spines["bottom"].set_linewidth(0.9)
            ax.invert_yaxis()

        # Place directional hints at the center gap between top and bottom rows
        for ax in (ax_b_grid[0], ax_b_grid[1]):
            ax.text(
                -0.5,
                -0.26,
                "← Not supported",
                ha="center",
                va="center",
                fontsize=7.2 * font_scale_b,
                color="#B5263A",
                fontweight="bold",
                transform=ax.get_xaxis_transform(),
            )
            ax.text(
                0.5,
                -0.26,
                "Supported →",
                ha="center",
                va="center",
                fontsize=7.2 * font_scale_b,
                color="#2E7D4F",
                fontweight="bold",
                transform=ax.get_xaxis_transform(),
            )

        for ax in (ax_b_grid[1], ax_b_grid[3]):
            ax.set_yticklabels([""] * len(MODELS))

        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        legend_order = [-2, -1, 0, 1, 2]
        handles = [
            Patch(
                facecolor=SCORE_DIVERGING_COLORS[s],
                edgecolor="white",
                label=SCORE_LABELS_DIV[s],
            )
            for s in legend_order
        ]
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                markerfacecolor="white",
                markeredgewidth=0.9,
                markersize=5,
                linewidth=0,
                label="Mean score",
            )
        )
        handles.append(
            Patch(
                facecolor="#9CC4A8",
                alpha=0.35,
                edgecolor="none",
                label="Self-consistent zone",
            )
        )
        labels = [h.get_label() for h in handles]

    if handles:
        fig_b.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.14),
            ncol=7,
            frameon=False,
            fontsize=7.3 * font_scale_b,
            handlelength=1.4,
            columnspacing=0.85,
            handletextpad=0.38,
        )
    fig_b.subplots_adjust(bottom=0.2, top=0.95, left=0.08, right=0.98)

    fig_b.savefig(str(out_b), bbox_inches="tight", dpi=300, facecolor="white")
    out_b_png = out_b.with_suffix(".png")
    fig_b.savefig(str(out_b_png), bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig_b)

    csv_path = out_b.with_name(f"{out_b.stem}_part_b_details.csv")
    _write_part_b_details_csv(csv_path, part_b_details)

    print(f"Saved: {out_a}", file=sys.stderr)
    print(f"Saved: {out_a_png}", file=sys.stderr)
    print(f"Saved: {out_b}", file=sys.stderr)
    print(f"Saved: {out_b_png}", file=sys.stderr)
    print(f"Saved: {csv_path}", file=sys.stderr)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-dir",
        type=Path,
        default=Path("output/full_conf_results/benchmarks/all_conferences"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/paper/images/novelty_detailed_analysis.pdf"),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("."),
        help=(
            "Local root containing raw per-paper outputs (defaults to the repo "
            "root so the benchmark's relative run_dirs like "
            "'output/full_conf_results/<model>/<conference>' resolve). Expected "
            "layouts: <root>/<model>/<conference>, "
            "<root>/full_conf_results/<model>/<conference>, or "
            "<root>/output/full_conf_results/<model>/<conference>."
        ),
    )
    args = parser.parse_args()

    benchmarks = load_benchmarks(args.bench_dir)
    if not benchmarks:
        raise SystemExit(f"No benchmark JSON files loaded from {args.bench_dir}")

    print(f"Loaded {len(benchmarks)} conferences", file=sys.stderr)
    create_figure(benchmarks, args.out, raw_root=args.raw_root)


if __name__ == "__main__":
    main()
