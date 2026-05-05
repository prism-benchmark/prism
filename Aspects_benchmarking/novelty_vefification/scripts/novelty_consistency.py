#!/usr/bin/env python3
"""
Combined "Diverging bars + line plot consistency summary" figure.

Top row: Four diverging stacked-bar panels showing P(Task 3 score | Task 1
         stance) per reviewer, with a shaded "self-consistent zone" marking
         the polarity expected when stance and evidence align.

Bottom row: A single line plot summarising self-consistency per reviewer.
            x-axis = ordinal Task 1 stance (Not novel -> Somewhat novel ->
            Novel); y-axis = mean Task 3 score in [-2, +2]. Each reviewer is
            one polyline; an "ideal" reference line connects (Not novel, -2)
            -> (Somewhat novel, 0) -> (Novel, +2).

Outputs:
    docs/paper/images/novelty_consistency.{pdf,png}
    docs/paper/images/novelty_consistency_means.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Reuse the data-loading + aggregation helpers from novelty_detail.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from novelty_detail import (  # type: ignore  # noqa: E402
    MODELS,
    MODEL_LABELS,
    MODEL_COLORS,
    MODEL_LINESTYLES,
    STANCE_KEYS,
    _aggregate_raw_stance_score_counts,
    load_benchmarks,
)

# ---------------------------------------------------------------------------
# Constants for the new figure
# ---------------------------------------------------------------------------

SCORE_DIVERGING_COLORS = {
    -2: "#B5263A",  # UNSUPPORTED
    -1: "#E37A87",  # UNDERSTATED
     0: "#BFBFBF",  # AMBIGUOUS
     1: "#7FB28A",  # OVERSTATED
     2: "#2E7D4F",  # SUPPORTED
}
SCORE_LABELS_DIV = {
    -2: "Unsupported",
    -1: "Understated",
     0: "Ambiguous",
     1: "Overstated",
     2: "Supported",
}

# Self-consistent expected zone in normalised proportion space [-1, +1]
EXPECTED_ZONE = {
    "not_novel":      (-1.00,  0.00),
    "somewhat_novel": (-0.40, +0.40),
    "novel":          ( 0.00, +1.00),
    "unclear":        None,
}
STANCE_TITLES = {
    "not_novel": "Not novel",
    "somewhat_novel": "Somewhat novel",
    "novel": "Novel",
    "unclear": "Unclear",
}

# Stance ordinal axis used by the consistency line plot. We deliberately
# exclude "unclear" because it has no expected polarity.
ORDINAL_STANCES = ["not_novel", "somewhat_novel", "novel"]
ORDINAL_LABELS = ["Not novel", "Somewhat novel", "Novel"]
# Ideal mean Task 3 score per stance for a perfectly self-consistent reviewer.
IDEAL_LINE = {"not_novel": -2.0, "somewhat_novel": 0.0, "novel": +2.0}


# ---------------------------------------------------------------------------
# Plot routines
# ---------------------------------------------------------------------------


def _draw_diverging_panel(ax, model_score_counts, stance):
    """Draw one diverging-stacked-bar panel for a given Task 1 stance."""
    zone = EXPECTED_ZONE.get(stance)
    if zone is not None:
        ax.axvspan(zone[0], zone[1], color="#9CC4A8", alpha=0.10, zorder=0)

    for i, model in enumerate(MODELS):
        score_counts = model_score_counts.get(model, {}).get(stance, {})
        n = float(sum(score_counts.values()))
        if n <= 0:
            ax.text(
                0.0, i, "no data",
                ha="center", va="center", fontsize=6.0,
                color="#888", style="italic",
            )
            continue

        # Negative side
        left_acc = 0.0
        for s in (-1, -2):
            p = score_counts.get(s, 0.0) / n
            if p > 0:
                ax.barh(
                    i, -p, left=left_acc, height=0.66,
                    color=SCORE_DIVERGING_COLORS[s],
                    edgecolor="white", linewidth=0.45, zorder=2,
                )
                if p >= 0.10:
                    ax.text(
                        left_acc - p / 2, i,
                        f"{int(round(p * 100))}",
                        ha="center", va="center",
                        fontsize=5.6, color="white", fontweight="bold",
                        zorder=3,
                    )
                left_acc -= p

        # Positive side
        right_acc = 0.0
        for s in (0, 1, 2):
            p = score_counts.get(s, 0.0) / n
            if p > 0:
                ax.barh(
                    i, p, left=right_acc, height=0.66,
                    color=SCORE_DIVERGING_COLORS[s],
                    edgecolor="white", linewidth=0.45, zorder=2,
                )
                if p >= 0.10:
                    text_color = "white" if s != 0 else "#222"
                    ax.text(
                        right_acc + p / 2, i,
                        f"{int(round(p * 100))}",
                        ha="center", va="center",
                        fontsize=5.6, color=text_color, fontweight="bold",
                        zorder=3,
                    )
                right_acc += p

        # Mean-score marker (normalised to [-1, +1])
        mean_score = sum(s * score_counts.get(s, 0.0) for s in (-2, -1, 0, 1, 2)) / n
        ax.scatter(
            mean_score / 2.0, i,
            s=26, color="white", edgecolor="black", linewidth=0.9, zorder=6,
        )

        # Sample size on the right
        ax.text(
            1.04, i, f"n={int(n)}",
            transform=ax.get_yaxis_transform(),
            ha="left", va="center",
            fontsize=5.8, color="#555",
        )

    ax.axvline(0.0, color="black", linewidth=0.85, zorder=4)

    ax.set_title(STANCE_TITLES[stance], fontsize=10.2, fontweight="bold", pad=4.0)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.6, len(MODELS) - 0.4)
    ax.set_yticks(np.arange(len(MODELS)))
    ax.set_yticklabels(MODEL_LABELS, fontsize=7.5)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xticklabels(["100%", "50%", "0", "50%", "100%"], fontsize=7.0)
    ax.tick_params(axis="both", which="major", width=0.9, length=4)
    ax.grid(axis="x", alpha=0.18, linewidth=0.35, color="#777777")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.invert_yaxis()

    ax.text(
        -0.5, -0.55, "← Not supported",
        ha="center", va="center",
        fontsize=6.5, color="#B5263A", fontweight="bold",
        transform=ax.get_xaxis_transform(),
    )
    ax.text(
        0.5, -0.55, "Supported →",
        ha="center", va="center",
        fontsize=6.5, color="#2E7D4F", fontweight="bold",
        transform=ax.get_xaxis_transform(),
    )


def _compute_mean_scores(model_score_counts):
    """Return {model: {stance: (mean_score, n)}} for ordinal stances."""
    means = {}
    for model in MODELS:
        per_stance = {}
        for stance in ORDINAL_STANCES + ["unclear"]:
            counts = model_score_counts.get(model, {}).get(stance, {})
            n = float(sum(counts.values()))
            if n <= 0:
                per_stance[stance] = (None, 0)
                continue
            mean_score = sum(s * counts.get(s, 0.0) for s in (-2, -1, 0, 1, 2)) / n
            per_stance[stance] = (mean_score, int(n))
        means[model] = per_stance
    return means


def _draw_consistency_line(ax, mean_scores):
    """Bottom panel: per-reviewer mean Task 3 score across ordinal stances."""
    x_idx = np.arange(len(ORDINAL_STANCES))

    # Ideal reference: y = -2, 0, +2 across the three ordinal stances.
    ideal_y = [IDEAL_LINE[s] for s in ORDINAL_STANCES]
    ax.plot(
        x_idx, ideal_y,
        linestyle=(0, (4, 3)), linewidth=1.6, color="#444444",
        zorder=2, label="Ideal (perfectly self-consistent)",
    )
    ax.scatter(
        x_idx, ideal_y,
        s=44, marker="X", color="#444444", zorder=3,
    )

    # Per-reviewer polylines
    for model, label in zip(MODELS, MODEL_LABELS):
        ys = []
        for stance in ORDINAL_STANCES:
            m, n = mean_scores[model][stance]
            ys.append(m if m is not None else np.nan)

        is_human = model == "human"
        ax.plot(
            x_idx, ys,
            color=MODEL_COLORS[model],
            linestyle=MODEL_LINESTYLES[model],
            linewidth=2.6 if is_human else 1.7,
            alpha=1.0 if is_human else 0.85,
            marker="o",
            markersize=7 if is_human else 5,
            markerfacecolor=MODEL_COLORS[model],
            markeredgecolor="white",
            markeredgewidth=0.9,
            label=label,
            zorder=8 if is_human else 5,
        )

    # Compute slope-based self-consistency index (Pearson r vs ordinal stance)
    consistency = {}
    for model in MODELS:
        ys = [mean_scores[model][s][0] for s in ORDINAL_STANCES]
        if any(v is None for v in ys):
            consistency[model] = None
            continue
        ys_arr = np.array(ys, dtype=float)
        # Slope of best-fit line; positive ~ 2.0 means perfectly consistent.
        slope = np.polyfit(x_idx, ys_arr, 1)[0]
        consistency[model] = slope

    # Annotate slopes on the right edge of the line plot
    txt_x = len(ORDINAL_STANCES) - 1 + 0.18
    used_y = []
    for model, label in zip(MODELS, MODEL_LABELS):
        slope = consistency[model]
        if slope is None:
            continue
        y_end = mean_scores[model][ORDINAL_STANCES[-1]][0]
        # Avoid label collision by nudging vertically
        for ref in used_y:
            if abs(y_end - ref) < 0.18:
                y_end = ref + 0.22
        used_y.append(y_end)
        ax.text(
            txt_x, y_end,
            f"{label}  Δ={slope:+.2f}",
            ha="left", va="center",
            fontsize=7.0, color=MODEL_COLORS[model],
            fontweight="bold" if model == "human" else "normal",
        )

    ax.set_xticks(x_idx)
    ax.set_xticklabels(ORDINAL_LABELS, fontsize=8.5)
    ax.set_xlabel("Task 1 stance (ordinal)", fontsize=9.5, fontweight="bold")
    ax.set_xlim(-0.35, len(ORDINAL_STANCES) - 1 + 1.55)
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_yticklabels(
        ["-2\nUnsupported", "-1", "0\nAmbiguous", "+1", "+2\nSupported"],
        fontsize=7.5,
    )
    ax.set_ylabel("Mean Task 3 score", fontsize=9.5, fontweight="bold")
    ax.set_ylim(-2.2, 2.2)
    ax.axhline(0, color="#999", linewidth=0.5, linestyle=":", zorder=1)
    ax.grid(axis="y", alpha=0.22, linewidth=0.35, color="#777777")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        "Self-consistency summary: mean Task 3 score by Task 1 stance",
        fontsize=10.4, fontweight="bold", pad=6.0,
    )

    return consistency


def _write_means_csv(path: Path, mean_scores, consistency):
    fieldnames = ["model", "model_label", "consistency_slope"]
    for stance in ORDINAL_STANCES + ["unclear"]:
        fieldnames += [f"{stance}_mean", f"{stance}_n"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model, label in zip(MODELS, MODEL_LABELS):
            row = {
                "model": model,
                "model_label": label,
                "consistency_slope": (
                    round(consistency[model], 4)
                    if consistency.get(model) is not None
                    else ""
                ),
            }
            for stance in ORDINAL_STANCES + ["unclear"]:
                m, n = mean_scores[model][stance]
                row[f"{stance}_mean"] = round(m, 4) if m is not None else ""
                row[f"{stance}_n"] = n
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------


def create_figure(benchmarks, out_path: Path, raw_root: Path | None = None):
    if not benchmarks:
        raise ValueError("No benchmark files were loaded.")

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
            "axes.titlesize": 9.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.0,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    counts, total = _aggregate_raw_stance_score_counts(benchmarks, raw_root=raw_root)
    if total <= 0:
        raise SystemExit(
            "No claim-level task1_result.json/task3_result.json pairs found."
        )

    fig = plt.figure(figsize=(13.4, 8.6))
    outer = GridSpec(
        2, 1,
        figure=fig,
        height_ratios=[1.05, 1.0],
        hspace=0.46,
    )
    top_grid = outer[0, 0].subgridspec(1, 4, wspace=0.62)
    ax_panels = [fig.add_subplot(top_grid[0, i]) for i in range(4)]
    ax_line = fig.add_subplot(outer[1, 0])

    # ---- Top: 4 diverging stacked-bar panels (one per stance) ----
    stance_order = ["not_novel", "somewhat_novel", "novel", "unclear"]
    for ax, stance in zip(ax_panels, stance_order):
        _draw_diverging_panel(ax, counts, stance)

    # Common x-label for top row (centered under the 4 panels)
    fig.text(
        0.5, 0.522,
        "Share of Task 3 outcomes (%)",
        ha="center", va="center",
        fontsize=10.0, fontweight="bold",
    )

    # Top-row legend (score categories, mean marker, self-consistent zone)
    legend_handles_top = [
        Patch(facecolor=SCORE_DIVERGING_COLORS[s], edgecolor="white",
              label=SCORE_LABELS_DIV[s])
        for s in (-2, -1, 0, 1, 2)
    ] + [
        Line2D([0], [0], marker="o", color="black", markerfacecolor="white",
               markeredgewidth=0.9, markersize=5, linewidth=0,
               label="Mean score"),
        Patch(facecolor="#9CC4A8", alpha=0.35, edgecolor="none",
              label="Self-consistent zone"),
    ]
    fig.legend(
        handles=legend_handles_top,
        labels=[h.get_label() for h in legend_handles_top],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=7,
        frameon=False,
        fontsize=7.6,
        handlelength=1.4,
        columnspacing=0.95,
        handletextpad=0.42,
    )

    # ---- Bottom: self-consistency line summary ----
    mean_scores = _compute_mean_scores(counts)
    consistency = _draw_consistency_line(ax_line, mean_scores)

    # Bottom-row legend (one entry per reviewer + ideal line)
    line_handles, line_labels = ax_line.get_legend_handles_labels()
    ax_line.legend(
        line_handles, line_labels,
        loc="lower right",
        ncol=2,
        frameon=False,
        fontsize=7.5,
        handlelength=2.2,
        columnspacing=1.1,
        handletextpad=0.5,
    )

    # Panel labels
    fig.text(0.04, 0.965, "(a)", fontsize=11.5, fontweight="bold", va="top")
    fig.text(0.04, 0.475, "(b)", fontsize=11.5, fontweight="bold", va="top")
    fig.text(
        0.5, 0.945,
        "Stance-conditioned Task 3 outcomes per reviewer",
        ha="center", va="top",
        fontsize=10.6, fontweight="bold",
    )

    fig.subplots_adjust(left=0.06, right=0.985, top=0.91, bottom=0.06)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight", dpi=300, facecolor="white")
    png_path = out_path.with_suffix(".png")
    fig.savefig(str(png_path), bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)

    csv_path = out_path.with_name(f"{out_path.stem}_means.csv")
    _write_means_csv(csv_path, mean_scores, consistency)

    print(f"Saved: {out_path}", file=sys.stderr)
    print(f"Saved: {png_path}", file=sys.stderr)
    print(f"Saved: {csv_path}", file=sys.stderr)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-dir", type=Path,
        default=Path("output/full_conf_results/benchmarks/all_conferences"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("docs/paper/images/novelty_consistency.pdf"),
    )
    parser.add_argument(
        "--raw-root", type=Path, default=Path("."),
        help="Root of raw per-paper outputs (see novelty_detail.py).",
    )
    args = parser.parse_args()

    benchmarks = load_benchmarks(args.bench_dir)
    if not benchmarks:
        raise SystemExit(f"No benchmark JSON files loaded from {args.bench_dir}")

    print(f"Loaded {len(benchmarks)} conferences", file=sys.stderr)
    create_figure(benchmarks, args.out, raw_root=args.raw_root)


if __name__ == "__main__":
    main()
