#!/usr/bin/env python3
"""
Visualize multi-conference benchmark results and ablation analysis.

Generates publication-quality figures from benchmark JSON outputs
and ablation aggregation JSON files.

Usage:
  python scripts/visualize_benchmarks.py
  python scripts/visualize_benchmarks.py --bench-dir output/full_conf_results/benchmarks/all_conferences
  python scripts/visualize_benchmarks.py --ablation-dir output/full_conf_results/benchmarks/ablation
  python scripts/visualize_benchmarks.py --error-decomp error_decomposition.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

CONFERENCES = ["ICLR2024", "ICLR2025", "ICLR2026", "ICML2025", "Neurlps2025"]
CONF_LABELS = ["ICLR'24", "ICLR'25", "ICLR'26", "ICML'25", "NeurIPS'25"]
MODELS = ["human", "sea", "deepreview", "reviewer2", "cyclereview", "tree"]
MODEL_LABELS = ["Human", "SEA", "DeepReview", "Reviewer2", "CycleReview", "Tree"]
MODEL_COLORS = {
    "human": "#2196F3",
    "sea": "#E91E63",
    "deepreview": "#FF9800",
    "reviewer2": "#4CAF50",
    "cyclereview": "#9C27B0",
    "tree": "#607D8B",
}

ALL_LABELS = ["SUPPORTED", "OVERSTATED", "AMBIGUOUS", "UNDERSTATED", "UNSUPPORTED"]
LABEL_COLORS = {
    "SUPPORTED": "#4CAF50",
    "OVERSTATED": "#FF9800",
    "AMBIGUOUS": "#9E9E9E",
    "UNDERSTATED": "#2196F3",
    "UNSUPPORTED": "#F44336",
}

POLICIES = ["max", "mean", "weighted", "top3_relevance"]
POLICY_LABELS = ["Max", "Mean", "Weighted", "Top-3 Relevance"]
POLICY_COLORS = ["#F44336", "#2196F3", "#FF9800", "#4CAF50"]


def _setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    if HAS_SEABORN:
        sns.set_palette("Set2")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_benchmarks(bench_dir: Path) -> Dict[str, Dict]:
    data = {}
    for conf in CONFERENCES:
        candidates = [
            bench_dir / f"benchmark_{conf}_6models.json",
            bench_dir / f"benchmark_{conf}.json",
        ]
        for p in candidates:
            if p.exists():
                with p.open() as f:
                    data[conf] = json.load(f)
                break
    return data


def load_ablations(ablation_dir: Path) -> Dict[Tuple[str, str], Dict]:
    data = {}
    if not ablation_dir.is_dir():
        return data
    for p in ablation_dir.glob("*.json"):
        parts = p.stem.replace("_ablation", "").split("_", 1)
        if len(parts) == 2:
            model, conf = parts
            with p.open() as f:
                data[(model, conf)] = json.load(f)
    return data


def load_error_decompositions(
    error_decomp_path: Optional[Path],
    bench_dir: Path,
) -> Dict[str, Dict]:
    """Load one or many error decomposition JSONs.

    Supports:
      - explicit file path
      - explicit directory path (loads *.json)
      - auto-discovery (error_decomposition.json in common locations)
    """
    paths: List[Path] = []

    if error_decomp_path:
        if error_decomp_path.is_file():
            paths.append(error_decomp_path)
        elif error_decomp_path.is_dir():
            paths.extend(sorted(error_decomp_path.glob("*.json")))
        else:
            print(f"  [WARN] --error-decomp not found: {error_decomp_path}", file=sys.stderr)
    else:
        auto_candidates = [
            Path("error_decomposition.json"),
            bench_dir.parent / "error_decomposition.json",
            bench_dir.parent.parent / "error_decomposition.json",
        ]
        for p in auto_candidates:
            if p.is_file():
                paths.append(p)

    data: Dict[str, Dict] = {}
    for p in paths:
        try:
            with p.open() as f:
                payload = json.load(f)
        except Exception as e:
            print(f"  [WARN] Failed to read {p}: {e}", file=sys.stderr)
            continue

        base_name = payload.get("name_b") or payload.get("name_a") or p.stem
        key = base_name
        idx = 2
        while key in data:
            key = f"{base_name}_{idx}"
            idx += 1
        data[key] = payload

    return data


# ---------------------------------------------------------------------------
# Figure 1: Claim counts per model across conferences
# ---------------------------------------------------------------------------

def fig_claim_counts(benchmarks: Dict[str, Dict], out_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(CONFERENCES))
    width = 0.12
    offsets = np.arange(len(MODELS)) - (len(MODELS) - 1) / 2

    for i, (model, label) in enumerate(zip(MODELS, MODEL_LABELS)):
        means, stds = [], []
        for conf in CONFERENCES:
            b = benchmarks.get(conf, {})
            stats = b.get("phase1_extraction", {}).get("claim_count_stats", {}).get(model, {})
            means.append(stats.get("mean", 0))
            stds.append(stats.get("std", 0))
        bars = ax.bar(x + offsets[i] * width, means, width, yerr=stds,
                      label=label, color=MODEL_COLORS[model],
                      capsize=2, error_kw={"lw": 0.8})
        for bar, m in zip(bars, means):
            if m > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{m:.1f}", ha="center", va="bottom", fontsize=6)

    ax.set_xlabel("Conference")
    ax.set_ylabel("Claims per paper (mean ± std)")
    ax.set_title("Phase 1: Claim Count per Model Across Conferences")
    ax.set_xticks(x)
    ax.set_xticklabels(CONF_LABELS)
    ax.legend(loc="upper right", ncol=3, framealpha=0.9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_claim_counts.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Stance distributions (stacked bar per model per conference)
# ---------------------------------------------------------------------------

def fig_stance_distributions(benchmarks: Dict[str, Dict], out_dir: Path):
    fig, axes = plt.subplots(1, len(CONFERENCES), figsize=(18, 4.5), sharey=True)
    stance_keys = ["not_novel", "somewhat_novel", "novel", "unclear"]
    stance_labels = ["Not Novel", "Somewhat Novel", "Novel", "Unclear"]
    stance_colors = ["#F44336", "#FF9800", "#4CAF50", "#9E9E9E"]

    for ci, (conf, conf_label) in enumerate(zip(CONFERENCES, CONF_LABELS)):
        ax = axes[ci]
        b = benchmarks.get(conf, {})
        dist = b.get("phase1_extraction", {}).get("stance_distributions", {}).get("per_model", {})

        y_pos = np.arange(len(MODELS))
        for mi, model in enumerate(MODELS):
            md = dist.get(model, {})
            left = 0
            for si, sk in enumerate(stance_keys):
                val = md.get(sk, 0) * 100
                ax.barh(mi, val, left=left, color=stance_colors[si], height=0.7,
                        label=stance_labels[si] if mi == 0 and ci == 0 else None)
                if val > 8:
                    ax.text(left + val / 2, mi, f"{val:.0f}%",
                            ha="center", va="center", fontsize=6, color="white", fontweight="bold")
                left += val

        ax.set_yticks(y_pos)
        ax.set_yticklabels(MODEL_LABELS if ci == 0 else [])
        ax.set_xlabel("Percentage")
        ax.set_title(conf_label, fontweight="bold")
        ax.set_xlim(0, 100)

    fig.legend(stance_labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Phase 1: Stance Distributions per Model", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_stance_distributions.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Retrieval Jaccard@K across conferences
# ---------------------------------------------------------------------------

def fig_retrieval_jaccard(benchmarks: Dict[str, Dict], out_dir: Path):
    k_vals = [5, 10, 20, 30]
    # Focus on human_vs_X pairs (excluding human_vs_tree which is always 1.0)
    target_models = ["sea", "deepreview", "reviewer2", "cyclereview"]
    target_labels = ["SEA", "DeepReview", "Reviewer2", "CycleReview"]
    colors = [MODEL_COLORS[m] for m in target_models]

    fig, axes = plt.subplots(1, len(k_vals), figsize=(16, 4), sharey=True)

    for ki, k in enumerate(k_vals):
        ax = axes[ki]
        x = np.arange(len(CONFERENCES))
        width = 0.18

        for mi, (model, label, color) in enumerate(zip(target_models, target_labels, colors)):
            vals = []
            for conf in CONFERENCES:
                b = benchmarks.get(conf, {})
                jac = b.get("phase2_retrieval", {}).get("pairwise_jaccard", {})
                pair_key = f"human_vs_{model}"
                vals.append(jac.get(pair_key, {}).get(f"mean_jaccard@{k}", 0))
            ax.bar(x + (mi - 1.5) * width, vals, width, label=label, color=color)

        ax.set_title(f"Jaccard@{k}", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(CONF_LABELS, rotation=30, ha="right")
        ax.set_ylim(0, 0.5)
        if ki == 0:
            ax.set_ylabel("Jaccard Index")
        if ki == len(k_vals) - 1:
            ax.legend(loc="upper right", fontsize=7)

    fig.suptitle("Phase 2: Retrieval Pool Overlap (Human vs. Each Model)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_retrieval_jaccard.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Agreement heatmaps (Cohen's κ) across conferences
# ---------------------------------------------------------------------------

def fig_agreement_heatmaps(benchmarks: Dict[str, Dict], out_dir: Path):
    fig, axes = plt.subplots(1, len(CONFERENCES), figsize=(22, 4.5))

    for ci, (conf, conf_label) in enumerate(zip(CONFERENCES, CONF_LABELS)):
        ax = axes[ci]
        b = benchmarks.get(conf, {})
        pp = b.get("phase3_verification", {}).get("pairwise_paper_level", {})

        n = len(MODELS)
        mat = np.full((n, n), np.nan)
        for i in range(n):
            mat[i, i] = 1.0
            for j in range(i + 1, n):
                pair1 = f"{MODELS[i]}_vs_{MODELS[j]}"
                pair2 = f"{MODELS[j]}_vs_{MODELS[i]}"
                val = pp.get(pair1, pp.get(pair2, {})).get("cohen_kappa", np.nan)
                mat[i, j] = val
                mat[j, i] = val

        im = ax.imshow(mat, cmap="RdYlGn", vmin=-0.1, vmax=0.5, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_xticklabels(MODEL_LABELS, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(n))
        ax.set_yticklabels(MODEL_LABELS if ci == 0 else [], fontsize=7)
        ax.set_title(conf_label, fontweight="bold", fontsize=10)

        for i in range(n):
            for j in range(n):
                v = mat[i, j]
                if not np.isnan(v):
                    color = "white" if v < 0.15 or v > 0.4 else "black"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6, color=color)

    fig.colorbar(im, ax=axes, label="Cohen's κ", shrink=0.8, pad=0.02)
    fig.suptitle("Phase 3: Pairwise Cohen's κ (Paper-Level) Across Conferences",
                 fontweight="bold", y=1.02)
    fig.savefig(out_dir / "fig_agreement_heatmaps.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: Label distributions across conferences
# ---------------------------------------------------------------------------

def fig_label_distributions(benchmarks: Dict[str, Dict], out_dir: Path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for mi, (model, label) in enumerate(zip(MODELS, MODEL_LABELS)):
        ax = axes[mi]
        x = np.arange(len(CONFERENCES))
        bottom = np.zeros(len(CONFERENCES))

        for li, lbl in enumerate(ALL_LABELS):
            vals = []
            for conf in CONFERENCES:
                b = benchmarks.get(conf, {})
                ld = b.get("phase3_verification", {}).get("label_distributions", {}).get(model, {})
                vals.append(ld.get(lbl, 0))
            vals_arr = np.array(vals, dtype=float)
            ax.bar(x, vals_arr, bottom=bottom, color=LABEL_COLORS[lbl],
                   label=lbl if mi == 0 else None, width=0.6)
            bottom += vals_arr

        ax.set_xticks(x)
        ax.set_xticklabels(CONF_LABELS, rotation=30, ha="right", fontsize=7)
        ax.set_title(label, fontweight="bold")
        ax.set_ylabel("Count")

    fig.legend(ALL_LABELS, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Phase 3: Label Distribution per Model Across Conferences",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_label_distributions.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: Multi-rater agreement (Fleiss κ, Krippendorff α) across confs
# ---------------------------------------------------------------------------

def fig_multi_rater(benchmarks: Dict[str, Dict], out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(CONFERENCES))
    width = 0.3

    fleiss_vals, kripp_vals = [], []
    for conf in CONFERENCES:
        b = benchmarks.get(conf, {})
        p3 = b.get("phase3_verification", {})
        fleiss_vals.append(p3.get("fleiss_kappa_paper_level", 0))
        kripp_vals.append(p3.get("krippendorff_alpha_paper_level", 0))

    bars1 = ax.bar(x - width / 2, fleiss_vals, width, label="Fleiss' κ", color="#2196F3")
    bars2 = ax.bar(x + width / 2, kripp_vals, width, label="Krippendorff's α", color="#FF9800")

    for bar, val in zip(bars1, fleiss_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, kripp_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(CONF_LABELS)
    ax.set_ylabel("Agreement Score")
    ax.set_title("Phase 3: Multi-Rater Agreement Across Conferences")
    ax.legend()
    ax.set_ylim(0, max(max(fleiss_vals), max(kripp_vals)) * 1.3)
    ax.axhline(y=0.2, color="gray", linestyle="--", alpha=0.5, label="Slight agreement")
    ax.axhline(y=0.4, color="gray", linestyle=":", alpha=0.5, label="Fair agreement")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_multi_rater_agreement.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7: Claim-level weighted κ with bootstrap CIs (human vs each)
# ---------------------------------------------------------------------------

def fig_claim_level_kappa(benchmarks: Dict[str, Dict], out_dir: Path):
    target_models = ["sea", "deepreview", "reviewer2", "cyclereview", "tree"]
    target_labels = ["SEA", "DeepReview", "Reviewer2", "CycleReview", "Tree"]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(CONFERENCES))
    width = 0.15

    for mi, (model, label) in enumerate(zip(target_models, target_labels)):
        vals, ci_lo, ci_hi = [], [], []
        for conf in CONFERENCES:
            b = benchmarks.get(conf, {})
            pc = b.get("phase3_verification", {}).get("pairwise_claim_level", {})
            pair_key = f"human_vs_{model}"
            entry = pc.get(pair_key, {})
            wk = entry.get("weighted_kappa_quadratic", 0)
            ci = entry.get("bootstrap_ci_weighted_kappa", {})
            vals.append(wk)
            ci_lo.append(wk - ci.get("ci_lower", wk))
            ci_hi.append(ci.get("ci_upper", wk) - wk)

        offset = (mi - 2) * width
        ax.bar(x + offset, vals, width, yerr=[ci_lo, ci_hi],
               label=label, color=MODEL_COLORS[model],
               capsize=2, error_kw={"lw": 0.8})

    ax.set_xticks(x)
    ax.set_xticklabels(CONF_LABELS)
    ax.set_ylabel("Weighted κ (quadratic)")
    ax.set_title("Phase 3: Claim-Level Weighted Cohen's κ (Human vs. Each Model) with 95% CI")
    ax.legend(loc="upper right", ncol=3)
    ax.axhline(y=0.2, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=0.4, color="gray", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_claim_level_kappa.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 8: Paper-level mean scores per model across conferences
# ---------------------------------------------------------------------------

def fig_paper_scores(benchmarks: Dict[str, Dict], out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(CONFERENCES))
    width = 0.12

    for i, (model, label) in enumerate(zip(MODELS, MODEL_LABELS)):
        means, stds = [], []
        for conf in CONFERENCES:
            b = benchmarks.get(conf, {})
            stats = b.get("phase3_verification", {}).get("paper_score_stats", {}).get(model, {})
            means.append(stats.get("mean", 0))
            stds.append(stats.get("std", 0))

        offset = (i - (len(MODELS) - 1) / 2) * width
        bars = ax.bar(x + offset, means, width, yerr=stds,
                      label=label, color=MODEL_COLORS[model],
                      capsize=2, error_kw={"lw": 0.7})
        for bar, m in zip(bars, means):
            if m > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        f"{m:.2f}", ha="center", va="bottom", fontsize=5.5)

    ax.set_xticks(x)
    ax.set_xticklabels(CONF_LABELS)
    ax.set_ylabel("Mean Verification Score")
    ax.set_title("Phase 3: Paper-Level Mean Score per Model Across Conferences")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    ax.axhline(y=0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_paper_scores.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 9: Ablation — aggregation policy comparison across models & confs
# ---------------------------------------------------------------------------

def fig_ablation_policies(ablations: Dict[Tuple[str, str], Dict], out_dir: Path):
    if not ablations:
        print("  [SKIP] No ablation data found", file=sys.stderr)
        return

    # Aggregate: for each (model, policy) → average across conferences
    abl_models = sorted(set(m for m, c in ablations.keys()))
    if not abl_models:
        return

    metrics = ["avg_ns", "avg_sr", "avg_ssr"]
    metric_labels = ["Normalized Score (NS)", "Support Rate (SR)", "Strong Support Rate (SSR)"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

    for mi, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[mi]
        x = np.arange(len(abl_models))
        width = 0.18

        for pi, (policy, plabel, pcolor) in enumerate(
            zip(POLICIES, POLICY_LABELS, POLICY_COLORS)
        ):
            vals = []
            for model in abl_models:
                model_vals = []
                for conf in CONFERENCES:
                    entry = ablations.get((model, conf))
                    if entry:
                        model_vals.append(
                            entry.get("policies", {}).get(policy, {}).get(metric, 0)
                        )
                vals.append(np.mean(model_vals) if model_vals else 0)

            offset = (pi - 1.5) * width
            bars = ax.bar(x + offset, vals, width, label=plabel if mi == 0 else None,
                          color=pcolor, alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=6)

        ax.set_xticks(x)
        model_display = {m: l for m, l in zip(MODELS, MODEL_LABELS)}
        ax.set_xticklabels([model_display.get(m, m) for m in abl_models], rotation=30, ha="right")
        ax.set_title(metric_label, fontweight="bold")
        ax.set_ylim(0, 1.05)

    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Ablation: Aggregation Policy Sensitivity (Averaged Across 5 Conferences)",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_ablation_policies.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 10: Ablation — policy effect per conference (line chart)
# ---------------------------------------------------------------------------

def fig_ablation_per_conference(ablations: Dict[Tuple[str, str], Dict], out_dir: Path):
    if not ablations:
        return

    abl_models = sorted(set(m for m, c in ablations.keys()))
    if not abl_models:
        return

    fig, axes = plt.subplots(1, len(abl_models), figsize=(5 * len(abl_models), 4.5), sharey=True)
    if len(abl_models) == 1:
        axes = [axes]

    for mi, model in enumerate(abl_models):
        ax = axes[mi]
        for pi, (policy, plabel, pcolor) in enumerate(
            zip(POLICIES, POLICY_LABELS, POLICY_COLORS)
        ):
            vals = []
            confs_present = []
            for ci, conf in enumerate(CONFERENCES):
                entry = ablations.get((model, conf))
                if entry:
                    vals.append(entry.get("policies", {}).get(policy, {}).get("avg_sr", 0))
                    confs_present.append(ci)

            ax.plot(confs_present, vals, "o-", label=plabel, color=pcolor, markersize=5)

        ax.set_xticks(range(len(CONFERENCES)))
        ax.set_xticklabels(CONF_LABELS, rotation=30, ha="right", fontsize=7)
        model_display = {m: l for m, l in zip(MODELS, MODEL_LABELS)}
        ax.set_title(model_display.get(model, model), fontweight="bold")
        if mi == 0:
            ax.set_ylabel("Support Rate (SR)")
        if mi == len(abl_models) - 1:
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Ablation: Support Rate by Aggregation Policy per Conference",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_ablation_per_conference.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 11: Summary radar / grouped bar — human vs. all models
# ---------------------------------------------------------------------------

def fig_summary_metrics(benchmarks: Dict[str, Dict], out_dir: Path):
    target_models = ["sea", "deepreview", "reviewer2", "cyclereview", "tree"]
    target_labels = ["SEA", "DeepReview", "Reviewer2", "CycleReview", "Tree"]
    metrics = ["cohen_kappa", "accuracy", "pearson_r", "spearman_rho"]
    metric_labels = ["Cohen's κ", "Accuracy", "Pearson r", "Spearman ρ"]

    # Average across all conferences
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics))
    width = 0.15

    for mi, (model, label) in enumerate(zip(target_models, target_labels)):
        avg_vals = []
        for metric in metrics:
            vals_across_confs = []
            for conf in CONFERENCES:
                b = benchmarks.get(conf, {})
                pp = b.get("phase3_verification", {}).get("pairwise_paper_level", {})
                pair_key = f"human_vs_{model}"
                val = pp.get(pair_key, {}).get(metric)
                if val is not None:
                    vals_across_confs.append(val)
            avg_vals.append(np.mean(vals_across_confs) if vals_across_confs else 0)

        offset = (mi - 2) * width
        bars = ax.bar(x + offset, avg_vals, width, label=label, color=MODEL_COLORS[model])
        for bar, v in zip(bars, avg_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_title("Phase 3: Human vs. Each Model — Paper-Level Metrics (Averaged Across 5 Conferences)")
    ax.legend(loc="upper right", ncol=3)
    ax.set_ylim(0, 0.65)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_summary_metrics.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 12: Claim-level macro F1 with CIs
# ---------------------------------------------------------------------------

def fig_claim_f1(benchmarks: Dict[str, Dict], out_dir: Path):
    target_models = ["sea", "deepreview", "reviewer2", "cyclereview", "tree"]
    target_labels = ["SEA", "DeepReview", "Reviewer2", "CycleReview", "Tree"]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(CONFERENCES))
    width = 0.15

    for mi, (model, label) in enumerate(zip(target_models, target_labels)):
        vals, ci_lo, ci_hi = [], [], []
        for conf in CONFERENCES:
            b = benchmarks.get(conf, {})
            pc = b.get("phase3_verification", {}).get("pairwise_claim_level", {})
            pair_key = f"human_vs_{model}"
            entry = pc.get(pair_key, {})
            f1 = entry.get("macro_f1", 0)
            ci = entry.get("bootstrap_ci_macro_f1", {})
            vals.append(f1)
            ci_lo.append(f1 - ci.get("ci_lower", f1))
            ci_hi.append(ci.get("ci_upper", f1) - f1)

        offset = (mi - 2) * width
        ax.bar(x + offset, vals, width, yerr=[ci_lo, ci_hi],
               label=label, color=MODEL_COLORS[model],
               capsize=2, error_kw={"lw": 0.8})

    ax.set_xticks(x)
    ax.set_xticklabels(CONF_LABELS)
    ax.set_ylabel("Macro F1")
    ax.set_title("Phase 3: Claim-Level Macro F1 (Human vs. Each Model) with 95% CI")
    ax.legend(loc="upper right", ncol=3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_claim_f1.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 13: Pairwise claim overlap (core task similarity) heatmap
# ---------------------------------------------------------------------------

def fig_core_task_similarity(benchmarks: Dict[str, Dict], out_dir: Path):
    fig, axes = plt.subplots(1, len(CONFERENCES), figsize=(22, 4.5))

    for ci, (conf, conf_label) in enumerate(zip(CONFERENCES, CONF_LABELS)):
        ax = axes[ci]
        b = benchmarks.get(conf, {})
        cts = b.get("phase1_extraction", {}).get("pairwise_core_task_similarity", {})

        n = len(MODELS)
        mat = np.full((n, n), np.nan)
        for i in range(n):
            mat[i, i] = 1.0
            for j in range(i + 1, n):
                pair1 = f"{MODELS[i]}_vs_{MODELS[j]}"
                pair2 = f"{MODELS[j]}_vs_{MODELS[i]}"
                val = cts.get(pair1, cts.get(pair2, {}))
                if isinstance(val, dict):
                    val = val.get("mean", np.nan)
                mat[i, j] = val
                mat[j, i] = val

        im = ax.imshow(mat, cmap="YlGnBu", vmin=0.6, vmax=0.8, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_xticklabels(MODEL_LABELS, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(n))
        ax.set_yticklabels(MODEL_LABELS if ci == 0 else [], fontsize=7)
        ax.set_title(conf_label, fontweight="bold", fontsize=10)

        for i in range(n):
            for j in range(n):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6)

    fig.colorbar(im, ax=axes, label="ROUGE-L Similarity", shrink=0.8, pad=0.02)
    fig.suptitle("Phase 1: Core Task Similarity (ROUGE-L) Across Conferences",
                 fontweight="bold", y=1.02)
    fig.savefig(out_dir / "fig_core_task_similarity.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 14: Claim-level confusion matrices (human vs each model)
# ---------------------------------------------------------------------------

def fig_claim_confusion_matrices(benchmarks: Dict[str, Dict], out_dir: Path):
    target_models = ["sea", "deepreview", "reviewer2", "cyclereview", "tree"]
    target_labels = ["SEA", "DeepReview", "Reviewer2", "CycleReview", "Tree"]

    n_labels = len(ALL_LABELS)
    fig, axes = plt.subplots(1, len(target_models), figsize=(4.4 * len(target_models), 4.5), sharey=True)
    if len(target_models) == 1:
        axes = [axes]

    for mi, (model, model_label) in enumerate(zip(target_models, target_labels)):
        ax = axes[mi]

        # Sum confusion matrices across conferences for stability.
        mat = np.zeros((n_labels, n_labels), dtype=float)
        for conf in CONFERENCES:
            b = benchmarks.get(conf, {})
            pc = b.get("phase3_verification", {}).get("pairwise_claim_level", {})
            pair_key = f"human_vs_{model}"
            cm = pc.get(pair_key, {}).get("confusion_matrix", {})

            for i, true_lbl in enumerate(ALL_LABELS):
                row = cm.get(true_lbl, {}) if isinstance(cm, dict) else {}
                for j, pred_lbl in enumerate(ALL_LABELS):
                    mat[i, j] += row.get(pred_lbl, 0)

        row_sums = mat.sum(axis=1, keepdims=True)
        mat_norm = np.divide(mat, row_sums, where=row_sums > 0)

        im = ax.imshow(mat_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(n_labels))
        ax.set_xticklabels(ALL_LABELS, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(n_labels))
        ax.set_yticklabels(ALL_LABELS if mi == 0 else [], fontsize=7)
        ax.set_title(model_label, fontweight="bold")

        for i in range(n_labels):
            for j in range(n_labels):
                v = mat_norm[i, j] if row_sums[i, 0] > 0 else 0
                color = "white" if v > 0.5 else "black"
                ax.text(j, i, f"{100 * v:.0f}%", ha="center", va="center", fontsize=6, color=color)

    fig.colorbar(im, ax=axes, label="Row-normalized percentage", shrink=0.85, pad=0.01)
    fig.suptitle("Phase 3: Claim-Level Confusion Matrices (Human vs. Each Model)",
                 fontweight="bold", y=1.02)
    axes[0].set_ylabel("Human label")
    for ax in axes:
        ax.set_xlabel("Model label")
    fig.savefig(out_dir / "fig_claim_confusion_matrices.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 15: Stage-wise error decomposition
# ---------------------------------------------------------------------------

def fig_error_decomposition(error_decomps: Dict[str, Dict], out_dir: Path):
    if not error_decomps:
        print("  [SKIP] No error decomposition data found", file=sys.stderr)
        return

    stage_labels = ["Extraction", "Retrieval", "Judging"]
    stage_colors = ["#F44336", "#FF9800", "#4CAF50"]

    names = list(error_decomps.keys())
    x = np.arange(len(names))

    attr_vals = np.array([
        [
            error_decomps[name].get("attribution", {}).get("extraction_pct", 0),
            error_decomps[name].get("attribution", {}).get("retrieval_pct", 0),
            error_decomps[name].get("attribution", {}).get("judging_pct", 0),
        ]
        for name in names
    ], dtype=float)

    dis_vals = np.array([
        [
            error_decomps[name].get("extraction", {}).get("disagreement", 0),
            error_decomps[name].get("retrieval", {}).get("disagreement", 0),
            error_decomps[name].get("judging", {}).get("disagreement", 0),
        ]
        for name in names
    ], dtype=float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, 1.8 * len(names)), 8), sharex=True)

    # Top panel: attribution percentages (stacked)
    bottom = np.zeros(len(names), dtype=float)
    for si, (stage_label, color) in enumerate(zip(stage_labels, stage_colors)):
        vals = attr_vals[:, si]
        ax1.bar(x, vals, bottom=bottom, color=color, label=stage_label, alpha=0.9)
        for xi, v, btm in zip(x, vals, bottom):
            if v >= 8:
                ax1.text(xi, btm + v / 2, f"{v:.1f}%", ha="center", va="center",
                         fontsize=7, color="white", fontweight="bold")
        bottom += vals

    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Attribution (%)")
    ax1.set_title("Stage-wise Disagreement Attribution")
    ax1.legend(loc="upper right", ncol=3)

    # Bottom panel: raw disagreement scores
    width = 0.24
    for si, (stage_label, color) in enumerate(zip(stage_labels, stage_colors)):
        vals = dis_vals[:, si]
        offset = (si - 1) * width
        bars = ax2.bar(x + offset, vals, width, label=stage_label, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    ax2.set_ylabel("Disagreement score")
    ax2.set_title("Raw Stage Disagreement (1 - agreement)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=25, ha="right")
    ax2.set_ylim(0, min(1.0, max(0.2, float(np.nanmax(dis_vals)) * 1.25)))

    fig.suptitle("Error Decomposition Across Compared Runs", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_error_decomposition.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 16: Coverage heatmap + bar by model
# ---------------------------------------------------------------------------

def fig_coverage_by_model(benchmarks: Dict[str, Dict], out_dir: Path):
    coverage_metrics = [
        ("claim_attempt_coverage_rate", "Attempt"),
        ("claim_success_coverage_rate", "Success"),
        ("evidence_coverage_rate", "Evidence"),
        ("decisive_coverage_rate", "Decisive"),
        ("pair_failure_rate", "Pair Fail"),
    ]

    # Aggregate mean over conferences for heatmap.
    heat = np.full((len(MODELS), len(coverage_metrics)), np.nan, dtype=float)
    for mi, model in enumerate(MODELS):
        for ki, (metric_key, _) in enumerate(coverage_metrics):
            vals = []
            for conf in CONFERENCES:
                b = benchmarks.get(conf, {})
                cov = b.get("phase3_verification", {}).get("coverage_stats", {}).get(model, {})
                entry = cov.get(metric_key, {})
                val = entry.get("mean") if isinstance(entry, dict) else None
                if isinstance(val, (int, float)):
                    vals.append(float(val))
            if vals:
                heat[mi, ki] = float(np.mean(vals))

    if np.all(np.isnan(heat)):
        print("  [SKIP] No coverage_stats found in benchmark data", file=sys.stderr)
        return

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.15, 1.0]}
    )

    # Left: heatmap (model × metric)
    im = ax1.imshow(heat, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax1.set_xticks(range(len(coverage_metrics)))
    ax1.set_xticklabels([lbl for _, lbl in coverage_metrics], rotation=35, ha="right")
    ax1.set_yticks(range(len(MODELS)))
    ax1.set_yticklabels(MODEL_LABELS)
    ax1.set_title("Coverage Metrics (Mean Across Conferences)", fontweight="bold")

    for i in range(len(MODELS)):
        for j in range(len(coverage_metrics)):
            v = heat[i, j]
            if not np.isnan(v):
                color = "white" if v < 0.35 or v > 0.7 else "black"
                ax1.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=color)

    fig.colorbar(im, ax=ax1, shrink=0.9, pad=0.02, label="Rate")

    # Right: grouped bars for the most decision-relevant rates.
    bar_metrics = ["claim_success_coverage_rate", "decisive_coverage_rate", "pair_failure_rate"]
    bar_labels = ["Success", "Decisive", "Pair Fail"]
    x = np.arange(len(MODELS))
    width = 0.24

    for bi, (mk, bl) in enumerate(zip(bar_metrics, bar_labels)):
        vals = []
        for model in MODELS:
            per_conf = []
            for conf in CONFERENCES:
                b = benchmarks.get(conf, {})
                cov = b.get("phase3_verification", {}).get("coverage_stats", {}).get(model, {})
                entry = cov.get(mk, {})
                val = entry.get("mean") if isinstance(entry, dict) else None
                if isinstance(val, (int, float)):
                    per_conf.append(float(val))
            vals.append(float(np.mean(per_conf)) if per_conf else 0.0)
        ax2.bar(x + (bi - 1) * width, vals, width, label=bl, alpha=0.9)

    ax2.set_xticks(x)
    ax2.set_xticklabels(MODEL_LABELS, rotation=30, ha="right")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Rate")
    ax2.set_title("Key Coverage Rates by Model", fontweight="bold")
    ax2.legend(loc="upper right")

    fig.suptitle("Phase 3: Coverage Diagnostics by Model", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_coverage_by_model.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 17: Best-evidence Jaccard by pair and conference
# ---------------------------------------------------------------------------

def fig_best_evidence_jaccard(benchmarks: Dict[str, Dict], out_dir: Path):
    # Union of all pair keys that contain best_evidence_jaccard.
    pair_keys: List[str] = []
    seen = set()
    for conf in CONFERENCES:
        b = benchmarks.get(conf, {})
        pc = b.get("phase3_verification", {}).get("pairwise_claim_level", {})
        for pk, entry in pc.items():
            if isinstance(entry, dict) and isinstance(entry.get("best_evidence_jaccard"), dict):
                if pk not in seen:
                    pair_keys.append(pk)
                    seen.add(pk)

    if not pair_keys:
        print("  [SKIP] No best_evidence_jaccard data found", file=sys.stderr)
        return

    # Prefer human_vs_* rows first for readability.
    pair_keys = sorted(pair_keys, key=lambda p: (0 if p.startswith("human_vs_") else 1, p))

    mat = np.full((len(pair_keys), len(CONFERENCES)), np.nan, dtype=float)
    n_claims_mat = np.zeros((len(pair_keys), len(CONFERENCES)), dtype=int)

    for pi, pk in enumerate(pair_keys):
        for ci, conf in enumerate(CONFERENCES):
            b = benchmarks.get(conf, {})
            pc = b.get("phase3_verification", {}).get("pairwise_claim_level", {})
            bej = pc.get(pk, {}).get("best_evidence_jaccard", {})
            mean_val = bej.get("mean") if isinstance(bej, dict) else None
            n_val = bej.get("n_aligned_claims_with_best_evidence") if isinstance(bej, dict) else None
            if isinstance(mean_val, (int, float)):
                mat[pi, ci] = float(mean_val)
            if isinstance(n_val, int):
                n_claims_mat[pi, ci] = n_val

    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.35 * len(pair_keys) + 1.8)))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(CONFERENCES)))
    ax.set_xticklabels(CONF_LABELS)
    ax.set_yticks(range(len(pair_keys)))
    ax.set_yticklabels(pair_keys, fontsize=7)
    ax.set_title("Phase 3: Best-Evidence Overlap (Jaccard) by Pair and Conference", fontweight="bold")

    for i in range(len(pair_keys)):
        for j in range(len(CONFERENCES)):
            v = mat[i, j]
            if not np.isnan(v):
                color = "white" if v > 0.6 else "black"
                # show mean and sample size on two lines when space permits
                n_claims = n_claims_mat[i, j]
                txt = f"{v:.2f}\n(n={n_claims})" if n_claims > 0 else f"{v:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=6, color=color)

    fig.colorbar(im, ax=ax, label="Best-evidence Jaccard", shrink=0.9, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_best_evidence_jaccard.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize benchmark results.")
    parser.add_argument(
        "--bench-dir", type=Path,
        default=Path("output/full_conf_results/benchmarks/all_conferences"),
        help="Directory with benchmark JSON files",
    )
    parser.add_argument(
        "--ablation-dir", type=Path,
        default=Path("output/full_conf_results/benchmarks/ablation"),
        help="Directory with ablation JSON files",
    )
    parser.add_argument(
        "--out-dir", "-o", type=Path,
        default=Path("results/figures/multi_conf"),
        help="Output directory for figures",
    )
    parser.add_argument(
        "--error-decomp", type=Path, default=None,
        help="Optional error decomposition JSON file (or directory of JSON files)",
    )
    args = parser.parse_args()

    _setup_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading benchmark data...", file=sys.stderr)
    benchmarks = load_benchmarks(args.bench_dir)
    print(f"  Loaded {len(benchmarks)} conferences", file=sys.stderr)

    print("Loading ablation data...", file=sys.stderr)
    ablations = load_ablations(args.ablation_dir)
    print(f"  Loaded {len(ablations)} ablation entries", file=sys.stderr)

    print("Loading error decomposition data...", file=sys.stderr)
    error_decomps = load_error_decompositions(args.error_decomp, args.bench_dir)
    print(f"  Loaded {len(error_decomps)} error decomposition entries", file=sys.stderr)

    figures = [
        ("fig_claim_counts", fig_claim_counts, [benchmarks]),
        ("fig_stance_distributions", fig_stance_distributions, [benchmarks]),
        ("fig_retrieval_jaccard", fig_retrieval_jaccard, [benchmarks]),
        ("fig_agreement_heatmaps", fig_agreement_heatmaps, [benchmarks]),
        ("fig_label_distributions", fig_label_distributions, [benchmarks]),
        ("fig_multi_rater_agreement", fig_multi_rater, [benchmarks]),
        ("fig_claim_level_kappa", fig_claim_level_kappa, [benchmarks]),
        ("fig_paper_scores", fig_paper_scores, [benchmarks]),
        ("fig_summary_metrics", fig_summary_metrics, [benchmarks]),
        ("fig_claim_f1", fig_claim_f1, [benchmarks]),
        ("fig_core_task_similarity", fig_core_task_similarity, [benchmarks]),
        ("fig_claim_confusion_matrices", fig_claim_confusion_matrices, [benchmarks]),
        ("fig_coverage_by_model", fig_coverage_by_model, [benchmarks]),
        ("fig_best_evidence_jaccard", fig_best_evidence_jaccard, [benchmarks]),
        ("fig_error_decomposition", fig_error_decomposition, [error_decomps]),
        ("fig_ablation_policies", fig_ablation_policies, [ablations]),
        ("fig_ablation_per_conference", fig_ablation_per_conference, [ablations]),
    ]

    for name, func, extra_args in figures:
        print(f"  Generating {name}...", file=sys.stderr)
        try:
            func(*extra_args, args.out_dir)
        except Exception as e:
            print(f"  [ERROR] {name}: {e}", file=sys.stderr)

    print(f"\n✓ All figures saved to: {args.out_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
