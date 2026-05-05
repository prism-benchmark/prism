# -*- coding: utf-8 -*-
"""
compare_gemini_mimo_metrics.py
==============================
Tinh va so sanh DoA metrics (HM, R_Premise, Avg_Grounding_Score)
giua Gemini Flash Lite va Mimo v2.5 Pro tren 50 paper IDs moi conference.

Metrics:
  - R_Premise           : so premise / tong so ADUs (0~1)
  - Avg_Grounding_Score : trung binh grounding score cua premise (0~2)
  - DoA_HM (HM)         : harmonic mean cua R_Premise va (Avg_GS/2) (0~1)

Ket qua duoc tinh trung binh tren TAT CA conferences (khong phan chia).

Cach chay:
    python pipeline/compare_gemini_mimo_metrics.py
    python pipeline/compare_gemini_mimo_metrics.py --save_csv
    python pipeline/compare_gemini_mimo_metrics.py --conference ICLR2024
"""

import sys
import os
import json
import argparse
import statistics
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipeline.config as config

# ================================================================
#  CONFIG: Paths
# ================================================================

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Paper IDs 50 cho tung conference
PAPER_IDS_50 = {
    conf: config.paper_ids_file(conf, 50)
    for conf in ["ICLR2024", "ICLR2025", "ICLR2026", "ICML2025", "NeurIPS2025"]
}

# Folder mapping cho tung source x conference
# Key = (evaluator, source_type, conference)
# Value = subfolder name trong OUTPUT_DIR

FOLDER_MAP = {
    # ---- GEMINI Human ----
    ("gemini", "human", "ICLR2024"):    "human_iclr2024",
    ("gemini", "human", "ICLR2025"):    "human_iclr2025",
    ("gemini", "human", "ICLR2026"):    "human_iclr2026",
    ("gemini", "human", "ICML2025"):    "human_icml2025",
    ("gemini", "human", "NeurIPS2025"): "human_neurips2025",

    # ---- MIMO Human ----
    ("mimo", "human", "ICLR2024"):    "human_mimo_iclr2024",
    ("mimo", "human", "ICLR2025"):    "human_mimo_iclr2025",
    ("mimo", "human", "ICLR2026"):    "human_mimo_iclr2026",
    ("mimo", "human", "ICML2025"):    "human_mimo_icml2025",
    ("mimo", "human", "NeurIPS2025"): "human_mimo_neurips2025",

    # ---- GEMINI LLM: sea ----
    ("gemini", "sea", "ICLR2024"):    "sea_iclr2024",
    ("gemini", "sea", "ICLR2025"):    "sea_iclr2025",
    ("gemini", "sea", "ICLR2026"):    "sea_iclr2026",
    ("gemini", "sea", "ICML2025"):    "sea_icml2025",
    ("gemini", "sea", "NeurIPS2025"): "sea_neurlps2025",

    # ---- MIMO LLM: sea ----
    ("mimo", "sea", "ICLR2024"):    "mimo_sea_iclr2024",
    ("mimo", "sea", "ICLR2025"):    "mimo_sea_iclr2025",
    ("mimo", "sea", "ICLR2026"):    "mimo_sea_iclr2026",
    ("mimo", "sea", "ICML2025"):    "mimo_sea_icml2025",
    ("mimo", "sea", "NeurIPS2025"): "mimo_sea_neurlps2025",

    # ---- GEMINI LLM: tree ----
    ("gemini", "tree", "ICLR2024"):    "tree_iclr2024",
    ("gemini", "tree", "ICLR2025"):    "tree_iclr2025",
    ("gemini", "tree", "ICLR2026"):    "tree_iclr2026",
    ("gemini", "tree", "ICML2025"):    "tree_icml2025",
    ("gemini", "tree", "NeurIPS2025"): "tree_neurips2025",

    # ---- MIMO LLM: tree ----
    ("mimo", "tree", "ICLR2024"):    "mimo_tree_iclr2024",
    ("mimo", "tree", "ICLR2025"):    "mimo_tree_iclr2025",
    ("mimo", "tree", "ICLR2026"):    "mimo_tree_iclr2026",
    ("mimo", "tree", "ICML2025"):    "mimo_tree_icml2025",
    ("mimo", "tree", "NeurIPS2025"): "mimo_tree_neurips2025",

    # ---- GEMINI LLM: reviewer2 ----
    ("gemini", "reviewer2", "ICLR2024"):    "reviewer2_iclr2024",
    ("gemini", "reviewer2", "ICLR2025"):    "reviewer2_iclr2025",
    ("gemini", "reviewer2", "ICLR2026"):    "reviewer2_iclr2026",
    ("gemini", "reviewer2", "ICML2025"):    "reviewer2_icml2025",
    ("gemini", "reviewer2", "NeurIPS2025"): "reviewer2_neurips2025",

    # ---- MIMO LLM: reviewer2 ----
    ("mimo", "reviewer2", "ICLR2024"):    "mimo_reviewer2_iclr2024",
    ("mimo", "reviewer2", "ICLR2025"):    "mimo_reviewer2_iclr2025",
    ("mimo", "reviewer2", "ICLR2026"):    "mimo_reviewer2_iclr2026",
    ("mimo", "reviewer2", "ICML2025"):    "mimo_reviewer2_icml2025",
    ("mimo", "reviewer2", "NeurIPS2025"): "mimo_reviewer2_neurips2025",

    # ---- GEMINI LLM: deepreview ----
    ("gemini", "deepreview", "ICLR2024"):    "deepreview_iclr2024",
    ("gemini", "deepreview", "ICLR2025"):    "deepreview_iclr2025",
    ("gemini", "deepreview", "ICLR2026"):    "deepreview_iclr2026",
    ("gemini", "deepreview", "ICML2025"):    "deepreview_icml2025",
    ("gemini", "deepreview", "NeurIPS2025"): "deepreview_neurips2025",

    # ---- MIMO LLM: deepreview ----
    ("mimo", "deepreview", "ICLR2024"):    "mimo_deepreview_iclr2024",
    ("mimo", "deepreview", "ICLR2025"):    "mimo_deepreview_iclr2025",
    ("mimo", "deepreview", "ICLR2026"):    "mimo_deepreview_iclr2026",
    ("mimo", "deepreview", "ICML2025"):    "mimo_deepreview_icml2025",
    ("mimo", "deepreview", "NeurIPS2025"): "mimo_deepreview_neurips2025",

    # ---- GEMINI LLM: cyclereview ----
    ("gemini", "cyclereview", "ICLR2024"):    "cyclereview_iclr2024",
    ("gemini", "cyclereview", "ICLR2025"):    "cyclereview_iclr2025",
    ("gemini", "cyclereview", "ICLR2026"):    "cyclereview_iclr2026",
    ("gemini", "cyclereview", "ICML2025"):    "cyclereview_icml2025",
    ("gemini", "cyclereview", "NeurIPS2025"): "cyclereview_neurlps2025",

    # ---- MIMO LLM: cyclereview ----
    ("mimo", "cyclereview", "ICLR2024"):    "mimo_cyclereview_iclr2024",
    ("mimo", "cyclereview", "ICLR2025"):    "mimo_cyclereview_iclr2025",
    ("mimo", "cyclereview", "ICLR2026"):    "mimo_cyclereview_iclr2026",
    ("mimo", "cyclereview", "ICML2025"):    "mimo_cyclereview_icml2025",
    ("mimo", "cyclereview", "NeurIPS2025"): "mimo_cyclereview_neurlps2025",

    # ---- P23MIMO Human (Gemini Seg + Mimo Clf) ----
    ("p23mimo", "human", "ICLR2024"):    "p23mimo_human_iclr2024",
    ("p23mimo", "human", "ICLR2025"):    "p23mimo_human_iclr2025",
    ("p23mimo", "human", "ICLR2026"):    "p23mimo_human_iclr2026",
    ("p23mimo", "human", "ICML2025"):    "p23mimo_human_icml2025",
    ("p23mimo", "human", "NeurIPS2025"): "p23mimo_human_neurips2025",

    # ---- P23MIMO LLM: sea ----
    ("p23mimo", "sea", "ICLR2024"):    "p23mimo_sea_iclr2024",
    ("p23mimo", "sea", "ICLR2025"):    "p23mimo_sea_iclr2025",
    ("p23mimo", "sea", "ICLR2026"):    "p23mimo_sea_iclr2026",
    ("p23mimo", "sea", "ICML2025"):    "p23mimo_sea_icml2025",
    ("p23mimo", "sea", "NeurIPS2025"): "p23mimo_sea_neurlps2025",

    # ---- P23MIMO LLM: tree ----
    ("p23mimo", "tree", "ICLR2024"):    "p23mimo_tree_iclr2024",
    ("p23mimo", "tree", "ICLR2025"):    "p23mimo_tree_iclr2025",
    ("p23mimo", "tree", "ICLR2026"):    "p23mimo_tree_iclr2026",
    ("p23mimo", "tree", "ICML2025"):    "p23mimo_tree_icml2025",
    ("p23mimo", "tree", "NeurIPS2025"): "p23mimo_tree_neurips2025",

    # ---- P23MIMO LLM: reviewer2 ----
    ("p23mimo", "reviewer2", "ICLR2024"):    "p23mimo_reviewer2_iclr2024",
    ("p23mimo", "reviewer2", "ICLR2025"):    "p23mimo_reviewer2_iclr2025",
    ("p23mimo", "reviewer2", "ICLR2026"):    "p23mimo_reviewer2_iclr2026",
    ("p23mimo", "reviewer2", "ICML2025"):    "p23mimo_reviewer2_icml2025",
    ("p23mimo", "reviewer2", "NeurIPS2025"): "p23mimo_reviewer2_neurips2025",

    # ---- P23MIMO LLM: deepreview ----
    ("p23mimo", "deepreview", "ICLR2024"):    "p23mimo_deepreview_iclr2024",
    ("p23mimo", "deepreview", "ICLR2025"):    "p23mimo_deepreview_iclr2025",
    ("p23mimo", "deepreview", "ICLR2026"):    "p23mimo_deepreview_iclr2026",
    ("p23mimo", "deepreview", "ICML2025"):    "p23mimo_deepreview_icml2025",
    ("p23mimo", "deepreview", "NeurIPS2025"): "p23mimo_deepreview_neurips2025",

    # ---- P23MIMO LLM: cyclereview ----
    ("p23mimo", "cyclereview", "ICLR2024"):    "p23mimo_cyclereview_iclr2024",
    ("p23mimo", "cyclereview", "ICLR2025"):    "p23mimo_cyclereview_iclr2025",
    ("p23mimo", "cyclereview", "ICLR2026"):    "p23mimo_cyclereview_iclr2026",
    ("p23mimo", "cyclereview", "ICML2025"):    "p23mimo_cyclereview_icml2025",
    ("p23mimo", "cyclereview", "NeurIPS2025"): "p23mimo_cyclereview_neurlps2025",
}

SOURCE_TYPES = ["human", "sea", "tree", "reviewer2", "deepreview", "cyclereview"]
EVALUATORS   = ["gemini", "mimo", "p23mimo"]
CONFERENCES  = ["ICLR2024", "ICLR2025", "ICLR2026", "ICML2025", "NeurIPS2025"]

EVALUATOR_LABELS = {
    "gemini":  "Gemini Flash Lite (All 3 phases)",
    "mimo":    "Mimo v2.5 Pro (All 3 phases)",
    "p23mimo": "Gemini Seg + Mimo Clf (P1:Gemini, P2+P3:Mimo)",
}


# ================================================================
#  Helpers
# ================================================================

def load_paper_ids(path: str) -> set:
    if not os.path.exists(path):
        print(f"  [WARN] Paper IDs file not found: {path}")
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def compute_paper_metrics(arguments: list) -> dict:
    """
    Tinh R_Premise, Avg_Grounding_Score, DoA_HM cho 1 reviewer/paper.

    - R_Premise           = #premises / #total_adus
    - Avg_Grounding_Score = mean(grounding_score) for premises, range [0,2]
    - S_depth_norm        = Avg_Grounding_Score / 2, range [0,1]
    - DoA_HM              = harmonic_mean(R_Premise, S_depth_norm) range [0,1]
    """
    if not arguments:
        return None

    total       = len(arguments)
    premises    = [a for a in arguments if a.get("role") == "Premise"]
    n_premises  = len(premises)
    r_premise   = n_premises / total if total > 0 else 0.0

    grounding_scores = [
        p["grounding_score"] for p in premises
        if p.get("grounding_score") is not None
    ]
    avg_gs      = statistics.mean(grounding_scores) if grounding_scores else 0.0
    s_norm      = avg_gs / 2.0   # normalize to [0,1]

    if r_premise > 0 and s_norm > 0:
        doa_hm = 2 * (r_premise * s_norm) / (r_premise + s_norm)
    else:
        doa_hm = 0.0

    return {
        "r_premise":           round(r_premise, 6),
        "avg_grounding_score": round(avg_gs,     6),
        "doa_hm":              round(doa_hm,     6),
        "n_args":              total,
        "n_premises":          n_premises,
    }


def load_folder_metrics(folder_path: str, paper_ids: set) -> list:
    """
    Doc tat ca JSON trong folder, filter theo paper_ids,
    tra ve list dict metrics (1 entry = 1 paper, trung binh qua reviewers).
    """
    if not os.path.isdir(folder_path):
        return []

    results = []
    for fname in os.listdir(folder_path):
        if not fname.endswith(".json"):
            continue
        paper_id = fname[:-5]
        if paper_ids and paper_id not in paper_ids:
            continue

        fpath = os.path.join(folder_path, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        reviews_analysis = data.get("reviews_analysis", {})
        if not reviews_analysis:
            continue

        # Trung binh qua tat ca reviewers trong 1 paper
        paper_r_premise    = []
        paper_avg_gs       = []
        paper_doa_hm       = []
        paper_n_args       = []
        paper_n_premises   = []

        for reviewer_id, arguments in reviews_analysis.items():
            if not arguments:
                continue
            m = compute_paper_metrics(arguments)
            if m is None:
                continue
            paper_r_premise.append(m["r_premise"])
            paper_avg_gs.append(m["avg_grounding_score"])
            paper_doa_hm.append(m["doa_hm"])
            paper_n_args.append(m["n_args"])
            paper_n_premises.append(m["n_premises"])

        if not paper_r_premise:
            continue

        results.append({
            "paper_id":           paper_id,
            "r_premise":          statistics.mean(paper_r_premise),
            "avg_grounding_score": statistics.mean(paper_avg_gs),
            "doa_hm":             statistics.mean(paper_doa_hm),
            "n_args":             statistics.mean(paper_n_args),
            "n_premises":         statistics.mean(paper_n_premises),
        })

    return results


def aggregate(paper_results: list) -> dict:
    """Tinh trung binh metrics tu list paper results."""
    if not paper_results:
        return None

    def avg(key):
        vals = [p[key] for p in paper_results]
        return round(statistics.mean(vals), 4)

    return {
        "r_premise":           avg("r_premise"),
        "avg_grounding_score": avg("avg_grounding_score"),
        "doa_hm":              avg("doa_hm"),
        "avg_n_args":          round(statistics.mean([p["n_args"]     for p in paper_results]), 1),
        "avg_n_premises":      round(statistics.mean([p["n_premises"] for p in paper_results]), 1),
        "n_papers":            len(paper_results),
    }


# ================================================================
#  Main computation
# ================================================================

def compute_all_metrics(filter_conference: str = None, verbose: bool = True) -> dict:
    """
    Tinh metrics cho tat ca (evaluator, source_type) combinations.
    Tra ve dict: {(evaluator, source_type): aggregated_metrics}
    """
    conferences = [filter_conference] if filter_conference else CONFERENCES

    # Load paper IDs cho tung conference
    ids_by_conf = {}
    for conf in conferences:
        ids_by_conf[conf] = load_paper_ids(PAPER_IDS_50[conf])

    # Collect per conference results per (evaluator, source_type)
    all_papers = defaultdict(list)   # key = (evaluator, source_type)

    for evaluator in EVALUATORS:
        for src in SOURCE_TYPES:
            for conf in conferences:
                key   = (evaluator, src, conf)
                folder_name = FOLDER_MAP.get(key)
                if folder_name is None:
                    continue

                folder_path = os.path.join(OUTPUT_DIR, folder_name)
                paper_ids   = ids_by_conf[conf]
                papers      = load_folder_metrics(folder_path, paper_ids)

                agg_key = (evaluator, src)
                all_papers[agg_key].extend(papers)

                if verbose and papers:
                    print(f"  [{evaluator:6}][{src:12}][{conf:11}] "
                          f"folder={folder_name:35} papers={len(papers):3}")

    # Aggregate
    results = {}
    for (evaluator, src), papers in all_papers.items():
        results[(evaluator, src)] = aggregate(papers)

    return results


# ================================================================
#  Display
# ================================================================

def print_comparison_table(results: dict):
    """In bang so sanh dep."""
    W_SRC  = 14
    W_VAL  = 12
    W_PAP  = 8

    sep     = "+" + "-"*(W_SRC+2) + "+" + ("-"*(W_VAL+2)+"+")*3 + "-"*(W_PAP+2) + "+"
    hdr_fmt = "| {:<{}} | {:>{}} | {:>{}} | {:>{}} | {:>{}} |"
    row_fmt = "| {:<{}} | {:>{}.4f} | {:>{}.4f} | {:>{}.4f} | {:>{}d} |"

    def section(evaluator_label):
        print(f"\n  Evaluator: {evaluator_label}")
        print("  " + sep)
        print("  " + hdr_fmt.format(
            "Source", W_SRC,
            "R_Premise", W_VAL,
            "Avg_GS(0-2)", W_VAL,
            "DoA_HM", W_VAL,
            "N_papers", W_PAP,
        ))
        print("  " + sep)

    def row(src_label, m):
        if m is None:
            print(f"  | {src_label:<{W_SRC}} | {'N/A':>{W_VAL}} | {'N/A':>{W_VAL}} | {'N/A':>{W_VAL}} | {'N/A':>{W_PAP}} |")
        else:
            print("  " + row_fmt.format(
                src_label,   W_SRC,
                m["r_premise"],           W_VAL,
                m["avg_grounding_score"], W_VAL,
                m["doa_hm"],              W_VAL,
                m["n_papers"],            W_PAP,
            ))

    print("\n" + "="*80)
    print("  DoA METRICS COMPARISON: Gemini vs Mimo vs Gemini-Seg+Mimo-Clf")
    print("  (Averaged across ALL conferences, filtered by 50 paper IDs each)")
    print("="*80)

    for evaluator in EVALUATORS:
        label = EVALUATOR_LABELS.get(evaluator, evaluator)
        section(label)
        for src in SOURCE_TYPES:
            m = results.get((evaluator, src))
            row(src, m)
        print("  " + sep)

    # Delta table vs Gemini baseline
    print("\n" + "="*80)
    print("  DELTA vs Gemini baseline  (positive = higher than Gemini)")
    print("="*80)
    W_DELTA = 11
    sep2    = "+" + "-"*(W_SRC+2) + "+" + ("-"*(W_DELTA+2)+"+")*6 + "+"
    hdr2    = ("| {:<{}} |" + " {:>{}} | {:>{}} | {:>{}} |"*2).format(
        "Source", W_SRC,
        "Mimo R_Pr", W_DELTA, "Mimo GS", W_DELTA, "Mimo HM", W_DELTA,
        "P23 R_Pr",  W_DELTA, "P23 GS",  W_DELTA, "P23 HM",  W_DELTA,
    )
    print("  " + sep2)
    print("  " + hdr2)
    print("  " + sep2)

    for src in SOURCE_TYPES:
        mg  = results.get(("gemini",  src))
        mm  = results.get(("mimo",    src))
        mp  = results.get(("p23mimo", src))
        def d(a, b, k):
            if a and b: return f"{b[k]-a[k]:>+{W_DELTA}.4f}"
            return f"{'N/A':>{W_DELTA}}"
        print(f"  | {src:<{W_SRC}} | {d(mg,mm,'r_premise')} | {d(mg,mm,'avg_grounding_score')} | {d(mg,mm,'doa_hm')} | {d(mg,mp,'r_premise')} | {d(mg,mp,'avg_grounding_score')} | {d(mg,mp,'doa_hm')} |")

    print("  " + sep2)


def draw_bar_chart(results: dict, out_path: str, conference_label: str = "All Conferences",
                   evaluators_to_plot: list = None):
    """
    Ve bieu do cot 3 subplots (R_Premise, Avg_GS, DoA_HM).
    evaluators_to_plot: list subset cua ["gemini","mimo","p23mimo"], mac dinh tat ca co data.
    """
    if not HAS_MPL:
        print("  [WARN] matplotlib not installed. Skip chart.")
        return

    ALL_STYLES = {
        "gemini":  {"color": "#4C72B0", "hatch": "",   "label": "Gemini Flash Lite (all P1-3)"},
        "mimo":    {"color": "#DD8452", "hatch": "//", "label": "Mimo v2.5 Pro (all P1-3)"},
        "p23mimo": {"color": "#55A868", "hatch": "..", "label": "Gemini Seg + Mimo Clf (P1:G, P2+3:M)"},
    }

    # Determine which evaluators to include
    if evaluators_to_plot is None:
        evaluators_to_plot = [ev for ev in ["gemini", "mimo", "p23mimo"]
                              if any(results.get((ev, src)) for src in SOURCE_TYPES)]

    n_ev   = len(evaluators_to_plot)
    width  = 0.8 / n_ev          # spread bars evenly
    offsets = [(i - (n_ev - 1) / 2) * width for i in range(n_ev)]

    metrics_cfg = [
        ("r_premise",           "R_Premise",           "Ratio (0-1)"),
        ("avg_grounding_score", "Avg Grounding Score", "Score (0-2)"),
        ("doa_hm",              "DoA HM",              "Score (0-1)"),
    ]

    sources    = SOURCE_TYPES
    src_labels = ["Human", "SEA", "Tree", "Reviewer2", "DeepReview", "CycleReview"]
    x          = np.arange(len(sources))

    fig, axes = plt.subplots(1, 3, figsize=(7 * len(metrics_cfg), 6))
    fig.suptitle(
        f"DoA Metrics: {' vs '.join(ALL_STYLES[ev]['label'].split('(')[0].strip() for ev in evaluators_to_plot)}\n({conference_label})",
        fontsize=13, fontweight="bold", y=1.02,
    )

    for ax, (metric_key, metric_label, y_label) in zip(axes, metrics_cfg):
        all_vals = []
        for i, ev in enumerate(evaluators_to_plot):
            st   = ALL_STYLES[ev]
            vals = [results[(ev, src)][metric_key] if results.get((ev, src)) else 0.0
                    for src in sources]
            all_vals.extend(vals)

            bars = ax.bar(x + offsets[i], vals, width,
                          label=st["label"],
                          color=st["color"], hatch=st["hatch"],
                          edgecolor="white", linewidth=0.4, alpha=0.92)

            for bar in bars:
                h = bar.get_height()
                if h > 0.001:
                    ax.text(bar.get_x() + bar.get_width() / 2., h + 0.003,
                            f"{h:.3f}", ha="center", va="bottom",
                            fontsize=7, color=st["color"], fontweight="bold",
                            rotation=90 if n_ev >= 3 else 0)

        ax.set_title(metric_label, fontsize=12, fontweight="bold")
        ax.set_ylabel(y_label, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(src_labels, fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_ylim(0, max(max(all_vals, default=0) * 1.35, 0.1))
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved -> {out_path}")


def save_csv(results: dict, out_path: str):
    """Luu ket qua ra CSV."""
    import csv
    rows = []
    for (evaluator, src), m in sorted(results.items()):
        if m is None:
            continue
        rows.append({
            "evaluator":           evaluator,
            "source":              src,
            "r_premise":           m["r_premise"],
            "avg_grounding_score": m["avg_grounding_score"],
            "doa_hm":              m["doa_hm"],
            "avg_n_args":          m["avg_n_args"],
            "avg_n_premises":      m["avg_n_premises"],
            "n_papers":            m["n_papers"],
        })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  CSV saved -> {out_path}")


# ================================================================
#  Entry Point
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare DoA metrics: Gemini vs Mimo across all conferences"
    )
    parser.add_argument(
        "--conference", type=str, default=None,
        choices=CONFERENCES,
        help="Chi tinh cho 1 conference. Mac dinh: tat ca 5 conferences."
    )
    parser.add_argument(
        "--save_csv", action="store_true",
        help="Luu ket qua ra CSV."
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Ve bieu do cot so sanh (tat ca evaluators co data)."
    )
    parser.add_argument(
        "--plot_2", action="store_true",
        help="Ve bieu do CHI CO Gemini vs Mimo (all 3 phases), bo qua p23mimo."
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Tat verbose loading log."
    )
    args = parser.parse_args()

    print("\nLoading results...")
    results = compute_all_metrics(
        filter_conference=args.conference,
        verbose=not args.quiet,
    )

    print_comparison_table(results)

    if args.save_csv:
        csv_name = f"gemini_vs_mimo_metrics{'_' + args.conference if args.conference else '_all'}.csv"
        csv_path = os.path.join(OUTPUT_DIR, csv_name)
        save_csv(results, csv_path)

    if args.plot:
        chart_name = f"gemini_vs_mimo_barchart{'_' + args.conference if args.conference else '_all'}.png"
        chart_path = os.path.join(OUTPUT_DIR, chart_name)
        conf_label = args.conference if args.conference else "All Conferences (ICLR2024/25/26 + ICML2025 + NeurIPS2025)"
        draw_bar_chart(results, chart_path, conf_label)

    if args.plot_2:
        chart_name = f"gemini_vs_mimo_2ev{'_' + args.conference if args.conference else '_all'}.png"
        chart_path = os.path.join(OUTPUT_DIR, chart_name)
        conf_label = args.conference if args.conference else "All Conferences (ICLR2024/25/26 + ICML2025 + NeurIPS2025)"
        draw_bar_chart(results, chart_path, conf_label, evaluators_to_plot=["gemini", "mimo"])

    # Also save JSON summary
    summary_path = os.path.join(
        OUTPUT_DIR,
        f"gemini_vs_mimo_summary{'_' + args.conference if args.conference else '_all'}.json"
    )
    clean = {f"{ev}_{src}": m for (ev, src), m in results.items() if m}
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
    print(f"\n  JSON summary saved -> {summary_path}")





