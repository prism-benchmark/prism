"""
evaluate_results.py — Analyse constructiveness results for Human vs. SEA vs. Reviewer2.

Loads:
    output/iclr2024/human/all_results_lite.jsonl
    output/iclr2024/sea/all_results_lite.jsonl
    output/iclr2024/reviewer2/all_results_lite.jsonl   (optional)

Produces:
    output/iclr2024/analysis/paired_records.jsonl           — human ↔ SEA per-paper
    output/iclr2024/analysis/paired_records_reviewer2.jsonl — human ↔ Reviewer2 per-paper
    output/iclr2024/analysis/summary_metrics_3way.csv       — mean ± std all 3 groups
    output/iclr2024/analysis/statistical_report.json        — Human vs SEA Wilcoxon+Cliff's
    output/iclr2024/analysis/statistical_report_reviewer2.json — Human vs Reviewer2
    output/iclr2024/analysis/report.txt                     — human-readable combined report

Usage:
    python evaluate_results.py
    python evaluate_results.py --no-stats
    python evaluate_results.py --no-reviewer2    # skip reviewer2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Optional

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE             = os.path.dirname(os.path.abspath(__file__))
_OUT_ROOT         = os.path.join(_HERE, "output", "iclr2024")
HUMAN_JSONL       = os.path.join(_OUT_ROOT, "human",      "all_results_lite.jsonl")
SEA_JSONL         = os.path.join(_OUT_ROOT, "sea",        "all_results_lite.jsonl")
REVIEWER2_JSONL   = os.path.join(_OUT_ROOT, "reviewer2",  "all_results_lite.jsonl")
DEEPREVIEW_JSONL  = os.path.join(_OUT_ROOT, "deepreview", "all_results_lite.jsonl")
TREE_JSONL        = os.path.join(_OUT_ROOT, "tree",       "all_results_lite.jsonl")
ANALYSIS_DIR      = os.path.join(_OUT_ROOT, "analysis")

PAIRED_OUT        = os.path.join(ANALYSIS_DIR, "paired_records.jsonl")
PAIRED_R2_OUT     = os.path.join(ANALYSIS_DIR, "paired_records_reviewer2.jsonl")
PAIRED_DR_OUT     = os.path.join(ANALYSIS_DIR, "paired_records_deepreview.jsonl")
PAIRED_TREE_OUT   = os.path.join(ANALYSIS_DIR, "paired_records_tree.jsonl")
CSV_OUT           = os.path.join(ANALYSIS_DIR, "summary_metrics_5way.csv")
STAT_OUT          = os.path.join(ANALYSIS_DIR, "statistical_report.json")
STAT_R2_OUT       = os.path.join(ANALYSIS_DIR, "statistical_report_reviewer2.json")
STAT_DR_OUT       = os.path.join(ANALYSIS_DIR, "statistical_report_deepreview.json")
STAT_TREE_OUT     = os.path.join(ANALYSIS_DIR, "statistical_report_tree.json")
REPORT_OUT        = os.path.join(ANALYSIS_DIR, "report.txt")

sys.path.insert(0, _HERE)
from src.metrics import compute_paper_comparison
from src.statistical import (
    METRIC_KEYS,
    run_full_analysis,
    run_subgroup_analysis,
)

# ── Metric display names ───────────────────────────────────────────────────────
METRIC_LABELS = {
    "MCS":                     "Mean Constructiveness Score",
    "AR":                      "Actionability Ratio",
    "SD":                      "Solution Density",
    "CD":                      "Constructiveness Density",
    "D1_actionability_mean":   "D1 Actionability",
    "D2_specificity_mean":     "D2 Specificity",
    "D3_justification_mean":   "D3 Justification",
    "D4_solution_mean":        "D4 Solution",
    "D5_tone_mean":            "D5 Tone",
}


# ══════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_jsonl(path: str) -> list[dict]:
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def save_jsonl(path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _safe(v: Any, decimals: int = 4) -> Any:
    if v is None:
        return None
    if isinstance(v, float):
        return round(v, decimals)
    return v


# ══════════════════════════════════════════════════════════════════════════════
# Build paired records  (human + any single-LLM reviewer merged per paper)
# ══════════════════════════════════════════════════════════════════════════════

def _build_paired_generic(
    human_records: list[dict],
    llm_records: list[dict],
) -> list[dict]:
    """Generic function: merge human records with any single-LLM records by paper_id.

    Both SEA and Reviewer2 have the same output schema
    (paper-level record with a top-level ``metrics`` field).
    """
    llm_by_id = {r["paper_id"]: r for r in llm_records if r.get("paper_id")}
    paired = []

    for h_rec in human_records:
        pid = h_rec.get("paper_id")
        if not pid:
            continue
        l_rec = llm_by_id.get(pid)
        if l_rec is None:
            continue  # no matching LLM record

        # Human: collect per-reviewer metrics (skip empty)
        human_metrics_list = []
        for rev in h_rec.get("reviewers", []):
            m = rev.get("metrics")
            if m and (m.get("n_arcs") or 0) > 0:
                human_metrics_list.append(m)

        # LLM metrics
        llm_metrics = l_rec.get("metrics")
        llm_valid   = llm_metrics is not None and (llm_metrics.get("n_arcs") or 0) > 0

        if not human_metrics_list and not llm_valid:
            continue

        comparison = compute_paper_comparison(human_metrics_list, llm_metrics)

        paired.append({
            "paper_id":               pid,
            "metadata":               h_rec.get("metadata", {}),
            "n_human_reviewers_valid": len(human_metrics_list),
            "llm_valid":              llm_valid,
            "human_avg_metrics":      _avg_metrics(human_metrics_list),
            "llm_metrics":            llm_metrics,
            "comparison":             comparison,
        })

    return paired


def build_paired_records(
    human_records: list[dict],
    sea_records:   list[dict],
) -> list[dict]:
    """Human ↔ SEA pairing (backward-compatible wrapper)."""
    return _build_paired_generic(human_records, sea_records)


def build_reviewer2_paired_records(
    human_records:     list[dict],
    reviewer2_records: list[dict],
) -> list[dict]:
    """Human ↔ Reviewer2 pairing."""
    return _build_paired_generic(human_records, reviewer2_records)


def _avg_metrics(metrics_list: list[dict]) -> Optional[dict]:
    if not metrics_list:
        return None
    keys = [k for k in metrics_list[0] if isinstance(metrics_list[0][k], (int, float))]
    result = {}
    for k in keys:
        vals = [m[k] for m in metrics_list if m.get(k) is not None]
        result[k] = round(sum(vals) / len(vals), 4) if vals else None
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Aggregate statistics  (3-way: human / sea / reviewer2)
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_stats_3way(
    sea_paired:    list[dict],
    r2_paired:     list[dict] | None = None,
    dr_paired:     list[dict] | None = None,
    tree_paired:   list[dict] | None = None,
) -> dict[str, dict]:
    """Compute mean ± std for Human_avg / SEA / Reviewer2 / DeepReview / Tree across all metrics."""
    human_vals: dict[str, list] = {k: [] for k in METRIC_KEYS}
    sea_vals:   dict[str, list] = {k: [] for k in METRIC_KEYS}
    r2_vals:    dict[str, list] = {k: [] for k in METRIC_KEYS}
    dr_vals:    dict[str, list] = {k: [] for k in METRIC_KEYS}
    tree_vals:  dict[str, list] = {k: [] for k in METRIC_KEYS}

    for rec in sea_paired:
        cmp = rec.get("comparison", {})
        for k in METRIC_KEYS:
            h = cmp.get(f"human_avg_{k}")
            s = cmp.get(f"llm_{k}")
            if h is not None:
                human_vals[k].append(h)
            if s is not None:
                sea_vals[k].append(s)

    if r2_paired:
        for rec in r2_paired:
            cmp = rec.get("comparison", {})
            for k in METRIC_KEYS:
                r = cmp.get(f"llm_{k}")
                if r is not None:
                    r2_vals[k].append(r)

    if dr_paired:
        for rec in dr_paired:
            cmp = rec.get("comparison", {})
            for k in METRIC_KEYS:
                d = cmp.get(f"llm_{k}")
                if d is not None:
                    dr_vals[k].append(d)

    if tree_paired:
        for rec in tree_paired:
            cmp = rec.get("comparison", {})
            for k in METRIC_KEYS:
                t = cmp.get(f"llm_{k}")
                if t is not None:
                    tree_vals[k].append(t)

    stats = {}
    for k in METRIC_KEYS:
        h    = np.array(human_vals[k]) if human_vals[k] else np.array([])
        s    = np.array(sea_vals[k])   if sea_vals[k]   else np.array([])
        r2   = np.array(r2_vals[k])    if r2_vals[k]    else np.array([])
        dr   = np.array(dr_vals[k])    if dr_vals[k]    else np.array([])
        tree = np.array(tree_vals[k])  if tree_vals[k]  else np.array([])

        def _stat(arr: np.ndarray, prefix: str) -> dict:
            if len(arr) == 0:
                return {f"{prefix}_n": 0, f"{prefix}_mean": None,
                        f"{prefix}_std": None, f"{prefix}_min": None,
                        f"{prefix}_max": None}
            return {
                f"{prefix}_n":    len(arr),
                f"{prefix}_mean": round(float(np.mean(arr)), 4),
                f"{prefix}_std":  round(float(np.std(arr)),  4),
                f"{prefix}_min":  round(float(np.min(arr)),  4),
                f"{prefix}_max":  round(float(np.max(arr)),  4),
            }

        row = {
            **_stat(h,    "human"),
            **_stat(s,    "sea"),
            **_stat(r2,   "reviewer2"),
            **_stat(dr,   "deepreview"),
            **_stat(tree, "tree"),
        }

        row["sea_delta"]        = (round(float(np.mean(s))    - float(np.mean(h)), 4)
                                   if len(h) and len(s)    else None)
        row["reviewer2_delta"]  = (round(float(np.mean(r2))   - float(np.mean(h)), 4)
                                   if len(h) and len(r2)   else None)
        row["deepreview_delta"] = (round(float(np.mean(dr))   - float(np.mean(h)), 4)
                                   if len(h) and len(dr)   else None)
        row["tree_delta"]       = (round(float(np.mean(tree)) - float(np.mean(h)), 4)
                                   if len(h) and len(tree) else None)
        stats[k] = row
    return stats


# backward-compat alias
def aggregate_stats(paired: list[dict]) -> dict[str, dict]:
    return aggregate_stats_3way(paired, r2_paired=None)


def save_csv_3way(stats: dict[str, dict], path: str) -> None:
    """Save 5-way comparison to CSV (human / sea / reviewer2 / deepreview / tree)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = [
        "metric", "label",
        "human_n",      "human_mean",      "human_std",      "human_min",      "human_max",
        "sea_n",        "sea_mean",        "sea_std",        "sea_min",        "sea_max",        "sea_delta",
        "reviewer2_n",  "reviewer2_mean",  "reviewer2_std",  "reviewer2_min",  "reviewer2_max",  "reviewer2_delta",
        "deepreview_n", "deepreview_mean", "deepreview_std", "deepreview_min", "deepreview_max", "deepreview_delta",
        "tree_n",       "tree_mean",       "tree_std",       "tree_min",       "tree_max",       "tree_delta",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for k, row in stats.items():
            w.writerow({"metric": k, "label": METRIC_LABELS.get(k, k), **row})


# backward compat
def save_csv(stats: dict[str, dict], path: str) -> None:
    save_csv_3way(stats, path)


# ══════════════════════════════════════════════════════════════════════════════
# Human-readable report
# ══════════════════════════════════════════════════════════════════════════════

_SEP  = "=" * 72
_SEP2 = "-" * 72

def _fmt(v: Any, w: int = 8) -> str:
    if v is None:
        return "N/A".rjust(w)
    return f"{v:.4f}".rjust(w)

def _sig_marker(sig: bool, p: float) -> str:
    if not sig:
        return "  ns"
    if p < 0.001:
        return " ***"
    if p < 0.01:
        return "  **"
    return "   *"


def _stat_section(
    label: str,
    stat_result: dict,
) -> list[str]:
    """Build the statistical-tests block for one comparison (e.g. Human vs SEA)."""
    lines = []
    a = lines.append
    a(_SEP)
    a(f"  STATISTICAL TESTS — {label}  (Wilcoxon + Holm-Bonferroni)")
    a(_SEP)
    a(f"  {'Metric':<32}  {'HumanMean':>9}  {'LLMMean':>9}  {'Delta':>7}  {'p-adj':>8}  {'d':>6}  {'Effect':>10}  Sig")
    a(_SEP2)
    pm = stat_result.get("per_metric", {})
    for k in METRIC_KEYS:
        data = pm.get(k, {})
        if data.get("skipped"):
            a(f"  {METRIC_LABELS.get(k,k):<32}  {'(skipped — insufficient data)':>48}")
            continue
        h_m   = _fmt(data.get("human_mean"))
        s_m   = _fmt(data.get("llm_mean"))
        diff  = _fmt(data.get("mean_diff"), w=7)
        padj  = data.get("adjusted_p")
        padj_s = f"{padj:.4f}" if padj is not None else "N/A"
        delta  = data.get("cliffs_delta", 0.0)
        mag    = data.get("cliffs_delta_magnitude", "")
        sig    = data.get("significant_after_correction", False)
        mark   = _sig_marker(sig, padj or 1.0)
        a(f"  {METRIC_LABELS.get(k,k):<32}  {h_m}  {s_m}  {diff}  {padj_s:>8}  {delta:>6.3f}  {mag:>10}{mark}")
    a("")
    a("  Significance: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant")
    a("")

    sm = stat_result.get("summary", {})
    llm_better  = sm.get("llm_significantly_better", [])
    hum_better  = sm.get("human_significantly_better", [])
    no_diff     = sm.get("no_significant_difference", [])
    a(f"  LLM significantly better   ({len(llm_better)}): " + (", ".join(llm_better) or "none"))
    a(f"  Human significantly better ({len(hum_better)}): " + (", ".join(hum_better) or "none"))
    a(f"  No significant difference  ({len(no_diff)}): "    + (", ".join(no_diff)    or "none"))
    a("")
    return lines


def _subgroup_section(stat_result: dict) -> list[str]:
    lines = []
    a = lines.append
    sg = stat_result.get("subgroup", {})
    if not sg:
        return lines
    a(_SEP)
    a("  SUBGROUP ANALYSIS")
    a(_SEP)
    for group_name, group_data in sg.items():
        a(f"\n  [{group_name}]")
        if not isinstance(group_data, dict):
            continue
        for tier, tier_data in group_data.items():
            if not tier_data or tier_data.get("n_papers") is None:
                continue
            n = tier_data.get("n_papers", 0)
            a(f"    {tier}  (n={n})")
            for k in ["MCS", "AR", "SD", "CD"]:
                d = tier_data.get(k, {})
                if d.get("skipped"):
                    continue
                h_m  = d.get("human_mean", 0)
                s_m  = d.get("llm_mean",   0)
                diff = d.get("diff", 0)
                mag  = d.get("magnitude", "")
                a(f"      {k:<6} Human={h_m:.4f}  LLM={s_m:.4f}  Δ={diff:+.4f}  ({mag})")
    a("")
    return lines


def build_report(
    sea_paired:    list[dict],
    agg:           dict[str, dict],
    stat_sea:      Optional[dict],
    stat_r2:       Optional[dict],
    stat_dr:       Optional[dict],
    stat_tree:     Optional[dict],
    n_human_raw:   int,
    n_sea_raw:     int,
    n_r2_raw:      int,
    n_dr_raw:      int,
    n_tree_raw:    int,
    r2_paired:     list[dict] | None = None,
    dr_paired:     list[dict] | None = None,
    tree_paired:   list[dict] | None = None,
) -> str:
    lines = []
    a = lines.append

    a(_SEP)
    a("  CONSTRUCTIVENESS EVALUATION — Human vs. SEA / Reviewer2 / DeepReview / Tree")
    a(_SEP)
    a(f"  Human records loaded       : {n_human_raw}")
    a(f"  SEA records loaded         : {n_sea_raw}")
    a(f"  Reviewer2 records loaded   : {n_r2_raw}")
    a(f"  DeepReview records loaded  : {n_dr_raw}")
    a(f"  Tree records loaded        : {n_tree_raw}")
    a(f"  Human ↔ SEA pairs          : {len(sea_paired)}")
    if r2_paired is not None:
        a(f"  Human ↔ Reviewer2 pairs    : {len(r2_paired)}")
    if dr_paired is not None:
        a(f"  Human ↔ DeepReview pairs   : {len(dr_paired)}")
    if tree_paired is not None:
        a(f"  Human ↔ Tree pairs         : {len(tree_paired)}")

    n_accept = sum(1 for r in sea_paired if "accept" in (r.get("metadata", {}).get("decision") or "").lower())
    n_reject = len(sea_paired) - n_accept
    a(f"  Accept / Reject (SEA)      : {n_accept} / {n_reject}")
    n_valid_sea = sum(1 for r in sea_paired if r.get("comparison", {}).get("comparison_valid"))
    a(f"  Valid SEA comparisons      : {n_valid_sea}")
    if r2_paired:
        n_valid_r2 = sum(1 for r in r2_paired if r.get("comparison", {}).get("comparison_valid"))
        a(f"  Valid Reviewer2 comps      : {n_valid_r2}")
    if dr_paired:
        n_valid_dr = sum(1 for r in dr_paired if r.get("comparison", {}).get("comparison_valid"))
        a(f"  Valid DeepReview comps     : {n_valid_dr}")
    if tree_paired:
        n_valid_tree = sum(1 for r in tree_paired if r.get("comparison", {}).get("comparison_valid"))
        a(f"  Valid Tree comps           : {n_valid_tree}")
    a("")

    # ── 5-way aggregate table ────────────────────────────────────────────────
    a(_SEP)
    a("  AGGREGATE METRICS  (mean ± std)")
    a(_SEP)
    has_r2   = any(row.get("reviewer2_mean")  is not None for row in agg.values())
    has_dr   = any(row.get("deepreview_mean") is not None for row in agg.values())
    has_tree = any(row.get("tree_mean")       is not None for row in agg.values())

    hdr_parts = [f"  {'Metric':<32}", f"{'Human':>16}", f"{'SEA':>16}"]
    if has_r2:   hdr_parts.append(f"{'Reviewer2':>16}")
    if has_dr:   hdr_parts.append(f"{'DeepReview':>16}")
    if has_tree: hdr_parts.append(f"{'Tree':>16}")
    hdr_parts += [f"{'Δ(SEA)':>8}"]
    if has_r2:   hdr_parts.append(f"{'Δ(R2)':>8}")
    if has_dr:   hdr_parts.append(f"{'Δ(DR)':>8}")
    if has_tree: hdr_parts.append(f"{'Δ(Tree)':>8}")
    a("  ".join(hdr_parts))
    a(_SEP2)

    for k in METRIC_KEYS:
        row   = agg[k]
        label = METRIC_LABELS.get(k, k)

        def _ms(prefix: str) -> str:
            m = row.get(f"{prefix}_mean")
            s = row.get(f"{prefix}_std")
            if m is None:
                return "N/A"
            return f"{m:.4f}±{s:.4f}" if s is not None else f"{m:.4f}"

        def _d(key: str) -> str:
            v = row.get(key)
            return f"{v:+.4f}" if v is not None else "N/A"

        parts = [f"  {label:<32}", f"{_ms('human'):>16}", f"{_ms('sea'):>16}"]
        if has_r2:   parts.append(f"{_ms('reviewer2'):>16}")
        if has_dr:   parts.append(f"{_ms('deepreview'):>16}")
        if has_tree: parts.append(f"{_ms('tree'):>16}")
        parts.append(f"{_d('sea_delta'):>8}")
        if has_r2:   parts.append(f"{_d('reviewer2_delta'):>8}")
        if has_dr:   parts.append(f"{_d('deepreview_delta'):>8}")
        if has_tree: parts.append(f"{_d('tree_delta'):>8}")
        a("  ".join(parts))
    a("")

    # ── Statistical tests ────────────────────────────────────────────────────
    if stat_sea:
        lines.extend(_stat_section("Human vs. SEA", stat_sea))
        lines.extend(_subgroup_section(stat_sea))
    if stat_r2:
        lines.extend(_stat_section("Human vs. Reviewer2", stat_r2))
        lines.extend(_subgroup_section(stat_r2))
    if stat_dr:
        lines.extend(_stat_section("Human vs. DeepReview", stat_dr))
        lines.extend(_subgroup_section(stat_dr))
    if stat_tree:
        lines.extend(_stat_section("Human vs. Tree", stat_tree))
        lines.extend(_subgroup_section(stat_tree))

    a(_SEP)
    a("  FILES SAVED")
    a(_SEP)
    a(f"  Human↔SEA pairs        : {PAIRED_OUT}")
    if r2_paired is not None:
        a(f"  Human↔Reviewer2 pairs  : {PAIRED_R2_OUT}")
    if dr_paired is not None:
        a(f"  Human↔DeepReview pairs : {PAIRED_DR_OUT}")
    if tree_paired is not None:
        a(f"  Human↔Tree pairs       : {PAIRED_TREE_OUT}")
    a(f"  5-way CSV              : {CSV_OUT}")
    a(f"  SEA stats JSON         : {STAT_OUT}")
    if stat_r2:
        a(f"  Reviewer2 stats JSON   : {STAT_R2_OUT}")
    if stat_dr:
        a(f"  DeepReview stats JSON  : {STAT_DR_OUT}")
    if stat_tree:
        a(f"  Tree stats JSON        : {STAT_TREE_OUT}")
    a(f"  This report            : {REPORT_OUT}")
    a(_SEP)

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Additional per-reviewer analysis
# ══════════════════════════════════════════════════════════════════════════════

def reviewer_level_stats(human_records: list[dict]) -> dict:
    """Stats across all individual human reviewers (not averaged per paper)."""
    all_metrics: dict[str, list] = {k: [] for k in METRIC_KEYS}
    n_reviewers, n_arcs_total = 0, 0

    for rec in human_records:
        for rev in rec.get("reviewers", []):
            m = rev.get("metrics")
            if not m or (m.get("n_arcs") or 0) == 0:
                continue
            n_reviewers += 1
            n_arcs_total += m.get("n_arcs", 0)
            for k in METRIC_KEYS:
                v = m.get(k)
                if v is not None:
                    all_metrics[k].append(v)

    result = {"n_reviewers": n_reviewers, "n_arcs_total": n_arcs_total}
    for k in METRIC_KEYS:
        arr = np.array(all_metrics[k]) if all_metrics[k] else np.array([])
        result[k] = {
            "mean": round(float(np.mean(arr)), 4) if len(arr) else None,
            "std":  round(float(np.std(arr)),  4) if len(arr) else None,
            "n":    len(arr),
        }
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate constructiveness results.")
    p.add_argument("--no-stats",       action="store_true", help="Skip all statistical tests.")
    p.add_argument("--no-reviewer2",   action="store_true", help="Skip Reviewer2 analysis.")
    p.add_argument("--no-deepreview",  action="store_true", help="Skip DeepReview analysis.")
    p.add_argument("--no-tree",        action="store_true", help="Skip Tree review analysis.")
    p.add_argument(
        "--paper-ids", default=None,
        help=(
            "Path to a text file with one paper_id per line. "
            "If given, only those papers are included in the analysis."
        ),
    )
    p.add_argument(
        "--out-tag", default=None,
        help=(
            "Suffix for the output sub-directory, e.g. '200subset' → "
            "results go to output/iclr2024/analysis_200subset/. "
            "Defaults to 'subset' when --paper-ids is given."
        ),
    )
    p.add_argument(
        "--human-jsonl",      default=HUMAN_JSONL,
        help=f"Path to human results JSONL (default: {HUMAN_JSONL})"
    )
    p.add_argument(
        "--sea-jsonl",        default=SEA_JSONL,
        help=f"Path to SEA results JSONL (default: {SEA_JSONL})"
    )
    p.add_argument(
        "--reviewer2-jsonl",  default=REVIEWER2_JSONL,
        help=f"Path to Reviewer2 results JSONL (default: {REVIEWER2_JSONL})"
    )
    p.add_argument(
        "--deepreview-jsonl", default=DEEPREVIEW_JSONL,
        help=f"Path to DeepReview results JSONL (default: {DEEPREVIEW_JSONL})"
    )
    p.add_argument(
        "--tree-jsonl",       default=TREE_JSONL,
        help=f"Path to Tree review results JSONL (default: {TREE_JSONL})"
    )
    return p.parse_args()


def _run_stats(paired: list[dict], label: str) -> Optional[dict]:
    """Run full statistical analysis + subgroup analysis on a paired set."""
    valid = [r for r in paired if r.get("comparison", {}).get("comparison_valid")]
    print(f"       → {len(valid)} valid pairs for {label}")
    if len(valid) < 10:
        print(f"[WARN] Not enough valid pairs for {label} (need ≥10, got {len(valid)}).")
        return None
    result = run_full_analysis(valid)
    result["subgroup"] = run_subgroup_analysis(valid)
    return result


def _load_paper_ids(filepath: str) -> set[str]:
    """Read a text file with one paper_id per line and return a set of IDs."""
    ids: set[str] = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            pid = line.strip()
            if pid:
                ids.add(pid)
    return ids


def _filter_records(records: list[dict], paper_ids: set[str]) -> list[dict]:
    """Keep only records whose paper_id is in the given set."""
    return [r for r in records if r.get("paper_id") in paper_ids]


def main():
    args = parse_args()

    # ── Resolve output directory (dynamic when --paper-ids is given) ──────────
    if args.paper_ids:
        tag = args.out_tag or os.path.splitext(os.path.basename(args.paper_ids))[0]
        analysis_dir  = os.path.join(_OUT_ROOT, f"analysis_{tag}")
    else:
        tag          = args.out_tag
        analysis_dir = ANALYSIS_DIR if not tag else os.path.join(_OUT_ROOT, f"analysis_{tag}")

    paired_out      = os.path.join(analysis_dir, "paired_records.jsonl")
    paired_r2_out   = os.path.join(analysis_dir, "paired_records_reviewer2.jsonl")
    paired_dr_out   = os.path.join(analysis_dir, "paired_records_deepreview.jsonl")
    paired_tree_out = os.path.join(analysis_dir, "paired_records_tree.jsonl")
    csv_out         = os.path.join(analysis_dir, "summary_metrics.csv")
    stat_out        = os.path.join(analysis_dir, "statistical_report.json")
    stat_r2_out     = os.path.join(analysis_dir, "statistical_report_reviewer2.json")
    stat_dr_out     = os.path.join(analysis_dir, "statistical_report_deepreview.json")
    stat_tree_out   = os.path.join(analysis_dir, "statistical_report_tree.json")
    report_out      = os.path.join(analysis_dir, "report.txt")

    # ── Load paper-id filter ───────────────────────────────────────────────────
    paper_ids: set[str] | None = None
    if args.paper_ids:
        paper_ids = _load_paper_ids(args.paper_ids)
        print(f"[INFO] Filtering to {len(paper_ids)} paper IDs from: {args.paper_ids}")

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"[INFO] Loading human results        : {args.human_jsonl}")
    human_records = load_jsonl(args.human_jsonl)
    print(f"       → {len(human_records)} papers (raw)")

    print(f"[INFO] Loading SEA results          : {args.sea_jsonl}")
    sea_records = load_jsonl(args.sea_jsonl)
    print(f"       → {len(sea_records)} papers (raw)")

    r2_records: list[dict] = []
    if not args.no_reviewer2:
        print(f"[INFO] Loading Reviewer2 results    : {args.reviewer2_jsonl}")
        r2_records = load_jsonl(args.reviewer2_jsonl)
        print(f"       → {len(r2_records)} papers (raw)")
        if not r2_records:
            print("       [WARN] No Reviewer2 results found — skipping.")
    else:
        print("[INFO] --no-reviewer2 set, skipping Reviewer2.")

    dr_records: list[dict] = []
    if not args.no_deepreview:
        print(f"[INFO] Loading DeepReview results   : {args.deepreview_jsonl}")
        dr_records = load_jsonl(args.deepreview_jsonl)
        print(f"       → {len(dr_records)} papers (raw)")
        if not dr_records:
            print("       [WARN] No DeepReview results found — skipping.")
    else:
        print("[INFO] --no-deepreview set, skipping DeepReview.")

    tree_records: list[dict] = []
    if not args.no_tree:
        print(f"[INFO] Loading Tree review results  : {args.tree_jsonl}")
        tree_records = load_jsonl(args.tree_jsonl)
        print(f"       → {len(tree_records)} papers (raw)")
        if not tree_records:
            print("       [WARN] No Tree results found — skipping.")
    else:
        print("[INFO] --no-tree set, skipping Tree reviews.")

    # ── Apply paper-id filter ─────────────────────────────────────────────────
    if paper_ids:
        human_records = _filter_records(human_records, paper_ids)
        sea_records   = _filter_records(sea_records,   paper_ids)
        r2_records    = _filter_records(r2_records,    paper_ids)
        dr_records    = _filter_records(dr_records,    paper_ids)
        tree_records  = _filter_records(tree_records,  paper_ids)
        print(f"\n[INFO] After filtering:")
        print(f"       human     : {len(human_records)}")
        print(f"       sea       : {len(sea_records)}")
        print(f"       reviewer2 : {len(r2_records)}")
        print(f"       deepreview: {len(dr_records)}")
        print(f"       tree      : {len(tree_records)}")

        # Warn if some IDs are missing from already-processed results
        processed_human = {r["paper_id"] for r in human_records if r.get("paper_id")}
        missing = paper_ids - processed_human
        if missing:
            print(f"\n  [WARN] {len(missing)} requested paper IDs are NOT yet in the human results.")
            print(f"         Run 'python run_constructiveness.py --mode human' first, then re-evaluate.")

    if not human_records and not sea_records:
        print("[FATAL] No records after filtering. Run run_constructiveness.py first.")
        sys.exit(1)

    os.makedirs(analysis_dir, exist_ok=True)

    # ── Build paired records ───────────────────────────────────────────────────
    print("\n[INFO] Building Human ↔ SEA paired records...")
    sea_paired = build_paired_records(human_records, sea_records)
    print(f"       → {len(sea_paired)} paired papers")
    save_jsonl(paired_out, sea_paired)
    print(f"[INFO] Human↔SEA pairs saved        : {paired_out}")

    r2_paired: list[dict] | None = None
    if r2_records:
        print("[INFO] Building Human ↔ Reviewer2 paired records...")
        r2_paired = build_reviewer2_paired_records(human_records, r2_records)
        print(f"       → {len(r2_paired)} paired papers")
        save_jsonl(paired_r2_out, r2_paired)
        print(f"[INFO] Human↔Reviewer2 pairs saved : {paired_r2_out}")

    dr_paired: list[dict] | None = None
    if dr_records:
        print("[INFO] Building Human ↔ DeepReview paired records...")
        dr_paired = _build_paired_generic(human_records, dr_records)
        print(f"       → {len(dr_paired)} paired papers")
        save_jsonl(paired_dr_out, dr_paired)
        print(f"[INFO] Human↔DeepReview pairs saved: {paired_dr_out}")

    tree_paired: list[dict] | None = None
    if tree_records:
        print("[INFO] Building Human ↔ Tree paired records...")
        tree_paired = _build_paired_generic(human_records, tree_records)
        print(f"       → {len(tree_paired)} paired papers")
        save_jsonl(paired_tree_out, tree_paired)
        print(f"[INFO] Human↔Tree pairs saved      : {paired_tree_out}")

    # ── Aggregate stats ────────────────────────────────────────────────────────
    print("[INFO] Computing aggregate statistics...")
    agg = aggregate_stats_3way(
        sea_paired,
        r2_paired   = r2_paired,
        dr_paired   = dr_paired,
        tree_paired = tree_paired,
    )
    save_csv_3way(agg, csv_out)
    print(f"[INFO] CSV saved                    : {csv_out}")

    # ── Per-reviewer stats ─────────────────────────────────────────────────────
    reviewer_stats = reviewer_level_stats(human_records)
    print(f"\n[INFO] Human reviewers summary:")
    print(f"       Total reviewers : {reviewer_stats['n_reviewers']}")
    print(f"       Total ARCs      : {reviewer_stats['n_arcs_total']}")

    # ── Statistical tests ──────────────────────────────────────────────────────
    stat_sea:  Optional[dict] = None
    stat_r2:   Optional[dict] = None
    stat_dr:   Optional[dict] = None
    stat_tree: Optional[dict] = None

    if not args.no_stats:
        print("\n[INFO] Running statistical analysis — Human vs. SEA...")
        try:
            stat_sea = _run_stats(sea_paired, "Human vs. SEA")
            if stat_sea:
                with open(stat_out, "w", encoding="utf-8") as f:
                    json.dump(stat_sea, f, indent=2, ensure_ascii=False)
                print(f"[INFO] SEA statistical report       : {stat_out}")
        except Exception as e:
            print(f"[WARN] SEA statistical tests failed: {type(e).__name__}: {e}")

        if r2_paired:
            print("[INFO] Running statistical analysis — Human vs. Reviewer2...")
            try:
                stat_r2 = _run_stats(r2_paired, "Human vs. Reviewer2")
                if stat_r2:
                    with open(stat_r2_out, "w", encoding="utf-8") as f:
                        json.dump(stat_r2, f, indent=2, ensure_ascii=False)
                    print(f"[INFO] Reviewer2 statistical report : {stat_r2_out}")
            except Exception as e:
                print(f"[WARN] Reviewer2 statistical tests failed: {type(e).__name__}: {e}")

        if dr_paired:
            print("[INFO] Running statistical analysis — Human vs. DeepReview...")
            try:
                stat_dr = _run_stats(dr_paired, "Human vs. DeepReview")
                if stat_dr:
                    with open(stat_dr_out, "w", encoding="utf-8") as f:
                        json.dump(stat_dr, f, indent=2, ensure_ascii=False)
                    print(f"[INFO] DeepReview statistical report: {stat_dr_out}")
            except Exception as e:
                print(f"[WARN] DeepReview statistical tests failed: {type(e).__name__}: {e}")

        if tree_paired:
            print("[INFO] Running statistical analysis — Human vs. Tree...")
            try:
                stat_tree = _run_stats(tree_paired, "Human vs. Tree")
                if stat_tree:
                    with open(stat_tree_out, "w", encoding="utf-8") as f:
                        json.dump(stat_tree, f, indent=2, ensure_ascii=False)
                    print(f"[INFO] Tree statistical report      : {stat_tree_out}")
            except Exception as e:
                print(f"[WARN] Tree statistical tests failed: {type(e).__name__}: {e}")
    else:
        print("[INFO] --no-stats set, skipping all statistical tests.")

    # ── Print & save report ───────────────────────────────────────────────────
    report = build_report(
        sea_paired   = sea_paired,
        agg          = agg,
        stat_sea     = stat_sea,
        stat_r2      = stat_r2,
        stat_dr      = stat_dr,
        stat_tree    = stat_tree,
        n_human_raw  = len(human_records),
        n_sea_raw    = len(sea_records),
        n_r2_raw     = len(r2_records),
        n_dr_raw     = len(dr_records),
        n_tree_raw   = len(tree_records),
        r2_paired    = r2_paired,
        dr_paired    = dr_paired,
        tree_paired  = tree_paired,
    )
    print("\n" + report)

    with open(report_out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[INFO] Report saved: {report_out}")


if __name__ == "__main__":
    main()
