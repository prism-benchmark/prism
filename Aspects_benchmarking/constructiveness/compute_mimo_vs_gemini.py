"""
compute_mimo_vs_gemini.py
=========================
Computes mean MCS and D1-D5 dimension scores for:
  - Mimo  v2.5-pro  (50-paper subset, all conferences)
  - Gemini 2.5-flash-lite (50-paper subset filtered from 200-paper runs)

Outputs:
  1. Per-conference tables printed to console
  2. Aggregated (macro-average across conferences) table
  3. CSV files saved to constructiveness_v2/output/analysis_all/
"""

from __future__ import annotations

import json
import os
import csv
from collections import defaultdict
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFERENCES = ["iclr2024", "iclr2025", "iclr2026", "icml2025", "neurips2025"]
REVIEWER_TYPES = ["human", "sea", "tree", "reviewer2", "deepreview", "cyclereview"]
DIMENSIONS = ["D1_actionability_mean", "D2_specificity_mean", "D3_justification_mean",
               "D4_solution_mean", "D5_tone_mean"]

DATA_ROOT = os.getenv("DATA_ROOT", "/path/to/Final_LLM_Reviewer_Data")
METRIC_DATA_ROOT = os.getenv("METRIC_DATA_ROOT", "/path/to/Final Metric Data")

PAPER_IDS_50 = {
    "iclr2024":    os.path.join(DATA_ROOT, "ICLR2024", "paper_ids_50_iclr2024.txt"),
    "iclr2025":    os.path.join(DATA_ROOT, "ICLR2025", "paper_ids_50_iclr2025.txt"),
    "iclr2026":    os.path.join(DATA_ROOT, "ICLR2026", "paper_ids_50_iclr2026.txt"),
    "icml2025":    os.path.join(DATA_ROOT, "ICML2025", "paper_ids_50_icml2025.txt"),
    "neurips2025": os.path.join(DATA_ROOT, "Neurlps2025", "paper_ids_50_neurips2025.txt"),
}

# Mimo paths: output/<conf>/mimo/<rtype>/all_results_lite.jsonl
MIMO_BASE = os.getenv("CONSTRUCTIVENESS_MIMO_OUTPUT_ROOT", os.path.join("output"))

def mimo_path(conf: str, rtype: str) -> str:
    return os.path.join(MIMO_BASE, conf, "mimo", rtype, "all_results_lite.jsonl")

# Gemini paths – each reviewer type has its own naming convention
def gemini_path(conf: str, rtype: str) -> Optional[str]:
    """Return the path to the Gemini JSONL file, or None if not applicable."""
    # Map conference to uppercased folder name used in Gemini results
    conf_upper_map = {
        "iclr2024":    "ICLR2024",
        "iclr2025":    "ICLR2025",
        "iclr2026":    "ICLR2026",
        "icml2025":    "ICML2025",
        "neurips2025": "NEURLPS2025",  # typo in folder name
    }
    # Human NeurIPS folder has different typo: neurilps2025
    human_conf_map = {
        "iclr2024":    "human_iclr2024",
        "iclr2025":    "human_iclr2025",
        "iclr2026":    "human_iclr2026",
        "icml2025":    "human_icml2025",
        "neurips2025": "human_neurilps2025",   # typo in folder
    }
    # LLM folder names within Constructiveness sub-folder
    llm_subfolder_map = {
        "sea":         ("SEA",         f"sea_{conf}"),
        "tree":        ("TreeReview",  f"tree_{conf}"),
        "reviewer2":   ("Reviewer2",   f"reviewer2_{conf}"),
        "deepreview":  ("DeepReview",  f"deepreview_{conf}"),
        "cyclereview": ("CycleReview", f"cyclereview_{conf}"),
    }

    base = METRIC_DATA_ROOT
    conf_up = conf_upper_map[conf]

    if rtype == "human":
        sub = human_conf_map[conf]
        candidates = [
            os.path.join(base, "Human", "Constructiveness", sub, "all_results_lite.jsonl"),
            os.path.join(base, "Human", "Constructiveness", sub, "all_results_lite_200.jsonl"),
        ]
    else:
        top_folder, inner_folder = llm_subfolder_map[rtype]
        candidates = [
            os.path.join(base, top_folder, conf_up, "Constructiveness", inner_folder, "all_results_lite.jsonl"),
            os.path.join(base, top_folder, conf_up, "Constructiveness", inner_folder, "all_results_lite_200.jsonl"),
        ]

    for p in candidates:
        if os.path.exists(p):
            return p
    return None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_paper_ids(path: str) -> set:
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            pid = line.strip()
            if pid and not pid.startswith("#"):
                ids.add(pid)
    return ids


def load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def compute_means(records: List[dict], paper_ids: set) -> Optional[Dict[str, float]]:
    """Filter by paper_ids and compute mean MCS + D1-D5.

    Handles two record formats:
      1. LLM reviewer: {paper_id, metrics: {...}, ...}  — one metrics per record
      2. Human reviewer: {paper_id, reviewers: [{reviewer_id, metrics:{...}}, ...]}
         — multiple reviewers per paper; flatten all reviewer-metrics then average
    """
    filtered = [r for r in records if r.get("paper_id") in paper_ids]
    if not filtered:
        return None

    metrics_list: List[dict] = []
    paper_count = 0
    for r in filtered:
        if "reviewers" in r:
            # Human format: aggregate all per-paper reviewer metrics
            paper_count += 1
            for rev in r["reviewers"]:
                m = rev.get("metrics", {})
                if m:
                    metrics_list.append(m)
        elif "metrics" in r:
            # LLM format: single metrics per record
            metrics_list.append(r["metrics"])
            paper_count += 1

    if not metrics_list:
        return None

    cols = ["MCS"] + DIMENSIONS
    result = {}
    for col in cols:
        vals = [m[col] for m in metrics_list if col in m and m[col] is not None]
        result[col] = round(sum(vals) / len(vals), 4) if vals else None
    result["n"] = paper_count
    result["n_reviewers"] = len(metrics_list)
    return result

# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyse():
    # storage: [conf][rtype][model] = metrics_dict
    results: Dict[str, Dict[str, Dict[str, Optional[Dict]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    missing_files = []

    for conf in CONFERENCES:
        ids_50 = load_paper_ids(PAPER_IDS_50[conf])
        print(f"\n{'='*60}")
        print(f"  Conference: {conf.upper()}  ({len(ids_50)} target IDs)")
        print(f"{'='*60}")

        for rtype in REVIEWER_TYPES:
            # --- Mimo ---
            mp = mimo_path(conf, rtype)
            if os.path.exists(mp):
                mimo_records = load_jsonl(mp)
                results[conf][rtype]["mimo"] = compute_means(mimo_records, ids_50)
            else:
                results[conf][rtype]["mimo"] = None
                missing_files.append(f"MIMO {conf}/{rtype}: {mp}")

            # --- Gemini ---
            gp = gemini_path(conf, rtype)
            if gp:
                gemini_records = load_jsonl(gp)
                results[conf][rtype]["gemini"] = compute_means(gemini_records, ids_50)
            else:
                results[conf][rtype]["gemini"] = None
                missing_files.append(f"GEMINI {conf}/{rtype}: NOT FOUND")

    return results, missing_files


def print_table(results, conf: str):
    cols = ["MCS"] + DIMENSIONS
    short_dim = {"MCS": "MCS",
                 "D1_actionability_mean": "D1",
                 "D2_specificity_mean":   "D2",
                 "D3_justification_mean": "D3",
                 "D4_solution_mean":      "D4",
                 "D5_tone_mean":          "D5"}
    header = f"{'Reviewer':<14} {'Model':<8}" + "".join(f" {short_dim[c]:>7}" for c in cols) + "  n"
    sep = "-" * len(header)
    print(f"\n  {conf.upper()}")
    print(f"  {sep}")
    print(f"  {header}")
    print(f"  {sep}")
    for rtype in REVIEWER_TYPES:
        for model in ["mimo", "gemini"]:
            m = results[conf][rtype].get(model)
            if m is None:
                print(f"  {rtype:<14} {model:<8}" + "  [NO DATA]")
                continue
            vals = "".join(
                f" {m[c]:>7.4f}" if m.get(c) is not None else f" {'N/A':>7}"
                for c in cols
            )
            n_info = f"{m.get('n','?')}"
            if "n_reviewers" in m:
                n_info += f"p/{m['n_reviewers']}r"
            print(f"  {rtype:<14} {model:<8}{vals}  {n_info}")
    print(f"  {sep}")


def compute_aggregate(results):
    """Macro-average across all 5 conferences."""
    agg: Dict[str, Dict[str, Dict[str, list]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for conf in CONFERENCES:
        for rtype in REVIEWER_TYPES:
            for model in ["mimo", "gemini"]:
                m = results[conf][rtype].get(model)
                if m is None:
                    continue
                cols = ["MCS"] + DIMENSIONS
                for col in cols:
                    if m.get(col) is not None:
                        agg[rtype][model][col].append(m[col])
    # compute means
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    cols = ["MCS"] + DIMENSIONS
    for rtype in REVIEWER_TYPES:
        out[rtype] = {}
        for model in ["mimo", "gemini"]:
            vals_dict = agg[rtype][model]
            out[rtype][model] = {
                col: round(sum(vals_dict[col]) / len(vals_dict[col]), 4) if vals_dict[col] else None
                for col in cols
            }
            out[rtype][model]["n_confs"] = len(vals_dict.get("MCS", []))
    return out


def print_aggregate(agg):
    cols = ["MCS"] + DIMENSIONS
    short_dim = {"MCS": "MCS",
                 "D1_actionability_mean": "D1",
                 "D2_specificity_mean":   "D2",
                 "D3_justification_mean": "D3",
                 "D4_solution_mean":      "D4",
                 "D5_tone_mean":          "D5"}
    header = f"{'Reviewer':<14} {'Model':<8}" + "".join(f" {short_dim[c]:>7}" for c in cols) + "  confs"
    sep = "-" * len(header)
    print(f"\n\n{'='*65}")
    print(f"  AGGREGATE (macro-average across {len(CONFERENCES)} conferences)")
    print(f"{'='*65}")
    print(f"  {sep}")
    print(f"  {header}")
    print(f"  {sep}")
    for rtype in REVIEWER_TYPES:
        for model in ["mimo", "gemini"]:
            m = agg[rtype].get(model, {})
            if not m:
                print(f"  {rtype:<14} {model:<8}  [NO DATA]")
                continue
            vals = "".join(
                f" {m[c]:>7.4f}" if m.get(c) is not None else f" {'N/A':>7}"
                for c in cols
            )
            print(f"  {rtype:<14} {model:<8}{vals}  {m.get('n_confs','?')}")
    print(f"  {sep}")


def save_csvs(results, agg, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    cols = ["MCS"] + DIMENSIONS
    col_labels = ["MCS", "D1", "D2", "D3", "D4", "D5"]

    # Per-conference CSVs
    for conf in CONFERENCES:
        rows = []
        for rtype in REVIEWER_TYPES:
            for model in ["mimo", "gemini"]:
                m = results[conf][rtype].get(model)
                if m is None:
                    continue
                row = {"conference": conf, "reviewer": rtype, "model": model}
                for col, label in zip(cols, col_labels):
                    row[label] = m.get(col, "")
                row["n"] = m.get("n", "")
                rows.append(row)
        csv_path = os.path.join(output_dir, f"constructiveness_{conf}.csv")
        if rows:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["conference","reviewer","model"] + col_labels + ["n"])
                writer.writeheader()
                writer.writerows(rows)
            print(f"  [SAVED] {csv_path}")

    # Aggregate CSV
    rows = []
    for rtype in REVIEWER_TYPES:
        for model in ["mimo", "gemini"]:
            m = agg[rtype].get(model, {})
            if not m:
                continue
            row = {"conference": "ALL", "reviewer": rtype, "model": model}
            for col, label in zip(cols, col_labels):
                row[label] = m.get(col, "")
            row["n"] = m.get("n_confs", "")
            rows.append(row)
    agg_csv = os.path.join(output_dir, "constructiveness_aggregate.csv")
    if rows:
        with open(agg_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["conference","reviewer","model"] + col_labels + ["n"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"  [SAVED] {agg_csv}")


def print_delta_table(agg):
    """Print Mimo - Gemini delta to see which model scores higher."""
    cols = ["MCS"] + DIMENSIONS
    short_dim = {"MCS": "MCS",
                 "D1_actionability_mean": "D1",
                 "D2_specificity_mean":   "D2",
                 "D3_justification_mean": "D3",
                 "D4_solution_mean":      "D4",
                 "D5_tone_mean":          "D5"}
    header = f"{'Reviewer':<14}" + "".join(f" {short_dim[c]:>9}" for c in cols)
    sep = "-" * len(header)
    print(f"\n\n{'='*65}")
    print(f"  DELTA: Mimo − Gemini (positive = Mimo higher)")
    print(f"{'='*65}")
    print(f"  {sep}")
    print(f"  {header}")
    print(f"  {sep}")
    for rtype in REVIEWER_TYPES:
        mm = agg[rtype].get("mimo", {})
        gm = agg[rtype].get("gemini", {})
        if not mm or not gm:
            print(f"  {rtype:<14}  [INCOMPLETE]")
            continue
        vals = ""
        for col in cols:
            mv = mm.get(col)
            gv = gm.get(col)
            if mv is not None and gv is not None:
                delta = mv - gv
                sign = "+" if delta >= 0 else ""
                vals += f" {sign}{delta:>7.4f}"
            else:
                vals += f" {'N/A':>9}"
        print(f"  {rtype:<14}{vals}")
    print(f"  {sep}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Computing Constructiveness: Mimo vs Gemini (50-paper subset)")
    print("=" * 65)

    results, missing = analyse()

    if missing:
        print("\n[WARNING] Missing files:")
        for f in missing:
            print(f"  - {f}")

    # Per-conference tables
    for conf in CONFERENCES:
        print_table(results, conf)

    # Aggregate
    agg = compute_aggregate(results)
    print_aggregate(agg)

    # Delta table
    print_delta_table(agg)

    # Save CSVs
    out_dir = os.path.join(MIMO_BASE, "analysis_all")
    print(f"\n\nSaving CSVs to {out_dir} ...")
    save_csvs(results, agg, out_dir)

    print("\nDone.")



