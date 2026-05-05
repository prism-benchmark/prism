"""
compute_flaw_metrics.py
════════════════════════════════════════════════════════════════════════════════
Tính metrics Flaw Identification (CFI + CPS) cho từng reviewer theo từng
conference và từng LLM type.

Cấu trúc mỗi file output:
  - Mỗi paper record chứa:
      metrics_report.cfi.Reviewer_Rankings  →  [Human_1, Human_2, ..., LLM_Reviewer]
      metrics_report.cps.Reviewer_Rankings  →  [Human_1, Human_2, ..., LLM_Reviewer]
  - LLM_Reviewer là LLM được paired với human trong file đó

Metrics tính:
  CFI: Minor_Recall, Critical_Recall (nếu có), Minor_Recall_CW, Critical_Recall_CW,
       Total_Valid_Flaws_Found, GT_Total_Valid_Flaws, GT_Minor_Flaws, GT_Critical_Flaws
  CPS: Raw_CPS, ICPS, nCPS, CPS_norm, Total_Arguments

Output
------
  flaw_results/
    all_conferences_all_llms.csv            ← bảng tổng hợp toàn bộ
    summary_llm_vs_human.csv                ← LLM vs Human_avg (so sánh)
    per_conference/
      <conf>_<llm_type>_metrics.csv         ← chi tiết từng file
    report.txt                              ← báo cáo text

Usage
-----
  python compute_flaw_metrics.py
  python compute_flaw_metrics.py --conf iclr2024
  python compute_flaw_metrics.py --paper-ids path/to/ids.txt
  python compute_flaw_metrics.py --out-dir flaw_results/custom/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Any, Optional

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))

# Conference → LLM type → path of the combined output JSONL
CONFERENCE_LLM_FILES: dict[str, dict[str, str]] = {
    "iclr2024": {
        "sea":         os.path.join(_HERE, "output_cfi_iclr2024",            "all_papers_results.jsonl"),
        "reviewer2":   os.path.join(_HERE, "output_cfi_iclr2024_reviewer2",  "all_papers_results.jsonl"),
        "deepreview":  os.path.join(_HERE, "output_cfi_iclr2024_deepreview", "all_papers_results.jsonl"),
        "tree":        os.path.join(_HERE, "output_cfi_iclr2024_tree",       "all_papers_results.jsonl"),
        "cyclereview": os.path.join(_HERE, "output_cfi_iclr2024_cyclereview","all_papers_results.jsonl"),
    },
    "iclr2025": {
        "sea":         os.path.join(_HERE, "output_cfi_iclr2025",            "all_papers_results.jsonl"),
        "reviewer2":   os.path.join(_HERE, "output_cfi_iclr2025_reviewer2",  "all_papers_results.jsonl"),
        "deepreview":  os.path.join(_HERE, "output_cfi_iclr2025_deepreview", "all_papers_results.jsonl"),
        "tree":        os.path.join(_HERE, "output_cfi_iclr2025_tree",       "all_papers_results.jsonl"),
        "cyclereview": os.path.join(_HERE, "output_cfi_iclr2025_cyclereview","all_papers_results.jsonl"),
    },
    "iclr2026": {
        "sea":         os.path.join(_HERE, "output_cfi_iclr2026",            "all_papers_results.jsonl"),
        "reviewer2":   os.path.join(_HERE, "output_cfi_iclr2026_reviewer2",  "all_papers_results.jsonl"),
        "deepreview":  os.path.join(_HERE, "output_cfi_iclr2026_deepreview", "all_papers_results.jsonl"),
        "tree":        os.path.join(_HERE, "output_cfi_iclr2026_tree",       "all_papers_results.jsonl"),
        "cyclereview": os.path.join(_HERE, "output_cfi_iclr2026_cyclereview","all_papers_results.jsonl"),
    },
    "icml2025": {
        "sea":         os.path.join(_HERE, "output_cfi_icml2025",            "all_papers_results.jsonl"),
        "reviewer2":   os.path.join(_HERE, "output_cfi_icml2025_reviewer2",  "all_papers_results.jsonl"),
        "deepreview":  os.path.join(_HERE, "output_cfi_icml2025_deepreview", "all_papers_results.jsonl"),
        "tree":        os.path.join(_HERE, "output_cfi_icml2025_tree",       "all_papers_results.jsonl"),
        "cyclereview": os.path.join(_HERE, "output_cfi_icml2025_cyclereview","all_papers_results.jsonl"),
    },
    "neurips2025": {
        "sea":         os.path.join(_HERE, "output_cfi_neurips2025",            "all_papers_results.jsonl"),
        "reviewer2":   os.path.join(_HERE, "output_cfi_neurips2025_reviewer2",  "all_papers_results.jsonl"),
        "deepreview":  os.path.join(_HERE, "output_cfi_neurips2025_deepreview", "all_papers_results.jsonl"),
        "tree":        os.path.join(_HERE, "output_cfi_neurips2025_tree",       "all_papers_results.jsonl"),
        "cyclereview": os.path.join(_HERE, "output_cfi_neurips2025_cyclereview","all_papers_results.jsonl"),
    },
}

# CFI metrics extracted per reviewer
CFI_KEYS = [
    "Minor_Recall",
    "Critical_Recall",
    "Minor_Recall_CW",       # ConsensusWeighted
    "Critical_Recall_CW",
    "Total_Valid_Flaws_Found",
]

# CPS metrics extracted per reviewer
CPS_KEYS = [
    "Raw_CPS",
    "ICPS",
    "nCPS",
    "CPS_norm",
    "Total_Arguments",
]

# Ground truth keys (paper-level, not per-reviewer)
GT_KEYS = [
    "GT_Total_Valid_Flaws",
    "GT_Critical_Flaws",
    "GT_Minor_Flaws",
    "GT_Minor_ConsensusWeight",
]

ALL_METRIC_KEYS = CFI_KEYS + CPS_KEYS


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _safe(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _agg(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    arr = np.array(values, dtype=float)
    return {
        "n":    len(arr),
        "mean": round(float(np.mean(arr)), 4),
        "std":  round(float(np.std(arr)),  4),
        "min":  round(float(np.min(arr)),  4),
        "max":  round(float(np.max(arr)),  4),
    }


# ── Extract per-paper, per-reviewer metrics ──────────────────────────────────

def _extract_cfi_row(ranking: dict) -> dict:
    """Extract CFI metrics from one reviewer's ranking entry."""
    return {
        "Minor_Recall":           _safe(ranking.get("Minor_Recall")),
        "Critical_Recall":        _safe(ranking.get("Critical_Recall")),
        "Minor_Recall_CW":        _safe(ranking.get("Minor_Recall_ConsensusWeighted")),
        "Critical_Recall_CW":     _safe(ranking.get("Critical_Recall_ConsensusWeighted")),
        "Total_Valid_Flaws_Found": _safe(ranking.get("Total_Valid_Flaws_Found")),
    }


def _extract_cps_row(ranking: dict) -> dict:
    """Extract CPS metrics from one reviewer's ranking entry."""
    return {
        "Raw_CPS":        _safe(ranking.get("Raw_CPS")),
        "ICPS":           _safe(ranking.get("ICPS")),
        "nCPS":           _safe(ranking.get("nCPS")),
        "CPS_norm":       _safe(ranking.get("CPS_norm")),
        "Total_Arguments": _safe(ranking.get("Total_Arguments")),
    }


def _extract_gt(record: dict) -> dict:
    """Extract ground truth summary from cfi section."""
    cfi = (record.get("metrics_report") or {}).get("cfi") or {}
    gt  = cfi.get("Ground_Truth_Summary") or {}
    return {
        "GT_Total_Valid_Flaws":    _safe(gt.get("Total_Valid_Flaws")),
        "GT_Critical_Flaws":       _safe(gt.get("Total_Critical_Flaws")),
        "GT_Minor_Flaws":          _safe(gt.get("Total_Minor_Flaws")),
        "GT_Minor_ConsensusWeight": _safe(gt.get("Total_Minor_ConsensusWeight")),
    }


def extract_all_reviewers(record: dict) -> dict[str, dict]:
    """
    From a paper record, extract per-reviewer CFI + CPS metrics.
    Returns: {reviewer_id -> {CFI metrics..., CPS metrics..., GT metrics...}}
    """
    mr  = record.get("metrics_report") or {}
    cfi = mr.get("cfi") or {}
    cps = mr.get("cps") or {}
    gt  = _extract_gt(record)

    cfi_rankings: list[dict] = cfi.get("Reviewer_Rankings") or []
    cps_rankings: list[dict] = cps.get("Reviewer_Rankings") or []

    # Index by Reviewer_ID
    cfi_by_id = {r.get("Reviewer_ID", ""): r for r in cfi_rankings}
    cps_by_id = {r.get("Reviewer_ID", ""): r for r in cps_rankings}

    all_ids = set(cfi_by_id.keys()) | set(cps_by_id.keys())

    result: dict[str, dict] = {}
    for rid in all_ids:
        if not rid:
            continue
        row: dict[str, Any] = {}
        if rid in cfi_by_id:
            row.update(_extract_cfi_row(cfi_by_id[rid]))
        else:
            row.update({k: None for k in CFI_KEYS})
        if rid in cps_by_id:
            row.update(_extract_cps_row(cps_by_id[rid]))
        else:
            row.update({k: None for k in CPS_KEYS})
        row.update(gt)
        result[rid] = row

    return result


# ── Aggregate across papers ───────────────────────────────────────────────────

def aggregate_reviewer_metrics(
    per_paper_list: list[dict],
) -> dict:
    """
    Aggregate list of per-paper metric dicts into mean ± std.
    Handles None values (skips them).
    """
    agg_data: dict[str, list[float]] = {k: [] for k in ALL_METRIC_KEYS + GT_KEYS}

    for m in per_paper_list:
        for k in ALL_METRIC_KEYS + GT_KEYS:
            v = m.get(k)
            if v is not None:
                agg_data[k].append(float(v))

    result: dict[str, Any] = {"n_papers": len(per_paper_list)}
    for k in ALL_METRIC_KEYS + GT_KEYS:
        s = _agg(agg_data[k])
        result[f"{k}_mean"] = s["mean"]
        result[f"{k}_std"]  = s["std"]
        result[f"{k}_n"]    = s["n"]

    return result


# ── Process one file ──────────────────────────────────────────────────────────

def process_file(
    conf: str,
    llm_type: str,
    fpath: str,
    paper_ids: Optional[set[str]] = None,
) -> list[dict]:
    """
    Load one results JSONL and return a list of row dicts (one per reviewer).
    Each row includes: conference, llm_type, reviewer_id + aggregated metrics.
    """
    records = load_jsonl(fpath)
    if not records:
        print(f"  [SKIP] {conf}/{llm_type}: file rong hoac khong ton tai")
        return []

    if paper_ids:
        records = [r for r in records if r.get("paper_id") in paper_ids]
        if not records:
            print(f"  [SKIP] {conf}/{llm_type}: 0 records sau filter")
            return []

    print(f"  [OK]   {conf}/{llm_type}: {len(records)} papers")

    # Collect per-paper metrics, grouped by reviewer_id
    per_reviewer: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        rev_metrics = extract_all_reviewers(rec)
        for rid, mdict in rev_metrics.items():
            per_reviewer[rid].append(mdict)

    rows: list[dict] = []
    human_ids = sorted(k for k in per_reviewer if k.startswith("Human_"))

    # LLM_Reviewer row
    if "LLM_Reviewer" in per_reviewer:
        agg = aggregate_reviewer_metrics(per_reviewer["LLM_Reviewer"])
        rows.append({
            "conference":   conf,
            "llm_type":     llm_type,
            "reviewer_id":  "LLM_Reviewer",
            "reviewer_role": "LLM",
            **agg,
        })

    # Individual Human rows
    for rid in human_ids:
        agg = aggregate_reviewer_metrics(per_reviewer[rid])
        rows.append({
            "conference":   conf,
            "llm_type":     llm_type,
            "reviewer_id":  rid,
            "reviewer_role": "Human",
            **agg,
        })

    # Human_avg row (average across all human reviewers, paper-by-paper)
    if human_ids:
        # Build a combined list: for each paper, average the human metrics
        # Collect per-paper per-human, then average per-paper
        n_papers = len(records)
        # Reuse per_reviewer data but need to average across humans per paper
        all_human_per_paper: list[dict] = []
        for rec in records:
            rev_metrics = extract_all_reviewers(rec)
            h_metrics = [rev_metrics[rid] for rid in human_ids if rid in rev_metrics]
            if not h_metrics:
                continue
            # Average per paper across humans
            avg_m: dict[str, Any] = {}
            for k in ALL_METRIC_KEYS + GT_KEYS:
                vals = [m[k] for m in h_metrics if m.get(k) is not None]
                avg_m[k] = round(sum(vals) / len(vals), 4) if vals else None
            all_human_per_paper.append(avg_m)

        agg = aggregate_reviewer_metrics(all_human_per_paper)
        rows.append({
            "conference":   conf,
            "llm_type":     llm_type,
            "reviewer_id":  "Human_avg",
            "reviewer_role": "Human_avg",
            **agg,
        })

    return rows


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _fieldnames() -> list[str]:
    base = ["conference", "llm_type", "reviewer_id", "reviewer_role", "n_papers"]
    for k in ALL_METRIC_KEYS + GT_KEYS:
        base += [f"{k}_mean", f"{k}_std", f"{k}_n"]
    return base


def save_csv(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = _fieldnames()
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if row.get(k) is None else row[k]) for k in fields})


# ── Build summary comparison CSV (LLM vs Human_avg) ─────────────────────────

def build_summary_comparison(all_rows: list[dict]) -> list[dict]:
    """
    Create a flattened comparison table:
    conference | llm_type | [LLM metrics] | [Human_avg metrics]
    """
    # Group by (conference, llm_type)
    groups: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in all_rows:
        key = (row["conference"], row["llm_type"])
        groups[key][row["reviewer_id"]] = row

    summary_rows: list[dict] = []
    COMPARE_KEYS = [
        "Minor_Recall", "Minor_Recall_CW",
        "Raw_CPS", "CPS_norm", "nCPS",
        "Total_Arguments", "Total_Valid_Flaws_Found",
        "GT_Total_Valid_Flaws",
    ]

    for (conf, llm_type), rev_map in sorted(groups.items()):
        llm_row   = rev_map.get("LLM_Reviewer", {})
        havg_row  = rev_map.get("Human_avg", {})

        out: dict[str, Any] = {
            "conference": conf,
            "llm_type":   llm_type,
            "n_papers":   llm_row.get("n_papers") or havg_row.get("n_papers"),
        }
        for k in COMPARE_KEYS:
            out[f"LLM_{k}"]     = llm_row.get(f"{k}_mean")
            out[f"Human_avg_{k}"] = havg_row.get(f"{k}_mean")
            # Delta: LLM - Human_avg
            lv = llm_row.get(f"{k}_mean")
            hv = havg_row.get(f"{k}_mean")
            out[f"delta_{k}"] = (
                round(float(lv) - float(hv), 4)
                if lv is not None and hv is not None
                else None
            )
        summary_rows.append(out)

    return summary_rows


def save_summary_csv(summary_rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    COMPARE_KEYS = [
        "Minor_Recall", "Minor_Recall_CW",
        "Raw_CPS", "CPS_norm", "nCPS",
        "Total_Arguments", "Total_Valid_Flaws_Found",
        "GT_Total_Valid_Flaws",
    ]
    fields = ["conference", "llm_type", "n_papers"]
    for k in COMPARE_KEYS:
        fields += [f"LLM_{k}", f"Human_avg_{k}", f"delta_{k}"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in summary_rows:
            w.writerow({k: ("" if row.get(k) is None else row[k]) for k in fields})


# ── Text report ───────────────────────────────────────────────────────────────

_SEP  = "=" * 110
_SEP2 = "-" * 110

def _fmt(v: Any, w: int = 8) -> str:
    if v is None:
        return "N/A".rjust(w)
    try:
        return f"{float(v):.4f}".rjust(w)
    except (TypeError, ValueError):
        return str(v).rjust(w)


def _fmt_ms(mean: Any, std: Any, w: int = 14) -> str:
    if mean is None:
        return "N/A".rjust(w)
    if std is None:
        return f"{float(mean):.4f}".rjust(w)
    return f"{float(mean):.4f}+-{float(std):.4f}".rjust(w)


def build_report(all_rows: list[dict], summary_rows: list[dict]) -> str:
    lines: list[str] = []
    a = lines.append

    a(_SEP)
    a("  FLAW IDENTIFICATION METRICS  (CFI + CPS)  — Per Reviewer x Conference x LLM Type")
    a(_SEP)

    # Group by conference → llm_type
    conf_llm_rows: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in all_rows:
        conf_llm_rows[row["conference"]][row["llm_type"]].append(row)

    LLM_ORDER = ["sea", "reviewer2", "deepreview", "tree", "cyclereview"]

    for conf in sorted(conf_llm_rows.keys()):
        a("")
        a(_SEP)
        a(f"  CONFERENCE: {conf.upper()}")
        a(_SEP)

        for llm_type in LLM_ORDER:
            if llm_type not in conf_llm_rows[conf]:
                continue
            rows = conf_llm_rows[conf][llm_type]
            a(f"\n  -- LLM paired: {llm_type.upper()} --")
            a(f"  {'Reviewer':<22} {'Role':<12} {'N_papers':>8}  "
              f"{'MinorR':>8}  {'MinorR_CW':>10}  "
              f"{'Raw_CPS':>8}  {'CPS_norm':>9}  {'nCPS':>7}  "
              f"{'TotArgs':>7}  {'FlawsFound':>11}  {'GT_Flaws':>9}")
            a("  " + "-" * 107)
            for row in rows:
                rid   = row["reviewer_id"]
                role  = row["reviewer_role"]
                n_p   = row["n_papers"]
                mr    = _fmt(row.get("Minor_Recall_mean"),    8)
                mr_cw = _fmt(row.get("Minor_Recall_CW_mean"), 10)
                rcps  = _fmt(row.get("Raw_CPS_mean"),         8)
                cnorm = _fmt(row.get("CPS_norm_mean"),        9)
                ncps  = _fmt(row.get("nCPS_mean"),            7)
                targs = _fmt(row.get("Total_Arguments_mean"), 7)
                found = _fmt(row.get("Total_Valid_Flaws_Found_mean"), 11)
                gt    = _fmt(row.get("GT_Total_Valid_Flaws_mean"),    9)
                a(f"  {rid:<22} {role:<12} {n_p:>8}  "
                  f"{mr}  {mr_cw}  {rcps}  {cnorm}  {ncps}  {targs}  {found}  {gt}")
            a("")

    # Summary comparison table
    a("")
    a(_SEP)
    a("  SUMMARY: LLM vs Human_avg — Minor Recall & CPS")
    a(_SEP)
    a(f"  {'Conference':<14} {'LLM Type':<14} {'N':>5}  "
      f"{'LLM_MinorR':>11}  {'Hum_MinorR':>11}  {'D_MinorR':>9}  "
      f"{'LLM_RawCPS':>11}  {'Hum_RawCPS':>11}  {'D_CPS':>7}  "
      f"{'LLM_CPS_norm':>13}  {'Hum_CPS_norm':>13}  {'D_norm':>7}")
    a(_SEP2)

    for row in summary_rows:
        conf      = row["conference"]
        llm_type  = row["llm_type"]
        n         = row.get("n_papers") or 0
        l_mr   = _fmt(row.get("LLM_Minor_Recall"),       11)
        h_mr   = _fmt(row.get("Human_avg_Minor_Recall"),  11)
        d_mr   = _fmt(row.get("delta_Minor_Recall"),       9)
        l_cps  = _fmt(row.get("LLM_Raw_CPS"),            11)
        h_cps  = _fmt(row.get("Human_avg_Raw_CPS"),       11)
        d_cps  = _fmt(row.get("delta_Raw_CPS"),            7)
        l_norm = _fmt(row.get("LLM_CPS_norm"),            13)
        h_norm = _fmt(row.get("Human_avg_CPS_norm"),      13)
        d_norm = _fmt(row.get("delta_CPS_norm"),            7)
        a(f"  {conf:<14} {llm_type:<14} {n:>5}  "
          f"{l_mr}  {h_mr}  {d_mr}  "
          f"{l_cps}  {h_cps}  {d_cps}  "
          f"{l_norm}  {h_norm}  {d_norm}")

    a("")
    a(_SEP)
    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Tinh metrics CFI + CPS cho tung reviewer, tung conference, tung LLM."
    )
    p.add_argument("--conf", default=None,
                   help="Chi chay 1 conference (vd: iclr2024). Mac dinh: tat ca.")
    p.add_argument("--llm", default=None,
                   help="Chi chay 1 llm_type (vd: sea). Mac dinh: tat ca.")
    p.add_argument("--paper-ids", default=None,
                   help="Path den file txt chua paper IDs can loc (1 ID/dong).")
    p.add_argument("--out-dir", default=None,
                   help="Thu muc output. Mac dinh: flaw_results/ trong flaw_identification/")
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = args.out_dir or os.path.join(_HERE, "flaw_results")
    os.makedirs(out_dir, exist_ok=True)
    per_conf_dir = os.path.join(out_dir, "per_conference")
    os.makedirs(per_conf_dir, exist_ok=True)

    # Load paper IDs filter
    paper_ids: Optional[set[str]] = None
    if args.paper_ids:
        with open(args.paper_ids, "r", encoding="utf-8") as f:
            paper_ids = {line.strip() for line in f if line.strip()}
        print(f"[INFO] Filtering to {len(paper_ids)} paper IDs from: {args.paper_ids}")

    # Select conferences
    confs_to_run = (
        {args.conf: CONFERENCE_LLM_FILES[args.conf]}
        if args.conf and args.conf in CONFERENCE_LLM_FILES
        else CONFERENCE_LLM_FILES
    )

    all_rows: list[dict] = []

    for conf, llm_map in confs_to_run.items():
        print(f"\n[CONF] {conf.upper()}")

        # Select LLM types
        llm_items = (
            {args.llm: llm_map[args.llm]}.items()
            if args.llm and args.llm in llm_map
            else llm_map.items()
        )

        for llm_type, fpath in llm_items:
            rows = process_file(conf, llm_type, fpath, paper_ids=paper_ids)
            all_rows.extend(rows)

            # Per-file CSV
            if rows:
                conf_csv = os.path.join(per_conf_dir, f"{conf}_{llm_type}_metrics.csv")
                save_csv(rows, conf_csv)
                print(f"    -> CSV: {conf_csv}")

    if not all_rows:
        print("[FATAL] Khong co du lieu! Kiem tra lai duong dan file output.")
        sys.exit(1)

    # Combined CSV
    combined_csv = os.path.join(out_dir, "all_conferences_all_llms.csv")
    save_csv(all_rows, combined_csv)
    print(f"\n[INFO] Combined CSV: {combined_csv}")

    # Summary comparison CSV
    summary_rows = build_summary_comparison(all_rows)
    summary_csv  = os.path.join(out_dir, "summary_llm_vs_human.csv")
    save_summary_csv(summary_rows, summary_csv)
    print(f"[INFO] Summary CSV:  {summary_csv}")

    # Report
    report = build_report(all_rows, summary_rows)
    report_path = os.path.join(out_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[INFO] Report:       {report_path}")

    print("\n" + report)
    print(f"\n[DONE] Ket qua da luu vao: {out_dir}")


if __name__ == "__main__":
    main()

