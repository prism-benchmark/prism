"""
compute_per_reviewer_metrics.py
════════════════════════════════════════════════════════════════════════════════
Tính metric constructiveness cho từng reviewer (human lẻ + từng LLM)
theo từng conference riêng biệt.

Output
------
  output/analysis_all/all_reviewers_all_conferences.csv   ← bảng tổng hợp
  output/analysis_all/per_conference/<conf>_metrics.csv   ← bảng từng conf
  output/analysis_all/report.txt                          ← báo cáo text

Usage
-----
  python compute_per_reviewer_metrics.py
  python compute_per_reviewer_metrics.py --conf iclr2024
  python compute_per_reviewer_metrics.py --paper-ids path/to/ids.txt
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
_HERE     = os.path.dirname(os.path.abspath(__file__))
_OUT_ROOT = os.path.join(_HERE, "output")

# Conferences và file paths tương ứng
#   key  = tên conference
#   value = dict  reviewer_type → đường dẫn JSONL
CONFERENCE_FILES: dict[str, dict[str, str]] = {
    "iclr2024": {
        "human":       os.path.join(_OUT_ROOT, "iclr2024", "human",       "all_results_lite.jsonl"),
        "sea":         os.path.join(_OUT_ROOT, "iclr2024", "sea",         "all_results_lite.jsonl"),
        "reviewer2":   os.path.join(_OUT_ROOT, "iclr2024", "reviewer2",   "all_results_lite.jsonl"),
        "deepreview":  os.path.join(_OUT_ROOT, "iclr2024", "deepreview",  "all_results_lite.jsonl"),
        "tree":        os.path.join(_OUT_ROOT, "iclr2024", "tree",        "all_results_lite.jsonl"),
        "cyclereview": os.path.join(_OUT_ROOT, "iclr2024", "cyclereview", "all_results_lite.jsonl"),
    },
    "iclr2025": {
        "human":       os.path.join(_OUT_ROOT, "iclr2025", "human",       "all_results_lite.jsonl"),
        "sea":         os.path.join(_OUT_ROOT, "iclr2025", "sea",         "all_results_lite.jsonl"),
        "reviewer2":   os.path.join(_OUT_ROOT, "iclr2025", "reviewer2",   "all_results_lite.jsonl"),
        "deepreview":  os.path.join(_OUT_ROOT, "iclr2025", "deepreview",  "all_results_lite.jsonl"),
        "tree":        os.path.join(_OUT_ROOT, "iclr2025", "tree",        "all_results_lite.jsonl"),
        "cyclereview": os.path.join(_OUT_ROOT, "iclr2025", "cyclereview", "all_results_lite.jsonl"),
    },
    "iclr2026": {
        "human":       os.path.join(_OUT_ROOT, "iclr2026", "human",       "all_results_lite.jsonl"),
        "sea":         os.path.join(_OUT_ROOT, "iclr2026", "sea",         "all_results_lite.jsonl"),
        "reviewer2":   os.path.join(_OUT_ROOT, "iclr2026", "reviewer2",   "all_results_lite.jsonl"),
        "deepreview":  os.path.join(_OUT_ROOT, "iclr2026", "deepreview",  "all_results_lite.jsonl"),
        "tree":        os.path.join(_OUT_ROOT, "iclr2026", "tree",        "all_results_lite.jsonl"),
        "cyclereview": os.path.join(_OUT_ROOT, "iclr2026", "cyclereview", "all_results_lite.jsonl"),
    },
    "icml2025": {
        "human":       os.path.join(_OUT_ROOT, "icml2025", "human",       "all_results_lite.jsonl"),
        "sea":         os.path.join(_OUT_ROOT, "icml2025", "sea",         "all_results_lite.jsonl"),
        "reviewer2":   os.path.join(_OUT_ROOT, "icml2025", "reviewer2",   "all_results_lite.jsonl"),
        "deepreview":  os.path.join(_OUT_ROOT, "icml2025", "deepreview",  "all_results_lite.jsonl"),
        "tree":        os.path.join(_OUT_ROOT, "icml2025", "tree",        "all_results_lite.jsonl"),
        "cyclereview": os.path.join(_OUT_ROOT, "icml2025", "cyclereview", "all_results_lite.jsonl"),
    },
    "neurips2025": {
        "human":       os.path.join(_OUT_ROOT, "neurips2025", "human",       "all_results_lite.jsonl"),
        "sea":         os.path.join(_OUT_ROOT, "neurips2025", "sea",         "all_results_lite.jsonl"),
        "reviewer2":   os.path.join(_OUT_ROOT, "neurips2025", "reviewer2",   "all_results_lite.jsonl"),
        "deepreview":  os.path.join(_OUT_ROOT, "neurips2025", "deepreview",  "all_results_lite.jsonl"),
        "tree":        os.path.join(_OUT_ROOT, "neurips2025", "tree",        "all_results_lite.jsonl"),
        "cyclereview": os.path.join(_OUT_ROOT, "neurips2025", "cyclereview", "all_results_lite.jsonl"),
    },
}

METRIC_KEYS = [
    "MCS", "AR", "SD", "CD",
    "D1_actionability_mean", "D2_specificity_mean",
    "D3_justification_mean", "D4_solution_mean", "D5_tone_mean",
]

METRIC_LABELS = {
    "MCS":                   "Mean Constructiveness Score",
    "AR":                    "Actionability Ratio",
    "SD":                    "Solution Density",
    "CD":                    "Constructiveness Density",
    "D1_actionability_mean": "D1 Actionability",
    "D2_specificity_mean":   "D2 Specificity",
    "D3_justification_mean": "D3 Justification",
    "D4_solution_mean":      "D4 Solution",
    "D5_tone_mean":          "D5 Tone",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    """Load JSONL, bỏ qua dòng lỗi."""
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
    """Tính mean, std, min, max, n cho danh sách giá trị."""
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


# ── Core extraction ───────────────────────────────────────────────────────────

def extract_human_metrics(records: list[dict]) -> dict[str, list[dict]]:
    """Trả về: {reviewer_id → list of metrics dict (1 per paper)}."""
    result: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        for rev in rec.get("reviewers", []):
            rid = rev.get("reviewer_id")
            m   = rev.get("metrics")
            if rid and m and (m.get("n_arcs") or 0) > 0:
                result[rid].append(m)
    return dict(result)


def extract_llm_metrics(records: list[dict]) -> dict[str, list[dict]]:
    """Trả về: {reviewer_id → list of metrics dict (1 per paper)}.
    
    LLM records có metrics ở top-level, reviewer_id ở top-level.
    """
    result: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        rid = rec.get("reviewer_id", "Unknown_LLM")
        m   = rec.get("metrics")
        if m and (m.get("n_arcs") or 0) > 0:
            result[rid].append(m)
    return dict(result)


def aggregate_reviewer(metrics_list: list[dict]) -> dict:
    """Tổng hợp metrics từ list paper-level metrics của 1 reviewer."""
    agg_data: dict[str, list[float]] = {k: [] for k in METRIC_KEYS}
    n_arcs_list: list[int] = []

    for m in metrics_list:
        for k in METRIC_KEYS:
            v = _safe(m.get(k))
            if v is not None:
                agg_data[k].append(v)
        n_arcs_list.append(int(m.get("n_arcs", 0)))

    result: dict[str, Any] = {
        "n_papers":    len(metrics_list),
        "n_arcs_mean": round(float(np.mean(n_arcs_list)), 2) if n_arcs_list else 0,
        "n_arcs_total": sum(n_arcs_list),
    }
    for k in METRIC_KEYS:
        s = _agg(agg_data[k])
        result[f"{k}_mean"] = s["mean"]
        result[f"{k}_std"]  = s["std"]
        result[f"{k}_n"]    = s["n"]
    return result


# ── Per-conference computation ────────────────────────────────────────────────

def compute_conference_metrics(
    conf: str,
    file_map: dict[str, str],
    paper_ids: Optional[set[str]] = None,
) -> list[dict]:
    """
    Tính metrics cho tất cả reviewer (human + LLM) của một conference.
    Trả về list of row dicts để xuất CSV.
    """
    rows: list[dict] = []

    for rtype, fpath in file_map.items():
        records = load_jsonl(fpath)
        if not records:
            print(f"  [SKIP] {conf}/{rtype}: file không tồn tại hoặc rỗng ({fpath})")
            continue

        # Lọc theo paper_ids nếu có
        if paper_ids:
            records = [r for r in records if r.get("paper_id") in paper_ids]
            if not records:
                print(f"  [SKIP] {conf}/{rtype}: 0 records sau filter")
                continue

        print(f"  [OK]   {conf}/{rtype}: {len(records)} papers")

        if rtype == "human":
            reviewer_map = extract_human_metrics(records)
        else:
            reviewer_map = extract_llm_metrics(records)

        for reviewer_id, mlist in sorted(reviewer_map.items()):
            agg = aggregate_reviewer(mlist)
            row = {
                "conference":   conf,
                "reviewer_type": rtype,
                "reviewer_id":  reviewer_id,
                "n_papers":     agg["n_papers"],
                "n_arcs_total": agg["n_arcs_total"],
                "n_arcs_mean":  agg["n_arcs_mean"],
            }
            for k in METRIC_KEYS:
                row[f"{k}_mean"] = agg.get(f"{k}_mean")
                row[f"{k}_std"]  = agg.get(f"{k}_std")
            rows.append(row)

    return rows


# ── CSV / report output ───────────────────────────────────────────────────────

def _csv_fieldnames() -> list[str]:
    base = ["conference", "reviewer_type", "reviewer_id",
            "n_papers", "n_arcs_total", "n_arcs_mean"]
    for k in METRIC_KEYS:
        base += [f"{k}_mean", f"{k}_std"]
    return base


def save_csv(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = _csv_fieldnames()
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if row.get(k) is None else row[k]) for k in fields})


_SEP  = "=" * 100
_SEP2 = "-" * 100

def _fmt(v: Any, w: int = 9) -> str:
    if v is None:
        return "N/A".rjust(w)
    try:
        return f"{float(v):.4f}".rjust(w)
    except (TypeError, ValueError):
        return str(v).rjust(w)


def _fmt_mean_std(mean: Any, std: Any) -> str:
    if mean is None:
        return "N/A".rjust(14)
    if std is None:
        return f"{float(mean):.4f}".rjust(14)
    return f"{float(mean):.4f}±{float(std):.4f}".rjust(14)


def build_report(all_rows: list[dict]) -> str:
    lines: list[str] = []
    a = lines.append

    a(_SEP)
    a("  CONSTRUCTIVENESS METRICS — Per Reviewer × Per Conference")
    a(_SEP)

    # Group by conference
    conf_groups: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        conf_groups[row["conference"]].append(row)

    for conf in sorted(conf_groups.keys()):
        rows = conf_groups[conf]
        a("")
        a(_SEP)
        a(f"  CONFERENCE: {conf.upper()}  ({len(rows)} reviewers with data)")
        a(_SEP)

        # Header
        hdr = f"  {'Reviewer':<30} {'N_papers':>8} {'N_arcs':>7}"
        for k in METRIC_KEYS:
            short = k.replace("_mean", "").replace("D1_actionability", "D1").\
                replace("D2_specificity", "D2").replace("D3_justification", "D3").\
                replace("D4_solution", "D4").replace("D5_tone", "D5")
            hdr += f" {short:>14}"
        a(hdr)
        a(_SEP2)

        # Group by reviewer_type for cleaner display
        rtype_groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            rtype_groups[row["reviewer_type"]].append(row)

        RTYPE_ORDER = ["human", "sea", "reviewer2", "deepreview", "tree", "cyclereview"]
        for rtype in RTYPE_ORDER:
            if rtype not in rtype_groups:
                continue
            for row in sorted(rtype_groups[rtype], key=lambda r: r["reviewer_id"]):
                rid     = row["reviewer_id"]
                n_p     = row["n_papers"]
                n_arcs  = row["n_arcs_total"]
                line = f"  {rid:<30} {n_p:>8} {n_arcs:>7}"
                for k in METRIC_KEYS:
                    line += _fmt_mean_std(row.get(f"{k}_mean"), row.get(f"{k}_std"))
                a(line)
            a("")  # blank line between reviewer types

    a(_SEP)
    a("")
    return "\n".join(lines)


def build_summary_report(all_rows: list[dict]) -> str:
    """Bảng tóm tắt: với mỗi reviewer_type, trung bình MCS qua tất cả conference."""
    lines: list[str] = []
    a = lines.append

    a(_SEP)
    a("  SUMMARY — Average MCS, AR, SD, CD per Reviewer Type (across all conferences)")
    a(_SEP)

    rtype_groups: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        rtype_groups[row["reviewer_type"]].append(row)

    header = f"  {'Reviewer Type':<30} {'Reviewer ID':<25} {'Confs':>6} {'N_papers':>8}"
    for k in ["MCS", "AR", "SD", "CD"]:
        header += f" {k:>14}"
    a(header)
    a(_SEP2)

    RTYPE_ORDER = ["human", "sea", "reviewer2", "deepreview", "tree", "cyclereview"]
    for rtype in RTYPE_ORDER:
        if rtype not in rtype_groups:
            continue

        # Group by reviewer_id within this type
        rid_groups: dict[str, list[dict]] = defaultdict(list)
        for row in rtype_groups[rtype]:
            rid_groups[row["reviewer_id"]].append(row)

        for rid in sorted(rid_groups.keys()):
            rrows = rid_groups[rid]
            n_confs = len(rrows)
            total_papers = sum(r["n_papers"] for r in rrows)

            line = f"  {rtype:<30} {rid:<25} {n_confs:>6} {total_papers:>8}"
            for k in ["MCS", "AR", "SD", "CD"]:
                vals = [r[f"{k}_mean"] for r in rrows if r.get(f"{k}_mean") is not None]
                if vals:
                    m = float(np.mean(vals))
                    s = float(np.std(vals))
                    line += f" {m:.4f}±{s:.4f}".rjust(14)
                else:
                    line += "N/A".rjust(14)
            a(line)
        a("")

    a(_SEP)
    return "\n".join(lines)


def build_heatmap_csv(all_rows: list[dict], path: str) -> None:
    """Xuất CSV dạng heatmap: rows = reviewer_id, cols = conference×metric."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    confs = sorted(set(r["conference"] for r in all_rows))
    reviewer_ids = sorted(set(r["reviewer_id"] for r in all_rows))

    # Build lookup: (reviewer_id, conf) → row
    lookup: dict[tuple, dict] = {}
    for row in all_rows:
        key = (row["reviewer_id"], row["conference"])
        lookup[key] = row

    # Fieldnames
    fields = ["reviewer_id", "reviewer_type"]
    for conf in confs:
        for k in METRIC_KEYS:
            fields.append(f"{conf}_{k}_mean")

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for rid in reviewer_ids:
            # Get reviewer_type from first occurrence
            rtype = next((r["reviewer_type"] for r in all_rows if r["reviewer_id"] == rid), "")
            out: dict[str, Any] = {"reviewer_id": rid, "reviewer_type": rtype}
            for conf in confs:
                row = lookup.get((rid, conf))
                for k in METRIC_KEYS:
                    col = f"{conf}_{k}_mean"
                    out[col] = (row.get(f"{k}_mean") if row else None)
            w.writerow({k: ("" if out.get(k) is None else out[k]) for k in fields})


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Tính metrics per reviewer per conference.")
    p.add_argument("--conf", default=None,
                   help="Chỉ chạy cho 1 conference (vd: iclr2024). Mặc định: tất cả.")
    p.add_argument("--paper-ids", default=None,
                   help="Path đến file txt chứa paper IDs cần lọc (1 ID/dòng).")
    p.add_argument("--out-dir", default=None,
                   help="Thư mục output tùy chỉnh. Mặc định: output/analysis_all/")
    return p.parse_args()


def main():
    args = parse_args()

    # Output directory
    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = os.path.join(_OUT_ROOT, "analysis_all")

    os.makedirs(out_dir, exist_ok=True)
    per_conf_dir = os.path.join(out_dir, "per_conference")
    os.makedirs(per_conf_dir, exist_ok=True)

    # Load paper IDs filter
    paper_ids: Optional[set[str]] = None
    if args.paper_ids:
        with open(args.paper_ids, "r", encoding="utf-8") as f:
            paper_ids = {line.strip() for line in f if line.strip()}
        print(f"[INFO] Lọc theo {len(paper_ids)} paper IDs từ: {args.paper_ids}")

    # Select conferences
    confs_to_run = (
        {args.conf: CONFERENCE_FILES[args.conf]}
        if args.conf and args.conf in CONFERENCE_FILES
        else CONFERENCE_FILES
    )

    all_rows: list[dict] = []

    for conf, file_map in confs_to_run.items():
        print(f"\n[CONF] {conf.upper()}")
        conf_rows = compute_conference_metrics(conf, file_map, paper_ids=paper_ids)
        all_rows.extend(conf_rows)

        # Save per-conference CSV
        conf_csv = os.path.join(per_conf_dir, f"{conf}_metrics.csv")
        save_csv(conf_rows, conf_csv)
        print(f"  -> CSV saved: {conf_csv}")

    if not all_rows:
        print("[FATAL] Không có dữ liệu nào! Kiểm tra lại đường dẫn file output.")
        sys.exit(1)

    # Save combined CSV
    combined_csv = os.path.join(out_dir, "all_reviewers_all_conferences.csv")
    save_csv(all_rows, combined_csv)
    print(f"\n[INFO] Combined CSV saved: {combined_csv}")

    # Save heatmap CSV (reviewer vs conference)
    heatmap_csv = os.path.join(out_dir, "heatmap_reviewer_x_conference.csv")
    build_heatmap_csv(all_rows, heatmap_csv)
    print(f"[INFO] Heatmap CSV saved:  {heatmap_csv}")

    # Build and save report
    report = build_report(all_rows)
    summary = build_summary_report(all_rows)
    full_report = report + "\n\n" + summary

    report_path = os.path.join(out_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"[INFO] Report saved:       {report_path}")

    print("\n" + full_report)

    print(f"\n[DONE] Kết quả đã lưu vào: {out_dir}")


if __name__ == "__main__":
    main()

