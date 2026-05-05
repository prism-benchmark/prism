#!/usr/bin/env python3
"""
Compare Task 3 scores across 2, 3, 4 or more folder runs.

Finds papers that have task3_result.json in all given directories,
computes per-paper aggregates (mean/max final_score) and label
distributions per folder, and outputs CSV + JSON with summary
(correlation matrix, per-folder averages).

Usage:
  python scripts/compare_human_llm_scores.py --dirs output/iclr2024 output/iclr2024_granite
  python scripts/compare_human_llm_scores.py --dirs d1 d2 d3 d4 -o comparison.json
  python scripts/compare_human_llm_scores.py --dirs d1 d2 d3 --names human granite other --format csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional: scipy for correlation; fallback to simple stats
try:
    from scipy.stats import pearsonr, spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _normalize_score(mean_score: float) -> float:
    """NS = (mean_score + 2) / 4, clipped to [0, 1]."""
    return round(max(0.0, min(1.0, (mean_score + 2.0) / 4.0)), 4)


def _support_rate(scores: List[float], threshold: float = 1.0) -> float:
    """Fraction of scores >= threshold.  SR (threshold=1) or SSR (threshold=2)."""
    if not scores:
        return 0.0
    return round(sum(1.0 for s in scores if s >= threshold) / len(scores), 4)


def load_task3(path: Path) -> Optional[Dict[str, Any]]:
    """Load task3_result.json; return None if missing or invalid."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def scores_from_task3(data: Dict[str, Any]) -> Tuple[List[float], Dict[str, int]]:
    """
    Extract list of final_score from aggregated and label counts from pair_results.
    Returns (list of final_score, label_counts dict).
    """
    aggregated = data.get("aggregated") or []
    scores: List[float] = []
    for item in aggregated:
        if isinstance(item, dict):
            s = item.get("final_score")
            if s is not None:
                try:
                    scores.append(float(s))
                except (TypeError, ValueError):
                    pass

    label_counts: Dict[str, int] = {}
    for item in data.get("pair_results") or []:
        if isinstance(item, dict):
            label = item.get("label")
            if isinstance(label, str) and label.strip():
                label_counts[label.strip()] = label_counts.get(label.strip(), 0) + 1

    return scores, label_counts


def per_paper_stats(
    dirs_with_names: List[Tuple[Path, str]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, int]]]:
    """
    For each paper that has task3_result.json in ALL given directories,
    compute per-paper stats for each folder.
    dirs_with_names: list of (directory Path, display name).
    Returns (list of row dicts, list of label_totals per folder).
    """
    if not dirs_with_names:
        return [], []

    # Paper IDs present in every directory (with task3_result.json)
    all_ids: Optional[set] = None
    for dpath, _ in dirs_with_names:
        ids = set()
        for sub in dpath.iterdir():
            if sub.is_dir() and (sub / "task3_result.json").exists():
                ids.add(sub.name)
        if all_ids is None:
            all_ids = ids
        else:
            all_ids &= ids
    common = sorted(all_ids or set())

    rows: List[Dict[str, Any]] = []
    label_totals_list: List[Dict[str, int]] = [{} for _ in dirs_with_names]

    for paper_id in common:
        row: Dict[str, Any] = {"paper_id": paper_id}
        per_dir_scores: List[Tuple[int, float, float]] = []  # (n, mean, max) per dir

        all_ok = True
        for i, (dpath, name) in enumerate(dirs_with_names):
            path = dpath / paper_id / "task3_result.json"
            data = load_task3(path)
            if data is None:
                all_ok = False
                break
            scores, labels = scores_from_task3(data)
            for k, v in labels.items():
                label_totals_list[i][k] = label_totals_list[i].get(k, 0) + v

            n = len(scores)
            mean_s = sum(scores) / n if n else 0.0
            max_s = max(scores) if scores else 0.0
            per_dir_scores.append((n, mean_s, max_s))
            row[f"n_claims_{name}"] = n
            row[f"mean_score_{name}"] = round(mean_s, 4)
            row[f"max_score_{name}"] = round(max_s, 4)
            row[f"ns_{name}"] = _normalize_score(mean_s)
            row[f"sr_{name}"] = _support_rate(scores)
            row[f"ssr_{name}"] = _support_rate(scores, threshold=2.0)

        if not all_ok:
            continue

        # Pairwise diffs relative to first folder (mean and max)
        ref_mean = per_dir_scores[0][1]
        ref_max = per_dir_scores[0][2]
        for i in range(1, len(dirs_with_names)):
            name = dirs_with_names[i][1]
            row[f"diff_mean_{name}"] = round(per_dir_scores[i][1] - ref_mean, 4)
            row[f"diff_max_{name}"] = round(per_dir_scores[i][2] - ref_max, 4)

        rows.append(row)

    return rows, label_totals_list


def correlation(x: List[float], y: List[float]) -> Optional[Tuple[float, float]]:
    """Pearson (r, p-value); returns None if not computable or no scipy."""
    if len(x) != len(y) or len(x) < 2:
        return None
    if not HAS_SCIPY:
        return None
    try:
        r, p = pearsonr(x, y)
        return (float(r), float(p))
    except Exception:
        return None


def correlation_matrix(
    names: List[str], rows: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Compute Pearson correlation matrix for mean_score_* columns."""
    if not HAS_SCIPY or len(names) < 2:
        return {}
    matrix: Dict[str, Dict[str, Any]] = {}
    for i, name_i in enumerate(names):
        matrix[name_i] = {}
        key_i = f"mean_score_{name_i}"
        vec_i = [r[key_i] for r in rows]
        for j, name_j in enumerate(names):
            if i == j:
                matrix[name_i][name_j] = 1.0
                continue
            key_j = f"mean_score_{name_j}"
            vec_j = [r[key_j] for r in rows]
            corr = correlation(vec_i, vec_j)
            if corr is not None:
                matrix[name_i][name_j] = round(corr[0], 4)
            else:
                matrix[name_i][name_j] = None
    return matrix


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Task 3 scores across 2, 3, 4 or more folder runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Two or more directories containing task3_result.json per paper (e.g. output/iclr2024 output/iclr2024_granite output/iclr2024_other)",
    )
    parser.add_argument(
        "--names",
        type=str,
        nargs="+",
        default=None,
        help="Display names for each directory (same order as --dirs); default is directory stem",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file (default: comparison.json or comparison.csv)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv", "both"],
        default="json",
        help="Output format (default: json)",
    )
    args = parser.parse_args()

    if len(args.dirs) < 2:
        print("Error: --dirs must list at least 2 directories.", file=sys.stderr)
        return 1

    for d in args.dirs:
        if not d.is_dir():
            print(f"Error: not a directory: {d}", file=sys.stderr)
            return 1

    if args.names is not None and len(args.names) != len(args.dirs):
        print("Error: --names must have the same number of items as --dirs.", file=sys.stderr)
        return 1

    names = args.names if args.names is not None else [d.name for d in args.dirs]
    dirs_with_names = list(zip(args.dirs, names))

    rows, label_totals_list = per_paper_stats(dirs_with_names)

    if not rows:
        print("No papers with task3 results in all given directories found.", file=sys.stderr)
        return 0

    # Summary: per-folder averages and label distributions
    summary: Dict[str, Any] = {"n_papers": len(rows)}
    for name in names:
        mean_key = f"mean_score_{name}"
        max_key = f"max_score_{name}"
        avg_mean = sum(r[mean_key] for r in rows) / len(rows)
        avg_max = sum(r[max_key] for r in rows) / len(rows)
        summary[f"avg_mean_score_{name}"] = round(avg_mean, 4)
        summary[f"avg_max_score_{name}"] = round(avg_max, 4)

    # Normalized metrics per folder
    for name in names:
        ns_key = f"ns_{name}"
        sr_key = f"sr_{name}"
        ssr_key = f"ssr_{name}"
        summary[f"avg_ns_{name}"] = round(sum(r[ns_key] for r in rows) / len(rows), 4)
        summary[f"avg_sr_{name}"] = round(sum(r[sr_key] for r in rows) / len(rows), 4)
        summary[f"avg_ssr_{name}"] = round(sum(r[ssr_key] for r in rows) / len(rows), 4)

    for name, totals in zip(names, label_totals_list):
        summary[f"label_distribution_{name}"] = totals

    # Pairwise avg diffs (relative to first folder)
    for i in range(1, len(names)):
        name = names[i]
        diff_mean_key = f"diff_mean_{name}"
        diff_max_key = f"diff_max_{name}"
        summary[f"avg_diff_mean_{name}"] = round(sum(r[diff_mean_key] for r in rows) / len(rows), 4)
        summary[f"avg_diff_max_{name}"] = round(sum(r[diff_max_key] for r in rows) / len(rows), 4)

    corr_matrix = correlation_matrix(names, rows)
    if corr_matrix:
        summary["pearson_correlation_matrix_mean"] = corr_matrix

    out_path = args.output
    if out_path is None:
        out_path = Path("comparison.json" if args.format == "json" else "comparison.csv" if args.format == "csv" else "comparison")

    if args.format in ("json", "both"):
        p = out_path if args.format == "json" else out_path.with_suffix(".json")
        if args.format == "both":
            p = out_path.parent / (out_path.stem + ".json")
        with p.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Wrote summary JSON to {p}", file=sys.stderr)

    if args.format in ("csv", "both"):
        p = out_path if args.format == "csv" else out_path.with_suffix(".csv")
        if args.format == "both":
            p = out_path.parent / (out_path.stem + ".csv")
        # Single-row summary CSV: flatten scalar fields only
        scalar_keys = [k for k in summary if not isinstance(summary[k], dict)]
        with p.open("w", encoding="utf-8") as f:
            f.write(",".join(scalar_keys) + "\n")
            f.write(",".join(str(summary[k]) for k in scalar_keys) + "\n")
        print(f"Wrote summary CSV to {p}", file=sys.stderr)

    # Print short summary to stdout
    print(f"Papers compared: {summary['n_papers']}")
    for name in names:
        print(f"Avg mean score ({name}): {summary[f'avg_mean_score_{name}']}")
    for name in names:
        ns = summary[f"avg_ns_{name}"]
        sr = summary[f"avg_sr_{name}"]
        ssr = summary[f"avg_ssr_{name}"]
        print(f"Normalized ({name}):  NS={ns:.4f}  SR={sr:.4f}  SSR={ssr:.4f}")
    for i in range(1, len(names)):
        name = names[i]
        print(f"Avg diff mean vs {names[0]} ({name}): {summary[f'avg_diff_mean_{name}']}")
        print(f"Avg diff max vs {names[0]} ({name}):  {summary[f'avg_diff_max_{name}']}")
    if "pearson_correlation_matrix_mean" in summary:
        print("Pearson correlation matrix (mean scores):")
        for row_name, col_dict in summary["pearson_correlation_matrix_mean"].items():
            print(f"  {row_name}: {col_dict}")
    for name in names:
        print(f"Label distribution ({name}): {summary[f'label_distribution_{name}']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
