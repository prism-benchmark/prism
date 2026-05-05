"""
Statistical analysis for Human vs. LLM constructiveness comparison.

Implements:
  - Wilcoxon signed-rank test (paired, non-parametric)
  - Cliff's delta (non-parametric effect size)
  - Bootstrap 95% confidence intervals
  - Holm-Bonferroni correction for multiple comparisons
  - Subgroup stratification (by decision, quality tier, confidence)
"""

from __future__ import annotations

import json
import warnings
from typing import Any, Optional

import numpy as np

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


METRIC_KEYS = [
    "MCS", "AR", "SD", "CD",
    "D1_actionability_mean", "D2_specificity_mean",
    "D3_justification_mean", "D4_solution_mean", "D5_tone_mean",
]

CLIFFS_DELTA_THRESHOLDS = {
    "negligible": 0.147,
    "small": 0.33,
    "medium": 0.474,
}


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    """Compute Cliff's delta effect size (non-parametric).

    Returns (delta, magnitude) where magnitude is one of:
    negligible, small, medium, large.
    """
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0, "negligible"

    dominance = 0.0
    for xi in x:
        for yj in y:
            if xi > yj:
                dominance += 1.0
            elif xi < yj:
                dominance -= 1.0

    delta = dominance / (n_x * n_y)
    abs_delta = abs(delta)

    if abs_delta < CLIFFS_DELTA_THRESHOLDS["negligible"]:
        magnitude = "negligible"
    elif abs_delta < CLIFFS_DELTA_THRESHOLDS["small"]:
        magnitude = "small"
    elif abs_delta < CLIFFS_DELTA_THRESHOLDS["medium"]:
        magnitude = "medium"
    else:
        magnitude = "large"

    return round(delta, 4), magnitude


def cliffs_delta_fast(x: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    """Vectorized Cliff's delta for large samples (N > 1000)."""
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0, "negligible"

    # Broadcast comparison
    diff = x[:, None] - y[None, :]
    dominance = np.sum(np.sign(diff))
    delta = float(dominance) / (n_x * n_y)
    abs_delta = abs(delta)

    if abs_delta < CLIFFS_DELTA_THRESHOLDS["negligible"]:
        magnitude = "negligible"
    elif abs_delta < CLIFFS_DELTA_THRESHOLDS["small"]:
        magnitude = "small"
    elif abs_delta < CLIFFS_DELTA_THRESHOLDS["medium"]:
        magnitude = "medium"
    else:
        magnitude = "large"

    return round(delta, 4), magnitude


def bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    rng_seed: int = 42,
) -> dict[str, float]:
    """Bootstrap confidence interval for the mean difference (x - y).

    x and y must be paired (same length).
    """
    assert len(x) == len(y), "x and y must have the same length for paired bootstrap"
    n = len(x)
    if n == 0:
        return {"mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    rng = np.random.default_rng(rng_seed)
    diffs = x - y
    observed_mean = float(np.mean(diffs))

    boot_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = np.mean(diffs[idx])

    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_means, 100 * alpha))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha)))

    return {
        "mean_diff": round(observed_mean, 4),
        "ci_lower": round(lower, 4),
        "ci_upper": round(upper, 4),
    }


def holm_bonferroni(p_values: list[tuple[str, float]]) -> list[dict[str, Any]]:
    """Holm-Bonferroni correction for multiple hypothesis tests.

    Args:
        p_values: list of (test_name, raw_p_value) tuples.

    Returns:
        list of dicts with test_name, raw_p, adjusted_p, significant (at alpha=0.05).
    """
    m = len(p_values)
    if m == 0:
        return []

    sorted_tests = sorted(p_values, key=lambda t: t[1])
    results = []

    for rank, (name, raw_p) in enumerate(sorted_tests):
        adjusted_p = min(raw_p * (m - rank), 1.0)
        results.append({
            "test_name": name,
            "raw_p": round(raw_p, 6),
            "adjusted_p": round(adjusted_p, 6),
            "significant": adjusted_p < 0.05,
        })

    # Enforce monotonicity: adjusted p-values should be non-decreasing
    for i in range(1, len(results)):
        if results[i]["adjusted_p"] < results[i - 1]["adjusted_p"]:
            results[i]["adjusted_p"] = results[i - 1]["adjusted_p"]
            results[i]["significant"] = results[i]["adjusted_p"] < 0.05

    return results


def _extract_paired_metric(
    records: list[dict],
    metric_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract paired (LLM, Human_avg) arrays for a given metric from JSONL records."""
    llm_vals = []
    human_avg_vals = []

    for record in records:
        comparison = record.get("comparison", {})
        llm_val = comparison.get(f"llm_{metric_key}")
        human_val = comparison.get(f"human_avg_{metric_key}")

        if llm_val is not None and human_val is not None:
            llm_vals.append(float(llm_val))
            human_avg_vals.append(float(human_val))

    return np.array(llm_vals), np.array(human_avg_vals)


def run_full_analysis(
    records: list[dict],
    n_bootstrap: int = 10_000,
) -> dict[str, Any]:
    """Run complete statistical analysis on cached constructiveness results.

    Args:
        records: list of paper-level result dicts (from JSONL).

    Returns:
        Comprehensive analysis dict with per-metric tests, corrections, and summary.
    """
    if not HAS_SCIPY:
        raise ImportError(
            "scipy is required for statistical analysis. Install: pip install scipy"
        )

    n_papers = len(records)
    print(f"[STAT] Running analysis on {n_papers} papers...")

    per_metric_results: dict[str, dict] = {}
    raw_p_values: list[tuple[str, float]] = []

    for metric_key in METRIC_KEYS:
        llm_arr, human_arr = _extract_paired_metric(records, metric_key)
        n_valid = len(llm_arr)

        if n_valid < 10:
            per_metric_results[metric_key] = {
                "n_valid_pairs": n_valid,
                "skipped": True,
                "reason": "insufficient pairs (< 10)",
            }
            continue

        # Descriptive stats
        llm_mean = float(np.mean(llm_arr))
        human_mean = float(np.mean(human_arr))
        diff_arr = llm_arr - human_arr

        # Wilcoxon signed-rank test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                stat, p_value = scipy_stats.wilcoxon(diff_arr, alternative="two-sided")
            except ValueError:
                stat, p_value = 0.0, 1.0

        raw_p_values.append((metric_key, p_value))

        # Cliff's delta (use fast version for large N)
        if n_valid > 500:
            delta, magnitude = cliffs_delta_fast(llm_arr, human_arr)
        else:
            delta, magnitude = cliffs_delta(llm_arr, human_arr)

        # Bootstrap CI
        boot = bootstrap_ci(llm_arr, human_arr, n_bootstrap=n_bootstrap)

        per_metric_results[metric_key] = {
            "n_valid_pairs": n_valid,
            "llm_mean": round(llm_mean, 4),
            "human_mean": round(human_mean, 4),
            "mean_diff": round(llm_mean - human_mean, 4),
            "wilcoxon_statistic": round(float(stat), 2),
            "wilcoxon_p": round(p_value, 6),
            "cliffs_delta": delta,
            "cliffs_delta_magnitude": magnitude,
            "bootstrap_95ci": boot,
        }

    # Multiple comparison correction
    corrected = holm_bonferroni(raw_p_values)
    for item in corrected:
        key = item["test_name"]
        if key in per_metric_results and not per_metric_results[key].get("skipped"):
            per_metric_results[key]["adjusted_p"] = item["adjusted_p"]
            per_metric_results[key]["significant_after_correction"] = item["significant"]

    # Winner summary
    summary = _build_winner_summary(per_metric_results)

    return {
        "n_papers": n_papers,
        "per_metric": per_metric_results,
        "holm_bonferroni_correction": corrected,
        "summary": summary,
    }


def run_subgroup_analysis(
    records: list[dict],
    n_bootstrap: int = 5_000,
) -> dict[str, Any]:
    """Stratified analysis by paper decision and quality tier."""
    # Group by decision
    accept_records = [r for r in records if _is_accept(r)]
    reject_records = [r for r in records if not _is_accept(r)]

    # Group by quality tier (avg rating)
    ratings = [
        r["metadata"]["avg_rating"]
        for r in records
        if r.get("metadata", {}).get("avg_rating") is not None
    ]

    result: dict[str, Any] = {}

    if accept_records and reject_records:
        result["by_decision"] = {
            "accept": _subgroup_summary(accept_records, n_bootstrap),
            "reject": _subgroup_summary(reject_records, n_bootstrap),
        }

    if ratings:
        sorted_ratings = sorted(ratings)
        n = len(sorted_ratings)
        q25 = sorted_ratings[n // 4]
        q75 = sorted_ratings[3 * n // 4]

        bottom = [r for r in records if (r.get("metadata", {}).get("avg_rating") or 0) <= q25]
        top = [r for r in records if (r.get("metadata", {}).get("avg_rating") or 0) >= q75]
        mid = [r for r in records if q25 < (r.get("metadata", {}).get("avg_rating") or 0) < q75]

        result["by_quality_tier"] = {
            "top_25": _subgroup_summary(top, n_bootstrap) if top else None,
            "middle_50": _subgroup_summary(mid, n_bootstrap) if mid else None,
            "bottom_25": _subgroup_summary(bottom, n_bootstrap) if bottom else None,
        }

    return result


def _is_accept(record: dict) -> bool:
    decision = (record.get("metadata", {}).get("decision") or "").lower()
    return "accept" in decision


def _subgroup_summary(records: list[dict], n_bootstrap: int) -> dict[str, Any]:
    """Compact per-metric summary for a subgroup."""
    result = {"n_papers": len(records)}
    for metric_key in METRIC_KEYS:
        llm_arr, human_arr = _extract_paired_metric(records, metric_key)
        if len(llm_arr) < 5:
            result[metric_key] = {"skipped": True}
            continue

        llm_mean = float(np.mean(llm_arr))
        human_mean = float(np.mean(human_arr))

        if len(llm_arr) > 500:
            delta, magnitude = cliffs_delta_fast(llm_arr, human_arr)
        else:
            delta, magnitude = cliffs_delta(llm_arr, human_arr)

        result[metric_key] = {
            "llm_mean": round(llm_mean, 4),
            "human_mean": round(human_mean, 4),
            "diff": round(llm_mean - human_mean, 4),
            "cliffs_delta": delta,
            "magnitude": magnitude,
        }
    return result


def _build_winner_summary(per_metric: dict) -> dict[str, Any]:
    """Build a human-readable summary of which side wins on each dimension."""
    llm_wins = []
    human_wins = []
    ties = []

    for key, data in per_metric.items():
        if data.get("skipped"):
            continue
        sig = data.get("significant_after_correction", False)
        diff = data.get("mean_diff", 0.0)

        if not sig:
            ties.append(key)
        elif diff > 0:
            llm_wins.append(key)
        else:
            human_wins.append(key)

    return {
        "llm_significantly_better": llm_wins,
        "human_significantly_better": human_wins,
        "no_significant_difference": ties,
    }
