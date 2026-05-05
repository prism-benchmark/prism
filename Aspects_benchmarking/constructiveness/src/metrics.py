"""
Constructiveness metrics computation.

Aggregates per-ARC dimension scores into review-level and paper-level metrics.
All computations are pure math — no LLM calls.
"""

from __future__ import annotations

from typing import Any

DIMENSION_KEYS = (
    "D1_actionability",
    "D2_specificity",
    "D3_justification",
    "D4_solution",
    "D5_tone",
)
MAX_SCORE_PER_DIM = 2
MAX_TOTAL = MAX_SCORE_PER_DIM * len(DIMENSION_KEYS)  # 10

CLC_CONSTRUCTIVE_THRESHOLD = 0.6


def _safe_numeric(value: Any, default: float = 0.0) -> float:
    """Convert possibly-missing metric values to a numeric default."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def compute_clc(arc: dict) -> float:
    """Comment-Level Constructiveness: CLC = sum(D1..D5) / 10."""
    total = sum(_safe_numeric(arc.get(k, 0)) for k in DIMENSION_KEYS)
    return total / MAX_TOTAL


def compute_review_metrics(arcs: list[dict]) -> dict[str, Any]:
    """Compute all review-level constructiveness metrics from a list of scored ARCs.

    Returns a dict with:
      MCS, AR, SD, CD, D1_mean .. D5_mean,
      n_arcs, n_weaknesses, n_strengths, n_questions, n_suggestions,
      per_type breakdown
    """
    n = len(arcs)
    if n == 0:
        return _empty_metrics()

    clc_values = [compute_clc(arc) for arc in arcs]

    # Mean Constructiveness Score
    mcs = sum(clc_values) / n

    # Actionability Ratio: fraction of ARCs with D1 >= 1
    ar = sum(1 for a in arcs if _safe_numeric(a.get("D1_actionability", 0)) >= 1) / n

    # Solution Density: fraction with D4 == 2
    sd = sum(1 for a in arcs if _safe_numeric(a.get("D4_solution", 0)) == 2) / n

    # Constructiveness Density: fraction with CLC >= threshold
    cd = sum(1 for c in clc_values if c >= CLC_CONSTRUCTIVE_THRESHOLD) / n

    # Per-dimension means
    dim_means = {}
    for k in DIMENSION_KEYS:
        dim_means[f"{k}_mean"] = sum(_safe_numeric(a.get(k, 0)) for a in arcs) / n

    # Comment-type distribution
    _PLURAL = {
        "weakness": "n_weaknesses",
        "strength": "n_strengths",
        "question": "n_questions",
        "suggestion": "n_suggestions",
        "observation": "n_observations",
    }
    type_counts = {}
    for t, key in _PLURAL.items():
        type_counts[key] = sum(1 for a in arcs if a.get("comment_type") == t)

    return {
        "MCS": round(mcs, 4),
        "AR": round(ar, 4),
        "SD": round(sd, 4),
        "CD": round(cd, 4),
        **{k: round(v, 4) for k, v in dim_means.items()},
        "n_arcs": n,
        **type_counts,
    }


def compute_paper_comparison(
    human_metrics_list: list[dict],
    llm_metrics: dict | None,
) -> dict[str, Any]:
    """Compute paper-level comparison between averaged Human metrics and LLM metrics.

    Only reviewers with n_arcs > 0 are included. If either side has no valid
    reviewers, comparison is marked as invalid.
    """
    comparison_keys = [
        "MCS", "AR", "SD", "CD",
        "D1_actionability_mean", "D2_specificity_mean",
        "D3_justification_mean", "D4_solution_mean", "D5_tone_mean",
    ]

    valid_humans = [h for h in human_metrics_list if (h.get("n_arcs") or 0) > 0]
    llm_valid = llm_metrics is not None and (llm_metrics.get("n_arcs") or 0) > 0

    result: dict[str, Any] = {
        "n_human_reviewers_total": len(human_metrics_list),
        "n_human_reviewers_valid": len(valid_humans),
        "llm_valid": llm_valid,
        "comparison_valid": len(valid_humans) > 0 and llm_valid,
    }

    n_valid = len(valid_humans)
    for key in comparison_keys:
        if n_valid > 0:
            human_vals = [_safe_numeric(h.get(key)) for h in valid_humans]
            human_avg = sum(human_vals) / n_valid
        else:
            human_avg = None

        llm_val = _safe_numeric(llm_metrics.get(key)) if llm_valid else None

        result[f"human_avg_{key}"] = round(human_avg, 4) if human_avg is not None else None
        result[f"llm_{key}"] = round(llm_val, 4) if llm_val is not None else None

        if human_avg is not None and llm_val is not None:
            result[f"delta_{key}"] = round(llm_val - human_avg, 4)
        else:
            result[f"delta_{key}"] = None

    result["n_arcs_llm"] = llm_metrics.get("n_arcs", 0) if llm_metrics else 0
    result["n_arcs_human_avg"] = (
        round(sum(h.get("n_arcs", 0) for h in valid_humans) / n_valid, 1)
        if n_valid > 0
        else 0
    )

    return result


def _empty_metrics() -> dict[str, Any]:
    result = {"MCS": None, "AR": None, "SD": None, "CD": None}
    for k in DIMENSION_KEYS:
        result[f"{k}_mean"] = None
    result["n_arcs"] = 0
    for key in ("n_weaknesses", "n_strengths", "n_questions", "n_suggestions", "n_observations"):
        result[key] = 0
    return result
