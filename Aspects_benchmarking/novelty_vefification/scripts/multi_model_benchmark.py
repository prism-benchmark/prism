#!/usr/bin/env python3
"""
Multi-model agreement benchmark for the novelty assessment pipeline.

Compares 2–N runs of the pipeline (e.g. different LLM backends) across all
3 phases using inter-model agreement metrics.  No ground-truth annotations
are required; models are treated as "raters" and agreement is measured.

Phase 1 (Extraction):   Claim count stats, pairwise ROUGE-L claim overlap,
                        stance distribution Jensen-Shannon Divergence,
                        core task ROUGE-L similarity.

Phase 2 (Retrieval):    Jaccard@K pool overlap (K=5,10,20,30),
                        Spearman rank correlation on shared candidates.

Phase 3 (Verification): Cohen's Kappa (pairwise), Fleiss' Kappa (multi-rater),
                        Krippendorff's Alpha (ordinal), Pearson/Spearman
                        correlation, pairwise confusion matrices, MAE,
                        label distributions, policy-aware best-evidence
                        overlap, and coverage stats. Claim alignment:
                        shared IDs first, ROUGE-L fallback.

Usage:
  python scripts/multi_model_benchmark.py \\
    --dirs output/iclr2024 output/iclr2024_gemini \\
           output/iclr2024_granite output/iclr2024_SEA \\
    --names default gemini granite sea \\
    --phases 1 2 3 \\
    --output benchmark_comparison \\
    --format both

  # Quick smoke test on 100 papers
  python scripts/multi_model_benchmark.py \\
    --dirs output/iclr2024 output/iclr2024_gemini \\
    --max-papers 100 --format json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

try:
    from scipy.stats import pearsonr, spearmanr  # type: ignore
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_SCORE_MAP: Dict[str, int] = {
    "SUPPORTED": 2,
    "OVERSTATED": 1,
    "AMBIGUOUS": 0,
    "UNDERSTATED": -1,
    "UNSUPPORTED": -2,
}
SCORE_LABEL_MAP: Dict[int, str] = {v: k for k, v in LABEL_SCORE_MAP.items()}
ALL_LABELS: List[str] = list(LABEL_SCORE_MAP.keys())

STANCE_VALUES: List[str] = ["not_novel", "somewhat_novel", "novel", "unclear"]
RETRIEVAL_K_VALUES: List[int] = [5, 10, 20, 30]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def safe_div(num: float, den: float) -> float:
    return num / den if den != 0 else 0.0


# ---------------------------------------------------------------------------
# Correlation helpers (scipy optional — manual fallbacks always available)
# ---------------------------------------------------------------------------

def _rank_data(x: List[float]) -> List[float]:
    """Return average ranks (1-indexed, handles ties) for a list of values."""
    n = len(x)
    if n == 0:
        return []
    indexed = sorted(range(n), key=lambda i: x[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and x[indexed[j + 1]] == x[indexed[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-indexed average rank for tie group
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg
        i = j + 1
    return ranks


def _pearson_corr(x: List[float], y: List[float]) -> Optional[float]:
    """Pearson correlation; returns None if undefined (constant input, n<2)."""
    n = len(x)
    if n < 2:
        return None
    if HAS_SCIPY:
        try:
            result = pearsonr(x, y)
            r = float(result[0])
            return None if math.isnan(r) else r
        except Exception:
            pass
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    denom_x = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    denom_y = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if denom_x == 0 or denom_y == 0:
        return None
    r = num / (denom_x * denom_y)
    return max(-1.0, min(1.0, r))


def _spearman_corr(x: List[float], y: List[float]) -> Optional[float]:
    """Spearman rank correlation; returns None if undefined (n<3)."""
    if len(x) < 3:
        return None
    if HAS_SCIPY:
        try:
            result = spearmanr(x, y)
            r = float(result[0])
            return None if math.isnan(r) else r
        except Exception:
            pass
    return _pearson_corr(_rank_data(x), _rank_data(y))


# ---------------------------------------------------------------------------
# ROUGE-L (adapted from benchmark_evaluation.py)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _lcs_length(x: Sequence[str], y: Sequence[str]) -> int:
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def rouge_l_f1(hyp: str, ref: str) -> float:
    h = _tokenize(hyp)
    r = _tokenize(ref)
    if not h or not r:
        return 0.0
    lcs = _lcs_length(h, r)
    p = safe_div(lcs, len(h))
    rec = safe_div(lcs, len(r))
    return safe_div(2.0 * p * rec, p + rec)


def _best_match_recall(texts_a: List[str], texts_b: List[str]) -> float:
    """For each text in A, find max ROUGE-L with any text in B. Return mean."""
    if not texts_a or not texts_b:
        return 0.0
    total = 0.0
    for ta in texts_a:
        total += max(rouge_l_f1(ta, tb) for tb in texts_b)
    return total / len(texts_a)


def symmetric_rouge_overlap(texts_a: List[str], texts_b: List[str]) -> float:
    """Symmetric ROUGE-L overlap: average of A→B recall and B→A recall."""
    if not texts_a and not texts_b:
        return 1.0  # both empty → full agreement
    if not texts_a or not texts_b:
        return 0.0
    return (_best_match_recall(texts_a, texts_b) + _best_match_recall(texts_b, texts_a)) / 2.0


# ---------------------------------------------------------------------------
# Claim alignment (Phase 3): shared IDs first, ROUGE-L fallback
# ---------------------------------------------------------------------------

def align_claims(
    claims_a: Dict[str, Dict[str, Any]],
    claims_b: Dict[str, Dict[str, Any]],
    rouge_threshold: float = 0.35,
) -> List[Tuple[str, str]]:
    """
    Return list of (id_a, id_b) matched pairs.
    Strategy:
      1. Exact ID match (e.g. both have "C1").
      2. For unmatched claims, greedy ROUGE-L text matching above threshold.
    """
    matched: List[Tuple[str, str]] = []
    used_b: Set[str] = set()

    # Step 1: exact ID intersection
    for cid in sorted(claims_a):
        if cid in claims_b:
            matched.append((cid, cid))
            used_b.add(cid)

    matched_a_ids: Set[str] = {m[0] for m in matched}

    # Step 2: ROUGE-L fallback for remaining
    remaining_a = [(cid, claims_a[cid]["text"]) for cid in claims_a if cid not in matched_a_ids]
    remaining_b = [(cid, claims_b[cid]["text"]) for cid in claims_b if cid not in used_b]

    for cid_a, text_a in remaining_a:
        if not remaining_b:
            break
        scores = [(rouge_l_f1(text_a, text_b), cid_b) for cid_b, text_b in remaining_b]
        best_score, best_cid_b = max(scores, key=lambda x: x[0])
        if best_score >= rouge_threshold:
            matched.append((cid_a, best_cid_b))
            remaining_b = [(c, t) for c, t in remaining_b if c != best_cid_b]

    return matched


# ---------------------------------------------------------------------------
# Jensen-Shannon Divergence
# ---------------------------------------------------------------------------

def _kl_div(p: float, q: float) -> float:
    if p <= 0.0:
        return 0.0
    if q <= 0.0:
        return float("inf")
    return p * math.log2(p / q)


def jensen_shannon_divergence(p: List[float], q: List[float]) -> float:
    """JSD between two discrete distributions (base-2, range [0,1])."""
    assert len(p) == len(q)
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
    jsd = 0.5 * sum(_kl_div(pi, mi) + _kl_div(qi, mi) for pi, qi, mi in zip(p, q, m))
    return max(0.0, min(1.0, jsd))


# ---------------------------------------------------------------------------
# Inter-rater agreement metrics
# ---------------------------------------------------------------------------

def cohen_kappa(labels_a: List[str], labels_b: List[str]) -> float:
    """Cohen's Kappa between two aligned label lists."""
    n = len(labels_a)
    if n == 0:
        return 0.0
    po = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    ca = Counter(labels_a)
    cb = Counter(labels_b)
    all_cats = set(ca) | set(cb)
    pe = sum((ca.get(k, 0) / n) * (cb.get(k, 0) / n) for k in all_cats)
    return safe_div(po - pe, 1.0 - pe) if pe < 1.0 else 1.0


def weighted_cohen_kappa(
    labels_a: List[str], labels_b: List[str], weight: str = "quadratic"
) -> float:
    """Weighted Cohen's Kappa (linear or quadratic) for ordinal labels."""
    n = len(labels_a)
    if n == 0:
        return 0.0
    # Ordinal ordering from ALL_LABELS
    label_idx = {l: i for i, l in enumerate(ALL_LABELS)}
    K = len(ALL_LABELS)
    if K <= 1:
        return 1.0

    # Build weight matrix
    w = [[0.0] * K for _ in range(K)]
    for i in range(K):
        for j in range(K):
            if weight == "linear":
                w[i][j] = 1.0 - abs(i - j) / (K - 1)
            else:  # quadratic
                w[i][j] = 1.0 - (i - j) ** 2 / (K - 1) ** 2

    # Build observed confusion matrix
    O = [[0] * K for _ in range(K)]
    for a, b in zip(labels_a, labels_b):
        ia = label_idx.get(a)
        ib = label_idx.get(b)
        if ia is not None and ib is not None:
            O[ia][ib] += 1

    # Marginals
    row_sums = [sum(O[i]) for i in range(K)]
    col_sums = [sum(O[i][j] for i in range(K)) for j in range(K)]

    # Expected matrix
    E = [[row_sums[i] * col_sums[j] / n for j in range(K)] for i in range(K)]

    # Weighted sums
    num = sum(w[i][j] * O[i][j] for i in range(K) for j in range(K))
    den = sum(w[i][j] * E[i][j] for i in range(K) for j in range(K))

    if den == 0:
        return 1.0
    # κ_w = 1 - (Σ w_ij * O_ij) / (Σ w_ij * E_ij)
    # Note: the disagreement formulation uses d_ij = 1 - w_ij
    # κ_w = 1 - Σ(1-w)*O / Σ(1-w)*E
    d_num = sum((1.0 - w[i][j]) * O[i][j] for i in range(K) for j in range(K))
    d_den = sum((1.0 - w[i][j]) * E[i][j] for i in range(K) for j in range(K))
    if d_den == 0:
        return 1.0
    return 1.0 - d_num / d_den


def macro_f1(labels_a: List[str], labels_b: List[str]) -> float:
    """Macro-averaged F1 score treating labels_a as gold and labels_b as predictions."""
    if not labels_a:
        return 0.0
    classes = sorted(set(labels_a) | set(labels_b))
    f1_scores: List[float] = []
    for c in classes:
        tp = sum(1 for a, b in zip(labels_a, labels_b) if a == c and b == c)
        fp = sum(1 for a, b in zip(labels_a, labels_b) if a != c and b == c)
        fn = sum(1 for a, b in zip(labels_a, labels_b) if a == c and b != c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


def bootstrap_ci(
    metric_fn,
    *args,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap confidence interval for a metric function."""
    rng = random.Random(seed)
    n = len(args[0])
    if n == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0}
    scores: List[float] = []
    for _ in range(n_bootstrap):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        resampled = tuple([a[i] for i in indices] for a in args)
        scores.append(metric_fn(*resampled))
    scores.sort()
    mean = sum(scores) / len(scores)
    std = math.sqrt(sum((s - mean) ** 2 for s in scores) / len(scores))
    alpha = 1.0 - ci
    lo = int(math.floor(alpha / 2.0 * len(scores)))
    hi = int(math.ceil((1.0 - alpha / 2.0) * len(scores))) - 1
    lo = max(0, min(lo, len(scores) - 1))
    hi = max(0, min(hi, len(scores) - 1))
    return {
        "mean": round(mean, 4),
        "ci_lower": round(scores[lo], 4),
        "ci_upper": round(scores[hi], 4),
        "std": round(std, 4),
    }


def fleiss_kappa(ratings: List[List[int]]) -> float:
    """
    Fleiss' Kappa for N raters and K categories.
    ratings: list of subjects × categories, each cell = count of raters
             who assigned that category to that subject.
    Assumes each subject has the same total number of raters.
    """
    N = len(ratings)
    if N == 0:
        return 0.0
    n_raters_per_row = [sum(row) for row in ratings]
    # Drop rows with fewer than 2 raters
    valid = [(row, nr) for row, nr in zip(ratings, n_raters_per_row) if nr >= 2]
    if not valid:
        return 0.0
    N_v = len(valid)
    # Use modal n (most common rater count) for robustness
    from collections import Counter as Ctr
    n = Ctr(nr for _, nr in valid).most_common(1)[0][0]
    valid_rows = [row for row, nr in valid if nr == n]
    N_v = len(valid_rows)
    if N_v == 0 or n <= 1:
        return 0.0

    n_cats = len(valid_rows[0])

    # P_i: extent of agreement for subject i
    P_i = [
        safe_div(sum(x * (x - 1) for x in row), n * (n - 1))
        for row in valid_rows
    ]
    P_bar = sum(P_i) / N_v

    # p_j: marginal proportion for category j
    total = N_v * n
    p_j = [sum(row[j] for row in valid_rows) / total for j in range(n_cats)]
    P_e = sum(p ** 2 for p in p_j)

    return safe_div(P_bar - P_e, 1.0 - P_e) if P_e < 1.0 else 1.0


def krippendorff_alpha_ordinal(rater_data: List[List[Optional[float]]]) -> float:
    """
    Krippendorff's Alpha using ordinal (squared difference) metric.
    rater_data: list of raters, each a list of values (None = missing).
    """
    n_raters = len(rater_data)
    if n_raters < 2:
        return 0.0
    n_units = len(rater_data[0]) if rater_data else 0
    if n_units == 0:
        return 0.0

    all_vals: List[float] = [
        v for rater in rater_data for v in rater if v is not None
    ]
    if len(all_vals) < 2:
        return 0.0

    # Observed disagreement: mean of within-unit pairwise squared diffs
    D_o_sum = 0.0
    D_o_count = 0
    for unit_idx in range(n_units):
        unit_vals = [
            rater_data[r][unit_idx]
            for r in range(n_raters)
            if rater_data[r][unit_idx] is not None
        ]
        if len(unit_vals) < 2:
            continue
        for i in range(len(unit_vals)):
            for j in range(i + 1, len(unit_vals)):
                D_o_sum += (unit_vals[i] - unit_vals[j]) ** 2
                D_o_count += 1

    if D_o_count == 0:
        return 1.0

    D_o = D_o_sum / D_o_count

    # Expected disagreement: mean of all pairwise squared diffs in global pool
    n_v = len(all_vals)
    D_e_sum = 0.0
    D_e_count = 0
    for i in range(n_v):
        for j in range(i + 1, n_v):
            D_e_sum += (all_vals[i] - all_vals[j]) ** 2
            D_e_count += 1

    if D_e_count == 0 or D_e_sum == 0:
        return 1.0 if D_o == 0 else 0.0

    D_e = D_e_sum / D_e_count
    return 1.0 - safe_div(D_o, D_e)


# ---------------------------------------------------------------------------
# Phase 1: Extraction
# ---------------------------------------------------------------------------

def _load_p1(paper_dir: Path) -> Optional[Dict[str, Any]]:
    data = load_json(paper_dir / "task1_result.json")
    if data is None:
        return None
    review = data.get("review") or {}
    paper = data.get("paper") or {}
    claims = [c for c in (review.get("novelty_claims") or []) if c.get("text", "").strip()]
    return {
        "claim_texts": [c["text"] for c in claims],
        "stances": [c.get("stance", "unclear") for c in claims],
        "core_task": (paper.get("core_task") or "").strip(),
        "n_claims": len(claims),
    }


def _p1_paper(model_data: Dict[str, Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    avail = {n: d for n, d in model_data.items() if d is not None}
    if len(avail) < 2:
        return None
    names = list(avail.keys())
    pairwise_overlap: Dict[str, float] = {}
    pairwise_core: Dict[str, float] = {}
    for na, nb in combinations(names, 2):
        pk = f"{na}_vs_{nb}"
        pairwise_overlap[pk] = round(
            symmetric_rouge_overlap(avail[na]["claim_texts"], avail[nb]["claim_texts"]), 4
        )
        pairwise_core[pk] = round(
            rouge_l_f1(avail[na]["core_task"], avail[nb]["core_task"]), 4
        )
    return {
        "n_claims": {n: avail[n]["n_claims"] for n in names},
        "pairwise_claim_overlap": pairwise_overlap,
        "pairwise_core_task": pairwise_core,
    }


def _agg_p1(
    per_paper: List[Dict[str, Any]],
    names: List[str],
    stance_counts: Dict[str, Dict[str, int]],
    n_reviews_map: Optional[Dict[str, int]] = None,
    human_model_name: Optional[str] = None,
    human_n_reviews_default: int = 3,
) -> Dict[str, Any]:
    if not per_paper:
        return {}

    # Claim count stats (human model: claims per review when n_reviews_map provided)
    counts: Dict[str, List[float]] = {n: [] for n in names}
    for p in per_paper:
        paper_id = p.get("paper_id", "")
        for n in names:
            v = p.get("n_claims", {}).get(n)
            if v is not None:
                if n == human_model_name and n_reviews_map is not None:
                    n_rev = max(1, n_reviews_map.get(paper_id, human_n_reviews_default))
                    counts[n].append(v / n_rev)
                else:
                    counts[n].append(float(v))

    claim_stats: Dict[str, Any] = {}
    normalization_applied = human_model_name and n_reviews_map is not None
    for n in names:
        vals = counts[n]
        if vals:
            mean = sum(vals) / len(vals)
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
            sorted_vals = sorted(vals)
            median = sorted_vals[len(vals) // 2]
            claim_stats[n] = {
                "mean": round(mean, 4),
                "median": round(float(median), 4),
                "std": round(std, 4),
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
                "n_papers": len(vals),
            }

    # Pairwise claim overlap
    all_pks: Set[str] = set()
    for p in per_paper:
        all_pks.update(p.get("pairwise_claim_overlap", {}).keys())

    overlap_agg: Dict[str, Any] = {}
    core_agg: Dict[str, Any] = {}
    for pk in sorted(all_pks):
        ov = [p["pairwise_claim_overlap"][pk] for p in per_paper if pk in p.get("pairwise_claim_overlap", {})]
        ct = [p["pairwise_core_task"][pk] for p in per_paper if pk in p.get("pairwise_core_task", {})]
        if ov:
            overlap_agg[pk] = {"mean": round(sum(ov) / len(ov), 4), "n_papers": len(ov)}
        if ct:
            core_agg[pk] = {"mean": round(sum(ct) / len(ct), 4), "n_papers": len(ct)}

    # Stance distributions and pairwise JSD
    total_stance: Dict[str, int] = {}
    distributions: Dict[str, Dict[str, float]] = {}
    for n in names:
        sc = stance_counts.get(n, {})
        total = sum(sc.values())
        distributions[n] = {s: round(sc.get(s, 0) / total, 4) if total > 0 else 0.0 for s in STANCE_VALUES}
        total_stance[n] = total

    pairwise_jsd: Dict[str, float] = {}
    for na, nb in combinations(names, 2):
        pk = f"{na}_vs_{nb}"
        p_vec = [distributions[na].get(s, 0.0) for s in STANCE_VALUES]
        q_vec = [distributions[nb].get(s, 0.0) for s in STANCE_VALUES]
        pairwise_jsd[pk] = round(jensen_shannon_divergence(p_vec, q_vec), 4)

    result: Dict[str, Any] = {
        "n_papers": len(per_paper),
        "claim_count_stats": claim_stats,
        "pairwise_claim_overlap": overlap_agg,
        "pairwise_core_task_similarity": core_agg,
        "stance_distributions": {
            "per_model": distributions,
            "total_claims_per_model": total_stance,
            "pairwise_jsd": pairwise_jsd,
        },
    }
    if normalization_applied:
        result["claim_count_stats_note"] = (
            "Human (default) normalized by n_reviews per paper; LLMs use 1 review per paper"
        )
    return result


# ---------------------------------------------------------------------------
# Phase 2: Retrieval
# ---------------------------------------------------------------------------

def _load_p2(paper_dir: Path) -> Optional[Dict[str, Any]]:
    data = load_json(paper_dir / "task2_result.json")
    if data is None:
        return None
    pool = data.get("candidate_pool_top30") or []
    cand_ids = [c.get("cand_id", "") for c in pool if c.get("cand_id", "")]
    return {"cand_ids": cand_ids, "n_candidates": len(cand_ids)}


def _jaccard_at_k(ids_a: List[str], ids_b: List[str], k: int) -> float:
    sa = set(ids_a[:k])
    sb = set(ids_b[:k])
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _spearman_shared(ids_a: List[str], ids_b: List[str]) -> Optional[float]:
    """Spearman rank correlation on candidates appearing in both lists."""
    set_b = set(ids_b)
    shared = [c for c in ids_a if c in set_b]
    if len(shared) < 3:
        return None
    rank_a = {c: float(i) for i, c in enumerate(ids_a)}
    rank_b = {c: float(i) for i, c in enumerate(ids_b)}
    # Use actual positions in each full list so re-ranking is meaningful
    ra = [rank_a[c] for c in shared]
    rb = [rank_b[c] for c in shared]
    return _spearman_corr(ra, rb)


def _p2_paper(model_data: Dict[str, Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    avail = {n: d for n, d in model_data.items() if d is not None}
    if len(avail) < 2:
        return None
    names = list(avail.keys())
    pairwise_jac: Dict[str, Dict[str, float]] = {}
    pairwise_spr: Dict[str, Optional[float]] = {}
    for na, nb in combinations(names, 2):
        pk = f"{na}_vs_{nb}"
        ia = avail[na]["cand_ids"]
        ib = avail[nb]["cand_ids"]
        pairwise_jac[pk] = {str(k): round(_jaccard_at_k(ia, ib, k), 4) for k in RETRIEVAL_K_VALUES}
        rho = _spearman_shared(ia, ib)
        pairwise_spr[pk] = round(rho, 4) if rho is not None else None
    return {
        "n_candidates": {n: avail[n]["n_candidates"] for n in names},
        "pairwise_jaccard": pairwise_jac,
        "pairwise_spearman": pairwise_spr,
    }


def _agg_p2(per_paper: List[Dict[str, Any]], names: List[str]) -> Dict[str, Any]:
    if not per_paper:
        return {}

    all_pks: Set[str] = set()
    for p in per_paper:
        all_pks.update(p.get("pairwise_jaccard", {}).keys())

    jac_agg: Dict[str, Any] = {}
    for pk in sorted(all_pks):
        by_k: Dict[str, List[float]] = {str(k): [] for k in RETRIEVAL_K_VALUES}
        for p in per_paper:
            jac = p.get("pairwise_jaccard", {}).get(pk)
            if jac:
                for k in RETRIEVAL_K_VALUES:
                    by_k[str(k)].append(jac[str(k)])
        entry: Dict[str, Any] = {"n_papers": len([p for p in per_paper if pk in p.get("pairwise_jaccard", {})])}
        for k in RETRIEVAL_K_VALUES:
            vals = by_k[str(k)]
            if vals:
                entry[f"mean_jaccard@{k}"] = round(sum(vals) / len(vals), 4)
        jac_agg[pk] = entry

    spr_agg: Dict[str, Any] = {}
    for pk in sorted(all_pks):
        vals = [
            p["pairwise_spearman"][pk]
            for p in per_paper
            if pk in p.get("pairwise_spearman", {}) and p["pairwise_spearman"][pk] is not None
        ]
        if vals:
            spr_agg[pk] = {"mean": round(sum(vals) / len(vals), 4), "n_papers": len(vals)}

    cand_stats: Dict[str, Any] = {}
    for n in names:
        vals = [p["n_candidates"][n] for p in per_paper if n in p.get("n_candidates", {})]
        if vals:
            cand_stats[n] = {"mean": round(sum(vals) / len(vals), 4), "n_papers": len(vals)}

    return {
        "n_papers": len(per_paper),
        "candidate_count_stats": cand_stats,
        "pairwise_jaccard": jac_agg,
        "pairwise_spearman": spr_agg,
    }


# ---------------------------------------------------------------------------
# Phase 3: Verification
# ---------------------------------------------------------------------------

def _set_jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _classify(score: float) -> str:
    r = max(-2, min(2, round(score)))
    return SCORE_LABEL_MAP.get(r, "AMBIGUOUS")


def _load_p3(paper_dir: Path) -> Optional[Dict[str, Any]]:
    data = load_json(paper_dir / "task3_result.json")
    if data is None:
        return None

    aggregated = data.get("aggregated") or []
    claims: Dict[str, Dict[str, Any]] = {}
    for item in aggregated:
        cid = item.get("review_sentence_id", "")
        fs = item.get("final_score")
        if fs is None or not cid:
            continue

        best_evidence_raw = item.get("best_evidence") or []
        best_evidence = {
            str(x) for x in best_evidence_raw
            if isinstance(x, (str, int, float)) and str(x)
        }

        claims[cid] = {
            "score": float(fs),
            "label": _classify(float(fs)),
            "text": item.get("text", ""),
            "best_evidence": best_evidence,
            "best_evidence_policy": item.get("best_evidence_policy"),
        }

    if not claims:
        return None

    scores = [c["score"] for c in claims.values()]
    paper_score = sum(scores) / len(scores)

    coverage_raw = (data.get("stats") or {}).get("coverage")
    coverage = coverage_raw if isinstance(coverage_raw, dict) else {}

    return {
        "claims": claims,
        "paper_score": paper_score,
        "paper_label": _classify(paper_score),
        "n_claims": len(claims),
        "coverage": coverage,
    }


def _p3_paper(model_data: Dict[str, Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    avail = {n: d for n, d in model_data.items() if d is not None}
    if len(avail) < 2:
        return None
    names = list(avail.keys())

    paper_scores = {n: avail[n]["paper_score"] for n in names}
    paper_labels = {n: avail[n]["paper_label"] for n in names}
    n_claims = {n: avail[n]["n_claims"] for n in names}
    coverage = {n: avail[n].get("coverage", {}) for n in names}

    # Align claims: shared ID first, then ROUGE-L fallback
    # Store aligned tuples per pair:
    # (score_a, score_b, label_a, label_b, best_evidence_jaccard)
    aligned_pairs: Dict[str, List[Tuple[float, float, str, str, float]]] = {}
    for na, nb in combinations(names, 2):
        pk = f"{na}_vs_{nb}"
        ca = avail[na]["claims"]
        cb = avail[nb]["claims"]
        matched = align_claims(ca, cb)
        pairs = []
        for id_a, id_b in matched:
            be_a = ca[id_a].get("best_evidence", set())
            be_b = cb[id_b].get("best_evidence", set())
            if not isinstance(be_a, set):
                be_a = set()
            if not isinstance(be_b, set):
                be_b = set()
            be_j = _set_jaccard(be_a, be_b)

            pairs.append((
                ca[id_a]["score"], cb[id_b]["score"],
                ca[id_a]["label"], cb[id_b]["label"],
                float(be_j),
            ))
        aligned_pairs[pk] = pairs

    return {
        "paper_scores": paper_scores,
        "paper_labels": paper_labels,
        "n_claims": n_claims,
        "coverage": coverage,
        "aligned_pairs": aligned_pairs,
    }


def _agg_p3(per_paper: List[Dict[str, Any]], names: List[str]) -> Dict[str, Any]:
    if not per_paper:
        return {}

    # Paper-level score/label lists per model
    ps_lists: Dict[str, List[float]] = {n: [] for n in names}
    pl_lists: Dict[str, List[str]] = {n: [] for n in names}
    for p in per_paper:
        for n in names:
            if n in p.get("paper_scores", {}):
                ps_lists[n].append(p["paper_scores"][n])
                pl_lists[n].append(p["paper_labels"][n])

    # Score stats per model
    score_stats: Dict[str, Any] = {}
    for n in names:
        vals = ps_lists[n]
        if vals:
            mean = sum(vals) / len(vals)
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
            score_stats[n] = {
                "mean": round(mean, 4),
                "std": round(std, 4),
                "n_papers": len(vals),
            }

    # Label distributions per model (paper-level)
    label_dist: Dict[str, Dict[str, int]] = {
        n: dict(Counter(pl_lists[n])) for n in names
    }

    # Coverage stats per model (from task3_result.stats.coverage)
    coverage_stats: Dict[str, Any] = {}
    for n in names:
        cov_rows = [
            p.get("coverage", {}).get(n)
            for p in per_paper
            if isinstance(p.get("coverage", {}).get(n), dict)
        ]
        if not cov_rows:
            continue

        numeric_keys: Set[str] = set()
        for row in cov_rows:
            for k, v in row.items():
                if isinstance(v, (int, float)):
                    numeric_keys.add(k)

        per_key: Dict[str, Any] = {}
        for k in sorted(numeric_keys):
            vals = [float(r[k]) for r in cov_rows if isinstance(r.get(k), (int, float))]
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
            per_key[k] = {
                "mean": round(mean, 4),
                "std": round(std, 4),
                "n_papers": len(vals),
            }

        if per_key:
            coverage_stats[n] = per_key

    # Pairwise paper-level metrics
    all_pks: Set[str] = set()
    for p in per_paper:
        all_pks.update(p.get("aligned_pairs", {}).keys())

    pairwise_paper: Dict[str, Any] = {}
    # For paper-level pairwise: use ps_lists and pl_lists (already aligned by paper)
    for na, nb in combinations(names, 2):
        pk = f"{na}_vs_{nb}"
        paired: List[Tuple[float, float]] = []
        labeled: List[Tuple[str, str]] = []
        for p in per_paper:
            ps = p.get("paper_scores", {})
            pl = p.get("paper_labels", {})
            if na in ps and nb in ps:
                paired.append((ps[na], ps[nb]))
                labeled.append((pl.get(na, "AMBIGUOUS"), pl.get(nb, "AMBIGUOUS")))

        if not paired:
            continue

        sa = [x[0] for x in paired]
        sb = [x[1] for x in paired]
        la = [x[0] for x in labeled]
        lb = [x[1] for x in labeled]

        mae = sum(abs(a - b) for a, b in paired) / len(paired)
        acc = sum(a == b for a, b in zip(la, lb)) / len(la)
        kappa = cohen_kappa(la, lb)

        r = _pearson_corr(sa, sb)
        pearson_r = round(r, 4) if r is not None else None
        rho = _spearman_corr(sa, sb)
        spearman_rho = round(rho, 4) if rho is not None else None

        # Confusion matrix (paper-level labels)
        cm: Dict[str, Dict[str, int]] = {la_: {lb_: 0 for lb_ in ALL_LABELS} for la_ in ALL_LABELS}
        for a, b in zip(la, lb):
            if a in cm and b in cm[a]:
                cm[a][b] += 1

        pairwise_paper[pk] = {
            "n_papers": len(paired),
            "mae": round(mae, 4),
            "accuracy": round(acc, 4),
            "cohen_kappa": round(kappa, 4),
            "pearson_r": pearson_r,
            "spearman_rho": spearman_rho,
            "confusion_matrix": cm,
        }

    # Pairwise claim-level metrics (aligned pairs)
    pairwise_claims: Dict[str, Any] = {}
    all_aligned: Dict[str, List[Tuple[float, float, str, str, float]]] = defaultdict(list)
    for p in per_paper:
        for pk, pairs in p.get("aligned_pairs", {}).items():
            all_aligned[pk].extend(pairs)

    for pk, pairs in sorted(all_aligned.items()):
        if not pairs:
            continue
        sa = [x[0] for x in pairs]
        sb = [x[1] for x in pairs]
        la = [x[2] for x in pairs]
        lb = [x[3] for x in pairs]
        be_jaccards = [x[4] for x in pairs if len(x) > 4 and isinstance(x[4], (int, float))]

        mae = sum(abs(a - b) for a, b in zip(sa, sb)) / len(sa)
        acc = sum(a == b for a, b in zip(la, lb)) / len(la)
        kappa = cohen_kappa(la, lb)

        r = _pearson_corr(sa, sb)
        pearson_r = round(r, 4) if r is not None else None
        rho = _spearman_corr(sa, sb)
        spearman_rho = round(rho, 4) if rho is not None else None

        w_kappa = weighted_cohen_kappa(la, lb, weight="quadratic")
        mf1 = macro_f1(la, lb)

        # Claim-level confusion matrix
        cm_claims: Dict[str, Dict[str, int]] = {
            la_: {lb_: 0 for lb_ in ALL_LABELS} for la_ in ALL_LABELS
        }
        for a, b in zip(la, lb):
            if a in cm_claims and b in cm_claims[a]:
                cm_claims[a][b] += 1

        # Bootstrap CIs for weighted kappa and macro F1
        ci_wk = bootstrap_ci(weighted_cohen_kappa, la, lb, n_bootstrap=1000, seed=42)
        ci_mf1 = bootstrap_ci(macro_f1, la, lb, n_bootstrap=1000, seed=42)

        pairwise_claims[pk] = {
            "n_aligned_claims": len(pairs),
            "mae": round(mae, 4),
            "accuracy": round(acc, 4),
            "cohen_kappa": round(kappa, 4),
            "weighted_kappa_quadratic": round(w_kappa, 4),
            "macro_f1": round(mf1, 4),
            "pearson_r": pearson_r,
            "spearman_rho": spearman_rho,
            "best_evidence_jaccard": {
                "mean": round(sum(be_jaccards) / len(be_jaccards), 4) if be_jaccards else None,
                "n_aligned_claims_with_best_evidence": len(be_jaccards),
            },
            "confusion_matrix": cm_claims,
            "bootstrap_ci_weighted_kappa": {
                "mean": ci_wk["mean"],
                "ci_lower": ci_wk["ci_lower"],
                "ci_upper": ci_wk["ci_upper"],
            },
            "bootstrap_ci_macro_f1": {
                "mean": ci_mf1["mean"],
                "ci_lower": ci_mf1["ci_lower"],
                "ci_upper": ci_mf1["ci_upper"],
            },
        }

    # Fleiss' Kappa: multi-rater, claim-level (papers with ALL models)
    fleiss_result: Optional[float] = None
    if len(names) >= 2:
        label_to_idx = {l: i for i, l in enumerate(ALL_LABELS)}
        ratings_matrix: List[List[int]] = []
        for p in per_paper:
            # Use aligned pairs where ALL n_models appear: approximate via paper labels
            pl = p.get("paper_labels", {})
            if len(pl) == len(names):
                row = [0] * len(ALL_LABELS)
                for n in names:
                    lbl = pl.get(n, "AMBIGUOUS")
                    if lbl in label_to_idx:
                        row[label_to_idx[lbl]] += 1
                ratings_matrix.append(row)
        if ratings_matrix:
            fleiss_result = round(fleiss_kappa(ratings_matrix), 4)

    # Krippendorff's Alpha: ordinal, paper-level scores
    kripp_result: Optional[float] = None
    if len(names) >= 2:
        n_units = len(per_paper)
        rater_data: List[List[Optional[float]]] = []
        for n in names:
            rater_row: List[Optional[float]] = []
            for p in per_paper:
                rater_row.append(p.get("paper_scores", {}).get(n))
            rater_data.append(rater_row)
        kripp_result = round(krippendorff_alpha_ordinal(rater_data), 4)

    result: Dict[str, Any] = {
        "n_papers": len(per_paper),
        "paper_score_stats": score_stats,
        "label_distributions": label_dist,
        "coverage_stats": coverage_stats,
        "pairwise_paper_level": pairwise_paper,
        "pairwise_claim_level": pairwise_claims,
    }
    if fleiss_result is not None:
        result["fleiss_kappa_paper_level"] = fleiss_result
    if kripp_result is not None:
        result["krippendorff_alpha_paper_level"] = kripp_result

    return result


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _build_n_reviews_from_json_dir(review_json_dir: Path) -> Dict[str, int]:
    """
    Build {paper_id: n_reviews} from directory of review JSON files.
    Each file is <paper_id>.json with a "reviews" array.
    """
    n_reviews_map: Dict[str, int] = {}
    if not review_json_dir or not review_json_dir.is_dir():
        return n_reviews_map
    for p in review_json_dir.glob("*.json"):
        pid = p.stem
        data = load_json(p)
        if data is None:
            continue
        reviews = data.get("reviews")
        if isinstance(reviews, list):
            n_reviews_map[pid] = len(reviews)
    return n_reviews_map


def _discover_common_papers(dirs: List[Path], required_files: List[str]) -> List[str]:
    """Papers that have all required task files in every directory."""
    common: Optional[Set[str]] = None
    for d in dirs:
        ids: Set[str] = set()
        if not d.is_dir():
            continue
        for sub in d.iterdir():
            if sub.is_dir() and all((sub / f).exists() for f in required_files):
                ids.add(sub.name)
        common = ids if common is None else common & ids
    return sorted(common or set())


def run_benchmark(
    dirs: List[Path],
    names: List[str],
    phases: List[int],
    max_papers: Optional[int] = None,
    verbose: bool = False,
    n_reviews_map: Optional[Dict[str, int]] = None,
    human_model_name: Optional[str] = None,
    human_n_reviews_default: int = 3,
    task2_shared_across_models: bool = False,
) -> Dict[str, Any]:
    required_non_p2: List[str] = []
    if 1 in phases:
        required_non_p2.append("task1_result.json")
    if 3 in phases:
        required_non_p2.append("task3_result.json")

    # Paper discovery:
    # - Default behavior: every requested phase file must exist in every model dir.
    # - Shared Task2 mode: task2_result.json can be missing in some models and reused
    #   from any model for that same paper.
    if 2 in phases and not task2_shared_across_models:
        required_all = list(required_non_p2) + ["task2_result.json"]
    else:
        required_all = list(required_non_p2)

    if 1 not in phases and 2 in phases and 3 not in phases and not required_all:
        # Phase-2-only run. In shared mode we only require paper directories to exist.
        paper_ids = _discover_common_papers(dirs, [])
    else:
        paper_ids = _discover_common_papers(dirs, required_all)

    if 2 in phases and task2_shared_across_models:
        # Keep only papers where at least one model has task2_result.json.
        paper_ids = [
            pid for pid in paper_ids
            if any((d / pid / "task2_result.json").exists() for d in dirs)
        ]

    if max_papers:
        paper_ids = paper_ids[:max_papers]

    print(
        f"Common papers across all {len(dirs)} models: {len(paper_ids)}",
        file=sys.stderr,
    )
    if 2 in phases and task2_shared_across_models:
        print(
            "Task2 shared mode: missing task2_result.json will be reused from another model for the same paper.",
            file=sys.stderr,
        )

    # Accumulators
    p1_per: List[Dict[str, Any]] = []
    p1_stance: Dict[str, Dict[str, int]] = {
        n: {s: 0 for s in STANCE_VALUES} for n in names
    }

    p2_per: List[Dict[str, Any]] = []
    p3_per: List[Dict[str, Any]] = []

    report_interval = max(1, len(paper_ids) // 10)

    for idx, pid in enumerate(paper_ids):
        if verbose and idx % report_interval == 0:
            print(f"  Processing paper {idx + 1}/{len(paper_ids)} ...", file=sys.stderr)

        if 1 in phases:
            md1: Dict[str, Optional[Dict[str, Any]]] = {}
            for d, n in zip(dirs, names):
                md1[n] = _load_p1(d / pid)
                if md1[n] is not None:
                    for stance in md1[n]["stances"]:
                        s = stance if stance in p1_stance[n] else "unclear"
                        p1_stance[n][s] += 1
            r = _p1_paper(md1)
            if r is not None:
                r["paper_id"] = pid
                p1_per.append(r)

        if 2 in phases:
            md2: Dict[str, Optional[Dict[str, Any]]] = {
                n: _load_p2(d / pid) for d, n in zip(dirs, names)
            }
            if task2_shared_across_models:
                shared_p2 = next((v for v in md2.values() if v is not None), None)
                if shared_p2 is not None:
                    for n in names:
                        if md2.get(n) is None:
                            md2[n] = shared_p2
            r = _p2_paper(md2)
            if r is not None:
                r["paper_id"] = pid
                p2_per.append(r)

        if 3 in phases:
            md3: Dict[str, Optional[Dict[str, Any]]] = {
                n: _load_p3(d / pid) for d, n in zip(dirs, names)
            }
            r = _p3_paper(md3)
            if r is not None:
                r["paper_id"] = pid
                p3_per.append(r)

    report: Dict[str, Any] = {
        "models": names,
        "n_papers_common": len(paper_ids),
        "phases_evaluated": phases,
    }

    if 1 in phases:
        report["phase1_extraction"] = _agg_p1(
            p1_per,
            names,
            p1_stance,
            n_reviews_map=n_reviews_map,
            human_model_name=human_model_name,
            human_n_reviews_default=human_n_reviews_default,
        )

    if 2 in phases:
        report["phase2_retrieval"] = _agg_p2(p2_per, names)

    if 3 in phases:
        report["phase3_verification"] = _agg_p3(p3_per, names)

    return report


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_summary(report: Dict[str, Any]) -> None:
    sep = "=" * 68
    print(f"\n{sep}")
    print("  MULTI-MODEL BENCHMARK COMPARISON SUMMARY")
    print(sep)
    print(f"  Models  : {', '.join(report.get('models', []))}")
    print(f"  Papers  : {report.get('n_papers_common', 0)}")
    print(f"  Phases  : {report.get('phases_evaluated', [])}")

    # ---- Phase 1 ----
    p1 = report.get("phase1_extraction")
    if p1:
        print(f"\n{'─'*68}")
        print(f"  PHASE 1 – Extraction Agreement  (n={p1.get('n_papers', 0)} papers)")
        print(f"{'─'*68}")

        ccs = p1.get("claim_count_stats", {})
        if ccs:
            print("  Claim counts per model:")
            for n, s in sorted(ccs.items()):
                print(f"    {n:<22s}  mean={s['mean']:.2f}  std={s['std']:.2f}  "
                      f"median={s['median']}  range=[{s['min']},{s['max']}]")

        ov = p1.get("pairwise_claim_overlap", {})
        if ov:
            print("  Pairwise claim overlap (symmetric ROUGE-L):")
            for pk, s in sorted(ov.items()):
                print(f"    {pk:<45s}  {s['mean']:.4f}")

        ct = p1.get("pairwise_core_task_similarity", {})
        if ct:
            print("  Pairwise core-task similarity (ROUGE-L):")
            for pk, s in sorted(ct.items()):
                print(f"    {pk:<45s}  {s['mean']:.4f}")

        stance = p1.get("stance_distributions", {})
        jsd = stance.get("pairwise_jsd", {})
        if jsd:
            print("  Stance distribution JSD (lower = more similar):")
            for pk, val in sorted(jsd.items()):
                print(f"    {pk:<45s}  {val:.4f}")
        dist = stance.get("per_model", {})
        if dist:
            print("  Stance distributions per model:")
            for n, d in sorted(dist.items()):
                parts = "  ".join(f"{s}={d.get(s,0):.3f}" for s in STANCE_VALUES)
                print(f"    {n:<22s}  {parts}")

    # ---- Phase 2 ----
    p2 = report.get("phase2_retrieval")
    if p2:
        print(f"\n{'─'*68}")
        print(f"  PHASE 2 – Retrieval Agreement  (n={p2.get('n_papers', 0)} papers)")
        print(f"{'─'*68}")

        jac = p2.get("pairwise_jaccard", {})
        if jac:
            header_parts = "  ".join(f"@{k}" for k in RETRIEVAL_K_VALUES)
            print(f"  Jaccard@K pool overlap  [{header_parts}]:")
            for pk, s in sorted(jac.items()):
                vals = "  ".join(
                    f"{s.get(f'mean_jaccard@{k}', 0):.4f}" for k in RETRIEVAL_K_VALUES
                )
                print(f"    {pk:<45s}  {vals}")

        spr = p2.get("pairwise_spearman", {})
        if spr:
            print("  Spearman rank correlation (shared candidates):")
            for pk, s in sorted(spr.items()):
                print(f"    {pk:<45s}  {s['mean']:.4f}")

    # ---- Phase 3 ----
    p3 = report.get("phase3_verification")
    if p3:
        print(f"\n{'─'*68}")
        print(f"  PHASE 3 – Verification Agreement  (n={p3.get('n_papers', 0)} papers)")
        print(f"{'─'*68}")

        ss = p3.get("paper_score_stats", {})
        if ss:
            print("  Paper-level mean score per model:")
            for n, s in sorted(ss.items()):
                print(f"    {n:<22s}  mean={s['mean']:+.4f}  std={s['std']:.4f}")

        ld = p3.get("label_distributions", {})
        if ld:
            print("  Label distributions (paper-level):")
            for n, d in sorted(ld.items()):
                parts = "  ".join(f"{l}={d.get(l,0)}" for l in ALL_LABELS)
                print(f"    {n:<22s}  {parts}")

        cov = p3.get("coverage_stats", {})
        if cov:
            print("  Coverage stats per model (mean across papers):")
            for n, c in sorted(cov.items()):
                attempt = c.get("claim_attempt_coverage_rate", {}).get("mean")
                success = c.get("claim_success_coverage_rate", {}).get("mean")
                evidence = c.get("evidence_coverage_rate", {}).get("mean")
                decisive = c.get("decisive_coverage_rate", {}).get("mean")
                fail_rate = c.get("pair_failure_rate", {}).get("mean")
                print(
                    f"    {n:<22s}  "
                    f"attempt={attempt if attempt is not None else float('nan'):.4f}  "
                    f"success={success if success is not None else float('nan'):.4f}  "
                    f"evidence={evidence if evidence is not None else float('nan'):.4f}  "
                    f"decisive={decisive if decisive is not None else float('nan'):.4f}  "
                    f"pair_fail={fail_rate if fail_rate is not None else float('nan'):.4f}"
                )

        fk = p3.get("fleiss_kappa_paper_level")
        ka = p3.get("krippendorff_alpha_paper_level")
        if fk is not None or ka is not None:
            print("  Multi-rater agreement (paper-level):")
            if fk is not None:
                print(f"    Fleiss' Kappa              κ = {fk:+.4f}")
            if ka is not None:
                print(f"    Krippendorff's Alpha       α = {ka:+.4f}")

        pp = p3.get("pairwise_paper_level", {})
        if pp:
            print("  Pairwise paper-level metrics:")
            hdr = f"    {'Pair':<45s}  {'κ':>7}  {'r':>7}  {'ρ':>7}  {'MAE':>6}  {'Acc':>6}"
            print(hdr)
            for pk, s in sorted(pp.items()):
                kap = s.get("cohen_kappa", 0)
                pr = s.get("pearson_r")
                sp = s.get("spearman_rho")
                mae = s.get("mae", 0)
                acc = s.get("accuracy", 0)
                pr_s = f"{pr:+.4f}" if pr is not None else "   N/A"
                sp_s = f"{sp:+.4f}" if sp is not None else "   N/A"
                print(f"    {pk:<45s}  {kap:+.4f}  {pr_s}  {sp_s}  {mae:>6.4f}  {acc:>6.4f}")

        pc = p3.get("pairwise_claim_level", {})
        if pc:
            print("  Pairwise claim-level metrics (aligned claims):")
            for pk, s in sorted(pc.items()):
                n_aligned = s.get("n_aligned_claims", 0)
                kap = s.get("cohen_kappa", 0)
                wk = s.get("weighted_kappa_quadratic", 0)
                mf1 = s.get("macro_f1", 0)
                mae = s.get("mae", 0)
                acc = s.get("accuracy", 0)
                pr = s.get("pearson_r")
                sp = s.get("spearman_rho")
                pr_s = f"{pr:+.4f}" if pr is not None else "  N/A"
                sp_s = f"{sp:+.4f}" if sp is not None else "  N/A"
                bej = s.get("best_evidence_jaccard", {}).get("mean")
                bej_s = f"{bej:.4f}" if isinstance(bej, (int, float)) else "N/A"
                print(f"    {pk:<45s}  n={n_aligned:<6d}  "
                      f"κ={kap:+.4f}  wκ={wk:+.4f}  F1={mf1:.4f}  "
                      f"BE-Jacc={bej_s}  r={pr_s}  ρ={sp_s}  MAE={mae:.4f}  Acc={acc:.4f}")
                ci_wk = s.get("bootstrap_ci_weighted_kappa")
                ci_mf1 = s.get("bootstrap_ci_macro_f1")
                if ci_wk:
                    print(f"      95% CI wκ: [{ci_wk['ci_lower']:+.4f}, {ci_wk['ci_upper']:+.4f}]"
                          f"  F1: [{ci_mf1['ci_lower']:.4f}, {ci_mf1['ci_upper']:.4f}]"
                          if ci_mf1 else "")

    print(f"\n{sep}\n")


def write_json_report(report: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"Wrote JSON report → {path}", file=sys.stderr)


def write_csv_report(report: Dict[str, Any], path: Path) -> None:
    """Flatten scalar metrics to a single CSV row."""
    SKIP_KEYS = {
        "confusion_matrix", "per_paper", "per_claim", "per_query",
        "matched_pairs", "paper_scores", "paper_labels", "claim_scores",
        "claim_labels", "aligned_pairs", "core_task",
    }
    flat: Dict[str, Any] = {}

    def _flatten(obj: Any, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in SKIP_KEYS:
                    continue
                _flatten(v, f"{prefix}{k}." if prefix else f"{k}.")
        elif isinstance(obj, list):
            flat[prefix.rstrip(".")] = json.dumps(obj)
        elif isinstance(obj, (int, float, str, bool)):
            flat[prefix.rstrip(".")] = obj
        elif obj is None:
            flat[prefix.rstrip(".")] = ""

    _flatten(report)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)
    print(f"Wrote CSV report → {path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multi-model agreement benchmark for novelty assessment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Two or more pipeline output directories (e.g. output/iclr2024 output/iclr2024_gemini)",
    )
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="Display names for each directory (same order as --dirs); default is directory stem",
    )
    parser.add_argument(
        "--phases",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        choices=[1, 2, 3],
        help="Which phases to evaluate (default: 1 2 3)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output file stem (default: multi_model_report)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv", "both"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Limit to the first N common papers; useful for quick tests",
    )
    parser.add_argument(
        "--task2-shared-across-models",
        action="store_true",
        help="Allow Phase 2 reuse across models: if a model is missing task2_result.json for a paper, reuse Task 2 from another model for that same paper.",
    )
    parser.add_argument(
        "--human-review-json-dir",
        type=Path,
        default=None,
        help="Directory of review JSON files (paper_id.json) to auto-detect n_reviews per paper; overrides --human-n-reviews-file",
    )
    parser.add_argument(
        "--human-n-reviews-file",
        type=Path,
        default=None,
        help="Path to JSON {paper_id: n_reviews} for human/default run; enables claims-per-review normalization",
    )
    parser.add_argument(
        "--human-n-reviews-default",
        type=int,
        default=3,
        help="Fallback n_reviews when paper not in file (default: 3). Set 0 to disable normalization",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress during paper processing",
    )
    args = parser.parse_args()

    if len(args.dirs) < 2:
        print("Error: --dirs must list at least 2 directories.", file=sys.stderr)
        return 1

    for d in args.dirs:
        if not d.is_dir():
            print(f"Error: not a directory: {d}", file=sys.stderr)
            return 1

    names = args.names if args.names else [d.name for d in args.dirs]
    if len(names) != len(args.dirs):
        print("Error: --names must have the same count as --dirs.", file=sys.stderr)
        return 1

    phases = sorted(set(args.phases))
    print(f"Models : {', '.join(names)}", file=sys.stderr)
    print(f"Phases : {phases}", file=sys.stderr)

    n_reviews_map: Optional[Dict[str, int]] = None
    human_n_reviews_default = args.human_n_reviews_default if args.human_n_reviews_default > 0 else 3
    if args.human_n_reviews_default > 0:
        if args.human_review_json_dir and args.human_review_json_dir.is_dir():
            n_reviews_map = _build_n_reviews_from_json_dir(args.human_review_json_dir)
            print(
                f"Auto-detected n_reviews for {len(n_reviews_map)} papers from {args.human_review_json_dir}",
                file=sys.stderr,
            )
        elif args.human_n_reviews_file and args.human_n_reviews_file.exists():
            with args.human_n_reviews_file.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            n_reviews_map = {str(k): int(v) for k, v in raw.items() if isinstance(v, (int, float))}
            print(f"Loaded n_reviews for {len(n_reviews_map)} papers from {args.human_n_reviews_file}", file=sys.stderr)
        elif args.human_n_reviews_file and not args.human_n_reviews_file.exists():
            print(f"Warning: --human-n-reviews-file not found: {args.human_n_reviews_file}", file=sys.stderr)

    report = run_benchmark(
        args.dirs, names, phases,
        max_papers=args.max_papers,
        verbose=args.verbose,
        n_reviews_map=n_reviews_map,
        human_model_name=names[0] if n_reviews_map else None,
        human_n_reviews_default=human_n_reviews_default,
        task2_shared_across_models=args.task2_shared_across_models,
    )
    report["run_dirs"] = [str(d) for d in args.dirs]

    out_stem = args.output or Path("multi_model_report")

    if args.format in ("json", "both"):
        p = out_stem.with_suffix(".json") if args.format == "both" else (
            out_stem if out_stem.suffix == ".json" else out_stem.with_suffix(".json")
        )
        write_json_report(report, p)

    if args.format in ("csv", "both"):
        p = out_stem.with_suffix(".csv") if args.format == "both" else (
            out_stem if out_stem.suffix == ".csv" else out_stem.with_suffix(".csv")
        )
        write_csv_report(report, p)

    print_summary(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
