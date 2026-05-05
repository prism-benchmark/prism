#!/usr/bin/env python3
"""
Phase-specific benchmark evaluation for the novelty assessment pipeline.

Computes metrics for each phase of the ICLR 2024 Task pipeline by comparing
LLM outputs (task1/2/3_result.json) against human annotations
(human_annotations.json).

Phase 1 (Extraction):  Precision, Recall, F1  (claim-level, ROUGE-L matching)
Phase 2 (Retrieval):   Recall@K, MRR, NDCG    (graded relevance)
Phase 3 (Verification): Micro-F1, Macro-F1     (5-class classification)

Usage:
  python scripts/benchmark_evaluation.py --run-dir output/iclr2024
  python scripts/benchmark_evaluation.py --run-dir output/iclr2024 --phases 1 3
  python scripts/benchmark_evaluation.py --run-dir output/iclr2024 -o benchmark_report.json
  python scripts/benchmark_evaluation.py --run-dir output/iclr2024 --run-dir2 output/iclr2024_granite --format both
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from scipy.optimize import linear_sum_assignment
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

RETRIEVAL_K_VALUES: List[int] = [5, 10, 20, 30]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def tokenize(text: str) -> List[str]:
    """Lowercase whitespace-and-punctuation tokenizer."""
    return re.findall(r"[a-z0-9]+", text.lower())


def safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


# ===================================================================
# Phase 1 – Extraction metrics (Precision / Recall / F1)
# ===================================================================

def _lcs_length(x: Sequence[str], y: Sequence[str]) -> int:
    """Length of the longest common subsequence (dynamic programming)."""
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


def rouge_l_f1(hypothesis: str, reference: str) -> float:
    """Token-level ROUGE-L F1 between two texts."""
    hyp_tokens = tokenize(hypothesis)
    ref_tokens = tokenize(reference)
    if not hyp_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(hyp_tokens, ref_tokens)
    p = safe_div(lcs, len(hyp_tokens))
    r = safe_div(lcs, len(ref_tokens))
    return safe_div(2 * p * r, p + r)


def _greedy_match(
    llm_texts: List[str],
    gt_texts: List[str],
    threshold: float = 0.5,
) -> List[Tuple[int, int, float]]:
    """
    Greedy bipartite matching: pick best (llm_i, gt_j) pairs by ROUGE-L F1
    above *threshold*.  Falls back to greedy when scipy is unavailable.
    """
    n_llm = len(llm_texts)
    n_gt = len(gt_texts)
    if n_llm == 0 or n_gt == 0:
        return []

    scores = [
        [rouge_l_f1(llm_texts[i], gt_texts[j]) for j in range(n_gt)]
        for i in range(n_llm)
    ]

    matches: List[Tuple[int, int, float]] = []

    if HAS_SCIPY:
        cost = [[-s for s in row] for row in scores]
        row_idx, col_idx = linear_sum_assignment(cost)
        for ri, ci in zip(row_idx, col_idx):
            ri, ci = int(ri), int(ci)
            if scores[ri][ci] >= threshold:
                matches.append((ri, ci, scores[ri][ci]))
    else:
        used_gt: set = set()
        used_llm: set = set()
        flat = []
        for i in range(n_llm):
            for j in range(n_gt):
                if scores[i][j] >= threshold:
                    flat.append((scores[i][j], i, j))
        flat.sort(reverse=True)
        for sc, i, j in flat:
            if i not in used_llm and j not in used_gt:
                matches.append((i, j, sc))
                used_llm.add(i)
                used_gt.add(j)

    return matches


def evaluate_phase1(
    llm_claims: List[Dict[str, Any]],
    gt_claims: List[Dict[str, Any]],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Evaluate extraction quality (claim-level P/R/F1).

    Both inputs are lists of dicts with at least a ``text`` key.
    """
    llm_texts = [c.get("text", "") for c in llm_claims]
    gt_texts = [c.get("text", "") for c in gt_claims]

    matches = _greedy_match(llm_texts, gt_texts, threshold)

    tp = len(matches)
    precision = safe_div(tp, len(llm_texts))
    recall = safe_div(tp, len(gt_texts))
    f1 = safe_div(2 * precision * recall, precision + recall)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "n_llm_claims": len(llm_texts),
        "n_gt_claims": len(gt_texts),
        "n_matched": tp,
        "match_threshold": threshold,
        "matched_pairs": [
            {
                "llm_idx": m[0],
                "gt_idx": m[1],
                "rouge_l_f1": round(m[2], 4),
            }
            for m in matches
        ],
    }


# ===================================================================
# Phase 2 – Retrieval metrics (Recall@K, MRR, NDCG)
# ===================================================================

def _dcg(relevances: List[float], k: int) -> float:
    """Discounted Cumulative Gain at position k."""
    val = 0.0
    for i, rel in enumerate(relevances[:k]):
        val += rel / math.log2(i + 2)  # i+2 because log2(1)=0
    return val


def evaluate_phase2_single_query(
    ranked_ids: List[str],
    gt_paper_ids: List[str],
    graded_relevance: Dict[str, int],
) -> Dict[str, Any]:
    """
    Compute retrieval metrics for one query (contribution / claim).

    ranked_ids:        ordered list of candidate paper IDs from the retrieval pool
    gt_paper_ids:      set of relevant paper IDs from human annotation
    graded_relevance:  {paper_id: grade} where grade in {0, 1, 2}
    """
    gt_set = set(gt_paper_ids)
    n_relevant = len(gt_set)
    if n_relevant == 0:
        return {
            "recall_at": {str(k): 0.0 for k in RETRIEVAL_K_VALUES},
            "mrr": 0.0,
            "ndcg": {str(k): 0.0 for k in RETRIEVAL_K_VALUES},
            "n_relevant": 0,
        }

    # Recall@K
    recall_at: Dict[str, float] = {}
    for k in RETRIEVAL_K_VALUES:
        found = sum(1 for pid in ranked_ids[:k] if pid in gt_set)
        recall_at[str(k)] = round(safe_div(found, n_relevant), 4)

    # MRR
    rr = 0.0
    for rank, pid in enumerate(ranked_ids, start=1):
        if pid in gt_set:
            rr = 1.0 / rank
            break

    # NDCG@K
    rel_vector = [float(graded_relevance.get(pid, 0)) for pid in ranked_ids]
    ideal_rels = sorted(
        [float(graded_relevance.get(pid, 0)) for pid in gt_paper_ids],
        reverse=True,
    )
    ndcg_at: Dict[str, float] = {}
    for k in RETRIEVAL_K_VALUES:
        dcg_val = _dcg(rel_vector, k)
        idcg_val = _dcg(ideal_rels, k)
        ndcg_at[str(k)] = round(safe_div(dcg_val, idcg_val), 4)

    return {
        "recall_at": recall_at,
        "mrr": round(rr, 4),
        "ndcg": ndcg_at,
        "n_relevant": n_relevant,
    }


def evaluate_phase2(
    task2_data: Dict[str, Any],
    gt_retrieval: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate retrieval quality across all queries for one paper.

    task2_data:    parsed task2_result.json
    gt_retrieval:  task2_retrieval section from human_annotations.json
    """
    pool = task2_data.get("candidate_pool_top30") or []
    ranked_ids = [c.get("cand_id", "") for c in pool]

    per_query: List[Dict[str, Any]] = []
    gt_keys = [k for k in gt_retrieval if not k.startswith("_")]

    for claim_id in gt_keys:
        entry = gt_retrieval[claim_id]
        if not isinstance(entry, dict):
            continue
        gt_paper_ids = entry.get("paper_ids", [])
        graded = entry.get("graded_relevance", {})
        if isinstance(graded, dict):
            graded = {str(k): int(v) for k, v in graded.items()
                      if str(v).lstrip("-").isdigit()}
        else:
            graded = {}

        q_result = evaluate_phase2_single_query(ranked_ids, gt_paper_ids, graded)
        q_result["claim_id"] = claim_id
        per_query.append(q_result)

    if not per_query:
        return {
            "recall_at": {str(k): 0.0 for k in RETRIEVAL_K_VALUES},
            "mrr": 0.0,
            "ndcg": {str(k): 0.0 for k in RETRIEVAL_K_VALUES},
            "n_queries": 0,
            "per_query": [],
        }

    n = len(per_query)
    avg_recall: Dict[str, float] = {}
    avg_ndcg: Dict[str, float] = {}
    for k in RETRIEVAL_K_VALUES:
        sk = str(k)
        avg_recall[sk] = round(sum(q["recall_at"][sk] for q in per_query) / n, 4)
        avg_ndcg[sk] = round(sum(q["ndcg"][sk] for q in per_query) / n, 4)
    avg_mrr = round(sum(q["mrr"] for q in per_query) / n, 4)

    return {
        "recall_at": avg_recall,
        "mrr": avg_mrr,
        "ndcg": avg_ndcg,
        "n_queries": n,
        "per_query": per_query,
    }


# ===================================================================
# Phase 3 – Verification metrics (Micro-F1, Macro-F1, confusion matrix)
# ===================================================================

def _classify_score(score: float) -> str:
    """Map a numeric final_score to the nearest label."""
    rounded = round(score)
    rounded = max(-2, min(2, rounded))
    return SCORE_LABEL_MAP.get(rounded, "AMBIGUOUS")


def evaluate_phase3(
    task3_data: Dict[str, Any],
    gt_verification: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate verification quality (Micro-F1, Macro-F1, confusion matrix).

    task3_data:        parsed task3_result.json
    gt_verification:   task3_verification section from human_annotations.json
    """
    gt_labels_raw = gt_verification.get("claim_labels", {})

    aggregated = task3_data.get("aggregated") or []
    pred_map: Dict[str, str] = {}
    for item in aggregated:
        cid = item.get("review_sentence_id", "")
        fs = item.get("final_score")
        if fs is not None and cid:
            pred_map[cid] = _classify_score(float(fs))

    y_true: List[str] = []
    y_pred: List[str] = []
    per_claim: List[Dict[str, Any]] = []

    for cid, gt_entry in gt_labels_raw.items():
        if not isinstance(gt_entry, dict):
            continue
        gt_label = gt_entry.get("label", "")
        if gt_label not in LABEL_SCORE_MAP:
            continue
        pred_label = pred_map.get(cid)
        if pred_label is None:
            continue

        y_true.append(gt_label)
        y_pred.append(pred_label)
        per_claim.append({
            "claim_id": cid,
            "gt_label": gt_label,
            "pred_label": pred_label,
            "correct": gt_label == pred_label,
        })

    n = len(y_true)
    if n == 0:
        return {
            "micro_f1": 0.0,
            "macro_f1": 0.0,
            "accuracy": 0.0,
            "n_claims": 0,
            "confusion_matrix": {},
            "per_class": {},
            "per_claim": [],
        }

    # Confusion matrix
    cm: Dict[str, Dict[str, int]] = {
        la: {lb: 0 for lb in ALL_LABELS} for la in ALL_LABELS
    }
    for gt, pr in zip(y_true, y_pred):
        cm[gt][pr] += 1

    # Per-class metrics
    per_class: Dict[str, Dict[str, float]] = {}
    macro_f1_sum = 0.0
    n_classes_present = 0

    # Micro accumulators
    micro_tp_total = 0
    micro_fp_total = 0
    micro_fn_total = 0

    for label in ALL_LABELS:
        tp = cm[label][label]
        fp = sum(cm[other][label] for other in ALL_LABELS) - tp
        fn = sum(cm[label][other] for other in ALL_LABELS) - tp

        micro_tp_total += tp
        micro_fp_total += fp
        micro_fn_total += fn

        p = safe_div(tp, tp + fp)
        r = safe_div(tp, tp + fn)
        f = safe_div(2 * p * r, p + r)
        support = tp + fn

        per_class[label] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "support": support,
        }

        if support > 0:
            macro_f1_sum += f
            n_classes_present += 1

    macro_f1 = safe_div(macro_f1_sum, n_classes_present)
    micro_p = safe_div(micro_tp_total, micro_tp_total + micro_fp_total)
    micro_r = safe_div(micro_tp_total, micro_tp_total + micro_fn_total)
    micro_f1 = safe_div(2 * micro_p * micro_r, micro_p + micro_r)
    accuracy = safe_div(sum(cm[l][l] for l in ALL_LABELS), n)

    return {
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(accuracy, 4),
        "n_claims": n,
        "n_classes_with_support": n_classes_present,
        "confusion_matrix": cm,
        "per_class": per_class,
        "per_claim": per_claim,
    }


# ===================================================================
# Paper-level orchestration
# ===================================================================

def evaluate_paper(
    paper_dir: Path,
    gt_dir: Path,
    phases: List[int],
    match_threshold: float = 0.5,
) -> Optional[Dict[str, Any]]:
    """Run requested phase evaluations for a single paper."""
    paper_id = paper_dir.name
    gt_path = gt_dir / paper_id / "human_annotations.json"
    gt = load_json(gt_path)
    if gt is None:
        return None

    result: Dict[str, Any] = {"paper_id": paper_id}

    if 1 in phases:
        task1 = load_json(paper_dir / "task1_result.json")
        gt_claims = (gt.get("task1_extraction") or {}).get("novelty_claims", [])
        gt_claims = [c for c in gt_claims if c.get("text", "").strip()
                     and not c.get("text", "").startswith("__")]
        if task1 and gt_claims:
            llm_claims = (task1.get("review") or {}).get("novelty_claims", [])
            result["phase1"] = evaluate_phase1(llm_claims, gt_claims, match_threshold)
        else:
            result["phase1"] = None

    if 2 in phases:
        task2 = load_json(paper_dir / "task2_result.json")
        gt_ret = gt.get("task2_retrieval") or {}
        has_gt = any(
            isinstance(gt_ret.get(k), dict) and gt_ret[k].get("paper_ids")
            for k in gt_ret if not k.startswith("_")
        )
        if task2 and has_gt:
            result["phase2"] = evaluate_phase2(task2, gt_ret)
        else:
            result["phase2"] = None

    if 3 in phases:
        task3 = load_json(paper_dir / "task3_result.json")
        gt_ver = gt.get("task3_verification") or {}
        has_gt = any(
            isinstance(v, dict) and v.get("label", "").strip()
            and not v.get("label", "").startswith("__")
            for v in (gt_ver.get("claim_labels") or {}).values()
        )
        if task3 and has_gt:
            result["phase3"] = evaluate_phase3(task3, gt_ver)
        else:
            result["phase3"] = None

    return result


# ===================================================================
# Aggregation across papers
# ===================================================================

def aggregate_results(
    per_paper: List[Dict[str, Any]],
    phases: List[int],
) -> Dict[str, Any]:
    """Compute corpus-level aggregate metrics from per-paper results."""
    agg: Dict[str, Any] = {"n_papers_total": len(per_paper)}

    # ---- Phase 1 aggregate ----
    if 1 in phases:
        p1_results = [p["phase1"] for p in per_paper if p.get("phase1")]
        if p1_results:
            n = len(p1_results)
            agg["phase1_extraction"] = {
                "n_papers": n,
                "precision": round(sum(r["precision"] for r in p1_results) / n, 4),
                "recall": round(sum(r["recall"] for r in p1_results) / n, 4),
                "f1": round(sum(r["f1"] for r in p1_results) / n, 4),
                "total_llm_claims": sum(r["n_llm_claims"] for r in p1_results),
                "total_gt_claims": sum(r["n_gt_claims"] for r in p1_results),
                "total_matched": sum(r["n_matched"] for r in p1_results),
            }
            # Corpus-level P/R/F1 (micro-averaged)
            total_matched = agg["phase1_extraction"]["total_matched"]
            total_llm = agg["phase1_extraction"]["total_llm_claims"]
            total_gt = agg["phase1_extraction"]["total_gt_claims"]
            cp = safe_div(total_matched, total_llm)
            cr = safe_div(total_matched, total_gt)
            cf = safe_div(2 * cp * cr, cp + cr)
            agg["phase1_extraction"]["corpus_precision"] = round(cp, 4)
            agg["phase1_extraction"]["corpus_recall"] = round(cr, 4)
            agg["phase1_extraction"]["corpus_f1"] = round(cf, 4)
        else:
            agg["phase1_extraction"] = None

    # ---- Phase 2 aggregate ----
    if 2 in phases:
        p2_results = [p["phase2"] for p in per_paper if p.get("phase2")]
        if p2_results:
            n = len(p2_results)
            avg_recall: Dict[str, float] = {}
            avg_ndcg: Dict[str, float] = {}
            for k in RETRIEVAL_K_VALUES:
                sk = str(k)
                avg_recall[sk] = round(
                    sum(r["recall_at"][sk] for r in p2_results) / n, 4
                )
                avg_ndcg[sk] = round(
                    sum(r["ndcg"][sk] for r in p2_results) / n, 4
                )
            avg_mrr = round(sum(r["mrr"] for r in p2_results) / n, 4)

            agg["phase2_retrieval"] = {
                "n_papers": n,
                "recall_at": avg_recall,
                "mrr": avg_mrr,
                "ndcg": avg_ndcg,
                "total_queries": sum(r["n_queries"] for r in p2_results),
            }
        else:
            agg["phase2_retrieval"] = None

    # ---- Phase 3 aggregate (corpus-level confusion matrix) ----
    if 3 in phases:
        p3_results = [p["phase3"] for p in per_paper if p.get("phase3")]
        if p3_results:
            corpus_cm: Dict[str, Dict[str, int]] = {
                la: {lb: 0 for lb in ALL_LABELS} for la in ALL_LABELS
            }
            for r in p3_results:
                for la in ALL_LABELS:
                    for lb in ALL_LABELS:
                        corpus_cm[la][lb] += r["confusion_matrix"].get(la, {}).get(lb, 0)

            total_n = sum(corpus_cm[la][lb] for la in ALL_LABELS for lb in ALL_LABELS)
            micro_tp = 0
            micro_fp = 0
            micro_fn = 0
            per_class: Dict[str, Dict[str, float]] = {}
            macro_sum = 0.0
            n_cls = 0

            for label in ALL_LABELS:
                tp = corpus_cm[label][label]
                fp = sum(corpus_cm[o][label] for o in ALL_LABELS) - tp
                fn = sum(corpus_cm[label][o] for o in ALL_LABELS) - tp
                micro_tp += tp
                micro_fp += fp
                micro_fn += fn
                p = safe_div(tp, tp + fp)
                r_val = safe_div(tp, tp + fn)
                f = safe_div(2 * p * r_val, p + r_val)
                support = tp + fn
                per_class[label] = {
                    "precision": round(p, 4),
                    "recall": round(r_val, 4),
                    "f1": round(f, 4),
                    "support": support,
                }
                if support > 0:
                    macro_sum += f
                    n_cls += 1

            macro_f1 = safe_div(macro_sum, n_cls)
            micro_p = safe_div(micro_tp, micro_tp + micro_fp)
            micro_r = safe_div(micro_tp, micro_tp + micro_fn)
            micro_f1 = safe_div(2 * micro_p * micro_r, micro_p + micro_r)
            accuracy = safe_div(sum(corpus_cm[l][l] for l in ALL_LABELS), total_n // 1 if total_n else 1)

            agg["phase3_verification"] = {
                "n_papers": len(p3_results),
                "n_claims": sum(r["n_claims"] for r in p3_results),
                "micro_precision": round(micro_p, 4),
                "micro_recall": round(micro_r, 4),
                "micro_f1": round(micro_f1, 4),
                "macro_f1": round(macro_f1, 4),
                "accuracy": round(accuracy, 4),
                "confusion_matrix": corpus_cm,
                "per_class": per_class,
            }
        else:
            agg["phase3_verification"] = None

    return agg


# ===================================================================
# Cross-model comparison
# ===================================================================

def cross_model_comparison(
    reports: List[Dict[str, Any]],
    names: List[str],
) -> Dict[str, Any]:
    """Side-by-side summary of aggregate metrics from multiple runs."""
    comparison: Dict[str, Any] = {"models": names}

    for phase_key in ["phase1_extraction", "phase2_retrieval", "phase3_verification"]:
        rows: Dict[str, Dict[str, Any]] = {}
        for name, report in zip(names, reports):
            section = report.get(phase_key)
            if section is None:
                continue
            rows[name] = {
                k: v for k, v in section.items()
                if k not in ("confusion_matrix", "per_class", "per_paper")
            }
        if rows:
            comparison[phase_key] = rows

    return comparison


# ===================================================================
# Report output
# ===================================================================

def write_json_report(report: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Wrote JSON report to {path}", file=sys.stderr)


def write_csv_report(report: Dict[str, Any], path: Path) -> None:
    """Flatten aggregate metrics into a single-row CSV."""
    flat: Dict[str, Any] = {}

    def _flatten(obj: Any, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in ("per_paper", "per_claim", "per_query",
                         "matched_pairs", "confusion_matrix"):
                    continue
                _flatten(v, f"{prefix}{k}." if prefix else f"{k}.")
        elif isinstance(obj, (int, float, str, bool)):
            key = prefix.rstrip(".")
            flat[key] = obj

    _flatten(report)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)
    print(f"Wrote CSV report to {path}", file=sys.stderr)


def print_summary(report: Dict[str, Any]) -> None:
    """Print a concise human-readable summary to stdout."""
    print(f"\n{'='*60}")
    print("BENCHMARK EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Papers evaluated: {report.get('n_papers_total', 0)}")

    p1 = report.get("phase1_extraction")
    if p1:
        print(f"\n--- Phase 1: Extraction (n={p1['n_papers']}) ---")
        print(f"  Macro-avg  P={p1['precision']:.4f}  R={p1['recall']:.4f}  F1={p1['f1']:.4f}")
        print(f"  Corpus     P={p1['corpus_precision']:.4f}  R={p1['corpus_recall']:.4f}  F1={p1['corpus_f1']:.4f}")

    p2 = report.get("phase2_retrieval")
    if p2:
        print(f"\n--- Phase 2: Retrieval (n={p2['n_papers']}) ---")
        for k in RETRIEVAL_K_VALUES:
            sk = str(k)
            print(f"  Recall@{k:<3} = {p2['recall_at'][sk]:.4f}   NDCG@{k:<3} = {p2['ndcg'][sk]:.4f}")
        print(f"  MRR       = {p2['mrr']:.4f}")

    p3 = report.get("phase3_verification")
    if p3:
        print(f"\n--- Phase 3: Verification (n={p3['n_papers']}, claims={p3['n_claims']}) ---")
        print(f"  Micro-F1 = {p3['micro_f1']:.4f}   Macro-F1 = {p3['macro_f1']:.4f}   Accuracy = {p3['accuracy']:.4f}")
        print("  Per-class:")
        for label in ALL_LABELS:
            c = p3["per_class"].get(label, {})
            print(f"    {label:<12s}  P={c.get('precision',0):.4f}  R={c.get('recall',0):.4f}  F1={c.get('f1',0):.4f}  support={c.get('support',0)}")

    print(f"\n{'='*60}\n")


# ===================================================================
# CLI
# ===================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase-specific benchmark evaluation for novelty assessment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory with per-paper LLM task results (e.g. output/iclr2024)",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=None,
        help="Directory with human_annotations.json per paper (default: same as --run-dir)",
    )
    parser.add_argument(
        "--phases",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Which phases to evaluate (default: 1 2 3)",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.5,
        help="ROUGE-L F1 threshold for Phase 1 claim matching (default: 0.5)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output file path (default: benchmark_report.json or .csv)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv", "both"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--run-dir2",
        type=Path,
        default=None,
        help="Optional second run directory for cross-model comparison",
    )
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="Display names for --run-dir (and --run-dir2 if given)",
    )
    parser.add_argument(
        "--papers",
        nargs="*",
        default=None,
        help="Evaluate only these paper IDs (default: all with annotations)",
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    gt_dir: Path = args.gt_dir if args.gt_dir else run_dir
    phases = sorted(set(args.phases))

    if not run_dir.is_dir():
        print(f"Error: {run_dir} is not a directory.", file=sys.stderr)
        return 1

    # Discover papers with annotations
    if args.papers:
        paper_ids = args.papers
    else:
        paper_ids = sorted(
            d.name for d in gt_dir.iterdir()
            if d.is_dir() and (d / "human_annotations.json").exists()
        )

    if not paper_ids:
        print("No papers with human_annotations.json found.", file=sys.stderr)
        return 0

    # Evaluate each paper
    per_paper: List[Dict[str, Any]] = []
    for pid in paper_ids:
        paper_dir = run_dir / pid
        if not paper_dir.is_dir():
            continue
        result = evaluate_paper(paper_dir, gt_dir, phases, args.match_threshold)
        if result is not None:
            per_paper.append(result)

    if not per_paper:
        print("No papers could be evaluated (check annotations).", file=sys.stderr)
        return 0

    # Aggregate
    report = aggregate_results(per_paper, phases)
    report["run_dir"] = str(run_dir)
    report["gt_dir"] = str(gt_dir)
    report["phases_evaluated"] = phases
    report["per_paper"] = per_paper

    # Cross-model comparison
    if args.run_dir2 and args.run_dir2.is_dir():
        per_paper2: List[Dict[str, Any]] = []
        for pid in paper_ids:
            pd2 = args.run_dir2 / pid
            if not pd2.is_dir():
                continue
            r2 = evaluate_paper(pd2, gt_dir, phases, args.match_threshold)
            if r2 is not None:
                per_paper2.append(r2)
        if per_paper2:
            report2 = aggregate_results(per_paper2, phases)
            names = args.names or [run_dir.name, args.run_dir2.name]
            if len(names) < 2:
                names = [run_dir.name, args.run_dir2.name]
            report["cross_model_comparison"] = cross_model_comparison(
                [report, report2], names
            )

    # Output
    out_path = args.output or Path("benchmark_report")

    if args.format in ("json", "both"):
        p = out_path.with_suffix(".json") if args.format == "both" else (
            out_path if out_path.suffix == ".json" else out_path.with_suffix(".json")
        )
        write_json_report(report, p)

    if args.format in ("csv", "both"):
        p = out_path.with_suffix(".csv") if args.format == "both" else (
            out_path if out_path.suffix == ".csv" else out_path.with_suffix(".csv")
        )
        write_csv_report(report, p)

    print_summary(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
