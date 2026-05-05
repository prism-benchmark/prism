#!/usr/bin/env python3
"""
Stage-wise error decomposition for novelty assessment pipeline.

Compares two pipeline runs (e.g., human vs SEA) and attributes
disagreement to extraction, retrieval, or judging stages.

Usage:
  python scripts/error_decomposition.py \
    --dir-a output/iclr2024 --dir-b output/iclr2024_SEA \
    --name-a human --name-b sea
"""

import argparse
import json
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# ROUGE-L helpers (self-contained, no external deps)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase whitespace tokenisation."""
    return text.lower().split()


def _lcs_length(x: list[str], y: list[str]) -> int:
    """Length of longest common subsequence."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        cur = [0] * (n + 1)
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(cur[j - 1], prev[j])
        prev = cur
    return prev[n]


def rouge_l_f1(text_a: str, text_b: str) -> float:
    """Symmetric ROUGE-L F1 between two texts."""
    toks_a = _tokenize(text_a)
    toks_b = _tokenize(text_b)
    if not toks_a or not toks_b:
        return 0.0
    lcs = _lcs_length(toks_a, toks_b)
    precision = lcs / len(toks_a)
    recall = lcs / len(toks_b)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict | None:
    """Load JSON file, return None on missing/corrupt."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _find_common_papers(dir_a: Path, dir_b: Path, max_papers: int | None = None) -> list[str]:
    """Return paper IDs that exist in both directories."""
    ids_a = {d.name for d in dir_a.iterdir() if d.is_dir()}
    ids_b = {d.name for d in dir_b.iterdir() if d.is_dir()}
    common = sorted(ids_a & ids_b)
    if max_papers is not None:
        common = common[:max_papers]
    return common


# ---------------------------------------------------------------------------
# Stage 1: Extraction agreement
# ---------------------------------------------------------------------------

def extraction_agreement(dir_a: Path, dir_b: Path, paper_ids: list[str]) -> dict:
    """Compare novelty_claims extracted in task1 between two runs."""
    rouge_scores: list[float] = []
    total_claims_a = 0
    total_matched = 0
    total_unmatched = 0

    for pid in paper_ids:
        t1_a = _load_json(dir_a / pid / "task1_result.json")
        t1_b = _load_json(dir_b / pid / "task1_result.json")
        if t1_a is None or t1_b is None:
            continue

        claims_a = t1_a.get("review", {}).get("novelty_claims", [])
        claims_b = t1_b.get("review", {}).get("novelty_claims", [])
        texts_a = [c.get("text", "") for c in claims_a if c.get("text")]
        texts_b = [c.get("text", "") for c in claims_b if c.get("text")]

        if not texts_a:
            continue
        total_claims_a += len(texts_a)

        # Greedy best-match: for each claim in A find best ROUGE-L in B
        used_b: set[int] = set()
        for ta in texts_a:
            best_score = 0.0
            best_idx = -1
            for j, tb in enumerate(texts_b):
                if j in used_b:
                    continue
                s = rouge_l_f1(ta, tb)
                if s > best_score:
                    best_score = s
                    best_idx = j
            rouge_scores.append(best_score)
            if best_score >= 0.35:
                total_matched += 1
                if best_idx >= 0:
                    used_b.add(best_idx)
            else:
                total_unmatched += 1

    mean_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
    pct_matched = total_matched / total_claims_a if total_claims_a else 0.0
    pct_unmatched = total_unmatched / total_claims_a if total_claims_a else 0.0

    return {
        "mean_rouge_l_overlap": round(mean_rouge, 4),
        "pct_matched_claims": round(pct_matched, 4),
        "pct_unmatched_claims": round(pct_unmatched, 4),
        "disagreement": round(1 - mean_rouge, 4),
    }


# ---------------------------------------------------------------------------
# Stage 2: Retrieval agreement
# ---------------------------------------------------------------------------

def retrieval_agreement(dir_a: Path, dir_b: Path, paper_ids: list[str]) -> dict:
    """Compare candidate_pool_top30 overlap (Jaccard) between two runs."""
    jaccards: list[float] = []
    papers_ge50 = 0

    for pid in paper_ids:
        t2_a = _load_json(dir_a / pid / "task2_result.json")
        t2_b = _load_json(dir_b / pid / "task2_result.json")
        if t2_a is None or t2_b is None:
            continue

        ids_a = {c["cand_id"] for c in t2_a.get("candidate_pool_top30", []) if "cand_id" in c}
        ids_b = {c["cand_id"] for c in t2_b.get("candidate_pool_top30", []) if "cand_id" in c}

        if not ids_a and not ids_b:
            jaccards.append(1.0)  # both empty → agree
        elif not ids_a or not ids_b:
            jaccards.append(0.0)
        else:
            j = len(ids_a & ids_b) / len(ids_a | ids_b)
            jaccards.append(j)

        if jaccards[-1] >= 0.5:
            papers_ge50 += 1

    mean_jaccard = sum(jaccards) / len(jaccards) if jaccards else 0.0
    pct_overlap_50 = papers_ge50 / len(jaccards) if jaccards else 0.0

    return {
        "mean_jaccard_30": round(mean_jaccard, 4),
        "pct_papers_overlap_50": round(pct_overlap_50, 4),
        "disagreement": round(1 - mean_jaccard, 4),
    }


# ---------------------------------------------------------------------------
# Stage 3: Judging agreement
# ---------------------------------------------------------------------------

def _score_to_label(score: float) -> str:
    """Bin final_score into coarse labels for agreement computation."""
    if score <= -1:
        return "refuted"
    elif score >= 1:
        return "supported"
    else:
        return "ambiguous"


def judging_agreement(dir_a: Path, dir_b: Path, paper_ids: list[str]) -> dict:
    """Compare final_score judgments on aligned claims between two runs."""
    labels_a: list[str] = []
    labels_b: list[str] = []

    for pid in paper_ids:
        t3_a = _load_json(dir_a / pid / "task3_result.json")
        t3_b = _load_json(dir_b / pid / "task3_result.json")
        if t3_a is None or t3_b is None:
            continue

        agg_a = {r["review_sentence_id"]: r for r in t3_a.get("aggregated", [])}
        agg_b = {r["review_sentence_id"]: r for r in t3_b.get("aggregated", [])}

        # Exact match on claim IDs first
        matched_ids = set(agg_a.keys()) & set(agg_b.keys())

        # Fallback: ROUGE-L for unmatched claims
        unmatched_a = {k: v for k, v in agg_a.items() if k not in matched_ids}
        unmatched_b = {k: v for k, v in agg_b.items() if k not in matched_ids}

        if unmatched_a and unmatched_b:
            used_b: set[str] = set()
            for ka, va in unmatched_a.items():
                best_score = 0.0
                best_kb = None
                for kb, vb in unmatched_b.items():
                    if kb in used_b:
                        continue
                    s = rouge_l_f1(va.get("text", ""), vb.get("text", ""))
                    if s > best_score:
                        best_score = s
                        best_kb = kb
                if best_score >= 0.35 and best_kb is not None:
                    la = _score_to_label(va.get("final_score", 0))
                    lb = _score_to_label(agg_b[best_kb].get("final_score", 0))
                    labels_a.append(la)
                    labels_b.append(lb)
                    used_b.add(best_kb)

        # Exact-matched pairs
        for cid in matched_ids:
            la = _score_to_label(agg_a[cid].get("final_score", 0))
            lb = _score_to_label(agg_b[cid].get("final_score", 0))
            labels_a.append(la)
            labels_b.append(lb)

    n_aligned = len(labels_a)
    if n_aligned == 0:
        return {
            "cohen_kappa": 0.0,
            "accuracy": 0.0,
            "n_aligned_claims": 0,
            "disagreement": 1.0,
        }

    # Accuracy
    accuracy = sum(a == b for a, b in zip(labels_a, labels_b)) / n_aligned

    # Cohen's kappa
    categories = sorted(set(labels_a) | set(labels_b))
    n = n_aligned
    po = accuracy
    pe = 0.0
    for cat in categories:
        fa = sum(1 for x in labels_a if x == cat) / n
        fb = sum(1 for x in labels_b if x == cat) / n
        pe += fa * fb
    kappa = (po - pe) / (1 - pe) if pe < 1.0 else 0.0

    return {
        "cohen_kappa": round(kappa, 4),
        "accuracy": round(accuracy, 4),
        "n_aligned_claims": n_aligned,
        "disagreement": round(1 - accuracy, 4),
    }


# ---------------------------------------------------------------------------
# Error decomposition
# ---------------------------------------------------------------------------

def decompose_error(dir_a: Path, dir_b: Path, paper_ids: list[str],
                    name_a: str, name_b: str) -> dict:
    """Run all three stage comparisons and produce attribution table."""
    ext = extraction_agreement(dir_a, dir_b, paper_ids)
    ret = retrieval_agreement(dir_a, dir_b, paper_ids)
    jdg = judging_agreement(dir_a, dir_b, paper_ids)

    e_dis = ext["disagreement"]
    r_dis = ret["disagreement"]
    j_dis = jdg["disagreement"]
    total = e_dis + r_dis + j_dis

    if total > 0:
        pct_e = e_dis / total * 100
        pct_r = r_dis / total * 100
        pct_j = j_dis / total * 100
    else:
        pct_e = pct_r = pct_j = 0.0

    return {
        "n_papers": len(paper_ids),
        "name_a": name_a,
        "name_b": name_b,
        "extraction": ext,
        "retrieval": ret,
        "judging": jdg,
        "attribution": {
            "extraction_pct": round(pct_e, 1),
            "retrieval_pct": round(pct_r, 1),
            "judging_pct": round(pct_j, 1),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage-wise error decomposition for novelty assessment pipeline."
    )
    parser.add_argument("--dir-a", required=True, type=Path,
                        help="First pipeline output directory")
    parser.add_argument("--dir-b", required=True, type=Path,
                        help="Second pipeline output directory")
    parser.add_argument("--name-a", default=None,
                        help="Display name for dir-a (default: directory stem)")
    parser.add_argument("--name-b", default=None,
                        help="Display name for dir-b (default: directory stem)")
    parser.add_argument("-o", "--output", default="error_decomposition.json",
                        help="Output JSON path (default: error_decomposition.json)")
    parser.add_argument("--max-papers", type=int, default=None,
                        help="Limit number of papers (for testing)")
    args = parser.parse_args()

    if not args.dir_a.is_dir():
        print(f"Error: {args.dir_a} is not a directory", file=sys.stderr)
        sys.exit(1)
    if not args.dir_b.is_dir():
        print(f"Error: {args.dir_b} is not a directory", file=sys.stderr)
        sys.exit(1)

    name_a = args.name_a or args.dir_a.stem
    name_b = args.name_b or args.dir_b.stem

    paper_ids = _find_common_papers(args.dir_a, args.dir_b, args.max_papers)
    if not paper_ids:
        print("Error: no common paper IDs found between the two directories",
              file=sys.stderr)
        sys.exit(1)

    print(f"Comparing {name_a} vs {name_b} on {len(paper_ids)} papers ...")
    result = decompose_error(args.dir_a, args.dir_b, paper_ids, name_a, name_b)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results written to {out_path}")

    # Pretty-print summary
    print(f"\n{'='*50}")
    print(f"  Error Decomposition: {name_a} vs {name_b}")
    print(f"  Papers: {result['n_papers']}")
    print(f"{'='*50}")
    print(f"  Extraction  ROUGE-L overlap : {result['extraction']['mean_rouge_l_overlap']:.4f}")
    print(f"              disagreement    : {result['extraction']['disagreement']:.4f}")
    print(f"  Retrieval   Jaccard@30      : {result['retrieval']['mean_jaccard_30']:.4f}")
    print(f"              disagreement    : {result['retrieval']['disagreement']:.4f}")
    print(f"  Judging     Cohen's κ       : {result['judging']['cohen_kappa']:.4f}")
    print(f"              accuracy        : {result['judging']['accuracy']:.4f}")
    print(f"              aligned claims  : {result['judging']['n_aligned_claims']}")
    print(f"              disagreement    : {result['judging']['disagreement']:.4f}")
    print(f"{'='*50}")
    print(f"  Attribution:")
    print(f"    Extraction : {result['attribution']['extraction_pct']:5.1f}%")
    print(f"    Retrieval  : {result['attribution']['retrieval_pct']:5.1f}%")
    print(f"    Judging    : {result['attribution']['judging_pct']:5.1f}%")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
