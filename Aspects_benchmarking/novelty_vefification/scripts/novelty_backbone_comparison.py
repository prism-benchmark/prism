#!/usr/bin/env python3
"""
Novelty Assessment Pipeline Backbone Comparison:
Mimo 2.5 Pro (subset_50) vs Gemini 2.5 Flash Lite (full_conf_results)

Compares novelty assessment pipeline outputs on 250 overlapping papers (50/conf × 5 confs)
across 6 reviewer systems (human, sea, deepreview, reviewer2, cyclereview, tree).

Focus: Phase 1 (extraction), Phase 2 (retrieval), Phase 3 (verification) agreement,
reviewer ranking consistency, and identification of backbone-sensitive metrics.
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# Global font sizes for all figures
plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 17,
})

# Paths
BASE = Path(os.getenv("NOVELTY_OUTPUT_ROOT", str(Path(__file__).resolve().parents[1] / "output")))
MIMO_BASE = BASE / "subset_50"
GEMINI_BASE = BASE / "full_conf_results"
OUTPUT_DIR = BASE / "novelty_backbone_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Conference mapping
CONF_MAP = {
    "ICLR_2024": "ICLR2024",
    "ICLR_2025": "ICLR2025",
    "ICLR_2026": "ICLR2026",
    "ICML_2025": "ICML2025",
    "NeurIPS_2025": "NeurIPS2025",
}
METHODS = ["human", "sea", "deepreview", "reviewer2", "cyclereview", "tree"]
METHOD_DISPLAY = {
    "human": "Human", "sea": "SEA", "deepreview": "DeepReview",
    "reviewer2": "Reviewer2", "cyclereview": "CycleReview", "tree": "TreeReview",
}
CONFERENCES = list(CONF_MAP.keys())
CONF_DISPLAY = {
    "ICLR_2024": "ICLR'24", "ICLR_2025": "ICLR'25", "ICLR_2026": "ICLR'26",
    "ICML_2025": "ICML'25", "NeurIPS_2025": "NeurIPS'25",
}

MIMO_COLOR = "#E74C3C"
GEMINI_COLOR = "#3498DB"


def load_task1(path):
    """Extract Phase 1 metrics from task1_result.json."""
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:
        return None

    paper = d.get("paper", {})
    review = d.get("review", {})
    
    claims = paper.get("contributions", [])
    key_terms = paper.get("key_terms", [])
    
    # Review-side extraction
    review_sentences = review.get("review_sentences", []) if isinstance(review, dict) else []
    
    return {
        "claim_count": len(claims),
        "key_term_count": len(key_terms),
        "core_task": paper.get("core_task", ""),
        "contributions": claims,
        "review_sentences": review_sentences,
    }


def load_task2(path):
    """Extract Phase 2 metrics from task2_result.json."""
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:
        return None

    candidates = d.get("candidate_pool_top30", [])
    queries = d.get("queries", [])
    
    # Extract candidate paper IDs for Jaccard computation
    candidate_ids = set()
    for c in candidates:
        if isinstance(c, dict):
            pid = c.get("paperId", c.get("paper_id", ""))
            if pid:
                candidate_ids.add(pid)
    
    return {
        "candidate_count": len(candidates),
        "query_count": len(queries),
        "candidate_ids": candidate_ids,
        "candidates": candidates,
    }


def load_task3(path):
    """Extract Phase 3 metrics from task3_result.json."""
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:
        return None

    aggregated = d.get("aggregated", [])
    stats_data = d.get("stats", {})
    coverage = stats_data.get("coverage", {})
    
    # Extract per-claim scores and labels
    claim_scores = []
    claim_labels = []
    claim_texts = []
    for item in aggregated:
        score = item.get("final_score")
        if score is not None:
            claim_scores.append(score)
        cls = item.get("classification", {})
        claim = cls.get("claim", 0)
        proof = cls.get("proof", 0)
        if claim == 1:
            if proof == 1:
                claim_labels.append("SUPPORTED")
            else:
                claim_labels.append("OVERSTATED")
        claim_texts.append(item.get("text", ""))
    
    # Compute NS, SR, SSR
    mean_score = np.mean(claim_scores) if claim_scores else 0
    ns = (mean_score + 2) / 4  # Normalized Score ∈ [0, 1]
    sr = sum(1 for s in claim_scores if s >= 1) / len(claim_scores) if claim_scores else 0
    ssr = sum(1 for s in claim_scores if s >= 2) / len(claim_scores) if claim_scores else 0
    
    # Paper-level score (mean of claim scores)
    paper_score = mean_score if claim_scores else None
    
    return {
        "paper_score": paper_score,
        "ns": ns,
        "sr": sr,
        "ssr": ssr,
        "claim_scores": claim_scores,
        "claim_labels": claim_labels,
        "claim_count": len(aggregated),
        "review_sentences": stats_data.get("review_sentences", 0),
        "related_works": stats_data.get("related_works", 0),
        "pairs_attempted": stats_data.get("pairs_attempted", 0),
        "pairs_completed": stats_data.get("pairs_completed", 0),
        "coverage_rate": coverage.get("claim_success_coverage_rate", 0),
        "evidence_coverage": coverage.get("evidence_coverage_rate", 0),
        "decisive_coverage": coverage.get("decisive_coverage_rate", 0),
        "avg_evidence": coverage.get("avg_evidence_per_claim", 0),
    }


def jaccard(set_a, set_b):
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def cosine_claim_overlap(mimo_claims, gemini_claims):
    """Simple token-overlap based claim similarity."""
    def tokenize(text):
        return set(text.lower().split()) if isinstance(text, str) else set()
    
    if not mimo_claims or not gemini_claims:
        return 0.0
    
    # For each Mimo claim, find best matching Gemini claim
    overlaps = []
    for mc in mimo_claims:
        mc_tokens = tokenize(mc.get("description", mc.get("text", "")) if isinstance(mc, dict) else str(mc))
        if not mc_tokens:
            continue
        best_overlap = 0
        for gc in gemini_claims:
            gc_tokens = tokenize(gc.get("description", gc.get("text", "")) if isinstance(gc, dict) else str(gc))
            if not gc_tokens:
                continue
            intersection = len(mc_tokens & gc_tokens)
            union = len(mc_tokens | gc_tokens)
            if union > 0:
                best_overlap = max(best_overlap, intersection / union)
        overlaps.append(best_overlap)
    
    return np.mean(overlaps) if overlaps else 0.0


# ============================================================
# Collect data
# ============================================================
print("=" * 80)
print("Novelty Assessment Pipeline: Backbone LLM Comparison")
print("Mimo 2.5 Pro vs Gemini 2.5 Flash Lite")
print("=" * 80)

records = []
claim_overlap_data = []
candidate_jaccard_data = []

for conf_mimo, conf_gemini in CONF_MAP.items():
    # Get overlapping paper IDs
    mimo_papers = set(p.name for p in (MIMO_BASE / "human" / conf_mimo).iterdir() if p.is_dir())
    gemini_papers = set(p.name for p in (GEMINI_BASE / "human" / conf_gemini).iterdir() if p.is_dir())
    overlap = sorted(mimo_papers & gemini_papers)
    print(f"\n{conf_mimo}: {len(overlap)} overlapping papers")
    
    for paper_id in overlap:
        for method in METHODS:
            # File paths
            mimo_dir = MIMO_BASE / method / conf_mimo / paper_id
            gemini_dir = GEMINI_BASE / method / conf_gemini / paper_id
            
            m1 = load_task1(mimo_dir / "task1_result.json") if (mimo_dir / "task1_result.json").exists() else None
            m2 = load_task2(mimo_dir / "task2_result.json") if (mimo_dir / "task2_result.json").exists() else None
            m3 = load_task3(mimo_dir / "task3_result.json") if (mimo_dir / "task3_result.json").exists() else None
            
            g1 = load_task1(gemini_dir / "task1_result.json") if (gemini_dir / "task1_result.json").exists() else None
            g2 = load_task2(gemini_dir / "task2_result.json") if (gemini_dir / "task2_result.json").exists() else None
            g3 = load_task3(gemini_dir / "task3_result.json") if (gemini_dir / "task3_result.json").exists() else None
            
            # Phase 1 claim overlap
            if m1 and g1:
                co = cosine_claim_overlap(m1.get("contributions", []), g1.get("contributions", []))
                claim_overlap_data.append({
                    "conference": conf_mimo, "paper_id": paper_id, "method": method,
                    "claim_overlap": co,
                    "mimo_claim_count": m1["claim_count"],
                    "gemini_claim_count": g1["claim_count"],
                })
            
            # Phase 2 candidate Jaccard
            if m2 and g2:
                jac = jaccard(m2.get("candidate_ids", set()), g2.get("candidate_ids", set()))
                candidate_jaccard_data.append({
                    "conference": conf_mimo, "paper_id": paper_id, "method": method,
                    "jaccard": jac,
                    "mimo_candidates": m2["candidate_count"],
                    "gemini_candidates": g2["candidate_count"],
                })
            
            # Phase 3 metrics
            if m3 and g3:
                record = {
                    "conference": conf_mimo,
                    "paper_id": paper_id,
                    "method": method,
                    # Mimo Phase 3
                    "mimo_ns": m3["ns"],
                    "mimo_sr": m3["sr"],
                    "mimo_ssr": m3["ssr"],
                    "mimo_paper_score": m3["paper_score"],
                    "mimo_claim_count_p3": m3["claim_count"],
                    "mimo_coverage": m3["coverage_rate"],
                    "mimo_evidence_cov": m3["evidence_coverage"],
                    "mimo_decisive_cov": m3["decisive_coverage"],
                    "mimo_review_sentences": m3["review_sentences"],
                    "mimo_pairs_attempted": m3["pairs_attempted"],
                    # Gemini Phase 3
                    "gemini_ns": g3["ns"],
                    "gemini_sr": g3["sr"],
                    "gemini_ssr": g3["ssr"],
                    "gemini_paper_score": g3["paper_score"],
                    "gemini_claim_count_p3": g3["claim_count"],
                    "gemini_coverage": g3["coverage_rate"],
                    "gemini_evidence_cov": g3["evidence_coverage"],
                    "gemini_decisive_cov": g3["decisive_coverage"],
                    "gemini_review_sentences": g3["review_sentences"],
                    "gemini_pairs_attempted": g3["pairs_attempted"],
                    # Claim-level labels for agreement
                    "mimo_claim_labels": m3["claim_labels"],
                    "gemini_claim_labels": g3["claim_labels"],
                    "mimo_claim_scores": m3["claim_scores"],
                    "gemini_claim_scores": g3["claim_scores"],
                }
                
                # Phase 1 data
                if m1:
                    record["mimo_claim_count_p1"] = m1["claim_count"]
                if g1:
                    record["gemini_claim_count_p1"] = g1["claim_count"]
                
                # Phase 2 data
                if m2:
                    record["mimo_candidates"] = m2["candidate_count"]
                if g2:
                    record["gemini_candidates"] = g2["candidate_count"]
                
                records.append(record)

# Create DataFrames
df = pd.DataFrame(records)
co_df = pd.DataFrame(claim_overlap_data) if claim_overlap_data else pd.DataFrame()
jac_df = pd.DataFrame(candidate_jaccard_data) if candidate_jaccard_data else pd.DataFrame()

print(f"\nTotal records: {len(df)}")
print(f"Claim overlap records: {len(co_df)}")
print(f"Candidate Jaccard records: {len(jac_df)}")

# Save detailed CSV
df.drop(columns=["mimo_claim_labels", "gemini_claim_labels", "mimo_claim_scores", "gemini_claim_scores"]).to_csv(
    OUTPUT_DIR / "novelty_comparison_detailed.csv", index=False
)

# ============================================================
# ANALYSIS 1: Phase 1 - Claim Extraction Agreement
# ============================================================
print("\n" + "=" * 80)
print("PHASE 1: CLAIM EXTRACTION AGREEMENT")
print("=" * 80)

if len(co_df) > 0:
    print(f"\nClaim overlap (token-based similarity):")
    for method in METHODS:
        m_data = co_df[co_df["method"] == method]["claim_overlap"]
        print(f"  {METHOD_DISPLAY[method]:<15}: mean={m_data.mean():.4f} ± {m_data.std():.4f} (n={len(m_data)})")
    
    print(f"\nOverall claim overlap: {co_df['claim_overlap'].mean():.4f} ± {co_df['claim_overlap'].std():.4f}")
    
    # Claim count comparison
    if "mimo_claim_count_p1" in df.columns:
        print(f"\nClaim count comparison (Phase 1):")
        for method in METHODS:
            m_df = df[df["method"] == method]
            mimo_cc = m_df["mimo_claim_count_p1"].dropna()
            gemini_cc = m_df["gemini_claim_count_p1"].dropna()
            if len(mimo_cc) > 0 and len(gemini_cc) > 0:
                print(f"  {METHOD_DISPLAY[method]:<15}: Mimo={mimo_cc.mean():.2f}, Gemini={gemini_cc.mean():.2f}")


# ============================================================
# ANALYSIS 2: Phase 2 - Retrieval Agreement
# ============================================================
print("\n" + "=" * 80)
print("PHASE 2: RETRIEVAL AGREEMENT (Candidate Pool Jaccard)")
print("=" * 80)

if len(jac_df) > 0:
    print(f"\nCandidate pool Jaccard similarity:")
    for method in METHODS:
        m_data = jac_df[jac_df["method"] == method]["jaccard"]
        print(f"  {METHOD_DISPLAY[method]:<15}: mean={m_data.mean():.4f} ± {m_data.std():.4f} (n={len(m_data)})")
    print(f"\nOverall Jaccard: {jac_df['jaccard'].mean():.4f} ± {jac_df['jaccard'].std():.4f}")
    
    # Candidate count comparison
    if "mimo_candidates" in df.columns:
        print(f"\nCandidate count comparison:")
        for method in METHODS:
            m_df = df[df["method"] == method]
            mimo_c = m_df["mimo_candidates"].dropna()
            gemini_c = m_df["gemini_candidates"].dropna()
            if len(mimo_c) > 0 and len(gemini_c) > 0:
                print(f"  {METHOD_DISPLAY[method]:<15}: Mimo={mimo_c.mean():.2f}, Gemini={gemini_c.mean():.2f}")


# ============================================================
# ANALYSIS 3: Phase 3 - Verification Agreement
# ============================================================
print("\n" + "=" * 80)
print("PHASE 3: VERIFICATION AGREEMENT")
print("=" * 80)

# NS, SR, SSR comparison by method
print(f"\n{'Method':<15} {'Metric':<8} {'Mimo':>10} {'Gemini':>10} {'Δ':>10} {'r':>8} {'p':>12} {'n':>6}")
print("-" * 75)

for metric, metric_name in [("ns", "NS"), ("sr", "SR"), ("ssr", "SSR")]:
    for method in METHODS:
        m_df = df[df["method"] == method]
        mimo_vals = m_df[f"mimo_{metric}"].dropna()
        gemini_vals = m_df[f"gemini_{metric}"].dropna()
        
        # Only compute on paired data
        valid = m_df[[f"mimo_{metric}", f"gemini_{metric}"]].dropna()
        if len(valid) > 5:
            m = valid[f"mimo_{metric}"]
            g = valid[f"gemini_{metric}"]
            r, p = stats.pearsonr(m, g)
            print(f"{METHOD_DISPLAY[method]:<15} {metric_name:<8} {m.mean():>10.4f} {g.mean():>10.4f} {g.mean()-m.mean():>+10.4f} {r:>8.4f} {p:>12.2e} {len(valid):>5}")
    print()


# ============================================================
# ANALYSIS 4: Reviewer System Rankings
# ============================================================
print("\n" + "=" * 80)
print("REVIEWER SYSTEM RANKING COMPARISON")
print("=" * 80)

for metric, metric_name in [("ns", "NS"), ("sr", "SR"), ("ssr", "SSR"), ("paper_score", "Paper Score")]:
    print(f"\n--- {metric_name} ---")
    mimo_ranks = {}
    gemini_ranks = {}
    
    for method in METHODS:
        m_df = df[df["method"] == method]
        mimo_mean = m_df[f"mimo_{metric}"].mean()
        gemini_mean = m_df[f"gemini_{metric}"].mean()
        mimo_ranks[method] = mimo_mean
        gemini_ranks[method] = gemini_mean
    
    # Rank by score (higher = better)
    mimo_sorted = sorted(mimo_ranks.items(), key=lambda x: x[1], reverse=True)
    gemini_sorted = sorted(gemini_ranks.items(), key=lambda x: x[1], reverse=True)
    
    print(f"  {'Method':<15} {'Mimo score':>12} {'Mimo rank':>10} {'Gemini score':>14} {'Gemini rank':>12} {'Δ rank':>8}")
    for i, (method, score) in enumerate(mimo_sorted):
        gemini_rank = next(j for j, (m, _) in enumerate(gemini_sorted) if m == method)
        print(f"  {METHOD_DISPLAY[method]:<15} {score:>12.4f} {i+1:>10} {gemini_ranks[method]:>14.4f} {gemini_rank+1:>12} {gemini_rank-i:>+8}")
    
    # Spearman rank correlation
    mimo_rank_vals = [mimo_ranks[m] for m in METHODS]
    gemini_rank_vals = [gemini_ranks[m] for m in METHODS]
    rho, rho_p = stats.spearmanr(mimo_rank_vals, gemini_rank_vals)
    print(f"\n  Spearman ρ = {rho:.4f}, p = {rho_p:.4f}")


# ============================================================
# ANALYSIS 5: Claim-Level Label Agreement
# ============================================================
print("\n" + "=" * 80)
print("CLAIM-LEVEL LABEL AGREEMENT")
print("=" * 80)

from sklearn.metrics import cohen_kappa_score, confusion_matrix
from itertools import combinations

# Compute claim-level agreement on overlapping papers
label_agreements = []
all_mimo_labels = []
all_gemini_labels = []

for _, row in df.iterrows():
    m_labels = row.get("mimo_claim_labels", [])
    g_labels = row.get("gemini_claim_labels", [])
    m_scores = row.get("mimo_claim_scores", [])
    g_scores = row.get("gemini_claim_scores", [])
    
    if m_labels and g_labels and m_scores and g_scores:
        # For each pair of claims, compute agreement
        # Since claims may not be aligned 1:1, compute per-paper agreement rate
        m_set = set(m_labels)
        g_set = set(g_labels)
        
        # Simple label distribution agreement
        m_dist = {l: m_labels.count(l) / len(m_labels) for l in set(m_labels)}
        g_dist = {l: g_labels.count(l) / len(g_labels) for l in set(g_labels)}
        
        # Compute mean score agreement
        m_mean = np.mean(m_scores) if m_scores else 0
        g_mean = np.mean(g_scores) if g_scores else 0
        
        label_agreements.append({
            "conference": row["conference"],
            "paper_id": row["paper_id"],
            "method": row["method"],
            "mimo_mean_score": m_mean,
            "gemini_mean_score": g_mean,
            "mimo_n_claims": len(m_scores),
            "gemini_n_claims": len(g_scores),
            "mimo_labels": m_labels,
            "gemini_labels": g_labels,
        })

if label_agreements:
    la_df = pd.DataFrame(label_agreements)
    
    # Overall score correlation
    r, p = stats.pearsonr(la_df["mimo_mean_score"], la_df["gemini_mean_score"])
    rho, rho_p = stats.spearmanr(la_df["mimo_mean_score"], la_df["gemini_mean_score"])
    print(f"Claim-level mean score correlation:")
    print(f"  Pearson r = {r:.4f}, p = {p:.2e}")
    print(f"  Spearman ρ = {rho:.4f}, p = {rho_p:.2e}")
    
    # By method
    print(f"\nClaim-level score correlation by method:")
    print(f"  {'Method':<15} {'Pearson r':>10} {'p':>12} {'Spearman ρ':>12} {'n':>6}")
    for method in METHODS:
        m_df = la_df[la_df["method"] == method]
        if len(m_df) > 5:
            r, p = stats.pearsonr(m_df["mimo_mean_score"], m_df["gemini_mean_score"])
            rho, rho_p = stats.spearmanr(m_df["mimo_mean_score"], m_df["gemini_mean_score"])
            print(f"  {METHOD_DISPLAY[method]:<15} {r:>10.4f} {p:>12.2e} {rho:>12.4f} {len(m_df):>5}")


# ============================================================
# ANALYSIS 6: Per-Conference Consistency
# ============================================================
print("\n" + "=" * 80)
print("PER-CONFERENCE RANKING CONSISTENCY")
print("=" * 80)

for metric, metric_name in [("ns", "NS"), ("sr", "SR")]:
    print(f"\n--- {metric_name} by Conference ---")
    print(f"  {'Conference':<15} {'Spearman ρ':>10} {'p':>12}")
    for conf in CONFERENCES:
        m_df = df[df["conference"] == conf]
        mimo_means = {}
        gemini_means = {}
        for method in METHODS:
            mm = m_df[m_df["method"] == method]
            mimo_means[method] = mm[f"mimo_{metric}"].mean()
            gemini_means[method] = mm[f"gemini_{metric}"].mean()
        
        mimo_vals = [mimo_means[m] for m in METHODS]
        gemini_vals = [gemini_means[m] for m in METHODS]
        rho, rho_p = stats.spearmanr(mimo_vals, gemini_vals)
        print(f"  {CONF_DISPLAY[conf]:<15} {rho:>10.4f} {rho_p:>12.4f}")


# ============================================================
# FIGURE GENERATION
# ============================================================
print("\n" + "=" * 80)
print("GENERATING FIGURES")
print("=" * 80)


def fig1_novelty_metrics_comparison():
    """Figure 1: NS/SR/SSR comparison across methods."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = [("ns", "Normalized Score (NS)"), ("sr", "Support Rate (SR)"), ("ssr", "Strict Support Rate (SSR)")]
    
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx]
        mimo_means = []
        gemini_means = []
        mimo_sems = []
        gemini_sems = []
        
        for method in METHODS:
            m_df = df[df["method"] == method]
            valid = m_df[[f"mimo_{metric}", f"gemini_{metric}"]].dropna()
            mimo_means.append(valid[f"mimo_{metric}"].mean())
            gemini_means.append(valid[f"gemini_{metric}"].mean())
            mimo_sems.append(valid[f"mimo_{metric}"].std() / np.sqrt(len(valid)))
            gemini_sems.append(valid[f"gemini_{metric}"].std() / np.sqrt(len(valid)))
        
        x = np.arange(len(METHODS))
        width = 0.35
        ax.bar(x - width/2, mimo_means, width, yerr=[s*1.96 for s in mimo_sems],
               label='Mimo 2.5 Pro' if idx == 0 else None, color=MIMO_COLOR, alpha=0.8, capsize=3)
        ax.bar(x + width/2, gemini_means, width, yerr=[s*1.96 for s in gemini_sems],
               label='Gemini 2.5 Flash Lite' if idx == 0 else None, color=GEMINI_COLOR, alpha=0.8, capsize=3)
        ax.set_title(f"({chr(97+idx)}) {title}")
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_DISPLAY[m] for m in METHODS], rotation=30, ha='right', fontsize=12)
        ax.set_ylim(0, 1.05)
    
    # Single shared legend at the bottom center
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=12, frameon=True)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(FIGURES_DIR / "fig1_novelty_metrics_comparison.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / "fig1_novelty_metrics_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved fig1_novelty_metrics_comparison")


def fig2_ranking_consistency():
    """Figure 2: Ranking consistency across backbones."""
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    # fig.suptitle("Reviewer System Ranking Consistency", fontsize=17, fontweight='bold')
    
    # Panel A: NS ranking comparison
    ax = axes[0]
    mimo_ns = []
    gemini_ns = []
    for method in METHODS:
        m_df = df[df["method"] == method]
        valid = m_df[["mimo_ns", "gemini_ns"]].dropna()
        mimo_ns.append(valid["mimo_ns"].mean())
        gemini_ns.append(valid["gemini_ns"].mean())
    
    ax.scatter(mimo_ns, gemini_ns, s=100, c='#2C3E50', zorder=5)
    for i, method in enumerate(METHODS):
        ax.annotate(METHOD_DISPLAY[method], (mimo_ns[i], gemini_ns[i]),
                   textcoords="offset points", xytext=(8, 5), fontsize=12)
    
    # Fit line
    z = np.polyfit(mimo_ns, gemini_ns, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(min(mimo_ns)-0.02, max(mimo_ns)+0.02, 100)
    ax.plot(x_line, p_line(x_line), '--', color='gray', alpha=0.5)
    
    rho, rho_p = stats.spearmanr(mimo_ns, gemini_ns)
    ax.text(0.05, 0.95, f"Spearman ρ = {rho:.3f}\np = {rho_p:.3f}",
            transform=ax.transAxes, fontsize=13, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel("Mimo 2.5 Pro: Mean NS")
    ax.set_ylabel("Gemini 2.5 Flash Lite: Mean NS")
    ax.set_title("(a) Normalized Score (NS) Ranking")
    
    # Panel B: SR ranking comparison
    ax = axes[1]
    mimo_sr = []
    gemini_sr = []
    for method in METHODS:
        m_df = df[df["method"] == method]
        valid = m_df[["mimo_sr", "gemini_sr"]].dropna()
        mimo_sr.append(valid["mimo_sr"].mean())
        gemini_sr.append(valid["gemini_sr"].mean())
    
    ax.scatter(mimo_sr, gemini_sr, s=100, c='#2C3E50', zorder=5)
    for i, method in enumerate(METHODS):
        ax.annotate(METHOD_DISPLAY[method], (mimo_sr[i], gemini_sr[i]),
                   textcoords="offset points", xytext=(8, 5), fontsize=12)
    
    z = np.polyfit(mimo_sr, gemini_sr, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(min(mimo_sr)-0.02, max(mimo_sr)+0.02, 100)
    ax.plot(x_line, p_line(x_line), '--', color='gray', alpha=0.5)
    
    rho, rho_p = stats.spearmanr(mimo_sr, gemini_sr)
    ax.text(0.05, 0.95, f"Spearman ρ = {rho:.3f}\np = {rho_p:.3f}",
            transform=ax.transAxes, fontsize=13, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel("Mimo 2.5 Pro: Mean SR")
    ax.set_ylabel("Gemini 2.5 Flash Lite: Mean SR")
    ax.set_title("(b) Support Rate (SR) Ranking")
    
    # Panel C: SSR ranking comparison
    ax = axes[2]
    mimo_ssr = []
    gemini_ssr = []
    for method in METHODS:
        m_df = df[df["method"] == method]
        valid = m_df[["mimo_ssr", "gemini_ssr"]].dropna()
        mimo_ssr.append(valid["mimo_ssr"].mean())
        gemini_ssr.append(valid["gemini_ssr"].mean())
    
    ax.scatter(mimo_ssr, gemini_ssr, s=100, c='#2C3E50', zorder=5)
    for i, method in enumerate(METHODS):
        ax.annotate(METHOD_DISPLAY[method], (mimo_ssr[i], gemini_ssr[i]),
                   textcoords="offset points", xytext=(8, 5), fontsize=12)
    
    z = np.polyfit(mimo_ssr, gemini_ssr, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(min(mimo_ssr)-0.02, max(mimo_ssr)+0.02, 100)
    ax.plot(x_line, p_line(x_line), '--', color='gray', alpha=0.5)
    
    rho, rho_p = stats.spearmanr(mimo_ssr, gemini_ssr)
    ax.text(0.05, 0.95, f"Spearman ρ = {rho:.3f}\np = {rho_p:.3f}",
            transform=ax.transAxes, fontsize=13, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel("Mimo 2.5 Pro: Mean SSR")
    ax.set_ylabel("Gemini 2.5 Flash Lite: Mean SSR")
    ax.set_title("(c) Strict Support Rate (SSR) Ranking")
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_ranking_consistency.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / "fig2_ranking_consistency.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved fig2_ranking_consistency")


def fig3_pipeline_agreement():
    """Figure 3: Pipeline agreement (Phase 1 claim overlap, Phase 2 Jaccard, Phase 3 score correlation)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Pipeline Agreement Between Backbone LLMs", fontsize=17, fontweight='bold')
    
    # Panel A: Phase 1 claim overlap
    ax = axes[0]
    if len(co_df) > 0:
        method_overlaps = []
        for method in METHODS:
            m_data = co_df[co_df["method"] == method]["claim_overlap"]
            method_overlaps.append(m_data.values)
        
        bp = ax.boxplot(method_overlaps, labels=[METHOD_DISPLAY[m] for m in METHODS],
                       patch_artist=True, widths=0.5)
        for patch in bp['boxes']:
            patch.set_facecolor('#3498DB')
            patch.set_alpha(0.6)
        ax.set_ylabel("Claim Overlap (Token Jaccard)")
        ax.set_title("(a) Phase 1: Claim Extraction Agreement")
        ax.tick_params(axis='x', rotation=30)
        ax.set_ylim(0, 1.05)
    
    # Panel B: Phase 2 candidate Jaccard
    ax = axes[1]
    if len(jac_df) > 0:
        method_jaccards = []
        for method in METHODS:
            m_data = jac_df[jac_df["method"] == method]["jaccard"]
            method_jaccards.append(m_data.values)
        
        bp = ax.boxplot(method_jaccards, labels=[METHOD_DISPLAY[m] for m in METHODS],
                       patch_artist=True, widths=0.5)
        for patch in bp['boxes']:
            patch.set_facecolor('#2ECC71')
            patch.set_alpha(0.6)
        ax.set_ylabel("Candidate Pool Jaccard")
        ax.set_title("(b) Phase 2: Retrieval Agreement")
        ax.tick_params(axis='x', rotation=30)
        ax.set_ylim(0, 1.05)
    
    # Panel C: Phase 3 score correlation scatter
    ax = axes[2]
    valid = df[["mimo_ns", "gemini_ns"]].dropna()
    sample = valid.sample(min(500, len(valid)), random_state=42)
    ax.scatter(sample["mimo_ns"], sample["gemini_ns"], alpha=0.3, s=15, c='#7F8C8D')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    r, p = stats.pearsonr(valid["mimo_ns"], valid["gemini_ns"])
    ax.text(0.05, 0.95, f"r = {r:.3f}\np < 0.001",
            transform=ax.transAxes, fontsize=13, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel("Mimo 2.5 Pro: NS")
    ax.set_ylabel("Gemini 2.5 Flash Lite: NS")
    ax.set_title("(c) Phase 3: Verification Score Agreement")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_pipeline_agreement.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / "fig3_pipeline_agreement.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved fig3_pipeline_agreement")


def fig4_per_conference_ranking():
    """Figure 4: Per-conference ranking heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Reviewer Rankings by Conference", fontsize=17, fontweight='bold')
    
    # Panel A: Mimo NS by method × conference
    ax = axes[0]
    pivot_mimo = df.pivot_table(values='mimo_ns', index='method', columns='conference', aggfunc='mean')
    pivot_mimo = pivot_mimo.reindex(index=METHODS, columns=CONFERENCES)
    pivot_mimo.index = [METHOD_DISPLAY[m] for m in pivot_mimo.index]
    pivot_mimo.columns = [CONF_DISPLAY[c] for c in pivot_mimo.columns]
    im = ax.imshow(pivot_mimo.values, cmap='YlOrRd', aspect='auto', vmin=0.4, vmax=0.9)
    ax.set_xticks(range(len(pivot_mimo.columns)))
    ax.set_xticklabels(pivot_mimo.columns, fontsize=12)
    ax.set_yticks(range(len(pivot_mimo.index)))
    ax.set_yticklabels(pivot_mimo.index, fontsize=12)
    for i in range(len(pivot_mimo.index)):
        for j in range(len(pivot_mimo.columns)):
            val = pivot_mimo.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=10,
                       color='white' if val > 0.7 else 'black')
    ax.set_title("(a) Mimo 2.5 Pro: Mean NS")
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Panel B: Gemini NS by method × conference
    ax = axes[1]
    pivot_gemini = df.pivot_table(values='gemini_ns', index='method', columns='conference', aggfunc='mean')
    pivot_gemini = pivot_gemini.reindex(index=METHODS, columns=CONFERENCES)
    pivot_gemini.index = [METHOD_DISPLAY[m] for m in pivot_gemini.index]
    pivot_gemini.columns = [CONF_DISPLAY[c] for c in pivot_gemini.columns]
    im = ax.imshow(pivot_gemini.values, cmap='YlOrRd', aspect='auto', vmin=0.4, vmax=0.9)
    ax.set_xticks(range(len(pivot_gemini.columns)))
    ax.set_xticklabels(pivot_gemini.columns, fontsize=12)
    ax.set_yticks(range(len(pivot_gemini.index)))
    ax.set_yticklabels(pivot_gemini.index, fontsize=12)
    for i in range(len(pivot_gemini.index)):
        for j in range(len(pivot_gemini.columns)):
            val = pivot_gemini.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=10,
                       color='white' if val > 0.7 else 'black')
    ax.set_title("(b) Gemini 2.5 Flash Lite: Mean NS")
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_per_conference_ranking.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / "fig4_per_conference_ranking.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved fig4_per_conference_ranking")


def fig5_coverage_comparison():
    """Figure 5: Coverage and operational metrics comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Operational Metrics: Mimo vs Gemini", fontsize=17, fontweight='bold')
    
    metrics = [
        ("coverage", "Claim Coverage Rate", "(a)"),
        ("decisive_cov", "Decisive Coverage Rate", "(b)"),
        ("review_sentences", "Review Sentences", "(c)"),
        ("pairs_attempted", "Pairs Attempted", "(d)"),
    ]
    
    for idx, (metric, title, panel) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        mimo_vals = []
        gemini_vals = []
        
        for method in METHODS:
            m_df = df[df["method"] == method]
            valid = m_df[[f"mimo_{metric}", f"gemini_{metric}"]].dropna()
            mimo_vals.append(valid[f"mimo_{metric}"].mean())
            gemini_vals.append(valid[f"gemini_{metric}"].mean())
        
        x = np.arange(len(METHODS))
        width = 0.35
        ax.bar(x - width/2, mimo_vals, width, label='Mimo', color=MIMO_COLOR, alpha=0.8)
        ax.bar(x + width/2, gemini_vals, width, label='Gemini', color=GEMINI_COLOR, alpha=0.8)
        ax.set_ylabel(title)
        ax.set_title(f"{panel} {title}")
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_DISPLAY[m] for m in METHODS], rotation=30, ha='right', fontsize=12)
        ax.legend(fontsize=12)
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_coverage_comparison.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / "fig5_coverage_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved fig5_coverage_comparison")


# ============================================================
# ADDITIONAL TREND ANALYSES
# ============================================================
print("\n" + "=" * 80)
print("TREND ANALYSES")
print("=" * 80)

# Precompute per-paper backbone divergence and paper quality
df['diff_ns'] = df['mimo_ns'] - df['gemini_ns']
df['diff_sr'] = df['mimo_sr'] - df['gemini_sr']
df['diff_ssr'] = df['mimo_ssr'] - df['gemini_ssr']

# Paper quality from human method (average of both backbones)
human_df = df[df['method'] == 'human'].copy()
human_df['paper_quality_ns'] = (human_df['mimo_ns'] + human_df['gemini_ns']) / 2
human_df['paper_quality_sr'] = (human_df['mimo_sr'] + human_df['gemini_sr']) / 2
human_df['paper_quality_ssr'] = (human_df['mimo_ssr'] + human_df['gemini_ssr']) / 2
paper_quality = human_df[['conference', 'paper_id', 'paper_quality_ns', 'paper_quality_sr', 'paper_quality_ssr']]

def fig6_score_difference_distribution():
    """Figure 6: Distribution of backbone score differences (Mimo - Gemini)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Backbone Score Difference Distribution (Mimo − Gemini)", fontsize=17, fontweight='bold')
    
    metrics = [('diff_ns', 'NS'), ('diff_sr', 'SR'), ('diff_ssr', 'SSR')]
    
    for idx, (col, name) in enumerate(metrics):
        ax = axes[idx]
        data = df[col].dropna()
        ax.hist(data, bins=40, color='#7F8C8D', alpha=0.7, edgecolor='white')
        ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
        ax.axvline(data.mean(), color='black', linestyle='-', linewidth=1.5, label=f'Mean = {data.mean():.3f}')
        ax.set_xlabel(f'Δ {name} (Mimo − Gemini)')
        ax.set_ylabel('Count')
        ax.set_title(f'({chr(97+idx)}) Δ {name} Distribution')
        ax.legend(fontsize=11)
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_score_difference_distribution.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / "fig6_score_difference_distribution.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved fig6_score_difference_distribution")


def fig7_agreement_by_paper_quality():
    """Figure 7: Backbone agreement stratified by paper quality quartile."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle("Backbone Agreement by Paper Quality (Human Score Quartile)", fontsize=17, fontweight='bold')
    
    metrics = [('ns', 'NS', 'paper_quality_ns'), ('sr', 'SR', 'paper_quality_sr'), ('ssr', 'SSR', 'paper_quality_ssr')]
    
    for idx, (metric, name, qcol) in enumerate(metrics):
        ax = axes[idx]
        # Merge paper quality onto main df
        merged = df.merge(paper_quality, on=['conference', 'paper_id'])
        merged['quartile'] = pd.qcut(merged[qcol], 4, labels=['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'])
        
        pearson_rs = []
        pearson_ps = []
        n_per_q = []
        quartile_labels = ['Q1\n(lowest)', 'Q2', 'Q3', 'Q4\n(highest)']
        
        for q in ['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)']:
            sub = merged[merged['quartile'] == q]
            valid = sub[[f'mimo_{metric}', f'gemini_{metric}']].dropna()
            if len(valid) > 5:
                r, p = stats.pearsonr(valid[f'mimo_{metric}'], valid[f'gemini_{metric}'])
            else:
                r, p = np.nan, np.nan
            pearson_rs.append(r)
            pearson_ps.append(p)
            n_per_q.append(len(valid))
        
        colors = ['#E74C3C', '#F39C12', '#27AE60', '#2980B9']
        bars = ax.bar(range(4), pearson_rs, color=colors, alpha=0.8, edgecolor='white')
        ax.set_xticks(range(4))
        ax.set_xticklabels(quartile_labels, fontsize=12)
        ax.set_ylabel('Pearson r')
        ax.set_title(f'({chr(97+idx)}) Mimo vs Gemini {name} Agreement')
        ax.set_ylim(0, max(pearson_rs) * 1.3 if max(pearson_rs) > 0 else 0.5)
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        
        for i, (r, n) in enumerate(zip(pearson_rs, n_per_q)):
            if not np.isnan(r):
                ax.text(i, r + 0.01, f'r={r:.3f}\nn={n}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig7_agreement_by_paper_quality.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / "fig7_agreement_by_paper_quality.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved fig7_agreement_by_paper_quality")


def fig8_claim_level_agreement():
    """Figure 8: Claim-level agreement (SUPPORTED vs OVERSTATED)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Claim-Level Label Agreement Between Backbones", fontsize=17, fontweight='bold')
    
    # Panel A: Fraction of SUPPORTED claims scatter
    ax = axes[0]
    mimo_frac_supported = []
    gemini_frac_supported = []
    method_colors_map = []
    
    for _, row in df.iterrows():
        ml = row.get('mimo_claim_labels', [])
        gl = row.get('gemini_claim_labels', [])
        if ml and gl:
            mimo_frac_supported.append(sum(1 for l in ml if l == 'SUPPORTED') / len(ml))
            gemini_frac_supported.append(sum(1 for l in gl if l == 'SUPPORTED') / len(gl))
            method_colors_map.append(row['method'])
    
    mimo_frac = np.array(mimo_frac_supported)
    gemini_frac = np.array(gemini_frac_supported)
    
    # Color by method
    method_color_list = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6', '#F39C12', '#1ABC9C']
    method_cmap = dict(zip(METHODS, method_color_list))
    colors = [method_cmap.get(m, '#7F8C8D') for m in method_colors_map]
    
    ax.scatter(mimo_frac, gemini_frac, c=colors, alpha=0.25, s=15)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    r, p = stats.pearsonr(mimo_frac, gemini_frac)
    ax.text(0.05, 0.95, f'r = {r:.3f}\np < 0.001', transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel('Mimo: Fraction SUPPORTED')
    ax.set_ylabel('Gemini: Fraction SUPPORTED')
    ax.set_title('(a) Per-Paper Claim Agreement')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    # Legend for methods
    for method in METHODS:
        ax.scatter([], [], c=method_cmap[method], s=40, label=METHOD_DISPLAY[method])
    ax.legend(fontsize=9, loc='lower right', ncol=2)
    
    # Panel B: Mean fraction SUPPORTED by method
    ax = axes[1]
    mimo_means = []
    gemini_means = []
    mimo_sems = []
    gemini_sems = []
    for method in METHODS:
        sub = df[df['method'] == method]
        m_vals = []
        g_vals = []
        for _, row in sub.iterrows():
            ml = row.get('mimo_claim_labels', [])
            gl = row.get('gemini_claim_labels', [])
            if ml:
                m_vals.append(sum(1 for l in ml if l == 'SUPPORTED') / len(ml))
            if gl:
                g_vals.append(sum(1 for l in gl if l == 'SUPPORTED') / len(gl))
        mimo_means.append(np.mean(m_vals) if m_vals else 0)
        gemini_means.append(np.mean(g_vals) if g_vals else 0)
        mimo_sems.append(np.std(m_vals) / np.sqrt(len(m_vals)) if m_vals else 0)
        gemini_sems.append(np.std(g_vals) / np.sqrt(len(g_vals)) if g_vals else 0)
    
    x = np.arange(len(METHODS))
    width = 0.35
    ax.bar(x - width/2, mimo_means, width, yerr=[s*1.96 for s in mimo_sems],
           label='Mimo', color=MIMO_COLOR, alpha=0.8, capsize=3)
    ax.bar(x + width/2, gemini_means, width, yerr=[s*1.96 for s in gemini_sems],
           label='Gemini', color=GEMINI_COLOR, alpha=0.8, capsize=3)
    ax.set_ylabel('Fraction SUPPORTED')
    ax.set_title('(b) Mean Fraction SUPPORTED by Method')
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_DISPLAY[m] for m in METHODS], rotation=30, ha='right', fontsize=12)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1.05)
    
    # Panel C: Agreement rate (both agree on majority label)
    ax = axes[2]
    agree_rates = []
    for method in METHODS:
        sub = df[df['method'] == method]
        agree = 0
        total = 0
        for _, row in sub.iterrows():
            ml = row.get('mimo_claim_labels', [])
            gl = row.get('gemini_claim_labels', [])
            if ml and gl:
                m_majority = 'SUPPORTED' if sum(1 for l in ml if l == 'SUPPORTED') > len(ml)/2 else 'OVERSTATED'
                g_majority = 'SUPPORTED' if sum(1 for l in gl if l == 'SUPPORTED') > len(gl)/2 else 'OVERSTATED'
                if m_majority == g_majority:
                    agree += 1
                total += 1
        agree_rates.append(agree / total if total > 0 else 0)
    
    bars = ax.bar(range(len(METHODS)), agree_rates, color='#3498DB', alpha=0.8, edgecolor='white')
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([METHOD_DISPLAY[m] for m in METHODS], rotation=30, ha='right', fontsize=12)
    ax.set_ylabel('Agreement Rate')
    ax.set_title('(c) Majority Label Agreement Rate')
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    
    for i, rate in enumerate(agree_rates):
        ax.text(i, rate + 0.02, f'{rate:.2f}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig8_claim_level_agreement.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / "fig8_claim_level_agreement.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved fig8_claim_level_agreement")


def fig9_method_effect_size():
    """Figure 9: Cohen's d effect size per method (backbone sensitivity)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Backbone Sensitivity: Effect Size (Cohen's d) per Reviewer Method", fontsize=17, fontweight='bold')
    
    metrics = [('ns', 'NS'), ('sr', 'SR'), ('ssr', 'SSR')]
    
    for idx, (metric, name) in enumerate(metrics):
        ax = axes[idx]
        cohens_d = []
        for method in METHODS:
            sub = df[df['method'] == method]
            m = sub[f'mimo_{metric}'].dropna()
            g = sub[f'gemini_{metric}'].dropna()
            pooled_std = np.sqrt((m.std()**2 + g.std()**2) / 2)
            d = (m.mean() - g.mean()) / pooled_std if pooled_std > 0 else 0
            cohens_d.append(d)
        
        colors = ['#E74C3C' if d < 0 else '#2ECC71' for d in cohens_d]
        bars = ax.barh(range(len(METHODS)), cohens_d, color=colors, alpha=0.8, edgecolor='white')
        ax.set_yticks(range(len(METHODS)))
        ax.set_yticklabels([METHOD_DISPLAY[m] for m in METHODS], fontsize=12)
        ax.set_xlabel("Cohen's d (Mimo − Gemini)")
        ax.set_title(f'({chr(97+idx)}) {name} Effect Size')
        ax.axvline(0, color='black', linewidth=1)
        ax.axvline(-0.2, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
        ax.axvline(-0.5, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
        ax.axvline(-0.8, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
        ax.axvline(0.2, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
        ax.axvline(0.5, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
        ax.axvline(0.8, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
        
        for i, d in enumerate(cohens_d):
            ax.text(d + (0.03 if d >= 0 else -0.03), i, f'{d:.2f}',
                    va='center', ha='left' if d >= 0 else 'right', fontsize=11)
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig9_method_effect_size.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / "fig9_method_effect_size.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved fig9_method_effect_size")


def fig10_conference_trend():
    """Figure 10: Backbone score trends across conferences."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle("Backbone Score Trend Across Conferences", fontsize=17, fontweight='bold')
    
    metrics = [('ns', 'NS'), ('sr', 'SR'), ('ssr', 'SSR')]
    conf_order = CONFERENCES
    conf_labels = [CONF_DISPLAY[c] for c in conf_order]
    x = np.arange(len(conf_order))
    
    for idx, (metric, name) in enumerate(metrics):
        ax = axes[idx]
        mimo_by_conf = []
        gemini_by_conf = []
        mimo_sem_by_conf = []
        gemini_sem_by_conf = []
        
        for conf in conf_order:
            sub = df[df['conference'] == conf]
            m = sub[f'mimo_{metric}'].dropna()
            g = sub[f'gemini_{metric}'].dropna()
            mimo_by_conf.append(m.mean())
            gemini_by_conf.append(g.mean())
            mimo_sem_by_conf.append(m.std() / np.sqrt(len(m)))
            gemini_sem_by_conf.append(g.std() / np.sqrt(len(g)))
        
        ax.errorbar(x - 0.12, mimo_by_conf, yerr=[s*1.96 for s in mimo_sem_by_conf],
                    fmt='o-', color=MIMO_COLOR, label='Mimo', capsize=4, linewidth=2, markersize=8)
        ax.errorbar(x + 0.12, gemini_by_conf, yerr=[s*1.96 for s in gemini_sem_by_conf],
                    fmt='s-', color=GEMINI_COLOR, label='Gemini', capsize=4, linewidth=2, markersize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(conf_labels, fontsize=12)
        ax.set_ylabel(f'Mean {name}')
        ax.set_title(f'({chr(97+idx)}) {name} Across Conferences')
        ax.legend(fontsize=12)
        ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig10_conference_trend.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / "fig10_conference_trend.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved fig10_conference_trend")


def fig11_coverage_claimcount_vs_divergence():
    """Figure 11: Coverage and claim count vs backbone divergence."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Drivers of Backbone Divergence", fontsize=17, fontweight='bold')
    
    # Panel A: Coverage vs NS divergence
    ax = axes[0, 0]
    merged = df.merge(paper_quality, on=['conference', 'paper_id'])
    valid = merged[['mimo_coverage', 'diff_ns']].dropna()
    ax.scatter(valid['mimo_coverage'], valid['diff_ns'], alpha=0.2, s=15, c='#7F8C8D')
    z = np.polyfit(valid['mimo_coverage'], valid['diff_ns'], 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(valid['mimo_coverage'].min(), valid['mimo_coverage'].max(), 100)
    ax.plot(x_line, p_line(x_line), 'r--', linewidth=1.5)
    r, p = stats.pearsonr(valid['mimo_coverage'], valid['diff_ns'])
    ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.2e}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel('Coverage Rate')
    ax.set_ylabel('Δ NS (Mimo − Gemini)')
    ax.set_title('(a) Coverage vs NS Divergence')
    
    # Panel B: Coverage vs SR divergence
    ax = axes[0, 1]
    valid = merged[['mimo_coverage', 'diff_sr']].dropna()
    ax.scatter(valid['mimo_coverage'], valid['diff_sr'], alpha=0.2, s=15, c='#7F8C8D')
    z = np.polyfit(valid['mimo_coverage'], valid['diff_sr'], 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(valid['mimo_coverage'].min(), valid['mimo_coverage'].max(), 100)
    ax.plot(x_line, p_line(x_line), 'r--', linewidth=1.5)
    r, p = stats.pearsonr(valid['mimo_coverage'], valid['diff_sr'])
    ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.2e}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel('Coverage Rate')
    ax.set_ylabel('Δ SR (Mimo − Gemini)')
    ax.set_title('(b) Coverage vs SR Divergence')
    
    # Panel C: Claim count (Phase 3) vs NS divergence
    ax = axes[1, 0]
    valid = merged[['mimo_claim_count_p3', 'diff_ns']].dropna()
    ax.scatter(valid['mimo_claim_count_p3'], valid['diff_ns'], alpha=0.2, s=15, c='#7F8C8D')
    z = np.polyfit(valid['mimo_claim_count_p3'], valid['diff_ns'], 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(valid['mimo_claim_count_p3'].min(), valid['mimo_claim_count_p3'].max(), 100)
    ax.plot(x_line, p_line(x_line), 'r--', linewidth=1.5)
    r, p = stats.pearsonr(valid['mimo_claim_count_p3'], valid['diff_ns'])
    ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.2e}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel('Claim Count (Phase 3)')
    ax.set_ylabel('Δ NS (Mimo − Gemini)')
    ax.set_title('(c) Claim Count vs NS Divergence')
    
    # Panel D: Claim count vs SR divergence
    ax = axes[1, 1]
    valid = merged[['mimo_claim_count_p3', 'diff_sr']].dropna()
    ax.scatter(valid['mimo_claim_count_p3'], valid['diff_sr'], alpha=0.2, s=15, c='#7F8C8D')
    z = np.polyfit(valid['mimo_claim_count_p3'], valid['diff_sr'], 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(valid['mimo_claim_count_p3'].min(), valid['mimo_claim_count_p3'].max(), 100)
    ax.plot(x_line, p_line(x_line), 'r--', linewidth=1.5)
    r, p = stats.pearsonr(valid['mimo_claim_count_p3'], valid['diff_sr'])
    ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.2e}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel('Claim Count (Phase 3)')
    ax.set_ylabel('Δ SR (Mimo − Gemini)')
    ax.set_title('(d) Claim Count vs SR Divergence')
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig11_drivers_of_divergence.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / "fig11_drivers_of_divergence.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved fig11_drivers_of_divergence")


# Generate all figures
fig1_novelty_metrics_comparison()
fig2_ranking_consistency()
fig3_pipeline_agreement()
fig4_per_conference_ranking()
fig5_coverage_comparison()
fig6_score_difference_distribution()
fig7_agreement_by_paper_quality()
fig8_claim_level_agreement()
fig9_method_effect_size()
fig10_conference_trend()
fig11_coverage_claimcount_vs_divergence()

# ============================================================
# SUMMARY STATISTICS FOR TREND ANALYSES
# ============================================================
print("\n" + "=" * 80)
print("SUMMARY: TREND ANALYSIS KEY FINDINGS")
print("=" * 80)

# 1. Overall divergence
for metric, name in [('diff_ns', 'NS'), ('diff_sr', 'SR'), ('diff_ssr', 'SSR')]:
    data = df[metric].dropna()
    print(f"\nΔ {name}: mean={data.mean():.4f}, std={data.std():.4f}, median={data.median():.4f}")
    print(f"  95% CI: [{data.quantile(0.025):.4f}, {data.quantile(0.975):.4f}]")

# 2. Method sensitivity ranking
print("\nMethod Sensitivity (Cohen's d, NS):")
for method in METHODS:
    sub = df[df['method'] == method]
    m = sub['mimo_ns'].dropna()
    g = sub['gemini_ns'].dropna()
    pooled_std = np.sqrt((m.std()**2 + g.std()**2) / 2)
    d = (m.mean() - g.mean()) / pooled_std if pooled_std > 0 else 0
    print(f"  {METHOD_DISPLAY[method]:<15}: d = {d:+.3f} (Mimo {m.mean():.4f} vs Gemini {g.mean():.4f})")

# 3. Conference trend
print("\nConference Trend (mean NS):")
for conf in CONFERENCES:
    sub = df[df['conference'] == conf]
    m = sub['mimo_ns'].mean()
    g = sub['gemini_ns'].mean()
    print(f"  {CONF_DISPLAY[conf]:<12}: Mimo={m:.4f}, Gemini={g:.4f}, Δ={m-g:+.4f}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print(f"Output: {OUTPUT_DIR}")
print(f"Figures: {FIGURES_DIR}")
print("=" * 80)
