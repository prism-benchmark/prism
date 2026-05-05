"""
compare_human_llm.py — So sánh metrics giữa Human và các LLM

So sánh 3 chỉ số:
  1. R_Premise: Tỷ lệ arguments là Premise
  2. S_Depth: Chất lượng bằng chứng (grounding score)
  3. Combined: Harmonic mean của 2 chỉ số trên

Cách chạy:
    python pipeline/compare_human_llm.py --llm sea
    python pipeline/compare_human_llm.py --llm tree_iclr2024
    python pipeline/compare_human_llm.py --llm reviewer2_iclr2024
    python pipeline/compare_human_llm.py --all

Output: pipeline/output/comparison_*.json
"""

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipeline.config as config


def load_metrics(source: str):
    """Load metrics file cho 1 source."""
    metrics_file = os.path.join(
        config.PROJECT_ROOT, "pipeline", "output", f"{source}_metrics.json"
    )
    
    if not os.path.exists(metrics_file):
        print(f"⚠️  Chưa tính metrics cho {source}")
        print(f"   Chạy: python pipeline/calculate_metrics.py --source {source}")
        return None
    
    with open(metrics_file, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_similarity(human_val: float, llm_val: float) -> float:
    """Tính độ tương đồng [0, 1]."""
    difference = abs(human_val - llm_val)
    return max(0.0, 1.0 - difference)


def compare_source(human_metrics: dict, llm_name: str):
    """So sánh 1 LLM với Human."""
    llm_metrics = load_metrics(llm_name)
    if not llm_metrics:
        return None
    
    # Lấy average của Human_1, Human_2, Human_3
    human_avg = {}
    human_count = 0
    for reviewer_id, metrics in human_metrics["metrics_by_reviewer"].items():
        if reviewer_id.startswith("Human_"):
            for key in ["avg_r_premise", "avg_s_depth", "avg_combined"]:
                if key not in human_avg:
                    human_avg[key] = 0
                human_avg[key] += metrics[key]
            human_count += 1
    
    for key in human_avg:
        human_avg[key] /= human_count if human_count > 0 else 1
    
    # Lấy average của LLM
    llm_avg = {}
    llm_count = 0
    for reviewer_id, metrics in llm_metrics["metrics_by_reviewer"].items():
        for key in ["avg_r_premise", "avg_s_depth", "avg_combined"]:
            if key not in llm_avg:
                llm_avg[key] = 0
            llm_avg[key] += metrics[key]
        llm_count += 1
    
    for key in llm_avg:
        llm_avg[key] /= llm_count if llm_count > 0 else 1
    
    # Tính comparison
    comparison = {
        "llm_name": llm_name,
        "metrics": {}
    }
    
    for metric_key in ["avg_r_premise", "avg_s_depth", "avg_combined"]:
        label = metric_key.replace("avg_", "").upper()
        human_val = round(human_avg[metric_key], 4)
        llm_val = round(llm_avg[metric_key], 4)
        similarity = round(calculate_similarity(human_val, llm_val), 4)
        diff = round(llm_val - human_val, 4)
        
        if diff > 0:
            trend = f"📈 LLM cao hơn Human {abs(diff):.1%}"
        elif diff < 0:
            trend = f"📉 LLM thấp hơn Human {abs(diff):.1%}"
        else:
            trend = "➡️  LLM = Human"
        
        comparison["metrics"][label] = {
            "human_avg": human_val,
            "llm_avg": llm_val,
            "similarity": similarity,
            "difference": diff,
            "trend": trend
        }
    
    return comparison


def run_comparison(llm_name: str = None, compare_all: bool = False):
    """Chạy comparison."""
    
    # Load human metrics
    human_metrics = load_metrics("human")
    if not human_metrics:
        print("❌ Chưa có human metrics")
        sys.exit(1)
    
    results = []
    
    if compare_all:
        llm_names = list(config.LLM_SOURCES.keys())
    else:
        llm_names = [llm_name]
    
    print(f"\n📊 So Sánh Human vs LLM")
    print("=" * 80)
    
    for llm in llm_names:
        print(f"\n🔍 Đang so sánh: {llm}")
        comparison = compare_source(human_metrics, llm)
        
        if not comparison:
            continue
        
        results.append(comparison)
        
        # Display
        print(f"\n📈 Kết Quả So Sánh: {llm}")
        print("-" * 80)
        for metric, data in comparison["metrics"].items():
            print(f"\n{metric}:")
            print(f"  Human:      {data['human_avg']:.4f}")
            print(f"  {llm:20} {data['llm_avg']:.4f}")
            print(f"  Similarity: {data['similarity']:.4f} (1.0 = giống hệt)")
            print(f"  Chênh lệch: {data['difference']:+.4f}")
            print(f"  {data['trend']}")
    
    # Save all results
    if results:
        output_file = os.path.join(
            config.PROJECT_ROOT, "pipeline", "output", 
            f"comparison_human_vs_llm_all.json"
        )
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Lưu tại: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="So sánh Human vs LLM metrics"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--llm", type=str,
        choices=list(config.LLM_SOURCES.keys()),
        help="Tên LLM cần so sánh"
    )
    group.add_argument(
        "--all", action="store_true",
        help="So sánh tất cả LLM"
    )
    
    args = parser.parse_args()
    run_comparison(llm_name=args.llm, compare_all=args.all)




