"""
calculate_metrics.py — Tính R_Premise, S_Depth và Combined Score

Metrics:
  - R_Premise: Tỷ lệ Premise arguments trong review
  - S_Depth: Trung bình grounding score của Premise arguments
  - Combined: Harmonic mean của R_Premise và S_Depth

Cách chạy:
    python pipeline/calculate_metrics.py --source human
    python pipeline/calculate_metrics.py --source sea
    python pipeline/calculate_metrics.py --source tree_iclr2024
    python pipeline/calculate_metrics.py --source reviewer2_iclr2024
    python pipeline/calculate_metrics.py --source deepreview_iclr2024

Output: pipeline/output/{source}_metrics.json
"""

import sys
import os
import json
import argparse
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipeline.config as config


def calculate_r_premise(arguments: list) -> float:
    """
    Tính R_Premise: Tỷ lệ Premise arguments.
    
    R_Premise = (Số Premise) / (Tổng số Arguments)
    
    Range: [0, 1]
      - 1.0: Tất cả arguments là Premise
      - 0.5: 50% arguments là Premise
      - 0.0: Không có Premise
    """
    if not arguments:
        return 0.0
    
    premise_count = sum(1 for arg in arguments if arg.get("role") == "Premise")
    return premise_count / len(arguments)


def calculate_s_depth(arguments: list) -> float:
    """
    Tính S_Depth: Trung bình grounding score của Premise arguments.
    
    S_Depth = Trung bình(grounding_score của các Premise)
    
    Range: [0, 1]
      - 1.0: Tất cả Premise có grounding score = 1.0
      - 0.5: Trung bình grounding score = 0.5
      - 0.0: Không có Premise hoặc tất cả = 0
    """
    premises = [
        arg.get("grounding_score")
        for arg in arguments
        if arg.get("role") == "Premise" and arg.get("grounding_score") is not None
    ]
    
    if not premises:
        return 0.0
    
    return statistics.mean(premises)


def calculate_combined_metric(r_premise: float, s_depth: float) -> float:
    """
    Tính Combined Score: Harmonic Mean của R_Premise và S_Depth.
    
    Combined = 2 * (R_Premise * S_Depth) / (R_Premise + S_Depth)
    
    Ý nghĩa:
      - Cân bằng giữa lượng premise (R_Premise) và chất lượng premise (S_Depth)
      - Nếu một trong hai = 0 → Combined = 0
      - Nếu cả hai bằng nhau → Combined = giá trị đó
    
    Range: [0, 1]
    """
    if r_premise == 0 or s_depth == 0:
        return 0.0
    
    combined = 2 * (r_premise * s_depth) / (r_premise + s_depth)
    return combined


def calculate_similarity(human_metric: float, llm_metric: float) -> float:
    """
    Tính độ tương đồng giữa Human và LLM.
    
    Similarity = 1 - |human_metric - llm_metric|
    
    Range: [0, 1]
      - 1.0: Giống hệt nhau
      - 0.5: Chênh lệch 50%
      - 0.0: Khác nhau hoàn toàn
    """
    difference = abs(human_metric - llm_metric)
    return max(0.0, 1.0 - difference)


def process_single_result(result_path: str) -> dict:
    """
    Xử lý 1 file kết quả (từ Phase 1 hoặc Phase 2).
    
    Returns:
    {
        "paper_id": "...",
        "reviewers": {
            "Human_1": {
                "r_premise": 0.6,
                "s_depth": 0.75,
                "combined": 0.67
            },
            "SEA": { ... }
        }
    }
    """
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    paper_id = data["paper_id"]
    reviews_analysis = data.get("reviews_analysis", {})
    
    metrics = {}
    for reviewer_id, arguments in reviews_analysis.items():
        if not arguments:
            continue
        
        r_premise = calculate_r_premise(arguments)
        s_depth = calculate_s_depth(arguments)
        combined = calculate_combined_metric(r_premise, s_depth)
        
        metrics[reviewer_id] = {
            "r_premise": round(r_premise, 4),
            "s_depth": round(s_depth, 4),
            "combined": round(combined, 4),
            "num_arguments": len(arguments),
            "num_premises": sum(1 for a in arguments if a.get("role") == "Premise")
        }
    
    return {
        "paper_id": paper_id,
        "reviewers": metrics
    }


def calculate_average_metrics(all_results: list) -> dict:
    """
    Tính trung bình metrics cho tất cả papers.
    
    Returns:
    {
        "Human_1": {"avg_r_premise": 0.6, "avg_s_depth": 0.7, "avg_combined": 0.65},
        "SEA": { ... }
    }
    """
    reviewer_metrics = {}
    
    for result in all_results:
        for reviewer_id, metrics in result.get("reviewers", {}).items():
            if reviewer_id not in reviewer_metrics:
                reviewer_metrics[reviewer_id] = {
                    "r_premise": [],
                    "s_depth": [],
                    "combined": [],
                    "num_arguments": [],
                    "num_premises": []
                }
            
            reviewer_metrics[reviewer_id]["r_premise"].append(metrics["r_premise"])
            reviewer_metrics[reviewer_id]["s_depth"].append(metrics["s_depth"])
            reviewer_metrics[reviewer_id]["combined"].append(metrics["combined"])
            reviewer_metrics[reviewer_id]["num_arguments"].append(metrics["num_arguments"])
            reviewer_metrics[reviewer_id]["num_premises"].append(metrics["num_premises"])
    
    # Tính trung bình
    averages = {}
    for reviewer_id, metrics in reviewer_metrics.items():
        averages[reviewer_id] = {
            "avg_r_premise": round(statistics.mean(metrics["r_premise"]), 4),
            "avg_s_depth": round(statistics.mean(metrics["s_depth"]), 4),
            "avg_combined": round(statistics.mean(metrics["combined"]), 4),
            "avg_num_arguments": round(statistics.mean(metrics["num_arguments"]), 2),
            "avg_num_premises": round(statistics.mean(metrics["num_premises"]), 2),
            "papers_processed": len(metrics["r_premise"])
        }
    
    return averages


def compare_human_vs_llm(human_metrics: dict, llm_name: str, llm_metrics: dict) -> dict:
    """
    So sánh Human vs 1 LLM source.
    
    Returns:
    {
        "combined": {
            "human_avg": 0.65,
            "llm_avg": 0.58,
            "similarity": 0.93,
            "diff": -0.07
        },
        ...
    }
    """
    comparison = {}
    
    for metric_key in ["r_premise", "s_depth", "combined"]:
        human_key = f"avg_{metric_key}"
        llm_key = f"avg_{metric_key}"
        
        human_val = next(
            (v[human_key] for k, v in human_metrics.items() if k.startswith("Human_")),
            0.0
        )
        llm_val = llm_metrics.get(llm_name, {}).get(llm_key, 0.0)
        
        similarity = calculate_similarity(human_val, llm_val)
        diff = llm_val - human_val
        
        comparison[metric_key] = {
            "human_avg": round(human_val, 4),
            "llm_avg": round(llm_val, 4),
            "similarity": round(similarity, 4),
            "diff": round(diff, 4),
            "interpretation": (
                f"LLM cao hơn Human {abs(diff):.1%}" if diff > 0
                else f"LLM thấp hơn Human {abs(diff):.1%}" if diff < 0
                else "LLM = Human"
            )
        }
    
    return comparison


def run_calculate_metrics(source_name: str):
    """Main pipeline để tính metrics."""
    
    # Determine source type
    if source_name == "human":
        source_dir = config.OUTPUT_HUMAN_DIR
        llm_name = None
    else:
        source_dir = config.get_llm_output_dir(source_name)
        llm_name = source_name.upper()
    
    if not os.path.isdir(source_dir):
        print(f"❌ Thư mục không tồn tại: {source_dir}")
        sys.exit(1)
    
    # Load all results
    json_files = sorted([f for f in os.listdir(source_dir) if f.endswith(".json")])
    if not json_files:
        print(f"❌ Không có file JSON trong {source_dir}")
        sys.exit(1)
    
    print(f"\n📊 Đang tính metrics cho: {source_name}")
    print(f"📁 Source: {source_dir}")
    print(f"📄 Tổng papers: {len(json_files)}")
    print("=" * 60)
    
    all_results = []
    for fname in json_files:
        fpath = os.path.join(source_dir, fname)
        try:
            result = process_single_result(fpath)
            all_results.append(result)
        except Exception as e:
            print(f"⚠️  Lỗi xử lý {fname}: {e}")
            continue
    
    if not all_results:
        print("❌ Không có dữ liệu hợp lệ")
        sys.exit(1)
    
    # Calculate averages
    print(f"\n✅ Xử lý thành công: {len(all_results)}/{len(json_files)}")
    averages = calculate_average_metrics(all_results)
    
    # Save results
    output_dir = os.path.join(config.PROJECT_ROOT, "pipeline", "output")
    os.makedirs(output_dir, exist_ok=True)
    
    result_file = os.path.join(output_dir, f"{source_name}_metrics.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({
            "source": source_name,
            "papers_processed": len(all_results),
            "metrics_by_reviewer": averages
        }, f, ensure_ascii=False, indent=2)
    
    # Display results
    print(f"\n📈 Kết Quả Metrics cho {source_name}:")
    print("-" * 60)
    for reviewer_id, metrics in averages.items():
        print(f"\n{reviewer_id}:")
        print(f"  R_Premise (Tỷ lệ Premise):        {metrics['avg_r_premise']:.4f}")
        print(f"  S_Depth (Chất lượng Premise):     {metrics['avg_s_depth']:.4f}")
        print(f"  Combined (Chỉ số kết hợp):       {metrics['avg_combined']:.4f}")
        print(f"  Trung bình arguments/paper:      {metrics['avg_num_arguments']:.1f}")
        print(f"  Trung bình premises/paper:       {metrics['avg_num_premises']:.1f}")
    
    print(f"\n✅ Lưu tại: {result_file}")
    
    return averages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tính R_Premise, S_Depth, Combined Score cho từng source"
    )
    parser.add_argument(
        "--source", type=str, required=True,
        choices=["human"] + list(config.LLM_SOURCES.keys()),
        help="Tên source (human hoặc tên LLM)"
    )
    args = parser.parse_args()
    run_calculate_metrics(source_name=args.source)


