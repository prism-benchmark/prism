"""
run_human_icml2025.py — Phase 1: Xử lý Human reviews cho ICML2025.

Cấu trúc file ICML2025 khác với ICLR2024:
  - Không có trường "Strengths" / "Weaknesses" riêng biệt
  - Thay vào đó có các mục chi tiết:
      Summary, Claims And Evidence, Methods And Evaluation Criteria,
      Theoretical Claims, Experimental Designs Or Analyses,
      Other Strengths And Weaknesses, Other Comments Or Suggestions, v.v.

Cách chạy:
    python pipeline/run_human_icml2025.py
    python pipeline/run_human_icml2025.py --paper_ids e:/path/to/paper_ids.txt
    python pipeline/run_human_icml2025.py --all       (chạy tất cả, không lọc theo ID)

Output: pipeline/output/human_icml2025/{paper_id}.json
"""

import sys
import os
import json
import argparse
from tqdm import tqdm

# Thêm root project vào sys.path để import src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipeline.config as config

# ================================================================
#  CẤU HÌNH ICML2025
# ================================================================

import sys as _sys
_sys.path.insert(0, _sys.path[0] if _sys.path else '') 
from config import HUMAN_DIRS as _HUMAN_DIRS
ICML2025_HUMAN_DIR = _HUMAN_DIRS['ICML2025']
ICML2025_PAPER_IDS_FILE = config.paper_ids_file("ICML2025", 200)

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR   = os.path.join(PIPELINE_DIR, "output", "human_icml2025")

# Các section cần trích xuất từ review ICML2025
# (map sang Summary / Strengths / Weaknesses tương đương)
ICML2025_SECTIONS = {
    "Summary": "Summary",
    "Other Strengths And Weaknesses": "Strengths & Weaknesses",
    "Claims And Evidence": "Claims And Evidence",
    "Methods And Evaluation Criteria": "Methods And Evaluation",
    "Theoretical Claims": "Theoretical Claims",
    "Experimental Designs Or Analyses": "Experimental Analyses",
    "Other Comments Or Suggestions": "Comments & Suggestions",
    "Relation To Broader Scientific Literature": "Relation To Broader Scientific Literature",
    "Essential References Not Discussed": "Essential References Not Discussed"
}


# ================================================================
#  Text Extraction — ICML2025 format
# ================================================================

def extract_icml2025_review_text(review_dict: dict) -> str:
    """
    Ghép các section quan trọng từ 1 review ICML2025 thành chuỗi văn bản.

    ICML2025 review format:
      - Review ID, Overall Recommendation
      - Summary
      - Claims And Evidence
      - Methods And Evaluation Criteria
      - Theoretical Claims
      - Experimental Designs Or Analyses
      - Supplementary Material
      - Relation To Broader Scientific Literature
      - Essential References Not Discussed
      - Other Strengths And Weaknesses
      - Other Comments Or Suggestions
      - Questions For Authors
      - Code Of Conduct
    """
    parts = []

    for field, label in ICML2025_SECTIONS.items():
        content = review_dict.get(field, "").strip()
        # Bỏ qua nội dung quá ngắn hoặc vô nghĩa
        if content and content.lower() not in ("none.", "n/a", "affirmed", "tbd", "na"):
            parts.append(f"**{label}:**\n{content}")

    return "\n\n".join(parts)


# ================================================================
#  Checkpoint helpers
# ================================================================

def is_processed(paper_id: str) -> bool:
    path = os.path.join(OUTPUT_DIR, f"{paper_id}.json")
    return os.path.exists(path)


def save_result(paper_id: str, result: dict):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{paper_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


# ================================================================
#  Load paper IDs
# ================================================================

def load_paper_ids(filepath: str) -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ================================================================
#  Main Pipeline
# ================================================================

def run_human_icml2025_pipeline(paper_ids_file: str = None, run_all: bool = False):
    # Khởi tạo evaluator (sử dụng central unified agent DepthOfAnalysisEvaluator)
    from src.evaluator import DepthOfAnalysisEvaluator
    evaluator = DepthOfAnalysisEvaluator()
    print(
        "🤖 Backend: Unified Pluggable Client"
        f"  |  Provider: {evaluator.client.provider}"
        f"  |  Model: {evaluator.client.model}"
    )

    # Xác định danh sách paper IDs cần xử lý
    if run_all:
        # Lấy tất cả file trong thư mục human_reviews
        all_files = sorted([f for f in os.listdir(ICML2025_HUMAN_DIR) if f.endswith(".json")])
        target_ids = [f.replace(".json", "") for f in all_files]
        print(f"📌 Chế độ: ALL papers ({len(target_ids)} papers)")
    else:
        ids_file = paper_ids_file or ICML2025_PAPER_IDS_FILE
        target_ids = load_paper_ids(ids_file)
        print(f"📌 Chế độ: Filter theo IDs  |  File: {ids_file}")
        print(f"📄 Số paper IDs: {len(target_ids)}")

    # Lọc chỉ những file tồn tại
    valid_ids = []
    missing_ids = []
    for pid in target_ids:
        fpath = os.path.join(ICML2025_HUMAN_DIR, f"{pid}.json")
        if os.path.exists(fpath):
            valid_ids.append(pid)
        else:
            missing_ids.append(pid)

    already_done = sum(1 for pid in valid_ids if is_processed(pid))

    print(f"\n📂 Human dir  : {ICML2025_HUMAN_DIR}")
    print(f"📁 Output dir : {OUTPUT_DIR}")
    print(f"📄 Tổng papers: {len(valid_ids)}  |  Đã xong: {already_done}  |  Còn lại: {len(valid_ids) - already_done}")
    if missing_ids:
        print(f"⚠️  Thiếu file: {len(missing_ids)} papers  → {missing_ids[:5]}{'...' if len(missing_ids) > 5 else ''}")
    print("=" * 60)

    for paper_id in tqdm(valid_ids, desc="👤 Human ICML2025", unit="paper"):
        # --- Checkpoint ---
        if is_processed(paper_id):
            continue

        print(f"\n📄 [{paper_id}]")

        # Đọc file review JSON
        fpath = os.path.join(ICML2025_HUMAN_DIR, f"{paper_id}.json")
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            human_data = json.load(f)

        reviews_list = human_data.get("reviews", [])
        if not reviews_list:
            print(f"  ⚠️  Không có reviews, bỏ qua.")
            continue

        result = {
            "paper_id": paper_id,
            "dataset": "icml2025",
            "reviews_analysis": {}
        }
        total_prompt     = 0
        total_completion = 0

        for idx, review_dict in enumerate(reviews_list, start=1):
            # Lấy Review ID nếu có, fallback về số thứ tự
            reviewer_id = f"Human_{review_dict.get('Review ID', idx)}"
            review_text = extract_icml2025_review_text(review_dict)

            if not review_text:
                print(f"  ⚠️  {reviewer_id}: text rỗng, bỏ qua.")
                continue

            print(f"  → {reviewer_id}")

            # Task 1 — Argument Segmentation
            arguments, u1 = evaluator.segment_arguments(review_text)

            # Task 2 — Role & Aspect Classification
            classified_args, u2 = evaluator.classify_arguments(review_text, arguments)

            # Task 3 — Grounding Score (chỉ cho Premise)
            premise_texts = [
                a["argument"] for a in classified_args if a.get("role") == "Premise"
            ]
            grounding_results, u3 = evaluator.score_grounding(review_text, premise_texts)

            # Gắn grounding_score vào từng Premise
            grounding_map = {item["premise"]: item for item in grounding_results}
            for arg in classified_args:
                if arg.get("role") == "Premise" and arg["argument"] in grounding_map:
                    arg["grounding_score"] = grounding_map[arg["argument"]]["grounding_score"]
                else:
                    arg["grounding_score"] = None

            result["reviews_analysis"][reviewer_id] = classified_args
            total_prompt     += u1["prompt_tokens"]  + u2["prompt_tokens"]  + u3["prompt_tokens"]
            total_completion += u1["completion_tokens"] + u2["completion_tokens"] + u3["completion_tokens"]

        result["usage_stats"] = {
            "prompt_tokens":     total_prompt,
            "completion_tokens": total_completion,
            "total_tokens":      total_prompt + total_completion
        }

        save_result(paper_id, result)
        reviewers = list(result["reviews_analysis"].keys())
        print(f"  ✅ Đã lưu  [{', '.join(reviewers)}]  "
              f"| tokens: {total_prompt + total_completion:,}")

    print(f"\n🎉 Human ICML2025 Phase hoàn tất! Kết quả tại: {OUTPUT_DIR}")
    print(f"📊 Tổng đã xử lý: {sum(1 for pid in valid_ids if is_processed(pid))}/{len(valid_ids)}")


# ================================================================
#  Entry Point
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DoA Pipeline — Phase 1 ICML2025: Process Human Reviews"
    )
    parser.add_argument(
        "--paper_ids",
        type=str,
        default=None,
        help=f"Path đến file paper IDs (mặc định: {ICML2025_PAPER_IDS_FILE})"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Chạy tất cả papers trong thư mục (không filter theo IDs)"
    )
    args = parser.parse_args()

    run_human_icml2025_pipeline(
        paper_ids_file=args.paper_ids,
        run_all=args.all
    )
