"""
run_human.py — Phase 1: Xử lý tất cả Human reviews.

Chỉ cần chạy MỘT LẦN DUY NHẤT cho toàn bộ dataset.
Kết quả dùng chung để so sánh với mọi LLM source sau này.

Cách chạy:
    python pipeline/run_human.py
    python pipeline/run_human.py --backend gemini     (mặc định)
    python pipeline/run_human.py --backend gpt

Output: pipeline/output/human/{paper_id}.json
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
#  Text Extraction
# ================================================================

def extract_human_review_text(review_dict: dict) -> str:
    """Ghép 4 mục cốt lõi từ 1 review dict thành chuỗi văn bản."""
    sections = ["Summary", "Strengths", "Weaknesses"]
    parts = []
    for sec in sections:
        content = review_dict.get(sec, "").strip()
        if content:
            parts.append(f"**{sec}:**\n{content}")
    return "\n\n".join(parts)


# ================================================================
#  Checkpoint helpers
# ================================================================

def is_processed(paper_id: str) -> bool:
    path = os.path.join(config.OUTPUT_HUMAN_DIR, f"{paper_id}.json")
    return os.path.exists(path)


def save_result(paper_id: str, result: dict):
    os.makedirs(config.OUTPUT_HUMAN_DIR, exist_ok=True)
    path = os.path.join(config.OUTPUT_HUMAN_DIR, f"{paper_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


# ================================================================
#  Main Pipeline
# ================================================================

def run_human_pipeline(conference: str = "ICLR2026", workers: int = 1):
    config.HUMAN_DIR = config.HUMAN_DIRS[conference]
    config.OUTPUT_HUMAN_DIR = os.path.join(
        config.OUTPUT_ROOT, f"human_{conference.lower()}"
    )
    # Khởi tạo evaluator (sử dụng central unified agent DepthOfAnalysisEvaluator)
    from src.evaluator import DepthOfAnalysisEvaluator
    evaluator = DepthOfAnalysisEvaluator(api_key=config.GEMINI_API_KEY, model=config.GEMINI_MODEL)
    print(f"🤖 Backend: Unified Pluggable Client  |  Model: {config.GEMINI_MODEL}")

    # Lấy danh sách tất cả file Human
    all_files = sorted([
        f for f in os.listdir(config.HUMAN_DIR) if f.endswith(".json")
    ])
    total = len(all_files)
    already_done = sum(1 for f in all_files if is_processed(f.replace(".json", "")))

    print(f"\n📂 Human dir  : {config.HUMAN_DIR}")
    print(f"📁 Output dir : {config.OUTPUT_HUMAN_DIR}")
    print(f"📄 Tổng papers: {total}  |  Đã xong: {already_done}  |  Còn lại: {total - already_done}")
    print("=" * 60)

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    print_lock = threading.Lock()

    def process_file(filename: str):
        paper_id = filename.replace(".json", "")
        if is_processed(paper_id):
            return

        with print_lock:
            print(f"\n📄 [{paper_id}]")

        # Đọc file Human JSON
        with open(os.path.join(config.HUMAN_DIR, filename), "r", encoding="utf-8") as f:
            human_data = json.load(f)

        reviews_list = human_data.get("reviews", [])
        if not reviews_list:
            with print_lock:
                print(f"  ⚠️  Không có reviews, bỏ qua.")
            return

        result = {
            "paper_id": paper_id,
            "reviews_analysis": {}
        }
        total_prompt     = 0
        total_completion = 0

        for idx, review_dict in enumerate(reviews_list, start=1):
            reviewer_id  = f"Human_{idx}"
            review_text  = extract_human_review_text(review_dict)
            if not review_text:
                with print_lock:
                    print(f"  ⚠️  {reviewer_id}: text rỗng, bỏ qua.")
                continue

            with print_lock:
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
        with print_lock:
            print(f"  ✅ Đã lưu  [{', '.join(reviewers)}]  "
                  f"| tokens: {total_prompt + total_completion:,}")

    if workers > 1:
        todo_files = [f for f in all_files if not is_processed(f.replace(".json", ""))]
        if todo_files:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(process_file, f): f for f in todo_files}
                for future in tqdm(as_completed(futures), total=len(futures), desc="👤 Human Phase", unit="paper"):
                    try:
                        future.result()
                    except Exception as e:
                        fname = futures[future]
                        print(f"  ❌ Lỗi xử lý file {fname}: {e}")
    else:
        for filename in tqdm(all_files, desc="👤 Human Phase", unit="paper"):
            if is_processed(filename.replace(".json", "")):
                continue
            process_file(filename)

    print(f"\n🎉 Human Phase hoàn tất! Kết quả tại: {config.OUTPUT_HUMAN_DIR}")


# ================================================================
#  Entry Point
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DoA Pipeline — Phase 1: Process Human Reviews"
    )
    parser.add_argument(
        "--conference",
        choices=list(config.HUMAN_DIRS),
        default="ICLR2026",
        help="Conference folder to process.",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Số lượng luồng xử lý song song (mặc định: 1, tuần tự)"
    )
    args = parser.parse_args()
    
    # Read environment variable override if set
    env_workers = os.getenv("PRISM_MAX_WORKERS")
    workers = args.workers
    if env_workers:
        try:
            workers = int(env_workers)
        except ValueError:
            pass

    run_human_pipeline(conference=args.conference, workers=workers)
