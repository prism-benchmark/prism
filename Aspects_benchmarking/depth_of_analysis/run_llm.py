"""
run_llm.py — Phase 2: Xử lý LLM reviews từ một source cụ thể.

Chạy riêng cho từng LLM source. Có thể chạy song song nhiều terminal.
Mỗi source lưu kết quả vào thư mục output riêng.
Tự động retry 5 lần nếu Gemini API bị quá tải.

Cách chạy:
    python pipeline/run_llm.py --source sea
    python pipeline/run_llm.py --source tree_iclr2024
    python pipeline/run_llm.py --source reviewer2_iclr2024
    python pipeline/run_llm.py --source deepreview_iclr2024

Danh sách source hiện có: xem LLM_SOURCES trong pipeline/config.py

Định dạng source được hỗ trợ:
    txt             — file .txt,  tên {paper_id}.txt  (SEA style: **Section:**)
    reviewer2_txt   — file .txt,  tên {paper_id}.txt  (có </think>, ## Section)
    tree_json       — file .json, tên {paper_id}_review.json, field "full_review"
    deepreview_json — file .json, tên {paper_id}.json,
                      generated_review[0].reviews[reviewer_id==1].text (### Section)

Output: pipeline/output/{source_name}/{paper_id}.json
"""

import sys
import os
import json
import re
import argparse
from tqdm import tqdm

# Thêm root project vào sys.path để import src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipeline.config as config


# ================================================================
#  Text Extraction — theo từng format
# ================================================================

def extract_from_txt(raw_text: str) -> str:
    """Trích 3 mục cốt lõi từ file TXT review của LLM (format SEA)."""
    sections = ["Summary", "Strengths", "Weaknesses"]
    parts = []
    for sec in sections:
        # Pattern: **Section:** followed by content until next **Section:** or end of text
        # Lookahead: \*\*(?:Summary|Strengths|Weaknesses):\*\* hoặc \Z
        pattern = rf"\*\*{sec}:\*\*(.*?)(?=\*\*(?:Summary|Strengths|Weaknesses):\*\*|\Z)"
        match = re.search(pattern, raw_text, re.DOTALL | re.IGNORECASE)
        if match:
            parts.append(f"**{sec}:**\n{match.group(1).strip()}")
    # Fallback: nếu không trích được gì thì dùng toàn bộ text
    return "\n\n".join(parts) if parts else raw_text.strip()


def extract_from_reviewer2_txt(raw_text: str) -> str:
    """
    Trích 3 mục cốt lõi từ file TXT reviewer2_iclr2024.
    Format: có thinking text trước </think>, sau đó dùng ## hoặc ### Section headers.
    Một số file dùng ## (H2), một số dùng ### (H3) — hỗ trợ cả hai.

    Sections cần lấy:
        ## (hay ###) Summary Of The Paper
        ## (hay ###) Strengths
        ## (hay ###) Weaknesses
    """
    # 1. Strip phần thinking text (trước và kể cả </think>)
    think_end = raw_text.find("</think>")
    if think_end != -1:
        text = raw_text[think_end + len("</think>"):].strip()
    else:
        text = raw_text.strip()

    # 2. Trích các section bằng ##/### header
    # Mapping: tên section → label hiển thị
    target_sections = [
        ("Summary Of The Paper", "Summary"),
        ("Strengths",            "Strengths"),
        ("Weaknesses",           "Weaknesses"),
        ("Limitations Not Addressed By The Authors", "Limitations Not Addressed By The Authors"),
        ("Brief Justification For Rating", "Brief Justification For Rating"),# Dự phòng: nếu có section khác không tên chuẩn, sẽ không trích được → fallback về toàn bộ text
    ]

    parts = []
    for section_name, label in target_sections:
        # Match ## hoặc ### Section Name (trailing spaces, bất kể case)
        # Lookahead: dừng tại header ##/### tiếp theo hoặc cuối chuỗi
        # Dùng {{2,3}} trong f-string để tạo regex quantifier {2,3}
        pattern = rf"#{{2,3}}\s+{re.escape(section_name)}\s*\n(.*?)(?=\n#{{2,3}}\s|\Z)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            parts.append(f"**{label}:**\n{content}")

    # Fallback: nếu không parse được → dùng toàn bộ text sau </think>
    return "\n\n".join(parts) if parts else text


def extract_from_tree_json(file_path: str) -> str:
    """
    Đọc file {paper_id}_review.json của tree_iclr2024_full.
    Lấy field 'full_review' rồi trích 3 mục cốt lõi.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    full_review = data.get("full_review", "").strip()
    if not full_review:
        return ""
    # full_review đã dùng markdown **Section:** nên dùng lại logic txt
    return extract_from_txt(full_review)


def extract_from_cyclereview_json(file_path: str) -> str:
    """
    Đọc file JSON của cyclereview (iclr2024/2025/2026, icml2025).
    Cấu trúc: generated_review.content  (dict, không phải list)
    content chứa nhiều reviewer phân tách bởi '**********', mỗi reviewer dùng:
        ## Reviewer
        ### Summary / ### Strengths / ### Weaknesses / ### Questions / ...

    Chỉ lấy reviewer đầu tiên, trích 3 section cốt lõi:
        ### Summary, ### Strengths, ### Weaknesses
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gr = data.get("generated_review", {})
    if not isinstance(gr, dict):
        return ""

    content = gr.get("content", "").strip()
    if not content:
        return ""

    # Lấy chỉ phần reviewer đầu tiên (trước dấu phân tách **********)
    first_reviewer = content.split("**********")[0].strip()

    # Trích ### Summary, ### Strengths, ### Weaknesses
    target_sections = [
        ("Summary",    "Summary"),
        ("Strengths",  "Strengths"),
        ("Weaknesses", "Weaknesses"),
    ]

    parts = []
    for section_name, label in target_sections:
        # Match ### Section, stop at next ### hoặc cuối chuỗi
        pattern = rf"###\s+{re.escape(section_name)}\s*\n(.*?)(?=\n###\s|\Z)"
        match = re.search(pattern, first_reviewer, re.DOTALL | re.IGNORECASE)
        if match:
            content_sec = match.group(1).strip()
            parts.append(f"**{label}:**\n{content_sec}")

    # Fallback: trả về toàn bộ reviewer đầu tiên nếu không parse được
    return "\n\n".join(parts) if parts else first_reviewer


def extract_from_deepreview_json(file_path: str) -> str:
    """
    Đọc file JSON của deepreview_iclr2024.
    Cấu trúc: generated_review[0].reviews[reviewer_id==1].text
    text dùng ### Section (H3) headers.

    Sections cần lấy:
        ### Summary
        ### Strengths
        ### Weaknesses
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Lấy reviews từ generated_review[0]
    generated = data.get("generated_review", [])
    if not generated:
        return ""

    reviews = generated[0].get("reviews", [])
    if not reviews:
        return ""

    # Lấy reviewer_id == 1, fallback về reviewer đầu tiên nếu không tìm thấy
    reviewer1 = next(
        (r for r in reviews if r.get("reviewer_id") == 1),
        reviews[0]
    )
    text = reviewer1.get("text", "").strip()
    if not text:
        return ""

    # Trích ### Summary, ### Strengths, ### Weaknesses từ text
    target_sections = [
        ("Summary",    "Summary"),
        ("Strengths",  "Strengths"),
        ("Weaknesses", "Weaknesses"),
    ]

    parts = []
    for section_name, label in target_sections:
        # Match ### Section (H3), stop at next ### hoặc cuối file
        pattern = rf"###\s+{re.escape(section_name)}\s*\n(.*?)(?=\n###\s|\Z)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            # Bỏ sub-section headers kiểu #### nếu cần giữ clean
            parts.append(f"**{label}:**\n{content}")

    # Fallback: không parse được → trả về toàn bộ text
    return "\n\n".join(parts) if parts else text


# ================================================================
#  Source loader — trả về list (paper_id, review_text)
# ================================================================

def load_source_files(source_dir: str, fmt: str):
    """
    Generator: yield (paper_id, review_text) cho từng file trong source_dir.

    fmt:
        "txt"       — glob *.txt, paper_id = filename không có .txt
        "tree_json" — glob *_review.json, paper_id = filename không có _review.json
    """
    if fmt == "txt":
        files = sorted(f for f in os.listdir(source_dir) if f.endswith(".txt"))
        for fname in files:
            paper_id = fname[:-4]          # strip ".txt"
            fpath    = os.path.join(source_dir, fname)
            with open(fpath, "r", encoding="utf-8") as fh:
                raw = fh.read()
            yield paper_id, extract_from_txt(raw), fname

    elif fmt == "reviewer2_txt":
        files = sorted(f for f in os.listdir(source_dir) if f.endswith(".txt"))
        for fname in files:
            paper_id = fname[:-4]          # strip ".txt"
            fpath    = os.path.join(source_dir, fname)
            with open(fpath, "r", encoding="utf-8") as fh:
                raw = fh.read()
            yield paper_id, extract_from_reviewer2_txt(raw), fname

    elif fmt == "tree_json":
        files = sorted(f for f in os.listdir(source_dir) if f.endswith("_review.json"))
        for fname in files:
            paper_id = fname[:-len("_review.json")]   # strip "_review.json"
            fpath    = os.path.join(source_dir, fname)
            text     = extract_from_tree_json(fpath)
            yield paper_id, text, fname

    elif fmt == "deepreview_json":
        files = sorted(f for f in os.listdir(source_dir) if f.endswith(".json"))
        for fname in files:
            paper_id = fname[:-5]          # strip ".json"
            fpath    = os.path.join(source_dir, fname)
            text     = extract_from_deepreview_json(fpath)
            yield paper_id, text, fname

    elif fmt == "cyclereview_json":
        files = sorted(f for f in os.listdir(source_dir) if f.endswith(".json"))
        for fname in files:
            paper_id = fname[:-5]          # strip ".json"
            fpath    = os.path.join(source_dir, fname)
            text     = extract_from_cyclereview_json(fpath)
            yield paper_id, text, fname

    else:
        raise ValueError(
            f"Format '{fmt}' chưa được hỗ trợ. "
            f"Các format hiện có: txt, reviewer2_txt, tree_json, deepreview_json"
        )


# ================================================================
#  Checkpoint helpers
# ================================================================

def is_processed(source_name: str, paper_id: str) -> bool:
    path = os.path.join(config.get_llm_output_dir(source_name), f"{paper_id}.json")
    return os.path.exists(path)


def save_result(source_name: str, paper_id: str, result: dict):
    out_dir = config.get_llm_output_dir(source_name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{paper_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


# ================================================================
#  Main Pipeline
# ================================================================

def run_llm_pipeline(source_name: str, workers: int = 1):
    # Kiểm tra source hợp lệ
    if source_name not in config.LLM_SOURCES:
        print(f"❌ Source '{source_name}' không tồn tại.")
        print(f"   Các source hiện có: {list(config.LLM_SOURCES.keys())}")
        print(f"   Thêm source mới trong pipeline/config.py → LLM_SOURCES")
        sys.exit(1)

    source_cfg = config.LLM_SOURCES[source_name]
    source_dir = source_cfg["dir"]
    fmt        = source_cfg.get("format", "txt")
    llm_key    = source_name.upper()   # reviewer_id, e.g. "SEA", "TREE_ICLR2024"

    # Kiểm tra thư mục source tồn tại
    if not os.path.isdir(source_dir):
        print(f"❌ Thư mục source không tồn tại: {source_dir}")
        sys.exit(1)

    # Khởi tạo evaluator (sử dụng central unified agent DepthOfAnalysisEvaluator)
    from src.evaluator import DepthOfAnalysisEvaluator
    evaluator = DepthOfAnalysisEvaluator(api_key=config.GEMINI_API_KEY, model=config.GEMINI_MODEL)
    print(f"🤖 Backend: Unified Pluggable Client  |  Model: {config.GEMINI_MODEL}  |  Max Retries: 5")

    # Đếm tổng file để hiển thị progress
    if fmt in ("txt", "reviewer2_txt"):
        all_fnames = [f for f in os.listdir(source_dir) if f.endswith(".txt")]
    elif fmt == "tree_json":
        all_fnames = [f for f in os.listdir(source_dir) if f.endswith("_review.json")]
    elif fmt in ("deepreview_json", "cyclereview_json"):
        all_fnames = [f for f in os.listdir(source_dir) if f.endswith(".json")]
    else:
        all_fnames = []

    total        = len(all_fnames)
    already_done = sum(
        1 for f in all_fnames
        if is_processed(
            source_name,
            f[:-4]  if fmt in ("txt", "reviewer2_txt") else
            f[:-5]  if fmt in ("deepreview_json", "cyclereview_json") else
            f[:-len("_review.json")]   # tree_json
        )
    )

    print(f"\n📂 Source     : {source_name}  (format: {fmt})")
    print(f"📁 Source dir : {source_dir}")
    print(f"📁 Output dir : {config.get_llm_output_dir(source_name)}")
    print(f"📄 Tổng papers: {total}  |  Đã xong: {already_done}  |  Còn lại: {total - already_done}")
    print("=" * 60)

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    print_lock = threading.Lock()

    def process_item(paper_id: str, review_text: str, fname: str):
        if is_processed(source_name, paper_id):
            return

        with print_lock:
            print(f"\n📄 [{paper_id}]  (file: {fname})")

        if not review_text:
            with print_lock:
                print(f"  ⚠️  Text rỗng, bỏ qua.")
            return

        with print_lock:
            print(f"  → {llm_key}")

        # Task 1 — Argument Segmentation
        arguments, u1 = evaluator.segment_arguments(review_text)

        # Task 2 — Role & Aspect Classification
        classified_args, u2 = evaluator.classify_arguments(review_text, arguments)

        # Task 3 — Grounding Score
        premise_texts = [
            a["argument"] for a in classified_args if a.get("role") == "Premise"
        ]
        grounding_results, u3 = evaluator.score_grounding(review_text, premise_texts)

        # Gắn grounding_score vào từng Premise
        grounding_map = {item["premise"]: item for item in grounding_results if "premise" in item}
        for arg in classified_args:
            if arg.get("role") == "Premise" and arg["argument"] in grounding_map:
                arg["grounding_score"] = grounding_map[arg["argument"]].get("grounding_score")
            else:
                arg["grounding_score"] = None

        total_prompt     = u1["prompt_tokens"]     + u2["prompt_tokens"]     + u3["prompt_tokens"]
        total_completion = u1["completion_tokens"] + u2["completion_tokens"] + u3["completion_tokens"]

        result = {
            "paper_id":         paper_id,
            "source":           source_name,
            "source_format":    fmt,
            "reviews_analysis": {
                llm_key: classified_args
            },
            "usage_stats": {
                "prompt_tokens":     total_prompt,
                "completion_tokens": total_completion,
                "total_tokens":      total_prompt + total_completion
            }
        }

        save_result(source_name, paper_id, result)
        with print_lock:
            print(f"  ✅ Đã lưu  | tokens: {total_prompt + total_completion:,}")

    source_files = list(load_source_files(source_dir, fmt))

    if workers > 1:
        todo_items = [item for item in source_files if not is_processed(source_name, item[0])]
        if todo_items:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(process_item, pid, rtxt, fn): pid for pid, rtxt, fn in todo_items}
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"🤖 LLM [{source_name}]", unit="paper"):
                    try:
                        future.result()
                    except Exception as e:
                        pid = futures[future]
                        print(f"  ❌ Lỗi xử lý paper {pid}: {e}")
    else:
        for paper_id, review_text, fname in tqdm(
            source_files,
            total=total,
            desc=f"🤖 LLM [{source_name}]",
            unit="paper"
        ):
            if is_processed(source_name, paper_id):
                continue
            process_item(paper_id, review_text, fname)

    print(f"\n🎉 LLM [{source_name}] Phase hoàn tất! Kết quả tại: {config.get_llm_output_dir(source_name)}")


# ================================================================
#  Entry Point
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DoA Pipeline — Phase 2: Process LLM Reviews"
    )
    parser.add_argument(
        "--source", type=str, required=True,
        help=f"Tên LLM source cần xử lý. Hiện có: {list(config.LLM_SOURCES.keys())}"
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

    run_llm_pipeline(source_name=args.source, workers=workers)

