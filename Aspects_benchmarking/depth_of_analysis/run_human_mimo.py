"""
run_human_mimo.py  Xu ly Human reviews bang Mimo v2.5 Pro evaluator.

Chay tren subset 50 paper IDs (paper_ids_50_iclr2024.txt).
Ho tro chay cho bat ky conference nao co human_reviews voi format JSON chuan.

Cach chay:
    # ICLR2024 (mac inh, 50 IDs)
    python pipeline/run_human_mimo.py

    # Chi inh thu muc human va file paper IDs tuy y
    python pipeline/run_human_mimo.py \\
        --human_dir "E:\\Final_LLM_Reviewer_Data\\ICLR2025\\human_reviews" \\
        --paper_ids "E:\\Final_LLM_Reviewer_Data\\ICLR2025\\paper_ids_50_iclr2025.txt" \\
        --output_suffix iclr2025

    # Chay tat ca (khong loc IDs)
    python pipeline/run_human_mimo.py --all

Output: pipeline/output/human_mimo_{suffix}/{paper_id}.json
"""

import sys
import os
import json
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipeline.config as config


# ================================================================
#  Text Extraction  (giong run_human.py)
# ================================================================

def extract_human_review_text(review_dict: dict) -> str:
    """Ghep Summary / Strengths / Weaknesses tu 1 review dict."""
    sections = ["Summary", "Strengths", "Weaknesses"]
    parts = []
    for sec in sections:
        content = review_dict.get(sec, "").strip()
        if content:
            parts.append(f"**{sec}:**\n{content}")
    return "\n\n".join(parts)


# ================================================================
#  Helpers
# ================================================================

def load_paper_ids(path: str) -> set:
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def is_processed(output_dir: str, paper_id: str) -> bool:
    return os.path.exists(os.path.join(output_dir, f"{paper_id}.json"))


def save_result(output_dir: str, paper_id: str, result: dict):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{paper_id}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


# ================================================================
#  Main Pipeline
# ================================================================

def run_human_mimo_pipeline(
    human_dir: str,
    output_dir: str,
    paper_ids_file: str | None,
    run_all: bool = False,
):
    # Khoi tao Mimo evaluator
    from src.mimo_client import MimoEvaluator
    evaluator = MimoEvaluator(
        api_key=config.MIMO_API_KEY,
        model=config.MIMO_MODEL,
        base_url=config.MIMO_BASE_URL,
        temperature=config.MIMO_TEMP,
    )
    print(f" Backend: Mimo  |  Model: {config.MIMO_MODEL}")

    # Loc paper IDs
    if run_all or paper_ids_file is None:
        target_ids = None   # None = khong loc
        print("  Chay ALL papers (khong loc theo IDs).")
    else:
        target_ids = load_paper_ids(paper_ids_file)
        print(f" Subset IDs    : {len(target_ids)} papers  ({paper_ids_file})")

    # Liet ke file
    all_files = sorted(f for f in os.listdir(human_dir) if f.endswith(".json"))
    if target_ids is not None:
        all_files = [f for f in all_files if f.replace(".json", "") in target_ids]

    total        = len(all_files)
    already_done = sum(1 for f in all_files if is_processed(output_dir, f.replace(".json", "")))

    print(f"\n Human dir  : {human_dir}")
    print(f" Output dir : {output_dir}")
    print(f" Tong papers: {total}  |  a xong: {already_done}  |  Con lai: {total - already_done}")
    print("=" * 60)

    for filename in tqdm(all_files, desc=" Human [Mimo]", unit="paper"):
        paper_id = filename.replace(".json", "")

        if is_processed(output_dir, paper_id):
            continue

        print(f"\n [{paper_id}]")

        with open(os.path.join(human_dir, filename), "r", encoding="utf-8") as f:
            human_data = json.load(f)

        reviews_list = human_data.get("reviews", [])
        if not reviews_list:
            print(f"    Khong co reviews, bo qua.")
            continue

        result = {
            "paper_id":         paper_id,
            "source":           "human",
            "evaluator_model":  config.MIMO_MODEL,
            "reviews_analysis": {},
        }
        total_prompt     = 0
        total_completion = 0

        for idx, review_dict in enumerate(reviews_list, start=1):
            reviewer_id = f"Human_{idx}"
            review_text = extract_human_review_text(review_dict)
            if not review_text:
                print(f"    {reviewer_id}: text rong, bo qua.")
                continue

            print(f"   {reviewer_id}")

            # Task 1
            arguments, u1 = evaluator.segment_arguments(review_text)

            # Task 2
            classified_args, u2 = evaluator.classify_arguments(review_text, arguments)

            # Task 3
            premise_texts = [a["argument"] for a in classified_args if a.get("role") == "Premise"]
            grounding_results, u3 = evaluator.score_grounding(review_text, premise_texts)

            grounding_map = {item["premise"]: item for item in grounding_results if "premise" in item}
            for arg in classified_args:
                if arg.get("role") == "Premise" and arg["argument"] in grounding_map:
                    arg["grounding_score"] = grounding_map[arg["argument"]].get("grounding_score")
                else:
                    arg["grounding_score"] = None

            result["reviews_analysis"][reviewer_id] = classified_args
            total_prompt     += u1["prompt_tokens"]     + u2["prompt_tokens"]     + u3["prompt_tokens"]
            total_completion += u1["completion_tokens"] + u2["completion_tokens"] + u3["completion_tokens"]

        result["usage_stats"] = {
            "prompt_tokens":     total_prompt,
            "completion_tokens": total_completion,
            "total_tokens":      total_prompt + total_completion,
        }

        save_result(output_dir, paper_id, result)
        reviewers = list(result["reviews_analysis"].keys())
        print(f"   a luu  [{', '.join(reviewers)}]  | tokens: {total_prompt + total_completion:,}")

    print(f"\n Human [Mimo] hoan tat! Ket qua tai: {output_dir}")


# ================================================================
#  Entry Point
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DoA Pipeline  Human reviews with Mimo v2.5 Pro evaluator"
    )
    parser.add_argument(
        "--human_dir", type=str,
        default=r"E:\Final_LLM_Reviewer_Data\ICLR2024\human_reviews",
        help="Thu muc chua human review JSON files."
    )
    parser.add_argument(
        "--paper_ids", type=str,
        default=config.PAPER_IDS_50_FILE,
        help="File .txt chua danh sach paper IDs can chay (1 ID/dong)."
    )
    parser.add_argument(
        "--output_suffix", type=str, default="iclr2024",
        help="Hau to output dir: pipeline/output/human_mimo_{suffix}/. Mac inh: iclr2024"
    )
    parser.add_argument(
        "--all", action="store_true", dest="run_all",
        help="Chay tat ca paper (bo qua --paper_ids)."
    )
    args = parser.parse_args()

    PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
    output_dir   = os.path.join(PIPELINE_DIR, "output", f"human_mimo_{args.output_suffix}")

    run_human_mimo_pipeline(
        human_dir     = args.human_dir,
        output_dir    = output_dir,
        paper_ids_file= None if args.run_all else args.paper_ids,
        run_all       = args.run_all,
    )

