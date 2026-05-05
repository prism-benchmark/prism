"""
run_human_icml2025_mimo.py  Xu ly Human reviews ICML2025 bang Mimo v2.5 Pro.

Mac inh chay tren subset 50 paper IDs (paper_ids_50_icml2025.txt).

Cach chay:
    python pipeline/run_human_icml2025_mimo.py
    python pipeline/run_human_icml2025_mimo.py --paper_ids /path/to/ids.txt
    python pipeline/run_human_icml2025_mimo.py --all   (chay tat ca)

Output: pipeline/output/human_icml2025_mimo/{paper_id}.json
"""

import sys
import os
import json
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipeline.config as config

# ================================================================
#  CAU HINH
# ================================================================

from config import HUMAN_DIRS as _HUMAN_DIRS
ICML2025_HUMAN_DIR = _HUMAN_DIRS['ICML2025']
ICML2025_PAPER_IDS_50 = config.paper_ids_file("ICML2025", 50)

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR   = os.path.join(PIPELINE_DIR, "output", "human_icml2025_mimo")

ICML2025_SECTIONS = {
    "Summary":                                "Summary",
    "Other Strengths And Weaknesses":         "Strengths & Weaknesses",
    "Claims And Evidence":                    "Claims And Evidence",
    "Methods And Evaluation Criteria":        "Methods And Evaluation",
    "Theoretical Claims":                     "Theoretical Claims",
    "Experimental Designs Or Analyses":       "Experimental Analyses",
    "Other Comments Or Suggestions":          "Comments & Suggestions",
    "Relation To Broader Scientific Literature": "Relation To Broader Scientific Literature",
    "Essential References Not Discussed":     "Essential References Not Discussed",
}


# ================================================================
#  Text Extraction  ICML2025 format
# ================================================================

def extract_icml2025_review_text(review_dict: dict) -> str:
    parts = []
    for field, label in ICML2025_SECTIONS.items():
        content = review_dict.get(field, "").strip()
        if content and content.lower() not in ("none.", "n/a", "affirmed", "tbd", "na"):
            parts.append(f"**{label}:**\n{content}")
    return "\n\n".join(parts)


# ================================================================
#  Helpers
# ================================================================

def load_paper_ids(filepath: str) -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def is_processed(paper_id: str) -> bool:
    return os.path.exists(os.path.join(OUTPUT_DIR, f"{paper_id}.json"))


def save_result(paper_id: str, result: dict):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, f"{paper_id}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


# ================================================================
#  Main Pipeline
# ================================================================

def run_pipeline(paper_ids_file: str = None, run_all: bool = False):
    from src.mimo_client import MimoEvaluator
    evaluator = MimoEvaluator(
        api_key    = config.MIMO_API_KEY,
        model      = config.MIMO_MODEL,
        base_url   = config.MIMO_BASE_URL,
        temperature= config.MIMO_TEMP,
    )
    print(f" Backend: Mimo  |  Model: {config.MIMO_MODEL}")

    if run_all:
        all_files  = sorted(f for f in os.listdir(ICML2025_HUMAN_DIR) if f.endswith(".json"))
        target_ids = [f.replace(".json", "") for f in all_files]
        print(f" Che o: ALL papers ({len(target_ids)} papers)")
    else:
        ids_file   = paper_ids_file or ICML2025_PAPER_IDS_50
        target_ids = load_paper_ids(ids_file)
        print(f" Subset IDs: {len(target_ids)} papers  ({ids_file})")

    valid_ids   = [pid for pid in target_ids
                   if os.path.exists(os.path.join(ICML2025_HUMAN_DIR, f"{pid}.json"))]
    missing_ids = [pid for pid in target_ids if pid not in valid_ids]
    already_done = sum(1 for pid in valid_ids if is_processed(pid))

    print(f"\n Human dir  : {ICML2025_HUMAN_DIR}")
    print(f" Output dir : {OUTPUT_DIR}")
    print(f" Tong papers: {len(valid_ids)}  |  a xong: {already_done}  |  Con lai: {len(valid_ids) - already_done}")
    if missing_ids:
        print(f"  Thieu file: {len(missing_ids)}  {missing_ids[:5]}{'...' if len(missing_ids) > 5 else ''}")
    print("=" * 60)

    for paper_id in tqdm(valid_ids, desc=" Human ICML2025 [Mimo]", unit="paper"):
        if is_processed(paper_id):
            continue

        print(f"\n [{paper_id}]")

        with open(os.path.join(ICML2025_HUMAN_DIR, f"{paper_id}.json"), "r", encoding="utf-8", errors="replace") as f:
            human_data = json.load(f)

        reviews_list = human_data.get("reviews", [])
        if not reviews_list:
            print(f"    Khong co reviews, bo qua.")
            continue

        result = {
            "paper_id":        paper_id,
            "dataset":         "icml2025",
            "evaluator_model": config.MIMO_MODEL,
            "reviews_analysis": {},
        }
        total_prompt = total_completion = 0

        for idx, review_dict in enumerate(reviews_list, start=1):
            reviewer_id = f"Human_{review_dict.get('Review ID', idx)}"
            review_text = extract_icml2025_review_text(review_dict)
            if not review_text:
                print(f"    {reviewer_id}: text rong, bo qua.")
                continue

            print(f"   {reviewer_id}")

            arguments,        u1 = evaluator.segment_arguments(review_text)
            classified_args,  u2 = evaluator.classify_arguments(review_text, arguments)
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
        save_result(paper_id, result)
        reviewers = list(result["reviews_analysis"].keys())
        print(f"   a luu  [{', '.join(reviewers)}]  | tokens: {total_prompt + total_completion:,}")

    print(f"\n Human ICML2025 [Mimo] hoan tat! Ket qua tai: {OUTPUT_DIR}")
    print(f" Tong a xu ly: {sum(1 for pid in valid_ids if is_processed(pid))}/{len(valid_ids)}")


# ================================================================
#  Entry Point
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DoA  Human ICML2025 with Mimo v2.5 Pro evaluator"
    )
    parser.add_argument("--paper_ids", type=str, default=None,
                        help=f"File paper IDs (mac inh: {ICML2025_PAPER_IDS_50})")
    parser.add_argument("--all", action="store_true",
                        help="Chay tat ca papers (bo qua --paper_ids)")
    args = parser.parse_args()

    run_pipeline(paper_ids_file=args.paper_ids, run_all=args.all)

