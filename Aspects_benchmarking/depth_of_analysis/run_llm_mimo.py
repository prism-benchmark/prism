"""
Run the Depth-of-Analysis pipeline on LLM reviews using the Mimo evaluator.

Examples:
    python pipeline/run_llm_mimo.py --source sea_iclr2024
    python pipeline/run_llm_mimo.py --source reviewer2_iclr2024

    # Custom paper-id subset
    python pipeline/run_llm_mimo.py --source sea_iclr2024 \\
        --paper_ids "/path/to/paper_ids.txt"

    # Run all papers in the source directory
    python pipeline/run_llm_mimo.py --source sea_iclr2024 --all
"""

import sys
import os
import json
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipeline.config as config

# Tai su dung toan bo ham extract/load tu run_llm.py
from pipeline.run_llm import load_source_files


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

def run_llm_mimo_pipeline(source_name: str, paper_ids_file: str | None, run_all: bool = False):
    if source_name not in config.LLM_SOURCES:
        print(f" Source '{source_name}' not found.")
        print(f"   Available sources: {list(config.LLM_SOURCES.keys())}")
        sys.exit(1)

    source_cfg = config.LLM_SOURCES[source_name]
    source_dir = source_cfg["dir"]
    fmt        = source_cfg.get("format", "txt")
    llm_key    = source_name.upper()

    if not os.path.isdir(source_dir):
        print(f" Source directory does not exist: {source_dir}")
        sys.exit(1)

    # Output dir: pipeline/output/mimo_{source_name}/
    output_dir = config.get_mimo_output_dir(source_name)

    # Loc paper IDs
    if run_all or paper_ids_file is None:
        target_ids = None
        print("  Running ALL papers (no ID filtering).")
    else:
        target_ids = load_paper_ids(paper_ids_file)
        print(f" Subset IDs: {len(target_ids)} papers ({paper_ids_file})")

    # Khoi tao Mimo evaluator
    from src.mimo_client import MimoEvaluator
    evaluator = MimoEvaluator(
        api_key    = config.MIMO_API_KEY,
        model      = config.MIMO_MODEL,
        base_url   = config.MIMO_BASE_URL,
        temperature= config.MIMO_TEMP,
    )
    print(f" Backend: Mimo  |  Model: {config.MIMO_MODEL}")

    # em file sau khi loc
    if fmt in ("txt", "reviewer2_txt"):
        all_fnames = [f for f in os.listdir(source_dir) if f.endswith(".txt")]
    elif fmt == "tree_json":
        all_fnames = [f for f in os.listdir(source_dir) if f.endswith("_review.json")]
    elif fmt in ("deepreview_json", "cyclereview_json"):
        all_fnames = [f for f in os.listdir(source_dir) if f.endswith(".json")]
    else:
        all_fnames = []

    if target_ids is not None:
        def _get_pid(fname):
            if fmt in ("txt", "reviewer2_txt"):
                return fname[:-4]
            elif fmt == "tree_json":
                return fname[:-len("_review.json")]
            else:
                return fname[:-5]
        all_fnames = [f for f in all_fnames if _get_pid(f) in target_ids]

    total        = len(all_fnames)
    already_done = sum(
        1 for f in all_fnames
        if is_processed(
            output_dir,
            f[:-4] if fmt in ("txt", "reviewer2_txt") else
            f[:-5] if fmt in ("deepreview_json", "cyclereview_json") else
            f[:-len("_review.json")]
        )
    )

    print(f"\n Source     : {source_name}  (format: {fmt})")
    print(f" Source dir : {source_dir}")
    print(f" Output dir : {output_dir}")
    print(f" Total papers: {total} | Completed: {already_done} | Remaining: {total - already_done}")
    print("=" * 60)

    for paper_id, review_text, fname in tqdm(
        load_source_files(source_dir, fmt),
        total=len(all_fnames),
        desc=f" Mimo [{source_name}]",
        unit="paper",
    ):
        # Loc theo target IDs
        if target_ids is not None and paper_id not in target_ids:
            continue

        # Checkpoint
        if is_processed(output_dir, paper_id):
            continue

        print(f"\n [{paper_id}]  (file: {fname})")

        if not review_text:
            print("    Empty review text, skipping.")
            continue

        print(f"   {llm_key}")

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

        total_prompt     = u1["prompt_tokens"]     + u2["prompt_tokens"]     + u3["prompt_tokens"]
        total_completion = u1["completion_tokens"] + u2["completion_tokens"] + u3["completion_tokens"]

        result = {
            "paper_id":         paper_id,
            "source":           source_name,
            "source_format":    fmt,
            "evaluator_model":  config.MIMO_MODEL,
            "reviews_analysis": {
                llm_key: classified_args,
            },
            "usage_stats": {
                "prompt_tokens":     total_prompt,
                "completion_tokens": total_completion,
                "total_tokens":      total_prompt + total_completion,
            },
        }

        save_result(output_dir, paper_id, result)
        print(f"   Saved | tokens: {total_prompt + total_completion:,}")

    print(f"\n Mimo [{source_name}] complete. Results: {output_dir}")


# ================================================================
#  Entry Point
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Depth of Analysis: LLM reviews with Mimo evaluator."
    )
    parser.add_argument(
        "--source", type=str, required=True,
        help=f"LLM source name. Available: {list(config.LLM_SOURCES.keys())}"
    )
    parser.add_argument(
        "--paper_ids", type=str,
        default=config.PAPER_IDS_50_FILE,
        help="Path to a .txt file of target paper IDs."
    )
    parser.add_argument(
        "--all", action="store_true", dest="run_all",
        help="Run all papers in the source directory (ignore --paper_ids)."
    )
    args = parser.parse_args()

    run_llm_mimo_pipeline(
        source_name    = args.source,
        paper_ids_file = None if args.run_all else args.paper_ids,
        run_all        = args.run_all,
    )

