"""
CycleReviewer-ML-Llama-3.1-8B Pipeline for ICML2025
Uses the official ai_researcher package.

Usage:
    python run_cyclereviewer_icml2025.py --check     # environment check only
    python run_cyclereviewer_icml2025.py --limit 2   # test run
    python run_cyclereviewer_icml2025.py --all-papers # process all papers
    python run_cyclereviewer_icml2025.py             # full run (200 papers from subset)
"""

import os
import json
import argparse
import glob
import re
import sys
import time
from pathlib import Path
from datetime import datetime
import traceback

from cyclereviewer_config_icml2025 import (
    MODEL_SIZE, GPU_ID,
    GPU_MEMORY_UTILIZATION, MAX_MODEL_LEN,
    MMD_FOLDER, JSON_FOLDER,
    PAPER_IDS_FILE,
    OUTPUT_FOLDER, SUMMARY_FILE, SKIP_COMPLETED,
    HF_HOME, HF_TOKEN,
)

# Must be set before any model import
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)


def truncate_paper_to_tokens(text: str, tokenizer, max_tokens: int = 16000) -> str:
    """
    Truncate paper text to fit within token limit.
    Accounts for system prompt (~1000 tokens) + output tokens (7000) in vLLM.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)

    if len(tokens) <= max_tokens:
        return text

    print(f"    [TRUNCATE] Paper has {len(tokens):,} tokens, cutting to {max_tokens:,}")
    truncated_tokens = tokens[:max_tokens]
    truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)

    for marker in [". ", ".\n", "!\n", "?\n"]:
        idx = truncated_text.rfind(marker)
        if idx > max_tokens * 0.8:
            return truncated_text[:idx+1]

    return truncated_text


def load_paper(mmd_path: str) -> str:
    """Load paper content up to the References section."""
    with open(mmd_path, "r", encoding="utf-8") as f:
        text = f.read()

    match = re.search(r"(?im)^##\s*references\b.*$", text)
    if match:
        text = text[:match.start()]

    text = text.rstrip()
    if len(text) > 250000:
        print(f"  [WARN] Paper is very long ({len(text):,} chars, ~{len(text)//4} tokens)")

    return text


def load_ground_truth(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_paper_ids_filter(all_papers: bool = False):
    """Load paper IDs from config. Returns a set of IDs, or None for all papers."""
    if all_papers or not PAPER_IDS_FILE:
        return None

    ids_file = Path(PAPER_IDS_FILE)
    if not ids_file.exists():
        print(f"[WARN] PAPER_IDS_FILE not found: {PAPER_IDS_FILE}")
        return None

    with open(ids_file, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def _discover_paper_files(folder: str) -> dict:
    """
    Auto-detect paper text files in a folder.
    Supports .txt, .grobid.txt, and .mmd extensions.
    Returns {paper_id: file_path}.
    """
    result = {}
    for f in Path(folder).glob("*.grobid.txt"):
        paper_id = f.name.replace(".grobid.txt", "")
        result[paper_id] = str(f)
    if result:
        return result
    for f in Path(folder).glob("*.txt"):
        result[f.stem] = str(f)
    if result:
        return result
    for f in Path(folder).glob("*.mmd"):
        result[f.stem] = str(f)
    return result


def _resolve_paper_file(folder: str, paper_id: str) -> str | None:
    folder_path = Path(folder)
    for candidate in (
        folder_path / f"{paper_id}.grobid.txt",
        folder_path / f"{paper_id}.txt",
        folder_path / f"{paper_id}.mmd",
    ):
        if candidate.exists():
            return str(candidate)
    return None


def get_paper_pairs(paper_ids_filter=None) -> list:
    pairs = []

    if paper_ids_filter is not None:
        for paper_id in sorted(paper_ids_filter):
            paper_path = _resolve_paper_file(MMD_FOLDER, paper_id)
            if paper_path is None:
                continue

            json_path = os.path.join(JSON_FOLDER, f"{paper_id}.json")
            if os.path.exists(json_path):
                pairs.append((paper_id, paper_path, json_path))
        return pairs

    paper_files = _discover_paper_files(MMD_FOLDER)
    for paper_id, paper_path in sorted(paper_files.items()):
        json_path = os.path.join(JSON_FOLDER, f"{paper_id}.json")
        if os.path.exists(json_path):
            pairs.append((paper_id, paper_path, json_path))

    return pairs


def is_completed(paper_id: str) -> bool:
    result_path = os.path.join(OUTPUT_FOLDER, f"{paper_id}_result.json")
    if not os.path.exists(result_path):
        return False

    try:
        with open(result_path, "r", encoding="utf-8") as f:
            result = json.load(f)
    except Exception:
        return False

    generated_review = result.get("generated_review")
    if not isinstance(generated_review, dict) or not generated_review:
        return False

    if generated_review.get("paper_decision") or generated_review.get("avg_rating") is not None:
        return True

    return bool(generated_review.get("reviews") or generated_review.get("summary"))


def extract_gt_ratings(gt: dict) -> dict:
    ratings = []
    decision = gt.get("Decision", "Unknown")
    for review in gt.get("reviews", []):
        rating_str = review.get("Rating", "")
        try:
            ratings.append(float(rating_str.split(":")[0].strip()))
        except (ValueError, IndexError):
            pass
    avg = sum(ratings) / len(ratings) if ratings else None
    return {"decision": decision, "individual_ratings": ratings, "avg_rating": avg}


def get_tensor_parallel_size() -> int:
    return len([gpu.strip() for gpu in GPU_ID.split(",") if gpu.strip()])


def build_result(paper_id: str, gt_ratings: dict, generated_review: dict) -> dict:
    if generated_review is None:
        generated_review = {}
    return {
        "paper_id": paper_id,
        "model": f"CycleReviewer-ML-Llama-3.1-{MODEL_SIZE}",
        "timestamp": datetime.now().isoformat(),
        "ground_truth": {
            "decision": gt_ratings["decision"],
            "avg_rating": gt_ratings["avg_rating"],
            "individual_ratings": gt_ratings["individual_ratings"],
        },
        "generated_review": generated_review,
        "comparison": {
            "ground_truth_decision": gt_ratings["decision"],
            "ground_truth_avg_rating": gt_ratings["avg_rating"],
            "generated_avg_rating": generated_review.get("avg_rating"),
            "generated_decision": generated_review.get("paper_decision"),
        }
    }


def save_result(result: dict):
    out_path = os.path.join(OUTPUT_FOLDER, f"{result['paper_id']}_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def print_result_summary(gt_ratings: dict, result: dict):
    gt_avg = gt_ratings["avg_rating"]
    gt_avg_text = f"{gt_avg:.2f}" if gt_avg is not None else "N/A"
    print(
        f"  GT avg: {gt_avg_text} | Generated: {result['comparison']['generated_avg_rating']} "
        f"| GT: {gt_ratings['decision']} | Generated: {result['comparison']['generated_decision']}"
    )


def check_environment():
    print("=== ICML2025 CycleReviewer Environment Check ===")
    ok = True
    paper_ids_filter = load_paper_ids_filter(all_papers=args.all_papers)

    try:
        from ai_researcher import CycleReviewer
        print("[OK] ai_researcher package found")
    except ImportError:
        print("[FAIL] ai_researcher not installed — run: pip install ai_researcher")
        ok = False

    for label, path in [("MMD", MMD_FOLDER), ("JSON", JSON_FOLDER)]:
        if os.path.exists(path):
            file_count = len(glob.glob(os.path.join(path, '*')))
            print(f"[OK] {label} folder: {file_count} files")
        else:
            print(f"[FAIL] {label} folder not found: {path}")
            ok = False

    print(f"[OK] GPU: {GPU_ID}")
    print(f"[OK] Output: {OUTPUT_FOLDER}")

    if paper_ids_filter is not None:
        print(f"[OK] Loaded {len(paper_ids_filter)} paper IDs from {PAPER_IDS_FILE}")

    pairs = get_paper_pairs(paper_ids_filter)
    print(f"[OK] Found {len(pairs)} matched paper pairs")

    if pairs:
        first_pair = pairs[0]
        print(f"[OK] Sample pair: {first_pair[0]}")

    print("================================================")
    return ok


def run_pipeline(limit: int = None, all_papers: bool = False):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(os.path.dirname(SUMMARY_FILE), exist_ok=True)

    from ai_researcher import CycleReviewer
    from transformers import AutoTokenizer

    tensor_parallel_size = get_tensor_parallel_size()

    print(
        f"Initializing CycleReviewer {MODEL_SIZE} on GPU(s) {GPU_ID} "
        f"(tensor_parallel_size={tensor_parallel_size}, "
        f"gpu_memory_utilization={GPU_MEMORY_UTILIZATION}, max_model_len={MAX_MODEL_LEN})..."
    )

    reviewer = CycleReviewer(
        model_size=MODEL_SIZE,
        custom_model_name="/mnt/duyna/review_assessment/model/CycleReviewer-8B",
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
    )
    print("Model loaded.")

    tokenizer = AutoTokenizer.from_pretrained("/mnt/duyna/review_assessment/model/CycleReviewer-8B")

    paper_ids_filter = load_paper_ids_filter(all_papers=all_papers)
    if paper_ids_filter is not None:
        print(f"Loaded {len(paper_ids_filter)} paper IDs from filter")

    pairs = get_paper_pairs(paper_ids_filter)
    if limit:
        pairs = pairs[:limit]

    print(f"Processing {len(pairs)} papers...")
    results, failed = [], []
    skipped = 0

    for paper_index, (paper_id, mmd_path, json_path) in enumerate(pairs, 1):
        print(f"\n[{paper_index}/{len(pairs)}] {paper_id}")

        if SKIP_COMPLETED and is_completed(paper_id):
            print("  Skipping (already done)")
            skipped += 1
            continue

        try:
            paper_text = load_paper(mmd_path)
            paper_text = truncate_paper_to_tokens(paper_text, tokenizer, max_tokens=16000)

            gt = load_ground_truth(json_path)
            gt_ratings = extract_gt_ratings(gt)

            print(f"  Running single-paper inference ({len(paper_text):,} chars)...")
            started_at = time.time()
            generated_review = reviewer.evaluate(paper_text)[0]
            print(f"  Inference completed in {time.time() - started_at:.1f}s")

            result = build_result(paper_id, gt_ratings, generated_review)
            save_result(result)
            results.append(result)
            print_result_summary(gt_ratings, result)

        except Exception as e:
            error_details = traceback.format_exc()
            print(f"  FAILED: {e}")
            print(f"  Details:\n{error_details}")
            failed.append({
                "paper_id": paper_id,
                "error": str(e),
                "traceback": error_details
            })

    summary = {
        "dataset": "ICML2025",
        "model": f"CycleReviewer-ML-Llama-3.1-{MODEL_SIZE}",
        "total_processed": len(results),
        "total_failed": len(failed),
        "total_skipped": skipped,
        "results": results,
        "failed": failed,
    }
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(results)} processed, {skipped} skipped, {len(failed)} failed.")
    print(f"Results: {OUTPUT_FOLDER}/")
    print(f"Summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CycleReviewer pipeline for ICML2025 dataset"
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N papers")
    parser.add_argument("--check", action="store_true",
                        help="Check environment and exit")
    parser.add_argument("--all-papers", action="store_true",
                        help="Process all papers and ignore PAPER_IDS_FILE in config")
    args = parser.parse_args()

    if args.check:
        check_environment()
    else:
        if not check_environment():
            print("\nFix errors above before running.")
            exit(1)
        run_pipeline(limit=args.limit, all_papers=args.all_papers)
