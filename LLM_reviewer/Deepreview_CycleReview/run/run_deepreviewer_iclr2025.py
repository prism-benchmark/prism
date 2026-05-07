#!/usr/bin/env python3
"""
DeepReviewer Pipeline Runner — ICLR2025 Dataset
Iterates over paired paper + review files from separate folders

Usage:
    python run_deepreviewer_iclr2025.py                  # run full pipeline
    python run_deepreviewer_iclr2025.py --check          # check environment only
    python run_deepreviewer_iclr2025.py --limit 2        # only process first 2 papers (for testing)
"""

import os
import sys
import json
import re
import time
import argparse
import subprocess
from pathlib import Path

# ── Load config ───────────────────────────────────────────────
try:
    import deepreviewer_config_iclr2025 as deepreviewer_config
except ImportError:
    print("ERROR: deepreviewer_config_iclr2025.py not found. Must be in the same directory.")
    sys.exit(1)

# ── Argument parsing ──────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--check", action="store_true")
parser.add_argument("--limit", type=int, default=None,
                    help="Process only first N papers (useful for testing)")
parser.add_argument("--all-papers", action="store_true",
                    help="Process all papers (ignore PAPER_IDS_FILE in config)")
args = parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_gpu_ids(raw_gpu_value):
    value = str(raw_gpu_value).strip()
    if not value or value.lower() == "auto":
        return None
    return [gpu.strip() for gpu in value.split(",") if gpu.strip()]


def select_gpus_for_deepreviewer():
    configured_gpus = parse_gpu_ids(deepreviewer_config.DEEPREVIEWER_GPU)
    if configured_gpus is not None:
        return configured_gpus

    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("Failed to query GPU memory with nvidia-smi")

    gpu_rows = []
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        gpu_rows.append(
            {
                "index": int(parts[0]),
                "free": int(parts[1]),
                "used": int(parts[2]),
            }
        )

    if len(gpu_rows) < deepreviewer_config.TENSOR_PARALLEL_SIZE:
        raise RuntimeError(
            f"Found only {len(gpu_rows)} GPUs, but TENSOR_PARALLEL_SIZE={deepreviewer_config.TENSOR_PARALLEL_SIZE}"
        )

    gpu_rows.sort(key=lambda row: (row["free"], -row["used"]), reverse=True)
    selected = gpu_rows[: deepreviewer_config.TENSOR_PARALLEL_SIZE]
    selected_ids = [str(row["index"]) for row in selected]
    selected_desc = ", ".join(
        f"GPU {row['index']} ({row['free']} MiB free, {row['used']} MiB used)" for row in selected
    )
    log(f"✓ Auto-selected DeepReviewer GPUs: {selected_desc}")
    return selected_ids


def paper_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".grobid.txt"):
        return name[:-len(".grobid.txt")]
    if name.endswith(".txt"):
        return name[:-len(".txt")]
    return path.stem


def list_paper_files(folder: Path):
    files = []
    for pattern in ("*.grobid.txt", "*.txt", "*.grobid.txt"):
        files.extend(folder.glob(pattern))
    seen = set()
    unique_files = []
    for path in sorted(files):
        if path in seen:
            continue
        seen.add(path)
        unique_files.append(path)
    return unique_files


def check_environment():
    log("Checking environment...")
    errors = []

    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    if result.returncode != 0:
        errors.append("nvidia-smi failed — no GPU detected")
    else:
        log("✓ GPU detected")

    try:
        import vllm
        log(f"✓ vLLM {vllm.__version__} installed")
    except ImportError:
        errors.append("vLLM not installed — run: pip install vllm")

    try:
        import ai_researcher
        log("✓ ai_researcher installed")
    except ImportError:
        errors.append("ai_researcher not installed — run: pip install ai_researcher")

    if "YOUR" in deepreviewer_config.HF_TOKEN:
        errors.append("HF_TOKEN not set in config.py")
    else:
        log("✓ HF_TOKEN set")

    if deepreviewer_config.REVIEW_MODE == "Best Mode":
        log("⚠ Best Mode is disabled in this runner; it will fall back to Standard Mode")
    else:
        log(f"✓ Mode is '{deepreviewer_config.REVIEW_MODE}'")

    try:
        gpu_ids = select_gpus_for_deepreviewer()
        log(f"✓ DeepReviewer: {len(gpu_ids)} GPU(s), tensor_parallel_size={deepreviewer_config.TENSOR_PARALLEL_SIZE}")
    except Exception as exc:
        errors.append(str(exc))

    papers_folder = Path(deepreviewer_config.PAPERS_FOLDER)
    json_folder = Path(deepreviewer_config.JSON_FOLDER)

    if not papers_folder.exists():
        errors.append(f"PAPERS_FOLDER not found: {deepreviewer_config.PAPERS_FOLDER}")
    if not json_folder.exists():
        errors.append(f"JSON_FOLDER not found: {deepreviewer_config.JSON_FOLDER}")

    if papers_folder.exists() and json_folder.exists():
        paper_files = list_paper_files(papers_folder)
        json_files = list(json_folder.glob("*.json"))
        paper_stems = {paper_stem(f) for f in paper_files}
        json_stems = {f.stem for f in json_files}
        paired = paper_stems & json_stems
        log(f"✓ PAPERS_FOLDER : {len(paper_files)} paper files")
        log(f"✓ JSON_FOLDER: {len(json_files)} .json files")
        log(f"✓ Paired     : {len(paired)} papers")
        if len(paired) == 0:
            errors.append("No paired files found — filenames must match across both folders")

    if errors:
        print("\n── ERRORS ──────────────────────────────")
        for e in errors:
            print(f"  ✗ {e}")
        print("────────────────────────────────────────\n")
        return False

    log("✓ Environment OK")
    return True


def load_paper_ids_filter():
    if args.all_papers or not deepreviewer_config.PAPER_IDS_FILE:
        return None

    ids_file = Path(deepreviewer_config.PAPER_IDS_FILE)
    if not ids_file.exists():
        log(f"WARNING: PAPER_IDS_FILE not found: {deepreviewer_config.PAPER_IDS_FILE}")
        return None

    with open(ids_file, "r", encoding="utf-8") as f:
        ids = {line.strip() for line in f if line.strip()}

    return ids


def get_paired_files(paper_ids_filter=None):
    paper_files = {paper_stem(f): f for f in list_paper_files(Path(deepreviewer_config.PAPERS_FOLDER))}
    json_files = {f.stem: f for f in Path(deepreviewer_config.JSON_FOLDER).glob("*.json")}

    paired = [
        (stem, paper_files[stem], json_files[stem])
        for stem in sorted(paper_files)
        if stem in json_files
    ]

    if paper_ids_filter is not None:
        paired = [p for p in paired if p[0] in paper_ids_filter]

    return paired


def analyze_cutoff(text):
    cutoff_match = re.search(
        r"(?im)^##\s*(references|bibliography|appendix|appendices)\b.*$",
        text,
    )
    if not cutoff_match:
        return text.rstrip(), None

    cutoff_heading = cutoff_match.group(0).strip()
    return text[:cutoff_match.start()].rstrip(), cutoff_heading


def count_text_tokens(text, tokenizer):
    return len(tokenizer.encode(text, add_special_tokens=False))


def load_ground_truth(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_completed(paper_id):
    if not getattr(deepreviewer_config, "SKIP_COMPLETED", True):
        return False

    result_path = Path(deepreviewer_config.OUTPUT_FOLDER) / f"{paper_id}.json"
    if not result_path.exists():
        return False

    try:
        with open(result_path, "r", encoding="utf-8") as f:
            result = json.load(f)
    except Exception:
        return False

    generated_review = result.get("generated_review")
    comparison = result.get("comparison")

    if not isinstance(generated_review, list) or not generated_review:
        return False

    first_review = generated_review[0]
    if not isinstance(first_review, dict):
        return False

    if not str(first_review.get("raw_text", "")).strip():
        return False

    reviews = first_review.get("reviews")
    if not isinstance(reviews, list) or not reviews:
        return False

    if not isinstance(comparison, dict) or comparison.get("generated_avg_rating") is None:
        return False

    return True


def avg_rating_from_ground_truth(ground_truth):
    try:
        ratings = []
        for r in ground_truth.get("reviews", []):
            num = float(str(r.get("Rating", "0")).split(":")[0].strip())
            ratings.append(num)
        return round(sum(ratings) / len(ratings), 2) if ratings else None
    except Exception:
        return None


def extract_generated_ratings(generated_review):
    try:
        return [
            float(r["rating"])
            for r in generated_review[0].get("reviews", [])
            if r.get("rating") is not None
        ]
    except Exception:
        return []


def save_paper_result(paper_id, generated_review, ground_truth):
    os.makedirs(deepreviewer_config.OUTPUT_FOLDER, exist_ok=True)
    gen_ratings = extract_generated_ratings(generated_review)
    result = {
        "paper_id": paper_id,
        "ground_truth": ground_truth,
        "generated_review": generated_review,
        "comparison": {
            "ground_truth_decision": ground_truth.get("Decision", "N/A"),
            "ground_truth_avg_rating": avg_rating_from_ground_truth(ground_truth),
            "generated_ratings": gen_ratings,
            "generated_avg_rating": round(sum(gen_ratings) / len(gen_ratings), 2)
            if gen_ratings else None,
        }
    }
    path = Path(deepreviewer_config.OUTPUT_FOLDER) / f"{paper_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    return result


def save_summary(all_results):
    summary = [
        {
            "paper_id": r["paper_id"],
            "ground_truth_decision": r["comparison"]["ground_truth_decision"],
            "ground_truth_avg_rating": r["comparison"]["ground_truth_avg_rating"],
            "generated_ratings": r["comparison"]["generated_ratings"],
            "generated_avg_rating": r["comparison"]["generated_avg_rating"],
        }
        for r in all_results
    ]
    with open(deepreviewer_config.SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"✓ Summary saved to {deepreviewer_config.SUMMARY_FILE}")


if __name__ == "__main__":
    if not check_environment():
        sys.exit(1)

    if args.check:
        log("--check flag set, exiting without running.")
        sys.exit(0)

    try:
        selected_gpus = select_gpus_for_deepreviewer()
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected_gpus)
        log(f"✓ CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
        from huggingface_hub import login
        login(token=deepreviewer_config.HF_TOKEN)
        log("✓ HuggingFace login successful")

        from ai_researcher import DeepReviewer
        log(f"Loading DeepReviewer-{deepreviewer_config.DEEPREVIEWER_SIZE} "
            f"on GPU(s) {deepreviewer_config.DEEPREVIEWER_GPU} "
            f"(tensor_parallel_size={deepreviewer_config.TENSOR_PARALLEL_SIZE})...")

        reviewer = DeepReviewer(
            model_size=deepreviewer_config.DEEPREVIEWER_SIZE,
            device="cuda",
            tensor_parallel_size=deepreviewer_config.TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=deepreviewer_config.GPU_MEMORY_UTILIZATION
        )
        log("✓ Model loaded")

        paper_ids_filter = load_paper_ids_filter()
        if paper_ids_filter:
            log(f"✓ Loaded {len(paper_ids_filter)} paper IDs from {deepreviewer_config.PAPER_IDS_FILE}")

        paired_files = get_paired_files(paper_ids_filter)
        if args.limit:
            paired_files = paired_files[:args.limit]
            log(f"--limit {args.limit}: processing first {args.limit} papers only")

        if paper_ids_filter:
            log(f"✓ Filtered to {len(paired_files)} papers from paper IDs file")
        else:
            log(f"✓ Processing all {len(paired_files)} paired papers")

        batch_size = deepreviewer_config.BATCH_SIZE
        log(f"Batch size: {batch_size}")
        log(f"Total papers to process: {len(paired_files)}")
        all_results, skipped, failed = [], 0, []

        for batch_idx in range(0, len(paired_files), batch_size):
            batch_papers = paired_files[batch_idx:batch_idx + batch_size]
            log(f"\n── Batch [{batch_idx+1}-{min(batch_idx+batch_size, len(paired_files))}/{len(paired_files)}] ──────────────────")

            for idx_in_batch, (paper_id, paper_path, json_path) in enumerate(batch_papers, 1):
                paper_idx = batch_idx + idx_in_batch
                log(f"\n  [{paper_idx}/{len(paired_files)}] {paper_id}")

                if is_completed(paper_id):
                    log("    Skipping (already completed)")
                    skipped += 1
                    continue

                try:
                    with open(paper_path, "r", encoding="utf-8") as f:
                        raw_paper_text = f.read()
                    paper_text, cutoff_heading = analyze_cutoff(raw_paper_text)
                    paper_tokens = count_text_tokens(paper_text, reviewer.tokenizer)
                    ground_truth = load_ground_truth(json_path)
                    log(f"    Paper: {len(paper_text)} chars | "
                        f"cut tokens: {paper_tokens} | "
                        f"GT decision: {ground_truth.get('Decision', 'N/A')} | "
                        f"GT avg rating: {avg_rating_from_ground_truth(ground_truth)}")
                    if cutoff_heading:
                        log(f"    Cut at heading: {cutoff_heading}")
                    else:
                        log("    Cut at heading: none")
                    if paper_tokens < 9000:
                        if cutoff_heading:
                            log("    WARNING: cut paper is under 9000 tokens after heading-based cutoff; inspect this cutoff before trusting the review")
                        else:
                            log("    WARNING: paper is under 9000 tokens with no cutoff applied; this looks like a short or incomplete source paper, not a cutoff bug")

                    generated_review = reviewer.evaluate(
                        paper_text,
                        mode=deepreviewer_config.REVIEW_MODE,
                        reviewer_num=deepreviewer_config.REVIEWER_NUM
                    )

                    result = save_paper_result(paper_id, generated_review, ground_truth)
                    all_results.append(result)
                    log(f"    ✓ Generated ratings: {result['comparison']['generated_ratings']} "
                        f"(avg: {result['comparison']['generated_avg_rating']})")

                except Exception as e:
                    log(f"    ✗ FAILED: {e}")
                    failed.append({"paper_id": paper_id, "error": str(e)})

        log(f"\n── Run complete ──────────────────────────────────")
        log(f"  Processed : {len(all_results)}")
        log(f"  Skipped   : {skipped}")
        log(f"  Failed    : {len(failed)}")

        if all_results:
            save_summary(all_results)

        if failed:
            with open("failed_papers.json", "w") as f:
                json.dump(failed, f, indent=2)
            log("  Failed papers logged to failed_papers.json")

    except KeyboardInterrupt:
        log("Interrupted by user")
    except Exception as e:
        log(f"ERROR: {e}")
        raise