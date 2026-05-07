import argparse
import concurrent.futures
import csv
import logging
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from batch_common import (
    BatchConfig,
    DEFAULT_CHECKPOINT_NAME,
    FINAL_OUTPUT_NAME,
    LOG_NAME,
    ManifestRecord,
    STANDARDIZED_REVIEWS_NAME,
    final_output_exists,
    load_existing_status,
    read_manifest,
    resolve_manifest_record,
    utc_now_iso,
    write_status_csv,
)
from run_one_paper import run_one_paper


def setup_batch_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("treereview.batch")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def parse_args():
    parser = argparse.ArgumentParser(description="Batch runner for TreeReview without modifying TreeReview core logic.")
    parser.add_argument("--manifest", required=True, help="CSV with columns: paper_id,paper_path,reviews_json,output_dir")
    parser.add_argument("--status-csv", default=None, help="Path to status.csv. Defaults to <manifest_dir>/status.csv")
    parser.add_argument("--batch-log", default=None, help="Path to batch log. Defaults to <manifest_dir>/batch.log")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--retrieval-top-k", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--ranker-model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--ranker-device", type=str, default="cuda")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--no-save-standardized-reviews", action="store_true")
    parser.add_argument("--max-papers", type=int, default=None)
    return parser.parse_args()


def materialize_skip_result(record: ManifestRecord) -> Dict[str, str]:
    output_dir = record.output_dir
    return {
        "paper_id": record.paper_id,
        "status": "skipped",
        "output_dir": output_dir,
        "final_output_path": str(Path(output_dir) / FINAL_OUTPUT_NAME),
        "checkpoint_path": str(Path(output_dir) / DEFAULT_CHECKPOINT_NAME),
        "log_path": str(Path(output_dir) / LOG_NAME),
        "standardized_reviews_path": str(Path(output_dir) / STANDARDIZED_REVIEWS_NAME),
        "started_at": utc_now_iso(),
        "finished_at": utc_now_iso(),
        "duration_seconds": "0.000",
        "error_message": "",
    }


def worker_entry(payload: Tuple[ManifestRecord, BatchConfig]):
    record, config = payload
    return run_one_paper(
        paper_id=record.paper_id,
        paper_path=record.paper_path,
        reviews_json=record.reviews_json,
        output_dir=record.output_dir,
        config=config,
    )


def main():
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    status_csv = Path(args.status_csv).resolve() if args.status_csv else manifest_path.parent / "status.csv"
    batch_log = Path(args.batch_log).resolve() if args.batch_log else manifest_path.parent / "batch.log"
    status_csv.parent.mkdir(parents=True, exist_ok=True)
    batch_log.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_batch_logger(str(batch_log))

    raw_records = read_manifest(str(manifest_path))
    records = [resolve_manifest_record(r, str(manifest_path)) for r in raw_records]
    if args.max_papers is not None:
        records = records[: args.max_papers]

    config = BatchConfig(
        num_workers=max(1, args.num_workers),
        max_depth=args.max_depth,
        retrieval_top_k=args.retrieval_top_k,
        chunk_size=args.chunk_size,
        ranker_model=args.ranker_model,
        ranker_device=args.ranker_device,
        force_rerun=args.force_rerun,
        save_standardized_reviews=not args.no_save_standardized_reviews,
    )

    existing_status = load_existing_status(str(status_csv))
    merged_status: Dict[str, Dict[str, str]] = dict(existing_status)
    run_queue: List[ManifestRecord] = []

    for record in records:
        output_done = final_output_exists(record.output_dir)
        prior = existing_status.get(record.paper_id, {})
        if not config.force_rerun and (output_done or prior.get("status") == "done"):
            merged_status[record.paper_id] = materialize_skip_result(record)
            logger.info("Skip queued paper_id=%s because final output already exists.", record.paper_id)
            continue
        run_queue.append(record)

    write_status_csv(str(status_csv), merged_status.values())
    logger.info(
        "Prepared batch: total_manifest=%s | queued=%s | skipped=%s | workers=%s",
        len(records), len(run_queue), len(records) - len(run_queue), config.num_workers,
    )

    if not run_queue:
        logger.info("Nothing to run.")
        return

    mp_ctx = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=config.num_workers, mp_context=mp_ctx) as executor:
        futures = {executor.submit(worker_entry, (record, config)): record for record in run_queue}
        for future in concurrent.futures.as_completed(futures):
            record = futures[future]
            try:
                result = future.result()
                row = result.to_status_row()
                merged_status[record.paper_id] = row
                logger.info("Finished paper_id=%s with status=%s", record.paper_id, row["status"])
            except Exception as e:
                logger.exception("Worker crashed for paper_id=%s", record.paper_id)
                merged_status[record.paper_id] = {
                    "paper_id": record.paper_id,
                    "status": "failed",
                    "output_dir": record.output_dir,
                    "final_output_path": str(Path(record.output_dir) / FINAL_OUTPUT_NAME),
                    "checkpoint_path": str(Path(record.output_dir) / DEFAULT_CHECKPOINT_NAME),
                    "log_path": str(Path(record.output_dir) / LOG_NAME),
                    "standardized_reviews_path": str(Path(record.output_dir) / STANDARDIZED_REVIEWS_NAME),
                    "started_at": utc_now_iso(),
                    "finished_at": utc_now_iso(),
                    "duration_seconds": "0.000",
                    "error_message": f"WorkerCrash: {type(e).__name__}: {e}",
                }
            finally:
                write_status_csv(str(status_csv), merged_status.values())

    counts: Dict[str, int] = {}
    for row in merged_status.values():
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    logger.info("Batch complete. Status counts: %s", counts)


if __name__ == "__main__":
    main()
