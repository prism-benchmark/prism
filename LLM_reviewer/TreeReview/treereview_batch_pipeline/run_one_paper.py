import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from batch_common import (
    DEFAULT_CHECKPOINT_NAME,
    FINAL_OUTPUT_NAME,
    LOG_NAME,
    STANDARDIZED_REVIEWS_NAME,
    RuntimeConfig,
    PaperRunResult,
    ensure_output_dir,
    utc_now_iso,
)
from phase_adapter import PhaseInputAdapter


_WORKER_RESOURCES: Optional[Dict[str, Any]] = None
_WORKER_CONFIG: Optional[RuntimeConfig] = None


def setup_paper_logger(log_path: str, paper_id: str) -> logging.Logger:
    logger_name = f"treereview.paper.{paper_id}"
    logger = logging.getLogger(logger_name)
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


def initialize_worker_resources(config: RuntimeConfig) -> None:
    global _WORKER_RESOURCES, _WORKER_CONFIG
    if _WORKER_RESOURCES is not None and _WORKER_CONFIG == config:
        return
    from treereview.agents.answer_synthesizer import AnswerSynthesizer
    from treereview.agents.question_generator import QuestionGenerator
    from treereview.utility.LLMClient import LLMClient
    from treereview.utility.context_ranker import ContextRanker

    llm = LLMClient()
    question_gen = QuestionGenerator(llm=llm)
    context_ranker = ContextRanker(model_name=config.ranker_model, device_map=config.ranker_device)
    answer_syn = AnswerSynthesizer(llm=llm)
    _WORKER_RESOURCES = {
        "question_gen": question_gen,
        "context_ranker": context_ranker,
        "answer_syn": answer_syn,
    }
    _WORKER_CONFIG = config


def load_paper(paper_id: str, paper_path: str, chunk_size: int):
    from treereview.utility.paper_loader import PaperLoader

    loader = PaperLoader(
        paper_id=paper_id,
        paper_path=paper_path,
        chunk_config={"chunk_size": chunk_size},
    )
    return loader.get_paper()


def run_one_paper(
    paper_id: str,
    paper_path: str,
    reviews_json: str,
    output_dir: str,
    config: RuntimeConfig,
) -> PaperRunResult:
    ensure_output_dir(output_dir)
    log_path = str(Path(output_dir) / LOG_NAME)
    logger = setup_paper_logger(log_path, paper_id)
    final_output_path = str(Path(output_dir) / FINAL_OUTPUT_NAME)
    checkpoint_path = str(Path(output_dir) / DEFAULT_CHECKPOINT_NAME)
    standardized_reviews_path = str(Path(output_dir) / STANDARDIZED_REVIEWS_NAME)

    if os.path.exists(final_output_path) and not config.force_rerun:
        logger.info("Skipping %s because final output already exists: %s", paper_id, final_output_path)
        return PaperRunResult(
            paper_id=paper_id,
            status="skipped",
            output_dir=output_dir,
            final_output_path=final_output_path,
            checkpoint_path=checkpoint_path,
            log_path=log_path,
            standardized_reviews_path=standardized_reviews_path if os.path.exists(standardized_reviews_path) else None,
            started_at=utc_now_iso(),
            finished_at=utc_now_iso(),
            duration_seconds=0.0,
        )

    started_at = utc_now_iso()
    start_ts = time.time()
    logger.info("Starting paper_id=%s", paper_id)
    logger.info("Inputs: paper=%s | reviews=%s | output_dir=%s", paper_path, reviews_json, output_dir)
    try:
        initialize_worker_resources(config)
        assert _WORKER_RESOURCES is not None

        paper = load_paper(paper_id=paper_id, paper_path=paper_path, chunk_size=config.chunk_size)
        adapter = PhaseInputAdapter(paper_id=paper_id, paper_path=paper_path, reviews_json_path=reviews_json)
        bundle = adapter.build_bundle(paper)
        standardized_reviews = bundle["human_reviews"]
        if config.save_standardized_reviews:
            adapter.save_json(standardized_reviews_path, standardized_reviews)
            logger.info("Saved standardized reviews to %s", standardized_reviews_path)

        from treereview.core import PipelineConfig, ReviewPipeline

        pipeline = ReviewPipeline(
            paper=paper,
            question_generator=_WORKER_RESOURCES["question_gen"],
            context_ranker=_WORKER_RESOURCES["context_ranker"],
            answer_synthesizer=_WORKER_RESOURCES["answer_syn"],
            config=PipelineConfig(max_depth=config.max_depth, retrieval_top_k=config.retrieval_top_k),
            state_file=checkpoint_path,
        )
        logger.info(
            "Running TreeReview: max_depth=%s retrieval_top_k=%s chunk_size=%s ranker_model=%s ranker_device=%s",
            config.max_depth,
            config.retrieval_top_k,
            config.chunk_size,
            config.ranker_model,
            config.ranker_device,
        )
        tree_review_result = pipeline.run()
        metadata = {
            "paper_id": paper_id,
            "input_paths": {"paper_path": paper_path, "reviews_json": reviews_json},
            "output_dir": output_dir,
            "checkpoint_path": checkpoint_path,
            "config": {
                "max_depth": config.max_depth,
                "retrieval_top_k": config.retrieval_top_k,
                "chunk_size": config.chunk_size,
                "ranker_model": config.ranker_model,
                "ranker_device": config.ranker_device,
            },
            "paper_stats": {
                "title": paper.title,
                "num_chunks": len(paper.chunks),
            },
            "started_at": started_at,
            "finished_at": utc_now_iso(),
        }
        final_output = {
            "paper_id": paper_id,
            "tree_review_result": tree_review_result,
            "standardized_reviews": standardized_reviews,
            "metadata": metadata,
        }
        with open(final_output_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        duration = time.time() - start_ts
        logger.info("Completed paper_id=%s in %.2f seconds", paper_id, duration)
        return PaperRunResult(
            paper_id=paper_id,
            status="done",
            output_dir=output_dir,
            final_output_path=final_output_path,
            checkpoint_path=checkpoint_path,
            log_path=log_path,
            standardized_reviews_path=standardized_reviews_path if config.save_standardized_reviews else None,
            started_at=started_at,
            finished_at=utc_now_iso(),
            duration_seconds=duration,
        )
    except Exception as e:
        duration = time.time() - start_ts
        err_text = f"{type(e).__name__}: {e}"
        logger.exception("Paper failed: %s", err_text)
        traceback.print_exc()
        return PaperRunResult(
            paper_id=paper_id,
            status="failed",
            output_dir=output_dir,
            final_output_path=final_output_path,
            checkpoint_path=checkpoint_path,
            log_path=log_path,
            standardized_reviews_path=standardized_reviews_path if os.path.exists(standardized_reviews_path) else None,
            error_message=err_text,
            started_at=started_at,
            finished_at=utc_now_iso(),
            duration_seconds=duration,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Run TreeReview for one paper without changing TreeReview core logic.")
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--paper-path", required=True)
    parser.add_argument("--reviews-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--retrieval-top-k", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--ranker-model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--ranker-device", type=str, default="cuda")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--no-save-standardized-reviews", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_one_paper(
        paper_id=args.paper_id,
        paper_path=args.paper_path,
        reviews_json=args.reviews_json,
        output_dir=args.output_dir,
        config=RuntimeConfig(
            max_depth=args.max_depth,
            retrieval_top_k=args.retrieval_top_k,
            chunk_size=args.chunk_size,
            ranker_model=args.ranker_model,
            ranker_device=args.ranker_device,
            force_rerun=args.force_rerun,
            save_standardized_reviews=not args.no_save_standardized_reviews,
        ),
    )
    print(json.dumps(result.to_status_row(), indent=2, ensure_ascii=False))
    raise SystemExit(0 if result.status in {"done", "skipped"} else 1)


if __name__ == "__main__":
    main()
