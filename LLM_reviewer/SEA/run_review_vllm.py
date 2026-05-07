"""
vLLM-based Review Generation - Optimized for Speed
Usage: python run_review_vllm.py --model model_name --input-dir INPUT_DIR --paper-ids PAPER_IDS_FILE --output-dir OUTPUT_DIR
"""

import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional
import logging

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def read_txt_file(path: str) -> str:
    """Read text file content."""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def read_json_file(path: str) -> dict:
    """Read JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_output(output: str, save_dir: str, data_id: str) -> None:
    """Save output to file."""
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"{data_id}.txt"), 'w', encoding='utf-8') as f:
        f.write(output)


def get_paper_ids(paper_ids_file: str) -> List[str]:
    """Load paper IDs from file (one per line)."""
    if not os.path.exists(paper_ids_file):
        logger.warning(f"Paper IDs file not found: {paper_ids_file}. Processing all available papers.")
        return None
    
    with open(paper_ids_file, 'r', encoding='utf-8') as f:
        paper_ids = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Loaded {len(paper_ids)} paper IDs from {paper_ids_file}")
    return paper_ids


def get_paper_files(input_dir: str, paper_ids: Optional[List[str]] = None) -> dict:
    """Get text files to process (.grobid.txt format).
    
    Returns dict of {paper_id: file_path}
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    grobid_files = {}

    def resolve_text_file(paper_id: str) -> Optional[Path]:
        candidates = [
            input_path / f"{paper_id}.grobid.txt",
            input_path / f"{paper_id}.txt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None
    
    if paper_ids:
        # Only process specified papers
        for paper_id in paper_ids:
            text_file = resolve_text_file(paper_id)

            if text_file is not None:
                grobid_files[paper_id] = str(text_file)
            else:
                logger.warning(
                    f"Text file not found for paper {paper_id}: "
                    f"{input_path / f'{paper_id}.grobid.txt'} or {input_path / f'{paper_id}.txt'}"
                )
    else:
        # Process all available .grobid.txt and .txt files
        all_files = sorted(
            list(input_path.glob("*.grobid.txt")) + list(input_path.glob("*.txt"))
        )
        seen = set()
        for text_file in all_files:
            if text_file in seen:
                continue
            seen.add(text_file)

            if text_file.name.endswith(".grobid.txt"):
                paper_id = text_file.name.replace(".grobid.txt", "")
            else:
                paper_id = text_file.stem
            grobid_files[paper_id] = str(text_file)
    
    logger.info(f"Found {len(grobid_files)} papers to process (GROBID format)")
    return grobid_files


def prepare_prompts(grobid_files: dict, template_path: str, max_model_len: int = 20480, max_output_tokens: int = 2000) -> tuple:
    """Prepare prompts and metadata for batch processing.
    
    Automatically truncates paper content to fit within model context window.
    
    Args:
        grobid_files: Dictionary of {paper_id: grobid_file_path}
        template_path: Path to template JSON file
        max_model_len: Model's maximum context length
        max_output_tokens: Reserved tokens for model output
    
    Returns (paper_ids, prompts)
    """
    template = read_json_file(template_path)
    instruction = template.get('instruction_e', '')
    
    # Load tokenizer to accurately count tokens
    try:
        # Using Qwen tokenizer as default
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct", trust_remote_code=True)
        logger.info("Loaded Qwen tokenizer for token counting")
    except Exception as e:
        logger.warning(f"Could not load tokenizer: {e}. Using rough estimation.")
        tokenizer = None
    
    # Calculate available tokens for paper content
    # Reserve some tokens for: instruction, template, and safety margin
    template_overhead = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n"
    reserve_tokens = max_output_tokens + 500  # 500 tokens buffer
    
    if tokenizer:
        instruction_tokens = len(tokenizer.encode(instruction))
        overhead_tokens = len(tokenizer.encode(template_overhead))
        available_tokens = max_model_len - instruction_tokens - overhead_tokens - reserve_tokens
        logger.info(f"Instruction uses {instruction_tokens} tokens, overhead {overhead_tokens} tokens")
        logger.info(f"Available for paper content: {available_tokens} tokens (max_input: {max_model_len}, reserved: {reserve_tokens})")
    else:
        # Rough estimation: 1 token ≈ 4 characters
        available_chars = (max_model_len - reserve_tokens) * 4 - len(instruction) - len(template_overhead)
        available_tokens = None
    
    paper_ids = []
    prompts = []
    truncated_count = 0
    
    for paper_id, grobid_file in grobid_files.items():
        # Read GROBID full text
        paper_content = read_txt_file(grobid_file)
        
        # Remove references section (common in academic papers)
        idx = paper_content.find("## References")
        if idx != -1:
            paper_content = paper_content[:idx].strip()
        
        # Truncate paper content to fit context window
        if tokenizer:
            paper_tokens = tokenizer.encode(paper_content)
            if len(paper_tokens) > available_tokens:
                # Truncate to available tokens (leave small safety margin)
                safe_tokens = int(available_tokens * 0.95)  # 95% to be safe
                paper_tokens = paper_tokens[:safe_tokens]
                paper_content = tokenizer.decode(paper_tokens, skip_special_tokens=True)
                truncated_count += 1
                logger.debug(f"Truncated {paper_id} to {safe_tokens} tokens")
        else:
            # Rough character-based truncation
            if len(paper_content) > available_chars:
                paper_content = paper_content[:int(available_chars * 0.95)]
                truncated_count += 1
                logger.debug(f"Truncated {paper_id} to ~{int(available_chars * 0.95)} chars")
        
        # Format prompt for Qwen model chat format
        # <|im_start|>system ... <|im_end|> <|im_start|>user ... <|im_end|> <|im_start|>assistant
        prompt = f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{paper_content}<|im_end|>\n<|im_start|>assistant\n"
        
        paper_ids.append(paper_id)
        prompts.append(prompt)
    
    logger.info(f"Prepared {len(paper_ids)} prompts for inference")
    if truncated_count > 0:
        logger.info(f"Truncated {truncated_count} papers to fit context window")
    return paper_ids, prompts


def run_vllm_inference(
    model_name: str,
    paper_ids: List[str],
    prompts: List[str],
    output_dir: str,
    batch_size: int = 8,
    max_tokens: int = 8192,
    temperature: float = 0.7,
    top_p: float = 0.95,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    skip_completed: bool = True,
) -> None:
    """Run vLLM inference on prompts.
    
    Args:
        model_name: HuggingFace model name or path
        paper_ids: List of paper IDs
        prompts: List of prompts
        output_dir: Directory to save outputs
        batch_size: Batch size for processing
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: GPU memory utilization fraction
        skip_completed: Skip papers already in output_dir
        
    Processes text files (.grobid.txt or .txt format).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter out already processed papers
    to_process = []
    to_process_ids = []
    
    for paper_id, prompt in zip(paper_ids, prompts):
        output_file = os.path.join(output_dir, f"{paper_id}.txt")
        if skip_completed and os.path.exists(output_file):
            logger.info(f"Skipping {paper_id} (already processed)")
            continue
        to_process.append(prompt)
        to_process_ids.append(paper_id)
    
    if not to_process:
        logger.info("All papers already processed!")
        return
    
    logger.info(f"Processing {len(to_process)} papers with vLLM")
    logger.info(f"Model: {model_name}")
    logger.info(f"Tensor Parallel Size: {tensor_parallel_size}")
    logger.info(f"GPU Memory Utilization: {gpu_memory_utilization}")
    
    # Initialize vLLM
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="auto",
        max_model_len=20480,  # Reduced to fit available KV cache on the current GPU setup
    )
    
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        skip_special_tokens=True,
    )
    
    # Process in batches
    for i in tqdm(range(0, len(to_process), batch_size), desc="Processing batches"):
        batch_prompts = to_process[i:i+batch_size]
        batch_ids = to_process_ids[i:i+batch_size]
        
        # Generate
        outputs = llm.generate(batch_prompts, sampling_params)
        
        # Save outputs
        for output, paper_id in zip(outputs, batch_ids):
            response = output.outputs[0].text.strip()
            save_output(response, output_dir, paper_id)
            logger.debug(f"Saved review for {paper_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate academic reviews using vLLM (optimized for speed)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-7B-Instruct",
        help="Model name or path (default: Qwen/Qwen2-7B-Instruct)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=os.getenv("SEA_INPUT_DIR", "data/grobid_fulltext"),
        help="Input directory with GROBID full text files (.grobid.txt)"
    )
    parser.add_argument(
        "--paper-ids",
        type=str,
        default=os.getenv("SEA_PAPER_IDS_FILE", "data/paper_ids.txt"),
        help="File with paper IDs (one per line)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.getenv("SEA_OUTPUT_DIR", "outputs/sea_reviews"),
        help="Output directory for generated reviews"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference (default: 8)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum tokens to generate (default: 8192)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling parameter (default: 0.95)"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization fraction (default: 0.9)"
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        default=True,
        help="Skip papers already in output directory"
    )
    parser.add_argument(
        "--no-skip-completed",
        action="store_false",
        dest="skip_completed",
        help="Process all papers, overwrite existing"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("vLLM Review Generation Pipeline")
    logger.info("=" * 60)
    
    try:
        # Load paper IDs
        paper_ids = get_paper_ids(args.paper_ids)
        
        # Get GROBID files
        grobid_files = get_paper_files(args.input_dir, paper_ids)
        
        if not grobid_files:
            logger.error("No GROBID files found!")
            return
        
        # Prepare prompts
        template_path = os.path.join(
            os.path.dirname(__file__),
            "paper_review",
            "template.json"
        )
        paper_ids_list, prompts = prepare_prompts(
            grobid_files, 
            template_path,
            max_model_len=20480,  # Reduced to fit available KV cache on the current GPU setup
            max_output_tokens=args.max_tokens
        )
        
        # Run inference
        run_vllm_inference(
            model_name=args.model,
            paper_ids=paper_ids_list,
            prompts=prompts,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            skip_completed=args.skip_completed,
        )
        
        logger.info("=" * 60)
        logger.info("Review generation completed!")
        logger.info(f"Output saved to: {args.output_dir}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
