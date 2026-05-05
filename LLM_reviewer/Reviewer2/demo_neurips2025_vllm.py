import argparse
import glob
import re
from pathlib import Path

import transformers

from llama_attn_replace import replace_llama_attn
from optimizations import generate_with_vllm, load_model_vllm, setup_torch_performance


SEQ_LEN = 40960
SYSTEM_PROMPT_RESERVE = 512
MAX_PAPER_TOKENS = 32000
PHASE1_MAX_TOKENS = 4096
PHASE2_MAX_TOKENS = 8192
PHASE1_MIN_TOKENS = 512
PHASE2_MIN_TOKENS = 1024
PROMPT_OVERHEAD_TOKENS = 2048
GENERATION_SAFETY_MARGIN = 100
MIN_ENGINE_MAX_MODEL_LEN = 24576
DEFAULT_ENGINE_MAX_MODEL_LEN = 30720
DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048
DEFAULT_MAX_NUM_SEQS = 32
MAX_GPU_MEMORY_UTILIZATION = 0.95


PHASE1_SYSTEM_PROMPT = (
    "You are an expert academic peer reviewer for top-tier AI/ML venues "
    "(NeurIPS, ICML, ICLR, ACL, AAAI). You have deep knowledge of machine learning, "
    "deep learning, NLP, and computer vision. Your role is to generate a precise set "
    "of deep, critical, and specific questions that will guide a thorough review of the paper. "
    "These questions must go well beyond surface-level observations and must probe the paper's technical depth, "
    "experimental rigor, and novelty. Do not output chain-of-thought, internal reasoning, or <think> tags. "
    "Return only the final requested answer."
)


PHASE2_SYSTEM_PROMPT = (
    "You are a senior expert reviewer at a top-tier AI/ML conference (NeurIPS, ICML, ICLR, ACL, or AAAI). "
    "You are known for writing thorough, technically rigorous, and constructive reviews that go well beyond superficial summaries. "
    "Your reviews are specific: you cite exact claims, equations, tables, or figures from the paper, and you back every criticism with evidence and reasoning. "
    "You are fair: you acknowledge genuine contributions before raising concerns, and you distinguish between major and minor issues. "
    "You do not write vague feedback such as 'the paper needs more experiments' without specifying exactly which experiments are missing and why they matter. "
    "Do not output chain-of-thought, internal reasoning, or <think> tags. Return only the final review."
)


CONCLUSION_HEADING_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:\d+(?:\.\d+)*\s*)?(?:conclusion(?:s)?(?:\s+and\s+limitations?)?|limitations?)\b",
    re.IGNORECASE,
)

POST_CONCLUSION_MARKERS = [
    "neurips paper checklist",
    "iclr paper checklist",
    "paper checklist",
    "reproducibility checklist",
    "references",
    "appendix",
    "appendices",
    "supplementary material",
    "supplementary materials",
    "supplemental material",
    "supplemental materials",
    "acknowledgements",
    "acknowledgments",
    "a case study",
]

TRAILING_SECTION_HEADING_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:\d+(?:\.\d+)*\s*)?(?:neurips paper checklist|iclr paper checklist|paper checklist|reproducibility checklist|references|appendix|appendices|supplementary material(?:s)?|supplemental material(?:s)?|acknowledg(?:e)?ments?)\b",
    re.IGNORECASE,
)
MIN_FALLBACK_TRIM_START_RATIO = 0.55


def parse_estimated_max_model_len(error_text: str):
    match = re.search(r"estimated maximum model length is (\d+)", error_text)
    if not match:
        return None
    return int(match.group(1))


def next_lower_max_model_len(max_model_len: int):
    next_value = max_model_len - 2048
    if next_value < MIN_ENGINE_MAX_MODEL_LEN:
        return None
    return next_value


def load_vllm_with_retry(
    model_path: str,
    requested_utilization: float,
    requested_max_model_len: int,
    requested_max_num_batched_tokens: int,
    requested_max_num_seqs: int,
):
    tried = []
    utilization = requested_utilization

    while utilization >= 0.35:
        max_model_len = requested_max_model_len
        while max_model_len >= MIN_ENGINE_MAX_MODEL_LEN:
            tried.append((utilization, max_model_len))
            try:
                print(
                    "Trying vLLM with "
                    f"gpu_memory_utilization={utilization:.2f}, max_model_len={max_model_len}, "
                    f"max_num_batched_tokens={requested_max_num_batched_tokens}, "
                    f"max_num_seqs={requested_max_num_seqs}"
                )
                return load_model_vllm(
                    model_path,
                    gpu_memory_utilization=utilization,
                    max_model_len=max_model_len,
                    max_num_batched_tokens=requested_max_num_batched_tokens,
                    max_num_seqs=requested_max_num_seqs,
                ), utilization, max_model_len
            except Exception as exc:
                error_text = str(exc)
                is_memory_pressure = (
                    "Free memory on device" in error_text
                    and "desired GPU memory utilization" in error_text
                )
                is_cache_block_error = "No available memory for the cache blocks" in error_text
                is_engine_core_init_failure = "Engine core initialization failed" in error_text
                is_sampler_warmup_oom = (
                    "CUDA out of memory occurred when warming up sampler" in error_text
                    or "Please try lowering `max_num_seqs` or `gpu_memory_utilization`" in error_text
                )

                if is_memory_pressure or is_sampler_warmup_oom:
                    print(
                        "vLLM failed to initialize due to insufficient free VRAM or sampler warmup memory pressure. "
                        f"Retrying with lower gpu_memory_utilization (last error: {error_text})"
                    )
                    break

                estimated_max_model_len = parse_estimated_max_model_len(error_text)
                if estimated_max_model_len is not None and estimated_max_model_len < max_model_len:
                    next_max_model_len = min(max_model_len - 256, estimated_max_model_len - 256)
                    if next_max_model_len < MIN_ENGINE_MAX_MODEL_LEN:
                        next_max_model_len = MIN_ENGINE_MAX_MODEL_LEN
                    print(
                        "vLLM initialized the model weights but did not have enough KV cache for the requested "
                        f"context length. Retrying with max_model_len={next_max_model_len} "
                        f"(last error: {error_text})"
                    )
                    if next_max_model_len >= max_model_len:
                        raise
                    max_model_len = next_max_model_len
                    continue

                if is_cache_block_error or is_engine_core_init_failure:
                    next_max_model_len = next_lower_max_model_len(max_model_len)
                    if next_max_model_len is not None:
                        print(
                            "vLLM engine initialization failed after model load. "
                            f"Retrying with smaller max_model_len={next_max_model_len} "
                            f"(last error: {error_text})"
                        )
                        max_model_len = next_max_model_len
                        continue

                raise

        utilization = round(utilization - 0.05, 2)

    tried_str = ", ".join(f"({util:.2f}, {max_len})" for util, max_len in tried)
    raise RuntimeError(
        "vLLM failed to initialize after retrying (gpu_memory_utilization, max_model_len) values: "
        f"{tried_str}. Use a less busy GPU or lower the requested utilization further."
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Reviewer2 NeurIPS2025 vLLM Batch Processing")
    parser.add_argument(
        "--grobid_dir",
        type=str,
        default="/datastore/npl/luannt/IHSD/Reviewer2/NeurIPS2025/grobid_fulltext",
        help="Directory containing full-text files (.grobid.txt and/or .txt)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/datastore/npl/luannt/IHSD/Reviewer2/output_reviewer2_neurips2025",
        help="Directory to save .txt reviews",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/datastore/npl/luannt/.cache/huggingface/Qwen3-14B",
        help="Path to the base model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="vLLM batch size",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit for the number of papers to process",
    )
    parser.add_argument(
        "--paper_ids",
        type=str,
        default="/datastore/npl/luannt/IHSD/Reviewer2/NeurIPS2025/data_subset/paper_ids.txt",
        help="Optional path to a paper_ids.txt file for filtering",
    )
    parser.add_argument(
        "--error_ids_output",
        type=str,
        default=None,
        help="Optional path to a .txt file where missing or empty-review paper IDs will be written after the run.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.92,
        help="vLLM GPU memory utilization",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=DEFAULT_ENGINE_MAX_MODEL_LEN,
        help="Initial vLLM max_model_len to request. Lower values reduce KV-cache pressure.",
    )
    parser.add_argument(
        "--max_num_batched_tokens",
        type=int,
        default=DEFAULT_MAX_NUM_BATCHED_TOKENS,
        help="Initial vLLM max_num_batched_tokens. Lower values reduce batching memory pressure.",
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=DEFAULT_MAX_NUM_SEQS,
        help="Initial vLLM max_num_seqs. Lower values reduce sampler warmup memory pressure.",
    )
    parser.add_argument(
        "--force_reprocess",
        action="store_true",
        help="Overwrite existing outputs",
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="Process files even if output does not exist check would normally skip",
    )
    return parser.parse_args()


def count_tokens(text, tokenizer):
    tokenized = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        verbose=False,
    )
    return len(tokenized["input_ids"])


def calculate_dynamic_paper_token_limit(max_model_len: int) -> int:
    return min(MAX_PAPER_TOKENS, max_model_len - PHASE2_MAX_TOKENS - PROMPT_OVERHEAD_TOKENS)


def calculate_safe_max_tokens(prompts, tokenizer, max_model_len: int, requested_max_tokens: int) -> int:
    if not prompts:
        return 0

    prompt_lengths = [count_tokens(prompt, tokenizer) for prompt in prompts]
    available_by_prompt = [max_model_len - prompt_length - SYSTEM_PROMPT_RESERVE for prompt_length in prompt_lengths]
    safe_max_tokens = min(requested_max_tokens, min(available_by_prompt))
    return max(0, safe_max_tokens)


def get_paper_id(file_path: str) -> str:
    name = Path(file_path).name
    if name.endswith(".grobid.txt"):
        return name[: -len(".grobid.txt")]
    if name.endswith(".txt"):
        return name[: -len(".txt")]
    return Path(file_path).stem


def collect_input_files(input_dir: Path):
    file_by_id = {}
    for path_str in sorted(glob.glob(str(input_dir / "*.grobid.txt"))):
        file_by_id[get_paper_id(path_str)] = path_str
    for path_str in sorted(glob.glob(str(input_dir / "*.txt"))):
        paper_id = get_paper_id(path_str)
        if paper_id not in file_by_id:
            file_by_id[paper_id] = path_str
    return [file_by_id[paper_id] for paper_id in sorted(file_by_id)]


def normalize_line(line: str) -> str:
    line = line.replace("\x00", " ")
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def clean_grobid_text(text: str) -> str:
    lines = []
    for raw_line in text.splitlines():
        line = normalize_line(raw_line)
        if not line:
            lines.append("")
            continue
        if re.fullmatch(r"\d+", line):
            continue
        if re.fullmatch(r"page \d+", line, flags=re.IGNORECASE):
            continue
        lines.append(line)

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def extract_title(text: str) -> str:
    for line in text.splitlines():
        line = normalize_line(line)
        if not line:
            continue
        if len(line.split()) < 3:
            continue
        return line.lstrip("# ")
    return "N/A"


def extract_abstract(text: str) -> str:
    pattern = re.compile(
        r"(?:^|\n)abstract\s*\n+(.*?)(?=\n(?:keywords|introduction|1\s+introduction|background|related work|methods?)\b|\n[A-Z][A-Z\s]{4,}\n|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return "N/A"
    abstract = match.group(1).strip()
    abstract = re.sub(r"\n{2,}", "\n", abstract)
    return abstract if abstract else "N/A"


def remove_appendix_or_supplement(text: str):
    pattern = re.compile(
        r"^\s*(?:appendix|appendices|supplementary material|supplementary materials|supplemental material|supplemental materials|a\.?\s+appendix)\b",
        re.IGNORECASE | re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        return text, False, 0
    main_text = text[: match.start()].rstrip()
    removed = text[match.start() :]
    return main_text, True, len(removed)


def truncate_paper_content(paper_content: str, tokenizer, max_input_tokens: int):
    token_count = count_tokens(paper_content, tokenizer)
    if token_count <= max_input_tokens:
        return paper_content, token_count, {"was_truncated": False, "method": "none"}

    text_no_appendix, appendix_removed, removed_chars = remove_appendix_or_supplement(paper_content)
    if appendix_removed:
        token_count = count_tokens(text_no_appendix, tokenizer)
        if token_count <= max_input_tokens:
            return text_no_appendix, token_count, {
                "was_truncated": True,
                "method": "appendix_removed",
                "removed_chars": removed_chars,
            }
        paper_content = text_no_appendix

    lines = paper_content.split("\n")
    kept_lines = []
    current_tokens = 0
    for line in lines:
        line_tokens = count_tokens(line + "\n", tokenizer)
        if current_tokens + line_tokens > max_input_tokens:
            kept_lines.append("[... CONTENT TRUNCATED DUE TO LENGTH ...]")
            break
        kept_lines.append(line)
        current_tokens += line_tokens

    truncated_content = "\n".join(kept_lines).strip()
    final_tokens = count_tokens(truncated_content, tokenizer)
    method = "appendix_removed + line_truncation" if appendix_removed else "line_truncation"
    return truncated_content, final_tokens, {
        "was_truncated": True,
        "method": method,
        "final_tokens": final_tokens,
    }


def trim_after_conclusion(text: str) -> str:
    conclusion_match = None
    for match in CONCLUSION_HEADING_PATTERN.finditer(text):
        conclusion_match = match

    if conclusion_match is None:
        return trim_after_known_trailing_markers(text)

    suffix = text[conclusion_match.end() :]
    cut_offset = find_post_conclusion_cut_offset(suffix)
    if cut_offset is None:
        return text

    trimmed = text[: conclusion_match.end() + cut_offset].rstrip()
    return trimmed if trimmed else text


def trim_after_known_trailing_markers(text: str) -> str:
    cut_positions = []
    min_start = int(len(text) * MIN_FALLBACK_TRIM_START_RATIO)
    for match in TRAILING_SECTION_HEADING_PATTERN.finditer(text):
        if match.start() >= min_start:
            cut_positions.append(match.start())
    if not cut_positions:
        return text
    return text[: min(cut_positions)].rstrip()


def find_post_conclusion_cut_offset(text: str):
    cut_positions = []

    for match in TRAILING_SECTION_HEADING_PATTERN.finditer(text):
        cut_positions.append(match.start())

    if not cut_positions:
        return None
    return min(cut_positions)


def build_paper_content(raw_text: str, tokenizer, max_paper_tokens: int):
    cleaned_text = clean_grobid_text(raw_text)
    cleaned_text = trim_after_conclusion(cleaned_text)
    title = extract_title(cleaned_text)
    abstract = extract_abstract(cleaned_text)
    paper_content = "\n".join([
        "Title",
        title,
        "Abstract",
        abstract,
        "Full Text",
        cleaned_text,
    ]).strip()
    token_count = count_tokens(paper_content, tokenizer)
    truncation_info = {"was_truncated": False, "method": "none"}
    if token_count > max_paper_tokens:
        paper_content, token_count, truncation_info = truncate_paper_content(
            paper_content, tokenizer, max_paper_tokens
        )
    return {
        "title": title,
        "abstract": abstract,
        "paper_content": paper_content,
        "token_count": token_count,
        "truncation_info": truncation_info,
    }


def _check_acknowledgements(text):
    hit = re.search(r"\b(acknowledgement|acknowledgment|funded by|supported by|grant)\b", text, re.IGNORECASE)
    if not hit:
        return []
    snippet = text[max(0, hit.start() - 30) : hit.start() + 80].replace("\n", " ")
    return [f"Acknowledgement-related text found: '...{snippet}...'"]


def _check_self_citation(text):
    hits = re.findall(r"\b(our previous work|in our earlier|we showed in \[|we proposed in \[|in our work \[)\b", text, re.IGNORECASE)
    return [f"Potential self-citation phrase: '{hit}'" for hit in hits[:3]]


def _check_identifying_links(text):
    urls = re.findall(r"https?://\S+", text)
    emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    issues = [f"Email address found in text: {email}" for email in emails[:5]]
    if not issues and urls:
        return []
    return issues


def _check_section_present(text, keywords, label):
    for keyword in keywords:
        if re.search(rf"\b{re.escape(keyword)}\b", text, re.IGNORECASE):
            return []
    return [f"{label} section may be missing"]


def build_compliance_guidance(text: str) -> str:
    issues = []
    issues.extend(_check_acknowledgements(text))
    issues.extend(_check_self_citation(text))
    issues.extend(_check_identifying_links(text))
    issues.extend(_check_section_present(text, ["reproducibility", "reproducible"], "Reproducibility"))
    issues.extend(_check_section_present(text, ["ethics", "ethical", "broader impact"], "Ethics"))
    if not issues:
        return ""
    guidance = ["[Reviewer Guidance - Potential Submission Format Issues]"]
    for issue in issues:
        guidance.append(f"- {issue}")
    guidance.append("")
    guidance.append("Please assess how these issues impact the paper's quality and contribution.")
    return "\n".join(guidance)


def build_chat_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        try:
            return tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        except ImportError:
            pass

    if getattr(tokenizer, "eos_token", None) == "<|im_end|>":
        return (
            "<|im_start|>system\n"
            f"{system_prompt}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    return (
        "[INST] <<SYS>>\n"
        f"{system_prompt}\n"
        "<</SYS>>\n"
        f"{user_prompt}\n"
        "[/INST]"
    )


def build_phase1_prompt(paper_content: str, tokenizer) -> str:
    user_prompt = (
        "Read the following paper carefully and in full:\n\n"
        f"{paper_content}\n\n"
        "---\n"
        "Generate a set of SPECIFIC, CRITICAL questions to guide a thorough review of this paper. "
        "Each question must be grounded in the actual content of the paper (cite specific sections, equations, tables, or claims). "
        "The questions should probe the following dimensions:\n"
        "1. Technical correctness: Are the core claims, proofs, and derivations technically sound? Are there errors or unvalidated assumptions?\n"
        "2. Experimental rigor: Are the baselines appropriate and up-to-date? Are ablation studies sufficient? Are results statistically validated?\n"
        "3. Novelty and contribution: What is the precise novel contribution beyond prior work? Is the contribution clearly distinguished?\n"
        "4. Clarity and reproducibility: Are all hyperparameters, datasets, and training details specified for replication?\n"
        "5. Scope and limitations: Are the limitations honestly discussed? Do the conclusions overgeneralize the results?\n\n"
        "The reviewer should structure their response in the following sections:\n"
        "Summary Of The Paper\n"
        "Strengths And Weaknesses\n"
        "Detailed Technical Questions\n"
        "Experimental Questions\n"
        "Questions About Novelty And Related Work"
    )
    return build_chat_prompt(tokenizer, PHASE1_SYSTEM_PROMPT, user_prompt)


def build_phase2_prompt(paper_content: str, compliance_guidance: str, questions: str, tokenizer) -> str:
    compliance_block = f"{compliance_guidance}\n\n" if compliance_guidance else ""
    user_prompt = (
        "Read the following paper carefully and in full:\n\n"
        f"{paper_content}\n\n"
        f"{compliance_block}"
        "---\n"
        "You have been guided by the following critical analysis questions:\n"
        f"{questions}\n\n"
        "---\n"
        "Using your analysis, write a DETAILED, SPECIFIC, and RIGOROUS peer review. "
        "Your review must:\n"
        "- Summarize the paper's claims and methods in your own words.\n"
        "- List concrete, evidence-backed strengths.\n"
        "- List concrete, evidence-backed weaknesses or concerns, ordered by severity.\n"
        "- Ask precise, answerable questions to the authors.\n"
        "- State clear limitations the authors have not addressed.\n"
        "- Provide a recommendation with justification.\n\n"
        "Write your complete review strictly in the following sections:\n"
        "Summary Of The Paper\n"
        "Strengths\n"
        "Weaknesses\n"
        "Questions For The Authors\n"
        "Limitations Not Addressed By The Authors\n"
        "Soundness (1-4: 1=Poor, 2=Fair, 3=Good, 4=Excellent)\n"
        "Contribution (1-4)\n"
        "Confidence (1-5: 1=Not sure, 5=Expert)\n"
        "Rating (1-10: 1=Reject, 5=Borderline, 8=Accept, 10=Strong Accept)\n"
        "Brief Justification For Rating"
    )
    return build_chat_prompt(tokenizer, PHASE2_SYSTEM_PROMPT, user_prompt)


def normalize_generated_text(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""

    if "</think>" in cleaned:
        final_answer = cleaned.split("</think>", 1)[1].strip()
        if final_answer:
            cleaned = final_answer

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()

    return cleaned


def is_effectively_empty(text: str, min_non_ws_chars: int = 32) -> bool:
    return len(re.sub(r"\s+", "", text or "")) < min_non_ws_chars


def retry_empty_generations(
    llm,
    prompts,
    outputs,
    tokenizer,
    effective_max_model_len: int,
    requested_max_tokens: int,
    phase_name: str,
):
    retried_outputs = list(outputs)
    empty_indices = [index for index, output in enumerate(outputs) if is_effectively_empty(output)]

    for index in empty_indices:
        safe_max_tokens = calculate_safe_max_tokens(
            [prompts[index]], tokenizer, effective_max_model_len, requested_max_tokens
        )
        if safe_max_tokens <= 0:
            print(f"  ! {phase_name} retry skipped for item {index + 1}: no generation budget left")
            continue

        print(
            f"  ! Retrying empty {phase_name} output for item {index + 1} "
            f"with max_tokens={safe_max_tokens}"
        )
        retry_output = generate_with_vllm(
            llm,
            [prompts[index]],
            max_tokens=safe_max_tokens,
            temperature=0.3,
            top_p=0.9,
            top_k=20,
            repetition_penalty=1.05,
        )[0]
        retried_outputs[index] = normalize_generated_text(retry_output)

    return retried_outputs


def fallback_phase1_questions() -> str:
    return "\n".join(
        [
            "Summary Of The Paper",
            "Strengths And Weaknesses",
            "Detailed Technical Questions",
            "1. Which core claims or derivations appear insufficiently justified, and where in the paper do those issues appear?",
            "2. Which experimental comparisons or ablations are missing to validate the main method against credible alternatives?",
            "3. What is the exact novelty over the most relevant prior work, and is that distinction supported by the paper's evidence?",
            "4. Which implementation, training, or evaluation details are missing for reproducibility?",
            "Experimental Questions",
            "Questions About Novelty And Related Work",
        ]
    )


def load_paper_filter(paper_ids_path: str):
    if not paper_ids_path:
        return None
    path = Path(paper_ids_path)
    if not path.exists():
        raise FileNotFoundError(f"paper_ids file not found: {paper_ids_path}")
    with open(path, "r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def save_review(output_dir: Path, paper_id: str, title: str, review: str):
    output_path = output_dir / f"{paper_id}.txt"
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(f"PAPER: {title}\n")
        handle.write(f"{'=' * 80}\n\n")
        handle.write("REVIEW\n")
        handle.write(f"{'-' * 80}\n")
        handle.write(review)
        handle.write(f"\n\n{'=' * 80}\n")


def extract_review_body(text: str) -> str:
    lines = text.splitlines()
    in_review = False
    review_lines = []

    for line in lines:
        stripped = line.strip()
        if not in_review:
            if stripped == "REVIEW":
                in_review = True
            continue

        if re.fullmatch(r"=+", stripped):
            break
        if not review_lines and re.fullmatch(r"-+", stripped):
            continue

        review_lines.append(line)

    return "\n".join(review_lines).strip()


def review_file_has_content(output_path: Path) -> bool:
    if not output_path.exists():
        return False

    with open(output_path, "r", encoding="utf-8", errors="ignore") as handle:
        review_body = extract_review_body(handle.read())

    return not is_effectively_empty(review_body)


def resolve_error_ids_output_path(explicit_path: str, paper_ids_path: str, output_dir: Path) -> Path:
    if explicit_path:
        return Path(explicit_path)

    if paper_ids_path:
        paper_ids = Path(paper_ids_path)
        if paper_ids.suffix == ".txt":
            if paper_ids.stem.endswith("_errors"):
                return paper_ids
            return paper_ids.with_name(f"{paper_ids.stem}_errors.txt")
        return paper_ids.with_name(f"{paper_ids.name}_errors.txt")

    return output_dir / "paper_ids_errors.txt"


def write_error_ids(error_ids_path: Path, paper_ids):
    error_ids_path.parent.mkdir(parents=True, exist_ok=True)
    with open(error_ids_path, "w", encoding="utf-8") as handle:
        for paper_id in sorted({paper_id for paper_id in paper_ids if paper_id}):
            handle.write(f"{paper_id}\n")


def iter_batches(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def main():
    args = parse_args()
    skip_existing = not args.no_skip_existing

    if args.gpu_memory_utilization <= 0:
        raise ValueError("gpu_memory_utilization must be positive")
    if args.gpu_memory_utilization > MAX_GPU_MEMORY_UTILIZATION:
        print(
            f"Requested gpu_memory_utilization is above {MAX_GPU_MEMORY_UTILIZATION}; "
            f"clamping to {MAX_GPU_MEMORY_UTILIZATION}"
        )
        args.gpu_memory_utilization = MAX_GPU_MEMORY_UTILIZATION
    if args.max_model_len <= 0:
        raise ValueError("max_model_len must be positive")
    if args.max_num_batched_tokens <= 0:
        raise ValueError("max_num_batched_tokens must be positive")
    if args.max_num_seqs <= 0:
        raise ValueError("max_num_seqs must be positive")

    setup_torch_performance()
    replace_llama_attn(inference=True)

    config = transformers.AutoConfig.from_pretrained(args.model_path)
    model_max_pos = getattr(config, "max_position_embeddings", SEQ_LEN)
    effective_seq_len = min(SEQ_LEN, model_max_pos)
    requested_engine_max_model_len = min(args.max_model_len, effective_seq_len)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length=effective_seq_len,
        padding_side="right",
        use_fast=False,
    )

    print(f"Model context window: {model_max_pos} tokens")
    print(f"Using effective sequence length: {effective_seq_len} tokens")
    print(f"Requested gpu_memory_utilization: {args.gpu_memory_utilization}")
    print(f"Requested vLLM max_model_len: {requested_engine_max_model_len}")
    print(f"Requested vLLM max_num_batched_tokens: {args.max_num_batched_tokens}")
    print(f"Requested vLLM max_num_seqs: {args.max_num_seqs}")
    print(f"Loading vLLM model with batch_size={args.batch_size} ...")
    llm, effective_gpu_memory_utilization, effective_max_model_len = load_vllm_with_retry(
        args.model_path,
        args.gpu_memory_utilization,
        requested_engine_max_model_len,
        args.max_num_batched_tokens,
        args.max_num_seqs,
    )
    print(f"Using effective gpu_memory_utilization: {effective_gpu_memory_utilization}")
    print(f"Using effective vLLM max_model_len: {effective_max_model_len}")

    dynamic_paper_token_limit = calculate_dynamic_paper_token_limit(effective_max_model_len)
    if dynamic_paper_token_limit <= 0:
        raise RuntimeError(
            f"Computed paper token budget is invalid: {dynamic_paper_token_limit}. "
            "Increase available GPU memory or lower generation requirements."
        )
    print(f"Using dynamic paper token limit: {dynamic_paper_token_limit}")

    grobid_dir = Path(args.grobid_dir)
    if not grobid_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.grobid_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    error_ids_output_path = resolve_error_ids_output_path(args.error_ids_output, args.paper_ids, output_dir)

    paper_filter = load_paper_filter(args.paper_ids)
    input_files = collect_input_files(grobid_dir)
    if paper_filter is not None:
        input_files = [path for path in input_files if get_paper_id(path) in paper_filter]
    if args.limit > 0:
        input_files = input_files[: args.limit]

    print(f"Found {len(input_files)} input papers")
    print(f"Input:  {args.grobid_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Error IDs Output: {error_ids_output_path}")
    print(f"Skip existing: {skip_existing and not args.force_reprocess}")

    pending_files = []
    skipped_count = 0
    for file_path in input_files:
        paper_id = get_paper_id(file_path)
        output_path = output_dir / f"{paper_id}.txt"
        if output_path.exists() and not args.force_reprocess and skip_existing:
            skipped_count += 1
            continue
        pending_files.append(file_path)

    print(f"Pending for processing: {len(pending_files)}")
    print(f"Skipped existing: {skipped_count}")

    success_count = 0
    failed = []
    expected_paper_ids = [get_paper_id(path) for path in input_files]

    for batch_index, batch_files in enumerate(iter_batches(pending_files, args.batch_size), start=1):
        batch_records = []
        for file_path in batch_files:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
                    raw_text = handle.read()
                parsed = build_paper_content(raw_text, tokenizer, dynamic_paper_token_limit)
                batch_records.append(
                    {
                        "file_path": file_path,
                        "paper_id": get_paper_id(file_path),
                        "title": parsed["title"],
                        "paper_content": parsed["paper_content"],
                        "token_count": parsed["token_count"],
                        "truncation_info": parsed["truncation_info"],
                        "compliance_guidance": build_compliance_guidance(parsed["paper_content"]),
                    }
                )
            except Exception as exc:
                failed.append((Path(file_path).name, f"preprocess error: {exc}"))

        if not batch_records:
            continue

        print(f"\n[BATCH {batch_index}] Preparing {len(batch_records)} papers")
        for record in batch_records:
            truncation = record["truncation_info"]
            if truncation["was_truncated"]:
                print(
                    f"  - {record['paper_id']}: truncated via {truncation['method']} to {record['token_count']} tokens"
                )
            else:
                print(f"  - {record['paper_id']}: {record['token_count']} tokens")

        try:
            phase1_prompts = [build_phase1_prompt(record["paper_content"], tokenizer) for record in batch_records]
            phase1_safe_max_tokens = calculate_safe_max_tokens(
                phase1_prompts,
                tokenizer,
                effective_max_model_len,
                PHASE1_MAX_TOKENS,
            )
            if phase1_safe_max_tokens < PHASE1_MIN_TOKENS:
                raise RuntimeError(
                    f"phase1 prompt too long: insufficient generation budget ({phase1_safe_max_tokens} tokens)"
                )
            phase1_outputs = generate_with_vllm(
                llm,
                phase1_prompts,
                max_tokens=phase1_safe_max_tokens,
                temperature=0.7,
                top_p=0.7,
                top_k=50,
                repetition_penalty=1.13,
            )
            phase1_outputs = [normalize_generated_text(output) for output in phase1_outputs]
            phase1_outputs = retry_empty_generations(
                llm,
                phase1_prompts,
                phase1_outputs,
                tokenizer,
                effective_max_model_len,
                phase1_safe_max_tokens,
                "phase1",
            )
        except Exception as exc:
            for record in batch_records:
                failed.append((Path(record["file_path"]).name, f"phase1 generation error: {exc}"))
            print(f"  ! Batch {batch_index} phase1 failed: {exc}")
            continue

        phase2_prompts = []
        for record, questions in zip(batch_records, phase1_outputs):
            if is_effectively_empty(questions):
                questions = fallback_phase1_questions()
            phase2_prompts.append(
                build_phase2_prompt(record["paper_content"], record["compliance_guidance"], questions, tokenizer)
            )

        try:
            phase2_safe_max_tokens = calculate_safe_max_tokens(
                phase2_prompts,
                tokenizer,
                effective_max_model_len,
                PHASE2_MAX_TOKENS,
            )
            if phase2_safe_max_tokens < PHASE2_MIN_TOKENS:
                raise RuntimeError(
                    f"phase2 prompt too long: insufficient generation budget ({phase2_safe_max_tokens} tokens)"
                )
            phase2_outputs = generate_with_vllm(
                llm,
                phase2_prompts,
                max_tokens=phase2_safe_max_tokens,
                temperature=0.7,
                top_p=0.7,
                top_k=50,
                repetition_penalty=1.13,
            )
            phase2_outputs = [normalize_generated_text(output) for output in phase2_outputs]
            phase2_outputs = retry_empty_generations(
                llm,
                phase2_prompts,
                phase2_outputs,
                tokenizer,
                effective_max_model_len,
                phase2_safe_max_tokens,
                "phase2",
            )
        except Exception as exc:
            for record in batch_records:
                failed.append((Path(record["file_path"]).name, f"phase2 generation error: {exc}"))
            print(f"  ! Batch {batch_index} phase2 failed: {exc}")
            continue

        for record, review in zip(batch_records, phase2_outputs):
            try:
                if is_effectively_empty(review):
                    failed.append((Path(record["file_path"]).name, "empty review after retry"))
                    print(f"  ✗ Empty review after retry: {record['paper_id']}")
                    continue

                save_review(output_dir, record["paper_id"], record["title"], review)
                success_count += 1
                print(f"  ✓ Saved: {record['paper_id']}.txt")
            except Exception as exc:
                failed.append((Path(record["file_path"]).name, f"save error: {exc}"))

    print("\n" + "=" * 70)
    print("NEURIPS2025 BATCH PROCESSING COMPLETE")
    print("=" * 70)
    print(f"✓ Successful: {success_count}")
    print(f"✗ Failed: {len(failed)}")
    print(f"⊘ Skipped existing: {skipped_count}")

    failed_ids = {get_paper_id(file_name) for file_name, _ in failed}
    missing_output_ids = []
    empty_review_ids = []
    for paper_id in expected_paper_ids:
        output_path = output_dir / f"{paper_id}.txt"
        if not output_path.exists():
            missing_output_ids.append(paper_id)
            continue
        if not review_file_has_content(output_path):
            empty_review_ids.append(paper_id)

    error_paper_ids = failed_ids | set(missing_output_ids) | set(empty_review_ids)
    write_error_ids(error_ids_output_path, error_paper_ids)

    print(f"! Missing outputs after run: {len(missing_output_ids)}")
    print(f"! Empty review outputs after run: {len(empty_review_ids)}")
    print(f"! Error paper IDs written: {len(error_paper_ids)}")
    if failed:
        print("\nFailed files:")
        for file_name, error in failed[:50]:
            print(f"  - {file_name}: {error}")
        if len(failed) > 50:
            print(f"  ... and {len(failed) - 50} more")
    if error_paper_ids:
        print(f"\nError paper IDs saved to: {error_ids_output_path}")
    print(f"\nReviews saved to: {args.output_dir}\n")


if __name__ == "__main__":
    main()