import sys
import json
import math
import os
import re
import torch
import argparse
import transformers
from llama_attn_replace import replace_llama_attn
from optimizations import RateLimiter, load_model_vllm, generate_with_vllm, setup_torch_performance  # Rate limiter for API calls
import glob
from pathlib import Path
from typing import List, Tuple, Optional

# ==================== TOKEN CONFIGURATION ====================
# CRITICAL: Understanding token budgets in this pipeline:
#
# Qwen3-14B actual max_position_embeddings: 40960 tokens
# SEQ_LEN = 40960 (actual model capacity, from config.json)
#
# When generating text:
#   input tokens = tokens in the prompt (encoded by tokenizer)
#   output tokens (max_new_tokens) = tokens the model generates (up to the limit)
#
# IMPORTANT: If paper content + prompts > 30K tokens:
#   - Paper will be TRUNCATED to preserve output space
#   - max_new_tokens adjusted: (40960 - input_tokens - 512 buffer)
#
# Token budget per paper:
#   input_tokens + max_new_tokens + 512(buffer) <= 40960
#   └─ If input > 32000: truncate paper content
#   └─ Remaining space used for output (min 4096)
#
# ============================================================

# Qwen3-14B max capacity
SEQ_LEN = 40960
# Reserved for system prompt + formatting
SYSTEM_PROMPT_RESERVE = 512
# Minimum output tokens to maintain quality
MIN_OUTPUT_TOKENS = 4096


def parse_args():
    parser = argparse.ArgumentParser(description="Reviewer2 Demo - Batch Processing")
    parser.add_argument('--json_path', type=str, default='', help="path to single paper json file")
    parser.add_argument('--mmd_dir', type=str, default=os.getenv("REVIEWER2_MMD_DIR", "data/paper_nougat_mmd"), 
                        help="directory with .mmd files to batch process")
    parser.add_argument('--output_dir', type=str, default=os.getenv("REVIEWER2_OUTPUT_DIR", "outputs/reviewer2"),
                        help="directory to save .txt reviews")
    parser.add_argument('--paper_ids', type=str, default='', 
                        help="Path to paper_ids.txt file to filter which .mmd files to process (optional)")
    parser.add_argument('--use_vllm', action='store_true', help="Use vLLM for faster inference (experimental)")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Batch size (only works with vLLM)")
    parser.add_argument('--skip_existing', action='store_true',
                        help="Skip papers that already have output files (resume interrupted batch)")
    parser.add_argument('--force_reprocess', action='store_true',
                        help="Force reprocess all papers, overwriting existing outputs")
    return parser.parse_args()


def build_generator(
    model, tokenizer, temperature=0.7, top_p=0.7, top_k=50, max_new_tokens=8192, min_new_tokens=64, repetition_penalty=1.13, max_length=32768
):
    def response(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        output = model.generate(
            **inputs,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            max_length=max_length,
        )
        
        out = tokenizer.decode(output[0], skip_special_tokens=True)

        try:
            out = out.split(prompt.lstrip("<s>"))[1].strip()
        except:
            out = []

        return out

    return response


def build_vllm_generator(
    llm, temperature=0.7, top_p=0.7, top_k=50, max_new_tokens=8192, repetition_penalty=1.13
):
    """Build a generator for vLLM model (single inference)."""
    def response(prompt):
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
        
        outputs = llm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        # Try to extract only the model's response (not the prompt)
        try:
            # Remove the prompt part if present
            if "[/INST]" in prompt:
                generated_text = generated_text.split("[/INST]")[-1].strip()
        except:
            pass
        
        return generated_text
    
    return response


# ==================== TOKEN MANAGEMENT & PAPER TRUNCATION ====================

def count_tokens(text, tokenizer):
    """Count tokens in text without adding special tokens."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def remove_appendix(paper_content: str, tokenizer) -> tuple:
    """
    Remove content after '## Appendix' section.
    
    Returns: (content_without_appendix, was_removed, removed_char_count)
    """
    # Look for ## Appendix or other appendix markers
    appendix_pattern = re.compile(r'^##\s+(Appendix|APPENDIX|Appendices|APPENDICES|A\s+)', re.MULTILINE | re.IGNORECASE)
    
    match = appendix_pattern.search(paper_content)
    if not match:
        return paper_content, False, 0
    
    # Split at appendix marker
    main_content = paper_content[:match.start()].rstrip()
    removed_content = paper_content[match.start():]
    
    return main_content, True, len(removed_content)


def truncate_paper_content(paper_content: str, tokenizer, max_input_tokens: int) -> tuple:
    """
    Truncate paper content intelligently to fit within max_input_tokens.
    
    Strategy:
    1. First, try removing content after '## Appendix' (saves many tokens!)
    2. If still too long, truncate line-by-line from the end
    
    Returns: (truncated_content, token_count, truncation_info_dict)
    """
    token_count = count_tokens(paper_content, tokenizer)
    
    if token_count <= max_input_tokens:
        return paper_content, token_count, {"was_truncated": False, "method": "none"}
    
    # ===== STRATEGY 1: Remove Appendix =====
    content_no_appendix, appendix_removed, appendix_length = remove_appendix(paper_content, tokenizer)
    if appendix_removed:
        token_count = count_tokens(content_no_appendix, tokenizer)
        if token_count <= max_input_tokens:
            return content_no_appendix, token_count, {
                "was_truncated": True, 
                "method": "appendix_removed",
                "removed_chars": appendix_length
            }
        # Still too long, continue to Strategy 2
        paper_content = content_no_appendix
    
    # ===== STRATEGY 2: Line-by-line truncation =====
    lines = paper_content.split('\n')
    truncated = []
    current_tokens = 0
    
    for line in lines:
        line_tokens = count_tokens(line, tokenizer)
        if current_tokens + line_tokens <= max_input_tokens:
            truncated.append(line)
            current_tokens += line_tokens
        else:
            # Add truncation marker
            truncated.append("\n[... CONTENT TRUNCATED DUE TO LENGTH ...]")
            break
    
    truncated_content = '\n'.join(truncated)
    final_count = count_tokens(truncated_content, tokenizer)
    
    truncation_method = "appendix_removed + line_truncation" if appendix_removed else "line_truncation"
    
    return truncated_content, final_count, {
        "was_truncated": True,
        "method": truncation_method,
        "final_tokens": final_count
    }


def calculate_max_new_tokens(input_token_count: int, base_max_output: int) -> int:
    """
    Calculate safe max_new_tokens based on available space.
    Prevents OOM by respecting total token budget.
    
    Returns: safe max_new_tokens value
    """
    available = SEQ_LEN - input_token_count - SYSTEM_PROMPT_RESERVE
    safe_max = min(base_max_output, max(MIN_OUTPUT_TOKENS, available - 100))
    return max(MIN_OUTPUT_TOKENS, safe_max)



# ==================== MMD to JSON Converter ====================
def _clean_body(text: str) -> str:
    """Remove Nougat artefacts (lone figure labels, page-break lines, etc.)."""
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if re.match(r'^(Figure|Table|Fig\.?)\s+\d+', stripped) and len(stripped.split()) <= 8:
            continue
        cleaned.append(line)
    return '\n'.join(cleaned).strip()


def _build_metadata(title, abstract_text, sections):
    return {
        "metadata": {
            "title": title,
            "abstractText": abstract_text,
            "sections": sections
        }
    }


def parse_mmd(text: str) -> dict:
    """Parse a Nougat MMD string into {title, abstractText, sections}."""
    lines = text.splitlines()

    # ── 1. Title: first line starting with a single '# ' ──────────────────
    title = "N/A"
    title_line_idx = 0
    for i, line in enumerate(lines):
        if re.match(r'^#\s+\S', line):
            title = line.lstrip('#').strip()
            title_line_idx = i
            break

    # ── 2. Split remaining text into blocks at every heading ───────────────
    remaining = '\n'.join(lines[title_line_idx + 1:])
    heading_pattern = re.compile(r'^(#{1,6})\s+(.+)', re.MULTILINE)
    splits = list(heading_pattern.finditer(remaining))

    abstract_text = "N/A"
    sections = []

    if not splits:
        abstract_text = remaining.strip() or "N/A"
        return _build_metadata(title, abstract_text, sections)

    for idx, match in enumerate(splits):
        hashes = match.group(1)
        heading = match.group(2).strip()
        start = match.end()
        end = splits[idx + 1].start() if idx + 1 < len(splits) else len(remaining)

        body = remaining[start:end].strip()
        body = _clean_body(body)

        if re.search(r'\bAbstract\b', heading, re.IGNORECASE) and len(hashes) >= 3:
            abstract_text = body if body else "N/A"
        else:
            sections.append({
                "heading": heading,
                "text": body if body else "N/A"
            })

    return _build_metadata(title, abstract_text, sections)


# ==================== END MMD Converter ====================


if __name__ == '__main__':

    args = parse_args()
    
    # Setup PyTorch performance
    setup_torch_performance()

    # prep model
    replace_llama_attn(inference=True)

    # ==========Model Loading: vLLM vs Transformers===========
    MODEL_PATH = '/datastore/npl/luannt/.cache/huggingface/Qwen3-14B'
    
    # Load config and check context length
    config = transformers.AutoConfig.from_pretrained(MODEL_PATH)
    model_max_pos = getattr(config, "max_position_embeddings", 40960)
    
    print(f"Model context window: {model_max_pos} tokens")
    print(f"Using SEQ_LEN: {SEQ_LEN} tokens")
    
    if SEQ_LEN > model_max_pos:
        print(f"\n⚠️  WARNING: SEQ_LEN ({SEQ_LEN}) > model max ({model_max_pos})")
        print(f"    Using model max instead: {model_max_pos}")
        SEQ_LEN = model_max_pos

    # Load tokenizer (needed for both vLLM and Transformers)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_PATH,
        model_max_length=SEQ_LEN,
        padding_side="right",
        use_fast=False,
    )
    
    # Load model based on user choice
    use_vllm = args.use_vllm
    model = None
    llm = None
    
    if use_vllm:
        print(f"\n🚀 Loading model with vLLM (batch_size={args.batch_size})...")
        try:
            llm = load_model_vllm(MODEL_PATH)
            if llm is None:
                print("⚠️  vLLM load failed, falling back to Transformers")
                use_vllm = False
            else:
                print(f"✓ vLLM model loaded successfully")
                print(f"  Using batch_size: {args.batch_size}")
        except Exception as e:
            print(f"⚠️  vLLM load failed: {e}, falling back to Transformers")
            use_vllm = False
    
    if not use_vllm:
        print(f"\n📦 Loading model with Transformers...")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model.generation_config.max_length = SEQ_LEN
        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)
        print(f"✓ Transformers model loaded: {MODEL_PATH}")
    
    print(f"  Model dtype: bfloat16")
    print(f"  Tokenizer vocab size: {len(tokenizer)}")
    print(f"  Max position embeddings: {model_max_pos}\n")

    # ==========Compliance Check & Review Helper Function==========
    
    def _check_anonymity(text, meta):
        authors = meta.get("authors", [])
        emails  = meta.get("emails", [])
        hits = []
        for author in authors:
            parts = [p for p in re.split(r'[\s,]+', author) if len(p) > 2]
            for part in parts:
                if part.lower() in text.lower():
                    hits.append(f"author name fragment '{part}' found in text")
        for email in emails:
            domain = email.split('@')[-1].rstrip('.,')
            if domain and domain.lower() in text.lower():
                hits.append(f"email domain '{domain}' found in text")
        if hits:
            return ("FAIL", hits[:3])
        return ("PASS", ["No direct author names or email domains detected in text."])

    def _check_acknowledgements(text):
        ack_match = re.search(r'\b(acknowledgement|acknowledgment|funded by|supported by|grant)\b', text, re.IGNORECASE)
        if ack_match:
            snippet = text[max(0, ack_match.start()-30):ack_match.start()+80].replace('\n', ' ')
            return ("WARN", [f"Acknowledgement-related text found: '...{snippet}...' — verify it does not reveal author identity."])
        return ("PASS", ["No acknowledgement section detected."])

    def _check_self_citation(text):
        hits = re.findall(r'\b(our previous work|in our earlier|we showed in \[|we proposed in \[|in our work \[)', text, re.IGNORECASE)
        if hits:
            return ("FAIL", [f"First-person self-citation phrase found: '{h}'" for h in hits[:3]])
        return ("PASS", ["No first-person self-citation phrasing detected."])

    def _check_identifying_links(text, meta):
        urls = re.findall(r'https?://\S+', text)
        emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
        hits = []
        author_domains = set()
        for e in meta.get("emails", []):
            if '@' in e:
                author_domains.add(e.split('@')[-1].rstrip('.,').lower())
        for url in urls:
            for domain in author_domains:
                if domain in url.lower():
                    hits.append(f"URL may identify authors: {url[:80]}")
        for email in emails:
            hits.append(f"Email address found in text: {email}")
        if hits:
            return ("WARN", hits[:5])
        return ("PASS", [f"{len(urls)} URL(s) found, none matched author domains. No email addresses in text."])

    def _check_section_present(text, keywords):
        for kw in keywords:
            if re.search(rf'\b{re.escape(kw)}\b', text, re.IGNORECASE):
                return ("PASS", [f"Section keyword '{kw}' found in text."])
        return ("WARN", [f"None of the keywords {keywords} found. Section may be missing."])

    COMPLIANCE_RULES = [
        {
            "id": "rule_01",
            "name": "Paper-level anonymity (double-blind)",
            "check": lambda text, meta: _check_anonymity(text, meta),
        },
        {
            "id": "rule_02",
            "name": "No identifying acknowledgements",
            "check": lambda text, meta: _check_acknowledgements(text),
        },
        {
            "id": "rule_03",
            "name": "Self-citation phrasing (third-person requirement)",
            "check": lambda text, meta: _check_self_citation(text),
        },
        {
            "id": "rule_05",
            "name": "No author-identifying external links or contact info",
            "check": lambda text, meta: _check_identifying_links(text, meta),
        },
        {
            "id": "rule_06",
            "name": "Reproducibility statement present",
            "check": lambda text, meta: _check_section_present(text, ["reproducibility", "reproducible"]),
        },
        {
            "id": "rule_07",
            "name": "Ethics statement when applicable",
            "check": lambda text, meta: _check_section_present(text, ["ethics", "ethical", "broader impact"]),
        },
    ]

    def process_paper(paper_data, filename, model=None, tokenizer=None, llm=None, output_dir=None):
        """Process a single paper to generate review.
        
        Args:
            paper_data: Parsed paper metadata
            filename: Original filename
            model: Transformers model (if using Transformers backend)
            tokenizer: Tokenizer instance
            llm: vLLM instance (if using vLLM backend)
            output_dir: Output directory for review
        """
        # Determine which backend to use
        use_vllm_backend = llm is not None
        
        # Build paper content
        paper_content = []
        paper_content.append('Title')
        paper_content.append(paper_data['metadata']['title'])
        paper_content.append('Abstract')
        paper_content.append(paper_data['metadata']['abstractText'])
        
        for section in paper_data['metadata']['sections']:
            paper_content.append(section['heading'])
            paper_content.append(section['text'])
        for i in range(len(paper_content)):
            if paper_content[i] is None:
                paper_content[i] = 'N/A'
        paper_content = "\n".join(paper_content).encode("utf-8", "ignore").decode("utf-8").strip()

        # ===== TOKEN MANAGEMENT: Check and truncate if necessary =====
        input_token_count = count_tokens(paper_content, tokenizer)
        max_paper_tokens = 32000  # Leave room for system prompts (8960 tokens)
        
        if input_token_count > max_paper_tokens:
            print(f"⚠️  Paper is LONG: {input_token_count} tokens (limit: {max_paper_tokens})")
            paper_content, input_token_count, truncation_info = truncate_paper_content(
                paper_content, tokenizer, max_paper_tokens
            )
            method = truncation_info.get("method", "unknown")
            print(f"   ✂️  {method.upper()}")
            print(f"   → Truncated to: {input_token_count} tokens")
        else:
            print(f"✓ Paper has {input_token_count} tokens (within limit)")
        
        # Calculate safe output tokens for this paper
        safe_max_output_phase1 = calculate_max_new_tokens(input_token_count, 4096)
        safe_max_output_phase2 = calculate_max_new_tokens(input_token_count, 8192)
        print(f"✓ Safe output tokens - Phase 1: {safe_max_output_phase1}, Phase 2: {safe_max_output_phase2}")

        # Compliance check
        full_text = paper_content
        meta      = paper_data.get("metadata", {})

        print(f"\n{'='*70}")
        print(f"Processing: {filename}")
        print(f"{'='*70}")
        print(f"SUBMISSION COMPLIANCE CHECK (Backend)")
        print(f"{'-'*70}")
        compliance_results = {}
        compliance_issues = []  # Track issues to inject into review
        
        for rule in COMPLIANCE_RULES:
            status, details = rule["check"](full_text, meta)
            compliance_results[rule["id"]] = {"name": rule["name"], "status": status, "details": details}
            icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️"}[status]
            print(f"{icon} [{rule['id']}] {rule['name']}: {status}")
            
            # Collect non-PASS issues to inform the reviewer
            if status != "PASS":
                compliance_issues.append({
                    "rule": rule["name"],
                    "status": status,
                    "details": details
                })

        # Build compliance guidance for reviewer (only issues, not passes)
        compliance_guidance = ""
        if compliance_issues:
            compliance_guidance = "\n[Reviewer Guidance - Potential Submission Format Issues]\n"
            for issue in compliance_issues:
                status_label = "⚠️ WARNING" if issue["status"] == "WARN" else "❌ FAILED COMPLIANCE"
                details_str = ", ".join(issue["details"]) if isinstance(issue["details"], list) else str(issue["details"])
                compliance_guidance += f"- {issue['rule']} ({status_label}): {details_str}\n"
            compliance_guidance += "\nPlease assess how these issues impact the paper's quality and contribution.\n"

        # Generate prompt
        prompt_Llama_2 = (
            "[INST] <<SYS>>\n"
            "You are an expert academic peer reviewer for top-tier AI/ML venues (NeurIPS, ICML, ICLR, ACL, AAAI). "
            "You have deep knowledge of machine learning, deep learning, NLP, and computer vision. "
            "Your role is to generate a precise set of deep, critical, and specific questions that will guide a thorough review of the paper. "
            "These questions must go well beyond surface-level observations and must probe the paper's technical depth, experimental rigor, and novelty.\n"
            "<</SYS>>\n"
            "Read the following paper carefully and in full:\n\n"
            "{paper_content}\n\n"
            "---\n"
            "Generate a set of SPECIFIC, CRITICAL questions to guide a thorough review of this paper. "
            "Each question must be grounded in the actual content of the paper (cite specific sections, equations, tables, or claims). "
            "The questions should probe the following dimensions:\n"
            "1. **Technical correctness**: Are the core claims, proofs, and derivations technically sound? Are there errors or unvalidated assumptions?\n"
            "2. **Experimental rigor**: Are the baselines appropriate and up-to-date? Are ablation studies sufficient? Are results statistically validated?\n"
            "3. **Novelty and contribution**: What is the precise novel contribution beyond prior work? Is the contribution clearly distinguished?\n"
            "4. **Clarity and reproducibility**: Are all hyperparameters, datasets, and training details specified for replication?\n"
            "5. **Scope and limitations**: Are the limitations honestly discussed? Do the conclusions overgeneralize the results?\n\n"
            "The reviewer should structure their response in the following sections:\n{format}\n"
            "[/INST]"
        )
        prompt_dict = {
            'paper_content': paper_content,
            'format': '\n'.join([
                'Summary Of The Paper',
                'Strengths And Weaknesses',
                'Detailed Technical Questions',
                'Experimental Questions',
                'Questions About Novelty And Related Work',
            ])
        }
        prompt = prompt_Llama_2.format_map(prompt_dict)

        # Generate analysis questions (Phase 1)
        # Use dynamically calculated safe max_new_tokens to avoid OOM
        if use_vllm_backend:
            prompt_generator = build_vllm_generator(llm, max_new_tokens=safe_max_output_phase1)
        else:
            prompt_generator = build_generator(model, tokenizer, max_new_tokens=safe_max_output_phase1, max_length=SEQ_LEN)
        gen_prompt = prompt_generator(prompt)
        print(f"[PROMPT PHASE] Analysis questions generated (length: {len(gen_prompt)} chars)")

        # Generate full review
        prompt_Llama_2 = (
            "[INST] <<SYS>>\n"
            "You are a senior expert reviewer at a top-tier AI/ML conference (NeurIPS, ICML, ICLR, ACL, or AAAI). "
            "You are known for writing thorough, technically rigorous, and constructive reviews that go well beyond superficial summaries. "
            "Your reviews are specific — you cite exact claims, equations, tables, or figures from the paper, and you back every criticism with evidence and reasoning. "
            "You are fair: you acknowledge genuine contributions before raising concerns, and you distinguish between major and minor issues. "
            "You do NOT write vague feedback such as 'the paper needs more experiments' without specifying exactly which experiments are missing and why they matter.\n"
            "<</SYS>>\n"
            "Read the following paper carefully and in full:\n\n"
            "{paper_content}\n\n"
            "{compliance_context}"
            "---\n"
            "You have been guided by the following critical analysis questions:\n"
            "{prompt_gen}\n\n"
            "---\n"
            "Using your analysis, write a DETAILED, SPECIFIC, and RIGOROUS peer review. "
            "Your review must:\n"
            "- Summarize the paper's claims and methods in your own words (not copy-paste from abstract).\n"
            "- List concrete, evidence-backed strengths (cite specific sections or results).\n"
            "- List concrete, evidence-backed weaknesses or concerns, ordered by severity. For each weakness: state the problem, cite the specific location in the paper, and suggest how it could be addressed.\n"
            "- Ask precise, answerable questions to the authors (not rhetorical).\n"
            "- State clear limitations the authors have not addressed.\n"
            "- Provide a recommendation with justification.\n\n"
            "Write your complete review strictly in the following sections:\n{format}\n"
            "[/INST]"
        )
        prompt_dict = {
            'paper_content': paper_content,
            'compliance_context': compliance_guidance,
            'prompt_gen': gen_prompt,
            'format': '\n'.join([
                'Summary Of The Paper',
                'Strengths',
                'Weaknesses',
                'Questions For The Authors',
                'Limitations Not Addressed By The Authors',
                'Soundness (1-4: 1=Poor, 2=Fair, 3=Good, 4=Excellent)',
                'Contribution (1-4)',
                'Confidence (1-5: 1=Not sure, 5=Expert)',
                'Rating (1-10: 1=Reject, 5=Borderline, 8=Accept, 10=Strong Accept)',
                'Brief Justification For Rating',
            ])
        }
        prompt = prompt_Llama_2.format_map(prompt_dict)

        # Generate full review (Phase 2)
        # Use dynamically calculated safe max_new_tokens to avoid OOM
        if use_vllm_backend:
            review_generator = build_vllm_generator(llm, max_new_tokens=safe_max_output_phase2)
        else:
            review_generator = build_generator(model, tokenizer, max_new_tokens=safe_max_output_phase2, max_length=SEQ_LEN)
        gen_review = review_generator(prompt)
        print(f"[REVIEW PHASE] Generated review (length: {len(gen_review)} chars)")

        # Save output
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            output_file = Path(output_dir) / f"{Path(filename).stem}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"PAPER: {paper_data['metadata']['title']}\n")
                f.write(f"{'='*80}\n\n")
                f.write("REVIEW\n")
                f.write(f"{'-'*80}\n")
                f.write(gen_review)
                f.write(f"\n\n{'='*80}\n")
            print(f"✓ Review saved to: {output_file}\n")
            return str(output_file)

    # ==========Batch Processing or Single File==========
    
    if args.mmd_dir and Path(args.mmd_dir).exists():
        # Batch mode: process all .mmd files (or filtered by paper_ids)
        all_mmd_files = sorted(glob.glob(str(Path(args.mmd_dir) / "*.mmd")))
        
        # Load paper IDs if provided
        paper_ids = None
        if args.paper_ids and Path(args.paper_ids).exists():
            with open(args.paper_ids, 'r', encoding='utf-8') as f:
                paper_ids = set(line.strip() for line in f if line.strip())
            # Filter mmd files by paper_ids
            mmd_files = [f for f in all_mmd_files if Path(f).stem in paper_ids]
            print(f"Loaded {len(paper_ids)} paper IDs → Found {len(mmd_files)} matching .mmd files")
        else:
            mmd_files = all_mmd_files
            print(f"No paper_ids filter provided → Processing all {len(mmd_files)} .mmd files")
        
        print(f"\n{'='*70}")
        print(f"BATCH MODE: Processing {len(mmd_files)} .mmd files")
        print(f"Input:  {args.mmd_dir}")
        print(f"Output: {args.output_dir}")
        print(f"Backend: {'vLLM' if use_vllm else 'Transformers'}")
        if args.skip_existing:
            print(f"Mode:   RESUME (skip existing outputs)")
        if args.force_reprocess:
            print(f"Mode:   FORCE REPROCESS (overwrite all)")
        print(f"{'='*70}\n")
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        success_count = 0
        failed_count = 0
        skipped_count = 0
        failed_files = []
        
        for i, mmd_file in enumerate(mmd_files, 1):
            # ===== CHECK: Paper already processed? =====
            output_file = output_dir / f"{Path(mmd_file).stem}.txt"
            
            if output_file.exists() and not args.force_reprocess:
                # Already exists, skip it
                skipped_count += 1
                print(f"[{i}/{len(mmd_files)}] ⊘ SKIP (already exists): {Path(mmd_file).name}")
                continue
            elif output_file.exists() and args.force_reprocess:
                # Exists but force reprocess
                print(f"[{i}/{len(mmd_files)}] ↻ REPROCESS (overwriting): {Path(mmd_file).name}")
            else:
                # Not exists, process normally
                print(f"[{i}/{len(mmd_files)}] → Processing: {Path(mmd_file).name}")
            
            try:
                # Convert .mmd to JSON
                with open(mmd_file, 'r', encoding='utf-8', errors='ignore') as f:
                    mmd_text = f.read()
                paper_data = parse_mmd(mmd_text)
                
                # Process paper (pass appropriate backend)
                if use_vllm:
                    process_paper(paper_data, mmd_file, model=None, tokenizer=tokenizer, llm=llm, output_dir=args.output_dir)
                else:
                    process_paper(paper_data, mmd_file, model=model, tokenizer=tokenizer, llm=None, output_dir=args.output_dir)
                success_count += 1
                
            except Exception as e:
                print(f"✗ ERROR processing {Path(mmd_file).name}: {str(e)}")
                failed_count += 1
                failed_files.append((Path(mmd_file).name, str(e)))
        
        print(f"\n{'='*70}")
        print(f"BATCH PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"✓ Successful: {success_count}/{len(mmd_files)}")
        print(f"✗ Failed: {failed_count}/{len(mmd_files)}")
        print(f"⊘ Skipped: {skipped_count}/{len(mmd_files)}")
        print(f"Total processed: {success_count + failed_count}/{len(mmd_files)}")
        if failed_files:
            print(f"\nFailed files:")
            for fname, err in failed_files:
                print(f"  - {fname}: {err}")
        print(f"\nReviews saved to: {args.output_dir}\n")
        
    elif args.json_path and Path(args.json_path).exists():
        # Single file mode: process single JSON
        print(f"\n{'='*70}")
        print(f"SINGLE FILE MODE")
        print(f"Input: {args.json_path}")
        print(f"Backend: {'vLLM' if use_vllm else 'Transformers'}")
        print(f"{'='*70}\n")
        
        with open(args.json_path, 'rb') as f:
            paper_data = json.load(f)
        
        if use_vllm:
            process_paper(paper_data, args.json_path, model=None, tokenizer=tokenizer, llm=llm, output_dir=args.output_dir)
        else:
            process_paper(paper_data, args.json_path, model=model, tokenizer=tokenizer, llm=None, output_dir=args.output_dir)
        
    else:
        print(f"\nERROR: Must provide either:")
        print(f"  --mmd_dir (default: {args.mmd_dir})")
        print(f"  --json_path (for single file mode)")
        print(f"\nUsage:")
        print(f"  # Batch with vLLM - ALL papers:")
        print(f"  python demo.py --use_vllm --batch_size 4")
        print(f"  ")
        print(f"  # Batch with vLLM - ONLY subset from paper_ids.txt:")
        print(f"  python demo.py --use_vllm --batch_size 4 --paper_ids data_subset/paper_ids.txt")
        print(f"  ")
        print(f"  # Batch with Transformers:")
        print(f"  python demo.py --mmd_dir /path/to/mmd --output_dir /path/to/output")
        print(f"  ")
        print(f"  # Single file mode:")
        print(f"  python demo.py --json_path paper.json\n")