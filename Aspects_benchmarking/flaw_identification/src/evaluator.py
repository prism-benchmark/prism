import json
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List

from src.azure_openai_client import (
    AzureChatClient,
    AzureOpenAIConfigError,
    get_default_deployment,
    get_preferred_gpt5mini_deployment,
)
from src.gemini_client import extract_first_json_object, json_loads_lenient
from src.cps_metrics import (
    calculate_cps,
    calculate_icps,
    calculate_ncps,
    calculate_section_breakdown,
    reorder_by_position,
)


def _extract_json_payload(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Model returned empty content instead of JSON.")
    try:
        return extract_first_json_object(cleaned)
    except Exception as exc:
        preview = cleaned[:300].replace("\n", "\\n")
        raise ValueError(f"Model response did not contain valid JSON. Preview: {preview}") from exc


def _load_json_response(response_text: str, step_name: str) -> dict[str, Any]:
    stripped = response_text.strip()
    # Fast path: try parsing the raw text directly first.
    # Gemini with response_mime_type="application/json" returns clean JSON without fences.
    try:
        return json_loads_lenient(stripped)
    except (json.JSONDecodeError, ValueError) as e:
        print(
            f"  [DEBUG] {step_name} direct parse failed ({type(e).__name__}): {str(e)[:120]}",
            file=sys.stderr,
        )
    # Fallback: strip markdown fences and extract the first balanced JSON object.
    try:
        return json_loads_lenient(_extract_json_payload(response_text))
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{step_name} returned invalid JSON: {exc}") from exc


def build_cps_issue_bank(micro_flaws_json: dict, evaluations_json: dict) -> dict[str, list[dict[str, str]]]:
    """
    Build a canonical issue bank for CPS from all-paper flaws.

    Rationale:
      - Issue identity should be consistent across reviewers.
      - Ordering should remain local to each reviewer's review text.
      - Therefore we reuse the paper-level flaw inventory (all reviews) as the
        canonical bank, then map each individual review onto that bank.
    """
    issues: list[dict[str, str]] = []
    evaluations = evaluations_json.get("evaluations", {})
    for flaw in micro_flaws_json.get("micro_flaws", []):
        flaw_id = flaw.get("flaw_id", "")
        if not flaw_id:
            continue
        eval_result = evaluations.get(flaw_id, {})
        if eval_result.get("is_valid") is not True:
            continue
        severity = eval_result.get("severity")
        if severity not in {"Critical", "Minor"}:
            continue
        issues.append(
            {
                "issue_id": flaw_id,
                "shared_severity": severity,
                "canonical_summary": (flaw.get("core_summary") or "").strip(),
                "macro_topic": (flaw.get("macro_topic") or "").strip(),
            }
        )
    return {"issues": issues}


def get_cps_mapping_prompt(review_text: str, issue_bank: dict[str, list[dict[str, str]]]) -> str:
    """
    Prompt for per-review mapping onto a paper-level canonical issue bank.

    Design rationale:
      - All reviews define the canonical issue inventory (consistent issue IDs).
      - One review at a time preserves reviewer-local order.
      - Regex/fuzzy anchor recovery restores the true textual order afterwards.
    """
    issue_bank_str = json.dumps(issue_bank, ensure_ascii=False, indent=2)
    return f"""
You are an expert meta-review analyst. Your task is to map ONE peer review onto a CANONICAL ISSUE BANK that was built from ALL reviews of the same paper.

GOAL:
Produce an ordered list of issue mentions from this ONE review only.
Each output item must reference an existing `issue_id` from the canonical issue bank.

STRICT RULES:
1. Use ONLY issue_ids that already exist in the canonical issue bank. DO NOT invent new issue_ids.
2. Output mentions in the EXACT ORDER they first appear in the review.
3. One canonical issue may appear AT MOST ONCE in the output for this review. If the reviewer repeats the same issue multiple times, merge them and keep the first occurrence.
4. If a sentence does not clearly match any issue in the canonical bank, omit it.
5. The `section_name` field must reflect the structural section of THIS review where the issue appears (e.g. "Weaknesses", "Summary", "Questions", "Suggestions"). If no explicit header exists, infer the most plausible local section.
6. The `anchor_quote` field MUST be a VERBATIM substring (5-20 words) copied EXACTLY from this review text.
7. The `content` field should be a concise 1-2 sentence paraphrase of how THIS reviewer expressed the issue.
8. Do not output praise, neutral summaries, or questions that do not clearly correspond to a critique issue in the canonical bank.

RETURN FORMAT (strict JSON, no markdown fences):
{{
  "mentions": [
    {{
      "issue_id": "F01",
      "section_name": "Weaknesses",
      "anchor_quote": "Theorem 1 is incorrect or at least it is incorrectly stated",
      "content": "The reviewer argues that Theorem 1 is mathematically incorrect because approximation assumptions are not stated."
    }},
    {{
      "issue_id": "F04",
      "section_name": "Weaknesses",
      "anchor_quote": "several existing works studying the implicit bias",
      "content": "The reviewer points out that relevant prior work on momentum-based implicit bias is missing."
    }}
  ]
}}

CANONICAL ISSUE BANK:
{issue_bank_str}

Review text:
\"\"\"
{review_text}
\"\"\"
""".strip()


def process_single_review_for_cps(
    pipeline: "ReviewEvaluatorPipeline",
    review_text: str,
    reviewer_id: str,
    issue_bank: dict[str, list[dict[str, str]]],
    *,
    deployment: str | None = None,
) -> dict:
    """
    Compute CPS metrics for one reviewer by mapping ordered mentions in that
    review onto a shared canonical issue bank built from all reviews.
    """
    issues = issue_bank.get("issues", [])
    if not issues:
        return {
            "reviewer_id": reviewer_id,
            "arguments": [],
            "metrics": {
                "Reviewer_ID": reviewer_id,
                "Raw_CPS": 0.0,
                "ICPS": 0.0,
                "nCPS": 0.0,
                "Total_Arguments": 0,
                "Critical_Count": 0,
                "Minor_Count": 0,
                "Section_Breakdown": [],
            },
        }

    issue_index = {issue["issue_id"]: issue for issue in issues if issue.get("issue_id")}
    prompt = get_cps_mapping_prompt(review_text, issue_bank)
    parsed = pipeline.generate_json_response(
        step_name=f"CPS mapping ({reviewer_id})",
        system_prompt="You map one review onto a canonical issue bank as structured JSON.",
        user_prompt=prompt,
        temperature=0.0,
        max_output_tokens=pipeline.step1_max_output_tokens,
        deployment=deployment or pipeline.step1_deployment,
    )
    mentions = parsed.get("mentions", [])

    arguments = []
    seen_issue_ids = set()
    for mention in mentions:
        issue_id = (mention.get("issue_id") or "").strip()
        if not issue_id or issue_id in seen_issue_ids or issue_id not in issue_index:
            continue
        issue = issue_index[issue_id]
        severity = issue.get("shared_severity")
        if severity not in {"Critical", "Minor"}:
            continue
        seen_issue_ids.add(issue_id)
        arguments.append(
            {
                "issue_id": issue_id,
                "section_name": (mention.get("section_name") or "Weaknesses").strip() or "Weaknesses",
                "anchor_quote": (mention.get("anchor_quote") or "").strip(),
                "content": (mention.get("content") or issue.get("canonical_summary") or "").strip(),
                "severity": severity,
                "canonical_summary": issue.get("canonical_summary", ""),
                "macro_topic": issue.get("macro_topic", ""),
            }
        )

    # Re-sort within each section by true position in review text.
    arguments = reorder_by_position(arguments, review_text)

    raw_cps = round(calculate_cps(arguments), 4)
    icps = round(calculate_icps(arguments), 4)
    ncps = calculate_ncps(arguments)
    n_critical = sum(1 for a in arguments if a.get("severity") == "Critical")
    n_minor = len(arguments) - n_critical
    section_breakdown = calculate_section_breakdown(arguments)
    return {
        "reviewer_id": reviewer_id,
        "arguments": arguments,
        "metrics": {
            "Reviewer_ID": reviewer_id,
            "Raw_CPS": raw_cps,
            "ICPS": icps,
            "nCPS": ncps,
            "Total_Arguments": len(arguments),
            "Critical_Count": n_critical,
            "Minor_Count": n_minor,
            "Section_Breakdown": section_breakdown,
        },
    }


def _normalize_reviewer_id(raw_reviewer_id: str) -> str:
    reviewer_id = (raw_reviewer_id or "").strip()
    lower = reviewer_id.lower()
    if "llm" in lower or "sea" in lower:
        return "LLM_Reviewer"
    match = re.search(r"human[_\s-]*(\d+)", lower)
    if match:
        return f"Human_{match.group(1)}"
    return reviewer_id


def _extract_section_from_macro_topic(macro_topic: str) -> str:
    """
    Convert "4. Experimental ... - Missing/Weak Baselines" into a stable section label.
    """
    topic = (macro_topic or "").strip()
    if not topic:
        return "Unknown"
    if " - " in topic:
        topic = topic.split(" - ", 1)[0].strip()
    return topic or "Unknown"


def build_cps_arguments_by_reviewer(
    micro_flaws_json: dict,
    evaluations_json: dict,
) -> dict[str, list[dict[str, str]]]:
    """
    Build CPS-ready arguments from flaw-identification outputs.

    Each argument contains:
      - reviewer_id
      - section
      - severity (Critical/Minor/None)
      - content
      - flaw_id
    """
    flaws = micro_flaws_json.get("micro_flaws", [])
    evaluations = evaluations_json.get("evaluations", {})
    args_by_reviewer: dict[str, list[dict[str, str]]] = defaultdict(list)

    for flaw in flaws:
        flaw_id = flaw.get("flaw_id", "")
        eval_result = evaluations.get(flaw_id, {})
        if eval_result.get("is_valid") is not True:
            continue

        severity = eval_result.get("severity")
        if severity not in {"Critical", "Minor"}:
            severity = "None"

        section = _extract_section_from_macro_topic(flaw.get("macro_topic", ""))
        raw_arguments = flaw.get("raw_arguments", {})

        for raw_reviewer_id, quote in raw_arguments.items():
            content = (quote or "").strip()
            if not content:
                continue
            reviewer_id = _normalize_reviewer_id(raw_reviewer_id)
            args_by_reviewer[reviewer_id].append(
                {
                    "reviewer_id": reviewer_id,
                    "section": section,
                    "severity": severity,
                    "content": content,
                    "flaw_id": flaw_id,
                }
            )

    return dict(args_by_reviewer)

class ReviewEvaluatorPipeline:
    def __init__(self, api_key: str | None = None, client: Any | None = None, provider: str | None = None, model: str | None = None):
        self.api_key = api_key

        # ── Token budgets ─────────────────────────────────────────────────────
        #
        # Step 1 output CAN be very large: each flaw has raw_arguments with
        # verbatim quotes from N reviewers. For 5 reviewers × 20 flaws,
        # output ≈ 20 × (5 quotes × 150 chars + overhead) ≈ 18K chars ≈ 5K tokens.
        # In practice with longer quotes the output can exceed 31K chars (≈9K tokens).
        # We default to 16000 to give ample headroom; override via env var.
        #
        # Step 2 output is compact: {"evaluations": {"F01": {...}, ...}}
        # For 30 flaws that's ≈ 30 × 60 chars ≈ 1800 chars ≈ 500 tokens.
        # 4096 is more than enough.
        self.step1_max_output_tokens = int(os.getenv("AZURE_OPENAI_STEP1_MAX_OUTPUT_TOKENS", "35000"))
        self.step2_max_output_tokens = int(os.getenv("AZURE_OPENAI_STEP2_MAX_OUTPUT_TOKENS", "4096"))
        self.step2_max_input_chars   = int(os.getenv("AZURE_OPENAI_STEP2_MAX_INPUT_CHARS", "45000"))

        # ── Use centralized config if available ──────────────────────────────
        _centralized = False
        if client is None:
            try:
                _sys_path_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
                if _sys_path_root not in sys.path:
                    sys.path.insert(0, _sys_path_root)
                from ai_config import get_llm_client
                overrides = {}
                if provider is not None:
                    overrides["provider"] = provider
                if model is not None:
                    overrides["model"] = model
                if api_key is not None:
                    overrides["api_key"] = api_key
                self.step1_client = get_llm_client("flaw_identification", step="step1", **overrides)
                self.step2_client = get_llm_client("flaw_identification", step="step2", **overrides)
                self.client = self.step1_client
                self.step1_deployment = self.step1_client.model
                self.step2_deployment = self.step2_client.model
                _centralized = True
                print(
                    "  [INFO] Flaw identification using centralized config "
                    f"(step1={self.step1_client.provider}/{self.step1_client.model}, "
                    f"step2={self.step2_client.provider}/{self.step2_client.model})"
                )
            except Exception as e:
                print(f"  [WARNING] Centralized config failed, falling back: {e}")

        if not _centralized:
            provider_name = getattr(client, "provider", None)
            if provider_name == "gemini-devmate":
                if "AZURE_OPENAI_STEP1_MAX_OUTPUT_TOKENS" not in os.environ:
                    self.step1_max_output_tokens = max(self.step1_max_output_tokens, 12000)
                if "AZURE_OPENAI_STEP2_MAX_OUTPUT_TOKENS" not in os.environ:
                    self.step2_max_output_tokens = max(self.step2_max_output_tokens, 8000)
            if client is not None:
                self.client = client
                self.step1_client = client
                self.step2_client = client
                shared_deployment = getattr(client, "model", None) or getattr(client, "provider", "configured-client")
                self.step1_deployment = shared_deployment
                self.step2_deployment = shared_deployment
            else:
                preferred_gpt5mini = get_preferred_gpt5mini_deployment()
                base_deployment = preferred_gpt5mini or get_default_deployment()
                self.client = AzureChatClient(
                    deployment=base_deployment,
                    api_key=self.api_key,
                    max_output_tokens=max(self.step1_max_output_tokens, self.step2_max_output_tokens),
                )
                self.step1_client = self.client
                self.step2_client = self.client
                shared_deployment = base_deployment
                self.step1_deployment = self._resolve_deployment(
                    "AZURE_OPENAI_STEP1_DEPLOYMENT",
                    shared_deployment,
                )
                self.step2_deployment = self._resolve_deployment(
                    "AZURE_OPENAI_STEP2_DEPLOYMENT",
                    shared_deployment,
                )
        
        # Store prompts
        self.prompt_step1 = """
        SYSTEM PROMPT (STEP 1):
You are an expert meta-reviewer for top-tier computer science conferences (e.g., ICLR, NeurIPS). Your task is to analyze raw review texts from multiple reviewers (both Human and AI) and consolidate their arguments into a structured list of unique "Micro-flaws".

For each unique weakness mentioned across the reviews, create a "Micro-flaw" object. 

CRITICAL RULES FOR GROUPING (STRICTLY ENFORCED):
You must avoid "Frankenstein" clustering (merging fundamentally different scientific issues just because they fall under the same broad Macro-Topic). Follow these rules:

1. CONCEPTUAL CONSISTENCY (Must Split): Arguments grouped into the same Micro-flaw MUST address the same fundamental conceptual, methodological, or experimental problem.
   - Example to SPLIT: If Reviewer A criticizes "scaling activation values to integers" and Reviewer B criticizes "using an LLM to validate the assessment", these are fundamentally different scientific issues. They MUST be separate Micro-flaws, even if both are "Experimental Design".
   - Example to SPLIT: A complaint about "limited dataset (MNIST)" and a complaint about "missing regularization details" are entirely different problems. DO NOT merge them.

2. ALLOWED AGGREGATION (Can Group): You MAY group arguments if they share the exact same nature or severity level, even if they point to different sections of the paper.
   - Example to GROUP: If Reviewer A says "broken references", Reviewer B says "Equation 7 dimensions don't match", and Reviewer C says "Section 3 is unclear", you CAN and SHOULD group them together under ONE Micro-flaw for "2. Clarity & Presentation - General writing & Clarity issues".

3. NO FORCED FIT: Do not force an argument into a mismatched Micro-flaw type just to group it. If an argument doesn't fit the existing types perfectly, use the "Other [Topic] Issues" category.
Categorize each Micro-flaw by selecting EXACTLY ONE Macro-topic and its corresponding Micro-flaw type from the hierarchical taxonomy below:

1. Novelty & Contribution
   - Limited Novelty
   - Incremental Contribution Only
   - Lack of Significance/Impact
   - Other Novelty Issues
2. Clarity & Presentation
   - General writing & Clarity issues
   - Unclear Math/ Notations
   - Poor Figures/Tables Quality
   - Grammar & Typos
   - Other Presentation Issues
3. Applicability, Scalability & Limitations
   - General Applicability Issues
   - Scalability & Complexity Concerns
   - Lack of Discussion on Limitations
   - Missing Broader Impact/Ethical Concerns
   - Other Limitation Issues
4. Experimental Design & Evaluation 
   - Missing/Weak Baselines
   - Insufficient Experimental Validation
   - Questionable Evaluation Metrics
   - Limited/Biased Datasets
   - Other Evaluation Issues
5. Related work & Citations
   - Missing Comparisons with Prior Work
   - Missing Relevant Citations
   - Missing Recent/Concurrent Works
   - Other Citation Issues
6. Methodology & Theoretical Soundness
   - Weak Theoretical Justification/Proofs
   - Methodological Flaws
   - Strong/Unrealistic Assumptions
   - Lack of Intuition/Justification
   - Other Methodology Issues
7. Reproducibility & Open Science
   - General Reproducibility Concerns
   - Insufficient Implementation Details
   - Missing Code/Data Repository
   - Other Reproducibility Issues
CRITICAL: DO NOT group arguments just because they belong to the same Macro-topic. Only group them if they point to the EXACT SAME specific error in the paper. If Human 1 talks about missing baseline A, and Human 2 talks about missing baseline B, they are TWO DIFFERENT Micro-flaws.
NO LIMIT ON NUMBER OF MICRO-FLAWS: There is NO upper bound on the total number of Micro-flaws you may output. A paper can have 5, 10, 15, or more Micro-flaws — output as many as the reviews warrant. Multiple Micro-flaws CAN and SHOULD share the same Macro-topic number and name (e.g., two separate flaws can both have "4. Experimental Design & Evaluation") as long as they address different specific problems. The 7 Macro-topics in the taxonomy are categories, NOT a cap on the number of Micro-flaws.
OUTPUT FORMAT:
You MUST output a valid JSON object strictly matching this schema:
{
  "micro_flaws": [
    {
      "flaw_id": "F01",
      "macro_topic": "<Macro-topic number and name> - <Micro-flaw type>",
      "core_summary": "<A concise 1-sentence summary of the weakness>",
      "raw_arguments": {
        "<EXACT_ID_FROM_INPUT_1>": "<Exact quote>",
        "<EXACT_ID_FROM_INPUT_2>": "<Exact quote>"
      }
    }
  ]
}
CRITICAL INSTRUCTION: The keys inside "raw_arguments" MUST exactly match the reviewer IDs provided in the input (e.g., "Human_1", "Human_2", "LLM_Reviewer"). DO NOT rename them to "Reviewer_ID_1" or anything else.
EXAMPLE OF MACRO_TOPIC FIELD:
"macro_topic": "4. Experimental Design & Evaluation - Missing/Weak Baselines"""
        self.prompt_step2 = """
SYSTEM PROMPT (STEP 2):
You are a strict and objective Meta-Reviewer in a top-tier Computer Science conference. You will be provided with the FULL TEXT of a submitted scientific paper and a JSON list of "Micro-flaws" raised by various reviewers.

Your task is to independently verify each Micro-flaw against the paper's text.
For EACH Micro-flaw, answer two questions based STRICTLY on the paper's content:
1. is_valid (True/False): Does this flaw actually exist in the paper? Is the reviewer's argument factually correct? (Return False if it's a hallucination, a misunderstanding, or an unreasonable request).
2. severity ("Critical" / "Minor"): If valid, you MUST assign a strict severity label based on the predefined ontology below.

SEVERITY SCORING POLICY (used downstream by metrics):
- Critical = 2 points
- Minor = 1 point
- None = 0 points (must be used when is_valid is False)

ONTOLOGY FOR SEVERITY MAPPING:
Assign CRITICAL if the flaw falls into these categories:
- Methodology & Theoretical Soundness: Weak Theoretical Justification/Proofs, Methodological Flaws, Strong/Unrealistic Assumptions, Lack of Intuition/Justification.
- Experimental Design & Evaluation: Missing/Weak Baselines, Insufficient Experimental Validation, Questionable Evaluation Metrics, Limited/Biased Datasets.
- Novelty & Contribution: Limited Novelty, Incremental Contribution Only, Lack of Significance/Impact.
- Applicability & Reproducibility (Severe): General Applicability Issues, Scalability & Complexity Concerns, General Reproducibility Concerns.
- Related Work (Severe): Missing Empirical Comparisons with Prior Work.

Assign MINOR if the flaw falls into these categories:
- Clarity & Presentation: General writing & Clarity issues, Unclear Math/Notations (ambiguous symbols, not fundamentally wrong math), Poor Figures/Tables Quality, Grammar & Typos.
- Applicability & Limitations (Textual): Lack of Discussion on Limitations, Missing Broader Impact/Ethical Concerns.
- Related Work & Citations: Missing Relevant Citations, Missing Recent/Concurrent Works.
- Reproducibility & Open Science (Documentation): Insufficient Implementation Details (e.g., missing hyperparameters in the appendix), Missing Code/Data Repository.

BORDERLINE DECISION RULES:
- Prefer Critical if fixing the issue requires new experiments, new analyses, core equation/proof changes, or changes that can alter main claims.
- Prefer Minor if fixing the issue is mainly textual/editorial (clarity wording, writing, citations, minor missing implementation details) and does not change core claims.
- If uncertain between Critical and Minor, choose Minor unless the flaw can plausibly invalidate conclusions.

CRUCIAL RULE: Do not guess. If is_valid is False, set severity to "None".

OUTPUT FORMAT (Strict JSON):
{
  "evaluations": {
    "F01": {
      "is_valid": true,
      "severity": "Critical"
    },
    "F02": {
      "is_valid": true,
      "severity": "Minor"
    }
  }
}"""

    # def step1_atomize_and_group(self, human_reviews: Dict[str, str], llm_review: str) -> dict:
    #     """
    #     Input: 
    #         human_reviews: dict format {"Human_1": "text...", "Human_2": "text..."}
    #         llm_review: string chứa review của LLM
    #     """
    #     # Trộn input
    #     input_text = ""
    #     for reviewer_id, review_text in human_reviews.items():
    #         input_text += f"\n\n[REVIEWER: {reviewer_id}]\n{review_text}"
    #     input_text += f"\n\n[REVIEWER: LLM_Reviewer]\n{llm_review}"

    #     response = self.client.chat.completions.create(
    #         model="gpt-5-mini",
    #         response_format={"type": "json_object"}, # Ép trả về chuẩn JSON
    #         messages=[
    #             {"role": "system", "content": self.prompt_step1},
    #             {"role": "user", "content": f"Here are the reviews to analyze:\n{input_text}"}
    #         ],
    #         # temperature=0.1,
    #     )
    #     return json.loads(response.choices[0].message.content)

    # def step2_judge_flaws(self, paper_text: str, micro_flaws_json: dict) -> dict:
    #     """
    #     Input: Text của bài báo gốc và file JSON kết quả từ Step 1
    #     """
    #     flaws_str = json.dumps(micro_flaws_json, indent=2)
    #     input_text = f"[PAPER TEXT]\n{paper_text[:20000]}...\n\n[MICRO-FLAWS]\n{flaws_str}" # Cắt bớt nếu paper quá dài so với context window

    #     response = self.client.chat.completions.create(
    #         model="gpt-4o", # Nên dùng model mạnh hơn cho bước này
    #         response_format={"type": "json_object"},
    #         messages=[
    #             {"role": "system", "content": self.prompt_step2},
    #             {"role": "user", "content": input_text}
    #         ],
    #         temperature=0.0,
    #     )
    #     return json.loads(response.choices[0].message.content)
    
    def _resolve_deployment(self, env_name: str, default: str) -> str:
        """Resolve Azure deployment name for a pipeline step.

        Azure requires a deployment name, not just a base model name. If no step-specific
        deployment is configured, we fall back to the shared deployment configured for the app.
        """
        preferred_gpt5mini = get_preferred_gpt5mini_deployment()
        deployment = os.getenv(env_name)
        if deployment:
            return deployment
        if preferred_gpt5mini:
            return preferred_gpt5mini
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("AZURE_CHAT_DEPLOYMENT") or default
        if deployment:
            return deployment
        raise AzureOpenAIConfigError(
            f"Missing Azure deployment for {env_name}. Set {env_name} or AZURE_OPENAI_DEPLOYMENT (or AZURE_CHAT_DEPLOYMENT) to an existing Azure deployment name."
        )

    def _repair_json_response(
        self,
        *,
        step_name: str,
        system_prompt: str,
        user_prompt: str,
        invalid_response: str,
        max_output_tokens: int,
        deployment: str | None,
        client: Any | None = None,
    ) -> dict[str, Any]:
        repair_system_prompt = (
            "You are a strict JSON repair assistant. "
            "Return exactly one valid JSON object. "
            "Do not include markdown fences or any extra commentary."
        )
        repair_user_prompt = (
            f"The previous model output for {step_name} was not valid JSON.\n\n"
            "[ORIGINAL SYSTEM PROMPT]\n"
            f"{system_prompt}\n\n"
            "[ORIGINAL USER PROMPT]\n"
            f"{user_prompt}\n\n"
            "[INVALID MODEL OUTPUT]\n"
            f"{invalid_response}\n\n"
            "Rewrite the answer as exactly one valid JSON object that follows the original task and schema."
        )
        llm_client = client or self.client
        repaired_text = llm_client.generate_text(
            repair_system_prompt,
            repair_user_prompt,
            response_format={"type": "json_object"},
            temperature=0.0,
            max_output_tokens=max_output_tokens,
            deployment=deployment,
        )
        return _load_json_response(repaired_text, f"{step_name} repair")

    def generate_json_response(
        self,
        *,
        step_name: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
        deployment: str | None,
        client: Any | None = None,
    ) -> dict[str, Any]:
        llm_client = client or self.client
        response_text = llm_client.generate_text(
            system_prompt,
            user_prompt,
            response_format={"type": "json_object"},
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            deployment=deployment,
        )
        try:
            return _load_json_response(response_text, step_name)
        except ValueError as exc:
            print(f"  [WARNING] {step_name} returned non-JSON output; attempting repair...")
            return self._repair_json_response(
                step_name=step_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                invalid_response=response_text,
                max_output_tokens=max_output_tokens,
                deployment=deployment,
                client=llm_client,
            )
    
    def step1_atomize_and_group(self, human_reviews: Dict[str, str], llm_review: str) -> dict:
        """
        Gộp và phân tách các luận điểm đánh giá (Sử dụng gpt-5-mini)
        """
        input_text = ""
        for reviewer_id, review_text in human_reviews.items():
            input_text += f"\n\n[REVIEWER: {reviewer_id}]\n{review_text}"
        input_text += f"\n\n[REVIEWER: LLM_Reviewer]\n{llm_review}"

        # ── Adaptive Step 1 token budget ───────────────────────────────────
        # Output size scales with (n_reviewers × avg_review_length).
        # Each micro-flaw contains raw_arguments with verbatim quotes from
        # every reviewer that mentioned it.
        # Empirical: total_input_chars × 0.35 gives a reasonable output estimate;
        # add 20% safety margin.  Floor = self.step1_max_output_tokens (env default).
        total_input_chars = len(input_text)
        estimated_output_tokens = int(total_input_chars * 0.35 / 3.5 * 1.2)  # chars→tokens×safety
        # Also cap at Gemini Flash Lite practical limit (65536 output tokens)
        adaptive_tokens = max(self.step1_max_output_tokens,
                              min(estimated_output_tokens, 32000))
        if adaptive_tokens != self.step1_max_output_tokens:
            print(
                f"  [INFO] Step 1 adaptive budget: input={total_input_chars:,} chars"
                f" → max_output_tokens={adaptive_tokens}",
                file=sys.stderr,
            )

        print(f"  [INFO] Calling model/deployment '{self.step1_deployment}' for Step 1...")
        return self.generate_json_response(
            step_name="Step 1",
            system_prompt=self.prompt_step1,
            user_prompt=f"Here are the reviews to analyze:\n{input_text}",
            temperature=0.1,
            max_output_tokens=adaptive_tokens,
            deployment=self.step1_deployment,
            client=self.step1_client,
        )

    def step2_judge_flaws(self, paper_text: str, micro_flaws_json: dict) -> dict:
        """
        Thẩm định lỗi và Gán nhãn Severity (Sử dụng gpt-5-mini)
        """
        flaws_str = json.dumps(micro_flaws_json, indent=2)
        
        safe_paper_text = paper_text[: self.step2_max_input_chars]
        
        input_text = f"[PAPER TEXT]\n{safe_paper_text}\n\n[MICRO-FLAWS]\n{flaws_str}"

        print(f"  [INFO] Calling model/deployment '{self.step2_deployment}' for Step 2...")
        return self.generate_json_response(
            step_name="Step 2",
            system_prompt=self.prompt_step2,
            user_prompt=input_text,
            temperature=0.0,
            max_output_tokens=self.step2_max_output_tokens,
            deployment=self.step2_deployment,
            client=self.step2_client,
        )

class MetricsCalculator:
    """
    Class này nhận kết quả từ Step 1 (Flaws Grouping) và Step 2 (Judgement) 
    để tính toán các chỉ số thống kê toán học.
    """
    def __init__(self, micro_flaws_json: dict, evaluations_json: dict):
        self.flaws = micro_flaws_json.get("micro_flaws", [])
        self.evals = evaluations_json.get("evaluations", {})
        
        # Khởi tạo các tập hợp (Sets) Ground Truth
        self.G_all = set()
        self.G_critical = set()
        self.G_minor = set()
        
        # Phân loại Ground Truth dựa trên kết quả của LLM Judge
        for flaw_id, result in self.evals.items():
            if result.get("is_valid") is True:
                self.G_all.add(flaw_id)
                if result.get("severity") == "Critical":
                    self.G_critical.add(flaw_id)
                elif result.get("severity") == "Minor":
                    self.G_minor.add(flaw_id)

    def get_reviewer_flaws(self, reviewer_id: str) -> set:
            """Lấy danh sách các flaw_id mà một reviewer cụ thể đã chỉ ra (Flexible Matching)"""
            detected_flaws = set()
            for flaw in self.flaws:
                raw_args = flaw.get("raw_arguments", {})
                
                for key in raw_args.keys():
                    # Xử lý cho LLM_Reviewer
                    if reviewer_id == "LLM_Reviewer" and ("llm" in key.lower() or "sea" in key.lower()):
                        detected_flaws.add(flaw["flaw_id"])
                        break
                    
                    # Xử lý cho Human_X (Ví dụ Input là "Human_1", nếu key là "Reviewer_ID_1" thì vẫn lấy)
                    elif reviewer_id.startswith("Human_"):
                        # Lấy con số định danh, ví dụ "1" từ "Human_1"
                        human_num = reviewer_id.split("_")[1] 
                        if human_num in key: 
                            detected_flaws.add(flaw["flaw_id"])
                            break
                            
            return detected_flaws

    def calculate_scores(self, reviewer_flaws: set) -> dict:
        """Tính toán Precision, Recall, F1 dựa trên toán học tập hợp"""
        # Precision
        true_positives = len(reviewer_flaws.intersection(self.G_all))
        precision = true_positives / len(reviewer_flaws) if len(reviewer_flaws) > 0 else 0.0
        
        # Recalls
        recall_critical = len(reviewer_flaws.intersection(self.G_critical)) / len(self.G_critical) if self.G_critical else 0.0
        recall_minor = len(reviewer_flaws.intersection(self.G_minor)) / len(self.G_minor) if self.G_minor else 0.0
        recall_overall = true_positives / len(self.G_all) if self.G_all else 0.0
        
        # F1-Score
        f1 = 2 * (precision * recall_overall) / (precision + recall_overall) if (precision + recall_overall) > 0 else 0.0
        
        return {
            "Precision": round(precision, 4),
            "Recall_Critical": round(recall_critical, 4),
            "Recall_Minor": round(recall_minor, 4),
            "Recall_Overall": round(recall_overall, 4),
            "F1_Score": round(f1, 4)
        }

    def generate_report(self, human_ids: List[str]) -> dict:
        """Tạo báo cáo so sánh xử lý Lỗ hổng 2 (Individual vs Collective)"""
        report = {}
        
        # 1. Điểm của LLM Reviewer
        llm_flaws = self.get_reviewer_flaws("LLM_Reviewer")
        report["LLM_Reviewer"] = self.calculate_scores(llm_flaws)
        
        # 2. Điểm của từng cá nhân Human và trung bình Human
        human_individual_scores = []
        human_collective_flaws = set()
        
        for h_id in human_ids:
            h_flaws = self.get_reviewer_flaws(h_id)
            report[h_id] = self.calculate_scores(h_flaws)
            human_individual_scores.append(report[h_id]["F1_Score"])
            
            # Gộp flaws cho Collective
            human_collective_flaws.update(h_flaws)
            
        # 3. Điểm của Hội đồng Human (Collective)
        report["Human_Collective"] = self.calculate_scores(human_collective_flaws)
        
        # 4. Tính Macro-Average F1 của Human để so sánh công bằng 1vs1
        report["Human_Average_F1"] = round(sum(human_individual_scores) / len(human_individual_scores) if human_individual_scores else 0, 4)
        
        return report

# --- CÁCH SỬ DỤNG TRONG PIPELINE CHÍNH ---
if __name__ == "__main__":
    # Khởi tạo
    pipeline = ReviewEvaluatorPipeline(api_key="YOUR_OPENAI_API_KEY")
    
    # Giả lập dữ liệu đã load từ hàm utils của bạn
    paper_content = "Toàn bộ text của bài báo..."
    human_reviews_dict = {
        "Human_1": "The baseline is missing...",
        "Human_2": "Equation 3 is wrong, and there are many typos."
    }
    llm_review_text = "It fails to compare with standard baselines. There are spelling mistakes."
    
    # Chạy Pipeline
    print("Running Step 1: Atomize & Group...")
    step1_output = pipeline.step1_atomize_and_group(human_reviews_dict, llm_review_text)
    
    print("Running Step 2: Evaluating Ground Truth...")
    step2_output = pipeline.step2_judge_flaws(paper_content, step1_output)
    
    print("Calculating Metrics...")
    calculator = MetricsCalculator(step1_output, step2_output)
    final_report = calculator.generate_report(human_ids=["Human_1", "Human_2"])
    
    print(json.dumps(final_report, indent=2))
