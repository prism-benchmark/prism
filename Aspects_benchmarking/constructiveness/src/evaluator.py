"""
LLM-as-Judge for Constructiveness Evaluation.

Single LLM call per review:
  1. Atomize review into Atomic Review Comments (ARCs)
  2. Score each ARC on 5 constructiveness dimensions (D1–D5)

With retry-on-empty: if the first call returns 0 ARCs, a focused
reprompt is sent once more before giving up.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
_FI_SRC = os.path.normpath(os.path.join(_HERE, "..", "..", "flaw_identification", "src"))

import importlib.util

def _import_fi_module(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(
        f"fi_{module_name}", os.path.join(_FI_SRC, file_name),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_azure_mod = _import_fi_module("azure_openai_client", "azure_openai_client.py")
AzureChatClient = _azure_mod.AzureChatClient
get_default_deployment = _azure_mod.get_default_deployment
get_preferred_gpt5mini_deployment = _azure_mod.get_preferred_gpt5mini_deployment
_unified_mod = _import_fi_module("unified_client", "unified_client.py")
UnifiedChatClient = _unified_mod.UnifiedChatClient


# ---------------------------------------------------------------------------
# Prompt — designed so Gemini/Devmate returns JSON even without json_mode
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert peer-review analyst. Your ONLY job is to output a single JSON object. Do NOT output markdown, commentary, or analysis outside JSON.

TASK
====
Given a peer review, decompose it into Atomic Review Comments (ARCs) and score each on 5 dimensions.

DIMENSIONS (score 0, 1, or 2 for each)
=======================================
D1_actionability  — Can the author act on this?
  0 = opinion with no guidance ("poorly written")
  1 = general direction ("needs more baselines")
  2 = specific, implementable ("add [MethodX] on CIFAR-10 with same splits as Table 2")

D2_specificity — References concrete paper elements?
  0 = vague ("has issues")
  1 = semi-specific ("methodology section unclear")
  2 = pinpoints exact element ("Eq 7 in Sec 4.2 missing regularization term")

D3_justification — Evidence-backed?
  0 = bare assertion ("not novel")
  1 = partial reasoning ("similar to prior work on X")
  2 = full evidence ("same loss as [Author2020] Eq 3, only activation differs")

D4_solution — Suggests improvements?
  0 = problem-only ("baselines weak")
  1 = implicit fix ("lacks recent SOTA" implies add them)
  2 = explicit fix ("add [M2023] and [M2024] achieving X% on same benchmark")

D5_tone — Respectful?
  0 = hostile/dismissive
  1 = neutral, factual
  2 = professional-constructive, encouraging

RULES
=====
- Extract ALL distinct points from Summary, Strengths, Weaknesses, Questions, Suggestions.
- One point per ARC. Two critiques in one sentence = two ARCs.
- anchor_quote: verbatim 5-25 word substring copied EXACTLY from the review.
- comment_type: one of "weakness", "strength", "question", "suggestion", "observation".
- Minimum 1 ARC for any non-empty review.

OUTPUT — Return ONLY this JSON, nothing else:
{"atomic_comments":[{"arc_id":"ARC_01","section":"Weaknesses","comment_type":"weakness","content":"concise paraphrase","anchor_quote":"verbatim quote from review","D1_actionability":1,"D2_specificity":2,"D3_justification":1,"D4_solution":0,"D5_tone":2}]}"""


RETRY_PROMPT = """\
Your previous response did not produce valid JSON with an "atomic_comments" array.

You MUST return ONLY a JSON object like:
{"atomic_comments":[{"arc_id":"ARC_01","section":"...","comment_type":"...","content":"...","anchor_quote":"...","D1_actionability":0,"D2_specificity":0,"D3_justification":0,"D4_solution":0,"D5_tone":0}]}

The review to analyze is below. Output ONLY the JSON object, no other text.

"""


def _extract_json_payload(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("LLM returned empty content.")

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    decoder = json.JSONDecoder()
    for start_index, ch in enumerate(cleaned):
        if ch not in "[{":
            continue
        try:
            _, end_index = decoder.raw_decode(cleaned[start_index:])
            return cleaned[start_index : start_index + end_index]
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Response did not contain valid JSON. Preview: {cleaned[:300]}")


def _load_json_response(response_text: str, label: str) -> dict[str, Any]:
    try:
        return json.loads(_extract_json_payload(response_text))
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} returned invalid JSON: {exc}") from exc


def _validate_arc(arc: dict) -> dict:
    """Normalize and clamp ARC dimension scores to [0, 2]."""
    for dim_key in ("D1_actionability", "D2_specificity", "D3_justification",
                    "D4_solution", "D5_tone"):
        val = arc.get(dim_key)
        if not isinstance(val, (int, float)):
            arc[dim_key] = 0
        else:
            arc[dim_key] = max(0, min(2, int(round(val))))

    if not arc.get("arc_id"):
        arc["arc_id"] = "ARC_00"
    if not arc.get("section"):
        arc["section"] = "Unknown"
    if arc.get("comment_type") not in {"weakness", "strength", "question", "suggestion", "observation"}:
        arc["comment_type"] = "observation"
    if not arc.get("content"):
        arc["content"] = ""
    if not arc.get("anchor_quote"):
        arc["anchor_quote"] = ""

    return arc


class ConstructivenessEvaluator:
    """LLM-as-Judge pipeline for scoring review constructiveness.
    
    Features:
      - Automatic token budget adjustment based on review text length
      - Adaptive max_output_tokens allocation
      - Smart retry mechanism with simplified prompts
    """

    MAX_ATTEMPTS = 2
    
    # ─── Automatic Token Adjustment Configuration ───────────────────────────
    # These thresholds determine max_output_tokens based on review length
    AUTO_ADJUST_ENABLED = True  # Enable automatic adjustment (can be disabled)
    
    # Review length → minimum recommended max_output_tokens.
    #
    # WHY these values are higher than you might expect:
    #   The OUTPUT is a JSON array of ARCs. Each ARC has 8+ fields
    #   (arc_id, section, comment_type, content ~150 chars, anchor_quote ~120 chars,
    #   D1–D5 scores). That's ~450 chars (~130 tokens) per ARC.
    #   A review with 8K chars typically yields 20–25 ARCs
    #   → output ≈ 20 × 130 = 2600 tokens + JSON overhead ≈ 3500+ tokens.
    #   Empirically observed: 8K input → 12K+ chars output (truncated at 3500 tokens).
    #   Use ratio estimate as the primary method; this map is the safety floor.
    TOKEN_BUDGET_MAP = {
        3000:   3000,    # Very short reviews (< 3K chars) — ≤ 10 ARCs
        6000:   4500,    # Short reviews (3–6K chars)     — ~15 ARCs
        10000:  6000,    # Medium reviews (6–10K chars)   — ~20–25 ARCs  ← main fix
        15000:  7000,    # Long reviews (10–15K chars)    — ~30+ ARCs
        20000:  8000,    # Very long reviews (15–20K chars)
        float('inf'): 8000,  # Extremely long (20K+ chars)
    }
    
    # Hard limits
    MIN_OUTPUT_TOKENS = 2500
    MAX_OUTPUT_TOKENS_HARD_LIMIT = 8000

    def __init__(
        self,
        provider: str = "devmate-gemini",
        api_key: str | None = None,
        model: str | None = None,
        auto_adjust_tokens: bool | None = None,
    ):
        self.provider = provider
        self.api_key = api_key
        
        # Determine if auto-adjustment is enabled
        env_auto = os.getenv("CONSTRUCTIVENESS_AUTO_ADJUST_TOKENS", "true").lower()
        if auto_adjust_tokens is not None:
            self.auto_adjust_enabled = auto_adjust_tokens
        else:
            self.auto_adjust_enabled = env_auto in ("true", "1", "yes")
        
        # Get static max_output_tokens (used as override or fallback)
        self.base_max_output_tokens = int(
            os.getenv("CONSTRUCTIVENESS_MAX_OUTPUT_TOKENS", "4096")
        )
        
        # Current max_output_tokens (will be adjusted per review)
        self.max_output_tokens = self.base_max_output_tokens
        
        self.deployment = None

        if provider == "gemini":
            self.client = UnifiedChatClient(
                provider="gemini",
                model=model,
                api_key=self.api_key,
                max_output_tokens=self.base_max_output_tokens,
            )
            self.deployment = self.client.model
        elif provider == "devmate-gemini":
            self.client = UnifiedChatClient(
                provider="gemini-devmate",
                model=model,
                api_key=self.api_key,
                max_output_tokens=self.base_max_output_tokens,
            )
            self.deployment = self.client.model
        elif provider == "azure":
            preferred_gpt5mini = get_preferred_gpt5mini_deployment()
            base_deployment = preferred_gpt5mini or get_default_deployment()
            self.client = AzureChatClient(
                deployment=base_deployment,
                api_key=self.api_key,
                max_output_tokens=self.base_max_output_tokens,
            )
            self.deployment = self._resolve_deployment(base_deployment)
        elif provider == "mimo":
            self.client = UnifiedChatClient(
                provider="mimo",
                model=model,
                api_key=self.api_key,
                max_output_tokens=self.base_max_output_tokens,
            )
            self.deployment = self.client.model
        else:
            raise ValueError(
                f"Unsupported provider: {provider}. Use 'gemini', 'devmate-gemini', 'azure', or 'mimo'."
            )
        
        print(f"[INFO] ConstructivenessEvaluator initialized (auto_adjust_tokens={self.auto_adjust_enabled})")

    def _resolve_deployment(self, default: str) -> str:
        deployment = os.getenv("CONSTRUCTIVENESS_DEPLOYMENT")
        if deployment:
            return deployment
        preferred = get_preferred_gpt5mini_deployment()
        if preferred:
            return preferred
        return (
            os.getenv("AZURE_OPENAI_DEPLOYMENT")
            or os.getenv("AZURE_CHAT_DEPLOYMENT")
            or default
        )

    def _calculate_adaptive_tokens(self, review_text: str, include_paper: bool = False) -> int:
        """Calculate recommended max_output_tokens based on review length.
        
        Uses TWO estimation methods and takes the larger:
        
        Method 1 — Lookup table (safety floor):
          Maps input char ranges to known-safe token budgets.
        
        Method 2 — ARC-count ratio (primary estimate):
          Each ARC in the output JSON costs ~130 output tokens.
          Estimate ~1 ARC per 350 input chars. Apply 1.5× safety margin.
          Formula: max(5, input_chars / 350) × 130 × 1.5
        
        Empirical basis:
          - Review 8K chars → model produced 12K+ chars JSON (truncated at 3500)
          - Actual needed: ~3700+ tokens for ~25 ARCs
          - Ratio method gives: (8000/350) × 130 × 1.5 ≈ 4457 tokens ✓
        
        Returns: Token budget clamped to [MIN, MAX_HARD_LIMIT]
        """
        text_length = len(review_text)
        
        # ── Method 1: Table lookup ─────────────────────────────────────────
        table_budget = self.MIN_OUTPUT_TOKENS
        for threshold in sorted(self.TOKEN_BUDGET_MAP.keys()):
            if text_length < threshold:
                table_budget = self.TOKEN_BUDGET_MAP[threshold]
                break
        
        # ── Method 2: ARC-count ratio estimate ────────────────────────────
        # ~1 ARC per 350 input chars, each ARC ~130 output tokens, 1.5× safety
        estimated_arcs  = max(5, text_length / 350)
        tokens_per_arc  = 130
        safety_factor   = 1.5
        ratio_budget    = int(estimated_arcs * tokens_per_arc * safety_factor)
        
        # Take the larger of the two estimates
        budget = max(table_budget, ratio_budget)
        
        # ── Paper context boost ────────────────────────────────────────────
        # When paper is included, model reads more context → slightly larger output
        if include_paper:
            budget = int(budget * 1.2)
        
        # ── Clamp to hard limits ───────────────────────────────────────────
        budget = max(self.MIN_OUTPUT_TOKENS,
                     min(budget, self.MAX_OUTPUT_TOKENS_HARD_LIMIT))
        
        return budget

    def score_review(
        self,
        review_text: str,
        reviewer_id: str,
        paper_text: str | None = None,
    ) -> dict[str, Any]:
        """Atomize a single review into ARCs and score each on D1–D5.

        Returns dict with keys:
          reviewer_id, atomic_comments, status ("success" | "empty" | "error")
        """
        if not review_text or not review_text.strip():
            return {
                "reviewer_id": reviewer_id,
                "atomic_comments": [],
                "status": "empty_input",
            }

        review_snippet = review_text[:200].replace("\n", " ")
        
        # ─── Automatically adjust max_output_tokens based on review length ───
        if self.auto_adjust_enabled:
            has_paper = paper_text is not None and len(paper_text) > 100
            adaptive_tokens = self._calculate_adaptive_tokens(review_text, include_paper=has_paper)
            # Update the current max_output_tokens
            self.max_output_tokens = adaptive_tokens
            print(
                f"  [DEBUG] Auto-adjusted tokens: "
                f"review={len(review_text):,} chars → max_output_tokens={adaptive_tokens}"
            )
        else:
            self.max_output_tokens = self.base_max_output_tokens

        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            is_retry = attempt > 1
            label = f"attempt {attempt}/{self.MAX_ATTEMPTS}"

            if is_retry:
                print(
                    f"  [RETRY] {reviewer_id} ({label}): "
                    f"previous attempt returned 0 ARCs, sending focused reprompt..."
                )
                system = RETRY_PROMPT
                user = self._build_user_prompt(review_text, None)
            else:
                print(
                    f"  [INFO] Scoring constructiveness for {reviewer_id} "
                    f"(provider={self.provider}, model={self.deployment})..."
                )
                system = SYSTEM_PROMPT
                user = self._build_user_prompt(review_text, paper_text)

            try:
                response_text = self.client.generate_text(
                    system,
                    user,
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_output_tokens=self.max_output_tokens,  # Use current (possibly adjusted) value
                    deployment=self.deployment,
                )

                parsed = _load_json_response(
                    response_text, f"Constructiveness ({reviewer_id} {label})"
                )

                arcs = parsed.get("atomic_comments", [])
                validated = [_validate_arc(arc) for arc in arcs if isinstance(arc, dict)]

                if validated:
                    for i, arc in enumerate(validated, 1):
                        arc["arc_id"] = f"ARC_{i:02d}"
                    print(f"  [OK] {reviewer_id}: {len(validated)} ARCs extracted")
                    return {
                        "reviewer_id": reviewer_id,
                        "atomic_comments": validated,
                        "status": "success",
                    }

                print(
                    f"  [WARN] {reviewer_id} ({label}): "
                    f"JSON parsed OK but atomic_comments is empty"
                )

            except Exception as exc:
                print(
                    f"  [WARN] {reviewer_id} ({label}): "
                    f"{type(exc).__name__}: {str(exc)[:150]}"
                )

        print(f"  [FAIL] {reviewer_id}: all {self.MAX_ATTEMPTS} attempts returned 0 ARCs")
        return {
            "reviewer_id": reviewer_id,
            "atomic_comments": [],
            "status": "failed",
        }

    @staticmethod
    def _build_user_prompt(
        review_text: str,
        paper_text: str | None = None,
    ) -> str:
        parts = []
        if paper_text:
            truncated = paper_text[:30000]
            parts.append(
                f"[PAPER TEXT (for context)]\n{truncated}\n"
            )
        parts.append(
            f"[REVIEW TO ANALYZE]\n\"\"\"\n{review_text}\n\"\"\""
        )
        return "\n\n".join(parts)
