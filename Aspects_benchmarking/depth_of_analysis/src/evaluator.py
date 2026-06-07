import json
import re
from typing import List, Tuple, Dict, Any
from llm_client import PRISMLLMClient

class DepthOfAnalysisEvaluator:
    def __init__(self, api_key: str = None, model: str = None):
        overrides = {}
        if api_key:
            overrides["api_key"] = api_key
        if model:
            overrides["model"] = model
        
        self.client = PRISMLLMClient.for_aspect("depth_of_analysis", **overrides)
        
    def segment_arguments(self, review_text: str) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        system_prompt = (
            "You are a precise peer-review analysis assistant. Your task is to segment a peer review "
            "into a list of individual Argumentative Discourse Units (ADUs). An ADU is a minimal unit "
            "of text (a sentence or clause) that conveys a claim, a premise, an observation, or a recommendation. "
            "Each segmented unit must be a verbatim or near-verbatim substring from the review text."
        )
        
        user_prompt = f"""
Segment the following peer review text into a list of individual Argumentative Discourse Units (ADUs).
Ensure that the output is formatted as a JSON object with a key "arguments" mapping to a list of objects, each containing an "argument" key.

Example output format:
{{
  "arguments": [
    {{
      "argument": "The paper presents a novel framework for image classification."
    }},
    {{
      "argument": "However, the evaluation is lacking comparisons with recent baselines."
    }}
  ]
}}

Review text:
{review_text}
"""
        raw = self.client.generate_text(system_prompt, user_prompt, response_format={"type": "json_object"})
        
        prompt_tokens = len(system_prompt + user_prompt) // 4
        completion_tokens = len(raw) // 4
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        }
        
        parsed = self.client._parse_json(raw)
        arguments = []
        if parsed and "arguments" in parsed:
            arguments = parsed["arguments"]
        elif parsed and isinstance(parsed, list):
            arguments = [{"argument": item} if isinstance(item, str) else item for item in parsed]
        else:
            sentences = re.split(r'(?<=[.!?])\s+', review_text)
            arguments = [{"argument": s.strip()} for s in sentences if s.strip()]
            
        return arguments, usage

    def classify_arguments(self, review_text: str, arguments: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        system_prompt = (
            "You are a precise peer-review analysis assistant. Your task is to classify a list of "
            "Argumentative Discourse Units (ADUs) extracted from a peer review. For each ADU, determine "
            "its Role ('Premise' or 'Claim') and its Aspect ('Soundness', 'Clarity', 'Contribution', or 'Other').\n\n"
            "Definitions:\n"
            "- 'Premise': An ADU providing concrete details, evidence, reasons, or specific support.\n"
            "- 'Claim': An ADU asserting a statement, judgment, or evaluation without direct evidence.\n\n"
            "- 'Soundness': Technical correctness, experiments, methodology, math.\n"
            "- 'Clarity': Writing, presentation, explanations, figures.\n"
            "- 'Contribution': Originality, novelty, significance, baseline comparisons.\n"
            "- 'Other': Anything else (formatting, general greeting)."
        )
        
        adus_list = [a.get("argument", "") for a in arguments]
        user_prompt = f"""
Given the following peer review text, classify the list of ADUs.
Review text:
{review_text}

List of ADUs to classify:
{json.dumps(adus_list, indent=2)}

Respond with a JSON object containing a key "classified_arguments" mapping to a list of objects.
Each object must have "argument", "role", and "aspect" keys.

Example output format:
{{
  "classified_arguments": [
    {{
      "argument": "The paper presents a novel framework for image classification.",
      "role": "Claim",
      "aspect": "Contribution"
    }},
    {{
      "argument": "However, the evaluation is lacking comparisons with recent baselines.",
      "role": "Premise",
      "aspect": "Soundness"
    }}
  ]
}}
"""
        raw = self.client.generate_text(system_prompt, user_prompt, response_format={"type": "json_object"})
        
        prompt_tokens = len(system_prompt + user_prompt) // 4
        completion_tokens = len(raw) // 4
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        }
        
        parsed = self.client._parse_json(raw)
        classified_args = []
        if parsed and "classified_arguments" in parsed:
            classified_args = parsed["classified_arguments"]
        else:
            for arg in arguments:
                classified_args.append({
                    "argument": arg.get("argument", ""),
                    "role": "Claim",
                    "aspect": "Other"
                })
        return classified_args, usage

    def score_grounding(self, review_text: str, premise_texts: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        if not premise_texts:
            return [], {"prompt_tokens": 0, "completion_tokens": 0}
            
        system_prompt = (
            "You are a precise peer-review analysis assistant. Your task is to score the 'Grounding Score' "
            "for a list of Premise arguments from a review. The Grounding Score measures how specific and concrete "
            "the evidence or explanation provided in the premise is.\n\n"
            "Scores:\n"
            "- 0: Generic, vague, or lacks details.\n"
            "- 1: Partially specific (mentions general parts but lacks full details).\n"
            "- 2: Highly specific (cites exact equations, sections, figures, specific metrics, datasets, or precise technical issues)."
        )
        
        user_prompt = f"""
Evaluate the grounding score (0, 1, or 2) for each of the following premises based on the review text.
Review text:
{review_text}

Premises:
{json.dumps(premise_texts, indent=2)}

Respond with a JSON object containing a key "grounding_results" mapping to a list of objects.
Each object must have "premise" and "grounding_score" keys.

Example output format:
{{
  "grounding_results": [
    {{
      "premise": "However, the evaluation is lacking comparisons with recent baselines.",
      "grounding_score": 1
    }}
  ]
}}
"""
        raw = self.client.generate_text(system_prompt, user_prompt, response_format={"type": "json_object"})
        
        prompt_tokens = len(system_prompt + user_prompt) // 4
        completion_tokens = len(raw) // 4
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        }
        
        parsed = self.client._parse_json(raw)
        grounding_results = []
        if parsed and "grounding_results" in parsed:
            grounding_results = parsed["grounding_results"]
        else:
            for p in premise_texts:
                grounding_results.append({
                    "premise": p,
                    "grounding_score": 0
                })
        return grounding_results, usage
