import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from treereview.models.paper import Paper


class PhaseInputAdapter:
    """
    Thin adapter between the upstream preparation pipeline and TreeReview.

    Design rule:
    - TreeReview generation consumes only parsed paper text.
    - Human reviews are standardized and attached to outputs for evaluation,
      but are never injected into TreeReview prompts.
    """

    REVIEW_FIELD_ALIASES = {
        "summary": ["Summary", "summary"],
        "strengths": ["Strengths", "strengths"],
        "weaknesses": ["Weaknesses", "weaknesses"],
        "questions": ["Questions", "questions"],
        "soundness": ["Soundness", "soundness"],
        "presentation": ["Presentation", "presentation"],
        "contribution": ["Contribution", "contribution"],
        "confidence": ["Confidence", "confidence"],
        "rating": ["Rating", "rating"],
    }

    META_FIELD_ALIASES = {
        "metareview": ["Metareview", "meta_review", "MetaReview", "meta review"],
        "justification_not_higher": [
            "Justification For Why Not Higher Score",
            "justification_for_why_not_higher_score",
        ],
        "justification_not_lower": [
            "Justification For Why Not Lower Score",
            "justification_for_why_not_lower_score",
        ],
    }

    def __init__(self, paper_id: str, paper_path: str, reviews_json_path: str):
        self.paper_id = paper_id
        self.paper_path = str(paper_path)
        self.reviews_json_path = str(reviews_json_path)

    def build_bundle(self, paper: Paper) -> Dict[str, Any]:
        raw_reviews = self.load_reviews_json(self.reviews_json_path)
        standardized_reviews = self.standardize_reviews(raw_reviews)
        return {
            "paper_id": self.paper_id,
            "paper_input": {
                "paper_path": self.paper_path,
                "title": paper.title,
                "abstract": paper.abstract,
                "toc": paper.toc,
                "num_chunks": len(paper.chunks),
            },
            "human_reviews": standardized_reviews,
        }

    @staticmethod
    def load_reviews_json(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("reviews JSON must be a top-level object/dictionary.")
        return data

    def standardize_reviews(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        decision = self._clean_text(raw_data.get("Decision", raw_data.get("decision", "")))
        meta_block = raw_data.get("Meta review", raw_data.get("meta_review", {})) or {}
        reviews = raw_data.get("reviews", []) or []

        standardized_meta = {
            "metareview": self._first_present(meta_block, self.META_FIELD_ALIASES["metareview"]),
            "justification_not_higher": self._first_present(
                meta_block, self.META_FIELD_ALIASES["justification_not_higher"]
            ),
            "justification_not_lower": self._first_present(
                meta_block, self.META_FIELD_ALIASES["justification_not_lower"]
            ),
        }

        standardized_reviews: List[Dict[str, Any]] = []
        rendered_reviews: List[str] = []
        for idx, review in enumerate(reviews, start=1):
            normalized = {"review_index": idx}
            for out_key, aliases in self.REVIEW_FIELD_ALIASES.items():
                normalized[out_key] = self._first_present(review, aliases)
            normalized["raw_review"] = review
            standardized_reviews.append(normalized)
            rendered_reviews.append(self._render_review_block(normalized))

        return {
            "decision": decision,
            "meta_review": standardized_meta,
            "reviews": standardized_reviews,
            "review_count": len(standardized_reviews),
            "standardized_human_reviews_text": self._render_all_reviews(
                decision=decision,
                meta_review=standardized_meta,
                rendered_reviews=rendered_reviews,
            ),
        }

    @staticmethod
    def save_json(output_path: str, payload: Dict[str, Any]) -> None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _clean_text(value: Optional[Any]) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return str(value).strip()

    def _first_present(self, data: Dict[str, Any], aliases: List[str]) -> str:
        for key in aliases:
            if key in data and data[key] is not None:
                return self._clean_text(data[key])
        return ""

    def _render_review_block(self, review: Dict[str, Any]) -> str:
        parts = [f"[Review {review['review_index']}]" ]
        ordered_keys = [
            ("summary", "Summary"),
            ("strengths", "Strengths"),
            ("weaknesses", "Weaknesses"),
            ("questions", "Questions"),
            ("soundness", "Soundness"),
            ("presentation", "Presentation"),
            ("contribution", "Contribution"),
            ("confidence", "Confidence"),
            ("rating", "Rating"),
        ]
        for key, label in ordered_keys:
            value = review.get(key, "")
            parts.append(f"{label}: {value}" if value else f"{label}: ")
        return "\n".join(parts)

    def _render_all_reviews(
        self,
        decision: str,
        meta_review: Dict[str, str],
        rendered_reviews: List[str],
    ) -> str:
        blocks = [
            f"Decision: {decision}",
            "[Meta Review]",
            f"Metareview: {meta_review.get('metareview', '')}",
            "Justification For Why Not Higher Score: " + meta_review.get("justification_not_higher", ""),
            "Justification For Why Not Lower Score: " + meta_review.get("justification_not_lower", ""),
        ]
        blocks.extend(rendered_reviews)
        return "\n\n".join(blocks)
