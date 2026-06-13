"""
Data loading utilities for Constructiveness Evaluation pipeline.
Reuses data formats from flaw_identification but extracts ALL review sections
(including Strengths) for constructiveness analysis.
"""

import glob
import json
import os
import re
from typing import Optional


def load_human_meta_json(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_llm_txt(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def load_paper_grobid(paper_id: str, papers_folder: str) -> Optional[str]:
    """Load paper full text from GROBID-extracted .grobid.txt file."""
    grobid_path = os.path.join(papers_folder, f"{paper_id}.grobid.txt")
    if not os.path.exists(grobid_path):
        print(f"  [WARNING] Missing .grobid.txt file for {paper_id}")
        return None

    with open(grobid_path, "r", encoding="utf-8") as f:
        return f.read()


def format_human_review_full(review_obj: dict) -> str:
    """Format ALL sections of a human review for constructiveness analysis.

    Unlike flaw_identification's format_human_review_text (Summary + Weaknesses +
    Questions only), this includes Strengths and any other textual fields because
    constructiveness assessment needs the complete review.
    """
    section_order = ["Summary", "Strengths", "Weaknesses", "Questions"]
    parts = []
    for section in section_order:
        if section in review_obj and review_obj[section]:
            text = str(review_obj[section]).strip()
            if text:
                parts.append(f"### {section}:\n{text}")

    for key, value in review_obj.items():
        if key in section_order:
            continue
        if key in {"Soundness", "Presentation", "Contribution", "Confidence", "Rating"}:
            continue
        if isinstance(value, str) and value.strip():
            parts.append(f"### {key}:\n{value.strip()}")

    return "\n\n".join(parts)


def extract_human_rating(review_obj: dict) -> Optional[int]:
    """Extract numeric rating from a human review for subgroup analysis."""
    rating_str = review_obj.get("Rating", "")
    if not rating_str:
        return None
    match = re.search(r"(\d+)", str(rating_str))
    return int(match.group(1)) if match else None


def extract_human_confidence(review_obj: dict) -> Optional[int]:
    """Extract numeric confidence from a human review for subgroup analysis."""
    conf_str = review_obj.get("Confidence", "")
    if not conf_str:
        return None
    match = re.search(r"(\d+)", str(conf_str))
    return int(match.group(1)) if match else None


def get_paper_pairs(human_folder: str, sea_folder: str) -> list[tuple[str, str, str]]:
    human_files = glob.glob(os.path.join(human_folder, "*.json"))
    pairs = []
    for h_path in human_files:
        basename = os.path.basename(h_path)
        paper_id = os.path.splitext(basename)[0]
        llm_path = os.path.join(sea_folder, f"{paper_id}.txt")
        if os.path.exists(llm_path):
            pairs.append((paper_id, h_path, llm_path))
        else:
            print(f"  [WARNING] Missing LLM review for {paper_id}")
    return pairs


def load_paper_metadata(human_data: dict) -> dict:
    """Extract paper-level metadata for subgroup analysis."""
    decision = human_data.get("Decision", "")
    reviews = human_data.get("reviews", [])
    if isinstance(human_data, list):
        reviews = human_data

    ratings = []
    confidences = []
    for r in reviews:
        rating = extract_human_rating(r)
        if rating is not None:
            ratings.append(rating)
        conf = extract_human_confidence(r)
        if conf is not None:
            confidences.append(conf)

    return {
        "decision": decision,
        "avg_rating": sum(ratings) / len(ratings) if ratings else None,
        "avg_confidence": sum(confidences) / len(confidences) if confidences else None,
        "num_reviewers": len(reviews),
    }


def load_deepreview_text(filepath: str, reviewer_id: int = 1) -> str:
    """Load a deepreview JSON and return the text for the specified reviewer_id.

    Structure: generated_review[0].reviews[].{reviewer_id, text, ...}
    Falls back to the first review if reviewer_id is not found.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        d = json.load(f)
    generated = d.get("generated_review", [])
    if not generated:
        return ""
    reviews = generated[0].get("reviews", [])
    # Try to find the exact reviewer_id
    for r in reviews:
        if r.get("reviewer_id") == reviewer_id:
            return r.get("text", "").strip()
    # Fallback: first review
    if reviews:
        return reviews[0].get("text", "").strip()
    return ""


def load_tree_review_text(filepath: str) -> str:
    """Load a tree_iclr2024_full review JSON and return the full_review text.

    Structure: {"full_review": "<string>"}
    """
    with open(filepath, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d.get("full_review", "").strip()


def load_reviewer2_txt(filepath: str) -> str:
    """Load and clean a reviewer2_iclr2024 .txt file.

    Format:
        PAPER: <title>
        ================================================================================
        REVIEW
        --------------------------------------------------------------------------------
        <think>
        ... internal reasoning ...
        </think>

        ## Summary Of The Paper
        ...
        ## Strengths
        ...

    Returns only the structured review content after </think>, with the
    PAPER/REVIEW header stripped.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Strip the <think>...</think> block (may or may not be present)
    think_end = content.find("</think>")
    if think_end != -1:
        content = content[think_end + len("</think>"):].strip()
    else:
        # No <think> block — strip the header lines manually
        # Remove "PAPER: ...", "===...", "REVIEW", "---..."
        lines = content.splitlines()
        cleaned = []
        skip_header = True
        for line in lines:
            if skip_header:
                stripped = line.strip()
                if (
                    stripped.startswith("PAPER:")
                    or stripped.startswith("===")
                    or stripped == "REVIEW"
                    or stripped.startswith("---")
                    or stripped == ""
                ):
                    continue
                else:
                    skip_header = False
            cleaned.append(line)
        content = "\n".join(cleaned).strip()

    # Remove trailing footer separator if present
    content = re.sub(r"\n={40,}\s*$", "", content).strip()

    return content


def format_human_review_full_icml(review_obj: dict) -> str:
    """Format ALL sections of an ICML2025 human review for constructiveness analysis.

    ICML2025 review fields:
      Review ID, Overall Recommendation, Summary, Claims And Evidence,
      Methods And Evaluation Criteria, Theoretical Claims,
      Experimental Designs Or Analyses, Supplementary Material,
      Relation To Broader Scientific Literature,
      Essential References Not Discussed, Other Strengths And Weaknesses,
      Other Comments Or Suggestions, Questions For Authors, Code Of Conduct
    """
    # Fields to include (skip meta/non-textual fields)
    SKIP_FIELDS = {"Review ID", "Overall Recommendation", "Code Of Conduct"}

    # Priority order for key sections
    section_order = [
        "Summary",
        "Claims And Evidence",
        "Methods And Evaluation Criteria",
        "Theoretical Claims",
        "Experimental Designs Or Analyses",
        "Relation To Broader Scientific Literature",
        "Essential References Not Discussed",
        "Other Strengths And Weaknesses",
        "Other Comments Or Suggestions",
        "Questions For Authors",
        "Supplementary Material",
    ]

    parts = []
    seen = set()
    for section in section_order:
        if section in review_obj and review_obj[section]:
            text = str(review_obj[section]).strip()
            if text:
                parts.append(f"### {section}:\n{text}")
                seen.add(section)

    # Include any remaining text fields not already added
    for key, value in review_obj.items():
        if key in seen or key in SKIP_FIELDS:
            continue
        if isinstance(value, str) and value.strip():
            parts.append(f"### {key}:\n{value.strip()}")

    return "\n\n".join(parts)


def extract_human_rating_icml(review_obj: dict) -> Optional[int]:
    """Extract numeric rating from an ICML2025 human review.
    Field name is 'Overall Recommendation' (e.g. '6', '4', '8').
    """
    rating_str = review_obj.get("Overall Recommendation", "")
    if not rating_str:
        return None
    match = re.search(r"(\d+)", str(rating_str))
    return int(match.group(1)) if match else None


def load_paper_metadata_icml(human_data: dict) -> dict:
    """Extract paper-level metadata for ICML2025 subgroup analysis.

    Uses 'Overall Recommendation' instead of 'Rating'; no Confidence field.
    """
    decision = human_data.get("Decision", "")
    reviews = human_data.get("reviews", [])
    if isinstance(human_data, list):
        reviews = human_data

    ratings = []
    for r in reviews:
        rating = extract_human_rating_icml(r)
        if rating is not None:
            ratings.append(rating)

    return {
        "decision": decision,
        "avg_rating": sum(ratings) / len(ratings) if ratings else None,
        "avg_confidence": None,   # Not available in ICML2025
        "num_reviewers": len(reviews),
    }


def get_paper_pairs_from_ids(
    human_folder: str,
    paper_ids_file: str | None,
) -> list[tuple[str, str]]:
    """Return list of (paper_id, human_path) for papers listed in paper_ids_file.

    If the optional manifest is absent, discover IDs from human_folder.
    Only papers whose .json file exists in human_folder are returned.
    """
    if paper_ids_file and os.path.exists(paper_ids_file):
        with open(paper_ids_file, "r", encoding="utf-8") as f:
            ids = [line.strip() for line in f if line.strip()]
    else:
        ids = sorted(
            os.path.splitext(name)[0]
            for name in os.listdir(human_folder)
            if name.endswith(".json")
        )

    pairs = []
    for pid in ids:
        h_path = os.path.join(human_folder, f"{pid}.json")
        if os.path.exists(h_path):
            pairs.append((pid, h_path))
        else:
            print(f"  [WARNING] Missing human JSON for {pid}")
    return pairs


def format_human_review_full_neurips(review_obj: dict) -> str:
    """Format ALL textual sections of a NeurIPS 2025 human review.

    NeurIPS 2025 review fields:
      Review ID, Rating, Confidence, Summary, Soundness, Presentation,
      Contribution, Strengths, Weaknesses, Questions, Limitations,
      Ethical Concerns, Flag For Ethics Review, Code Of Conduct

    Skips non-textual / meta fields; adds Limitations as a new section
    compared to ICLR2024.
    """
    SKIP_FIELDS = {
        "Review ID", "Rating", "Confidence",
        "Soundness", "Presentation", "Contribution",
        "Ethical Concerns", "Flag For Ethics Review", "Code Of Conduct",
    }

    section_order = [
        "Summary",
        "Strengths",
        "Weaknesses",
        "Questions",
        "Limitations",
    ]

    parts = []
    seen = set()
    for section in section_order:
        if section in review_obj and review_obj[section]:
            text = str(review_obj[section]).strip()
            if text:
                parts.append(f"### {section}:\n{text}")
                seen.add(section)

    # Include any remaining text fields not already handled
    for key, value in review_obj.items():
        if key in seen or key in SKIP_FIELDS:
            continue
        if isinstance(value, str) and value.strip():
            parts.append(f"### {key}:\n{value.strip()}")

    return "\n\n".join(parts)


def load_paper_metadata_neurips(human_data: dict) -> dict:
    """Extract paper-level metadata for NeurIPS 2025 subgroup analysis.

    NeurIPS top-level fields include title, abstract, primary_area,
    keywords, decision, statistics (avg_rating etc.).
    Uses 'Rating' and 'Confidence' from reviews (same as ICLR2024).
    """
    decision     = human_data.get("Decision", "")
    reviews      = human_data.get("reviews", [])

    ratings = []
    confidences = []
    for r in reviews:
        rating = extract_human_rating(r)
        if rating is not None:
            ratings.append(rating)
        conf = extract_human_confidence(r)
        if conf is not None:
            confidences.append(conf)

    return {
        "decision":       decision,
        "title":          human_data.get("title", ""),
        "primary_area":   human_data.get("primary_area", ""),
        "keywords":       human_data.get("keywords", []),
        "avg_rating":     sum(ratings) / len(ratings) if ratings else None,
        "avg_confidence": sum(confidences) / len(confidences) if confidences else None,
        "num_reviewers":  len(reviews),
    }


def load_cyclereview_first_text(filepath: str) -> str:
    """Load only the FIRST reviewer's text from a CycleReview JSON file.

    CycleReview structure:
        generated_review.reviews[]    – per-reviewer text with ### headers (primary)
        generated_review.content      – all reviewers concatenated, separated by **********
        generated_review.weaknesses[] – pre-parsed weaknesses (parallel array, fallback)
        generated_review.questions[]  – pre-parsed questions (parallel array, fallback)

    Extraction priority:
      1. reviews[0]  — cleanest source; directly the first reviewer's block
      2. content     — parse first ## Reviewer block (fallback if reviews is empty)
      3. Pre-parsed weaknesses[0] + questions[0] arrays (last resort)

    Sections extracted: Summary, Weaknesses, Questions.
    Returns a single formatted string.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        d = json.load(f)

    generated       = d.get("generated_review", {})
    reviews         = generated.get("reviews", [])
    weaknesses_list = generated.get("weaknesses", [])
    questions_list  = generated.get("questions", [])

    # ── 1. Primary: reviews[0] ────────────────────────────────────────────────
    raw = reviews[0].strip() if reviews else ""

    # ── 2. Fallback: first ## Reviewer block from content ─────────────────────
    if not raw:
        content_field = str(generated.get("content", "")).strip()
        if content_field:
            # Split on ## Reviewer header, keep first block
            # (content can have hundreds of repeated "## Paper Decision" lines at end)
            blocks = re.split(r"(?m)^##\s+Reviewer\b", content_field)
            # blocks[0] is before any "## Reviewer" — skip it; blocks[1] is first reviewer
            if len(blocks) >= 2:
                raw = ("## Reviewer\n" + blocks[1]).strip()
            elif blocks:
                raw = blocks[0].strip()

    if not raw:
        # Last resort: rebuild from pre-parsed arrays
        parts: list[str] = []
        if weaknesses_list and str(weaknesses_list[0]).strip():
            parts.append(f"**Weaknesses:**\n{weaknesses_list[0].strip()}")
        if questions_list and str(questions_list[0]).strip():
            parts.append(f"**Questions:**\n{questions_list[0].strip()}")
        return "\n\n".join(parts)

    # ── Extract Summary, Weaknesses, Questions via ### headers ────────────────
    parts = []
    for section_name, label in [
        ("Summary",    "Summary"),
        ("Weaknesses", "Weaknesses"),
        ("Questions",  "Questions"),
    ]:
        pattern = rf"###\s+{re.escape(section_name)}\s*\n(.*?)(?=\n###\s|\Z)"
        m = re.search(pattern, raw, re.DOTALL | re.IGNORECASE)
        if m:
            content = m.group(1).strip()
            content = re.sub(r"[\n\r]+[-=]{3,}\s*$", "", content).strip()
            if content:
                parts.append(f"**{label}:**\n{content}")

    # Fallback to pre-parsed arrays if regex failed
    if not parts:
        if weaknesses_list and str(weaknesses_list[0]).strip():
            parts.append(f"**Weaknesses:**\n{weaknesses_list[0].strip()}")
        if questions_list and str(questions_list[0]).strip():
            parts.append(f"**Questions:**\n{questions_list[0].strip()}")

    return "\n\n".join(parts) if parts else raw


def load_cyclereview_metadata(filepath: str) -> dict:
    """Extract metadata from a CycleReview JSON (uses ground_truth fields).

    Returns dict with decision, avg_rating, individual_ratings, num_reviewers.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        d = json.load(f)

    gt       = d.get("ground_truth", {})
    generated = d.get("generated_review", {})
    reviews  = generated.get("reviews", [])

    return {
        "decision":           gt.get("decision", ""),
        "avg_rating":         gt.get("avg_rating", None),
        "individual_ratings": gt.get("individual_ratings", []),
        "num_reviewers":      len(reviews),
    }


def get_cyclereview_pairs_from_ids(
    cyclereview_folder: str,
    paper_ids_file: str | None,
) -> list[tuple[str, str]]:
    """Return list of (paper_id, cyclereview_path) for papers listed in paper_ids_file.

    If the optional manifest is absent, discover IDs from cyclereview_folder.
    Only papers whose .json file exists in cyclereview_folder are returned.
    """
    if paper_ids_file and os.path.exists(paper_ids_file):
        with open(paper_ids_file, "r", encoding="utf-8") as f:
            ids = [line.strip() for line in f if line.strip()]
    else:
        ids = sorted(
            os.path.splitext(name)[0]
            for name in os.listdir(cyclereview_folder)
            if name.endswith(".json")
        )

    pairs = []
    for pid in ids:
        cr_path = os.path.join(cyclereview_folder, f"{pid}.json")
        if os.path.exists(cr_path):
            pairs.append((pid, cr_path))
        else:
            print(f"  [WARNING] Missing CycleReview JSON for {pid}")
    return pairs


def truncate_review_text(review_text: str, max_chars: int = 20000) -> str:
    """
    Truncate review text if it exceeds max_chars (default 20K).
    Helps avoid MAX_TOKENS truncation in LLM calls.
    
    Args:
        review_text: Text to potentially truncate
        max_chars: Maximum characters to keep (default 20000)
    
    Returns:
        Original or truncated text with "[...truncated...]" marker
    """
    if len(review_text) <= max_chars:
        return review_text
    
    truncated = review_text[:max_chars]
    
    # Try to cut at sentence boundary (period)
    last_period = truncated.rfind(".")
    if last_period > max_chars * 0.8:  # Only if close to the end (>80%)
        truncated = truncated[:last_period + 1]
    
    return truncated + "\n\n[...truncated...]"
