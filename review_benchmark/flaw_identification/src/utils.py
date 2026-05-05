# source/utils.py
import json
import os
import glob
import re

def load_human_meta_json(filepath):
    """Đọc file JSON Human & Meta review"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_llm_txt(filepath):
    """Đọc file TXT LLM review"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def load_paper_mmd(paper_id, mmd_folder):
    """
    Đọc file nội dung bài báo (.mmd).
    Tối ưu: Cắt bỏ "## References" bằng string split siêu tốc, 
    nhưng vẫn quét và giữ lại Appendix ở phía sau.
    """
    mmd_path = os.path.join(mmd_folder, f"{paper_id}.mmd")
    if not os.path.exists(mmd_path):
        print(f"[WARNING] Missing .mmd file for {paper_id}")
        return None
        
    with open(mmd_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Sử dụng hàm split cơ bản: Nhanh hơn và chính xác 100% theo format của bạn
    parts = content.split("## References")
    
    # Nếu không có chữ "## References" nào (mảng parts chỉ có 1 phần tử)
    if len(parts) == 1:
        return content
        
    # Nội dung chính là phần đầu tiên trước "## References"
    core_content = parts[0]
    
    # Nội dung đuôi là tất cả những gì nằm sau "## References"
    # Dùng join trong trường hợp hiếm hoi có nhiều chữ "## References" trong text
    tail_content = "## References".join(parts[1:])
    
    # Quét xem trong phần đuôi có Phụ lục không
    appx_pattern = r'\n#+\s*(Appendix|Appendices|Supplementary)\b'
    appx_match = re.search(appx_pattern, tail_content, re.IGNORECASE)
    
    if appx_match:
        # Nếu có Appendix, nối nó vào ngay sau phần core_content
        appendix_content = tail_content[appx_match.start():]
        core_content += "\n\n" + appendix_content
        print(f"  -> Optimized: trimmed References and kept Appendix for paper {paper_id}")
    else:
        print(f"  -> Optimized: trimmed the References tail for paper {paper_id}")
        
    return core_content
def get_paper_pairs(human_folder, sea_folder):
    """
    Tìm các cặp file (json, txt) trùng tên paper_id.
    """
    human_files = glob.glob(os.path.join(human_folder, "*.json"))
    pairs = []
    
    for h_path in human_files:
        basename = os.path.basename(h_path)
        paper_id = os.path.splitext(basename)[0]
        llm_path = os.path.join(sea_folder, f"{paper_id}.txt")
        
        if os.path.exists(llm_path):
            pairs.append((paper_id, h_path, llm_path))
        else:
            print(f"[WARNING] Missing LLM review for {paper_id}")
            
    return pairs

def format_human_review_text(review_obj):
    """
    Chỉ trích xuất các phần liên quan đến PHẢN BIỆN.
    """
    text_parts = []
    if "Summary" in review_obj:
        text_parts.append(f"### Summary:\n{review_obj['Summary']}")
    if "Weaknesses" in review_obj:
        text_parts.append(f"### Weaknesses:\n{review_obj['Weaknesses']}")
    if "Questions" in review_obj:
        text_parts.append(f"### Questions:\n{review_obj['Questions']}")
    return "\n\n".join(text_parts)


def format_human_review_text_extended(review_obj: dict) -> str:
    """
    Extended version of format_human_review_text that also includes the
    'Limitations' field present in ICLR 2025 / 2026 human reviews.
    Keeps: Summary, Weaknesses, Questions, Limitations.
    """
    text_parts = []
    if "Summary" in review_obj:
        text_parts.append(f"### Summary:\n{review_obj['Summary']}")
    if "Weaknesses" in review_obj:
        text_parts.append(f"### Weaknesses:\n{review_obj['Weaknesses']}")
    if "Questions" in review_obj:
        text_parts.append(f"### Questions:\n{review_obj['Questions']}")
    limitations = str(review_obj.get("Limitations", "")).strip()
    if limitations and limitations.lower() not in {"", "n/a", "na", "none", "not applicable"}:
        text_parts.append(f"### Limitations:\n{limitations}")
    return "\n\n".join(text_parts)


def extract_sea_relevant_sections(raw_text: str) -> str:
    """
    From a SEA review TXT file, keep ONLY the sections relevant to
    flaw identification: Summary, Weaknesses, Questions.

    Drops: Strengths, Rating, Soundness, Presentation, Contribution,
           Paper Decision, etc.

    Handles all ICLR SEA formats (2024 / 2025 / 2026) which use headers
    of the form:
        **Section Name:**          ← standalone header line
        **Section Name:** content  ← header + content on same line
    """
    KEEP = {"summary", "weaknesses", "questions"}

    lines = raw_text.split('\n')

    # SEA files use headers like  **Summary:**  where the colon is INSIDE the
    # bold markers.  We anchor with $ so we do NOT match inline bold phrases
    # such as "- **Limited comparison**: content here..." that appear in bullets.
    #
    # Accepted formats (entire line):
    #   **Section Name:**      ← colon inside  (all observed ICLR SEA files)
    #   **Section Name**:      ← colon outside (rare variant)
    #   **Section Name**       ← no colon
    header_re = re.compile(r'^\s*\*\*([^*\n]+?)\*\*\s*:?\s*$')

    # Collect (line_index, normalised_name) for every section header
    sections: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        m = header_re.match(line)
        if m:
            name = m.group(1).strip().lower().rstrip(':').strip()
            sections.append((i, name))

    if not sections:
        # No bold headers found – return the full text as fallback
        return raw_text

    kept_parts: list[str] = []
    for idx, (start_line, section_name) in enumerate(sections):
        if section_name not in KEEP:
            continue

        end_line = sections[idx + 1][0] if idx + 1 < len(sections) else len(lines)
        section_content = '\n'.join(lines[start_line:end_line]).rstrip()
        if section_content.strip():
            kept_parts.append(section_content)

    if not kept_parts:
        return raw_text  # Fallback: nothing matched, return full text

    return '\n\n'.join(kept_parts)


def format_human_review_text_icml(review_obj: dict) -> str:
    """
    Format a single ICML 2025 human review for flaw identification.

    ICML uses a structured rubric very different from ICLR.  We keep the
    fields most likely to contain critique / weaknesses and drop fields that
    are meta-information or purely positive (e.g. "Code Of Conduct",
    "Supplementary Material").

    Kept fields (in order):
      Summary
      Claims And Evidence
      Methods And Evaluation Criteria
      Theoretical Claims
      Experimental Designs Or Analyses
      Relation To Broader Scientific Literature
      Essential References Not Discussed
      Other Strengths And Weaknesses
      Other Comments Or Suggestions
      Questions For Authors
    """
    FIELDS = [
        ("Summary",                          "### Summary"),
        ("Claims And Evidence",              "### Claims and Evidence"),
        ("Methods And Evaluation Criteria",  "### Methods and Evaluation Criteria"),
        ("Theoretical Claims",               "### Theoretical Claims"),
        ("Experimental Designs Or Analyses", "### Experimental Designs or Analyses"),
        ("Relation To Broader Scientific Literature", "### Relation to Broader Literature"),
        ("Essential References Not Discussed",        "### Essential Missing References"),
        ("Other Strengths And Weaknesses",  "### Other Strengths and Weaknesses"),
        ("Other Comments Or Suggestions",   "### Other Comments or Suggestions"),
        ("Questions For Authors",           "### Questions for Authors"),
    ]
    SKIP_VALUES = {"none", "n/a", "na", "not applicable", "affirmed.", "affirmed"}

    text_parts = []
    for field_key, section_header in FIELDS:
        value = str(review_obj.get(field_key, "")).strip()
        if not value or value.lower() in SKIP_VALUES:
            continue
        text_parts.append(f"{section_header}:\n{value}")

    return "\n\n".join(text_parts)


def format_human_review_text_neurips(review_obj: dict) -> str:
    """
    Format a single NeurIPS 2025 human review for flaw identification.

    NeurIPS uses a rubric similar to ICLR but with important differences:
      - The "Strengths" field sometimes embeds BOTH strengths AND weaknesses
        (reviewers write "**Weaknesses**" / "### Weaknesses" headers inside it),
        while the separate "Weaknesses" field remains empty.
      - "Limitations" is usually just "Yes." (acknowledgment) or substantive text.

    Strategy:
      - Always include Summary and Strengths (the Strengths field is kept even
        when it contains only praise, because it may embed critique).
      - Include Weaknesses if non-empty.
      - Include Questions if non-empty.
      - Include Limitations only when it is substantive (not just "yes/no").
    """
    _YES_ONLY = {"yes", "yes.", "no", "no.", "n/a", "na", "none",
                 "not applicable", "affirmed", "affirmed."}

    text_parts = []

    summary = str(review_obj.get("Summary", "")).strip()
    if summary:
        text_parts.append(f"### Summary:\n{summary}")

    strengths = str(review_obj.get("Strengths", "")).strip()
    if strengths:
        text_parts.append(f"### Strengths (may include weaknesses):\n{strengths}")

    weaknesses = str(review_obj.get("Weaknesses", "")).strip()
    if weaknesses:
        text_parts.append(f"### Weaknesses:\n{weaknesses}")

    questions = str(review_obj.get("Questions", "")).strip()
    if questions:
        text_parts.append(f"### Questions:\n{questions}")

    limitations = str(review_obj.get("Limitations", "")).strip()
    if limitations and limitations.lower() not in _YES_ONLY:
        text_parts.append(f"### Limitations:\n{limitations}")

    return "\n\n".join(text_parts)


def load_tree_review_from_path(tree_path: str) -> str | None:
    """
    Load a tree-review JSON file (named  {paper_id}_review.json),
    extract the ``full_review`` text, then apply
    ``extract_sea_relevant_sections`` to keep only Summary / Weaknesses /
    Questions.  Returns None if the file is missing or the field is empty.
    """
    try:
        with open(tree_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[WARNING] Could not load tree review {tree_path}: {e}")
        return None
    full_review = str(data.get("full_review", "")).strip()
    if not full_review:
        return None
    return extract_sea_relevant_sections(full_review)


def get_paper_pairs_tree(human_folder: str, tree_folder: str):
    """
    Find (paper_id, h_path, tree_path) pairs where tree-review files are
    named  {paper_id}_review.json  (unlike SEA files which are {paper_id}.txt).
    """
    human_files = glob.glob(os.path.join(human_folder, "*.json"))
    pairs = []
    for h_path in human_files:
        paper_id = os.path.splitext(os.path.basename(h_path))[0]
        tree_path = os.path.join(tree_folder, f"{paper_id}_review.json")
        if os.path.exists(tree_path):
            pairs.append((paper_id, h_path, tree_path))
        else:
            print(f"[WARNING] Missing tree review for {paper_id}")
    return pairs


def extract_from_deepreview_json(file_path: str) -> str:
    """
    Extract Summary, Strengths, Weaknesses from a deepreview JSON file.
    Only uses reviewer_id == 1 (first reviewer as fallback).

    Two layouts are handled transparently:

      Layout A  (ICLR2024, ICLR2025, NeurIPS2025)
        generated_review[0].reviews  is populated
        → use reviews[reviewer_id==1].text  (pre-parsed, uses ### headers)

      Layout B  (ICLR2026, ICML2025)
        generated_review[0].reviews  is empty
        → parse raw_text for the "## Reviewer 1" block first,
          then extract ### sections from that block.

    Sections extracted: Summary, Weaknesses, Questions.
    Strengths intentionally excluded (not relevant for flaw identification).
    Fallback: returns the full Reviewer-1 text when section parsing fails.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    generated = data.get("generated_review", [])
    if not generated:
        return ""

    entry = generated[0]
    reviews = entry.get("reviews") or []
    text = ""

    # ── Layout A: structured reviews array ────────────────────────────────
    if reviews:
        reviewer1 = next(
            (r for r in reviews if r.get("reviewer_id") == 1),
            reviews[0],
        )
        text = str(reviewer1.get("text", "")).strip()

    # ── Layout B: empty reviews → find ## Reviewer 1 in raw_text ─────────
    if not text:
        raw_text = str(entry.get("raw_text", "")).strip()
        if raw_text:
            m = re.search(
                r"##\s+Reviewer\s+1\s*\n(.*?)(?=\n##\s+Reviewer\s+\d|\Z)",
                raw_text,
                re.DOTALL | re.IGNORECASE,
            )
            text = m.group(1).strip() if m else raw_text

    if not text:
        return ""

    # ── Extract ### Summary, ### Weaknesses, ### Questions ───────────────
    # Strengths intentionally excluded (not relevant for flaw identification).
    target_sections = [
        ("Summary",    "Summary"),
        ("Weaknesses", "Weaknesses"),
        ("Questions",  "Questions"),
    ]

    parts: list[str] = []
    for section_name, label in target_sections:
        pattern = rf"###\s+{re.escape(section_name)}\s*\n(.*?)(?=\n###\s|\Z)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            # Strip trailing separator lines (--- / ===)
            content = re.sub(r"[\n\r]+[-=]{3,}\s*$", "", content).strip()
            if content:
                parts.append(f"**{label}:**\n{content}")

    return "\n\n".join(parts) if parts else text


def get_paper_pairs_deepreview(human_folder: str, deepreview_folder: str):
    """
    Find (paper_id, h_path, dr_path) pairs for deepreview .json files.

    Both human and deepreview files are named {paper_id}.json, but live
    in different directories.
    """
    human_files = glob.glob(os.path.join(human_folder, "*.json"))
    pairs = []
    for h_path in human_files:
        paper_id = os.path.splitext(os.path.basename(h_path))[0]
        dr_path = os.path.join(deepreview_folder, f"{paper_id}.json")
        if os.path.exists(dr_path):
            pairs.append((paper_id, h_path, dr_path))
        else:
            print(f"[WARNING] Missing deepreview file for {paper_id}")
    return pairs


def extract_from_reviewer2_txt(raw_text: str) -> str:
    """
    Extract core critique sections from a Reviewer2 TXT file.

    Format: may contain a thinking block before </think>, followed by
    ## or ### section headers (H2 or H3 — both are handled).

    Sections extracted (in order):
        ## (or ###) Summary Of The Paper         → "Summary"
        ## (or ###) Weaknesses                   → "Weaknesses"
        ## (or ###) Questions For The Authors    → "Questions"
        ## (or ###) Limitations Not Addressed By The Authors
        ## (or ###) Brief Justification For Rating

    Strengths is intentionally excluded (not relevant for flaw ID).
    Fallback: returns the full post-think text when no sections are matched.
    """
    # 1. Strip thinking block (everything up to and including </think>)
    think_end = raw_text.find("</think>")
    if think_end != -1:
        text = raw_text[think_end + len("</think>"):].strip()
    else:
        text = raw_text.strip()

    # 2. Define target sections (section_name_regex, display_label)
    target_sections = [
        (r"Summary\s+Of\s+The\s+Paper",                "Summary"),
        (r"Weaknesses",                                 "Weaknesses"),
        (r"Questions\s+For\s+(?:The\s+)?Authors?",      "Questions"),
        (r"Limitations\s+Not\s+Addressed\s+By\s+The\s+Authors",
                                                        "Limitations Not Addressed By The Authors"),
        (r"Brief\s+Justification\s+For\s+Rating",       "Brief Justification For Rating"),
    ]

    parts: list[str] = []
    for section_re, label in target_sections:
        # Match ## or ### followed by the section name (trailing spaces OK),
        # capture everything until the next ## / ### header or end-of-string.
        pattern = rf"(?m)^#{{2,3}}\s+{section_re}\s*$\n(.*?)(?=\n#{{2,3}}\s|\Z)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            # Remove trailing separator lines ("---" / "===")
            content = re.sub(r'[\n\r]+[-=]{3,}\s*$', '', content).strip()
            if content:
                parts.append(f"**{label}:**\n{content}")

    # Fallback: if nothing matched return full text after </think>
    return "\n\n".join(parts) if parts else text


def get_paper_pairs_reviewer2(human_folder: str, reviewer2_folder: str):
    """
    Find (paper_id, h_path, r2_path) pairs for Reviewer2 .txt files.

    Reviewer2 files follow the same naming convention as SEA files:
    {paper_id}.txt  →  paired with  {paper_id}.json  in human_folder.
    """
    return get_paper_pairs(human_folder, reviewer2_folder)


def load_paper_content(paper_id: str, papers_folder: str) -> str | None:
    """
    Universal paper loader that tries multiple file extensions:
      1. {paper_id}.mmd        – Nougat-extracted markdown (ICLR 2024)
      2. {paper_id}.grobid.txt – GROBID-extracted text    (ICLR 2025)
      3. {paper_id}.txt        – plain text extraction     (ICLR 2026)

    For .mmd files the existing References-trimming logic is applied.
    For .grobid.txt and .txt files the content is returned as-is
    (the pipeline already caps input size via step2_max_input_chars).
    """
    # 1. Try .mmd  (with References trimming)
    mmd_path = os.path.join(papers_folder, f"{paper_id}.mmd")
    if os.path.exists(mmd_path):
        return load_paper_mmd(paper_id, papers_folder)

    # 2. Try .grobid.txt
    grobid_path = os.path.join(papers_folder, f"{paper_id}.grobid.txt")
    if os.path.exists(grobid_path):
        with open(grobid_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"  -> Loaded GROBID text for {paper_id} ({len(content):,} chars)")
        return content

    # 3. Try .txt
    txt_path = os.path.join(papers_folder, f"{paper_id}.txt")
    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"  -> Loaded plain text for {paper_id} ({len(content):,} chars)")
        return content

    print(f"[WARNING] No paper file found for {paper_id} in {papers_folder}")
    return None


def extract_from_cyclereview_json(file_path: str) -> str:
    """
    Extract Summary, Weaknesses, Questions from a CycleReview JSON file.
    Uses ONLY the first reviewer's text (reviews[0]).

    CycleReview structure:
        generated_review.reviews[]  – per-reviewer text (## Reviewer + ### sections)
        generated_review.content    – all reviewers concatenated (fallback)
        generated_review.weaknesses[]/questions[] – pre-parsed arrays (last resort)

    Sections extracted: Summary, Weaknesses, Questions.
    Strengths intentionally excluded (not relevant for flaw identification).
    Fallback: returns the first reviewer's raw text when section parsing fails.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    generated       = data.get("generated_review", {})
    reviews         = generated.get("reviews", [])
    weaknesses_list = generated.get("weaknesses", [])
    questions_list  = generated.get("questions", [])

    # ── 1. Primary: reviews[0] ────────────────────────────────────────────────
    raw = reviews[0].strip() if reviews else ""

    # ── 2. Fallback: first ## Reviewer block from content ─────────────────────
    if not raw:
        content_field = str(generated.get("content", "")).strip()
        if content_field:
            blocks = re.split(r"(?m)^##\s+Reviewer\b", content_field)
            if len(blocks) >= 2:
                raw = ("## Reviewer\n" + blocks[1]).strip()
            elif blocks:
                raw = blocks[0].strip()

    if not raw:
        # Last resort: rebuild from pre-parsed arrays (no Strengths)
        parts: list[str] = []
        if weaknesses_list and str(weaknesses_list[0]).strip():
            parts.append(f"**Weaknesses:**\n{weaknesses_list[0].strip()}")
        if questions_list and str(questions_list[0]).strip():
            parts.append(f"**Questions:**\n{questions_list[0].strip()}")
        return "\n\n".join(parts)

    # ── Extract ### Summary, ### Weaknesses, ### Questions ────────────────────
    # Strengths intentionally excluded (not relevant for flaw identification).
    target_sections = [
        ("Summary",    "Summary"),
        ("Weaknesses", "Weaknesses"),
        ("Questions",  "Questions"),
    ]

    parts: list[str] = []
    for section_name, label in target_sections:
        pattern = rf"###\s+{re.escape(section_name)}\s*\n(.*?)(?=\n###\s|\Z)"
        match = re.search(pattern, raw, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            content = re.sub(r"[\n\r]+[-=]{3,}\s*$", "", content).strip()
            if content:
                parts.append(f"**{label}:**\n{content}")

    # Fallback to pre-parsed arrays if regex failed (no Strengths)
    if not parts:
        if weaknesses_list and str(weaknesses_list[0]).strip():
            parts.append(f"**Weaknesses:**\n{weaknesses_list[0].strip()}")
        if questions_list and str(questions_list[0]).strip():
            parts.append(f"**Questions:**\n{questions_list[0].strip()}")

    return "\n\n".join(parts) if parts else raw


def get_paper_pairs_cyclereview(human_folder: str, cyclereview_folder: str):
    """
    Find (paper_id, h_path, cr_path) pairs for CycleReview .json files.

    Both human and cyclereview files are named {paper_id}.json, but live
    in different directories.
    """
    human_files = glob.glob(os.path.join(human_folder, "*.json"))
    pairs = []
    for h_path in human_files:
        paper_id = os.path.splitext(os.path.basename(h_path))[0]
        cr_path = os.path.join(cyclereview_folder, f"{paper_id}.json")
        if os.path.exists(cr_path):
            pairs.append((paper_id, h_path, cr_path))
        else:
            print(f"[WARNING] Missing cyclereview file for {paper_id}")
    return pairs

