"""
CPS (Critique Prioritization Score) — per-section nDCG formulation (v3).

Measures whether a reviewer front-loads their most severe critiques
**within each structural section** of their review (e.g. Weaknesses,
Suggestions, Summary).

Algorithm
---------
1. LLM extraction assigns each argument a ``section_name``.
2. Position recovery (regex + fuzzy on ``anchor_quote``) re-sorts
   arguments within each section by their true appearance order,
   eliminating LLM ordering bias.
3. Per-section DCG is computed independently:

     CPS_s  = Σ_i  w(severity_i)  / log2(i + 2)   [presentation order]
     ICPS_s = Σ_i  w(sorted_i)    / log2(i + 2)   [ideal: severity desc]

4. Final aggregation follows standard nDCG (Jarvelin & Kekalainen, 2002):

     nCPS = Σ_s CPS_s  /  Σ_s ICPS_s

   This is mathematically equivalent to a weighted average where sections
   with more arguments contribute proportionally, and single-argument
   sections never inflate the score (their CPS_s = ICPS_s contribution
   cancels in the ratio).

Why per-section?
   A reviewer who leads with all Critical issues in "Weaknesses" but buries
   a Critical in "Suggestions" after several Minors should score differently
   from one who consistently front-loads Critical issues in every section.
   Per-section reset captures this intra-section prioritisation signal.
"""

import difflib
import math
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


SEVERITY_MAP: Dict[str, float] = {
    "Critical": 2.0,
    "Minor": 1.0,
    "None": 0.0,
}

_FUZZY_THRESHOLD = 0.45
_DEFAULT_SECTION = "Weaknesses"


def _weight(severity_label: str) -> float:
    return SEVERITY_MAP.get(severity_label, 0.0)


def _filter_critique_weights(arguments: List[Dict]) -> List[float]:
    """Extract non-zero severity weights in original presentation order."""
    return [
        _weight(arg.get("severity", "None"))
        for arg in arguments
        if _weight(arg.get("severity", "None")) > 0.0
    ]


# ---------------------------------------------------------------------------
# Position recovery helpers
# ---------------------------------------------------------------------------

def _find_position_exact(anchor: str, text: str) -> Optional[int]:
    """Exact regex search (escaped, case-insensitive) — returns char offset."""
    m = re.search(re.escape(anchor), text, re.IGNORECASE)
    return m.start() if m else None


def _find_position_fuzzy(anchor: str, text: str) -> Optional[int]:
    """Sliding-window fuzzy match with a tight window for positional accuracy."""
    if not anchor or not text:
        return None
    best_pos: Optional[int] = None
    best_ratio = 0.0
    anchor_lower = anchor.lower()
    text_lower = text.lower()
    win = max(len(anchor_lower), int(len(anchor_lower) * 1.6))
    step = max(1, len(anchor_lower) // 4)
    limit = max(1, len(text_lower) - len(anchor_lower) + 1)
    for start in range(0, limit, step):
        chunk = text_lower[start : min(start + win, len(text_lower))]
        ratio = difflib.SequenceMatcher(None, anchor_lower, chunk).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_pos = start
    return best_pos if best_ratio >= _FUZZY_THRESHOLD else None


def reorder_by_position(
    arguments: List[Dict],
    review_text: str,
) -> List[Dict]:
    """Re-sort *arguments* by their true character position in *review_text*.

    Strategy per argument:
      1. Exact regex on ``anchor_quote``
      2. Fuzzy match on ``anchor_quote``
      3. Fuzzy match on ``content`` (last resort)
      4. Unresolved → pushed to end, preserving relative LLM order
    """
    if not review_text or not arguments:
        return arguments

    UNMATCHED = len(review_text) + 1
    positioned: List[Tuple[int, int, Dict]] = []

    for idx, arg in enumerate(arguments):
        anchor = (arg.get("anchor_quote") or "").strip()
        content = (arg.get("content") or "").strip()
        pos: Optional[int] = None

        if anchor:
            pos = _find_position_exact(anchor, review_text)
            if pos is None:
                pos = _find_position_fuzzy(anchor, review_text)
        if pos is None and content:
            pos = _find_position_fuzzy(content, review_text)

        positioned.append((pos if pos is not None else UNMATCHED, idx, arg))

    positioned.sort(key=lambda t: (t[0], t[1]))
    return [arg for _, _, arg in positioned]


# ---------------------------------------------------------------------------
# Per-section DCG helpers
# ---------------------------------------------------------------------------

def _dcg(weights: List[float]) -> float:
    """Standard DCG: Σ w_i / log2(i + 2)  for i = 0, 1, 2, ..."""
    return sum(w / math.log2(i + 2) for i, w in enumerate(weights))


def _group_by_section(arguments: List[Dict]) -> Dict[str, List[Dict]]:
    """Bucket arguments by section_name, preserving within-bucket order."""
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for arg in arguments:
        section = (arg.get("section_name") or _DEFAULT_SECTION).strip() or _DEFAULT_SECTION
        groups[section].append(arg)
    return dict(groups)


# ---------------------------------------------------------------------------
# Public API: CPS / ICPS / nCPS
# ---------------------------------------------------------------------------

def calculate_cps(arguments: List[Dict]) -> float:
    """Sum of per-section DCG scores (presentation order within each section)."""
    total = 0.0
    for args in _group_by_section(arguments).values():
        total += _dcg(_filter_critique_weights(args))
    return total


def calculate_icps(arguments: List[Dict]) -> float:
    """Sum of per-section ideal DCG scores (severity-descending within each section)."""
    total = 0.0
    for args in _group_by_section(arguments).values():
        total += _dcg(sorted(_filter_critique_weights(args), reverse=True))
    return total


def calculate_ncps(arguments: List[Dict]) -> float:
    """nCPS = Σ CPS_s / Σ ICPS_s  (Option C aggregate — standard nDCG).

    Returns 0.0 when there are no valid critique arguments.
    """
    cps = calculate_cps(arguments)
    icps = calculate_icps(arguments)
    if icps == 0.0:
        return 0.0
    return round(cps / icps, 4)


def calculate_section_breakdown(arguments: List[Dict]) -> List[Dict]:
    """Return per-section CPS/ICPS/nCPS for diagnostic output."""
    breakdown = []
    for section, args in _group_by_section(arguments).items():
        weights = _filter_critique_weights(args)
        cps_s = _dcg(weights)
        icps_s = _dcg(sorted(weights, reverse=True))
        ncps_s = round(cps_s / icps_s, 4) if icps_s > 0 else 0.0
        breakdown.append({
            "section": section,
            "n_args": len(args),
            "CPS_s": round(cps_s, 4),
            "ICPS_s": round(icps_s, 4),
            "nCPS_s": ncps_s,
        })
    breakdown.sort(key=lambda x: x["n_args"], reverse=True)
    return breakdown
