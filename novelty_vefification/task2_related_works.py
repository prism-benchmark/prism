from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from services.semantic_scholar_client import SemanticScholarClientPool
from utils.paper_id import make_canonical_id
from utils.text_cleaning import sanitize_unicode


DEFAULT_FIELDS = (
    "title,abstract,year,venue,authors,externalIds,url,openAccessPdf,publicationVenue"
)


@dataclass
class QuerySpec:
    query_id: str
    query: str
    source: str


_NON_TECH_TITLE_RE = re.compile(
    r"\b(editorial|foreword|preface|erratum|corrigendum|retraction|book review|news|"
    r"call for papers|conference report|invited commentary)\b",
    re.IGNORECASE,
)

_WORD_RE = re.compile(r"[a-zA-Z0-9]{2,}")

_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "in", "into", "is", "it", "its", "of", "on", "or", "our", "that",
    "the", "their", "these", "this", "to", "we", "with", "without", "via",
    "using", "use", "based", "within", "across", "paper", "method", "methods",
    "approach", "approaches", "model", "models", "results", "result",
}


def build_task2_queries(
    paper: Dict[str, Any],
    *,
    mode: str = "per_contribution",
    max_key_terms: int = 5,
    max_entities: int = 5,
) -> List[QuerySpec]:
    """
    Build deterministic Task 2 queries from Task 1 output.

    Modes:
    - per_contribution: core_task + each contribution name (N queries)
    - fixed: two queries (core_task + key_terms, contribution[0] name + must_have_entities)
    """
    paper_section = paper or {}
    core_task = (paper_section.get("core_task") or "").strip()
    contributions_raw = paper_section.get("contributions") or []
    contribution_names = _extract_contribution_names(contributions_raw)
    key_terms = _ensure_str_list(paper_section.get("key_terms"))
    must_have_entities = _ensure_str_list(paper_section.get("must_have_entities"))

    core_task_compact = _compact_query_text(core_task, max_words=12)

    queries: List[QuerySpec] = []

    if mode == "fixed":
        task_terms = [t for t in key_terms if t.strip()]
        task_terms = task_terms[: max(0, int(max_key_terms))]
        if core_task_compact or task_terms:
            q1 = " ".join([core_task_compact] + task_terms).strip()
            if q1:
                queries.append(
                    QuerySpec(query_id="Q1", query=_limit_query_words(q1, 25), source="core_task")
                )

        contrib_names = contribution_names[:1]
        entity_terms = [e for e in must_have_entities if e.strip()]
        entity_terms = entity_terms[: max(0, int(max_entities))]
        if contrib_names or entity_terms:
            contrib_compact = _compact_query_text(contrib_names[0] if contrib_names else "", max_words=12)
            q2 = " ".join([contrib_compact] + entity_terms).strip()
            if q2:
                queries.append(
                    QuerySpec(query_id="Q2", query=_limit_query_words(q2, 25), source="contribution")
                )

        return queries

    # Default: per contribution
    for idx, contrib_name in enumerate(contribution_names, start=1):
        contrib_name = (contrib_name or "").strip()
        if not contrib_name:
            continue
        contrib_compact = _compact_query_text(contrib_name, max_words=12)
        parts = [p for p in (core_task_compact, contrib_compact) if p]
        query = _limit_query_words(" ".join(parts).strip(), 25)
        if query:
            queries.append(
                QuerySpec(query_id=f"C{idx}", query=query, source="contribution")
            )

    if not queries and core_task_compact:
        fallback_terms = [t for t in key_terms if t.strip()]
        fallback_terms = fallback_terms[: max(0, int(max_key_terms))]
        query = _limit_query_words(" ".join([core_task_compact] + fallback_terms).strip(), 25)
        if query:
            queries.append(QuerySpec(query_id="Q1", query=query, source="core_task"))

    return queries


def retrieve_related_works(
    task1_output: Dict[str, Any],
    *,
    paper_year: Optional[int] = None,
    mode: str = "per_contribution",
    limit_per_query: int = 10,
    top_k: int = 30,
    max_total: int = 30,
    dedup_threshold: float = 0.96,
    mmr_lambda: float = 0.7,
    use_cache: bool = False,
    cache_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Task 2: retrieve related works using Semantic Scholar (no LLM).

    Returns a dict with `candidate_pool_top30` and metadata.
    """
    log = logger or logging.getLogger(__name__)
    paper_section = (task1_output or {}).get("paper", {})

    queries = build_task2_queries(paper_section, mode=mode)
    if not queries:
        log.warning("Task2: no queries could be built from Task 1 output.")
        return {
            "mode": mode,
            "paper_year": paper_year,
            "queries": [],
            "candidate_pool_top30": [],
            "stats": {"total_candidates": 0, "final": 0},
        }

    cache_path = None
    if use_cache:
        cache_dir = cache_dir or Path("output") / "task2_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = _make_cache_key(
            queries,
            paper_year,
            mode,
            limit_per_query,
            top_k,
            max_total,
            dedup_threshold,
            mmr_lambda,
        )
        cache_path = cache_dir / f"task2_{cache_key}.json"
        if cache_path.exists():
            try:
                with cache_path.open("r", encoding="utf-8") as f:
                    cached = json.load(f)
                log.info("Task2: loaded cached results from %s", cache_path)
                return cached
            except Exception as exc:
                log.warning("Task2: failed to load cache (%s); recomputing.", exc)

    client = SemanticScholarClientPool()

    query_meta: List[Dict[str, Any]] = []
    raw_candidates: List[Dict[str, Any]] = []

    for spec in queries:
        try:
            response = client.search(
                query=spec.query,
                limit=int(limit_per_query),
                offset=0,
                fields=DEFAULT_FIELDS,
            )
        except Exception as exc:
            log.warning("Task2: query failed (%s): %s", spec.query, exc)
            query_meta.append(
                {
                    "id": spec.query_id,
                    "query": spec.query,
                    "status": "error",
                    "error": str(exc),
                    "count": 0,
                }
            )
            continue

        items = response.get("data") or []
        normalized = [
            _normalize_candidate(item, rank=idx, total=len(items), source=spec.query_id)
            for idx, item in enumerate(items)
        ]
        raw_candidates.extend(normalized)
        query_meta.append(
            {
                "id": spec.query_id,
                "query": spec.query,
                "status": "ok",
                "count": len(normalized),
            }
        )

    total_candidates = len(raw_candidates)
    deduped = _dedup_approx(raw_candidates, threshold=dedup_threshold)
    after_dedup = len(deduped)
    filtered = [c for c in deduped if not _looks_non_technical(c.get("title"), c.get("abstract"))]
    after_nontech = len(filtered)
    if paper_year is not None:
        filtered = [c for c in filtered if _allow_by_year(c.get("year"), paper_year)]
    after_year = len(filtered)

    k = min(int(top_k), int(max_total)) if max_total else int(top_k)
    k = max(1, k) if filtered else 0
    diversified = _mmr_select(filtered, k=k, lambda_param=mmr_lambda)

    output_candidates = [_format_candidate(c) for c in diversified]

    result = {
        "mode": mode,
        "paper_year": paper_year,
        "queries": query_meta,
        "candidate_pool_top30": output_candidates,
        "stats": {
            "total_candidates": total_candidates,
            "after_dedup": after_dedup,
            "after_nontechnical_filter": after_nontech,
            "after_year_filter": after_year,
            "final": len(output_candidates),
        },
    }

    if cache_path is not None:
        try:
            with cache_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            log.info("Task2: cached results at %s", cache_path)
        except Exception as exc:
            log.warning("Task2: failed to write cache (%s)", exc)

    return result


def _make_cache_key(
    queries: Sequence[QuerySpec],
    paper_year: Optional[int],
    mode: str,
    limit_per_query: int,
    top_k: int,
    max_total: int,
    dedup_threshold: float,
    mmr_lambda: float,
) -> str:
    payload = {
        "mode": mode,
        "paper_year": paper_year,
        "limit_per_query": int(limit_per_query),
        "top_k": int(top_k),
        "max_total": int(max_total),
        "dedup_threshold": float(dedup_threshold),
        "mmr_lambda": float(mmr_lambda),
        "queries": [q.query for q in queries],
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _normalize_candidate(
    item: Dict[str, Any],
    *,
    rank: int,
    total: int,
    source: str,
) -> Dict[str, Any]:
    external_ids = item.get("externalIds") or {}
    doi = external_ids.get("DOI") or external_ids.get("doi")
    arxiv_id = (
        external_ids.get("ArXiv")
        or external_ids.get("arXiv")
        or external_ids.get("arxiv")
    )

    year_val = item.get("year")
    year = None
    if year_val is not None:
        try:
            year = int(year_val)
        except (TypeError, ValueError):
            year = None

    title = sanitize_unicode(item.get("title") or "")
    abstract = sanitize_unicode(item.get("abstract") or "")
    venue = item.get("venue") or ""
    publication_venue = item.get("publicationVenue")
    if not venue and isinstance(publication_venue, dict):
        venue = publication_venue.get("name") or ""

    url = item.get("url")
    if not url:
        open_access = item.get("openAccessPdf") or {}
        if isinstance(open_access, dict):
            url = open_access.get("url")

    score = item.get("score")
    if score is None:
        score = float(max(total - rank, 0)) / float(max(total, 1))

    cand_id = item.get("paperId") or ""
    if not cand_id:
        if doi:
            cand_id = f"doi:{doi}"
        elif arxiv_id:
            cand_id = f"arxiv:{arxiv_id}"
        else:
            cand_id = make_canonical_id(title=title)

    return {
        "cand_id": cand_id,
        "title": title,
        "year": year,
        "venue": venue,
        "abstract": abstract,
        "url": url,
        "embedding": None,
        "relevance_score": float(score) if score is not None else 0.0,
        "source_query": source,
    }


def _format_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "cand_id": candidate.get("cand_id"),
        "title": candidate.get("title"),
        "year": candidate.get("year"),
        "venue": candidate.get("venue"),
        "abstract": candidate.get("abstract"),
        "url": candidate.get("url"),
        "embedding": candidate.get("embedding"),
    }


def _score_value(item: Dict[str, Any]) -> float:
    for key in ("relevance_score", "score"):
        val = item.get(key)
        if val is None:
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return 0.0


def _normalize_meta_text(title: str, abstract: str) -> str:
    base = f"{title} {abstract}".lower()
    base = re.sub(r"[^a-z0-9 ]+", " ", base)
    base = re.sub(r"\s+", " ", base).strip()
    return base


def _dedup_approx(
    candidates: List[Dict[str, Any]],
    *,
    threshold: float = 0.96,
) -> List[Dict[str, Any]]:
    if len(candidates) <= 1:
        return candidates

    kept: List[Dict[str, Any]] = []
    kept_norms: List[str] = []

    for cand in candidates:
        norm = _normalize_meta_text(cand.get("title", ""), cand.get("abstract", ""))
        dup_idx = None
        for i, existing in enumerate(kept_norms):
            if not norm or not existing:
                continue
            if norm == existing:
                dup_idx = i
                break
            if SequenceMatcher(None, norm, existing).ratio() >= threshold:
                dup_idx = i
                break
        if dup_idx is None:
            kept.append(cand)
            kept_norms.append(norm)
        else:
            if _score_value(cand) > _score_value(kept[dup_idx]):
                kept[dup_idx] = cand
                kept_norms[dup_idx] = norm

    return kept


def _looks_non_technical(title: Optional[str], abstract: Optional[str]) -> bool:
    title = (title or "").strip()
    abstract = (abstract or "").strip()
    if not title:
        return True
    if _NON_TECH_TITLE_RE.search(title):
        return True

    title_tokens = _tokenize(title)
    abstract_tokens = _tokenize(abstract)

    if len(title_tokens) < 3 and len(abstract_tokens) < 10:
        return True
    return False


def _allow_by_year(candidate_year: Optional[int], paper_year: int) -> bool:
    if candidate_year is None:
        return True
    try:
        return int(candidate_year) <= int(paper_year)
    except Exception:
        return True


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def _compact_query_text(text: str, *, max_words: int = 12) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"\([^)]*\)", " ", text)
    cleaned = re.sub(r"[^A-Za-z0-9\- ]+", " ", cleaned)
    tokens = [t for t in cleaned.split() if t]
    if not tokens:
        return ""
    filtered = [t for t in tokens if t.lower() not in _QUERY_STOPWORDS]
    if not filtered:
        filtered = tokens
    return " ".join(filtered[:max_words]).strip()


def _limit_query_words(text: str, max_words: int) -> str:
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip()


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a or not set_b:
        return 0.0
    return float(len(set_a & set_b)) / float(len(set_a | set_b))


def _mmr_select(
    candidates: List[Dict[str, Any]],
    *,
    k: int,
    lambda_param: float = 0.7,
) -> List[Dict[str, Any]]:
    if not candidates or k <= 0:
        return []

    scored = sorted(
        candidates,
        key=lambda c: (_score_value(c), (c.get("title") or "")),
        reverse=True,
    )

    tokens = {id(c): _tokenize((c.get("title") or "") + " " + (c.get("abstract") or "")) for c in scored}

    selected: List[Dict[str, Any]] = []
    selected_tokens: List[List[str]] = []

    while scored and len(selected) < k:
        best = None
        best_score = None
        for cand in scored:
            relevance = _score_value(cand)
            if not selected_tokens:
                mmr_score = relevance
            else:
                max_sim = 0.0
                cand_tokens = tokens.get(id(cand), [])
                for chosen_tokens in selected_tokens:
                    sim = _jaccard(cand_tokens, chosen_tokens)
                    if sim > max_sim:
                        max_sim = sim
                mmr_score = lambda_param * relevance - (1.0 - lambda_param) * max_sim

            if best is None or mmr_score > best_score:
                best = cand
                best_score = mmr_score

        if best is None:
            break

        selected.append(best)
        selected_tokens.append(tokens.get(id(best), []))
        scored = [c for c in scored if c is not best]

    return selected


def _extract_contribution_names(contributions_raw: Any) -> List[str]:
    """
    Extract contribution names from structured or legacy format.
    
    Handles:
    - New format: list of dicts with 'name' field
    - Legacy format: list of strings
    """
    if not isinstance(contributions_raw, list):
        return []
    
    names: List[str] = []
    for item in contributions_raw:
        if isinstance(item, dict):
            # New structured format: extract 'name' field
            name = item.get("name", "")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
        elif isinstance(item, str) and item.strip():
            # Legacy format: use string directly
            names.append(item.strip())
    
    return names


def _ensure_str_list(value: Any) -> List[str]:
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        return out
    return []
