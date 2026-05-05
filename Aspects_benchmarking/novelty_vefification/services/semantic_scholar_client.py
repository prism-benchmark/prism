"""
Semantic Scholar API client for Phase 2 search.

Uses the Semantic Scholar Graph API /paper/search endpoint.
An API key is optional but recommended to increase rate limits.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
from typing import Any, Dict, List, Optional

import requests

from config import (
    API_TIMEOUT,
    PHASE2_MAX_QUERY_ATTEMPTS,
    RETRY_DELAY,
    SEMANTIC_SCHOLAR_API_BASE,
    SEMANTIC_SCHOLAR_API_KEY,
    SEMANTIC_SCHOLAR_API_KEYS,
)

logger = logging.getLogger(__name__)


class SemanticScholarClient:
    """Lightweight client for Semantic Scholar Graph API search."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: str = SEMANTIC_SCHOLAR_API_BASE,
        timeout: int = API_TIMEOUT,
        max_attempts: int = PHASE2_MAX_QUERY_ATTEMPTS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = int(timeout)
        self.max_attempts = max(1, int(max_attempts))
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "paper-novelty-pipeline/phase2",
            }
        )
        key = api_key or SEMANTIC_SCHOLAR_API_KEY
        if key:
            self.session.headers["x-api-key"] = key
        
        # Rate limiting: 1 request per 1.5 seconds
        self._last_request_time = 0.0
        self._min_request_interval = 1.5

    def search(
        self,
        *,
        query: str,
        limit: int = 10,
        offset: int = 0,
        fields: str,
    ) -> Dict[str, Any]:
        """Search papers by query string."""
        url = f"{self.base_url}/paper/search"
        params = {
            "query": query,
            "limit": int(limit),
            "offset": int(offset),
            "fields": fields,
        }

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_attempts):
            try:
                # Rate limiting: ensure 1.5 seconds between requests
                elapsed = time.time() - self._last_request_time
                if elapsed < self._min_request_interval:
                    time.sleep(self._min_request_interval - elapsed)
                
                self._last_request_time = time.time()
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code in (401, 403):
                    raise RuntimeError(
                        f"HTTP {resp.status_code} – API key rejected (not retryable)"
                    )
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise requests.RequestException(f"HTTP {resp.status_code}")
                resp.raise_for_status()
                return resp.json()
            except RuntimeError:
                raise  # 401/403 – propagate immediately, no retry
            except Exception as exc:  # requests.RequestException or JSON error
                last_exc = exc
                if attempt + 1 >= self.max_attempts:
                    break
                sleep_s = min(RETRY_DELAY * (2 ** attempt), 60)
                time.sleep(sleep_s)

        raise RuntimeError(
            f"Semantic Scholar search failed after {self.max_attempts} attempts: {last_exc}"
        )

    def get_paper_references(
        self,
        *,
        paper_id: str,
        limit: int = 5,
        fields: str = "title,abstract,year,venue,externalIds,url,openAccessPdf,publicationVenue",
    ) -> Dict[str, Any]:
        """Retrieve the reference list (papers cited *by* this paper).

        Uses ``/paper/{paper_id}/references`` from the Semantic Scholar
        Graph API.  Returns ``{"data": [{"citedPaper": {...}}, ...]}``
        on success.
        """
        url = f"{self.base_url}/paper/{paper_id}/references"
        params = {"limit": int(limit), "fields": fields}

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_attempts):
            try:
                elapsed = time.time() - self._last_request_time
                if elapsed < self._min_request_interval:
                    time.sleep(self._min_request_interval - elapsed)

                self._last_request_time = time.time()
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code in (401, 403):
                    raise RuntimeError(
                        f"HTTP {resp.status_code} – API key rejected (not retryable)"
                    )
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise requests.RequestException(f"HTTP {resp.status_code}")
                resp.raise_for_status()
                return resp.json()
            except RuntimeError:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt + 1 >= self.max_attempts:
                    break
                sleep_s = min(RETRY_DELAY * (2 ** attempt), 60)
                time.sleep(sleep_s)

        raise RuntimeError(
            f"Semantic Scholar references failed after {self.max_attempts} attempts: {last_exc}"
        )


class SemanticScholarClientPool:
    """Round-robin load balancer across N Semantic Scholar API keys.

    Each key gets its own ``SemanticScholarClient`` (and therefore its own
    per-key rate limiter).  Calls are distributed round-robin so that with
    *N* keys the effective throughput is ~N req/s instead of 1 req/s.

    When only one key (or none) is configured the pool degrades gracefully
    to a single-client setup – no behaviour change from before.
    """

    def __init__(
        self,
        *,
        api_keys: Optional[List[str]] = None,
        base_url: str = SEMANTIC_SCHOLAR_API_BASE,
        timeout: int = API_TIMEOUT,
        max_attempts: int = PHASE2_MAX_QUERY_ATTEMPTS,
    ) -> None:
        keys = api_keys if api_keys else SEMANTIC_SCHOLAR_API_KEYS
        if not keys:
            keys = [None]  # type: ignore[list-item]

        self._clients: List[SemanticScholarClient] = [
            SemanticScholarClient(
                api_key=k,
                base_url=base_url,
                timeout=timeout,
                max_attempts=max_attempts,
            )
            for k in keys
        ]
        self._cycle = itertools.cycle(range(len(self._clients)))
        self._lock = threading.Lock()
        logger.info(
            "SemanticScholarClientPool initialised with %d API key(s)", len(self._clients)
        )

    def _next_client(self) -> SemanticScholarClient:
        with self._lock:
            return self._clients[next(self._cycle)]

    def _call_with_failover(self, method: str, **kwargs: Any) -> Dict[str, Any]:
        """Try round-robin client; on failure fall back to remaining clients."""
        tried: set[int] = set()
        last_exc: Optional[Exception] = None
        while len(tried) < len(self._clients):
            with self._lock:
                idx = next(self._cycle)
            if idx in tried:
                continue
            tried.add(idx)
            try:
                return getattr(self._clients[idx], method)(**kwargs)
            except Exception as exc:
                logger.warning(
                    "Client %d failed for %s: %s – trying next key", idx, method, exc
                )
                last_exc = exc
        raise RuntimeError(
            f"All {len(self._clients)} Semantic Scholar clients failed: {last_exc}"
        )

    def search(self, **kwargs: Any) -> Dict[str, Any]:
        return self._call_with_failover("search", **kwargs)

    def get_paper_references(self, **kwargs: Any) -> Dict[str, Any]:
        return self._call_with_failover("get_paper_references", **kwargs)
