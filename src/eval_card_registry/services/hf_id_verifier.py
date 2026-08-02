"""Runtime HF repo-id verifier — the injected `hf_id_checker` for the
resolver chain (policy step 1: a string that IS a valid HF model id resolves
to itself).

Lookup order per raw value:
  1. In-memory result cache. Positive entries (id confirmed) live 24 h;
     negative entries (authoritative 404) live 1 h, so a repo created today
     becomes resolvable without a restart.
  2. The cron-built `hub_stats_index` dict (normalized id -> HF-true id),
     provided by the service. Answers the bulk of lookups with no network.
  3. Live Hub API fallback: one GET to `/api/models/{id}`. The Hub API is
     the source hub-stats itself is built from, answers a single-id question
     with one small request, and has no staleness. (The old shard-0-limited
     `HubStatsClient` is NOT used here — it stays draft-enrichment-only.)

The live path is defended against rate limits and thread starvation. All
acquisitions are NON-BLOCKING — when a guard says no, the verifier answers
"unknown" (None, uncached) immediately instead of pinning a request thread:
  - single-flight per id (a concurrent duplicate answers unknown rather
    than waiting on the in-flight call);
  - a global concurrency cap (default 4 simultaneous live calls);
  - a token-bucket budget (default 60 live calls per minute);
  - a circuit breaker (default: 5 consecutive failures open the live path
    for 5 minutes);
  - the `hf_live_id_check_enabled` config flag turns the live path off
    entirely.

Outcome semantics: only an authoritative 404 is negative-cached. Budget
exhaustion, breaker-open, transport errors, and the disabled flag all
return None WITHOUT caching, so the resolver chain falls through to the
alias/fuzzy steps exactly as it does today, and a later call may retry.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Optional

from eval_entity_resolver.models import HfIdHit, looks_like_hf_id

POSITIVE_TTL_SECONDS = 24 * 3600.0
NEGATIVE_TTL_SECONDS = 3600.0
MAX_CONCURRENT_LIVE = 4
LIVE_BUDGET_PER_MINUTE = 60
BREAKER_FAILURE_THRESHOLD = 5
BREAKER_COOLDOWN_SECONDS = 300.0

# Sentinel stored in the cache for an authoritative "not on the Hub".
_MISS = ""


class HfIdVerifier:
    """Checks whether a raw string is a valid HF model repo id.

    `index_provider` returns the current `normalized id -> HF-true id` dict
    (the service rebuilds it after entity churn; the verifier re-reads it on
    every check, so invalidation needs no coupling here). `clock` is
    injectable for tests and must be monotonic."""

    def __init__(
        self,
        index_provider: Callable[[], dict],
        *,
        positive_ttl: float = POSITIVE_TTL_SECONDS,
        negative_ttl: float = NEGATIVE_TTL_SECONDS,
        max_concurrent: int = MAX_CONCURRENT_LIVE,
        budget_per_minute: int = LIVE_BUDGET_PER_MINUTE,
        breaker_threshold: int = BREAKER_FAILURE_THRESHOLD,
        breaker_cooldown: float = BREAKER_COOLDOWN_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._index_provider = index_provider
        self._positive_ttl = positive_ttl
        self._negative_ttl = negative_ttl
        self._budget_per_minute = budget_per_minute
        self._breaker_threshold = breaker_threshold
        self._breaker_cooldown = breaker_cooldown
        self._clock = clock
        # cache: lowercased raw id -> (expires_at, hf_true_id | _MISS)
        self._cache: dict[str, tuple[float, str]] = {}
        self._cache_lock = threading.Lock()
        self._live_semaphore = threading.BoundedSemaphore(max_concurrent)
        self._in_flight: set[str] = set()
        self._in_flight_lock = threading.Lock()
        self._budget_window: deque[float] = deque()
        self._budget_lock = threading.Lock()
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0
        self._breaker_lock = threading.Lock()
        self._hf_api = None

    # ------------------------------------------------------------------
    # Public entry point (the resolver's hf_id_checker callable)
    # ------------------------------------------------------------------

    def check(self, raw_value: str) -> Optional[HfIdHit]:
        if not looks_like_hf_id(raw_value):
            return None
        key = raw_value.lower()

        cached = self._cache_get(key)
        if cached is not None:
            if cached == _MISS:
                return None
            return self._hit(raw_value, cached, "hf_live")

        index_hit = self._check_index(raw_value)
        if index_hit is not None:
            return index_hit

        return self._check_live(raw_value, key)

    # ------------------------------------------------------------------
    # Tiers
    # ------------------------------------------------------------------

    def _check_index(self, raw_value: str) -> Optional[HfIdHit]:
        try:
            index = self._index_provider() or {}
        except Exception:
            return None
        if not index:
            return None
        from eval_card_registry.services.hub_stats import normalize as _hsnorm

        hf_id = index.get(_hsnorm(raw_value))
        if not hf_id:
            return None
        return self._hit(raw_value, hf_id, "hub_stats_index")

    def _check_live(self, raw_value: str, key: str) -> Optional[HfIdHit]:
        from eval_card_registry.config import settings

        if not settings.hf_live_id_check_enabled:
            return None
        now = self._clock()
        with self._breaker_lock:
            if now < self._breaker_open_until:
                return None
        if not self._budget_take(now):
            return None
        # Single-flight: a concurrent duplicate answers unknown immediately.
        with self._in_flight_lock:
            if key in self._in_flight:
                return None
            self._in_flight.add(key)
        acquired = self._live_semaphore.acquire(blocking=False)
        try:
            if not acquired:
                return None
            return self._live_lookup(raw_value, key)
        finally:
            if acquired:
                self._live_semaphore.release()
            with self._in_flight_lock:
                self._in_flight.discard(key)

    def _live_lookup(self, raw_value: str, key: str) -> Optional[HfIdHit]:
        from eval_card_registry.config import settings

        try:
            from huggingface_hub import HfApi
            from huggingface_hub.errors import (
                GatedRepoError,
                RepositoryNotFoundError,
            )
        except Exception:
            return None
        try:
            if self._hf_api is None:
                self._hf_api = HfApi(token=settings.hf_token or None)
            info = self._hf_api.model_info(
                raw_value, timeout=settings.hf_live_id_check_timeout
            )
        except RepositoryNotFoundError:
            # Authoritative miss (includes private repos we can't see, which
            # is the right answer for resolution purposes).
            self._record_success()
            self._cache_put(key, _MISS, self._negative_ttl)
            return None
        except GatedRepoError:
            # Gated repos exist; metadata may be withheld, so fall back to
            # the raw spelling for casing.
            self._record_success()
            self._cache_put(key, raw_value, self._positive_ttl)
            return self._hit(raw_value, raw_value, "hf_live")
        except Exception:
            self._record_failure()
            return None
        hf_id = getattr(info, "id", None) or raw_value
        self._record_success()
        self._cache_put(key, hf_id, self._positive_ttl)
        return self._hit(raw_value, hf_id, "hf_live")

    # ------------------------------------------------------------------
    # Guards and bookkeeping
    # ------------------------------------------------------------------

    def _hit(self, raw_value: str, hf_id: str, source: str) -> HfIdHit:
        # HF repo ids are case-insensitively unique, so a case-insensitive
        # match identifies the repo exactly — that's "verbatim". Anything
        # else (separator collapse via the index, a followed rename redirect
        # from the live API) is the weaker "normalized" tier.
        return HfIdHit(
            hf_id=hf_id,
            verbatim=raw_value.lower() == hf_id.lower(),
            source=source,
        )

    def _cache_get(self, key: str) -> Optional[str]:
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if self._clock() >= expires_at:
                del self._cache[key]
                return None
            return value

    def _cache_put(self, key: str, value: str, ttl: float) -> None:
        with self._cache_lock:
            self._cache[key] = (self._clock() + ttl, value)

    def _budget_take(self, now: float) -> bool:
        with self._budget_lock:
            while self._budget_window and now - self._budget_window[0] >= 60.0:
                self._budget_window.popleft()
            if len(self._budget_window) >= self._budget_per_minute:
                return False
            self._budget_window.append(now)
            return True

    def _record_failure(self) -> None:
        with self._breaker_lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._breaker_threshold:
                self._breaker_open_until = self._clock() + self._breaker_cooldown
                self._consecutive_failures = 0

    def _record_success(self) -> None:
        with self._breaker_lock:
            self._consecutive_failures = 0
