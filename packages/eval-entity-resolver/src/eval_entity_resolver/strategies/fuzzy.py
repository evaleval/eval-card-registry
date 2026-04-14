"""
Fuzzy matching strategy.

Instead of generic string similarity (which falsely merges distinct versions,
sizes, and variants), this uses two targeted approaches:

1. **Stem matching** — strip known non-semantic suffixes (evaluation mode,
   hosting provider, effort level) and check if the stem has an exact or
   normalized match.  This catches real duplicates like
   ``model-name-fc`` → ``model-name`` without collapsing ``gpt-5-mini`` into
   ``gpt-5``.

2. **Org normalization** — handle cases where the org prefix differs between
   configs (``deepseek-ai/model`` vs ``deepseek/model``).

If neither approach produces a match the strategy returns None so the resolver
can fall through to auto-draft.
"""
from __future__ import annotations

import re
from typing import Optional

from eval_entity_resolver.normalization import normalize


# Suffixes stripped only from the *end* of the raw value.  Order matters:
# longer suffixes first to avoid partial stripping.
#
# Many of these patterns were identified from raw_model_ids in the
# evaleval/card_backend dataset, where a single canonical model family
# (e.g. ``anthropic/claude-opus-4-5``) has variants like
# ``claude-opus-4-5-20251101-thinking-16k``, ``-fc``, ``-prompt``, etc.
_STRIP_SUFFIXES = [
    # Evaluation-mode suffixes (BFCL, etc.)
    "-fc",
    "-prompt",
    # Hosting-provider suffixes
    "-together",
    "-bedrock",
    # Reasoning-effort suffixes
    "-high",
    "-medium",
    "-low",
    "-minimal",
    # Thinking-style suffixes
    "-nothink",
    "-thinking-none",
]

# Regex-based suffix patterns applied after the literal suffixes.
# These capture variants with numeric parameters (thinking budgets, dates).
# Each pattern must anchor with $ and match only the tail of the string.
_STRIP_SUFFIX_PATTERNS: list[re.Pattern[str]] = [
    # Thinking-budget suffix: "-thinking-8k", "-thinking-16k", "-thinking-64k"
    re.compile(r"-thinking-\d+k$", re.IGNORECASE),
    # Date version suffix (YYYYMMDD): "-20251101", "-20240315"
    # Only strip dates (8 consecutive digits) to avoid touching version numbers.
    re.compile(r"-\d{8}$"),
]

# Known org aliases: {variant_prefix: canonical_prefix}
# Convention: simplify HF org names (e.g. "deepseek-ai" → "deepseek") to the
# shorter form used as canonical in this registry.
_ORG_ALIASES: dict[str, str] = {
    "deepseek-ai": "deepseek",
    "cohereforai": "cohere",
    "cohere-labs": "cohere",
    "tii-uae": "tiiuae",
    "meta-llama": "meta",
    "mistral-ai": "mistralai",
    "nvidia-nemo": "nvidia",
}

# Confidence assigned to stem-match results.  Below 1.0 (exact) and 0.95
# (normalized) so the provenance is clear in the resolution log.
_STEM_CONFIDENCE = 0.90


def _strip_suffix(value: str) -> str | None:
    """Strip a single known suffix.  Returns the stem or None if no suffix matched."""
    lower = value.lower()
    for suffix in _STRIP_SUFFIXES:
        if lower.endswith(suffix):
            return value[: len(value) - len(suffix)]
    for pattern in _STRIP_SUFFIX_PATTERNS:
        m = pattern.search(value)
        if m:
            return value[: m.start()]
    return None


def _normalize_org(value: str) -> str | None:
    """Replace a known org-alias prefix.  Returns the rewritten string or None."""
    if "/" not in value:
        return None
    org, rest = value.split("/", 1)
    canonical_org = _ORG_ALIASES.get(org.lower())
    if canonical_org is None:
        return None
    return f"{canonical_org}/{rest}"


def fuzzy_match(
    raw_value: str,
    entity_type: str,
    threshold: float,  # kept for API compat; not used by stem matching
    alias_store,
    source_config: Optional[str] = None,
) -> tuple[Optional[str], float]:
    """
    Attempt targeted fuzzy resolution.

    Returns ``(canonical_id, confidence)``; canonical_id is None on no match.
    """
    candidates_to_try: list[str] = []

    # 1. Suffix stripping (may produce multiple stems: strip one, strip two, etc.)
    stripped = _strip_suffix(raw_value)
    if stripped:
        candidates_to_try.append(stripped)
        # Try double-strip (e.g. "model-fc-together" — unlikely but cheap)
        double = _strip_suffix(stripped)
        if double:
            candidates_to_try.append(double)

    # 2. Org normalization — on both original and stripped forms
    for val in [raw_value] + candidates_to_try[:]:
        rewritten = _normalize_org(val)
        if rewritten:
            candidates_to_try.append(rewritten)

    # 3. Check each candidate against exact then normalized lookups.
    # Scoped-aware: config-scoped aliases for ``source_config`` count as
    # candidates; unrelated scoped aliases are excluded.
    norm_lookup = alias_store.get_normalized_lookup(entity_type, source_config)

    for candidate in candidates_to_try:
        exact_id = alias_store.lookup(candidate, entity_type, source_config)
        if exact_id is not None:
            return exact_id, _STEM_CONFIDENCE

        norm = normalize(candidate)
        canonical_id = norm_lookup.get(norm)
        if canonical_id is not None:
            return canonical_id, _STEM_CONFIDENCE

    return None, 0.0
