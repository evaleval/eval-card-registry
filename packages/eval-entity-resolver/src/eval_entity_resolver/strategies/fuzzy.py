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

PRECISION-LOSS POLICY for stem stripping
=========================================
Some suffixes are stripped at the cost of conflating variants the registry
considers "the same model" but a precision-sensitive consumer might not.
The current strip list collapses:

- ``-hf``: HuggingFace re-uploaded copy of an official model release. The
  weights are the same; the upload path differs.
- ``-fp8`` / ``-fp16`` / ``-fp4`` / ``-bf16`` / ``-int4`` / ``-int8`` /
  ``-q4`` / ``-q8`` / ``-quant`` / ``-gguf`` / ``-awq`` / ``-gptq``:
  quantization variants. The architecture is the same; numerical precision
  differs. **Benchmark scores can differ measurably between quantization
  levels** — e.g., a 70B model at FP16 may outperform the same model at
  INT4. We collapse them anyway so the registry treats them as one canonical
  identity, but this means a row labeled "Llama-3.1-70B" in the catalog may
  represent any quantization that was reported.

Consumers needing per-quantization precision should NOT use the canonical_id
alone — they need the original `model_route_id` (which preserves the suffix
verbatim) and the `generation_args` payload.

This collapse is intentional: it's far more useful to match a quantized
inference run to the model family it represents than to leave it unresolved.
But it's a deliberate precision sacrifice and downstream code that compares
scores must respect that.
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
    # HuggingFace re-upload variants — same weights as official release.
    "-hf",
    # Quantization variants — see PRECISION-LOSS POLICY in module docstring.
    # Order: longer first so e.g. ``-int8`` doesn't pre-empt ``-int8-awq``.
    "-int4-awq",
    "-int8-awq",
    "-int4-gptq",
    "-int8-gptq",
    "-fp4",
    "-fp8",
    "-fp16",
    "-bf16",
    "-int4",
    "-int8",
    "-q4",
    "-q8",
    "-awq",
    "-gptq",
    "-gguf",
    "-quant",
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
#
# Zhipu/Z.ai cluster: the GLM-family canonical org is `zai` (short form used
# in this registry for canonical_ids like `zai/glm-4.5`). HF and various
# leaderboards spell it as `zhipu`, `zhipu-ai`, `z-ai`, or `zai-org` — all
# refer to the same Beijing AI startup behind GLM.
#
# Moonshot AI cluster: canonical org is `moonshotai` (matches HF
# `moonshotai/Kimi-*` namespace); aliases cover `moonshot` and `moonshot-ai`
# spellings seen in the corpus.
#
# `alibaba` → `qwen` was considered but skipped: the corpus has 1
# non-Qwen entry (`alibaba__mineru2-pipeline`) which would be wrongly
# rewritten. Qwen models under `alibaba/` are handled via explicit
# overrides instead.
_ORG_ALIASES: dict[str, str] = {
    "deepseek-ai": "deepseek",
    "cohereforai": "cohere",
    "cohere-labs": "cohere",
    "tii-uae": "tiiuae",
    "meta-llama": "meta",
    "mistral-ai": "mistralai",
    "nvidia-nemo": "nvidia",
    # Zhipu/Z.ai → zai
    "zhipu": "zai",
    "zhipu-ai": "zai",
    "z-ai": "zai",
    "zai-org": "zai",
    # Moonshot → moonshotai
    "moonshot": "moonshotai",
    "moonshot-ai": "moonshotai",
}

# Host / gateway / placeholder prefixes that should be DROPPED entirely
# (not rewritten to a canonical org). These are not model authors —
# they're hosting platforms, gateways, or placeholders for missing
# developer fields. When raw_value uses one of these as the org prefix,
# the resolver tries the bare suffix in addition to the full string.
#
# Identified from corpus surveys: alphaxiv leaderboard uses `unknown/`
# when developer field is absent; Bedrock/Vertex/Azure/Fireworks/etc.
# are inference platforms re-hosting other companies' models.
_HOST_PREFIXES_TO_STRIP: set[str] = {
    "unknown",
    "bedrock", "amazon-bedrock", "aws-bedrock",
    "azure", "azure-openai", "azure-cognitive-services",
    "vertex", "google-vertex", "vertex-anthropic",
    "fireworks", "fireworks-ai",
    "groq",
    "together", "togetherai", "together-ai",
    "openrouter",
    "perplexity-agent",
    "deepinfra", "anyscale", "novita", "novita-ai", "replicate",
    "ollama", "ollama-cloud",
    "github-models", "github-copilot",
    "lambda", "baseten", "modal", "runpod", "cerebras",
    "sap-ai-core", "cloudflare-ai-gateway", "aihubmix",
    "kilo", "vercel", "llmgateway", "poe",
}


def _drop_duplicated_org_prefix(value: str) -> str | None:
    """Detect and collapse a repeated-org-prefix typo.

    Recognized shapes (token equality is case-insensitive, but the
    returned string preserves the original casing of `value` so the
    downstream lookups can still match exact aliases):

      - ``<org>/<org>-<rest>``           → ``<org>/<rest>``
      - ``<org>/<org>_<rest>``           → ``<org>/<rest>``
      - ``<org>/<org>/<rest>``           → ``<org>/<rest>`` (literal double slash)
      - ``<org>__<org>-<rest>``          → ``<org>__<rest>`` (slug form;
        the pipeline rewrites ``/`` → ``__`` for route_ids and the resolver
        may receive either)
      - ``<org>__<org>__<rest>``         → ``<org>__<rest>`` (slug form
        of the literal double-slash variant)

    Returns ``None`` when the prefix is not duplicated, or when the
    repeated-prefix slug shape is followed by something that doesn't
    cleanly separate (e.g. ``gpt-4/gpt-4-turbo`` — the second ``gpt-4``
    is the START of the model name, not a duplicated prefix).

    The match requires exact token equality of the two leading tokens.
    A substring overlap (``gpt-4`` ⊂ ``gpt-4-turbo``) is intentionally
    NOT enough — that's a real two-segment HF path, not a typo.

    To disambiguate the org-typo case (``openai/openai-o1``) from the
    model-family-prefix case (``gpt-4/gpt-4-turbo``): the heuristic
    only fires when the leading org token has no internal hyphen.
    Real org names (``openai``, ``moonshotai``, ``anthropic``) are
    single tokens; model-family prefixes (``gpt-4``, ``llama-3``,
    ``claude-opus-4-5``) contain hyphens. This is imperfect — a
    hyphenated org like ``mistral-ai`` would slip through — but
    those are already captured upstream by the org-alias pass.
    """
    if not value:
        return None

    # Slash forms first (canonical HF path style).
    if "/" in value:
        first_slash = value.index("/")
        org = value[:first_slash]
        rest = value[first_slash + 1:]
        if not org or not rest:
            return None
        # Skip when the leading token contains a hyphen — likely a
        # model-family prefix (e.g. `gpt-4/gpt-4-turbo`), not a
        # duplicated-org typo. Hyphenated orgs like `mistral-ai` are
        # canonicalized via the org-alias pass first.
        if "-" in org:
            return None
        org_lower = org.lower()
        # `<org>/<org>/<rest>` literal double slash
        if "/" in rest:
            second, after = rest.split("/", 1)
            if second.lower() == org_lower and after:
                return f"{org}/{after}"
        # `<org>/<org>-<rest>` and `<org>/<org>_<rest>`
        for sep in ("-", "_"):
            prefix = org_lower + sep
            if rest.lower().startswith(prefix) and len(rest) > len(prefix):
                return f"{org}/{rest[len(prefix):]}"

    # Slug forms (route_id style with `__`).
    if "__" in value:
        first = value.index("__")
        org = value[:first]
        rest = value[first + 2:]
        if not org or not rest:
            return None
        # Same hyphen-in-org guard (see slash branch above).
        if "-" in org:
            return None
        org_lower = org.lower()
        # `<org>__<org>__<rest>`
        if "__" in rest:
            second, after = rest.split("__", 1)
            if second.lower() == org_lower and after:
                return f"{org}__{after}"
        # `<org>__<org>-<rest>` (and `_<rest>` — note we already consumed `__`,
        # so the next separator is a single `-` or `_`).
        for sep in ("-", "_"):
            prefix = org_lower + sep
            if rest.lower().startswith(prefix) and len(rest) > len(prefix):
                return f"{org}__{rest[len(prefix):]}"

    return None


def _drop_host_prefix(value: str) -> str | None:
    """If value's developer prefix is a known hosting platform, return the
    bare suffix portion (everything after the first separator). Otherwise None.

    Handles both `host/model` and `host.model` separators."""
    if "/" in value:
        org, rest = value.split("/", 1)
        if org.lower() in _HOST_PREFIXES_TO_STRIP and rest:
            return rest
    if "." in value:
        # Bedrock-style: "anthropic.claude-3-5-sonnet" → "anthropic.claude-3-5-sonnet"
        # is itself a host format, but the prefix BEFORE the dot is the host.
        # Only strip if everything-before-first-dot is a host name.
        first_dot = value.index(".")
        org = value[:first_dot]
        rest = value[first_dot + 1:]
        if org.lower() in _HOST_PREFIXES_TO_STRIP and rest:
            return rest
    return None

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


# Letter-then-dash-then-digit pattern: ``qwen-2`` → ``qwen2``. Models commonly
# appear in two spellings: ``qwen-2-72b`` (pipeline route_id form) vs
# ``qwen2-72b`` (registry canonical form). Collapsing the boundary dash lets
# them resolve to the same canonical without having to enumerate every variant
# as an alias. Safe because real distinguishing tokens are digits or words on
# the OTHER side of separators (e.g. ``gpt-4`` vs ``gpt-4-mini`` stays
# distinct because the ``-mini`` separator survives).
_LETTER_DIGIT_DASH = re.compile(r"([a-zA-Z])-(\d)")


def _collapse_letter_digit_dashes(value: str) -> str | None:
    """Return value with letter-digit boundary dashes removed, or None if no change."""
    collapsed = _LETTER_DIGIT_DASH.sub(r"\1\2", value)
    if collapsed == value:
        return None
    return collapsed


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
    # The heuristics below are intentionally model-specific: they strip
    # hosting prefixes, org aliases, dated model snapshots, and inference-mode
    # suffixes. Applying them to benchmarks/metrics/harnesses can merge
    # unrelated entities that merely share a host-like prefix or model-ish tail.
    if entity_type != "model":
        return None, 0.0

    candidates_to_try: list[str] = []

    # 1. Suffix stripping (may produce multiple stems: strip one, strip two, etc.)
    stripped = _strip_suffix(raw_value)
    if stripped:
        candidates_to_try.append(stripped)
        # Try double-strip (e.g. "model-fc-together" — unlikely but cheap)
        double = _strip_suffix(stripped)
        if double:
            candidates_to_try.append(double)

    # 2. Host-prefix dropping — if raw_value's developer prefix is a known
    # hosting platform / gateway / placeholder, also try the bare suffix.
    # Apply on the original AND any suffix-stripped forms.
    for val in [raw_value] + candidates_to_try[:]:
        bare = _drop_host_prefix(val)
        if bare:
            candidates_to_try.append(bare)
            # The bare form might itself need suffix stripping
            stripped_bare = _strip_suffix(bare)
            if stripped_bare:
                candidates_to_try.append(stripped_bare)

    # 3. Duplicated-org-prefix collapse — catches typos like
    # `moonshotai/moonshotai-kimi-k2-instruct` (and the slug-form
    # `moonshotai__moonshotai-kimi-k2-instruct`). Runs AFTER suffix /
    # host strip so the deduped form goes through the rest of the
    # pipeline (org alias + lookup), and BEFORE org alias so the
    # collapsed string can pick up `_ORG_ALIASES` rewriting on the
    # next step.
    for val in [raw_value] + candidates_to_try[:]:
        deduped = _drop_duplicated_org_prefix(val)
        if deduped:
            candidates_to_try.append(deduped)

    # 4. Org normalization — on original, suffix-stripped, host-stripped,
    # and duplicate-org-collapsed forms.
    for val in [raw_value] + candidates_to_try[:]:
        rewritten = _normalize_org(val)
        if rewritten:
            candidates_to_try.append(rewritten)

    # 5. Letter-digit dash collapse — try every candidate with the
    # ``letter-digit`` boundary dash removed (e.g. ``qwen-2-72b`` →
    # ``qwen2-72b``). Run last so it composes with all earlier rewrites.
    for val in [raw_value] + candidates_to_try[:]:
        collapsed = _collapse_letter_digit_dashes(val)
        if collapsed:
            candidates_to_try.append(collapsed)

    # 6. Check each candidate against exact then normalized lookups.
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
