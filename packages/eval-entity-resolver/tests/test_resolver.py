import pandas as pd
import pytest

from eval_entity_resolver import AliasStore, Resolver, ResolverConfig


def _store_with_aliases(*rows) -> AliasStore:
    """Build an AliasStore from (raw_value, entity_type, canonical_id, source_config, status) tuples."""
    from datetime import datetime, timezone
    import uuid

    now = datetime.now(timezone.utc).isoformat()
    records = []
    for raw_value, entity_type, canonical_id, source_config, status in rows:
        records.append(
            {
                "id": str(uuid.uuid4()),
                "raw_value": raw_value,
                "entity_type": entity_type,
                "canonical_id": canonical_id,
                "source_config": source_config,
                "source_field": None,
                "status": status,
                "strategy": "confirmed",
                "confidence": 1.0,
                "notes": None,
                "created_at": now,
                "updated_at": now,
            }
        )
    from eval_entity_resolver.alias_store import _empty_df
    df = pd.DataFrame(records) if records else _empty_df()
    return AliasStore(df)


class TestExactStrategy:
    def test_exact_match(self):
        store = _store_with_aliases(("IFEval", "benchmark", "ifeval", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("IFEval", "benchmark")
        assert result.canonical_id == "ifeval"
        assert result.strategy == "exact"
        assert result.confidence == 1.0

    def test_config_scoped_before_global(self):
        store = _store_with_aliases(
            ("MATH", "benchmark", "math-global", None, "confirmed"),
            ("MATH", "benchmark", "math-helm", "helm_lite", "confirmed"),
        )
        resolver = Resolver(store)
        result = resolver.resolve("MATH", "benchmark", source_config="helm_lite")
        assert result.canonical_id == "math-helm"

    def test_falls_back_to_global(self):
        store = _store_with_aliases(("MATH", "benchmark", "math-global", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("MATH", "benchmark", source_config="some_other_config")
        assert result.canonical_id == "math-global"

    def test_rejected_alias_skipped(self):
        store = _store_with_aliases(("IFEval", "benchmark", "ifeval", None, "rejected"))
        resolver = Resolver(store)
        result = resolver.resolve("IFEval", "benchmark")
        assert result.canonical_id is None
        assert result.strategy == "no_match"


class TestScopedAliasIsolation:
    """Scoped aliases (source_config != None) must not leak into unrelated lookups."""

    def test_scoped_isolated_from_other_config(self):
        store = _store_with_aliases(
            ("Overall", "benchmark", "ace", "ace", "confirmed"),
            ("Overall", "benchmark", "apex-v1", "apex-v1", "confirmed"),
        )
        resolver = Resolver(store)
        # Different config — scoped alias must not match.
        result = resolver.resolve("Overall", "benchmark", source_config="hfopenllm_v2")
        assert result.canonical_id is None
        assert result.strategy == "no_match"

    def test_scoped_isolated_without_source_config(self):
        store = _store_with_aliases(
            ("Arabic", "benchmark", "global-mmlu-lite", "global-mmlu-lite", "confirmed"),
        )
        resolver = Resolver(store)
        # No source_config provided — scoped alias must not match.
        result = resolver.resolve("Arabic", "benchmark")
        assert result.canonical_id is None

    def test_scoped_normalized_match_respects_scope(self):
        store = _store_with_aliases(
            ("Abstract Algebra", "benchmark", "mmlu", "helm_mmlu", "confirmed"),
        )
        resolver = Resolver(store)
        # Same scope, different casing — normalized strategy must match.
        result = resolver.resolve("abstract algebra", "benchmark", source_config="helm_mmlu")
        assert result.canonical_id == "mmlu"
        assert result.strategy == "normalized"
        # Different scope — must NOT match via normalized either.
        result = resolver.resolve("abstract algebra", "benchmark", source_config="helm_lite")
        assert result.canonical_id is None

    def test_global_alias_still_matches_any_scope(self):
        store = _store_with_aliases(("MMLU", "benchmark", "mmlu", None, "confirmed"))
        resolver = Resolver(store)
        for sc in [None, "helm_mmlu", "helm_lite", "some_other"]:
            result = resolver.resolve("MMLU", "benchmark", source_config=sc)
            assert result.canonical_id == "mmlu", f"failed for source_config={sc}"


class TestNormalizedStrategy:
    def test_normalized_match(self):
        store = _store_with_aliases(("MATH Level 5", "benchmark", "math-level-5", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("math level 5", "benchmark")
        assert result.canonical_id == "math-level-5"
        assert result.strategy == "normalized"

    def test_punctuation_stripped(self):
        store = _store_with_aliases(("GPQA!", "benchmark", "gpqa", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("gpqa", "benchmark")
        assert result.canonical_id == "gpqa"
        assert result.strategy in ("exact", "normalized")


class TestFuzzyStrategy:
    def test_suffix_strip_matches_base(self):
        """model-name-fc should match model-name via suffix stripping."""
        store = _store_with_aliases(("writer/palmyra-x-004", "model", "writer/palmyra-x-004", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("writer/palmyra-x-004-fc", "model")
        assert result.canonical_id == "writer/palmyra-x-004"
        assert result.strategy == "fuzzy"

    def test_org_normalization_matches(self):
        """deepseek-ai/model should match deepseek/model via org alias."""
        store = _store_with_aliases(("deepseek/deepseek-r1-0528", "model", "deepseek/deepseek-r1-0528", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("deepseek-ai/deepseek-r1-0528", "model")
        assert result.canonical_id == "deepseek/deepseek-r1-0528"
        assert result.strategy == "fuzzy"

    def test_distinct_versions_not_merged(self):
        """gpt-5-mini should NOT fuzzy-match to gpt-5 — they are different models."""
        store = _store_with_aliases(("openai/gpt-5-2025-08-07", "model", "openai/gpt-5-2025-08-07", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("openai/gpt-5-mini-2025-08-07", "model")
        assert result.canonical_id is None
        assert result.strategy == "no_match"

    def test_distinct_benchmarks_not_merged(self):
        """fibble2 should NOT fuzzy-match to fibble1."""
        store = _store_with_aliases(("fibble1_arena_win_rate", "benchmark", "fibble1-arena-win-rate", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("fibble2_arena_win_rate", "benchmark")
        assert result.canonical_id is None
        assert result.strategy == "no_match"

    def test_no_match_on_unrelated_string(self):
        store = _store_with_aliases(("completely-different", "harness", "x", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("unrelated string xyz", "harness")
        assert result.canonical_id is None
        assert result.strategy == "no_match"

    def test_thinking_budget_suffix_stripped(self):
        """claude-opus-4-5-thinking-16k should match claude-opus-4-5 (card_backend pattern)."""
        store = _store_with_aliases(
            ("anthropic/claude-opus-4-5", "model", "anthropic/claude-opus-4-5", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve("anthropic/claude-opus-4-5-thinking-16k", "model")
        assert result.canonical_id == "anthropic/claude-opus-4-5"
        assert result.strategy == "fuzzy"

    def test_thinking_none_suffix_stripped(self):
        store = _store_with_aliases(
            ("anthropic/claude-opus-4-5", "model", "anthropic/claude-opus-4-5", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve("anthropic/claude-opus-4-5-thinking-none", "model")
        assert result.canonical_id == "anthropic/claude-opus-4-5"
        assert result.strategy == "fuzzy"

    def test_date_version_suffix_stripped(self):
        """Date suffixes like -20251101 should be stripped (card_backend pattern)."""
        store = _store_with_aliases(
            ("anthropic/claude-opus-4-5", "model", "anthropic/claude-opus-4-5", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve("anthropic/claude-opus-4-5-20251101", "model")
        assert result.canonical_id == "anthropic/claude-opus-4-5"
        assert result.strategy == "fuzzy"

    def test_date_plus_thinking_double_strip(self):
        """Combined date + thinking suffix should resolve via double-strip."""
        store = _store_with_aliases(
            ("anthropic/claude-opus-4-5", "model", "anthropic/claude-opus-4-5", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve("anthropic/claude-opus-4-5-20251101-thinking-16k", "model")
        assert result.canonical_id == "anthropic/claude-opus-4-5"
        assert result.strategy == "fuzzy"

    def test_dot_version_normalizes_to_hyphen(self):
        """claude-opus-4.5 should normalize the same as claude-opus-4-5."""
        store = _store_with_aliases(
            ("anthropic/claude-opus-4-5", "model", "anthropic/claude-opus-4-5", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve("anthropic/claude-opus-4.5", "model")
        assert result.canonical_id == "anthropic/claude-opus-4-5"
        assert result.strategy == "normalized"

    def test_meta_llama_org_alias(self):
        """meta-llama/ → meta/ via expanded org alias map."""
        store = _store_with_aliases(("meta/llama-3-70b", "model", "meta/llama-3-70b", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("meta-llama/llama-3-70b", "model")
        assert result.canonical_id == "meta/llama-3-70b"
        assert result.strategy == "fuzzy"

    def test_qwen_org_alias_to_alibaba(self):
        """`Qwen/<model>` (HF-namespace upload form) → `alibaba/<model>`
        via the qwen → alibaba org alias. The reverse direction
        (alibaba → qwen) was rejected because of the non-Qwen
        `alibaba/mineru2-pipeline` entry; this direction has no analogous
        collision."""
        store = _store_with_aliases(
            ("alibaba/qwen2-vl-7b-instruct", "model", "alibaba/qwen2-vl-7b-instruct", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve("Qwen/Qwen2-VL-7B-Instruct", "model")
        assert result.canonical_id == "alibaba/qwen2-vl-7b-instruct"

    def test_year_only_strips_for_openai_shaped_via_iso_date_peel(self):
        """`openai/gpt-5-2024` peels the trailing year to `openai/gpt-5`
        via the OpenAI ISO-date strip. Strict year-range guard
        (2015–2035) + lookup verification means non-year 4-digit tails
        (e.g. `-1024`) and non-aliased truncations are unaffected; see
        the explicit regression tests below."""
        store = _store_with_aliases(("openai/gpt-5", "model", "openai/gpt-5", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("openai/gpt-5-2024", "model")
        assert result.canonical_id == "openai/gpt-5"
        assert result.strategy == "fuzzy"

    def test_iso_date_strip_does_not_apply_to_non_openai(self):
        """The OpenAI date-peel is org-scoped: `meta/llama-3-2024` must NOT
        strip to `meta/llama-3` because Meta's release cadence doesn't use
        the OpenAI YYYY-MM-DD truncated-month convention."""
        store = _store_with_aliases(("meta/llama-3", "model", "meta/llama-3", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("meta/llama-3-2024", "model")
        assert result.canonical_id is None
        assert result.strategy == "no_match"

    def test_iso_date_strip_rejects_non_year_4digit_tail(self):
        """The year-range guard (2015–2035) prevents arbitrary 4-digit
        tails like `-1024` (a number, not a year) from triggering the
        peel even on OpenAI-shaped raws."""
        store = _store_with_aliases(("openai/gpt-5", "model", "openai/gpt-5", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("openai/gpt-5-1024", "model")
        assert result.canonical_id is None
        assert result.strategy == "no_match"

    def test_iso_date_strip_full_date_progressive_peel(self):
        """`openai/gpt-5-2025-08-07` peels in three steps until it hits an
        existing alias. Lookup-verified: when only the bare family is
        aliased, that's what wins."""
        store = _store_with_aliases(("openai/gpt-5", "model", "openai/gpt-5", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("openai/gpt-5-2025-08-07", "model")
        assert result.canonical_id == "openai/gpt-5"
        assert result.strategy == "fuzzy"

    def test_iso_date_strip_prefers_truncated_month_canonical(self):
        """When the registry has both the truncated-month canonical and
        the family root, the peel stops at the first hit (truncated
        month) — preserves snapshot identity instead of over-collapsing."""
        store = _store_with_aliases(
            ("openai/gpt-5", "model", "openai/gpt-5", None, "confirmed"),
            ("openai/gpt-5-2025-08", "model", "openai/gpt-5-2025-08", None, "confirmed"),
        )
        resolver = Resolver(store)
        result = resolver.resolve("openai/gpt-5-2025-08-07", "model")
        assert result.canonical_id == "openai/gpt-5-2025-08"
        assert result.strategy == "fuzzy"

    def test_iso_date_strip_handles_unknown_host_prefix(self):
        """`unknown/gpt-5-2025-08-07` (placeholder host prefix) — host
        strip drops `unknown/`, then ISO-date peel reduces the
        OpenAI-shaped body. Production registry registers both the
        bare alias (`gpt-5 → openai/gpt-5`) and the prefixed canonical;
        the test mirrors that so the host-stripped candidate finds a hit."""
        store = _store_with_aliases(
            ("openai/gpt-5", "model", "openai/gpt-5", None, "confirmed"),
            ("gpt-5",        "model", "openai/gpt-5", None, "confirmed"),
        )
        resolver = Resolver(store)
        result = resolver.resolve("unknown/gpt-5-2025-08-07", "model")
        assert result.canonical_id == "openai/gpt-5"


class TestNoMatch:
    def test_empty_store(self):
        store = _store_with_aliases()
        resolver = Resolver(store)
        result = resolver.resolve("anything", "benchmark")
        assert result.canonical_id is None
        assert result.strategy == "no_match"
        assert result.confidence == 0.0


class TestPromotedVariantResolution:
    """After the alias-promotion pass, instruct/chat/quantized/snapshot
    variants are first-class canonicals with their own aliases. Resolving
    a raw value matching one of those variants must NOT collapse to the
    base — eval scores on `Llama-3-8B-Instruct` aren't comparable to
    scores on the base `Llama-3-8B`. Regression coverage: there is no
    `_FAMILY_STAGE_SUFFIXES` strip in fuzzy.py — the resolver relies on
    explicit alias entries for instruct/chat/etc., and stays out of the
    way for unknown post-training suffixes."""

    def _registry_like_store(self):
        """Mini fixture mimicking the post-promotion registry: base + promoted
        instruct + promoted instruct-quant, each with their own surface-form
        aliases."""
        return _store_with_aliases(
            # Base
            ("Llama-3-8B", "model", "meta/llama-3-8b", None, "confirmed"),
            ("meta/llama-3-8b", "model", "meta/llama-3-8b", None, "confirmed"),
            # Promoted instruct (variant/mode of the base)
            ("Llama-3-8B-Instruct", "model", "meta/llama-3-8b-instruct", None, "confirmed"),
            ("meta-llama/Meta-Llama-3-8B-Instruct", "model", "meta/llama-3-8b-instruct", None, "confirmed"),
            ("meta/llama-3-8b-instruct", "model", "meta/llama-3-8b-instruct", None, "confirmed"),
            # Promoted instruct-turbo (quantized of the instruct)
            ("Llama-3-8B-Instruct-Turbo", "model", "meta/llama-3-8b-instruct-turbo", None, "confirmed"),
            ("meta/llama-3-8b-instruct-turbo", "model", "meta/llama-3-8b-instruct-turbo", None, "confirmed"),
            # Promoted snapshot (variant/version of a base)
            ("gpt-4-0613", "model", "openai/gpt-4-0613", None, "confirmed"),
            ("openai/gpt-4-0613", "model", "openai/gpt-4-0613", None, "confirmed"),
            ("openai/gpt-4", "model", "openai/gpt-4", None, "confirmed"),
        )

    def test_instruct_resolves_to_instruct_canonical(self):
        resolver = Resolver(self._registry_like_store())
        result = resolver.resolve("Llama-3-8B-Instruct", "model")
        assert result.canonical_id == "meta/llama-3-8b-instruct"
        assert result.strategy == "exact"

    def test_base_still_resolves_to_base(self):
        resolver = Resolver(self._registry_like_store())
        result = resolver.resolve("Llama-3-8B", "model")
        assert result.canonical_id == "meta/llama-3-8b"
        assert result.strategy == "exact"

    def test_doubled_org_prefix_instruct_resolves_to_instruct(self):
        """HF-style `meta-llama/Meta-Llama-3-8B-Instruct` (org form duplicated
        inside the bare model id) should land on the instruct canonical."""
        resolver = Resolver(self._registry_like_store())
        result = resolver.resolve("meta-llama/Meta-Llama-3-8B-Instruct", "model")
        assert result.canonical_id == "meta/llama-3-8b-instruct"

    def test_quantized_variant_resolves_to_quantized_canonical(self):
        resolver = Resolver(self._registry_like_store())
        result = resolver.resolve("Llama-3-8B-Instruct-Turbo", "model")
        assert result.canonical_id == "meta/llama-3-8b-instruct-turbo"

    def test_snapshot_resolves_to_snapshot_canonical(self):
        resolver = Resolver(self._registry_like_store())
        result = resolver.resolve("gpt-4-0613", "model")
        assert result.canonical_id == "openai/gpt-4-0613"
        # Sanity check that the snapshot is NOT collapsed onto the base
        assert result.canonical_id != "openai/gpt-4"

    def test_quant_falls_through_to_nearest_parent(self):
        """When a specific quantization isn't a canonical, the fuzzy stem
        strip drops the `-fp8` suffix and lands on the next-up canonical
        (the unquantized instruct, in this case). This is the
        precision-loss policy we explicitly opted into."""
        resolver = Resolver(self._registry_like_store())
        result = resolver.resolve("meta/llama-3-8b-instruct-fp8", "model")
        # -fp8 is in _STRIP_SUFFIXES, so fuzzy strips and lands on instruct
        assert result.canonical_id == "meta/llama-3-8b-instruct"
        assert result.strategy == "fuzzy"

    def test_unknown_finetune_suffix_does_not_collapse_to_base(self):
        """If a raw value has an unrecognized suffix (a community finetune
        we haven't catalogued), the resolver must NOT silently strip it
        and land on the base — that would misattribute scores. Returns
        no_match so the caller can auto-draft a separate canonical."""
        resolver = Resolver(self._registry_like_store())
        result = resolver.resolve("Llama-3-8B-Instruct-CommunityFinetune", "model")
        assert result.canonical_id is None
        assert result.strategy == "no_match"

    def test_unknown_instruct_does_not_collapse_to_base(self):
        """Specifically: there is no resolver-side `-instruct` strip. An
        instruct variant we haven't promoted yet does NOT silently fall
        through to the base."""
        store = _store_with_aliases(
            ("Llama-3-8B", "model", "meta/llama-3-8b", None, "confirmed"),
            ("meta/llama-3-8b", "model", "meta/llama-3-8b", None, "confirmed"),
            # NB: no -instruct alias / canonical seeded
        )
        resolver = Resolver(store)
        result = resolver.resolve("Llama-3-8B-Instruct", "model")
        assert result.canonical_id is None
        assert result.strategy == "no_match"


class TestSnakeCaseEquivalence:
    """Snake_case forms of seeded display-form aliases resolve via normalized matcher,
    without requiring the snake_case alias to be listed explicitly."""

    @pytest.mark.parametrize(
        "raw,expected_canonical",
        [
            ("easy_problems", "livecodebench-pro"),
            ("medium_problems", "livecodebench-pro"),
            ("hard_problems", "livecodebench-pro"),
        ],
    )
    def test_lcb_pro_difficulty_tiers_normalize(self, raw, expected_canonical):
        store = _store_with_aliases(
            ("Easy Problems", "benchmark", "livecodebench-pro", "livecodebenchpro", "confirmed"),
            ("Medium Problems", "benchmark", "livecodebench-pro", "livecodebenchpro", "confirmed"),
            ("Hard Problems", "benchmark", "livecodebench-pro", "livecodebenchpro", "confirmed"),
        )
        resolver = Resolver(store)
        result = resolver.resolve(raw, "benchmark", source_config="livecodebenchpro")
        assert result.canonical_id == expected_canonical
        assert result.strategy == "normalized"
