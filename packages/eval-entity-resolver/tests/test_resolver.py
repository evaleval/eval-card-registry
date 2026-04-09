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


class TestNoMatch:
    def test_empty_store(self):
        store = _store_with_aliases()
        resolver = Resolver(store)
        result = resolver.resolve("anything", "benchmark")
        assert result.canonical_id is None
        assert result.strategy == "no_match"
        assert result.confidence == 0.0
