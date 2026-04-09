"""Tests for resolution_service: entity lifecycle, orphan cleanup, rerun behavior."""
import json
import pytest

from eval_card_registry.store.hf_store import RegistryStore
from eval_card_registry.store import schemas
from eval_card_registry.services.resolution_service import ResolutionService


def _fresh_store() -> RegistryStore:
    store = RegistryStore()
    from eval_card_registry.store import schemas as s
    store._tables = {name: s.empty(name) for name in [
        "canonical_models", "canonical_benchmarks", "canonical_metrics",
        "eval_harnesses", "aliases", "resolution_log", "eval_results", "sync_runs",
    ]}
    store._loaded = True
    return store


def _seed_benchmark(store: RegistryStore, id: str, display_name: str):
    from eval_card_registry.store import queries
    import json
    queries.upsert_entity(store, "canonical_benchmarks", {
        "id": id,
        "display_name": display_name,
        "description": None,
        "dataset_repo": None,
        "parent_benchmark_id": None,
        "tags": "[]",
        "metadata": "{}",
        "review_status": "reviewed",
    })
    # Add alias for exact lookup
    queries.add_alias(store, {
        "raw_value": display_name,
        "entity_type": "benchmark",
        "canonical_id": id,
        "source_config": None,
        "source_field": None,
        "status": "confirmed",
        "strategy": "exact",
        "confidence": 1.0,
        "notes": None,
    })


class TestResolutionService:
    def test_resolves_known_entity(self):
        store = _fresh_store()
        _seed_benchmark(store, "ifeval", "IFEval")
        svc = ResolutionService(store)
        result = svc.resolve("IFEval", "benchmark", None, None)
        assert result["canonical_id"] == "ifeval"
        assert result["created_new"] is False

    def test_auto_creates_draft_on_no_match(self):
        store = _fresh_store()
        svc = ResolutionService(store)
        result = svc.resolve("Some Unknown Benchmark", "benchmark", "test_config", None)
        assert result["canonical_id"] is not None
        assert result["created_new"] is True
        assert result["review_status"] == "draft"

    def test_idempotent_on_second_call(self):
        from eval_card_registry.store import queries
        store = _fresh_store()
        svc = ResolutionService(store)
        r1 = svc.resolve("Novel Benchmark X", "benchmark", "cfg", None)
        r2 = svc.resolve("Novel Benchmark X", "benchmark", "cfg", None)
        assert r1["canonical_id"] == r2["canonical_id"]
        # Flush pending writes and verify only one entity was created
        queries.flush_pending(store)
        df = store.table("canonical_benchmarks")
        assert len(df) == 1

    def test_rerun_bypasses_cache(self):
        store = _fresh_store()
        _seed_benchmark(store, "real-benchmark-y", "Novel Benchmark Y")
        svc = ResolutionService(store)

        # First call populates alias
        r1 = svc.resolve("Novel Benchmark Y", "benchmark", "cfg", None)
        assert r1["canonical_id"] == "real-benchmark-y"

        # Second call without rerun uses alias cache
        r2 = svc.resolve("Novel Benchmark Y", "benchmark", "cfg", None, rerun=False)
        assert r2["canonical_id"] == "real-benchmark-y"
        assert r2["created_new"] is False

        # rerun=True re-evaluates; result is the same since seed data hasn't changed
        svc.invalidate_resolver()
        r3 = svc.resolve("Novel Benchmark Y", "benchmark", "cfg", None, rerun=True)
        assert r3["canonical_id"] == "real-benchmark-y"

    def test_subset_alias_resolves_to_parent(self):
        """A subset aliased to a parent entity resolves to the parent, not a new entity."""
        store = _fresh_store()
        _seed_benchmark(store, "parent-bench", "Parent Bench")
        from eval_card_registry.store import queries
        queries.add_alias(store, {
            "raw_value": "Parent Bench Subset X",
            "entity_type": "benchmark",
            "canonical_id": "parent-bench",
            "source_config": None,
            "source_field": None,
            "status": "confirmed",
            "strategy": "exact",
            "confidence": 1.0,
            "notes": None,
        })
        svc = ResolutionService(store)
        result = svc.resolve("Parent Bench Subset X", "benchmark", "some_config", None)
        assert result["canonical_id"] == "parent-bench"
        assert result["created_new"] is False

    def test_empty_raw_value_returns_none(self):
        store = _fresh_store()
        svc = ResolutionService(store)
        result = svc.resolve("", "benchmark", "cfg", None)
        assert result["canonical_id"] is None
        assert result["strategy"] == "no_match"
