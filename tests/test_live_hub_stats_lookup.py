"""Tests for live hub-stats lookup at draft creation.

Covers the integration in `services/resolution_service._auto_create_entity`
that pre-populates a model draft with hub-stats metadata when the
unmatched raw value looks like an HF id."""
from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import patch

import pytest

from eval_card_registry.config import settings
from eval_card_registry.services import hub_stats as _hs
from eval_card_registry.services.resolution_service import ResolutionService
from eval_card_registry.store import hf_store, queries, schemas


@pytest.fixture
def fresh_store():
    store = hf_store.RegistryStore()
    store._tables = {n: schemas.empty(n) for n in [
        "canonical_orgs", "canonical_models", "canonical_benchmarks",
        "canonical_metrics", "eval_harnesses", "aliases",
        "resolution_log", "eval_results", "sync_runs",
    ]}
    store._loaded = True
    return store


@pytest.fixture
def enable_lookup():
    """Re-enable hub-stats lookup for the duration of a test (the
    autouse conftest fixture turns it off globally)."""
    settings.hub_stats_lookup_enabled = True
    yield
    settings.hub_stats_lookup_enabled = False


# ---------- _looks_like_hf_id heuristic ----------

@pytest.mark.parametrize("value,expected", [
    ("meta-llama/Llama-3.1-8B", True),
    ("meta-llama/Llama-3.1-8B-Instruct", True),
    ("Just-A-Model-Name", False),       # no slash
    ("multi/segment/path", False),      # multiple slashes — not a clean HF id
    ("/leading-slash", False),
    ("trailing-slash/", False),
    ("", False),
])
def test_looks_like_hf_id(value, expected):
    assert ResolutionService._looks_like_hf_id(value) is expected


# ---------- Lookup gated by config flag ----------

def test_lookup_disabled_returns_none(fresh_store):
    """When hub_stats_lookup_enabled is False (default in tests), the
    auto-create path skips the lookup entirely — no hub-stats import,
    no network call."""
    svc = ResolutionService(fresh_store)
    # The autouse conftest already disables it; verify behavior:
    assert settings.hub_stats_lookup_enabled is False
    assert svc._lookup_hub_stats("meta-llama/Llama-3.1-8B") is None


# ---------- Auto-create with mocked hub-stats client ----------

def _seed_org(store, oid, hf_org=None):
    queries.upsert_entity(store, "canonical_orgs", {
        "id": oid, "display_name": oid, "parent_org_id": None,
        "website": None, "hf_org": hf_org, "kind": "lab",
        "tags": "[]", "metadata": "{}", "review_status": "reviewed",
    })
    queries.add_alias(store, {
        "raw_value": oid, "entity_type": "org", "canonical_id": oid,
        "source_config": None, "source_field": "test",
        "status": "confirmed", "strategy": "seed",
        "confidence": 1.0, "notes": None,
    })


def _seed_model(store, mid, org_id):
    queries.upsert_entity(store, "canonical_models", {
        "id": mid, "display_name": mid, "developer": None,
        "org_id": org_id, "family": None, "architecture": None,
        "params_billions": None, "parents": "[]",
        "root_model_id": None, "lineage_origin_org_id": org_id,
        "tags": "[]", "metadata": "{}", "review_status": "reviewed",
    })
    queries.add_alias(store, {
        "raw_value": mid, "entity_type": "model", "canonical_id": mid,
        "source_config": None, "source_field": "test",
        "status": "confirmed", "strategy": "seed",
        "confidence": 1.0, "notes": None,
    })


def test_auto_create_enriches_with_hub_stats_metadata(fresh_store, enable_lookup):
    """When an unknown HF-shaped raw value comes through, the auto-create
    path queries hub-stats (mocked here) and pre-populates release_date,
    params_billions, and lineage_origin_org_id."""
    _seed_org(fresh_store, "meta", hf_org="meta-llama")
    _seed_model(fresh_store, "meta/llama-3.1-8b", "meta")

    fake_row = {
        "id": "casperhansen/llama-3.1-8b-instruct-awq",
        "author": "casperhansen",
        "createdAt": datetime(2024, 12, 6),
        "tags": ["safetensors", "license:llama3.1"],
        "cardData": {"license": "llama3.1"},
        "safetensors": {"total": 16_000_000_000},  # ~8B params
        "baseModels": {
            "relation": "quantized",
            "models": [{"id": "meta-llama/Llama-3.1-8B"}],  # in our aliases
        },
        "library_name": "transformers",
        "pipeline_tag": "text-generation",
        "downloads": 100, "downloadsAllTime": 5000,
        "likes": 10, "trendingScore": 0, "lastModified": None,
    }
    # Seed an alias so baseModels resolution finds our canonical
    queries.add_alias(fresh_store, {
        "raw_value": "meta-llama/Llama-3.1-8B", "entity_type": "model",
        "canonical_id": "meta/llama-3.1-8b", "source_config": None,
        "source_field": "test", "status": "confirmed",
        "strategy": "seed", "confidence": 1.0, "notes": None,
    })

    svc = ResolutionService(fresh_store)
    with patch.object(_hs.HubStatsClient, "lookup", return_value=fake_row):
        cid = svc._auto_create_entity("model", "casperhansen/Llama-3.1-8B-Instruct-AWQ")

    queries.flush_pending(fresh_store)
    df = fresh_store.table("canonical_models")
    new_row = df[df["id"] == cid].iloc[0]

    # Enriched fields landed on the draft
    assert new_row["release_date"] == "2024-12-06"
    assert new_row["params_billions"] == 8.0
    parents = json.loads(new_row["parents"])
    assert parents == [{"id": "meta/llama-3.1-8b", "relationship": "quantized"}]
    assert new_row["lineage_origin_org_id"] == "meta"
    # Defaults that hub-stats had no data for stay as defaults
    assert new_row["root_model_id"] is None or queries._is_na(new_row["root_model_id"])
    # Metadata carries the source marker
    md = json.loads(new_row["metadata"])
    assert md["source"] == "hub_stats"
    assert md["license"] == "llama3.1"


def test_auto_create_falls_back_when_lookup_returns_none(fresh_store, enable_lookup):
    """When hub-stats has no row for the HF id, the draft is created
    with NULL/empty defaults — no field is corrupted."""
    svc = ResolutionService(fresh_store)
    with patch.object(_hs.HubStatsClient, "lookup", return_value=None):
        cid = svc._auto_create_entity("model", "totally-new-author/Some-Model-7B")

    queries.flush_pending(fresh_store)
    df = fresh_store.table("canonical_models")
    new_row = df[df["id"] == cid].iloc[0]
    # All defaults — no enrichment
    assert new_row["params_billions"] is None or queries._is_na(new_row["params_billions"])
    assert new_row["release_date"] is None or queries._is_na(new_row["release_date"])
    assert new_row["parents"] == "[]"
    assert new_row["review_status"] == "draft"


def test_auto_create_falls_back_when_lookup_raises(fresh_store, enable_lookup):
    """Any exception from the lookup (network down, malformed parquet,
    DuckDB error) must NOT block draft creation. The draft lands with
    plain defaults."""
    svc = ResolutionService(fresh_store)
    with patch.object(_hs.HubStatsClient, "lookup",
                      side_effect=RuntimeError("network unreachable")):
        cid = svc._auto_create_entity("model", "some-org/some-model")

    queries.flush_pending(fresh_store)
    df = fresh_store.table("canonical_models")
    new_row = df[df["id"] == cid].iloc[0]
    assert new_row["review_status"] == "draft"
    assert new_row["parents"] == "[]"


def test_auto_create_skips_lookup_for_non_hf_shaped_value(fresh_store, enable_lookup):
    """Raw values that don't look like an HF id (no slash, multiple
    slashes) must NOT trigger a hub-stats lookup — saves a network call
    we know will return nothing."""
    svc = ResolutionService(fresh_store)
    with patch.object(_hs.HubStatsClient, "lookup") as mock_lookup:
        svc._auto_create_entity("model", "Just-A-Model-Name")
        svc._auto_create_entity("model", "multi/path/segments")
    mock_lookup.assert_not_called()


def test_auto_create_skips_lookup_for_non_model_entity_types(fresh_store, enable_lookup):
    """Live lookup is model-specific. Auto-creating a benchmark with an
    HF-id-shaped raw value (unusual but possible) must NOT hit hub-stats."""
    svc = ResolutionService(fresh_store)
    with patch.object(_hs.HubStatsClient, "lookup") as mock_lookup:
        svc._auto_create_entity("benchmark", "some/weird-benchmark-name")
    mock_lookup.assert_not_called()


# ---------- enrich_draft_from_row helper ----------

def test_enrich_draft_resolves_known_base_to_canonical():
    """Hub-stats baseModels references an HF id; helper resolves it to
    OUR canonical id via the alias index."""
    aliases_to_canonical = {
        "meta-llama-llama-3-1-8b": "meta/llama-3.1-8b",
    }
    org_alias_map = {"meta-llama": "meta", "meta": "meta"}
    row = {
        "id": "casperhansen/llama-3.1-8b-fp8",
        "createdAt": None,
        "tags": [],
        "cardData": None,
        "safetensors": None,
        "baseModels": {
            "relation": "quantized",
            "models": [{"id": "meta-llama/Llama-3.1-8B"}],
        },
        "downloadsAllTime": None, "likes": None,
        "library_name": None, "pipeline_tag": None,
    }
    out = _hs.enrich_draft_from_row(row, aliases_to_canonical, org_alias_map)
    parents = json.loads(out["parents"])
    assert parents == [{"id": "meta/llama-3.1-8b", "relationship": "quantized"}]
    # quantized is non-variant → lineage origin set
    assert out["lineage_origin_org_id"] == "meta"


def test_enrich_draft_sets_open_weights_for_hf_artifact_rows():
    """Live lookup mirrors the bulk script: presence of safetensors/gguf
    → open_weights=True. Absence → field omitted (so the merge into the
    draft leaves it NULL rather than asserting closed-weight)."""
    row_with = {
        "id": "x/y", "createdAt": None, "tags": [], "cardData": None,
        "safetensors": {"total": 14_000_000_000},
        "baseModels": None,
        "downloadsAllTime": None, "likes": None,
        "library_name": None, "pipeline_tag": None,
    }
    out = _hs.enrich_draft_from_row(row_with, {}, {})
    assert out.get("open_weights") is True

    row_without = {**row_with, "safetensors": None}
    out2 = _hs.enrich_draft_from_row(row_without, {}, {})
    assert "open_weights" not in out2


def test_enrich_draft_drops_unresolved_base_edges():
    """If baseModels references something we don't track, drop the edge
    rather than emit a dangling parent. lineage_origin_org_id stays
    empty in that case."""
    row = {
        "id": "x/y",
        "createdAt": None, "tags": [], "cardData": None,
        "safetensors": None,
        "baseModels": {
            "relation": "finetune",
            "models": [{"id": "unknown-org/totally-unknown"}],
        },
        "downloadsAllTime": None, "likes": None,
        "library_name": None, "pipeline_tag": None,
    }
    out = _hs.enrich_draft_from_row(row, {}, {})
    assert "parents" not in out
    assert "lineage_origin_org_id" not in out


# ---------- HubStatsClient cache ----------

def test_hub_stats_client_caches_lookups():
    """Two lookups for the same id must hit the cache, not the DuckDB
    backend, on the second call."""
    client = _hs.HubStatsClient()
    fake = {"id": "x/y", "createdAt": None}
    call_count = 0

    def _fake_ensure_con():
        nonlocal call_count
        call_count += 1

        class _C:
            def execute(self, sql):
                class _Cur:
                    description = [("id",), ("createdAt",)]
                    def fetchone(self):
                        return ("x/y", None)
                return _Cur()
        return _C()

    with patch.object(client, "_ensure_con", side_effect=_fake_ensure_con):
        r1 = client.lookup("x/y")
        r2 = client.lookup("x/y")
    assert r1 == r2
    assert call_count == 1, "second lookup must hit cache, not backend"
