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
        "model_group_id": None, "lineage_origin_model_org_id": org_id,
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
    params_billions, and lineage_origin_model_org_id."""
    _seed_org(fresh_store, "meta", hf_org="meta-llama")
    _seed_model(fresh_store, "meta/llama-3.1-8b", "meta")

    fake_row = {
        "id": "casperhansen/llama-3.1-8b-instruct-awq",
        "author": "casperhansen",
        "createdAt": datetime(2024, 12, 6),
        "tags": ["safetensors", "license:llama3.1"],
        "cardData": {"license": "llama3.1"},
        "safetensors": {"total": 8_000_000_000},  # total = param count -> 8.0B
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
    assert new_row["lineage_origin_model_org_id"] == "meta"
    # Defaults that hub-stats had no data for stay as defaults
    assert new_row["model_group_id"] is None or queries._is_na(new_row["model_group_id"])
    # Metadata carries the source marker
    md = json.loads(new_row["metadata"])
    assert md["source"] == "hub_stats"
    assert md["license"] == "llama3.1"


def test_auto_create_mints_hf_cased_id_and_marks_source_hf(fresh_store, enable_lookup):
    """HF source-of-truth casing: when hub-stats confirms the repo,
    the auto-create path derives the canonical id + display_name from the HF
    TRUE casing (via the two-tier org rule: HF-org -> developer slug) instead
    of a lowercased slug, and
    stamps `resolution_source='hf'` / `review_status='reviewed'`. The HF
    model-name casing is preserved; the curated dev namespace remaps the org
    (meta-llama -> meta) while the curated slug keeps its authored casing."""
    _seed_org(fresh_store, "meta", hf_org="meta-llama")

    fake_row = {
        "id": "meta-llama/Llama-3.1-8B-Instruct",   # HF-true casing
        "author": "meta-llama",
        "createdAt": datetime(2024, 7, 23),
        "tags": ["safetensors"],
        "cardData": {"license": "llama3.1"},
        "safetensors": {"total": 16_000_000_000},
        "baseModels": None,
        "library_name": "transformers",
        "pipeline_tag": "text-generation",
        "downloads": 100, "downloadsAllTime": 5000,
        "likes": 10, "trendingScore": 0, "lastModified": None,
    }

    svc = ResolutionService(fresh_store)
    # The raw EEE id arrives lowercased — the lookup matches case-insensitively
    # and the HF-true id drives the canonical casing.
    with patch.object(_hs.HubStatsClient, "lookup", return_value=fake_row):
        cid = svc._auto_create_entity("model", "meta-llama/llama-3.1-8b-instruct")

    # canonical_id is the real HF repo id (org never folded into the id);
    # only org_id remaps to `meta`.
    assert cid == "meta-llama/Llama-3.1-8B-Instruct"

    queries.flush_pending(fresh_store)
    df = fresh_store.table("canonical_models")
    new_row = df[df["id"] == cid].iloc[0]
    assert new_row["display_name"] == "Llama-3.1-8B-Instruct"  # HF NAME casing
    assert new_row["org_id"] == "meta"
    assert new_row["resolution_source"] == "hf"
    assert new_row["review_status"] == "reviewed"


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


def test_auto_create_infers_family_parent_when_hub_stats_misses(fresh_store, enable_lookup):
    """The family-version inference fallback: even when hub-stats returns
    None (parquet stale, brand-new model on release day), if a snapshot
    HF id strips to an existing canonical via the alias index, the draft
    still gets a `{variant, axis: version}` parent edge. Without this,
    same-day releases would orphan until the next hub-stats refresh."""
    _seed_org(fresh_store, "allenai")
    _seed_model(fresh_store, "allenai/olmo-3-32b", "allenai")
    svc = ResolutionService(fresh_store)
    with patch.object(_hs.HubStatsClient, "lookup", return_value=None):
        cid = svc._auto_create_entity("model", "allenai/Olmo-3-1125-32B")

    queries.flush_pending(fresh_store)
    df = fresh_store.table("canonical_models")
    new_row = df[df["id"] == cid].iloc[0]
    parents = json.loads(new_row["parents"])
    assert {
        "id": "allenai/olmo-3-32b",
        "relationship": "variant",
        "axis": "version",
    } in parents
    assert new_row["review_status"] == "draft"


def test_auto_create_handles_anthropic_yyyymmdd_snapshot(fresh_store, enable_lookup):
    """End-to-end for the Anthropic shape: resolver fuzzy USED to strip
    `-20251101` and silently match the family pointer. Now: that path
    returns no_match → auto-create runs. With the family pointer aliased
    AND hub-stats returning None (parquet stale), the family-version
    inference fallback in _auto_create_entity still attaches the edge.
    Snapshot canonical lands with a proper version-axis parent."""
    _seed_org(fresh_store, "anthropic")
    _seed_model(fresh_store, "anthropic/claude-opus-4-5", "anthropic")
    svc = ResolutionService(fresh_store)
    with patch.object(_hs.HubStatsClient, "lookup", return_value=None):
        cid = svc._auto_create_entity("model", "anthropic/claude-opus-4-5-20251101")

    queries.flush_pending(fresh_store)
    df = fresh_store.table("canonical_models")
    new_row = df[df["id"] == cid].iloc[0]
    parents = json.loads(new_row["parents"])
    assert {
        "id": "anthropic/claude-opus-4-5",
        "relationship": "variant",
        "axis": "version",
    } in parents


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
    assert out["lineage_origin_model_org_id"] == "meta"


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
    rather than emit a dangling parent. lineage_origin_model_org_id stays
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
    assert "lineage_origin_model_org_id" not in out


# ---------- Family-version parent inference ----------

def test_infer_family_parent_internal_yymm():
    """`Olmo-3-1125-32B`-shape: strip the internal `1125-` MMDD token,
    look up the family in the alias index, return a version-axis edge.
    Without this, dated AllenAI snapshots auto-create as orphans."""
    aliases = {"allenai-olmo-3-32b": "allenai/olmo-3-32b"}
    edge = _hs.infer_family_parent_edge("allenai/Olmo-3-1125-32B", aliases)
    assert edge == {
        "id": "allenai/olmo-3-32b",
        "relationship": "variant",
        "axis": "version",
    }


def test_infer_family_parent_internal_yymm_with_mode_suffix():
    """The strip preserves the mode/size suffix so an `-Instruct` snapshot
    points at the corresponding mode-family pointer when one exists."""
    aliases = {"allenai-olmo-3-7b-instruct": "allenai/olmo-3-7b-instruct"}
    edge = _hs.infer_family_parent_edge(
        "allenai/Olmo-3-1125-7B-Instruct", aliases
    )
    assert edge["id"] == "allenai/olmo-3-7b-instruct"
    assert edge["axis"] == "version"


def test_infer_family_parent_trailing_mmdd():
    """`kimi-k2-0905`-shape (Moonshot/Kimi): trailing 4-digit MMDD."""
    aliases = {"moonshotai-kimi-k2": "moonshotai/kimi-k2"}
    edge = _hs.infer_family_parent_edge("moonshotai/kimi-k2-0905", aliases)
    assert edge["id"] == "moonshotai/kimi-k2"


def test_infer_family_parent_trailing_yyyymmdd():
    """Anthropic-style YYYYMMDD: `claude-haiku-4-5-20251001` → family."""
    aliases = {"anthropic-claude-haiku-4-5": "anthropic/claude-haiku-4-5"}
    edge = _hs.infer_family_parent_edge(
        "anthropic/claude-haiku-4-5-20251001", aliases
    )
    assert edge["id"] == "anthropic/claude-haiku-4-5"


def test_infer_family_parent_trailing_yyyymm():
    """`stepfun/step-2-16k-202411`-shape: trailing 6-digit YYYYMM
    (year+month, no day). Used by Stepfun and several Chinese-lab
    release tags."""
    aliases = {"stepfun-step-2-16k": "stepfun/step-2-16k"}
    edge = _hs.infer_family_parent_edge("stepfun/step-2-16k-202411", aliases)
    assert edge["id"] == "stepfun/step-2-16k"


def test_infer_family_parent_trailing_yyyymm_rejects_invalid_month():
    """`-202413` has an invalid month — must NOT strip."""
    aliases = {"foo-bar": "foo/bar"}
    assert _hs.infer_family_parent_edge("foo/bar-202413", aliases) is None


def test_infer_family_parent_iso_full_date():
    """OpenAI-style `gpt-5-2025-08-07` — ISO ladder strips through
    `-2025-08` and `-2025` to bare. With the family aliased, the bare
    candidate hits."""
    aliases = {"openai-gpt-5": "openai/gpt-5"}
    edge = _hs.infer_family_parent_edge("openai/gpt-5-2025-08-07", aliases)
    assert edge["id"] == "openai/gpt-5"


def test_infer_family_parent_iso_full_date_prefers_intermediate_snapshot():
    """When an intermediate snapshot canonical (`gpt-5-2025-08`) is
    aliased, version-axis edge points at it rather than the family
    root. Preserves snapshot-of-snapshot lineage instead of over-
    collapsing to the topmost family."""
    aliases = {
        "openai-gpt-5": "openai/gpt-5",
        "openai-gpt-5-2025-08": "openai/gpt-5-2025-08",
    }
    edge = _hs.infer_family_parent_edge("openai/gpt-5-2025-08-07", aliases)
    assert edge["id"] == "openai/gpt-5-2025-08"


def test_infer_family_parent_iso_year_only():
    """`gpt-5-2024` strips to bare family via the year-only branch."""
    aliases = {"openai-gpt-5": "openai/gpt-5"}
    edge = _hs.infer_family_parent_edge("openai/gpt-5-2024", aliases)
    assert edge["id"] == "openai/gpt-5"


def test_infer_family_parent_user_examples():
    """The three examples surfaced during planning. Each has the
    family-pointer canonical aliased and gets a version-axis edge."""
    aliases = {
        "google-gemini-exp": "google/gemini-exp",
        "stepfun-step-2-16k": "stepfun/step-2-16k",
        "tencent-hunyuan-turbos": "tencent/hunyuan-turbos",
    }
    cases = [
        ("google/gemini-exp-1114", "google/gemini-exp"),
        ("stepfun/step-2-16k-202411", "stepfun/step-2-16k"),
        ("tencent/hunyuan-turbos-20250313", "tencent/hunyuan-turbos"),
    ]
    for hf_id, expected_family in cases:
        edge = _hs.infer_family_parent_edge(hf_id, aliases)
        assert edge is not None, f"no edge inferred for {hf_id!r}"
        assert edge["id"] == expected_family
        assert edge["axis"] == "version"


def test_infer_family_parent_returns_none_when_no_match():
    """No alias hit → no edge manufactured (don't dangle on stripping
    alone). 4-digit token only triggers when shape matches an MMDD."""
    aliases = {"meta-llama-3-1-8b": "meta/llama-3.1-8b"}
    # No `8888` family in aliases
    assert _hs.infer_family_parent_edge("foo/bar-1125-7b", aliases) is None
    # `8000` is not a valid MMDD (month 80) → not stripped
    assert _hs.infer_family_parent_edge("foo/bar-8000-7b", aliases) is None


def test_infer_family_parent_ignores_non_date_4digit_token():
    """Numeric tokens that aren't valid MMDD shouldn't be stripped, even
    if the stripped form happens to alias something. Guards against
    e.g. ContextLength-shape `-8000-` or version-shape `-2024` collisions."""
    aliases = {"foo-bar-7b": "foo/bar-7b"}
    # 13-something months / 32+ days fail the MMDD shape check
    assert _hs.infer_family_parent_edge("foo/bar-1399-7b", aliases) is None
    assert _hs.infer_family_parent_edge("foo/bar-0099-7b", aliases) is None


def test_enrich_draft_adds_family_version_edge_when_basemodels_misses_it():
    """The combined behavior: hub-stats baseModels records a finetune
    edge to an upstream lab's model, but the snapshot ↔ pointer family
    edge isn't on HF (the pointer is registry-only). Inference adds the
    version edge on top of the upstream edge so root-collapse works."""
    aliases_to_canonical = {
        "allenai-olmo-3-32b": "allenai/olmo-3-32b",
    }
    row = {
        "id": "allenai/Olmo-3-1125-32B",
        "createdAt": "2025-11-25T00:00:00",
        "tags": [],
        "cardData": None,
        "safetensors": {"total": 64_000_000_000},
        "baseModels": None,
        "downloadsAllTime": None, "likes": None,
        "library_name": None, "pipeline_tag": None,
    }
    out = _hs.enrich_draft_from_row(row, aliases_to_canonical, {})
    parents = json.loads(out["parents"])
    assert {
        "id": "allenai/olmo-3-32b",
        "relationship": "variant",
        "axis": "version",
    } in parents
    assert out["release_date"] == "2025-11-25"


def test_infer_family_parent_suppresses_self_edge():
    """When the inferred family equals the target canonical (the row is
    being merged INTO the family pointer rather than a separate snapshot
    canonical), inference must skip — otherwise the family pointer gains
    a parent edge to itself, breaking the lineage walker."""
    aliases = {"allenai-olmo-3-32b": "allenai/olmo-3-32b"}
    edge = _hs.infer_family_parent_edge(
        "allenai/Olmo-3-1125-32B",
        aliases,
        target_canonical="allenai/olmo-3-32b",
    )
    assert edge is None


def test_enrich_draft_suppresses_self_edge_via_target_canonical():
    """End-to-end: the bulk-refresh path passes target_canonical when an
    HF id maps to a family-pointer canonical. enrich_draft_from_row must
    not write a self-edge into parents."""
    aliases_to_canonical = {"allenai-olmo-3-32b": "allenai/olmo-3-32b"}
    row = {
        "id": "allenai/Olmo-3-1125-32B",
        "createdAt": "2025-11-25T00:00:00",
        "tags": [], "cardData": None, "safetensors": None,
        "baseModels": None,
        "downloadsAllTime": None, "likes": None,
        "library_name": None, "pipeline_tag": None,
    }
    out = _hs.enrich_draft_from_row(
        row, aliases_to_canonical, {},
        target_canonical="allenai/olmo-3-32b",
    )
    # No parents key (the inferred edge would be self; nothing else to add)
    assert "parents" not in out
    # Other fields still land
    assert out.get("release_date") == "2025-11-25"


def test_enrich_draft_does_not_double_add_version_edge():
    """If hub-stats already provided a version-axis edge (rare but
    possible if HF ever surfaces it), inference should NOT add a second
    one. Idempotency check."""
    aliases_to_canonical = {
        "allenai-olmo-3-32b": "allenai/olmo-3-32b",
    }
    # Construct a row where extract_base_models would yield a variant
    # edge directly (synthetic — HF doesn't currently emit `variant`,
    # but the shape is what matters here).
    row = {
        "id": "allenai/Olmo-3-1125-32B",
        "createdAt": None, "tags": [], "cardData": None, "safetensors": None,
        "baseModels": {
            "relation": "variant",
            "models": [{"id": "allenai/Olmo-3-32B"}],
        },
        "downloadsAllTime": None, "likes": None,
        "library_name": None, "pipeline_tag": None,
    }
    # NB: hub-stats `baseModels.relation == "variant"` arrives without an
    # axis. The extracted edge has no axis, so inference will still see
    # "no version-axis edge present" and add one. This documents that
    # behavior — the safety belt is the alias-existence check, not the
    # idempotency guard. For the realistic case (relation is a lineage
    # type), see test_enrich_draft_adds_family_version_edge_above.
    out = _hs.enrich_draft_from_row(row, aliases_to_canonical, {})
    parents = json.loads(out["parents"])
    version_edges = [
        p for p in parents
        if p.get("relationship") == "variant" and p.get("axis") == "version"
    ]
    assert len(version_edges) == 1


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
