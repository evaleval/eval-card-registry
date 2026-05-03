"""Tests for CanonicalStore + Resolver enrichment.

Locks in the standalone-vs-API parity: when a resolver is given a
CanonicalStore, its `ResolutionResult` carries the same fields the HTTP
API returns (root-collapse for quantized chains, parent_canonical_id,
parents list, lineage_origin_org_id, open_weights, release_date,
params_billions, etc.)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
import uuid

import pandas as pd
import pytest

from eval_entity_resolver import AliasStore, CanonicalStore, Resolver, ResolverConfig
from eval_entity_resolver.alias_store import _empty_df as _empty_alias_df


def _alias_store(*rows) -> AliasStore:
    now = datetime.now(timezone.utc).isoformat()
    records = []
    for raw_value, entity_type, canonical_id, source_config, status in rows:
        records.append({
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
        })
    df = pd.DataFrame(records) if records else _empty_alias_df()
    return AliasStore(df)


def _models_df(*rows) -> pd.DataFrame:
    """Build a canonical_models DataFrame with the columns the resolver
    reads. Each row is a dict — caller passes only the fields they care
    about; missing fields are None / NA."""
    cols = [
        "id", "display_name", "org_id", "parents", "root_model_id",
        "lineage_origin_org_id", "open_weights", "release_date",
        "params_billions", "tags", "metadata", "review_status",
    ]
    out = []
    for r in rows:
        row = {c: r.get(c) for c in cols}
        out.append(row)
    return pd.DataFrame(out)


# ---------- bare resolver (no canonical_store) — fields stay None ----------

def test_bare_resolver_returns_only_basic_fields():
    """Without a CanonicalStore, the resolver returns just the matching
    fields; rich-shape fields are all None. Backwards-compatible with
    the original API."""
    store = _alias_store(("meta/llama-3-8b", "model", "meta/llama-3-8b", None, "confirmed"))
    resolver = Resolver(store)  # no canonical_store
    r = resolver.resolve("meta/llama-3-8b", "model")
    assert r.canonical_id == "meta/llama-3-8b"
    assert r.strategy == "exact"
    assert r.review_status is None
    assert r.parents is None
    assert r.open_weights is None
    assert r.release_date is None


# ---------- enriched resolver matches API shape ----------

def test_enriched_resolver_populates_model_metadata():
    """A model resolution with canonical_store attached carries the
    matched canonical's metadata: parents, lineage_origin_org_id,
    open_weights, release_date, params_billions, review_status."""
    aliases = _alias_store(("Llama-3.1-8B", "model", "meta/llama-3.1-8b", None, "confirmed"))
    canonicals = CanonicalStore(models_df=_models_df({
        "id": "meta/llama-3.1-8b",
        "display_name": "Llama 3.1 8B",
        "org_id": "meta",
        "parents": json.dumps([{"id": "meta/llama-3.1", "relationship": "variant", "axis": "size"}]),
        "lineage_origin_org_id": "meta",
        "open_weights": True,
        "release_date": "2024-07-18",
        "params_billions": 8.0,
        "review_status": "reviewed",
    }))
    resolver = Resolver(aliases, canonical_store=canonicals)
    r = resolver.resolve("Llama-3.1-8B", "model")
    assert r.canonical_id == "meta/llama-3.1-8b"
    assert r.review_status == "reviewed"
    assert r.parent_canonical_id == "meta/llama-3.1"  # variant edge
    assert r.parents == [{"id": "meta/llama-3.1", "relationship": "variant", "axis": "size"}]
    assert r.lineage_origin_org_id == "meta"
    assert r.open_weights is True
    assert r.release_date == "2024-07-18"
    assert r.params_billions == 8.0


def test_enriched_resolver_collapses_quantized_chain_to_root():
    """Resolving a quantized leaf returns the identity root as
    `canonical_id`; the original leaf goes in `resolved_leaf_id`. All
    metadata comes from the root (quants preserve identity)."""
    aliases = _alias_store(
        ("meta/llama-3.1-8b-instruct-turbo", "model", "meta/llama-3.1-8b-instruct-turbo", None, "confirmed"),
    )
    canonicals = CanonicalStore(models_df=_models_df(
        # Root: open-weight base instruct
        {
            "id": "meta/llama-3.1-8b-instruct",
            "org_id": "meta", "parents": "[]", "open_weights": True,
            "release_date": "2024-07-18", "params_billions": 8.0,
            "lineage_origin_org_id": "meta", "review_status": "reviewed",
        },
        # Leaf: quantized variant pointing at the root
        {
            "id": "meta/llama-3.1-8b-instruct-turbo",
            "org_id": "meta",
            "parents": json.dumps([{"id": "meta/llama-3.1-8b-instruct", "relationship": "quantized"}]),
            "root_model_id": "meta/llama-3.1-8b-instruct",
            "lineage_origin_org_id": "meta",
            "review_status": "reviewed",
        },
    ))
    resolver = Resolver(aliases, canonical_store=canonicals)
    r = resolver.resolve("meta/llama-3.1-8b-instruct-turbo", "model")
    # canonical_id collapses to the root
    assert r.canonical_id == "meta/llama-3.1-8b-instruct"
    # leaf id preserved
    assert r.resolved_leaf_id == "meta/llama-3.1-8b-instruct-turbo"
    assert r.root_model_id == "meta/llama-3.1-8b-instruct"
    # Metadata sourced from the root (so the response is internally consistent)
    assert r.open_weights is True
    assert r.release_date == "2024-07-18"
    assert r.params_billions == 8.0
    # Parents reflect the matched leaf's edges
    assert r.parents == [{"id": "meta/llama-3.1-8b-instruct", "relationship": "quantized"}]


def test_enriched_resolver_benchmark_surfaces_parent():
    """For benchmarks the parent_canonical_id comes from the
    `parent_benchmark_id` scalar column. Other rich fields are
    correctly None for non-models."""
    aliases = _alias_store(("MATH Level 5", "benchmark", "math-level-5", None, "confirmed"))
    bm_df = pd.DataFrame([{
        "id": "math-level-5", "display_name": "MATH Level 5",
        "parent_benchmark_id": "math", "review_status": "reviewed",
    }])
    canonicals = CanonicalStore(benchmarks_df=bm_df)
    resolver = Resolver(aliases, canonical_store=canonicals)
    r = resolver.resolve("MATH Level 5", "benchmark")
    assert r.canonical_id == "math-level-5"
    assert r.parent_canonical_id == "math"
    assert r.review_status == "reviewed"
    # Model-only fields stay None
    assert r.resolved_leaf_id is None
    assert r.parents is None
    assert r.open_weights is None


def test_no_match_carries_no_enrichment():
    """A no_match result returns None for canonical_id and all
    enrichment fields stay None — there's nothing to enrich."""
    canonicals = CanonicalStore(models_df=_models_df())
    resolver = Resolver(_alias_store(), canonical_store=canonicals)
    r = resolver.resolve("totally-unknown-model", "model")
    assert r.canonical_id is None
    assert r.strategy == "no_match"
    assert r.parents is None
    assert r.open_weights is None
    assert r.review_status is None


# ---------- CanonicalStore directly ----------

def test_canonical_store_lookup_returns_dict_with_na_coerced():
    """CanonicalStore.lookup returns a row dict with NaN/NA coerced to
    None so callers don't have to handle pd.NA at every access."""
    df = pd.DataFrame([
        {"id": "x/y", "open_weights": True, "release_date": None, "params_billions": float("nan")},
    ])
    cs = CanonicalStore(models_df=df)
    row = cs.lookup("model", "x/y")
    assert row is not None
    assert row["open_weights"] is True
    assert row["release_date"] is None
    assert row["params_billions"] is None  # NaN coerced


def test_canonical_store_lookup_missing_id_returns_none():
    cs = CanonicalStore(models_df=_models_df())
    assert cs.lookup("model", "does-not-exist") is None
    assert cs.lookup("benchmark", "does-not-exist") is None


# ---------- backwards compat ----------

def test_resolver_positional_two_arg_constructor():
    """The original `Resolver(store, config)` two-arg form must keep
    working — downstream pipelines (AutoBenchmarkCard etc.) construct
    this way. Adding `canonical_store` as a third optional kwarg must
    not break positional callers."""
    store = _alias_store(("MATH", "benchmark", "math", None, "confirmed"))
    resolver = Resolver(store, ResolverConfig(threshold=0.85))
    r = resolver.resolve("MATH", "benchmark")
    # Bare matcher behavior — basic fields populated, enrichment None
    assert r.canonical_id == "math"
    assert r.strategy == "exact"
    assert r.confidence == 1.0
    assert r.review_status is None  # no canonical_store → no enrichment
    assert r.parents is None


def test_root_collapse_falls_back_when_root_missing_from_store():
    """If the matched canonical's `root_model_id` points at an id that
    isn't in the canonical store (e.g. broken FK), enrichment must not
    crash — partial data is fine, but the response should still be
    well-formed."""
    aliases = _alias_store(("orphan/leaf", "model", "orphan/leaf", None, "confirmed"))
    canonicals = CanonicalStore(models_df=_models_df({
        "id": "orphan/leaf",
        "org_id": "orphan",
        "parents": json.dumps([{"id": "orphan/missing-root", "relationship": "quantized"}]),
        "root_model_id": "orphan/missing-root",  # FK to non-existent canonical
        "review_status": "draft",
    }))
    resolver = Resolver(aliases, canonical_store=canonicals)
    r = resolver.resolve("orphan/leaf", "model")
    # Doesn't crash; canonical_id collapses to the dangling root id
    assert r.canonical_id == "orphan/missing-root"
    assert r.resolved_leaf_id == "orphan/leaf"
    # Metadata fields fall back to None since root entity wasn't found
    assert r.open_weights is None
    assert r.release_date is None
