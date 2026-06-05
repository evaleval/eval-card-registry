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
        "id", "display_name", "org_id", "parents", "model_group_id",
        "lineage_origin_model_org_id", "open_weights", "release_date",
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
        "lineage_origin_model_org_id": "meta",
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


def test_enriched_resolver_flips_quantized_chain_to_leaf_with_group_root():
    """Resolving a quantized leaf returns the LEAF as
    `canonical_id` (the precise quant artifact); the identity root moves to
    `model_group_id` (and `root_model_id` for compat). `resolved_leaf_id ==
    canonical_id`. Metadata now comes from the matched LEAF row — at seed
    `derive_model_lineage_fields` would inherit open_weights onto the leaf,
    but this unit fixture sets values only on the root and runs no derive
    pass, so the leaf's own (unset) metadata is what surfaces."""
    aliases = _alias_store(
        ("meta/llama-3.1-8b-instruct-turbo", "model", "meta/llama-3.1-8b-instruct-turbo", None, "confirmed"),
    )
    canonicals = CanonicalStore(models_df=_models_df(
        # Root: open-weight base instruct
        {
            "id": "meta/llama-3.1-8b-instruct",
            "org_id": "meta", "parents": "[]", "open_weights": True,
            "release_date": "2024-07-18", "params_billions": 8.0,
            "lineage_origin_model_org_id": "meta", "review_status": "reviewed",
        },
        # Leaf: quantized variant pointing at the root. Metadata is set on
        # the leaf row directly (as the seed-time derive pass would
        # materialise) so the flipped response is internally consistent.
        {
            "id": "meta/llama-3.1-8b-instruct-turbo",
            "org_id": "meta",
            "parents": json.dumps([{"id": "meta/llama-3.1-8b-instruct", "relationship": "quantized"}]),
            "model_group_id": "meta/llama-3.1-8b-instruct",
            "lineage_origin_model_org_id": "meta",
            "open_weights": True, "release_date": "2024-07-18",
            "params_billions": 8.0,
            "review_status": "reviewed",
        },
    ))
    resolver = Resolver(aliases, canonical_store=canonicals)
    r = resolver.resolve("meta/llama-3.1-8b-instruct-turbo", "model")
    # canonical_id is now the LEAF
    assert r.canonical_id == "meta/llama-3.1-8b-instruct-turbo"
    # resolved_leaf_id == canonical_id
    assert r.resolved_leaf_id == "meta/llama-3.1-8b-instruct-turbo"
    # the identity root moves to model_group_id (and root_model_id for compat)
    assert r.model_group_id == "meta/llama-3.1-8b-instruct"
    assert r.root_model_id == "meta/llama-3.1-8b-instruct"
    # Metadata sourced from the matched LEAF row
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


def test_flip_handles_dangling_group_root_fk():
    """If the matched canonical's `model_group_id` points at an id that
    isn't in the canonical store (e.g. broken FK), enrichment must not
    crash. The canonical_id is the matched LEAF (which always
    exists), so the response is well-formed regardless of the dangling
    group root; the dangling root is still surfaced in model_group_id, and
    metadata reads off the present leaf row."""
    aliases = _alias_store(("orphan/leaf", "model", "orphan/leaf", None, "confirmed"))
    canonicals = CanonicalStore(models_df=_models_df({
        "id": "orphan/leaf",
        "org_id": "orphan",
        "parents": json.dumps([{"id": "orphan/missing-root", "relationship": "quantized"}]),
        "model_group_id": "orphan/missing-root",  # FK to non-existent canonical
        "review_status": "draft",
    }))
    resolver = Resolver(aliases, canonical_store=canonicals)
    r = resolver.resolve("orphan/leaf", "model")
    # canonical_id is the LEAF (present); the dangling root only
    # appears in model_group_id. No crash.
    assert r.canonical_id == "orphan/leaf"
    assert r.resolved_leaf_id == "orphan/leaf"
    assert r.model_group_id == "orphan/missing-root"
    # Metadata reads off the leaf row, which set none → None.
    assert r.open_weights is None
    assert r.release_date is None


# ---------- ancestry + typed resolution_detail ----------

def _benchmarks_df(*rows) -> pd.DataFrame:
    cols = ["id", "display_name", "parent_benchmark_id", "review_status"]
    return pd.DataFrame([{c: r.get(c) for c in cols} for r in rows])


def _families_df(*rows) -> pd.DataFrame:
    cols = ["id", "display_name", "category", "benchmark_ids",
            "composite_keys", "review_status"]
    return pd.DataFrame([{c: r.get(c) for c in cols} for r in rows])


def _composites_df(*rows) -> pd.DataFrame:
    cols = ["id", "display_name", "source_configs", "family_id", "review_status"]
    return pd.DataFrame([{c: r.get(c) for c in cols} for r in rows])


def test_model_ancestry_and_detail():
    aliases = _alias_store(("acme/w-7b-it", "model", "acme/w-7b-it", None, "confirmed"))
    df = _models_df({"id": "acme/w-7b-it", "org_id": "acme", "parents": "[]"})
    df["model_family_id"] = ["acme/w"]
    df["model_group_id"] = ["acme/w-7b"]
    df["resolution_granularity"] = ["variant"]
    cs = CanonicalStore(models_df=df)
    r = Resolver(aliases, canonical_store=cs).resolve("acme/w-7b-it", "model")
    assert r.ancestry == [
        {"canonical_id": "acme/w-7b", "level": "group"},
        {"canonical_id": "acme/w", "level": "family"},
    ]
    # Off-HF model (no resolution_source / metadata): hf_repo_id is None.
    assert r.resolution_detail == {"granularity": "variant", "hf_repo_id": None}


def test_model_detail_hf_repo_id():
    """resolution_detail.hf_repo_id surfaces the real HF repo id for an
    HF-backed canonical (resolution_source==hf OR metadata.hf_id==id), and
    None for an off-HF slug whose metadata.hf_id points elsewhere (shadow)."""
    aliases = _alias_store(
        ("meta/llama", "model", "meta/llama", None, "confirmed"),
        ("acme/mistral", "model", "acme/mistral", None, "confirmed"),
        ("kimi/slug", "model", "kimi/slug", None, "confirmed"),
        ("junk/meta", "model", "junk/meta", None, "confirmed"),
    )
    df = _models_df(
        {"id": "meta/llama", "org_id": "meta", "parents": "[]"},
        {"id": "acme/mistral", "org_id": "acme", "parents": "[]",
         "metadata": '{"hf_id": "acme/mistral"}'},
        {"id": "kimi/slug", "org_id": "moonshotai", "parents": "[]",
         "metadata": '{"hf_id": "moonshotai/Real-Kimi"}'},
        # metadata that parses to a NON-dict JSON value must not crash the resolve.
        {"id": "junk/meta", "org_id": "junk", "parents": "[]",
         "metadata": "[1, 2, 3]"},
    )
    df["resolution_source"] = ["hf", "models_dev", "inferred", "inferred"]
    cs = CanonicalStore(models_df=df)
    res = lambda raw: Resolver(aliases, canonical_store=cs).resolve(raw, "model")
    assert res("meta/llama").resolution_detail["hf_repo_id"] == "meta/llama"       # source==hf
    assert res("acme/mistral").resolution_detail["hf_repo_id"] == "acme/mistral"   # metadata.hf_id==id
    assert res("kimi/slug").resolution_detail["hf_repo_id"] is None                # shadow: hf_id != id
    assert res("junk/meta").resolution_detail["hf_repo_id"] is None                # non-dict metadata: no crash


def test_benchmark_full_chain_ancestry():
    aliases = _alias_store(("bench-pro", "benchmark", "bench-pro", None, "confirmed"))
    cs = CanonicalStore(
        benchmarks_df=_benchmarks_df({"id": "bench-pro", "display_name": "BP"}),
        families_df=_families_df({
            "id": "fam-x", "benchmark_ids": json.dumps(["bench-pro"]),
            "composite_keys": json.dumps(["comp-suite"]), "category": "reasoning"}),
        composites_df=_composites_df({
            "id": "comp-suite", "family_id": "fam-x",
            "source_configs": json.dumps(["cfg-a"])}),
    )
    r = Resolver(aliases, canonical_store=cs).resolve("bench-pro", "benchmark")
    assert r.ancestry == [
        {"canonical_id": "fam-x", "level": "family"},
        {"canonical_id": "comp-suite", "level": "composite"},
    ]
    assert r.resolution_detail == {"level": "benchmark", "matched_subset": None}


def test_benchmark_slice_detail():
    aliases = _alias_store(("bench-sub", "benchmark", "bench-sub", None, "confirmed"))
    cs = CanonicalStore(benchmarks_df=_benchmarks_df(
        {"id": "bench-pro", "display_name": "BP"},
        {"id": "bench-sub", "display_name": "BS", "parent_benchmark_id": "bench-pro"},
    ))
    r = Resolver(aliases, canonical_store=cs).resolve("bench-sub", "benchmark")
    assert r.resolution_detail["level"] == "slice"
    # bench-sub's family defaults to its parent walk terminus (bench-pro).
    assert r.ancestry == [{"canonical_id": "bench-pro", "level": "family"}]


def test_benchmark_subset_fold_matched_subset():
    aliases = _alias_store(("Anatomy", "benchmark", "mmlu", None, "confirmed"))
    cs = CanonicalStore(benchmarks_df=_benchmarks_df(
        {"id": "mmlu", "display_name": "MMLU"}))
    r = Resolver(aliases, canonical_store=cs).resolve("Anatomy", "benchmark")
    assert r.canonical_id == "mmlu"
    assert r.resolution_detail == {"level": "benchmark", "matched_subset": "Anatomy"}


def test_family_resolves_with_composite_ancestry():
    aliases = _alias_store(("fam-x", "family", "fam-x", None, "confirmed"))
    cs = CanonicalStore(
        families_df=_families_df({
            "id": "fam-x", "composite_keys": json.dumps(["comp-suite"])}),
        composites_df=_composites_df({"id": "comp-suite", "family_id": "fam-x"}),
    )
    r = Resolver(aliases, canonical_store=cs).resolve("fam-x", "family")
    assert r.canonical_id == "fam-x"
    assert r.ancestry == [{"canonical_id": "comp-suite", "level": "composite"}]
    assert r.resolution_detail == {}


def test_composite_resolves_as_root():
    aliases = _alias_store(("comp-suite", "composite", "comp-suite", None, "confirmed"))
    cs = CanonicalStore(composites_df=_composites_df(
        {"id": "comp-suite", "family_id": "fam-x"}))
    r = Resolver(aliases, canonical_store=cs).resolve("comp-suite", "composite")
    assert r.canonical_id == "comp-suite"
    assert r.ancestry == []
    assert r.resolution_detail == {}
