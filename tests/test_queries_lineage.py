"""Tests for `queries.derive_model_lineage_fields` — the post-seed
denormalization that walks the `parents` graph to populate
`model_group_id` and `lineage_origin_model_org_id`."""
from __future__ import annotations

import json

import pandas as pd
import pytest

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


def _add_model(store, cid, org_id, parents, release_date=None):
    queries.upsert_entity(store, "canonical_models", {
        "id": cid, "display_name": cid, "developer": None,
        "org_id": org_id, "family": None, "architecture": None,
        "params_billions": None,
        "parents": json.dumps(parents) if parents else "[]",
        "model_group_id": None, "lineage_origin_model_org_id": None,
        "release_date": release_date,
        "tags": "[]", "metadata": "{}", "review_status": "reviewed",
    })


def test_derive_handles_cycle_without_infinite_loop(fresh_store):
    """The `_walk` helper tracks visited ids; a cycle in `parents`
    (data corruption) must terminate cleanly. Pathological case: A
    points to B, B points to A — both via `quantized` (so the walk
    would otherwise loop forever)."""
    _add_model(fresh_store, "lab/a", "lab", [{"id": "lab/b", "relationship": "quantized"}])
    _add_model(fresh_store, "lab/b", "lab", [{"id": "lab/a", "relationship": "quantized"}])

    # Should return cleanly, not hang
    counts = queries.derive_model_lineage_fields(fresh_store)
    assert counts["group_set"] >= 0  # just verify it returned

    df = fresh_store.table("canonical_models")
    # Both entries got SOME model_group_id assigned (not relevant which —
    # the important thing is the walk terminated).
    assert len(df) == 2


def test_derive_lineage_origin_walks_finetune_and_quantized(fresh_store):
    """`lineage_origin_model_org_id` walks through any non-variant relationship
    (quantized/finetune/merge/adapter) to the deepest ancestor and copies
    its org_id. Variant edges DO NOT count for this walk — they're
    within-family hierarchy, not lineage."""
    # Meta's base
    _add_model(fresh_store, "meta/llama-3.1-70b", "meta", [])
    # Nous's finetune of Meta's base
    _add_model(fresh_store, "nous/hermes-3-llama-70b", "nous-research", [
        {"id": "meta/llama-3.1-70b", "relationship": "finetune"},
    ])
    # Quantized of the Nous finetune
    _add_model(fresh_store, "nous/hermes-3-llama-70b-fp8", "nous-research", [
        {"id": "nous/hermes-3-llama-70b", "relationship": "quantized"},
    ])

    queries.derive_model_lineage_fields(fresh_store)
    df = fresh_store.table("canonical_models")
    by_id = {r["id"]: r for _, r in df.iterrows()}

    # Meta original: lineage = self.org_id; no quantized ancestor → group is
    # SELF (group membership is total; a singleton is a group of one).
    assert by_id["meta/llama-3.1-70b"]["lineage_origin_model_org_id"] == "meta"
    assert by_id["meta/llama-3.1-70b"]["model_group_id"] == "meta/llama-3.1-70b"
    # family membership is also total → SELF at the family root.
    assert by_id["meta/llama-3.1-70b"]["model_family_id"] == "meta/llama-3.1-70b"
    # lineage_origin_model_id is NULL at origin (NO self-fallback — unlike the
    # org_id variant which DID self-fall-back to "meta" above).
    assert pd.isna(by_id["meta/llama-3.1-70b"]["lineage_origin_model_id"])

    # Nous finetune: lineage = upstream lab (Meta), via the finetune edge.
    # No root collapse — finetune isn't identity-preserving → group is SELF.
    assert by_id["nous/hermes-3-llama-70b"]["lineage_origin_model_org_id"] == "meta"
    assert by_id["nous/hermes-3-llama-70b"]["model_group_id"] == "nous/hermes-3-llama-70b"
    assert by_id["nous/hermes-3-llama-70b"]["model_family_id"] == "nous/hermes-3-llama-70b"
    # The id of that upstream ancestor is the Meta base (NOT self, NOT null).
    assert by_id["nous/hermes-3-llama-70b"]["lineage_origin_model_id"] == "meta/llama-3.1-70b"

    # Quantized of finetune: lineage walks BOTH edges to Meta
    assert by_id["nous/hermes-3-llama-70b-fp8"]["lineage_origin_model_org_id"] == "meta"
    # Root collapses to the unquantized Hermes (NOT all the way to Llama —
    # the chain only follows `quantized`, stops at the `finetune` edge).
    assert by_id["nous/hermes-3-llama-70b-fp8"]["model_group_id"] == "nous/hermes-3-llama-70b"
    # lineage_origin_model_id walks BOTH the quantized + finetune edges to the
    # deepest non-variant ancestor (Meta's base) — same id as its parent's.
    assert by_id["nous/hermes-3-llama-70b-fp8"]["lineage_origin_model_id"] == "meta/llama-3.1-70b"


def test_derive_lineage_origin_falls_back_to_self_org(fresh_store):
    """When a model has no walkable non-variant edge (a true root), its
    `lineage_origin_model_org_id` is its own `org_id`."""
    _add_model(fresh_store, "meta/llama-3", "meta", [])
    queries.derive_model_lineage_fields(fresh_store)
    df = fresh_store.table("canonical_models")
    row = df[df["id"] == "meta/llama-3"].iloc[0]
    assert row["lineage_origin_model_org_id"] == "meta"


def test_open_weights_inherits_from_parent_via_variant_edges(fresh_store):
    """A variant/mode of an open-weight base inherits open_weights=True
    when the variant doesn't have its own value set. Same identity, just
    different post-training."""
    _add_model(fresh_store, "meta/llama-3-8b", "meta", [])
    _add_model(fresh_store, "meta/llama-3-8b-instruct", "meta", [
        {"id": "meta/llama-3-8b", "relationship": "variant", "axis": "mode"},
    ])
    # Set parent explicitly open
    df = fresh_store.table("canonical_models")
    df.loc[df["id"] == "meta/llama-3-8b", "open_weights"] = True
    fresh_store.set_table("canonical_models", df)

    queries.derive_model_lineage_fields(fresh_store)
    df = fresh_store.table("canonical_models")
    # pandas BooleanDtype returns numpy booleans, so use `==` not `is`.
    assert df[df["id"] == "meta/llama-3-8b"].iloc[0]["open_weights"] == True
    assert df[df["id"] == "meta/llama-3-8b-instruct"].iloc[0]["open_weights"] == True


def test_open_weights_inherits_through_quantized_chain(fresh_store):
    """Quantized of open base → open. Inheritance walks both variant
    and quantized edges (identity-preserving)."""
    _add_model(fresh_store, "meta/llama-3-8b", "meta", [])
    _add_model(fresh_store, "meta/llama-3-8b-instruct", "meta", [
        {"id": "meta/llama-3-8b", "relationship": "variant", "axis": "mode"},
    ])
    _add_model(fresh_store, "meta/llama-3-8b-instruct-fp8", "meta", [
        {"id": "meta/llama-3-8b-instruct", "relationship": "quantized"},
    ])
    df = fresh_store.table("canonical_models")
    df.loc[df["id"] == "meta/llama-3-8b", "open_weights"] = True
    fresh_store.set_table("canonical_models", df)

    queries.derive_model_lineage_fields(fresh_store)
    df = fresh_store.table("canonical_models")
    assert df[df["id"] == "meta/llama-3-8b-instruct-fp8"].iloc[0]["open_weights"] == True


def test_open_weights_does_not_inherit_through_finetune(fresh_store):
    """Finetune of an open base does NOT auto-inherit open_weights — a
    finetune is its own release whose openness depends on whether the
    finetuner published the weights, not on the base."""
    _add_model(fresh_store, "meta/llama-3.1-70b", "meta", [])
    _add_model(fresh_store, "nous/hermes-3-llama-70b", "nous-research", [
        {"id": "meta/llama-3.1-70b", "relationship": "finetune"},
    ])
    df = fresh_store.table("canonical_models")
    df.loc[df["id"] == "meta/llama-3.1-70b", "open_weights"] = True
    fresh_store.set_table("canonical_models", df)

    queries.derive_model_lineage_fields(fresh_store)
    df = fresh_store.table("canonical_models")
    val = df[df["id"] == "nous/hermes-3-llama-70b"].iloc[0]["open_weights"]
    assert val is None or queries._is_na(val), \
        "finetune must not auto-inherit open_weights from its base"


def test_open_weights_explicit_value_never_overwritten(fresh_store):
    """If a child has an explicit open_weights value, the inheritance
    walk MUST NOT overwrite it — even if the parent says otherwise.
    Curated values take precedence over derived inference."""
    _add_model(fresh_store, "lab/parent-open", "lab", [])
    _add_model(fresh_store, "lab/child-explicitly-closed", "lab", [
        {"id": "lab/parent-open", "relationship": "variant", "axis": "mode"},
    ])
    df = fresh_store.table("canonical_models")
    df.loc[df["id"] == "lab/parent-open", "open_weights"] = True
    df.loc[df["id"] == "lab/child-explicitly-closed", "open_weights"] = False
    fresh_store.set_table("canonical_models", df)

    queries.derive_model_lineage_fields(fresh_store)
    df = fresh_store.table("canonical_models")
    # Explicit False on child must survive — even though the inheritance
    # walk would have inherited True from the open parent.
    assert df[df["id"] == "lab/child-explicitly-closed"].iloc[0]["open_weights"] == False


def test_root_collapses_through_variant_version_edge(fresh_store):
    """`variant axis=version` edges represent dated snapshots of the same
    release (e.g. `gpt-4o-2024-05-13` -> `gpt-4o`, `grok-4-0709` -> `grok-4`).
    These collapse to root just like quantized edges — same model identity
    at the API."""
    _add_model(fresh_store, "xai/grok-4", "xai", [])
    _add_model(fresh_store, "xai/grok-4-0709", "xai", [
        {"id": "xai/grok-4", "relationship": "variant", "axis": "version"},
    ])
    # Quantized of the dated snapshot — full chain: quant -> snapshot -> base
    _add_model(fresh_store, "xai/grok-4-0709-fp8", "xai", [
        {"id": "xai/grok-4-0709", "relationship": "quantized"},
    ])
    queries.derive_model_lineage_fields(fresh_store)
    df = fresh_store.table("canonical_models")
    by_id = {r["id"]: r for _, r in df.iterrows()}

    # Base: no chain → group is SELF (singleton group)
    assert by_id["xai/grok-4"]["model_group_id"] == "xai/grok-4"
    # Snapshot: collapses to base via variant-version
    assert by_id["xai/grok-4-0709"]["model_group_id"] == "xai/grok-4"
    # Quant of snapshot: walks both edges all the way to base
    assert by_id["xai/grok-4-0709-fp8"]["model_group_id"] == "xai/grok-4"


def test_root_does_not_collapse_through_non_version_variant(fresh_store):
    """Other variant axes (mode/size/modality/domain) are NOT identity-
    preserving for model_group_id — `gpt-4o-mini` is a separate model
    from `gpt-4o`, with different scores."""
    _add_model(fresh_store, "openai/gpt-4o", "openai", [])
    _add_model(fresh_store, "openai/gpt-4o-mini", "openai", [
        {"id": "openai/gpt-4o", "relationship": "variant", "axis": "size"},
    ])
    queries.derive_model_lineage_fields(fresh_store)
    df = fresh_store.table("canonical_models")
    by_id = {r["id"]: r for _, r in df.iterrows()}
    # Size variant: stays at the leaf, no root collapse → group is SELF
    assert by_id["openai/gpt-4o-mini"]["model_group_id"] == "openai/gpt-4o-mini"


def test_derive_variant_edges_do_not_set_lineage_origin_to_parent(fresh_store):
    """Variant edges are within-family hierarchy and DO NOT count toward
    lineage origin. A size variant of a Meta family stays attributed to
    Meta (its own org), not walked to a different parent."""
    _add_model(fresh_store, "meta/llama-3", "meta", [])
    _add_model(fresh_store, "meta/llama-3-8b", "meta", [
        {"id": "meta/llama-3", "relationship": "variant", "axis": "size"},
    ])
    queries.derive_model_lineage_fields(fresh_store)
    df = fresh_store.table("canonical_models")
    by_id = {r["id"]: r for _, r in df.iterrows()}
    # Same org throughout — variant edge doesn't change this, but the
    # walk must not loop or chain across the variant edge to a sibling.
    assert by_id["meta/llama-3-8b"]["lineage_origin_model_org_id"] == "meta"
    # No quantized chain → model_group_id is SELF (self is the identity root,
    # which is a singleton group of one).
    assert by_id["meta/llama-3-8b"]["model_group_id"] == "meta/llama-3-8b"
    # But model_family_id FOLDS the size variant up to the family root
    # (`llama-3`): group keeps size-variant identity, family does not. These two
    # diverging is the whole point of the group-vs-family split.
    assert by_id["meta/llama-3-8b"]["model_family_id"] == "meta/llama-3"
    # Variant edges do NOT count toward lineage origin → null at origin (NOT
    # self, NOT the variant parent).
    assert pd.isna(by_id["meta/llama-3-8b"]["lineage_origin_model_id"])


# ---------------------------------------------------------------------------
# release_date derivation — fills in NULL release_dates by parsing the id's
# date suffix. Explicit values always win.
# ---------------------------------------------------------------------------


def test_release_date_derived_from_iso_full_suffix(fresh_store):
    """A canonical with id `<prefix>-YYYY-MM-DD` and NULL release_date
    inherits the parsed date. Mirrors the openai daily-snapshot pattern."""
    _add_model(fresh_store, "openai/gpt-4.1-mini-2025-04-14", "openai", [], release_date=None)
    queries.derive_model_lineage_fields(fresh_store)
    df = fresh_store.table("canonical_models")
    row = df[df["id"] == "openai/gpt-4.1-mini-2025-04-14"].iloc[0]
    assert row["release_date"] == "2025-04-14"


def test_release_date_derived_from_packed_suffix(fresh_store):
    """`<prefix>-YYYYMMDD` (8-digit Anthropic-style) parses correctly."""
    _add_model(fresh_store, "anthropic/claude-haiku-4.5-20251001", "anthropic", [], release_date=None)
    queries.derive_model_lineage_fields(fresh_store)
    df = fresh_store.table("canonical_models")
    row = df[df["id"] == "anthropic/claude-haiku-4.5-20251001"].iloc[0]
    assert row["release_date"] == "2025-10-01"


def test_release_date_derived_from_month_only_suffix(fresh_store):
    """`<prefix>-YYYY-MM` (truncated month) parses with day defaulted to 01."""
    _add_model(fresh_store, "openai/gpt-5-2025-08", "openai", [], release_date=None)
    queries.derive_model_lineage_fields(fresh_store)
    df = fresh_store.table("canonical_models")
    row = df[df["id"] == "openai/gpt-5-2025-08"].iloc[0]
    assert row["release_date"] == "2025-08-01"


def test_explicit_release_date_wins_over_derivation(fresh_store):
    """When the YAML hand-curated `release_date` differs from the date
    parseable off the id, the explicit value MUST stick. Anthropic
    sometimes ships a `-YYYYMMDD` snapshot a few days after the actual
    public release; the curated date is more accurate."""
    _add_model(
        fresh_store, "anthropic/claude-sonnet-4-20250514", "anthropic", [],
        release_date="2025-05-22",  # actual public release date
    )
    queries.derive_model_lineage_fields(fresh_store)
    df = fresh_store.table("canonical_models")
    row = df[df["id"] == "anthropic/claude-sonnet-4-20250514"].iloc[0]
    assert row["release_date"] == "2025-05-22"  # NOT 2025-05-14


def test_release_date_year_range_guard(fresh_store):
    """Non-year 4-digit tails (e.g. parameter counts, batch numbers) must
    NOT be treated as release dates. `model-1024` could be misread as a
    1024 release; the guard rejects it."""
    _add_model(fresh_store, "lab/model-1024", "lab", [], release_date=None)
    _add_model(fresh_store, "lab/model-9999-12-31", "lab", [], release_date=None)
    queries.derive_model_lineage_fields(fresh_store)
    df = fresh_store.table("canonical_models")
    by_id = {r["id"]: r for _, r in df.iterrows()}
    # 1024 is not in 2015-2035 range → no derivation
    assert queries._is_na(by_id["lab/model-1024"]["release_date"])
    # 9999 is not in range either → no derivation
    assert queries._is_na(by_id["lab/model-9999-12-31"]["release_date"])


def test_release_date_no_derivation_when_no_date_suffix(fresh_store):
    """Canonicals whose id doesn't end in a parseable date suffix stay
    NULL — no manufactured value."""
    _add_model(fresh_store, "openai/gpt-5", "openai", [], release_date=None)
    _add_model(fresh_store, "alibaba/qwen2-72b", "alibaba", [], release_date=None)
    queries.derive_model_lineage_fields(fresh_store)
    df = fresh_store.table("canonical_models")
    by_id = {r["id"]: r for _, r in df.iterrows()}
    assert queries._is_na(by_id["openai/gpt-5"]["release_date"])
    assert queries._is_na(by_id["alibaba/qwen2-72b"]["release_date"])


def test_release_date_invalid_month_or_day_rejected(fresh_store):
    """Out-of-range month/day in the id (data corruption) → no derivation
    rather than invalid date."""
    _add_model(fresh_store, "lab/model-2025-13-01", "lab", [], release_date=None)  # month 13
    _add_model(fresh_store, "lab/model-2025-02-32", "lab", [], release_date=None)  # day 32
    queries.derive_model_lineage_fields(fresh_store)
    df = fresh_store.table("canonical_models")
    by_id = {r["id"]: r for _, r in df.iterrows()}
    assert queries._is_na(by_id["lab/model-2025-13-01"]["release_date"])
    assert queries._is_na(by_id["lab/model-2025-02-32"]["release_date"])
