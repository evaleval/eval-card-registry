"""Offline reproducibility of the hub-stats lineage/enrichment layer.

The candidate subset of cfahlgren1/hub-stats — including the `baseModels` lineage
column — is frozen into a committed parquet (curation/hub_stats_frozen.parquet) so
a clean regen reproduces `parents` / `model_group_id` / `model_family_id` /
`lineage_origin` WITHOUT flaky live HF.

These tests pin: (1) the durable cache is committed and carries lineage; (2) the
generator defaults to reading it offline; (3) the enrichment derivation
reproduces a baseModels parent edge from a frozen row. All OFFLINE, no network.
They deliberately do NOT hardcode model ids so they survive a cache refresh or a
change to the curated model set.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from conftest import load_script_module

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE = REPO_ROOT / "curation" / "hub_stats_frozen.parquet"


@pytest.fixture(scope="module")
def mod():
    return load_script_module("refresh_from_hub_stats")


def test_frozen_cache_is_committed_and_nonempty():
    assert CACHE.exists(), f"durable offline cache missing at {CACHE}"
    con = duckdb.connect()
    n = con.execute(f"SELECT count(*) FROM read_parquet('{CACHE}')").fetchone()[0]
    # The cache covers our HF-aliased canonical universe (a few thousand rows).
    assert n >= 3000, f"cache unexpectedly small ({n} rows) — re-freeze it"


def test_cache_carries_basemodels_lineage():
    """The lineage SOURCE (baseModels) must be frozen, not just scalar
    enrichment — this is the whole point of the cache (parents reproduce
    offline)."""
    con = duckdb.connect()
    with_bm = con.execute(
        f"SELECT count(*) FROM read_parquet('{CACHE}') "
        f"WHERE baseModels.models IS NOT NULL AND len(baseModels.models) > 0"
    ).fetchone()[0]
    assert with_bm >= 1000, (
        f"only {with_bm} rows carry baseModels lineage — the cache is missing the "
        f"lineage column needed for reproducible parents"
    )


def test_generator_defaults_to_the_offline_cache(mod):
    assert mod.FROZEN_CACHE_PATH == CACHE
    assert mod.FROZEN_CACHE_PATH.exists()


def test_enrichment_reproduces_a_basemodels_parent_offline(mod):
    """Drive the derivation (enrich_draft_from_row) on a REAL frozen row and
    assert it reproduces a baseModels parent edge offline. The row + its base are
    read from the cache (no hardcoded ids), and a minimal alias map resolves the
    base to a canonical — so the test is independent of core.yaml's contents."""
    import json

    from eval_card_registry.services.hub_stats import (
        QUERY_COLUMNS,
        enrich_draft_from_row,
        normalize as _nz,
    )

    con = duckdb.connect()
    # A finetune row with exactly one base, so the expected edge is unambiguous.
    # Read it the SAME way query_hub_stats does (fetchall + dict(zip)) so values
    # are plain Python lists/dicts, not numpy arrays (which break tag filtering).
    cur = con.execute(
        f"SELECT {QUERY_COLUMNS} FROM read_parquet('{CACHE}') "
        f"WHERE baseModels.relation = 'finetune' AND len(baseModels.models) = 1 "
        f"ORDER BY id LIMIT 1"
    )
    cols = [d[0] for d in cur.description]
    row = dict(zip(cols, cur.fetchall()[0]))

    base_id = row["baseModels"]["models"][0]["id"]
    child_id = row["id"]
    # Minimal alias maps: the base resolves to itself as a canonical; no orgs.
    aliases_to_canonical = {_nz(base_id): base_id, _nz(child_id): child_id}
    enrichment = enrich_draft_from_row(
        row, aliases_to_canonical, {}, target_canonical=child_id
    )
    assert "parents" in enrichment, f"no parents derived for {child_id}"
    parents = json.loads(enrichment["parents"])
    assert any(p.get("id") == base_id for p in parents), (
        f"expected a parent edge to {base_id} for {child_id}, got {parents}"
    )
