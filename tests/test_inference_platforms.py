"""Invariant tests for the inference_platforms entity.

Asserts the seed YAML, the seeded table, and the single-sourced host-token map
are mutually consistent. Runs the real `seed --local` against the repo's
seed/ dir into an isolated fixtures dir (LOCAL_MODE).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from eval_card_registry.cli import app
from eval_card_registry.lib.inference_platforms_map import get_host_token_platform
from eval_card_registry.store import hf_store

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED_DIR = REPO_ROOT / "seed"
PROPOSAL_JSON = (
    REPO_ROOT
    / "specs"
    / "model-resolution-rework"
    / "inference_platforms.proposed.json"
)

VALID_KINDS = {
    "author_lab",
    "inference_platform",
    "aggregator_gateway",
    "regional_variant",
    "coding_plan",
}


def _strip_marker(form: str) -> str:
    for marker in ("prefix:", "suffix:"):
        if form.startswith(marker):
            return form[len(marker):]
    return form


@pytest.fixture
def seeded_store(tmp_path, monkeypatch):
    """Run `seed --local` against the repo seed/ dir into an isolated
    fixtures dir, returning the loaded store."""
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()

    monkeypatch.setenv("LOCAL_MODE", "true")
    monkeypatch.setenv("FIXTURES_PATH", str(fixtures_dir))
    hf_store._store = None

    result = CliRunner().invoke(
        app, ["seed", "--local", "--seed-dir", str(SEED_DIR)]
    )
    assert result.exit_code == 0, result.output

    hf_store._store = None
    store = hf_store.get_store()
    store.load()
    yield store
    hf_store._store = None


def test_seed_produces_137_rows(seeded_store):
    df = seeded_store.table("canonical_inference_platforms")
    assert len(df) == 137


def test_kind_enum(seeded_store):
    df = seeded_store.table("canonical_inference_platforms")
    bad = set(df["kind"].dropna().unique()) - VALID_KINDS
    assert not bad, f"unexpected kind values: {bad}"
    assert df["kind"].notna().all()


def test_author_lab_canonical_org_fk(seeded_store):
    import pandas as pd

    plat = seeded_store.table("canonical_inference_platforms")
    org_ids = set(seeded_store.table("canonical_orgs")["id"])
    authors = plat[plat["kind"] == "author_lab"]
    for _, row in authors.iterrows():
        co = row["canonical_org"]
        # `canonical_org` is only set when the org is itself curated in
        # seed/orgs.yaml; assert FK validity when present.
        if pd.isna(co) or co == "":
            continue
        assert co in org_ids, (
            f"author_lab {row['id']!r} canonical_org {co!r} not in canonical_orgs"
        )


def test_variant_of_points_at_platform(seeded_store):
    import pandas as pd

    df = seeded_store.table("canonical_inference_platforms")
    all_ids = set(df["id"])
    variants = df[df["kind"].isin(["regional_variant", "coding_plan"])]
    for _, row in variants.iterrows():
        vo = row["variant_of"]
        # `variant_of` is only set when the base platform is itself in the
        # curated list (a few coding_plan rows have no curated base). Assert
        # FK validity when present.
        if pd.isna(vo):
            continue
        assert vo in all_ids, (
            f"{row['kind']} {row['id']!r} variant_of {vo!r} not an existing platform id"
        )


def test_eee_host_tokens_resolve(seeded_store):
    data = json.loads(PROPOSAL_JSON.read_text())
    eee = data["eee_host_tokens"]
    assert len(eee) == 8, f"expected 8 EEE host tokens, got {len(eee)}"
    seeded_ids = set(seeded_store.table("canonical_inference_platforms")["id"])
    for token_name, info in eee.items():
        expected_platform = info["platform"]
        for form in info["forms"]:
            raw = _strip_marker(form)
            resolved = get_host_token_platform(raw)
            assert resolved == expected_platform, (
                f"host token {raw!r} ({token_name}) resolved to {resolved!r}, "
                f"expected {expected_platform!r}"
            )
            assert resolved in seeded_ids, (
                f"resolved platform {resolved!r} not in seeded table"
            )


def test_unknown_sentinel_maps_to_none():
    assert get_host_token_platform("unknown") is None


def test_seed_yaml_author_lab_org_consistency():
    """Cross-check the YAML directly (not just the table) for author_lab
    rows that declare canonical_org."""
    rows = yaml.safe_load((SEED_DIR / "inference_platforms.yaml").read_text())
    org_ids = {
        o["id"] for o in yaml.safe_load((SEED_DIR / "orgs.yaml").read_text())
    }
    for row in rows:
        if row.get("kind") == "author_lab" and row.get("canonical_org"):
            assert row["canonical_org"] in org_ids, (
                f"{row['id']}: canonical_org {row['canonical_org']!r} "
                "not in seed/orgs.yaml"
            )


def test_fuzzy_captured_platforms_are_seeded(seeded_store):
    """Invariant (fuzzy half): every platform_id the fuzzy strategy
    can CAPTURE — from `_HOST_PREFIXES_TO_STRIP` (prefix) and
    `_SUFFIX_PLATFORM_MAP` (suffix) — must exist in the seeded
    `canonical_inference_platforms` table. Guards drift between the resolver's
    single-source map loader and the seed table."""
    from eval_entity_resolver.strategies.fuzzy import (
        _HOST_PREFIXES_TO_STRIP,
        _SUFFIX_PLATFORM_MAP,
    )

    seeded_ids = set(seeded_store.table("canonical_inference_platforms")["id"])
    captured = {
        p
        for p in list(_HOST_PREFIXES_TO_STRIP.values())
        + list(_SUFFIX_PLATFORM_MAP.values())
        if p is not None
    }
    assert captured, "fuzzy captured no platforms — single-source loader failed"
    missing = captured - seeded_ids
    assert not missing, (
        f"fuzzy captures platform ids absent from the seed table: {sorted(missing)}"
    )
