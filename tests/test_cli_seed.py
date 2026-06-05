"""Tests for the seed CLI — particularly its rename-collision handling.

When seed YAML moves an alias from canonical-A to canonical-B, the seed CLI must
repoint the existing alias row at the new canonical (YAML is the source of truth).
The failure mode this guards against: swallowing the uniqueness ValueError from
add_alias() and then letting end-of-seed stale-removal delete the row entirely.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from eval_card_registry.cli import app
from eval_card_registry.store import hf_store


@pytest.fixture
def fresh_seed_env(tmp_path, monkeypatch):
    """Isolated fixtures-backed registry per test.

    - LOCAL_MODE=true → store reads/writes parquet under tmp_path/fixtures/.
    - FIXTURES_PATH points at tmp_path/fixtures/.
    - The module-level singleton is reset so each test gets a fresh store.
    - All four seed YAMLs are created (empty stubs) so seed_specs entries
      that aren't under test don't dirty the assertions.
    """
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()

    monkeypatch.setenv("LOCAL_MODE", "true")
    monkeypatch.setenv("FIXTURES_PATH", str(fixtures_dir))

    # Reset the singleton so a previous test's store doesn't leak.
    hf_store._store = None

    # Empty stubs for the YAMLs we don't exercise — keep the loop quiet.
    (seed_dir / "metrics.yaml").write_text("[]\n")
    (seed_dir / "harnesses.yaml").write_text("[]\n")
    # Models layout: seed/models/{core.yaml,sources/,enrichments/}. The loader
    # tolerates missing files, but core.yaml as an empty list keeps the
    # surface area explicit for any future tests that read these stubs.
    (seed_dir / "models").mkdir()
    (seed_dir / "models" / "sources").mkdir()
    (seed_dir / "models" / "enrichments").mkdir()
    (seed_dir / "models" / "core.yaml").write_text("[]\n")

    yield seed_dir, fixtures_dir

    # Teardown: clear singleton so other tests start clean too.
    hf_store._store = None


def _write_benchmarks(seed_dir: Path, entries: list[dict]) -> None:
    (seed_dir / "benchmarks.yaml").write_text(yaml.safe_dump(entries))


def _read_aliases(fixtures_dir: Path):
    import pandas as pd
    return pd.read_parquet(fixtures_dir / "aliases.parquet")


def _seed(seed_dir: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["seed", "--seed-dir", str(seed_dir)])
    assert result.exit_code == 0, f"seed failed: {result.output}"


def test_alias_canonical_rename_doesnt_drop(fresh_seed_env):
    """The bug: rename canonical of an alias and the alias survives, repointed.

    Before the fix this test would fail because the second seed run silently
    swallowed the ValueError from add_alias and then stale-removal deleted
    the orphaned row.
    """
    seed_dir, fixtures_dir = fresh_seed_env

    # Round 1: alias 'special-token' belongs to canonical 'old-bench'.
    _write_benchmarks(seed_dir, [
        {
            "id": "old-bench",
            "display_name": "Old Bench",
            "review_status": "reviewed",
            "aliases": ["special-token"],
        },
    ])
    _seed(seed_dir)

    aliases_df = _read_aliases(fixtures_dir)
    row = aliases_df[(aliases_df["raw_value"] == "special-token")
                     & (aliases_df["entity_type"] == "benchmark")]
    assert len(row) == 1, "alias should exist after first seed"
    assert row.iloc[0]["canonical_id"] == "old-bench"
    original_alias_uuid = row.iloc[0]["id"]

    # Round 2: YAML now puts 'special-token' on a NEW canonical 'new-bench'.
    # The old canonical is also kept (different alias) so we're testing only
    # the rename path, not the entity-removal path.
    _write_benchmarks(seed_dir, [
        {
            "id": "old-bench",
            "display_name": "Old Bench",
            "review_status": "reviewed",
            # 'special-token' moved away
        },
        {
            "id": "new-bench",
            "display_name": "New Bench",
            "review_status": "reviewed",
            "aliases": ["special-token"],
        },
    ])
    _seed(seed_dir)

    aliases_df = _read_aliases(fixtures_dir)
    row = aliases_df[(aliases_df["raw_value"] == "special-token")
                     & (aliases_df["entity_type"] == "benchmark")]
    assert len(row) == 1, (
        f"alias 'special-token' was deleted by the rename "
        f"(this is the bug). Aliases now: "
        f"{aliases_df[['raw_value', 'canonical_id']].to_dict('records')}"
    )
    assert row.iloc[0]["canonical_id"] == "new-bench", (
        f"alias should have been repointed to 'new-bench', got "
        f"{row.iloc[0]['canonical_id']!r}"
    )
    # The alias row identity is preserved (same UUID) — we updated, not
    # deleted-and-recreated. This matters for any downstream join that
    # carries the alias UUID.
    assert row.iloc[0]["id"] == original_alias_uuid


def test_repeat_seed_is_idempotent(fresh_seed_env):
    """Sanity check: same YAML seeded twice doesn't multiply or churn rows."""
    seed_dir, fixtures_dir = fresh_seed_env

    _write_benchmarks(seed_dir, [
        {
            "id": "bench-a",
            "display_name": "Bench A",
            "review_status": "reviewed",
            "aliases": ["tok-a", "tok-b"],
        },
    ])
    _seed(seed_dir)
    first = _read_aliases(fixtures_dir)
    first_count = len(first[first["entity_type"] == "benchmark"])

    _seed(seed_dir)
    second = _read_aliases(fixtures_dir)
    second_count = len(second[second["entity_type"] == "benchmark"])

    assert first_count == second_count, (
        f"alias count changed across identical re-seed: "
        f"{first_count} -> {second_count}"
    )


def _read_models(fixtures_dir: Path):
    import pandas as pd
    return pd.read_parquet(fixtures_dir / "canonical_models.parquet")


def _read_orgs(fixtures_dir: Path):
    import pandas as pd
    return pd.read_parquet(fixtures_dir / "canonical_orgs.parquet")


def test_parents_union_by_id_across_sources(fresh_seed_env):
    """Two seed sources contributing edges for the same model union by id:
    same parent id from both sides yields one edge; different ids yield
    multiple edges."""
    import json
    seed_dir, fixtures_dir = fresh_seed_env

    # Two parents: one shared between source and core (axis only on source),
    # one source-only, one core-only. After merge we expect three distinct
    # edges with axis preserved on the shared one.
    sources_dir = seed_dir / "models" / "sources"
    (sources_dir / "test.generated.yaml").write_text(yaml.safe_dump([
        {
            "id": "lab/model-a",
            "display_name": "Model A",
            "parents": [
                {"id": "lab/parent-shared", "relationship": "variant", "axis": "size"},
                {"id": "lab/parent-source-only", "relationship": "variant"},
            ],
            "review_status": "reviewed",
        },
    ]))
    (seed_dir / "models" / "core.yaml").write_text(yaml.safe_dump([
        {
            "id": "lab/model-a",
            "display_name": "Model A",
            "parents": [
                # Same id as source's first edge — should fold, not duplicate.
                # Core omits axis; source's axis must survive the merge.
                {"id": "lab/parent-shared", "relationship": "variant"},
                {"id": "lab/parent-core-only", "relationship": "finetune"},
            ],
            "review_status": "reviewed",
        },
    ]))
    _seed(seed_dir)

    df = _read_models(fixtures_dir)
    row = df[df["id"] == "lab/model-a"]
    assert len(row) == 1
    parents = json.loads(row.iloc[0]["parents"])
    by_id = {p["id"]: p for p in parents}
    assert set(by_id) == {"lab/parent-shared", "lab/parent-source-only", "lab/parent-core-only"}
    # Shared edge: source brought axis=size, core didn't, axis must survive.
    assert by_id["lab/parent-shared"]["axis"] == "size"
    assert by_id["lab/parent-shared"]["relationship"] == "variant"
    # Distinct edges retain their relationships.
    assert by_id["lab/parent-source-only"]["relationship"] == "variant"
    assert by_id["lab/parent-core-only"]["relationship"] == "finetune"


def test_orgs_generated_curated_wins_on_collision(fresh_seed_env):
    """seed/orgs.generated.yaml entries are dropped when the id is already
    in seed/orgs.yaml; non-conflicting entries are still loaded."""
    seed_dir, fixtures_dir = fresh_seed_env

    (seed_dir / "orgs.yaml").write_text(yaml.safe_dump([
        {
            "id": "anthropic",
            "display_name": "Anthropic",
            "kind": "lab",
            "review_status": "reviewed",
        },
    ]))
    (seed_dir / "orgs.generated.yaml").write_text(yaml.safe_dump([
        # Collides with curated — must be dropped (curated display_name wins).
        {
            "id": "anthropic",
            "display_name": "Anthropic Auto",
            "kind": "unknown",
            "review_status": "auto",
        },
        # Doesn't collide — must be loaded.
        {
            "id": "some-uploader",
            "display_name": "Some Uploader",
            "kind": "individual",
            "review_status": "auto",
        },
    ]))
    _seed(seed_dir)

    orgs = _read_orgs(fixtures_dir)
    anthropic = orgs[orgs["id"] == "anthropic"].iloc[0]
    assert anthropic["display_name"] == "Anthropic", "curated entry must win on collision"
    assert anthropic["kind"] == "lab"
    assert anthropic["review_status"] == "reviewed"

    auto = orgs[orgs["id"] == "some-uploader"].iloc[0]
    assert auto["display_name"] == "Some Uploader"
    assert auto["kind"] == "individual"


def test_prune_stale_keeps_org_referenced_by_model(fresh_seed_env):
    """--prune-stale must NOT drop an org still referenced by a surviving model's
    org_id FK. A model whose org_id is DERIVED from its id-prefix (no curated org
    row in any YAML) gets an org row auto-created at seed time; pruning it as
    'stale' (not in the orgs YAML) would orphan the model with a dangling FK."""
    seed_dir, fixtures_dir = fresh_seed_env
    (seed_dir / "models" / "core.yaml").write_text(yaml.safe_dump([
        {
            "id": "novelorg/some-model",
            "display_name": "Some Model",
            "review_status": "reviewed",
        },
    ]))
    runner = CliRunner()
    result = runner.invoke(app, ["seed", "--seed-dir", str(seed_dir), "--prune-stale"])
    assert result.exit_code == 0, f"seed --prune-stale failed: {result.output}"

    models = _read_models(fixtures_dir)
    orgs = _read_orgs(fixtures_dir)
    row = models[models["id"] == "novelorg/some-model"]
    assert len(row) == 1
    org_id = row.iloc[0]["org_id"]
    assert org_id and str(org_id) != "nan", "model should have a derived org_id"
    assert str(org_id) in set(orgs["id"].astype(str)), (
        f"--prune-stale dropped org {org_id!r} still referenced by a surviving "
        f"model (dangling org_id FK). Org ids: {sorted(orgs['id'].astype(str))}"
    )


def test_prune_stale_keeps_org_referenced_by_parent_org_id(fresh_seed_env):
    """--prune-stale must NOT drop an org still referenced by a surviving org's
    parent_org_id FK. Round 1 seeds a child org pointing at a parent; round 2
    drops the parent from the YAML (so it looks stale) but keeps the child — the
    parent must survive because the child still references it."""
    seed_dir, fixtures_dir = fresh_seed_env

    orgs_path = seed_dir / "orgs.yaml"
    orgs_path.write_text(yaml.safe_dump([
        {"id": "parent-lab", "display_name": "Parent Lab",
         "kind": "lab", "review_status": "reviewed"},
        {"id": "child-lab", "display_name": "Child Lab", "kind": "lab",
         "review_status": "reviewed", "parent_org_id": "parent-lab"},
    ]))
    _seed(seed_dir)
    assert "parent-lab" in set(_read_orgs(fixtures_dir)["id"].astype(str))

    # Round 2: parent-lab is no longer in the YAML, but child-lab still points at it.
    orgs_path.write_text(yaml.safe_dump([
        {"id": "child-lab", "display_name": "Child Lab", "kind": "lab",
         "review_status": "reviewed", "parent_org_id": "parent-lab"},
    ]))
    runner = CliRunner()
    result = runner.invoke(app, ["seed", "--seed-dir", str(seed_dir), "--prune-stale"])
    assert result.exit_code == 0, f"seed --prune-stale failed: {result.output}"

    orgs = _read_orgs(fixtures_dir)
    assert "parent-lab" in set(orgs["id"].astype(str)), (
        "--prune-stale dropped an org still referenced by a surviving org's "
        f"parent_org_id FK. Org ids: {sorted(orgs['id'].astype(str))}"
    )


def test_orgs_generated_alone_loads(fresh_seed_env):
    """When seed/orgs.yaml is absent, seed/orgs.generated.yaml still loads."""
    seed_dir, fixtures_dir = fresh_seed_env
    (seed_dir / "orgs.generated.yaml").write_text(yaml.safe_dump([
        {
            "id": "lone-uploader",
            "display_name": "Lone",
            "kind": "individual",
            "review_status": "auto",
        },
    ]))
    _seed(seed_dir)

    orgs = _read_orgs(fixtures_dir)
    assert "lone-uploader" in set(orgs["id"])


def test_scoped_alias_rename_doesnt_drop(fresh_seed_env):
    """Same bug, but for scoped aliases (source_config != None)."""
    seed_dir, fixtures_dir = fresh_seed_env

    _write_benchmarks(seed_dir, [
        {
            "id": "old-bench",
            "display_name": "Old Bench",
            "review_status": "reviewed",
            "scoped_aliases": {"some_config": ["Overall"]},
        },
    ])
    _seed(seed_dir)

    aliases_df = _read_aliases(fixtures_dir)
    row = aliases_df[(aliases_df["raw_value"] == "Overall")
                     & (aliases_df["source_config"] == "some_config")]
    assert len(row) == 1
    assert row.iloc[0]["canonical_id"] == "old-bench"

    _write_benchmarks(seed_dir, [
        {
            "id": "old-bench",
            "display_name": "Old Bench",
            "review_status": "reviewed",
        },
        {
            "id": "new-bench",
            "display_name": "New Bench",
            "review_status": "reviewed",
            "scoped_aliases": {"some_config": ["Overall"]},
        },
    ])
    _seed(seed_dir)

    aliases_df = _read_aliases(fixtures_dir)
    row = aliases_df[(aliases_df["raw_value"] == "Overall")
                     & (aliases_df["source_config"] == "some_config")]
    assert len(row) == 1, (
        "scoped alias 'Overall' was deleted by the rename "
        "(this is the bug, scoped variant)"
    )
    assert row.iloc[0]["canonical_id"] == "new-bench"
