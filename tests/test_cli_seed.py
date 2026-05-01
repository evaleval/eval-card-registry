"""Tests for the seed CLI — particularly its rename-collision handling.

Regression coverage for a real bug we hit during Phase 1.6: when seed YAML
moves an alias from canonical-A to canonical-B, the seed CLI used to
silently swallow the resulting uniqueness ValueError from add_alias() and
let stale-removal at the end of seeding delete the row entirely. The
correct behavior is to repoint the existing alias row at the new canonical
(YAML is the source of truth).
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
