"""Coverage for the core.yaml skip_ids / skip_source_ids loader (cli.py
_load_models_merged). skip_ids drops an id from BOTH core and the generated
sources; skip_source_ids drops it from sources/enrichments only, leaving the
curated core entry authoritative (so a source's bad aliases can't be re-merged).

Runs the real seed CLI against a minimal temp seed dir with FIXTURES_PATH pointed
at a temp dir, so it neither touches the repo fixtures nor needs HF.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml
from typer.testing import CliRunner

from eval_card_registry.cli import app


def _write(p: Path, data) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def test_skip_ids_and_skip_source_ids(tmp_path, monkeypatch):
    seed = tmp_path / "seed"
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    monkeypatch.setenv("FIXTURES_PATH", str(fixtures))

    _write(seed / "orgs.yaml", [{"id": "testorg", "display_name": "Test Org", "kind": "lab"}])

    # generated source ships: an id to drop entirely, an id whose bad alias must
    # be suppressed (core curates it), and a normal id to keep.
    _write(seed / "models" / "sources" / "z.generated.yaml", [
        {"id": "testorg/dropme", "org_id": "testorg", "aliases": ["dropme-alias"]},
        {"id": "testorg/curated", "org_id": "testorg", "aliases": ["bad-source-alias"]},
        {"id": "testorg/keep", "org_id": "testorg", "aliases": ["keep-alias"]},
    ])
    # core: skip_ids drops dropme everywhere; skip_source_ids drops curated from
    # the SOURCE only (core's clean version wins, without the bad alias).
    _write(seed / "models" / "core.yaml", {
        "skip_ids": ["testorg/dropme"],
        "skip_source_ids": ["testorg/curated"],
        "entries": [
            {"id": "testorg/curated", "org_id": "testorg", "aliases": ["good-alias"]},
        ],
    })

    res = CliRunner().invoke(app, ["seed", "--local", "--seed-dir", str(seed)])
    assert res.exit_code == 0, res.output

    m = pd.read_parquet(fixtures / "canonical_models.parquet")
    ids = set(m["id"])
    assert "testorg/dropme" not in ids, "skip_ids did not drop the id"
    assert "testorg/keep" in ids and "testorg/curated" in ids

    # the curated entry must NOT carry the source's bad alias (skip_source_ids)
    aliases = pd.read_parquet(fixtures / "aliases.parquet")
    curated_aliases = set(
        aliases[aliases["canonical_id"] == "testorg/curated"]["raw_value"]
    )
    assert "bad-source-alias" not in curated_aliases, "skip_source_ids did not suppress source alias"
    assert "good-alias" in curated_aliases, "curated alias missing"
    # the dropped id must not appear as a resolvable alias either
    assert "dropme-alias" not in set(aliases["raw_value"])
