"""Read-only HF id CONFIRMATION via the local hub_stats_index.

The read-only resolve path consults a periodically-refreshed local index of HF
model ids so it can CONFIRM an exact HF model id that was never minted into the
registry — with NO minting / NO persistence. The confirmation comes back in the
EXISTING ResolveResponse shape:
  canonical_id=<HF-true id>, strategy="exact", confidence=1.0, created_new=False,
  resolution_source="hub_stats_index", review_status=None, ancestry=[],
  resolution_detail={"granularity": None, "hf_repo_id": <same id>}.

Mirrors tests/test_d1_hierarchy.py fixture/TestClient style. Synthetic
hub_stats_index rows are built in-memory — never the real index.
"""
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from eval_card_registry.config import settings
from eval_card_registry.main import app
from eval_card_registry.store import hf_store, schemas as s
from eval_card_registry.services.resolution_service import ResolutionService
from eval_card_registry.services.log_writer import ResolveLogWriter
from eval_card_registry.services.hub_stats import normalize


def _row(table: str, **vals) -> dict:
    base = {col: None for col in s._SCHEMAS[table]}
    base.update({"review_status": "reviewed",
                 "created_at": "2026-01-01T00:00:00+00:00",
                 "updated_at": "2026-01-01T00:00:00+00:00"})
    for k, v in vals.items():
        base[k] = json.dumps(v) if isinstance(v, (list, dict)) else v
    return base


def _alias(raw, etype, cid):
    return {"id": f"{etype}:{raw}", "raw_value": raw, "entity_type": etype,
            "canonical_id": cid, "source_config": None, "source_field": None,
            "status": "confirmed", "strategy": "exact", "confidence": 1.0,
            "notes": None, "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00"}


def _index_row(hf_id: str, *, with_norm: bool = True) -> dict:
    base = {col: None for col in s._SCHEMAS["hub_stats_index"]}
    base.update({
        "id": hf_id,
        "id_norm": normalize(hf_id) if with_norm else None,
        "release_date": "2024-07-23",
        "pipeline_tag": "text-generation",
        "params_billions": 8.03,
        "downloads": 12345,
        "open_weights": True,
    })
    return base


_ALL_TABLES = [
    "canonical_models", "canonical_benchmarks", "canonical_metrics",
    "eval_harnesses", "canonical_orgs", "canonical_families",
    "canonical_composites", "aliases", "resolution_log", "eval_results",
    "sync_runs",
]


def _base_store(monkeypatch, *, index_rows=None, include_index=True):
    """A read-only store: one registry-attested model + an optional
    hub_stats_index. The registry model is HF-backed (resolution_source=hf)
    so it carries hf_repo_id. The index holds HF ids that are NOT in the
    registry."""
    store = hf_store.RegistryStore()
    tables = {name: s.empty(name) for name in _ALL_TABLES}

    tables["canonical_models"] = pd.DataFrame([
        _row("canonical_models", id="meta-llama/Llama-3.1-8B",
             display_name="Llama 3.1 8B", org_id="meta",
             resolution_source="hf", parents="[]"),
    ])
    tables["aliases"] = pd.DataFrame([
        _alias("meta-llama/Llama-3.1-8B", "model", "meta-llama/Llama-3.1-8B"),
    ])

    if include_index:
        rows = index_rows if index_rows is not None else [
            _index_row("Qwen/Qwen2.5-7B-Instruct"),
            _index_row("mistralai/Mistral-Nemo-Instruct-2407"),
        ]
        tables["hub_stats_index"] = (
            pd.DataFrame(rows) if rows else s.empty("hub_stats_index"))

    store._tables = tables
    store._loaded = True
    monkeypatch.setattr(hf_store, "_store", store)
    app.state.resolution_service = ResolutionService(store)
    app.state.log_writer = ResolveLogWriter("")
    return store


@pytest.fixture
def read_only(monkeypatch):
    """Force READ_ONLY for the duration of a test."""
    original = settings.read_only
    settings.read_only = True
    yield
    settings.read_only = original


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=True)


# --- (i) index-only HF id confirms ----------------------------------------

def test_index_only_id_confirms(monkeypatch, read_only, client):
    _base_store(monkeypatch)
    r = client.post("/api/v1/resolve", json={
        "raw_value": "Qwen/Qwen2.5-7B-Instruct", "entity_type": "model"})
    assert r.status_code == 200
    d = r.json()
    assert d["canonical_id"] == "Qwen/Qwen2.5-7B-Instruct"
    assert d["resolution_source"] == "hub_stats_index"
    assert d["strategy"] == "exact"
    assert d["confidence"] == 1.0
    assert d["created_new"] is False
    assert d["review_status"] is None
    assert d["ancestry"] == []
    assert d["resolution_detail"] == {
        "granularity": None, "hf_repo_id": "Qwen/Qwen2.5-7B-Instruct"}


# --- (ii) registry precedence — index NOT consulted on a registry hit ------

def test_registry_entity_wins(monkeypatch, read_only, client):
    # Put the registry-attested id ALSO in the index, but with a different
    # casing, to prove the index isn't what produced the answer.
    _base_store(monkeypatch, index_rows=[_index_row("META-LLAMA/llama-3.1-8b")])
    r = client.post("/api/v1/resolve", json={
        "raw_value": "meta-llama/Llama-3.1-8B", "entity_type": "model"})
    assert r.status_code == 200
    d = r.json()
    assert d["canonical_id"] == "meta-llama/Llama-3.1-8B"
    # Registry match -> NOT a hub_stats_index confirmation.
    assert d["resolution_source"] != "hub_stats_index"
    assert d["resolution_detail"]["hf_repo_id"] == "meta-llama/Llama-3.1-8B"


# --- (iii) non-HF-shaped miss stays no_match -------------------------------

def test_non_hf_shaped_miss_no_match(monkeypatch, read_only, client):
    _base_store(monkeypatch)
    r = client.post("/api/v1/resolve", json={
        "raw_value": "just-a-name", "entity_type": "model"})
    d = r.json()
    assert d["canonical_id"] is None
    assert d["strategy"] == "no_match"
    assert d["resolution_source"] is None


# --- (iv) id absent from both stays no_match -------------------------------

def test_absent_from_both_no_match(monkeypatch, read_only, client):
    _base_store(monkeypatch)
    r = client.post("/api/v1/resolve", json={
        "raw_value": "someorg/totally-unknown-model", "entity_type": "model"})
    d = r.json()
    assert d["canonical_id"] is None
    assert d["strategy"] == "no_match"
    assert d["resolution_source"] is None


# --- (v) missing / empty index degrades to no_match ------------------------

def test_missing_index_table_degrades(monkeypatch, read_only, client):
    _base_store(monkeypatch, include_index=False)
    r = client.post("/api/v1/resolve", json={
        "raw_value": "Qwen/Qwen2.5-7B-Instruct", "entity_type": "model"})
    d = r.json()
    assert d["canonical_id"] is None
    assert d["strategy"] == "no_match"


def test_empty_index_table_degrades(monkeypatch, read_only, client):
    _base_store(monkeypatch, index_rows=[])
    r = client.post("/api/v1/resolve", json={
        "raw_value": "Qwen/Qwen2.5-7B-Instruct", "entity_type": "model"})
    d = r.json()
    assert d["canonical_id"] is None
    assert d["strategy"] == "no_match"


# --- (vi) case/separator-variant input still hits via id_norm --------------

def test_case_separator_variant_hits(monkeypatch, read_only, client):
    _base_store(monkeypatch)
    # Lowercased + underscore separators — normalizes to the same id_norm.
    r = client.post("/api/v1/resolve", json={
        "raw_value": "qwen/qwen2.5_7b_instruct", "entity_type": "model"})
    d = r.json()
    assert d["canonical_id"] == "Qwen/Qwen2.5-7B-Instruct"
    assert d["resolution_source"] == "hub_stats_index"
    assert d["resolution_detail"]["hf_repo_id"] == "Qwen/Qwen2.5-7B-Instruct"
    # A non-byte-equal (normalized) hit is labelled honestly, not "exact".
    assert d["strategy"] == "normalized"
    assert d["confidence"] == 0.95


def test_variant_hits_even_without_id_norm_column_value(monkeypatch, read_only, client):
    # Index row has no precomputed id_norm -> build falls back to normalize(id).
    _base_store(monkeypatch, index_rows=[
        _index_row("Qwen/Qwen2.5-7B-Instruct", with_norm=False)])
    r = client.post("/api/v1/resolve", json={
        "raw_value": "qwen/qwen2.5-7b-instruct", "entity_type": "model"})
    d = r.json()
    assert d["canonical_id"] == "Qwen/Qwen2.5-7B-Instruct"
    assert d["resolution_source"] == "hub_stats_index"


# --- (vii) hf_repo_id populated for BOTH a registry hit and an index hit ----

def test_hf_repo_id_populated_both_paths(monkeypatch, read_only, client):
    _base_store(monkeypatch)
    reg = client.post("/api/v1/resolve", json={
        "raw_value": "meta-llama/Llama-3.1-8B", "entity_type": "model"}).json()
    idx = client.post("/api/v1/resolve", json={
        "raw_value": "Qwen/Qwen2.5-7B-Instruct", "entity_type": "model"}).json()
    assert reg["resolution_detail"]["hf_repo_id"] == "meta-llama/Llama-3.1-8B"
    assert idx["resolution_detail"]["hf_repo_id"] == "Qwen/Qwen2.5-7B-Instruct"


# --- precedence: exact index confirmation beats a FUZZY registry match ------

def _fuzzy_collision_store(monkeypatch, *, include_variant_in_index: bool):
    """Registry holds a base canonical `acme/widget-7b` that the raw value
    `acme/widget-7b-fp8` FUZZY-matches (quant suffix stripped). The index
    optionally holds the exact variant id."""
    store = hf_store.RegistryStore()
    tables = {name: s.empty(name) for name in _ALL_TABLES}
    tables["canonical_models"] = pd.DataFrame([
        _row("canonical_models", id="acme/widget-7b", display_name="Widget 7B",
             org_id="acme", resolution_source="hf", parents="[]"),
    ])
    tables["aliases"] = pd.DataFrame([_alias("acme/widget-7b", "model", "acme/widget-7b")])
    tables["hub_stats_index"] = (
        pd.DataFrame([_index_row("acme/widget-7b-fp8")])
        if include_variant_in_index else s.empty("hub_stats_index")
    )
    store._tables = tables
    store._loaded = True
    monkeypatch.setattr(hf_store, "_store", store)
    app.state.resolution_service = ResolutionService(store)
    app.state.log_writer = ResolveLogWriter("")
    return store


def test_exact_index_overrides_fuzzy_registry_match(monkeypatch, read_only, client):
    # The exact HF-id confirmation must WIN over the fuzzy registry stem-match
    # to the slug base, else a real-but-unminted HF id is shadowed by a slug.
    _fuzzy_collision_store(monkeypatch, include_variant_in_index=True)
    r = client.post("/api/v1/resolve", json={
        "raw_value": "acme/widget-7b-fp8", "entity_type": "model"})
    d = r.json()
    assert d["resolution_source"] == "hub_stats_index"
    assert d["canonical_id"] == "acme/widget-7b-fp8"
    assert d["resolution_detail"]["hf_repo_id"] == "acme/widget-7b-fp8"
    assert d["strategy"] == "exact"


def test_fuzzy_registry_match_kept_when_not_in_index(monkeypatch, read_only, client):
    # No index hit -> the fuzzy registry match is preserved (precedence change
    # must NOT turn an existing fuzzy match into a no_match).
    _fuzzy_collision_store(monkeypatch, include_variant_in_index=False)
    r = client.post("/api/v1/resolve", json={
        "raw_value": "acme/widget-7b-fp8", "entity_type": "model"})
    d = r.json()
    assert d["canonical_id"] == "acme/widget-7b"
    assert d["strategy"] == "fuzzy"


# --- drift guard: build-script SQL id_norm must equal Python normalize() ----

def test_sql_id_norm_matches_python_normalize():
    """The build script computes id_norm in DuckDB SQL; the resolver looks up
    with Python normalize(). If the two ever drift, separator/case-variant
    confirmation silently breaks. Exercise the REAL SQL over frozen rows."""
    import importlib.util
    from pathlib import Path

    import duckdb

    root = Path(__file__).resolve().parents[1]
    frozen = root / "curation" / "hub_stats_frozen.parquet"
    if not frozen.exists():
        pytest.skip("frozen hub-stats parquet not present")
    spec = importlib.util.spec_from_file_location(
        "build_hub_stats_index", root / "scripts" / "build_hub_stats_index.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    q = mod.build_query([str(frozen)], limit=300)
    rows = duckdb.connect().execute(q).fetchall()  # (id, id_norm, ...)
    assert rows, "no rows from frozen parquet"
    mismatches = [(r[0], r[1]) for r in rows if r[1] != normalize(r[0])]
    assert not mismatches, f"SQL id_norm drifted from normalize(): {mismatches[:5]}"
