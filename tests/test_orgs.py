"""Tests for the org entity type — resolution paths, FK linking, CRUD routes."""
import pytest
from fastapi.testclient import TestClient

from eval_card_registry.main import app
from eval_card_registry.services.log_writer import ResolveLogWriter
from eval_card_registry.services.resolution_service import ResolutionService
from eval_card_registry.store import hf_store, queries, schemas


@pytest.fixture(autouse=True)
def fresh_store(monkeypatch):
    store = hf_store.RegistryStore()
    store._tables = {n: schemas.empty(n) for n in [
        "canonical_orgs", "canonical_models", "canonical_benchmarks", "canonical_metrics",
        "eval_harnesses", "aliases", "resolution_log", "eval_results", "sync_runs",
    ]}
    store._loaded = True
    monkeypatch.setattr(hf_store, "_store", store)
    app.state.resolution_service = ResolutionService(store)
    app.state.log_writer = ResolveLogWriter("")
    return store


@pytest.fixture
def client():
    return TestClient(app)


def _seed_org(store, id, display_name, aliases=None):
    queries.upsert_entity(store, "canonical_orgs", {
        "id": id, "display_name": display_name, "parent_org_id": None,
        "website": None, "hf_org": None,
        "tags": "[]", "metadata": "{}", "review_status": "reviewed",
    })
    for raw in [id, display_name] + (aliases or []):
        try:
            queries.add_alias(store, {
                "raw_value": raw, "entity_type": "org", "canonical_id": id,
                "source_config": None, "source_field": "test",
                "status": "confirmed", "strategy": "seed",
                "confidence": 1.0, "notes": None,
            })
        except ValueError:
            pass


class TestOrgResolution:
    def test_exact_match(self, fresh_store):
        _seed_org(fresh_store, "anthropic", "Anthropic")
        svc = app.state.resolution_service
        r = svc.resolve("Anthropic", "org", None, None)
        assert r["canonical_id"] == "anthropic"
        assert r["strategy"] == "seed"

    def test_normalized_collapses_case_and_whitespace(self, fresh_store):
        _seed_org(fresh_store, "anthropic", "Anthropic", aliases=["Anthropic, PBC"])
        svc = app.state.resolution_service
        # Casing variant
        r = svc.resolve("anthropic", "org", None, None)
        assert r["canonical_id"] == "anthropic"
        # Formal-name alias hit
        r2 = svc.resolve("Anthropic, PBC", "org", None, None)
        assert r2["canonical_id"] == "anthropic"

    def test_auto_draft_for_unknown(self, fresh_store):
        svc = app.state.resolution_service
        r = svc.resolve("Some New Lab", "org", None, None)
        assert r["created_new"] is True
        assert r["canonical_id"] == "some-new-lab"
        assert r["review_status"] == "draft"


class TestModelOrgFK:
    def test_auto_draft_model_with_org_prefix_links_org(self, fresh_store):
        """Model auto-drafts derive org_id from `org/model` slug shape."""
        _seed_org(fresh_store, "meta", "Meta", aliases=["meta-llama"])
        svc = app.state.resolution_service
        # New HF-format model id triggers auto-draft. Resolver should
        # extract `meta-llama` prefix, resolve to org `meta`, and set org_id.
        r = svc.resolve("meta-llama/Some-New-Model-1B", "model", None, None)
        assert r["created_new"] is True
        # Flush pending and inspect the entity row
        queries.flush_pending(fresh_store)
        df = fresh_store.table("canonical_models")
        new_row = df[df["id"] == r["canonical_id"]].iloc[0]
        assert new_row["org_id"] == "meta", f"expected meta, got {new_row['org_id']!r}"

    def test_auto_draft_model_without_slash_has_null_org(self, fresh_store):
        """No slash in raw value -> can't derive org -> org_id stays null."""
        svc = app.state.resolution_service
        r = svc.resolve("Custom-Model-Name", "model", None, None)
        assert r["created_new"] is True
        queries.flush_pending(fresh_store)
        df = fresh_store.table("canonical_models")
        new_row = df[df["id"] == r["canonical_id"]].iloc[0]
        # pandas NA for unset string column
        import pandas as pd
        assert pd.isna(new_row["org_id"])


class TestOrgRoutes:
    def test_post_get_org(self, client):
        r = client.post("/api/v1/orgs", json={
            "id": "test-org", "display_name": "Test Org",
        })
        assert r.status_code == 201
        r2 = client.get("/api/v1/orgs/test-org")
        assert r2.status_code == 200
        assert r2.json()["display_name"] == "Test Org"

    def test_patch_org(self, client):
        client.post("/api/v1/orgs", json={"id": "test-org", "display_name": "Test"})
        r = client.patch("/api/v1/orgs/test-org", json={"website": "https://x.com"})
        assert r.status_code == 200
        assert r.json()["website"] == "https://x.com"

    def test_list_orgs_search(self, client):
        client.post("/api/v1/orgs", json={"id": "alpha", "display_name": "Alpha"})
        client.post("/api/v1/orgs", json={"id": "beta", "display_name": "Beta"})
        r = client.get("/api/v1/orgs?search=alpha")
        assert r.status_code == 200
        ids = [e["id"] for e in r.json()]
        assert "alpha" in ids
        assert "beta" not in ids


class TestModelDeveloperDerivedField:
    def test_get_model_returns_developer_from_org_display_name(self, client):
        client.post("/api/v1/orgs", json={"id": "anthropic", "display_name": "Anthropic"})
        client.post("/api/v1/models", json={
            "id": "anthropic/claude", "display_name": "Claude", "org_id": "anthropic",
        })
        r = client.get("/api/v1/models/anthropic/claude")
        body = r.json()
        assert body["org_id"] == "anthropic"
        # `developer` is derived from canonical_orgs.display_name
        assert body["developer"] == "Anthropic"

    def test_get_model_with_unknown_org_id_has_null_developer(self, client):
        client.post("/api/v1/models", json={
            "id": "x/y", "display_name": "X", "org_id": "missing-org",
        })
        r = client.get("/api/v1/models/x/y")
        body = r.json()
        assert body["org_id"] == "missing-org"
        assert body["developer"] is None


class TestParentCanonicalId:
    def test_populates_from_parent_benchmark_id(self, fresh_store):
        # Parent
        queries.upsert_entity(fresh_store, "canonical_benchmarks", {
            "id": "helm", "display_name": "HELM",
            "description": None, "dataset_repo": None, "parent_benchmark_id": None,
            "tags": "[]", "metadata": "{}", "review_status": "reviewed",
        })
        # Child
        queries.upsert_entity(fresh_store, "canonical_benchmarks", {
            "id": "helm-air-bench", "display_name": "HELM AIR-Bench",
            "description": None, "dataset_repo": None, "parent_benchmark_id": "helm",
            "tags": "[]", "metadata": "{}", "review_status": "reviewed",
        })
        queries.add_alias(fresh_store, {
            "raw_value": "helm_air_bench", "entity_type": "benchmark",
            "canonical_id": "helm-air-bench", "source_config": None,
            "source_field": "test", "status": "confirmed",
            "strategy": "seed", "confidence": 1.0, "notes": None,
        })
        svc = app.state.resolution_service
        r = svc.resolve("helm_air_bench", "benchmark", None, None)
        assert r["canonical_id"] == "helm-air-bench"
        assert r["parent_canonical_id"] == "helm"

    def test_null_for_top_of_family(self, fresh_store):
        queries.upsert_entity(fresh_store, "canonical_benchmarks", {
            "id": "math", "display_name": "MATH",
            "description": None, "dataset_repo": None, "parent_benchmark_id": None,
            "tags": "[]", "metadata": "{}", "review_status": "reviewed",
        })
        queries.add_alias(fresh_store, {
            "raw_value": "MATH", "entity_type": "benchmark", "canonical_id": "math",
            "source_config": None, "source_field": "test", "status": "confirmed",
            "strategy": "seed", "confidence": 1.0, "notes": None,
        })
        svc = app.state.resolution_service
        r = svc.resolve("MATH", "benchmark", None, None)
        assert r["canonical_id"] == "math"
        assert r["parent_canonical_id"] is None
