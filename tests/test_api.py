"""API route tests against a fixture-backed in-memory store."""
import pytest
from fastapi.testclient import TestClient

from eval_card_registry.main import app
from eval_card_registry.store.hf_store import get_store
from eval_card_registry.store import schemas as s
from eval_card_registry.services.resolution_service import ResolutionService
from eval_card_registry.services.log_writer import ResolveLogWriter


@pytest.fixture(autouse=True)
def fresh_store(monkeypatch):
    """Replace the module-level store singleton with a fresh in-memory store."""
    from eval_card_registry.store import hf_store

    store = hf_store.RegistryStore()
    store._tables = {name: s.empty(name) for name in [
        "canonical_models", "canonical_benchmarks", "canonical_metrics",
        "eval_harnesses", "aliases", "resolution_log", "eval_results", "sync_runs",
    ]}
    store._loaded = True
    monkeypatch.setattr(hf_store, "_store", store)

    # Set up app.state so route dependencies work without lifespan
    app.state.resolution_service = ResolutionService(store)
    app.state.log_writer = ResolveLogWriter("")  # disabled (no bucket)
    return store


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=True)


class TestHealth:
    def test_health(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_stats(self, client):
        r = client.get("/api/v1/stats")
        assert r.status_code == 200
        data = r.json()
        assert "models" in data
        assert "benchmarks" in data


class TestResolve:
    def test_resolve_unknown_creates_draft(self, client):
        r = client.post("/api/v1/resolve", json={
            "raw_value": "UnknownBenchmark",
            "entity_type": "benchmark",
            "source_config": "test_cfg",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["canonical_id"] is not None
        assert data["created_new"] is True
        assert data["review_status"] == "draft"

    def test_resolve_batch(self, client):
        r = client.post("/api/v1/resolve/batch", json=[
            {"raw_value": "BenchA", "entity_type": "benchmark"},
            {"raw_value": "BenchB", "entity_type": "benchmark"},
        ])
        assert r.status_code == 200
        assert len(r.json()) == 2


class TestEntityCRUD:
    def test_create_and_get_benchmark(self, client):
        r = client.post("/api/v1/benchmarks", json={
            "id": "my-bench",
            "display_name": "My Benchmark",
            "review_status": "reviewed",
        })
        assert r.status_code == 201

        r2 = client.get("/api/v1/benchmarks/my-bench")
        assert r2.status_code == 200
        assert r2.json()["display_name"] == "My Benchmark"

    def test_patch_benchmark(self, client):
        client.post("/api/v1/benchmarks", json={"id": "patch-bench", "display_name": "Old Name"})
        r = client.patch("/api/v1/benchmarks/patch-bench", json={"display_name": "New Name"})
        assert r.status_code == 200
        assert r.json()["display_name"] == "New Name"

    def test_get_nonexistent_returns_404(self, client):
        r = client.get("/api/v1/benchmarks/does-not-exist")
        assert r.status_code == 404

    def test_list_models(self, client):
        client.post("/api/v1/models", json={"id": "org/model-1", "display_name": "Model 1"})
        r = client.get("/api/v1/models")
        assert r.status_code == 200
        assert len(r.json()) >= 1

    def test_list_search_with_null_columns_serializes(self, client):
        """Regression: list endpoints with a search term used to 500 when
        matching rows had nullable columns (pd.NA / NaN) — those fields
        must serialize to JSON null, not raise."""
        client.post("/api/v1/benchmarks", json={"id": "math", "display_name": "MATH"})
        r = client.get("/api/v1/benchmarks?search=math")
        assert r.status_code == 200
        rows = r.json()
        assert len(rows) == 1
        assert rows[0]["parent_benchmark_id"] is None
        assert rows[0]["description"] is None

    def test_get_entity_with_null_columns_serializes(self, client):
        client.post("/api/v1/benchmarks", json={"id": "nullish", "display_name": "N"})
        r = client.get("/api/v1/benchmarks/nullish")
        assert r.status_code == 200
        body = r.json()
        assert body["parent_benchmark_id"] is None
        assert body["dataset_repo"] is None
