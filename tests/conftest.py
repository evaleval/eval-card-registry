"""Shared test fixtures — reset module-level caches between tests."""
import pytest
from eval_card_registry.store import queries, hf_store


@pytest.fixture(autouse=True)
def _reset_query_caches():
    """Clear module-level caches and pending buffers before each test for isolation."""
    queries._alias_index.clear()
    queries._pending_result_ids.clear()
    # Clear any pending buffer on the singleton store
    store = getattr(hf_store, "_store", None)
    if store is not None and hasattr(store, "_pending"):
        store._pending = {}
    yield
    queries._alias_index.clear()
    queries._pending_result_ids.clear()
    store = getattr(hf_store, "_store", None)
    if store is not None and hasattr(store, "_pending"):
        store._pending = {}
