"""Shared test fixtures — reset module-level caches between tests."""
import importlib.util
from pathlib import Path

import pytest
from eval_card_registry.config import settings
from eval_card_registry.store import queries, hf_store

# Repo root (parent of tests/) — scripts/ live one level up from this file.
_REPO_ROOT = Path(__file__).resolve().parent.parent


def load_script_module(script_stem, module_name=None):
    """Import scripts/<script_stem>.py as a standalone module without making
    scripts/ a package, so tests can call its functions without running its
    destructive main().

    module_name overrides the registered module name (defaults to script_stem);
    a few tests load the same script under a distinct synthetic name.
    """
    path = _REPO_ROOT / "scripts" / f"{script_stem}.py"
    spec = importlib.util.spec_from_file_location(module_name or script_stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


@pytest.fixture(autouse=True)
def _disable_hub_stats_lookup():
    """Hub-stats live lookup hits HF — disable globally for tests.
    Individual tests that exercise the lookup path can re-enable + mock."""
    original = settings.hub_stats_lookup_enabled
    settings.hub_stats_lookup_enabled = False
    yield
    settings.hub_stats_lookup_enabled = original
