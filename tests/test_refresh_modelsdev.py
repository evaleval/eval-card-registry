"""Tests for the models.dev refresh script — synthetic data, no network."""
import importlib.util
import json
from pathlib import Path

import pytest


def _load_module():
    """Import scripts/refresh_from_modelsdev.py without making it a package."""
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "scripts" / "refresh_from_modelsdev.py"
    spec = importlib.util.spec_from_file_location("refresh_from_modelsdev", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def mod():
    return _load_module()


SYNTHETIC_API = {
    # Author provider — should be processed. Names mirror real models.dev
    # convention: `name` carries the lab's spelling (with dots, no dates),
    # `id` is the slugified form (date in id when it's a snapshot).
    "anthropic": {
        "id": "anthropic", "name": "Anthropic", "doc": "...", "env": [], "npm": "",
        "models": {
            "claude-opus-4-5": {
                "id": "claude-opus-4-5", "name": "Claude Opus 4.5",
                "family": "claude-opus", "release_date": "2025-11-01",
                "open_weights": False, "knowledge": "2025-05",
            },
            "claude-opus-4-5-20251101": {
                "id": "claude-opus-4-5-20251101", "name": "Claude Opus 4.5",
                "family": "claude-opus", "release_date": "2025-11-01",
                "open_weights": False,
            },
            "claude-opus-4-7": {
                "id": "claude-opus-4-7", "name": "Claude Opus 4.7",
                "family": "claude-opus", "open_weights": False,
            },
            # Mirror entry — must be skipped (multi-segment id)
            "anthropic/claude-opus-4-5": {
                "id": "anthropic/claude-opus-4-5", "name": "(mirror)", "family": "claude-opus",
            },
        },
    },
    # Inference provider — must be skipped entirely (not in PROVIDER_TO_ORG)
    "fireworks-ai": {
        "id": "fireworks-ai", "name": "Fireworks", "models": {
            "some-model": {"id": "some-model", "name": "X", "family": "x"},
        },
    },
}


KNOWN_ORGS = {"anthropic", "openai", "google", "meta", "deepseek"}


def test_family_for_uses_name_field_with_dots(mod):
    """The `name` field carries the lab's spelling (dots preserved); we
    prefer it over `id` to avoid models.dev's slug-mangled separators."""
    assert mod._family_for({
        "id": "qwen2-5-7b-instruct",
        "name": "Qwen2.5 7B Instruct",
    }) == "qwen2.5-7b"


def test_family_for_strips_date_suffix(mod):
    """Dated-snapshot ids fold via the date pattern (id-only, name has no date)."""
    assert mod._family_for({"id": "claude-opus-4-5-20251101"}) == "claude-opus-4-5"


def test_family_for_strips_latest_suffix(mod):
    assert mod._family_for({"id": "claude-3-5-haiku-latest"}) == "claude-3-5-haiku"


def test_family_for_strips_training_stage_suffixes(mod):
    """`-instruct`/`-chat`/`-it`/`-base` strip per the training-stage merge rule."""
    assert mod._family_for({"id": "qwen3-30b-instruct"}) == "qwen3-30b"
    assert mod._family_for({"id": "yi-34b-chat"}) == "yi-34b"
    assert mod._family_for({"id": "gemma-2-9b-it"}) == "gemma-2-9b"
    assert mod._family_for({"id": "olmo-7b-base"}) == "olmo-7b"


def test_family_for_passes_through_when_no_date(mod):
    assert mod._family_for({"id": "gpt-5-mini"}) == "gpt-5-mini"


def test_generate_filters_to_known_providers(mod):
    """fireworks-ai is not in PROVIDER_TO_ORG -> entire provider skipped."""
    out, _missing = mod._generate_models(SYNTHETIC_API, KNOWN_ORGS)
    org_ids = {e["org_id"] for e in out}
    assert "anthropic" in org_ids
    # No fireworks output even though it has models, because it's not a model author
    assert all(e["org_id"] != "fireworks-ai" for e in out)


def test_generate_skips_mirror_entries(mod):
    """Multi-segment ids like 'anthropic/claude-opus-4-5' are mirrors -> skip."""
    out, _missing = mod._generate_models(SYNTHETIC_API, KNOWN_ORGS)
    # All canonical ids should be of form `org_id/family-slug` (one slash)
    for e in out:
        assert e["id"].count("/") == 1, f"expected single slash, got {e['id']!r}"


def test_generate_collapses_dated_to_family(mod):
    """claude-opus-4-5 + claude-opus-4-5-20251101 collapse to same canonical.
    Slug uses dots (from `name` field) so id is `claude-opus-4.5`."""
    out, _missing = mod._generate_models(SYNTHETIC_API, KNOWN_ORGS)
    by_id = {e["id"]: e for e in out}
    entry = by_id["anthropic/claude-opus-4.5"]
    # The dated snapshot is recorded as an alias, not a separate entry
    assert "claude-opus-4-5-20251101" in entry["aliases"]


def test_generate_keeps_distinct_major_versions(mod):
    """claude-opus-4-5 and claude-opus-4-7 are separate canonicals."""
    out, _missing = mod._generate_models(SYNTHETIC_API, KNOWN_ORGS)
    ids = {e["id"] for e in out}
    assert "anthropic/claude-opus-4.5" in ids
    assert "anthropic/claude-opus-4.7" in ids


def test_generate_reports_missing_org_ids(mod):
    """When PROVIDER_TO_ORG references an org_id not in known_orgs, report it."""
    api = {
        "anthropic": {
            "id": "anthropic", "name": "Anthropic", "doc": "...", "env": [], "npm": "",
            "models": {"claude-x": {"id": "claude-x", "name": "X", "family": "claude"}},
        },
    }
    # Pass an empty known_orgs set — `anthropic` provider's org_id will be missing
    out, missing = mod._generate_models(api, set())
    assert out == []
    assert any("anthropic" in m for m in missing)


# Override-merge tests previously lived here. The refresh script no longer
# merges overrides into the generated YAML — the seed CLI loader
# (`_load_models_merged` in `eval_card_registry.cli`) applies
# `seed/models/core.yaml` and `seed/models/enrichments/aliases.yaml` at
# load time. Add merge coverage there if needed.
