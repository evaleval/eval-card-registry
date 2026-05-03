"""Tests for the hub-stats refresh script — synthetic rows, no network."""
from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from pathlib import Path

import pytest


def _load_module():
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "scripts" / "refresh_from_hub_stats.py"
    spec = importlib.util.spec_from_file_location("refresh_from_hub_stats", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def mod():
    return _load_module()


# Mini fixtures mimicking the registry's shape — bypass loading from disk
# so the tests don't depend on the live seed contents.
ORG_ALIAS_MAP = {
    "meta": "meta",
    "meta-llama": "meta",
    "alibaba": "alibaba",
    "qwen": "alibaba",
    "anthropic": "anthropic",
    "huggingface": "huggingface",
    "huggingfaceh4": "huggingface",
    "nous-research": "nous-research",
    "nousresearch": "nous-research",
}

# Existing canonicals our script knows about. Keys are *normalized* forms
# (the `_normalize` helper collapses ./-/_/:/ all to single dashes).
ALIASES_TO_CANONICAL = {
    "meta-llama-3-1-70b": "meta/llama-3.1-70b",
    "meta-llama-llama-3-1-70b": "meta/llama-3.1-70b",
    "meta-llama-3-1-70b-instruct": "meta/llama-3.1-70b-instruct",
    "alibaba-qwen2-5-7b": "alibaba/qwen2.5-7b",
    "qwen-qwen2-5-7b": "alibaba/qwen2.5-7b",
}


def test_hf_id_to_canonical_maps_known_org_alias(mod):
    """`meta-llama` is the HF org for Meta — should map to canonical `meta`."""
    cid, org = mod.hf_id_to_canonical("meta-llama/Llama-3.1-70B", ORG_ALIAS_MAP)
    assert cid == "meta/llama-3.1-70b"
    assert org == "meta"


def test_hf_id_to_canonical_unknown_org_keeps_slug(mod):
    """An author we don't have in seed/orgs.yaml stays as its own org id."""
    cid, org = mod.hf_id_to_canonical("random-uploader/some-model", ORG_ALIAS_MAP)
    assert cid == "random-uploader/some-model"
    assert org == "random-uploader"


def test_hf_id_to_canonical_no_slash_uses_unknown_placeholder(mod):
    """No org prefix in the id → `unknown/...` placeholder."""
    cid, org = mod.hf_id_to_canonical("standalone-model-name", ORG_ALIAS_MAP)
    assert cid == "unknown/standalone-model-name"
    assert org == "unknown"


def test_coerce_date_handles_datetime_str_none(mod):
    assert mod._coerce_date(datetime(2025, 11, 1, 10, 30)) == "2025-11-01"
    assert mod._coerce_date("2025-11-01T10:30:00") == "2025-11-01"
    assert mod._coerce_date(None) is None


def test_extract_license_from_dict(mod):
    assert mod._extract_license({"license": "apache-2.0", "tags": []}) == "apache-2.0"


def test_extract_license_from_json_string(mod):
    assert mod._extract_license('{"license": "mit"}') == "mit"


def test_extract_license_returns_none_when_missing(mod):
    assert mod._extract_license({"tags": []}) is None
    assert mod._extract_license(None) is None


def test_filter_useful_tags(mod):
    tags = [
        "transformers", "safetensors", "deepseek_v4", "text-generation",
        "conversational", "license:mit", "eval-results", "endpoints_compatible",
        "8-bit", "fp8", "region:us", "en", "zh",
    ]
    out = mod._filter_useful_tags(tags)
    # Useful: license:mit, eval-results, safetensors (format), en, zh
    assert "license:mit" in out
    assert "eval-results" in out
    assert "safetensors" in out
    assert "en" in out
    assert "zh" in out
    # Filtered out: noisy library/region markers
    assert "transformers" not in out
    assert "region:us" not in out
    assert "conversational" not in out


def test_build_entry_job_a_existing_canonical(mod):
    """Job A: hub-stats id matches an existing canonical → emit
    enrichment entry (no parents set, even if hub-stats has baseModels —
    curated parents in core.yaml stay authoritative)."""
    row = {
        "id": "meta-llama/Llama-3.1-70B",
        "author": "meta-llama",
        "createdAt": datetime(2024, 7, 14),
        "tags": ["text-generation", "eval-results", "en"],
        "cardData": {"license": "llama3.1", "library_name": "transformers"},
        "safetensors": {"total": 141_000_000_000},  # ~70.5B params at BF16
        "baseModels": None,
        "library_name": "transformers",
        "pipeline_tag": "text-generation",
        "downloadsAllTime": 5_000_000,
        "likes": 12_345,
    }
    e = mod.build_entry(row, ORG_ALIAS_MAP, ALIASES_TO_CANONICAL)
    assert e is not None
    assert e["id"] == "meta/llama-3.1-70b"
    assert e["org_id"] == "meta"
    assert e["release_date"] == "2024-07-14"
    assert e["params_billions"] == 70.5  # bytes / 2 / 1e9
    assert "parents" not in e, "Job A enrichment must not set parents"
    assert "eval-results" in e["tags"]
    metadata = json.loads(e["metadata"])
    assert metadata["source"] == "hub_stats"
    assert metadata["license"] == "llama3.1"


def test_build_entry_skips_unknown_hf_ids(mod):
    """Backfill-only: hub-stats rows whose HF id isn't already an alias
    on one of our canonicals are skipped. The lineage-descendant
    pre-load (community quants/finetunes) is deferred."""
    row = {
        "id": "casperhansen/llama-3.3-70b-instruct-awq",  # not in our aliases
        "author": "casperhansen",
        "createdAt": datetime(2024, 12, 6),
        "tags": ["safetensors"],
        "cardData": None,
        "safetensors": None,
        "library_name": None,
        "pipeline_tag": None,
        "downloadsAllTime": None,
        "likes": None,
    }
    assert mod.build_entry(row, ORG_ALIAS_MAP, ALIASES_TO_CANONICAL) is None


def test_build_entry_uses_registry_canonical_id_not_slugified(mod):
    """When the registry's canonical id keeps dots (`alibaba/qwen2.5-7b`)
    but the HF id has dashes only (`Qwen/Qwen2.5-7B`), the emitted entry
    must use the registry's spelling so the seed merge lands on the
    right row."""
    row = {
        "id": "Qwen/Qwen2.5-7B",
        "author": "Qwen",
        "createdAt": datetime(2024, 9, 16),
        "tags": ["en"],
        "cardData": {"license": "apache-2.0"},
        "safetensors": None,
        "library_name": "transformers",
        "pipeline_tag": "text-generation",
        "downloadsAllTime": 1000,
        "likes": 50,
    }
    e = mod.build_entry(row, ORG_ALIAS_MAP, ALIASES_TO_CANONICAL)
    assert e is not None
    # Note: registry canonical uses the dotted form (`qwen2.5-7b`)
    assert e["id"] == "alibaba/qwen2.5-7b"
    assert e["org_id"] == "alibaba"
    # Original dashed HF id stays as alias for resolver coverage
    assert "Qwen/Qwen2.5-7B" in e["aliases"]


def test_approx_params_billions_from_safetensors_total(mod):
    """Param count estimated from total bytes / 2 (BF16 default)."""
    assert mod._approx_params_billions({"total": 16_000_000_000}) == 8.0
    assert mod._approx_params_billions({"total": 0}) is None
    assert mod._approx_params_billions(None) is None


def test_build_entry_sets_open_weights_when_safetensors_present(mod):
    """A row with safetensors data is downloadable HF weights → open."""
    row = {
        "id": "Qwen/Qwen2.5-7B",
        "author": "Qwen",
        "createdAt": datetime(2024, 9, 16),
        "tags": ["en"],
        "cardData": None,
        "safetensors": {"total": 14_000_000_000},
        "library_name": "transformers",
        "pipeline_tag": "text-generation",
        "downloadsAllTime": None, "likes": None,
    }
    e = mod.build_entry(row, ORG_ALIAS_MAP, ALIASES_TO_CANONICAL)
    assert e is not None
    assert e["open_weights"] is True


def test_build_entry_skips_open_weights_when_no_artifacts(mod):
    """A row with neither safetensors nor gguf data → don't infer
    open_weights (could be a metadata-only repo, deleted weights, etc.).
    Leaving it absent lets the seed loader default to NA."""
    row = {
        "id": "Qwen/Qwen2.5-7B",
        "author": "Qwen",
        "createdAt": datetime(2024, 9, 16),
        "tags": [],
        "cardData": None,
        "safetensors": None,
        "library_name": None, "pipeline_tag": None,
        "downloadsAllTime": None, "likes": None,
    }
    e = mod.build_entry(row, ORG_ALIAS_MAP, ALIASES_TO_CANONICAL)
    assert e is not None
    assert "open_weights" not in e
