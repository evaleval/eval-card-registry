"""Tests for the models.dev refresh script — synthetic data, no network."""
import json

import pytest

from conftest import load_script_module


@pytest.fixture
def mod():
    return load_script_module("refresh_from_modelsdev")


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
    """Multi-segment mirror ids like 'anthropic/claude-opus-4-5' never produce a
    two-slash canonical. After the full-catalog refactor, a re-host/closed-API
    model with no extractable org (fireworks-ai's `some-model`) IS now minted as
    a bare org-less slug (flagged `org-unknown`) — only true `org/a/b` mirrors
    are dropped."""
    out, _missing = mod._generate_models(SYNTHETIC_API, KNOWN_ORGS)
    for e in out:
        assert e["id"].count("/") <= 1, f"expected <=1 slash, got {e['id']!r}"
    by_id = {e["id"]: e for e in out}
    # fireworks-ai's some-model is now seeded org-less (full-catalog seed).
    assert "some-model" in by_id
    assert by_id["some-model"]["org_id"] is None
    assert "org-unknown" in by_id["some-model"]["tags"]


def test_generate_dated_snapshot_becomes_child_canonical(mod):
    """After the parents-shape refactor, dated snapshots are emitted as
    separate canonical entries with a typed `parents` edge back to the
    family root (axis: version) — not collapsed into the root's aliases.

    The leaf canonical id inherits the family's dotted spelling (from the
    lab's `name` field), and the original dashed models.dev id stays on
    the leaf as an alias so the resolver still maps it via exact match."""
    out, _missing = mod._generate_models(SYNTHETIC_API, KNOWN_ORGS)
    by_id = {e["id"]: e for e in out}
    root = by_id["anthropic/claude-opus-4.5"]
    assert root["parents"] == []
    # Leaf canonical: dotted base + dashed snapshot token
    snap_id = "anthropic/claude-opus-4.5-20251101"
    assert snap_id in by_id, f"missing snapshot canonical {snap_id!r} in output"
    snap = by_id[snap_id]
    assert snap["parents"] == [{
        "id": "anthropic/claude-opus-4.5",
        "relationship": "variant",
        "axis": "version",
    }]
    assert snap["release_date"] == "2025-11-01"
    # Original models.dev id (dashed) survives as an alias for resolver coverage
    assert "claude-opus-4-5-20251101" in snap["aliases"]
    assert "anthropic/claude-opus-4-5-20251101" in snap["aliases"]
    # Snapshot id is in family root's metadata.snapshots (still a useful index).
    assert "claude-opus-4-5-20251101" in json.loads(root["metadata"]).get("snapshots", [])


def test_generate_emits_mode_variants_as_children(mod):
    """A dated -instruct snapshot in models.dev produces a multi-level chain:
    family root, instruct intermediate (variant/training_stage), and the dated
    leaf (variant/version of the instruct intermediate). Mirrors the
    post-promotion shape of curated core.yaml so the loader's parents-merge
    agrees on edges. `-instruct` is a TRAINING STAGE axis, not a runtime
    mode."""
    api = {
        "mistral": {
            "id": "mistral", "name": "Mistral", "models": {
                "mistral-7b-instruct-v0-3": {
                    "id": "mistral-7b-instruct-v0-3",
                    "name": "Mistral 7B Instruct v0.3",
                    "release_date": "2024-05-22",
                    "open_weights": True,
                },
            },
        },
    }
    known = {"mistralai"}
    out, _missing = mod._generate_models(api, known)
    by_id = {e["id"]: e for e in out}
    # All three levels emitted:
    assert "mistralai/mistral-7b" in by_id, "family root missing"
    assert "mistralai/mistral-7b-instruct" in by_id, "instruct intermediate missing"
    assert "mistralai/mistral-7b-instruct-v0-3" in by_id, "leaf snapshot missing"
    # Edges chain correctly:
    assert by_id["mistralai/mistral-7b"]["parents"] == []
    assert by_id["mistralai/mistral-7b-instruct"]["parents"] == [{
        "id": "mistralai/mistral-7b", "relationship": "variant", "axis": "training_stage",
    }]
    assert by_id["mistralai/mistral-7b-instruct-v0-3"]["parents"] == [{
        "id": "mistralai/mistral-7b-instruct", "relationship": "variant", "axis": "version",
    }]
    # Leaf carries the source release_date; intermediate is anchor-only.
    assert by_id["mistralai/mistral-7b-instruct-v0-3"]["release_date"] == "2024-05-22"
    assert by_id["mistralai/mistral-7b-instruct"]["release_date"] is None


def test_generate_keeps_distinct_major_versions(mod):
    """claude-opus-4-5 and claude-opus-4-7 are separate canonicals."""
    out, _missing = mod._generate_models(SYNTHETIC_API, KNOWN_ORGS)
    ids = {e["id"] for e in out}
    assert "anthropic/claude-opus-4.5" in ids
    assert "anthropic/claude-opus-4.7" in ids


def test_generate_propagates_open_weights_flag(mod):
    """`open_weights` must round-trip from models.dev to the emitted entries.
    Closed-API models stay False; open-weight models become True; NULL when
    the source omits the field. Family root aggregates with `any()` so
    a single open snapshot in a mixed family marks the root open."""
    api = {
        "anthropic": {
            "id": "anthropic", "name": "Anthropic", "models": {
                "claude-foo-1": {
                    "id": "claude-foo-1", "name": "Claude Foo 1",
                    "release_date": "2025-06-15", "open_weights": False,
                },
            },
        },
        "mistral": {
            "id": "mistral", "name": "Mistral", "models": {
                "mistral-7b-instruct-v0-3": {
                    "id": "mistral-7b-instruct-v0-3",
                    "name": "Mistral 7B Instruct v0.3",
                    "release_date": "2024-05-22", "open_weights": True,
                },
            },
        },
    }
    out, _missing = mod._generate_models(api, {"anthropic", "mistralai"})
    by_id = {e["id"]: e for e in out}
    # Closed-API: family root + child both False
    assert by_id["anthropic/claude-foo-1"]["open_weights"] is False
    # Open-weight: every level of the chain inherits True
    assert by_id["mistralai/mistral-7b"]["open_weights"] is True
    assert by_id["mistralai/mistral-7b-instruct"]["open_weights"] is True
    assert by_id["mistralai/mistral-7b-instruct-v0-3"]["open_weights"] is True


def test_generate_promotes_earliest_release_date(mod):
    """`release_date` is a first-class field — earliest snapshot date for the
    family. Models without any release_date in the source get None."""
    api = {
        "anthropic": {
            "id": "anthropic", "name": "Anthropic", "models": {
                "claude-foo-1": {
                    "id": "claude-foo-1", "name": "Claude Foo 1",
                    "release_date": "2025-06-15",
                },
                "claude-foo-1-20250901": {
                    "id": "claude-foo-1-20250901", "name": "Claude Foo 1",
                    "release_date": "2025-09-01",
                },
                "claude-bar-1": {
                    "id": "claude-bar-1", "name": "Claude Bar 1",
                    # no release_date
                },
            },
        },
    }
    out, _missing = mod._generate_models(api, KNOWN_ORGS)
    by_id = {e["id"]: e for e in out}
    assert by_id["anthropic/claude-foo-1"]["release_date"] == "2025-06-15"
    assert by_id["anthropic/claude-bar-1"]["release_date"] is None


def test_tee_prefix_is_serving_marker_for_org_derivation(mod):
    """A `TEE/`-prefixed id (nano-gpt's trusted-execution namespace) is a
    serving marker, not an uploader org: the developer comes from the name."""
    org, rehost = mod._derive_group_org(
        [{"raw": "TEE/gemma9x-31b", "name": "Gemma 9X 31B"}], mod._dev_alias_index()
    )
    assert org == "google"
    assert rehost is None


def test_provider_alias_forms_keep_tee_prefixed_raw(mod):
    """The raw `TEE/...` spelling stays resolvable as an alias on whatever the
    stripped name resolves to (fold-keeps-alias)."""
    forms = mod._provider_alias_forms("TEE/gemma9x-31b", "google")
    assert "TEE/gemma9x-31b" in forms
    assert "gemma9x-31b" in forms
    assert "google/gemma9x-31b" in forms


def test_generate_tee_group_mints_under_name_vendor_with_tee_alias(mod):
    """End-to-end TEE strip: a TEE-only group mints via the bare-name path
    (developer from the leading name token), never under org `TEE`, and the
    raw TEE/ spellings survive as aliases."""
    api = {
        "nano-gpt": {
            "id": "nano-gpt", "name": "NanoGPT", "models": {
                "TEE/gemma9x-31b": {
                    "id": "TEE/gemma9x-31b", "name": "Gemma 9X 31B",
                    "release_date": "2026-04-04", "open_weights": False,
                },
                "TEE/gemma9x-26b-uncensored": {
                    "id": "TEE/gemma9x-26b-uncensored",
                    "name": "Gemma 9X 26B Uncensored TEE", "open_weights": False,
                },
            },
        },
    }
    out, _missing = mod._generate_models(api, KNOWN_ORGS)
    out = mod._finalize_entries(out)
    assert all(not e["id"].startswith("TEE/") for e in out)
    assert all(e.get("org_id") != "TEE" for e in out)
    by_id = {e["id"]: e for e in out}
    assert "TEE/gemma9x-31b" in by_id["google/gemma9x-31b"]["aliases"]
    assert "TEE/gemma9x-26b-uncensored" in by_id["google/gemma9x-26b-uncensored"]["aliases"]


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


# The refresh script does NOT merge overrides into the generated YAML: the seed
# CLI loader (`_load_models_merged` in `eval_card_registry.cli`) applies
# `seed/models/core.yaml` and `seed/models/enrichments/aliases.yaml` at load time.
# Override-merge coverage belongs against that loader, not this script.
