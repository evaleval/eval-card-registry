"""Dedup-vs-core guard at the hub-stats emit point.

OFFLINE. Drives the hub-stats refresh script's mint/guard FUNCTION
(`build_entry`) directly with synthetic rows + a synthetic curated-core
canonical — never invokes the destructive `main()` and touches no real seed
file or network.

Guarantee under test: no emitted id may collide (normalized) with a curated core
canonical under a DIFFERENT id. The curated id WINS id + casing; the generated
row becomes an alias-only enrichment that merges onto core, never a colliding
twin.

The guard in this BACKFILL-ONLY script is structural: `build_entry` returns None
unless the row's normalized canonical form already maps to an existing canonical
in the core-inclusive `aliases_to_canonical` index, and then emits under THAT
canonical's exact id (line `canonical_id = aliases_to_canonical[norm_canon]`). So
a different-casing twin defers to the curated id; a genuinely-novel HF id is
suppressed entirely. The script never rewrites an id, it only defers to /
enriches the existing one.
"""
from __future__ import annotations

import pytest

from conftest import load_script_module


@pytest.fixture
def mod():
    return load_script_module("refresh_from_hub_stats")


# A synthetic curated-core canonical: HF-true casing, decoupled developer org.
# Its NORMALIZED form is `foo-bar-7b` (via the script's `_normalize`, which
# collapses case + ./-/_/: to single dashes).
CORE_CANONICAL_ID = "Foo/Bar-7B"
CORE_ORG_ID = "foo"

# The core-inclusive existing-canonical index `build_entry` consults. Keyed by
# normalized form -> the curated canonical's EXACT id (this is exactly the shape
# `load_existing_canonical_aliases` produces, with the core id claiming its own
# normalized form first). Mirrors the real index built over _MODEL_SOURCES which
# includes CORE_PATH.
ALIASES_TO_CANONICAL = {"foo-bar-7b": CORE_CANONICAL_ID}

# Two-tier HF-org -> developer slug map (lowercase keys). `Foo` -> dev `foo`.
HF_TO_DEV = {"foo": CORE_ORG_ID}

ORG_ALIAS_MAP = {"foo": CORE_ORG_ID}


def _build(mod, row, **overrides):
    kw = dict(
        org_alias_map=ORG_ALIAS_MAP,
        aliases_to_canonical=ALIASES_TO_CANONICAL,
        hf_to_dev=HF_TO_DEV,
        canonical_id_norms={mod._normalize(CORE_CANONICAL_ID): CORE_CANONICAL_ID},
        ambiguous_canonicals=set(),
    )
    kw.update(overrides)
    return mod.build_entry(row, **kw)


def test_emit_defers_to_curated_core_id_never_normalized_colliding_twin(mod):
    """A hub-stats row for a DIFFERENT-casing twin of the curated core canonical
    (`foo/bar-7b` vs core `Foo/Bar-7B`) must NOT emit a colliding different id —
    it defers to the curated id + casing, emitting an enrichment that MERGES onto
    core. This is the dedup-vs-core guarantee."""
    # The parquet `id` carries the lowercase twin spelling.
    row = {"id": "foo/bar-7b"}

    entry = _build(mod, row)

    assert entry is not None, "a row whose normalized form matches core must enrich it"
    # HARD GUARANTEE: the emitted id is the CURATED core id (casing preserved),
    # NOT the lowercase twin `foo/bar-7b`.
    assert entry["id"] == CORE_CANONICAL_ID
    assert entry["id"] != "foo/bar-7b"
    # And it normalized-equals core under the SAME id (no different-id collision).
    assert mod._normalize(entry["id"]) == mod._normalize(CORE_CANONICAL_ID)


def test_emit_never_produces_a_different_id_with_colliding_norm(mod):
    """Sweep: across an exact-cased hit, a separator-drift twin, and a
    case-drift twin, EVERY emitted id whose normalized form collides with the
    curated core canonical resolves to the core id itself — never a twin."""
    core_norm = mod._normalize(CORE_CANONICAL_ID)
    for hf_id in ("Foo/Bar-7B", "foo/bar-7b", "foo/bar_7b", "FOO/BAR-7B"):
        entry = _build(mod, {"id": hf_id})
        assert entry is not None, f"{hf_id} should match core (normalized)"
        if mod._normalize(entry["id"]) == core_norm:
            assert entry["id"] == CORE_CANONICAL_ID, (
                f"row {hf_id!r} emitted id {entry['id']!r} which normalized-collides "
                f"with curated core {CORE_CANONICAL_ID!r} under a DIFFERENT id"
            )


def test_novel_hf_id_is_suppressed_not_minted_as_canonical(mod):
    """A hub-stats row whose normalized form is NOT a known canonical is
    suppressed (returns None) — the backfill-only script never mints a fresh
    canonical that could later collide with curated core. (No different-id twin
    can ever be created at this emit point.)"""
    entry = _build(mod, {"id": "brand-new-org/totally-novel-model-99b"})
    assert entry is None


def test_emitted_aliases_never_steal_a_different_core_canonical(mod):
    """The secondary emit point (aliases) is also core-aware: an emitted alias
    whose normalized form is owned by a DIFFERENT canonical is dropped via
    `_alias_ok`/`canonical_id_norms`, so no enrichment alias can intercept
    another core canonical's resolution."""
    # A second curated core canonical occupies the norm `other-thing-13b`.
    other_id = "Other/Thing-13B"
    aliases_to_canonical = dict(ALIASES_TO_CANONICAL)
    aliases_to_canonical[mod._normalize(other_id)] = other_id
    cid_norms = {
        mod._normalize(CORE_CANONICAL_ID): CORE_CANONICAL_ID,
        mod._normalize(other_id): other_id,
        # bare-name claim of the other canonical, as load_canonical_id_norms emits
        mod._normalize("Thing-13B"): other_id,
    }
    # Row resolves to Foo/Bar-7B but pretend its raw HF id collides in norm with
    # the OTHER canonical's bare name — _alias_ok must drop it.
    row = {"id": "foo/Thing-13B"}  # normalizes to foo-thing-13b for the canonical gate
    # Force the canonical gate to hit Foo/Bar-7B by aliasing this norm to it.
    aliases_to_canonical[mod._normalize("foo/Thing-13B")] = CORE_CANONICAL_ID

    entry = _build(
        mod, row,
        aliases_to_canonical=aliases_to_canonical,
        canonical_id_norms=cid_norms,
    )
    assert entry is not None
    assert entry["id"] == CORE_CANONICAL_ID
    # The raw HF id `foo/Thing-13B` must NOT be carried as an alias, because its
    # bare-name normalized form (`thing-13b`) is owned by the DIFFERENT canonical
    # Other/Thing-13B — that would steal its resolution.
    for a in entry.get("aliases", []):
        name_norm = mod._normalize(a.split("/", 1)[1]) if "/" in a else mod._normalize(a)
        assert cid_norms.get(name_norm, CORE_CANONICAL_ID) == CORE_CANONICAL_ID, (
            f"emitted alias {a!r} steals canonical {cid_norms.get(name_norm)!r}"
        )
