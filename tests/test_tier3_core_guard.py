"""Phase-1 regression test: the Tier-3 inferred-seed generator's mint/guard
logic must NOT emit a canonical id that normalized-collides with a curated
`core.yaml` canonical under a DIFFERENT id.

OFFLINE + non-destructive: loads the script via importlib (like
tests/test_refresh_modelsdev_dedup.py) and exercises the guard FUNCTIONS
directly (`build_core_norm_index` / `core_steals`). It never invokes the
script's `main()`, never touches fixtures, and never runs the network.

The guarantee under test mirrors `refresh_from_modelsdev`'s `regenerate_catalog`
steal-guard: HF/curated WIN id + casing; a generated row that would collide
(normalized, via the resolver `normalize`) with a curated core canonical under a
different id must defer to the curated id (suffix-disambiguate), never mint a
colliding twin.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from eval_entity_resolver.normalization import normalize as _rnz

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    path = REPO_ROOT / "scripts" / "generate_tier3_inferred_seed.py"
    spec = importlib.util.spec_from_file_location("generate_tier3_inferred_seed", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_module()


# A synthetic curated-core doc in the `{skip_ids, entries}` shape, with one
# curated canonical whose real HF-cased id is `Foo/Bar-7B`.
CORE_DOC = {
    "skip_ids": [],
    "skip_source_ids": [],
    "entries": [
        {
            "id": "Foo/Bar-7B",
            "display_name": "Foo Bar 7B",
            "aliases": ["foo-bar-7b", "Foo/Bar-7B"],
            # Guardrail §5: this curated id IS its own real HF repo.
            "metadata": '{"hf_id": "Foo/Bar-7B"}',
        }
    ],
}


def test_build_core_norm_index_maps_forms_to_curated_owner(mod):
    idx = mod.build_core_norm_index(CORE_DOC)
    owner = "Foo/Bar-7B"
    # id, display_name, and aliases all map to the curated owner under their
    # normalized form (case + separators incl. `/` collapsed).
    assert idx[_rnz("Foo/Bar-7B")] == owner
    assert idx[_rnz("foo/bar-7b")] == owner       # lowercase twin
    assert idx[_rnz("Foo Bar 7B")] == owner       # display_name
    assert idx[_rnz("foo-bar-7b")] == owner       # alias


def test_build_core_norm_index_handles_flat_list_shape(mod):
    """core.yaml may be a flat list rather than the {entries:...} dict."""
    flat = [{"id": "Baz/Qux-3B", "aliases": []}]
    idx = mod.build_core_norm_index(flat)
    assert idx[_rnz("baz/qux-3b")] == "Baz/Qux-3B"


def test_core_steals_flags_normalized_twin_under_different_id(mod):
    idx = mod.build_core_norm_index(CORE_DOC)
    # A tier-3 mint that is a normalized twin of the curated id under a DIFFERENT
    # id (lowercase / re-separated) is a steal -> must defer.
    assert mod.core_steals("foo/bar-7b", idx) is True
    assert mod.core_steals("Foo_Bar_7B", idx) is True
    assert mod.core_steals("foo-bar-7b", idx) is True


def test_core_steals_allows_curated_id_itself_and_novel_mints(mod):
    idx = mod.build_core_norm_index(CORE_DOC)
    # The curated id is its OWN owner -> not a steal (same id).
    assert mod.core_steals("Foo/Bar-7B", idx) is False
    # A genuinely novel mint that does not normalize-collide -> not a steal.
    assert mod.core_steals("openai/gpt-4o", idx) is False
    assert mod.core_steals("acme/widget-13b", idx) is False


def test_core_collision_is_SKIPPED_not_minted_as_inferred_twin(mod):
    """The PINNED guarantee (§5: merge, never dup): a residual raw whose minted
    id normalized-collides with a curated core canonical is the SAME model as
    that curated entry — the policy must SKIP it (drop the row so the raw
    resolves to the curated canonical), NOT mint a `{cid}-inferred` duplicate
    that splits one model into two and shadows the curated fix.

    Tests the extracted `mint_collision_decision` policy directly."""
    idx = mod.build_core_norm_index(CORE_DOC)

    raw = "foo/bar-7b"  # lowercase twin of curated Foo/Bar-7B
    cid, _org = mod.canon_id_for_org_present(raw, {})
    assert cid == "foo/bar-7b"  # naive mint would be the colliding twin
    assert mod.core_steals(cid, idx) is True

    # The decision is SKIP — NOT 'inferred' (the old band-aid) and NOT 'mint'.
    decision = mod.mint_collision_decision(
        cid, core_skip_ids=set(), core_norm_index=idx, resolver_hit=False
    )
    assert decision == "skip", (
        "a same-model core collision must be skipped (raw resolves to curated), "
        "never minted as a -inferred twin"
    )


def test_genuine_cross_entity_clash_is_disambiguated_not_skipped(mod):
    """A mint that does NOT core-collide but clashes with a core skip_ids entry
    or a DIFFERENT existing canonical (resolver_hit) is a genuine cross-entity
    clash → 'inferred' (suffix-disambiguate), never silently dropped."""
    idx = mod.build_core_norm_index(CORE_DOC)
    novel = "acme/widget-13b"  # does not core-steal
    assert mod.core_steals(novel, idx) is False

    # core skip_ids clash -> inferred
    assert mod.mint_collision_decision(novel, {novel}, idx, resolver_hit=False) == "inferred"
    # different existing canonical (resolver hit) -> inferred
    assert mod.mint_collision_decision(novel, set(), idx, resolver_hit=True) == "inferred"
    # no collision at all -> mint
    assert mod.mint_collision_decision(novel, set(), idx, resolver_hit=False) == "mint"


def test_guard_never_rewrites_id_equal_to_its_hf_id(mod):
    """Guardrail §5: the guard never rewrites an id whose metadata.hf_id equals
    that id. The curated core id `Foo/Bar-7B` (hf_id == id) is the WIN target —
    `core_steals` returns False for it, so it is never suffix-disambiguated."""
    idx = mod.build_core_norm_index(CORE_DOC)
    assert mod.core_steals("Foo/Bar-7B", idx) is False
