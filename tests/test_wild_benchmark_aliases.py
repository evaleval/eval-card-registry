"""WILD benchmark slugs resolve to the right canonical (added aliases + new
canonicals). Guards the ARC disambiguation (AI2 Reasoning Challenge, not ARC-AGI).
Skips if fixtures aren't built (run `eval-card-registry seed --local`)."""
from pathlib import Path

import pytest
from eval_entity_resolver import Resolver

_ROOT = Path(__file__).resolve().parent.parent
_FIXTURES = _ROOT / "fixtures"

# WILD slug -> expected canonical id. Covers the 8 that previously failed:
# arc_easy/arc_challenge (alias -> AI2 ARC), race_h (alias -> race), and the five
# new canonicals (squad, paws, chembench, finance_fundamentals, pre_flight).
WILD_BENCHMARKS = {
    "arc_easy": "ai2-reasoning-challenge-arc",
    "arc_challenge": "ai2-reasoning-challenge-arc",
    "race_h": "race",
    "squad": "squad",
    "paws": "paws",
    "chembench": "chembench",
    "finance_fundamentals": "finance_fundamentals",
    "pre_flight": "pre_flight",
}

pytestmark = pytest.mark.skipif(
    not (_FIXTURES / "aliases.parquet").exists(),
    reason="fixtures not built; run `eval-card-registry seed --local`",
)


@pytest.fixture(scope="module")
def resolver():
    return Resolver.from_parquet(str(_FIXTURES))


@pytest.mark.parametrize("slug,expected", WILD_BENCHMARKS.items())
def test_wild_benchmark_resolves(resolver, slug, expected):
    res = resolver.resolve(slug, entity_type="benchmark")
    assert res.canonical_id == expected, f"{slug} -> {res.canonical_id} (expected {expected})"


def test_arc_not_arc_agi(resolver):
    # WILD's arc_* is the AI2 Reasoning Challenge, never ARC-AGI (Chollet) or the
    # generic 'arc' canonical.
    for slug in ("arc_easy", "arc_challenge"):
        cid = resolver.resolve(slug, entity_type="benchmark").canonical_id
        assert cid == "ai2-reasoning-challenge-arc"
        assert cid not in ("arc-agi", "arc")
