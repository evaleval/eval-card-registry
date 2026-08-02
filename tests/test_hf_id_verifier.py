"""Unit tests for HfIdVerifier — the resolver's injected HF id checker.

All live-tier tests mock the Hub API client; nothing here touches the
network. The autouse conftest fixture disables the live flag, so tests that
exercise the live tier re-enable it explicitly."""
from unittest.mock import MagicMock, patch

import pytest

from eval_card_registry.config import settings
from eval_card_registry.services.hf_id_verifier import HfIdVerifier


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _verifier(index=None, **kwargs) -> tuple[HfIdVerifier, FakeClock]:
    clock = FakeClock()
    v = HfIdVerifier(index_provider=lambda: index or {}, clock=clock, **kwargs)
    return v, clock


class _FakeRepoNotFound(Exception):
    pass


class _FakeGated(Exception):
    pass


def _patch_live(verifier, side_effect=None, return_id=None):
    """Install a fake HfApi on the verifier and route the error classes the
    live path imports to fakes we can raise."""
    api = MagicMock()
    if side_effect is not None:
        api.model_info.side_effect = side_effect
    else:
        info = MagicMock()
        info.id = return_id
        api.model_info.return_value = info
    verifier._hf_api = api
    errors = MagicMock()
    errors.RepositoryNotFoundError = _FakeRepoNotFound
    errors.GatedRepoError = _FakeGated
    patches = patch.dict(
        "sys.modules",
        {"huggingface_hub.errors": errors},
    )
    return api, patches


@pytest.fixture
def live_enabled():
    original = settings.hf_live_id_check_enabled
    settings.hf_live_id_check_enabled = True
    yield
    settings.hf_live_id_check_enabled = original


def test_non_hf_shaped_returns_none():
    v, _ = _verifier(index={"someindex": "Some/Index"})
    assert v.check("gpt-4o") is None
    assert v.check("a/b/c") is None
    assert v.check("/x") is None


def test_index_verbatim_hit():
    v, _ = _verifier(index={"qwen-qwen2-5-7b": "Qwen/Qwen2.5-7B"})
    hit = v.check("qwen/qwen2.5-7b")
    assert hit is not None
    assert hit.hf_id == "Qwen/Qwen2.5-7B"
    assert hit.verbatim is True
    assert hit.source == "hub_stats_index"


def test_index_normalized_hit_is_not_verbatim():
    v, _ = _verifier(index={"qwen-qwen2-5-7b": "Qwen/Qwen2.5-7B"})
    hit = v.check("qwen/qwen2_5_7b")
    assert hit is not None
    assert hit.verbatim is False


def test_live_disabled_and_index_miss_returns_none():
    v, _ = _verifier(index={})
    assert v.check("org/unknown-model") is None


def test_live_success_caches_positive(live_enabled):
    v, clock = _verifier(index={})
    api, patches = _patch_live(v, return_id="NousResearch/Hermes-4-70B")
    with patches:
        hit = v.check("nousresearch/hermes-4-70b")
    assert hit is not None
    assert hit.hf_id == "NousResearch/Hermes-4-70B"
    assert hit.verbatim is True
    assert hit.source == "hf_live"
    assert api.model_info.call_count == 1
    # Second call answers from cache — no new HTTP call.
    with patches:
        hit2 = v.check("nousresearch/hermes-4-70b")
    assert hit2 is not None and hit2.hf_id == "NousResearch/Hermes-4-70B"
    assert api.model_info.call_count == 1


def test_live_404_negative_cached_with_ttl(live_enabled):
    v, clock = _verifier(index={})
    api, patches = _patch_live(v, side_effect=_FakeRepoNotFound())
    with patches:
        assert v.check("org/nope") is None
        assert v.check("org/nope") is None
    assert api.model_info.call_count == 1
    # After the negative TTL, the live path is retried.
    clock.now += v._negative_ttl + 1
    with patches:
        v.check("org/nope")
    assert api.model_info.call_count == 2


def test_transport_error_not_cached_and_breaker_opens(live_enabled):
    v, clock = _verifier(index={}, breaker_threshold=3, breaker_cooldown=100.0)
    api, patches = _patch_live(v, side_effect=RuntimeError("boom"))
    with patches:
        for i in range(3):
            assert v.check(f"org/m{i}") is None
    assert api.model_info.call_count == 3
    # Breaker is open — no further calls.
    with patches:
        assert v.check("org/m9") is None
    assert api.model_info.call_count == 3
    # After cooldown the live path closes again.
    clock.now += 101.0
    with patches:
        v.check("org/m9")
    assert api.model_info.call_count == 4


def test_budget_exhaustion_answers_unknown_without_caching(live_enabled):
    v, clock = _verifier(index={}, budget_per_minute=2)
    api, patches = _patch_live(v, return_id="Org/M")
    with patches:
        assert v.check("org/m1") is not None
        assert v.check("org/m2") is not None
        # Budget spent — this answers unknown, uncached.
        assert v.check("org/m3") is None
    assert api.model_info.call_count == 2
    # A minute later the bucket refills.
    clock.now += 61.0
    with patches:
        assert v.check("org/m3") is not None
    assert api.model_info.call_count == 3


def test_single_flight_duplicate_answers_unknown(live_enabled):
    v, _ = _verifier(index={})
    v._in_flight.add("org/dup")
    api, patches = _patch_live(v, return_id="Org/Dup")
    with patches:
        assert v.check("org/dup") is None
    assert api.model_info.call_count == 0


def test_gated_repo_counts_as_existing(live_enabled):
    v, _ = _verifier(index={})
    api, patches = _patch_live(v, side_effect=_FakeGated())
    with patches:
        hit = v.check("meta-llama/Llama-Gated")
    assert hit is not None
    assert hit.hf_id == "meta-llama/Llama-Gated"
    assert hit.source == "hf_live"


def test_index_error_degrades_to_live_or_none():
    def boom():
        raise RuntimeError("index unavailable")

    clock = FakeClock()
    v = HfIdVerifier(index_provider=boom, clock=clock)
    assert v.check("org/model") is None
