"""Unit tests for the publish script's transient-error backoff.

The HF publish job intermittently hits 429 (rate limit) when several publish
runs hit the Hub at once; huggingface_hub's own retry caps at 8s and gives up.
_with_backoff adds a longer outer retry that distinguishes transient errors
(429 / 5xx — worth waiting out) from permanent ones (auth / not-found).
"""
from __future__ import annotations

import sys

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError

from conftest import load_script_module


@pytest.fixture(scope="module")
def mod():
    return load_script_module("publish_registry_data")


def _http_error(status_code: int) -> HfHubHTTPError:
    resp = httpx.Response(status_code, request=httpx.Request("GET", "http://hf.test"))
    return HfHubHTTPError("boom", response=resp)


def test_is_retryable_only_for_rate_limit_and_5xx(mod):
    assert mod._is_retryable_http(_http_error(429)) is True
    assert mod._is_retryable_http(_http_error(503)) is True
    assert mod._is_retryable_http(_http_error(500)) is True
    assert mod._is_retryable_http(_http_error(403)) is False  # auth — won't heal
    assert mod._is_retryable_http(_http_error(404)) is False  # not found
    assert mod._is_retryable_http(ValueError("nope")) is False  # no response attr


def test_retries_on_429_then_succeeds(mod):
    waits: list[float] = []
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _http_error(429)
        return "ok"

    out = mod._with_backoff(fn, what="test", base_seconds=15.0, sleep=waits.append)
    assert out == "ok"
    assert calls["n"] == 3
    assert waits == [15.0, 30.0]  # exponential backoff before attempts 2 and 3


def test_does_not_retry_on_auth_error(mod):
    waits: list[float] = []
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise _http_error(403)

    with pytest.raises(HfHubHTTPError):
        mod._with_backoff(fn, what="test", sleep=waits.append)
    assert calls["n"] == 1  # raised immediately, no retry
    assert waits == []


def test_exhausts_attempts_then_raises(mod):
    waits: list[float] = []
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise _http_error(429)

    with pytest.raises(HfHubHTTPError):
        mod._with_backoff(fn, what="test", attempts=4, base_seconds=15.0, sleep=waits.append)
    assert calls["n"] == 4  # all attempts used
    assert waits == [15.0, 30.0, 60.0]  # slept before attempts 2, 3, 4; not after the last


def test_production_freshness_guard_accepts_current_main(mod, monkeypatch):
    monkeypatch.setattr(mod, "_git_sha", lambda: "abc123")
    monkeypatch.setattr(mod, "_origin_main_sha", lambda: "abc123")

    mod._assert_checkout_is_current_main()


def test_production_freshness_guard_rejects_stale_or_unverifiable(
    mod, monkeypatch
):
    monkeypatch.setattr(mod, "_git_sha", lambda: "old123")
    monkeypatch.setattr(mod, "_origin_main_sha", lambda: "new456")
    with pytest.raises(RuntimeError, match="stale checkout"):
        mod._assert_checkout_is_current_main()

    monkeypatch.setattr(mod, "_origin_main_sha", lambda: None)
    with pytest.raises(RuntimeError, match="cannot verify"):
        mod._assert_checkout_is_current_main()


def test_expected_head_guard_accepts_matching_checkout(mod, monkeypatch):
    monkeypatch.setattr(mod, "_git_sha", lambda: "a" * 40)

    mod._assert_checkout_is_expected("a" * 40)


def test_expected_head_guard_rejects_mismatch_or_unknown_head(mod, monkeypatch):
    monkeypatch.setattr(mod, "_git_sha", lambda: "b" * 40)
    with pytest.raises(RuntimeError, match="is not the expected head"):
        mod._assert_checkout_is_expected("a" * 40)

    monkeypatch.setattr(mod, "_git_sha", lambda: None)
    with pytest.raises(RuntimeError, match="cannot read checkout HEAD"):
        mod._assert_checkout_is_expected("a" * 40)


def _main_with_argv(mod, monkeypatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["publish_registry_data.py", *argv])
    return mod.main()


def test_cli_rejects_malformed_expected_head(mod, monkeypatch):
    with pytest.raises(SystemExit) as exc:
        _main_with_argv(
            mod,
            monkeypatch,
            "--skip-seed",
            "--require-origin-main-head",
            "--expected-head",
            "not-a-sha",
        )
    assert exc.value.code == 2


def test_cli_rejects_expected_head_without_production_guard(mod, monkeypatch):
    with pytest.raises(SystemExit) as exc:
        _main_with_argv(
            mod, monkeypatch, "--skip-seed", "--expected-head", "a" * 40
        )
    assert exc.value.code == 2
