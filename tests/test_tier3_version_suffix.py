"""The Tier-3 inferred-seed generator must classify a dated/version SNAPSHOT of a
confirmed base as an identity-preserving `variant/version` edge (so it folds into
the base's `model_group_id`), not a blanket `finetune`.

Regression for the bug where every inferred child was stamped `finetune`, which
broke the model-group walk and surfaced each dated snapshot (e.g.
`gpt-5.4-pro-2026-03-05`) as a SEPARATE model page from its base `gpt-5.4-pro`.

OFFLINE + non-destructive: loads the script via importlib and exercises the pure
`_is_pure_version_suffix` helper directly. Never invokes `main()`.
"""
from __future__ import annotations

import pytest

from conftest import load_script_module


@pytest.fixture(scope="module")
def mod():
    return load_script_module("generate_tier3_inferred_seed")


@pytest.mark.parametrize(
    "tokens",
    [
        ["2026", "03", "05"],   # ISO date split on separators: gpt-5.4-pro-2026-03-05
        ["2024", "05", "13"],   # gpt-4o-2024-05-13
        ["20250929"],           # compact 8-digit date: claude-...-20250929
        ["2025", "08"],         # year-month snapshot
        ["202608"],             # compact year-month
        ["v0", "3"],            # slugified v0.3
        ["v2"],                 # bare vN
    ],
)
def test_pure_version_suffix_folds(mod, tokens):
    assert mod._is_pure_version_suffix(tokens) is True


@pytest.mark.parametrize(
    "tokens",
    [
        [],                       # no delta -> not a snapshot edge
        ["instruct"],             # training stage, not a version
        ["dpo"],                  # finetune marker
        ["thinking", "20250929"], # mode token present alongside a date
        ["mini"],                 # tier token
        ["turbo"],                # named variant
        ["8b"],                   # disclosed size
    ],
)
def test_non_version_suffix_stays_finetune(mod, tokens):
    assert mod._is_pure_version_suffix(tokens) is False
