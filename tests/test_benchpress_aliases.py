"""BenchPress score-matrix slug aliases resolve to a canonical model.

These 21 raw slugs (from microsoft/benchpress-score-matrix) resolved to no_match
before the aliases in seed/models/enrichments/aliases.yaml were added. This guards
that each now attaches to a canonical entity. Skips if fixtures aren't built
(run `eval-card-registry seed --local` first).
"""
from pathlib import Path

import pytest
from eval_entity_resolver import Resolver

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FIXTURES = _REPO_ROOT / "fixtures"

# Raw BenchPress slug -> the HF repo id it should map to (each verified HTTP 200).
BENCHPRESS_SLUGS = {
    "deepseek-3.1": "deepseek-ai/DeepSeek-V3.1",
    "deepseek-v2.5-0905": "deepseek-ai/DeepSeek-V2.5",
    "exaone-deep-32b": "LGAI-EXAONE/EXAONE-Deep-32B",
    "gemma-3n-e2b-instructed": "google/gemma-3n-E2B-it",
    "gemma-3n-e2b-instructed-litert-preview": "google/gemma-3n-E2B-it-litert-preview",
    "gemma-3n-e4b-instructed": "google/gemma-3n-E4B-it",
    "gemma-3n-e4b-instructed-litert-preview": "google/gemma-3n-E4B-it-litert-preview",
    "granite-4.0-1b": "ibm-granite/granite-4.0-1b",
    "granite-4.0-h-1b": "ibm-granite/granite-4.0-h-1b",
    "lfm2.5-1.2b-instruct": "LiquidAI/LFM2.5-1.2B-Instruct",
    "lfm2.5-1.2b-thinking": "LiquidAI/LFM2.5-1.2B-Thinking",
    "mistral-large-2": "mistralai/Mistral-Large-Instruct-2407",
    "mistral-small-3-24b-instruct": "mistralai/Mistral-Small-24B-Instruct-2501",
    "nemotron-ultra-253b": "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1",
    "olmo-1b": "allenai/OLMo-1B",
    "olmo-2-13b": "allenai/OLMo-2-1124-13B",
    "olmo-2-1b": "allenai/OLMo-2-0425-1B",
    "olmo-2-32b": "allenai/OLMo-2-0325-32B",
    "olmo-2-7b": "allenai/OLMo-2-1124-7B",
    "openthinker2-32b": "open-thoughts/OpenThinker2-32B",
    "phi-3-14b": "microsoft/Phi-3-medium-4k-instruct",
}

pytestmark = pytest.mark.skipif(
    not (_FIXTURES / "aliases.parquet").exists(),
    reason="fixtures not built; run `eval-card-registry seed --local`",
)


@pytest.fixture(scope="module")
def resolver():
    return Resolver.from_parquet(str(_FIXTURES))


@pytest.mark.parametrize("slug,expected", BENCHPRESS_SLUGS.items())
def test_benchpress_slug_resolves(resolver, slug, expected):
    res = resolver.resolve(slug, entity_type="model")
    # Must attach to a canonical (no_match -> None) AND use the exact HF repo id
    # (HF-true casing) per the registry's canonical-id standard.
    assert res.canonical_id is not None, f"{slug} did not resolve"
    assert res.canonical_id == expected, (
        f"{slug} -> {res.canonical_id} (expected {expected})"
    )
