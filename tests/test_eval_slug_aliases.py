"""Raw eval-harness model slugs resolve to a real, unique canonical model.

Guards that each slug in SLUG_CANONICALS attaches to an existing canonical entity
(not an alias-minted phantom) at its HF-true id, with no shadow duplicate under a
different spelling or org namespace. Skips if fixtures aren't built (run
`eval-card-registry seed --local` first).
"""
import re
from pathlib import Path

import pandas as pd
import pytest
import yaml
from eval_entity_resolver import Resolver
from eval_entity_resolver.fold import build_curated_org_map

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FIXTURES = _REPO_ROOT / "fixtures"
_ORGS_YAML = _REPO_ROOT / "seed" / "orgs.yaml"


def _ndup_leaf(name: str) -> str:
    """Normalized model leaf (mirrors the `ndup` in
    test_gate_invariants.test_no_real_hf_id_duplicated_by_slug): case-fold, glue
    letter->digit boundaries, split a version dot (3.1 -> 3-1) while keeping size
    tokens (7b), collapse separators. Two leaves that normalize equal are the same
    model spelled differently."""
    s = name.lower()
    s = re.sub(r"([a-z])[-_ /]+(\d)", r"\1\2", s)
    s = re.sub(r"(\d)\.(\d)(?![bmkt])", r"\1-\2", s)
    return re.sub(r"[-_ /]+", "-", s)


def _dev_identity(cid: str, fold) -> tuple[str, str]:
    """(developer-org, normalized-leaf) — the org-AWARE identity. Catches
    cross-namespace shadows (deepseek/deepseek-v3-1 vs deepseek-ai/DeepSeek-V3.1)
    that a plain collision_key misses because the org prefix differs."""
    org, _, name = cid.partition("/")
    if not name:  # bare id, no org
        org, name = "", org
    return (fold(org.lower()), _ndup_leaf(name))

# Raw eval-harness slug -> the HF-true repo id it must resolve to.
SLUG_CANONICALS = {
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


@pytest.fixture(scope="module")
def models_df():
    return pd.read_parquet(_FIXTURES / "canonical_models.parquet")


@pytest.fixture(scope="module")
def org_fold():
    hf_to_dev = build_curated_org_map(yaml.safe_load(_ORGS_YAML.read_text()) or [])
    return lambda org: hf_to_dev.get(org, org)


@pytest.mark.parametrize("slug,expected", SLUG_CANONICALS.items())
def test_eval_slug_resolves(resolver, models_df, org_fold, slug, expected):
    res = resolver.resolve(slug, entity_type="model")
    # Must attach to a canonical (no_match -> None) AND use the exact HF repo id
    # (HF-true casing) per the registry's canonical-id standard.
    assert res.canonical_id is not None, f"{slug} did not resolve"
    assert res.canonical_id == expected, (
        f"{slug} -> {res.canonical_id} (expected {expected})"
    )
    # The target must be a REAL canonical entity — not an alias-minted phantom
    # (a dangling alias would still "resolve" to a string with no backing row).
    ids = set(models_df["id"])
    assert expected in ids, (
        f"{slug} resolves to {expected} which is NOT a canonical_models row "
        f"(alias-minted phantom / dangling target)"
    )
    # And it must be the SOLE canonical for its ORG-AWARE identity — no shadow
    # duplicate under a different spelling OR org namespace (e.g. deepseek/deepseek-v3-1
    # shadowing deepseek-ai/DeepSeek-V3.1, which a plain collision_key misses because
    # the org prefix differs).
    want = _dev_identity(expected, org_fold)
    twins = sorted(i for i in ids if _dev_identity(i, org_fold) == want)
    assert twins == [expected], (
        f"{slug}: shadow-duplicate canonicals share the developer-scoped identity "
        f"of {expected}: {twins}"
    )
