"""Guard for scripts/generate_hf_oracle_seed.py: hf_oracle mints are
AUTHORITATIVE real HF repos that must WIN id+casing, never be suppressed.

The authoritative invariant these tests pin: an hf_oracle mint's id IS the real
HF repo id, and every mint is marked `metadata.hf_id == id` so any downstream
dedup/reconcile consumer recognises it as authoritative and must not rewrite or
suppress it in favor of a colliding curated-slug canonical under a different id.
Suppressing such a mint is backwards — it would drop an authoritative HF repo to
keep a transient dev-org slug.

OFFLINE. Does NOT run the generator's destructive main() (it rewrites seed files
in place). Loads the module via importlib and exercises the mint id-derivation +
authoritative-marker logic in isolation.
"""
from __future__ import annotations

import json

from conftest import load_script_module


def _load_module():
    return load_script_module("generate_hf_oracle_seed", "gen_hf_oracle_core_guard")


def test_authoritative_metadata_marks_repo_as_its_own_hf_id():
    mod = _load_module()
    meta = json.loads(mod.authoritative_metadata("meta-llama/Llama-2-7b"))
    assert meta["hf_id"] == "meta-llama/Llama-2-7b"


def test_canon_id_returns_the_real_hf_repo_verbatim():
    """The mint id is the REAL HF repo id (org never folded INTO the id); only
    org_id is folded to the curated developer. So the mint is authoritative."""
    mod = _load_module()
    hf_to_dev = {"meta-llama": "meta", "qwen": "alibaba"}
    cid, org_id = mod.canon_id("meta-llama/Llama-2-7b", hf_to_dev)
    assert cid == "meta-llama/Llama-2-7b"   # id = real repo, NOT meta/llama-2-7b
    assert org_id == "meta"                 # org folded only for the FK
    cid2, org2 = mod.canon_id("Qwen/Qwen2.5-7B", hf_to_dev)
    assert cid2 == "Qwen/Qwen2.5-7B"
    assert org2 == "alibaba"


def test_mint_id_equals_its_metadata_hf_id_so_it_is_never_rewritten():
    """The authoritative invariant for an hf_oracle mint: id == metadata.hf_id,
    so a downstream consumer recognises it as the authoritative real repo and must
    not rewrite or suppress it in favor of a colliding dev-org slug. Reproduces the
    mint-entry construction in main()."""
    mod = _load_module()
    tgt, _org = mod.canon_id("meta-llama/Meta-Llama-3-8B-Instruct", {"meta-llama": "meta"})
    entry = {"id": tgt, "metadata": mod.authoritative_metadata(tgt)}
    assert json.loads(entry["metadata"])["hf_id"] == entry["id"]


def test_nearmiss_separator_rename_is_aliased_but_cross_uploader_is_blocked():
    """A same-uploader org RENAME (separator/case variant, same model name) is NOT
    identity-changing -> aliased onto the HF-true canonical. A genuine
    cross-uploader migration IS identity-changing -> blocked (resolves to self,
    never mis-aliased cross-developer)."""
    mod = _load_module()
    hf_to_dev = {"meta-llama": "meta", "qwen": "alibaba"}  # neither side curated here
    # separator-only same-uploader rename: NOT blocked (aliased).
    assert mod.nearmiss_changes_identity(
        "DeepAutoAI/Explore_Llama-3.1-8B-Inst",
        "DeepAuto-AI/Explore_Llama-3.1-8B-Inst", hf_to_dev) is None
    # genuine cross-uploader migration (same name, different owner): blocked.
    assert mod.nearmiss_changes_identity(
        "AI4free/Dhanishtha", "HelpingAI/Dhanishtha", hf_to_dev) is not None
    # genuine size change: still blocked.
    assert mod.nearmiss_changes_identity(
        "foo/model-7b", "foo/model-13b", hf_to_dev) is not None


def test_no_mint_suppressor_against_core():
    """No reconcile pass may suppress an authoritative HF-repo mint because it
    normalized-collides with a curated-core canonical under a different id: that is
    backwards (it drops the authoritative repo to keep a transient dev-org slug).
    Pinned by asserting the generator exposes no `reconcile_mints_against_core`."""
    mod = _load_module()
    assert not hasattr(mod, "reconcile_mints_against_core"), (
        "do not add a reconcile pass that suppresses authoritative HF mints "
        "(id == metadata.hf_id) in favor of a colliding curated slug"
    )
