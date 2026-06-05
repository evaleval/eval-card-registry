"""Unit test for the generate_hf_oracle_seed.py near-miss identity guard, which
stops the generator from turning an HF "did-you-mean" redirect that changes model
identity (a size/version/cross-developer change) into a confirmed alias — that
would mis-merge two distinct models. Does NOT run the generator — just exercises
the guard predicate."""
from conftest import load_script_module


def _mod():
    return load_script_module("generate_hf_oracle_seed", "gen_hf_oracle")


def test_nearmiss_identity_guard():
    m = _mod()
    g = m.nearmiss_changes_identity
    h = {"meta-llama": "meta", "facebook": "meta", "cohereforai": "cohere"}

    # --- must BLOCK (identity-changing) ---
    assert g("pankajmathur/orca_mini_v9_2_14B", "pankajmathur/orca_mini_phi-4", h)  # denylist
    assert g("foo/bar-3b", "foo/bar-7b", h)            # genuine size change 3B->7B
    assert g("aorg/some-model", "borg/some-model", h)  # cross-developer org

    # --- must ALLOW (None — same model / benign normalisation) ---
    assert g("meta-llama/Meta-Llama-3.1-8B", "meta-llama/Llama-3.1-8B", h) is None   # prefix drop
    assert g("Qwen/Qwen3-235B-A22B", "Qwen/Qwen3-235B-A22B-Instruct-2507", h) is None  # MoE active-param, not a 2nd size
    assert g("x/Yi-1.5-34B", "x/Yi-1.5-34B-Chat", h) is None                          # version token, not a size
    assert g("unknown/yi-vl", "01-ai/yi-vl", h) is None                               # org-less placeholder -> real org is fine
    assert g("meta-llama/foo-7b", "facebook/foo-7b", h) is None                       # curated org fold (meta-llama==facebook==meta)

    # denylist actually loaded from the sidecar
    assert len(m._AUDIT_BAD_NEARMISS) >= 50
