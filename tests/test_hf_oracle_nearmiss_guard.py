"""Unit test for the generate_hf_oracle_seed.py near-miss identity guard, which
stops the generator from propagating an HF "did-you-mean" redirect that changes
model identity into a confirmed alias (the wrong-alias class found by the alias
audit). Does NOT run the generator — just exercises the guard predicate."""
import importlib.util
from pathlib import Path


def _mod():
    p = Path(__file__).resolve().parent.parent / "scripts" / "generate_hf_oracle_seed.py"
    spec = importlib.util.spec_from_file_location("gen_hf_oracle", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


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
