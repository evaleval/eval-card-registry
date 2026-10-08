"""scripts/rekey_promoted_ids.py: the text-level re-key of seed references to
ids the models.dev refresh promoted to HF-true ids."""
from __future__ import annotations

from conftest import load_script_module


def _mod():
    return load_script_module("rekey_promoted_ids")


MAPPING = {"lab/model-a": "Lab/Model-A-Instruct", "lab/model-b": "Lab/Model-B"}


def test_rewrite_edges_repoints_and_drops_self_edges():
    text = (
        "# curated\n"
        "- id: lab/model-a-reasoning\n"
        "  parents:\n"
        "  - id: lab/model-a\n"
        "    relationship: variant\n"
        "    axis: mode\n"
        "- id: Lab/Model-A-Instruct\n"
        "  parents:\n"
        "  - id: lab/model-a\n"
        "    relationship: variant\n"
        "    axis: training_stage\n"
        "  - id: lab/base\n"
        "    relationship: finetune\n"
        "- id: lab/model-a\n"
        "  aliases:\n"
        "  - keep-me\n"
    )
    m = _mod()
    out, n = m.rewrite_edges(text, MAPPING)
    assert n == 2
    out, dropped = m.strip_self_edges(out)
    assert dropped == 1
    assert out == (
        "# curated\n"
        "- id: lab/model-a-reasoning\n"
        "  parents:\n"
        "  - id: Lab/Model-A-Instruct\n"
        "    relationship: variant\n"
        "    axis: mode\n"
        "- id: Lab/Model-A-Instruct\n"
        "  parents:\n"
        "  - id: lab/base\n"
        "    relationship: finetune\n"
        "- id: lab/model-a\n"
        "  aliases:\n"
        "  - keep-me\n"
    )
    # A record whose only parent becomes itself keeps a well-formed list, and
    # a nested `- id:` list that is not `parents` is never an edge.
    only = "- id: Lab/Model-A-Instruct\n  parents:\n  - id: Lab/Model-A-Instruct\n    relationship: variant\n  other:\n  - id: lab/model-a\n"
    out, dropped = m.strip_self_edges(only)
    assert (out, dropped) == ("- id: Lab/Model-A-Instruct\n  parents: []\n  other:\n  - id: lab/model-a\n", 1)
    assert m.rewrite_edges(only, MAPPING) == (only, 0)


def test_rewrite_keys_touches_only_column_zero_ids():
    text = "- id: lab/model-a\n  aliases:\n  - lab/model-a\n- id: lab/other\n  parents:\n  - id: lab/model-b\n"
    out, n = _mod().rewrite_keys(text, MAPPING)
    assert n == 1
    assert out == "- id: Lab/Model-A-Instruct\n  aliases:\n  - lab/model-a\n- id: lab/other\n  parents:\n  - id: lab/model-b\n"


def test_tier3_records_drop_and_bridge_as_aliases():
    m = _mod()
    tier3 = (
        "- id: lab/model-a\n  display_name: lab/model-a\n  aliases:\n  - host/model-a\n"
        "- id: lab/keep\n  display_name: keep\n"
    )
    kept, dropped = m.drop_tier3_records(tier3, MAPPING)
    assert dropped == [("lab/model-a", ["host/model-a"])]
    assert kept == "- id: lab/keep\n  display_name: keep\n"
    bridged = m.bridge_aliases(
        "- id: x\n  aliases:\n  - y\n", {"Lab/Model-A-Instruct": ["lab/model-a", "host/model-a"]}
    )
    assert bridged.endswith("- id: Lab/Model-A-Instruct\n  aliases:\n  - host/model-a\n  - lab/model-a\n")


def test_merge_duplicate_records_unions_into_the_first_block():
    text = (
        "# head comment\n"
        "- id: Lab/Model-A-Instruct\n  aliases:\n  - a1\n"
        "# between\n"
        "- id: lab/other\n  aliases:\n  - o1\n"
        "- id: Lab/Model-A-Instruct\n  aliases:\n  - a1\n  - a2\n"
        "  parents:\n  - id: lab/base\n    relationship: finetune\n"
    )
    out, n = _mod().merge_duplicate_records(text)
    assert n == 1
    assert out == (
        "# head comment\n"
        "- id: Lab/Model-A-Instruct\n  aliases:\n  - a1\n  - a2\n"
        "  parents:\n  - id: lab/base\n    relationship: finetune\n"
        "# between\n"
        "- id: lab/other\n  aliases:\n  - o1\n"
    )
    assert _mod().merge_duplicate_records(out) == (out, 0)


def test_promotion_map_reads_hf_true_aliases_and_skips_umbrellas(tmp_path, monkeypatch):
    import yaml
    m = _mod()
    gen = tmp_path / "models_dev.generated.yaml"
    gen.write_text(yaml.safe_dump([
        {"id": "Lab/Model-A-Instruct", "display_name": "A", "resolution_source": "models_dev",
         "metadata": '{"hf_deferred": true}', "aliases": ["lab/model-a", "model-a", "lab/lab"]},
        {"id": "lab/minted", "display_name": "M", "resolution_source": "models_dev",
         "aliases": ["lab/minted-old"]},
    ]))
    oracle = tmp_path / "hf_oracle.generated.yaml"
    oracle.write_text(yaml.safe_dump([{"id": "Lab/Model-B", "display_name": "B", "resolution_source": "hf"}]))
    catalog = tmp_path / "models_dev_catalog.generated.yaml"
    catalog.write_text(yaml.safe_dump([{"id": "Lab/Model-B", "aliases": ["lab/model-b"]}]))
    (tmp_path / "tier3.yaml").write_text("[]\n")
    (tmp_path / "overrides.yaml").write_text(yaml.safe_dump({"non_lineage_bases": ["lab/lab"]}))
    monkeypatch.setattr(m, "GENERATED", [gen, catalog])
    monkeypatch.setattr(m, "HF_ORACLE", oracle)
    monkeypatch.setattr(m, "TIER3", tmp_path / "tier3.yaml")
    monkeypatch.setattr(m, "OVERRIDES", tmp_path / "overrides.yaml")
    assert m.promotion_map() == {"lab/model-a": "Lab/Model-A-Instruct", "lab/model-b": "Lab/Model-B"}
