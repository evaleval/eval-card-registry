#!/usr/bin/env python3
"""The deferred HIGH-confidence alias fixes that are cleanly own_canonical (the
alias is its own distinct org/name model). Removes the wrong alias from its owner
and mints the alias as its own canonical. Explicit reviewed list (the audit-doc
rows the generic parser in scripts/m10_b3_03_apply_alias_audit.py skipped: the
BlackBeenie block had its disposition in the section header; the
meituan/baidu/ai2/longcat/PowerInfer/unknown rows pointed at a bare shorthand
target). Dry-run default; --apply writes core.yaml."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml

CORE = Path(__file__).resolve().parents[1] / "seed" / "models" / "core.yaml"

OWN = [
    "BlackBeenie/Neos-Llama-3.1-base", "BlackBeenie/Bloslain-8B-v0.2",
    "BlackBeenie/Llama-3.1-8B-OpenO1-SFT-v0.1",
    "BlackBeenie/Llama-3.1-8B-pythonic-passthrough-merge",
    "BlackBeenie/Neos-Gemma-2-9b", "BlackBeenie/Neos-Llama-3.1-8B",
    "baidu/ernie-5-0-thinking-preview", "ai2/molmo2-8b",
    "unknown/r1-0528-qwen3-8b-thinking", "unknown/deepseek-r1-0528-qwen3-8b",
    "PowerInfer/SmallThinker-3B-Preview",
    # DEFERRED to the manual/medium pass (entangled with an existing bare
    # `longcat-flash-thinking-2601` / shared `longcat-flash-lite` name -> need a
    # repoint-vs-mint decision, not a blind own_canonical):
    #   meituan/longcat-flash-thinking-2601, meituan/longcat-flash-lite,
    #   longcat/longcat-flash-lite
]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--apply", action="store_true"); args = ap.parse_args()
    doc = yaml.safe_load(CORE.read_text())
    entries = doc["entries"] if isinstance(doc, dict) else doc
    by_id = {e["id"]: e for e in entries if isinstance(e, dict) and isinstance(e.get("id"), str)}
    ids = set(by_id)
    owner = {}
    for e in entries:
        if isinstance(e, dict):
            for a in (e.get("aliases") or []):
                if isinstance(a, str): owner.setdefault(a, e["id"])

    apply_, skip = [], []
    for a in OWN:
        if a in ids:
            skip.append((a, "already a canonical")); continue
        if a not in owner:
            skip.append((a, "alias not found in any aliases list")); continue
        apply_.append((a, owner[a]))
    print(f"own_canonical to mint: {len(apply_)}; skip: {len(skip)}")
    for a, o in apply_: print(f"   MINT {a}  (remove alias from {o})")
    for a, w in skip: print(f"   SKIP {a} :: {w}")
    if not args.apply:
        print("(dry-run)"); return 0

    for a, o in apply_:
        e = by_id[o]
        if isinstance(e.get("aliases"), list):
            e["aliases"] = [x for x in e["aliases"] if x != a]
        orgless = a.split("/", 1)[0].lower() == "unknown"
        entries.append({
            # display_name = the full id (unique) so its auto-emitted display
            # alias can't collide with an existing bare-name canonical/alias.
            "id": a, "display_name": a, "org_id": None,
            "family": None, "architecture": None, "params_billions": None,
            "parents": [], "open_weights": None, "release_date": None,
            "input_modalities": None, "output_modalities": None,
            "tags": ["org-unknown"] if orgless else [], "aliases": [],
            "metadata": "{}", "resolution_source": "inferred", "review_status": "draft",
        })
    if isinstance(doc, dict): doc["entries"] = entries
    CORE.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=10_000))
    print(f"APPLIED {len(apply_)} own_canonical mints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
