#!/usr/bin/env python3
"""Apply batch 1 of the alias-sweep 'unsure' triage (decisions confirmed with the
user). Every minted id was verified on HF; every link edge points at an existing
canonical so the relationship is preserved (size siblings share model_family_id
via a variant/size edge to a shared root; derived models keep a finetune/merge
edge to their base).

DETERMINISTIC. Dry-run by default; --apply writes core.yaml. Validates mint ids
are new, parents/repoint targets exist, aliases are present on exactly one
canonical, no dup id/alias results.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "seed" / "models" / "core.yaml"

# (aliases, {id, ow, parent:(id,rel[,axis])})
MINTS = [
    # distinct models wrongly folded onto another model
    (["unknown/infimm-zephyr-7b"], {"id": "Infi-MM/infimm-zephyr", "ow": True}),
    (["unknown/hammer2-0-7b"], {"id": "MadeAgents/Hammer2.0-7b", "ow": True}),
    (["unknown/qwen3-0-6b-emb", "unknown/qwen3-0-6b-emb-a2a", "unknown/qwen3-0-6b-emb-f2f"],
     {"id": "Qwen/Qwen3-Embedding-0.6B", "ow": True,
      "parent": ("qwen/qwen3-0.6b", "finetune")}),
    (["unknown/internvl-4b-chat-1-5"], {"id": "OpenGVLab/Mini-InternVL-Chat-4B-V1-5", "ow": True}),
    # NOTE: piotr25691/thea-{c,rp}-3b-25r are NOT split — HF redirects them to
    # lunahr/thea-{c,rp}-3b-25r (same repo, moved), so the existing fold onto
    # lunahr is correct; leaving them as aliases there.
]
# Bielik v2.3 already exists as a canonical; repoint the mis-namespaced alias
# onto it (the merge->v2 link is added by the link-session-mints pass).
REPOINTS = {
    "speakleash-ack-cyfronet-agh/bielik-11b-v2-3-instruct": "speakleash/Bielik-11B-v2.3-Instruct",
    # the per-size Skywork reward canonicals already exist; the size aliases were
    # resolving to the generic root — move them to their size (family edge added
    # by the link-session-mints pass).
    "skywork/skywork-reward-v2-qwen3-0-6b": "Skywork/Skywork-Reward-V2-Qwen3-0.6B",
    "skywork/skywork-reward-v2-qwen3-1-7b": "Skywork/Skywork-Reward-V2-Qwen3-1.7B",
    "skywork/skywork-reward-v2-qwen3-4b": "Skywork/Skywork-Reward-V2-Qwen3-4B",
    "skywork/skywork-reward-v2-qwen3-8b": "Skywork/Skywork-Reward-V2-Qwen3-8B",
}
DROPS = ["unknown/internvl-1-5-chat-20b"]  # no 20B InternVL-1.5 exists; not an EEE raw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    doc = yaml.safe_load(CORE.read_text())
    entries = doc["entries"] if isinstance(doc, dict) else doc
    by_id = {e["id"]: e for e in entries if isinstance(e, dict) and isinstance(e.get("id"), str)}

    errors: list[str] = []
    actions: list[str] = []

    def owner_of(alias: str):
        ows = [e for e in entries if isinstance(e, dict) and alias in (e.get("aliases") or [])]
        if len(ows) != 1:
            errors.append(f"alias on {len(ows)} canonicals (need 1): {alias}")
            return None
        return ows[0]

    def detach(alias: str, e: dict) -> None:
        e["aliases"] = [a for a in (e.get("aliases") or []) if a != alias]

    for aliases, spec in MINTS:
        nid = spec["id"]
        if nid in by_id:
            errors.append(f"mint id already exists: {nid}")
            continue
        ok = True
        for a in aliases:
            ow = owner_of(a)
            if ow is None:
                ok = False
                break
            detach(a, ow)
        if not ok:
            continue
        parents = []
        if spec.get("parent"):
            pt = spec["parent"]
            pid, rel = pt[0], pt[1]
            if pid not in by_id:
                errors.append(f"parent missing: {pid}")
                continue
            edge = {"id": pid, "relationship": rel}
            if len(pt) > 2:
                edge["axis"] = pt[2]
            parents = [edge]
        # an alias equal to the new id is redundant (id self-resolves) — drop it
        new_aliases = [a for a in aliases if a != nid]
        row = {
            "id": nid, "display_name": nid, "org_id": None, "family": None,
            "architecture": None, "params_billions": None, "parents": parents,
            "open_weights": spec["ow"], "release_date": None, "input_modalities": None,
            "output_modalities": None, "tags": [], "metadata": "{}",
            "review_status": "reviewed", "aliases": new_aliases,
        }
        entries.append(row)
        by_id[nid] = row
        plink = f" parent={parents[0]}" if parents else ""
        actions.append(f"MINT {nid}  aliases={new_aliases}{plink}")

    for alias, target in REPOINTS.items():
        if target not in by_id:
            errors.append(f"repoint target missing: {target}")
            continue
        ow = owner_of(alias)
        if ow is None:
            continue
        detach(alias, ow)
        tal = by_id[target].get("aliases") or []
        if alias not in tal:
            tal.append(alias)
        by_id[target]["aliases"] = tal
        actions.append(f"REPOINT {alias}: {ow['id']} -> {target}")

    for alias in DROPS:
        ow = owner_of(alias)
        if ow is None:
            continue
        detach(alias, ow)
        actions.append(f"DROP {alias} (off {ow['id']})")

    for a in actions:
        print(f"   {a}")
    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors:
            print(f"   ! {e}")
        print("ABORT.")
        return 1
    if not args.apply:
        print(f"\n[{len(actions)} actions] (dry-run)")
        return 0
    if isinstance(doc, dict):
        doc["entries"] = entries
    CORE.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=10_000))
    print(f"\nAPPLIED {len(actions)} actions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
