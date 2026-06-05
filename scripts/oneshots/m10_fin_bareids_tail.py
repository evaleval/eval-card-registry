#!/usr/bin/env python3
"""Resolve the bare-id cases the bulk converter (m10_fin_bareids.py) left as
collision skips, each via a curated rule grounded in the EEE-raw evidence.

These eight bare ids could not be auto-converted because their natural target
already existed as a canonical id or as an alias of a DIFFERENT canonical. The
collision exposed pre-existing bad aliases that fold genuinely distinct models
together. Handling, per case:

  FOLD  (bare duplicates an already-minted org-prefixed canonical — drop the
        bare entry, union its aliases onto the survivor, rewrite parent edges):
          azerogpt                          -> unknown/azerogpt
          doubao-seed-1-6-thinking-250615   -> bytedance/doubao-seed-1-6-thinking-250615
          ernie-5-0-thinking-preview        -> baidu/ernie-5-0-thinking-preview
          gemini-2-0-flash-thinking-exp-1219-> google/gemini-2.0-flash-thinking   (dated snapshot)

  REKEY (bare is a real, distinct model wrongly aliased onto another canonical —
        free it: strip the bad alias from the wrong owner, set its developer
        org, rename to the org-prefixed id, keep the bare form as an alias):
          cohere-march-2024  -> cohere/cohere-march-2024   (Cohere reward-model checkpoint, off-HF)
          cohere-may-2024    -> cohere/cohere-may-2024     (Cohere reward-model checkpoint, off-HF)
          solar-pro-3        -> upstage/solar-pro-3        (matches the EEE raw `upstage/solar-pro-3`)

  MINT  (underspecified pointer — the Ollama `deepseek-coder` default tag is
        1.3b-instruct, NOT the 6.7b it was wrongly aliased onto):
          mint deepseek/deepseek-coder-1.3b-instruct (training_stage variant of
          deepseek/deepseek-coder-1.3b), carry `deepseek-coder` as its alias,
          drop the bogus bare `deepseek-coder` node.

  Also folds upstage/solar-pro3 (no-dash duplicate) into upstage/solar-pro-3.

Bad aliases removed from their wrong owners:
  cohere/command-r                  : unknown/cohere-march-2024, unknown/cohere-may-2024
  deepseek/deepseek-coder-6.7b      : unknown/deepseek-coder
  upstage/solar-pro2                : upstage/solar-pro-3

DETERMINISTIC. Dry-run by default; --apply writes core.yaml. Validates the
result: no duplicate canonical ids, no alias colliding with a canonical id or
shared across canonicals, no parent edge pointing at a removed id.

Usage:
    LOCAL_MODE=true uv run python scripts/m10_fin_bareids_tail.py
    LOCAL_MODE=true uv run python scripts/m10_fin_bareids_tail.py --apply
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "seed" / "models" / "core.yaml"

# bare -> survivor canonical that already exists; bare entry is folded in
FOLDS = {
    "azerogpt": "unknown/azerogpt",
    "doubao-seed-1-6-thinking-250615": "bytedance/doubao-seed-1-6-thinking-250615",
    "ernie-5-0-thinking-preview": "baidu/ernie-5-0-thinking-preview",
    "gemini-2-0-flash-thinking-exp-1219": "google/gemini-2.0-flash-thinking",
}

# bare -> (new id, developer org). bare form is preserved as an alias.
REKEYS = {
    "cohere-march-2024": ("cohere/cohere-march-2024", "cohere"),
    "cohere-may-2024": ("cohere/cohere-may-2024", "cohere"),
    "solar-pro-3": ("upstage/solar-pro-3", "upstage"),
}

# no-dash duplicate folded into the dashed survivor (applied after the solar rekey)
LATE_FOLDS = {
    "upstage/solar-pro3": "upstage/solar-pro-3",
}

# owner canonical -> aliases to strip (they fold distinct models together)
BAD_ALIASES = {
    "cohere/command-r": ["unknown/cohere-march-2024", "unknown/cohere-may-2024"],
    "deepseek/deepseek-coder-6.7b": ["unknown/deepseek-coder"],
    "upstage/solar-pro2": ["upstage/solar-pro-3"],
}

DEEPSEEK_INSTRUCT = {
    "id": "deepseek/deepseek-coder-1.3b-instruct",
    "display_name": "deepseek/deepseek-coder-1.3b-instruct",
    "org_id": "deepseek",
    "family": "deepseek-coder-1.3b",
    "architecture": None,
    "params_billions": 1.3,
    "parents": [
        {"id": "deepseek/deepseek-coder-1.3b", "relationship": "variant", "axis": "training_stage"}
    ],
    "open_weights": True,
    "release_date": None,
    "input_modalities": None,
    "output_modalities": None,
    "tags": ["open-weight", "code"],
    "metadata": "{}",
    "review_status": "reviewed",
    # the underspecified Ollama default tag resolves here
    "aliases": ["deepseek-coder"],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    doc = yaml.safe_load(CORE.read_text())
    entries = doc["entries"] if isinstance(doc, dict) else doc
    by_id = {e["id"]: e for e in entries if isinstance(e, dict) and isinstance(e.get("id"), str)}

    actions: list[str] = []
    errors: list[str] = []

    def require(cid: str, why: str) -> dict | None:
        e = by_id.get(cid)
        if e is None:
            errors.append(f"missing canonical: {cid} ({why})")
        return e

    # ---- 1. strip bad aliases from their wrong owners ----
    for owner, bad in BAD_ALIASES.items():
        e = require(owner, "bad-alias owner")
        if e is None:
            continue
        al = e.get("aliases") or []
        for b in bad:
            if b in al:
                al = [a for a in al if a != b]
                actions.append(f"strip alias {b!r} from {owner}")
            else:
                errors.append(f"expected alias {b!r} on {owner}, not found")
        e["aliases"] = al

    # ---- 2. mint deepseek instruct leaf; drop the bogus bare node ----
    if "deepseek/deepseek-coder-1.3b-instruct" in by_id:
        errors.append("deepseek/deepseek-coder-1.3b-instruct already exists")
    require("deepseek/deepseek-coder-1.3b", "deepseek instruct parent")
    bare_ds = by_id.get("deepseek-coder")
    if bare_ds is None:
        errors.append("bare deepseek-coder not found")
    else:
        entries.append(dict(DEEPSEEK_INSTRUCT))
        entries.remove(bare_ds)
        by_id.pop("deepseek-coder", None)
        actions.append("mint deepseek/deepseek-coder-1.3b-instruct (alias deepseek-coder); drop bare deepseek-coder")

    # ---- 3. rekeys (rename, set org, keep old id as alias) ----
    rekey_map: dict[str, str] = {}
    for old, (new, org) in REKEYS.items():
        e = by_id.get(old)
        if e is None:
            errors.append(f"rekey source not found: {old}")
            continue
        if new in by_id:
            errors.append(f"rekey target already exists: {new}")
            continue
        e["id"] = new
        e["display_name"] = new
        e["org_id"] = org
        al = e.get("aliases") or []
        if old not in al:
            al.append(old)
        e["aliases"] = al
        tags = [t for t in (e.get("tags") or []) if t != "org-unknown"]
        e["tags"] = tags
        by_id[new] = e
        by_id.pop(old, None)
        rekey_map[old] = new
        actions.append(f"rekey {old} -> {new} (org={org})")

    # ---- 4. folds (drop bare/dup, union aliases onto survivor) ----
    def fold(src: str, dst: str) -> None:
        s = by_id.get(src)
        d = by_id.get(dst)
        if s is None:
            errors.append(f"fold source not found: {src}")
            return
        if d is None:
            errors.append(f"fold survivor not found: {dst}")
            return
        dal = d.get("aliases") or []
        merged = list(dal)
        for a in [src, *(s.get("aliases") or [])]:
            if a not in merged and a != dst:
                merged.append(a)
        d["aliases"] = merged
        # union parents by edge id
        dpar = d.get("parents") or []
        have = {p.get("id") for p in dpar if isinstance(p, dict)}
        for p in (s.get("parents") or []):
            if isinstance(p, dict) and p.get("id") not in have and p.get("id") != dst:
                dpar.append(p)
        d["parents"] = dpar
        entries.remove(s)
        by_id.pop(src, None)
        actions.append(f"fold {src} -> {dst}")

    for src, dst in FOLDS.items():
        fold(src, dst)
    for src, dst in LATE_FOLDS.items():
        fold(src, dst)

    # ---- 5. rewrite parent edges that referenced any renamed/removed id ----
    removed = set(FOLDS) | set(LATE_FOLDS)
    edge_fixes = 0
    for e in entries:
        if not isinstance(e, dict):
            continue
        for p in (e.get("parents") or []):
            if not isinstance(p, dict):
                continue
            pid = p.get("id")
            if pid in rekey_map:
                p["id"] = rekey_map[pid]
                edge_fixes += 1
            elif pid in {**FOLDS, **LATE_FOLDS}:
                p["id"] = {**FOLDS, **LATE_FOLDS}[pid]
                edge_fixes += 1

    # ---- 6. validate ----
    ids = [e["id"] for e in entries if isinstance(e, dict) and isinstance(e.get("id"), str)]
    seen: set[str] = set()
    for i in ids:
        if i in seen:
            errors.append(f"duplicate canonical id after migration: {i}")
        seen.add(i)
    id_set = set(ids)
    alias_owners: dict[str, list[str]] = defaultdict(list)
    for e in entries:
        if isinstance(e, dict):
            for a in (e.get("aliases") or []):
                if isinstance(a, str):
                    alias_owners[a].append(e["id"])
    for a, owners in alias_owners.items():
        if len(owners) > 1:
            errors.append(f"alias {a!r} shared across canonicals: {owners}")
        if a in id_set and a not in owners:
            errors.append(f"alias {a!r} collides with a different canonical id")
    for e in entries:
        if isinstance(e, dict):
            for p in (e.get("parents") or []):
                if isinstance(p, dict) and p.get("id") not in id_set:
                    errors.append(f"{e['id']} has parent edge to missing id {p.get('id')!r}")

    # ---- report ----
    print(f"actions planned : {len(actions)}")
    for a in actions:
        print(f"   + {a}")
    print(f"parent edge rewrites: {edge_fixes}")
    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors:
            print(f"   ! {e}")
        print("\nABORT — refusing to write with validation errors.")
        return 1

    if not args.apply:
        print("\n(dry-run — no files written)")
        return 0

    if isinstance(doc, dict):
        doc["entries"] = entries
    CORE.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=10_000))
    print("\nAPPLIED — core.yaml written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
