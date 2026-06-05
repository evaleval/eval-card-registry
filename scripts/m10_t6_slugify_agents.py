#!/usr/bin/env python3
"""Slugify malformed inferred canonical ids (spaces / colon / '+' / parens) —
SWE-bench-style agent/scaffold systems like
'unknown/OpenHands + CodeAct v2.1 (claude-3-5-sonnet-20241022)'. Lowercase the
name-part and collapse non-alphanumerics to single dashes; keep the original raw
string as an alias (so the EEE submission still resolves) and as display_name.

DETERMINISTIC. Dry-run by default; --apply writes core.yaml.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "seed" / "models" / "core.yaml"
_MALFORMED = re.compile(r"[ :+()]")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    doc = yaml.safe_load(CORE.read_text())
    entries = doc["entries"] if isinstance(doc, dict) else doc
    by_id = {e["id"]: e for e in entries if isinstance(e, dict) and isinstance(e.get("id"), str)}
    alias_owner: dict[str, str] = {}
    for e in entries:
        if isinstance(e, dict) and isinstance(e.get("id"), str):
            for a in (e.get("aliases") or []):
                if isinstance(a, str):
                    alias_owner.setdefault(a, e["id"])

    errors, actions = [], []
    rekeys: dict[str, str] = {}
    reserved: set[str] = set()
    for e in list(entries):
        if not isinstance(e, dict):
            continue
        cid = e.get("id")
        if not isinstance(cid, str) or "/" not in cid or not _MALFORMED.search(cid):
            continue
        if e.get("resolution_source") != "inferred":
            continue
        org, name = cid.split("/", 1)
        new_id = f"{org}/{slug(name)}"
        if new_id == cid:
            continue
        # new_id is an existing canonical, or an alias of a different canonical
        # -> MERGE the malformed entry into that owner (e.g. a Claude snapshot)
        merge_into = None
        if new_id in by_id and by_id[new_id] is not e:
            merge_into = new_id
        elif new_id in alias_owner and alias_owner[new_id] != cid:
            merge_into = alias_owner[new_id]
        if merge_into is not None and merge_into != cid:
            tgt = by_id[merge_into]
            tal = tgt.get("aliases") or []
            for a in [cid, new_id, *(e.get("aliases") or [])]:
                if isinstance(a, str) and a not in tal and a != merge_into:
                    tal.append(a)
            tgt["aliases"] = tal
            entries.remove(e)
            by_id.pop(cid, None)
            rekeys[cid] = merge_into
            actions.append(f"MERGE {cid!r} -> {merge_into}")
            continue
        if new_id in reserved:
            errors.append(f"slug collision (this run): {cid} -> {new_id}")
            continue
        # keep original readable display_name; preserve raw id as alias
        if not e.get("display_name") or _MALFORMED.search(str(e.get("display_name", ""))) is None:
            e["display_name"] = name
        e["id"] = new_id
        al = e.get("aliases") or []
        if cid not in al:
            al.append(cid)
        e["aliases"] = al
        by_id[new_id] = e
        by_id.pop(cid, None)
        reserved.add(new_id)
        rekeys[cid] = new_id
        actions.append(f"{cid!r} -> {new_id}")

    edge_fixes = 0
    for e in entries:
        if isinstance(e, dict):
            for p in (e.get("parents") or []):
                if isinstance(p, dict) and p.get("id") in rekeys:
                    p["id"] = rekeys[p["id"]]
                    edge_fixes += 1

    for a in actions:
        print(f"   {a}")
    print(f"   slugified: {len(actions)}; edge rewrites: {edge_fixes}")
    if errors:
        print("ERRORS:", errors)
        return 1
    if not args.apply:
        print("\n(dry-run)")
        return 0
    if isinstance(doc, dict):
        doc["entries"] = entries
    CORE.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=10_000))
    print("\nAPPLIED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
