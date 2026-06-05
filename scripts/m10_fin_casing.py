#!/usr/bin/env python3
"""Re-case lowercased canonical leaves to their HF-true casing.

Re-case canonical LEAVES in seed/models/core.yaml whose id is lowercased but
equals (case-insensitively) a real oracle `fixed_hf_model_id` (the HF-true
casing). For each such leaf:

  1. re-key its id: lowercase -> HF-cased;
  2. append the old lowercase id as an alias (dedup, case-insensitive);
  3. rewrite EVERY reference to the old id across ALL entries so nothing
     dangles: parents[].id, model_group_id, model_family_id,
     lineage_origin_model_id.

GUARD: if the HF-cased target id already exists as a DIFFERENT entry (re-keying
would collide), do NOT re-key — emit an ALIAS-MERGE instead (fold the lowercase
loser into the HF-cased winner: winner gains the loser's id + aliases, loser
entry is removed, edges repoint loser->winner). The seed CLI aborts on duplicate
canonical ids / aliases, so a blind re-key into an occupied id is illegal.

Deterministic + re-runnable. VALIDATES every edit against live data
(seed/models/core.yaml + the frozen oracle) and stages only what validates;
anything that fails a guard is REPORTED, not applied.

Mirrors scripts/generate_hf_oracle_seed.py's rekey + global edge-rewrite logic
(rename_map -> parents[].id / EDGE_ID_KEYS rewrite over every entry).

Usage (DRY-RUN by default; do not pass --apply in this phase):
    LOCAL_MODE=true uv run python scripts/m10_fin_casing.py
    LOCAL_MODE=true uv run python scripts/m10_fin_casing.py --apply
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REGISTRY_ROOT = Path(__file__).resolve().parents[1]            # eval-card-registry/
EVALEVAL_ROOT = REGISTRY_ROOT.parent                           # evaleval/
ORACLE = EVALEVAL_ROOT / "hf_model_id_resolution.json"
CORE = REGISTRY_ROOT / "seed" / "models" / "core.yaml"

# Scalar edge pointers that must be rewritten on a re-key (same set the
# generator rewrites). parents[].id is handled separately (it is a list).
EDGE_ID_KEYS = ("model_group_id", "model_family_id", "lineage_origin_model_id")


def _load_core() -> tuple[dict | list, list[dict]]:
    doc = yaml.safe_load(CORE.read_text())
    entries = doc["entries"] if isinstance(doc, dict) else doc
    return doc, entries


def _hf_true_ids() -> dict[str, str]:
    """{lowercase_id -> HF-true-cased id} for every oracle fixed_exact /
    fixed_near_miss `fixed_hf_model_id` (the real HF repo id)."""
    oracle = json.loads(ORACLE.read_text())["resolutions"]
    by_lower: dict[str, str] = {}
    for _raw, meta in oracle.items():
        if meta.get("resolution_status") not in ("fixed_exact", "fixed_near_miss"):
            continue
        f = meta.get("fixed_hf_model_id")
        if isinstance(f, str) and "/" in f:
            # If the oracle ever carried two casings under one lower-key the
            # data would be ambiguous; first-seen wins deterministically (the
            # current oracle has zero such collisions — verified).
            by_lower.setdefault(f.lower(), f)
    return by_lower


def _add_alias(entry: dict, alias: str) -> bool:
    """Append `alias` to entry['aliases'] if not already present (case-insens).
    Returns True if added."""
    if not alias:
        return False
    aliases = list(entry.get("aliases") or [])
    seen = {a.lower() for a in aliases if isinstance(a, str)}
    if alias.lower() in seen or alias == entry.get("id"):
        return False
    aliases.append(alias)
    entry["aliases"] = aliases
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write seed/models/core.yaml")
    args = ap.parse_args()

    doc, entries = _load_core()
    by_id: dict[str, dict] = {}
    for e in entries:
        if isinstance(e, dict) and isinstance(e.get("id"), str):
            # core.yaml ids are unique (a duplicate would already break the seed
            # CLI); last-write-wins is harmless if a dup somehow exists.
            by_id[e["id"]] = e
    ids = set(by_id)

    hf_by_lower = _hf_true_ids()

    # --- Candidate selection -------------------------------------------------
    # A leaf is a casing candidate iff its id case-insensitively equals a real
    # HF-true id but differs in EXACT casing (i.e. the seed stored it lowercased
    # while HF carries it cased). VALIDATED against live data: the id must be a
    # real entry, and the HF-true target must come from the oracle.
    rekeys: list[tuple[str, str]] = []        # (old_lower_id, hf_cased_id)
    merges: list[tuple[str, str]] = []        # (loser_lower_id, winner_hf_cased_id)
    for cid in sorted(ids):
        hf = hf_by_lower.get(cid.lower())
        if hf is None or hf == cid:
            continue  # not in oracle, or already HF-cased -> nothing to do
        # GUARD: HF-cased target already exists as a different entry -> merge.
        if hf in ids:
            merges.append((cid, hf))
        else:
            rekeys.append((cid, hf))

    # Global rename map old_id -> new_id (re-keys AND merge losers both repoint).
    rename_map: dict[str, str] = {}
    for old, new in rekeys:
        rename_map[old] = new
    for loser, winner in merges:
        rename_map[loser] = winner

    # --- Apply (in-memory staging; written only under --apply) --------------
    edge_rewrites = 0
    aliases_added = 0
    removed_entries = 0

    def _rewrite_edges(entry: dict) -> int:
        nonlocal edge_rewrites
        n = 0
        parents = entry.get("parents")
        if isinstance(parents, list):
            for p in parents:
                if isinstance(p, dict) and isinstance(p.get("id"), str) and p["id"] in rename_map:
                    p["id"] = rename_map[p["id"]]
                    n += 1
        for key in EDGE_ID_KEYS:
            v = entry.get(key)
            if isinstance(v, str) and v in rename_map:
                entry[key] = rename_map[v]
                n += 1
        edge_rewrites += n
        return n

    # 1. re-key in place + append old lowercase id as alias
    for old, new in rekeys:
        e = by_id[old]
        e["id"] = new
        if _add_alias(e, old):
            aliases_added += 1

    # 2. merge losers into winners (fold id + aliases, then drop the loser)
    loser_ids = {l for l, _ in merges}
    for loser, winner in merges:
        win = by_id[winner]
        le = by_id[loser]
        if _add_alias(win, loser):
            aliases_added += 1
        for a in le.get("aliases") or []:
            if isinstance(a, str) and _add_alias(win, a):
                aliases_added += 1

    # 3. rewrite all edges across every surviving entry, drop merge losers
    new_entries: list[dict] = []
    for e in entries:
        if isinstance(e, dict) and e.get("id") in loser_ids and e is by_id.get(e.get("id")):
            # this is the loser entry (its id was NOT re-keyed) -> drop it
            removed_entries += 1
            continue
        if isinstance(e, dict):
            _rewrite_edges(e)
        new_entries.append(e)

    # --- Validation: no dangling edges, no case-dup ids ---------------------
    final_ids = {e["id"] for e in new_entries if isinstance(e, dict) and isinstance(e.get("id"), str)}
    dangling: list[tuple[str, str]] = []
    for e in new_entries:
        if not isinstance(e, dict):
            continue
        for p in e.get("parents") or []:
            if isinstance(p, dict) and isinstance(p.get("id"), str) and p["id"] not in final_ids:
                dangling.append((e["id"], p["id"]))
        for key in EDGE_ID_KEYS:
            v = e.get(key)
            if isinstance(v, str) and v not in final_ids:
                dangling.append((e["id"], f"{key}={v}"))
    from collections import Counter
    lc = Counter(i.lower() for i in final_ids)
    case_dups = {k: v for k, v in lc.items() if v > 1}

    # --- Report --------------------------------------------------------------
    print("=== m10_fin_casing ===")
    print(f"core.yaml entries: {len(entries)}")
    print(f"oracle HF-true ids: {len(hf_by_lower)}")
    print(f"PLANNED re-keys (lowercase -> HF-cased): {len(rekeys)}")
    for old, new in rekeys[:10]:
        print(f"    rekey  {old}  ->  {new}")
    print(f"PLANNED alias-merges (HF-cased id already exists): {len(merges)}")
    for loser, winner in merges[:10]:
        print(f"    merge  {loser}  ->  {winner}")
    print(f"edge references rewritten: {edge_rewrites}")
    print(f"aliases added: {aliases_added}")
    print(f"merge-loser entries removed: {removed_entries}")
    print(f"VALIDATION: dangling edges after stage: {len(dangling)}; "
          f"case-dup ids after stage: {len(case_dups)}")
    if dangling:
        print(f"  ABORT-WORTHY dangling (NOT applying): {dangling[:10]}", file=sys.stderr)
    if case_dups:
        print(f"  ABORT-WORTHY case-dups (NOT applying): {dict(list(case_dups.items())[:10])}",
              file=sys.stderr)

    if dangling or case_dups:
        print("validation failed — refusing to apply", file=sys.stderr)
        return 2

    if args.apply:
        if isinstance(doc, dict):
            doc["entries"] = new_entries
        else:
            doc = new_entries
        CORE.write_text(
            yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=10_000)
        )
        print("APPLIED.")
    else:
        print("(dry-run — no files written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
