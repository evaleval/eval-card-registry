#!/usr/bin/env python3
"""Fix models.dev re-host mints whose canonical id wears a BASE-MODEL vendor
prefix that contradicts the resolved org_id (and isn't a real HF repo) — the
review's split-identity HIGH finding.

Decision table: curation/rehost_repoint.json (HF-grounded with retries; each
entry kind ∈ {REKEY_REAL, CLOSED, UNCERTAIN, MERGE}):
  REKEY_REAL — a real HF repo exists (correct casing); rename the id to it
               (merge if that id is already a canonical), keep the junk id as
               an alias so EEE coverage holds.
  CLOSED     — no HF repo found; re-key to the org-decoupled {org_id}/{slug}
               (org_id is the verified real owner — removes the false vendor
               prefix). target field already holds {org}/{slug}.
  UNCERTAIN  — HF lookups stayed flaky AND the id is already an agent-scaffold
               under unknown/; just clear the stale vendor org_id to `unknown`.
  MERGE/self — target == junk: the id IS a real repo; skip.

DETERMINISTIC. Dry-run by default; --apply writes core.yaml.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "seed" / "models" / "core.yaml"
TABLE = ROOT / "curation" / "rehost_repoint.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    actions = json.loads(TABLE.read_text())
    doc = yaml.safe_load(CORE.read_text())
    entries = doc["entries"] if isinstance(doc, dict) else doc
    by_id = {e["id"]: e for e in entries if isinstance(e, dict) and isinstance(e.get("id"), str)}
    # alias -> owning canonical id: a new id that is already an alias of a
    # DIFFERENT canonical means that canonical is the real model — MERGE into it
    # rather than minting a colliding duplicate (the seed gate forbids dup aliases).
    alias_owner: dict[str, str] = {}
    for e in entries:
        if isinstance(e, dict) and isinstance(e.get("id"), str):
            for al in (e.get("aliases") or []):
                if isinstance(al, str):
                    alias_owner.setdefault(al, e["id"])

    errors: list[str] = []
    rekeys: dict[str, str] = {}   # old id -> new id (for edge rewrite)
    n_rekey = n_merge = n_orgfix = n_skip = 0
    reserved: set[str] = set()

    def detach_attach(old: str, e: dict) -> None:
        al = e.get("aliases") or []
        if old not in al:
            al.append(old)
        e["aliases"] = al

    for a in actions:
        junk = a["junk"]
        kind = a["kind"]
        target = a.get("target")
        e = by_id.get(junk)
        if e is None:
            continue  # already handled / not present
        # Authority guard: if the row already records its own id as the real HF
        # repo (metadata.hf_id == id), the id is CORRECT (two-tier design: real
        # HF namespace + decoupled dev org). Skip — flaky HF lookups must never
        # rewrite an already-correct id.
        md = e.get("metadata")
        if isinstance(md, str):
            try:
                if json.loads(md).get("hf_id") == junk:
                    n_skip += 1
                    continue
            except (ValueError, TypeError):
                pass
        if kind == "UNCERTAIN" or (junk.startswith("unknown/")):
            # agent-scaffold under unknown/: just clear the stale vendor org
            if e.get("org_id") not in (None, "unknown"):
                e["org_id"] = "unknown"
                n_orgfix += 1
            continue
        if not target or target == junk:
            n_skip += 1
            continue
        # SAFETY: only act on HF-VERIFIED real targets (REKEY_REAL/MERGE). The
        # CLOSED bucket (no HF repo found) is NOT rekeyed — flaky HF lookups
        # mis-classify real-namespace repos (e.g. ibm-granite/*) as CLOSED, and
        # blindly re-keying them to {org}/{slug} would mangle a correct id. The
        # CLOSED residual keeps its current id (imperfect prefix, but not newly
        # broken); the generator fix prevents new such mints.
        if kind not in ("REKEY_REAL", "MERGE"):
            n_skip += 1
            continue
        # collision guards: target is an existing canonical id OR an existing
        # alias of a different canonical -> MERGE junk into that owner.
        merge_into = None
        if target in by_id and by_id[target] is not e:
            merge_into = target
        elif target in alias_owner and alias_owner[target] != junk:
            merge_into = alias_owner[target]
        if merge_into is not None:
            tgt = by_id[merge_into]
            tal = tgt.get("aliases") or []
            for al in [junk, target, *(e.get("aliases") or [])]:
                if isinstance(al, str) and al not in tal and al != merge_into:
                    tal.append(al)
            tgt["aliases"] = tal
            entries.remove(e)
            by_id.pop(junk, None)
            rekeys[junk] = merge_into
            n_merge += 1
            continue
        if target in reserved:
            errors.append(f"two junk ids map to same new id: {target}")
            continue
        # pure REKEY: rename id, keep junk id as alias
        e["id"] = target
        e["display_name"] = target
        detach_attach(junk, e)
        by_id[target] = e
        by_id.pop(junk, None)
        reserved.add(target)
        rekeys[junk] = target
        n_rekey += 1

    # rewrite parent edges referencing any renamed/merged id
    edge_fixes = 0
    for e in entries:
        if not isinstance(e, dict):
            continue
        for p in (e.get("parents") or []):
            if isinstance(p, dict) and p.get("id") in rekeys:
                p["id"] = rekeys[p["id"]]
                edge_fixes += 1

    # validate: no duplicate canonical id
    ids = [e["id"] for e in entries if isinstance(e, dict) and isinstance(e.get("id"), str)]
    seen: set[str] = set()
    for i in ids:
        if i in seen:
            errors.append(f"duplicate canonical id: {i}")
        seen.add(i)

    print(f"rekey_real/closed: {n_rekey}  merge: {n_merge}  org-fix(agents): {n_orgfix}  skip: {n_skip}  edge_rewrites: {edge_fixes}")
    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors[:20]:
            print(f"   ! {e}")
        print("ABORT.")
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
