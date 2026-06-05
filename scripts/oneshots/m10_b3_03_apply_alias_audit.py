#!/usr/bin/env python3
"""Apply the HIGH-confidence wrong-alias fixes from the curated wrong-alias audit.

Parses the wrong-alias audit tables (the markdown at the DOC path below), takes
HIGH-confidence rows only, and for each:
  - own_canonical : remove the alias from its (wrong) owner; mint a new canonical
                    with id == the alias (must be a clean org/name form).
  - repoint       : remove the alias from its (wrong) owner; add it to the stated
                    target (target must already exist as a canonical).
  - remove_alias  : remove the alias from its (wrong) owner.

Every entry is VALIDATED against live data; anything that does not validate is
SKIPPED and reported (never applied blind). Dry-run by default; --apply writes
seed/models/core.yaml.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import yaml
import pandas as pd
from eval_entity_resolver.strategies.fuzzy import _ORG_ALIASES

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "specs" / "model-resolution-rework" / "alias-audit.md"
CORE = ROOT / "seed" / "models" / "core.yaml"

hf_to_dev = {k.lower(): v for k, v in _ORG_ALIASES.items()}
for e in yaml.safe_load((ROOT / "seed" / "orgs.yaml").read_text()) or []:
    if isinstance(e, dict) and isinstance(e.get("hf_org"), str) and e["hf_org"].strip() and isinstance(e.get("id"), str):
        hf_to_dev[e["hf_org"].lower()] = e["id"]
def dev(prefix): return hf_to_dev.get(prefix.lower(), prefix)

ID_RE = re.compile(r'^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$')

def parse_doc():
    lines = DOC.read_text().splitlines()
    s = next(i for i, l in enumerate(lines) if l.startswith("## 2."))
    e = next(i for i, l in enumerate(lines) if l.startswith("## 3."))
    out = []
    for l in lines[s:e]:
        if not l.strip().startswith("|"):
            continue
        cells = [c.strip().strip("`").strip() for c in l.strip().strip("|").split("|")]
        if len(cells) < 3 or cells[0].lower() == "alias" or set(cells[0]) <= set("-"):
            continue
        alias = cells[0]
        conf = cells[-1].lower()
        if "high" not in conf:
            continue
        rest = cells[1:-1]
        wrong = rest[0] if rest else ""
        blob = " ".join(rest).lower()
        # candidate repoint targets = clean org/name ids in the columns AFTER the
        # wrong-canonical, excluding the wrong canonical itself.
        cands = [c for cell in rest[1:]
                 for c in re.findall(r'([A-Za-z0-9._-]+/[A-Za-z0-9._-]+)', cell)
                 if c != wrong]
        disp = None; target = None
        if "own_canonical" in blob and "repoint" in blob:
            disp = "own_or_repoint"          # ambiguous (cohere trio) -> skip, manual
        elif "own_canonical" in blob:
            disp = "own"
        elif "remove" in blob:
            disp = "remove"
        elif cands:                          # implicit/explicit repoint_to column
            disp = "repoint"; target = cands[-1]
        out.append({"alias": alias, "wrong_doc": wrong, "disp": disp, "target": target, "conf": conf})
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--apply", action="store_true"); args = ap.parse_args()
    doc = yaml.safe_load(CORE.read_text())
    entries = doc["entries"] if isinstance(doc, dict) else doc
    by_id = {e["id"]: e for e in entries if isinstance(e, dict) and isinstance(e.get("id"), str)}
    ids = set(by_id)
    # alias -> owning entry id (search aliases lists)
    alias_owner = {}
    for e in entries:
        if not isinstance(e, dict): continue
        for a in (e.get("aliases") or []):
            if isinstance(a, str): alias_owner.setdefault(a, e["id"])

    rows = parse_doc()
    plan = {"own": [], "repoint": [], "remove": []}
    skip = []
    for r in rows:
        a, disp, tgt = r["alias"], r["disp"], r["target"]
        owner = alias_owner.get(a)
        if disp == "own_or_repoint":
            skip.append((a, "ambiguous own_canonical-OR-repoint — manual (human picks target)")); continue
        if disp is None:
            skip.append((a, "no disposition parsed")); continue
        if owner is None:
            skip.append((a, "alias not found in any entry's aliases list (display-form or generated) — manual")); continue
        if disp == "own":
            if not ID_RE.match(a):
                skip.append((a, "own_canonical but alias not a clean org/name id (bare/display) — manual")); continue
            if a in ids:
                skip.append((a, f"own_canonical id already exists — manual")); continue
            plan["own"].append((a, owner))
        elif disp == "repoint":
            if not tgt:
                skip.append((a, "repoint but no target parsed")); continue
            if tgt not in ids:
                skip.append((a, f"repoint target {tgt} does not exist as canonical — manual")); continue
            if tgt == owner:
                skip.append((a, "repoint target == current owner (already correct)")); continue
            plan["repoint"].append((a, owner, tgt))
        elif disp == "remove":
            plan["remove"].append((a, owner))

    print(f"parsed high-confidence rows: {len(rows)}")
    print(f"  will apply: own_canonical={len(plan['own'])} repoint={len(plan['repoint'])} remove={len(plan['remove'])}")
    print(f"  SKIP (validation/edge — reported, NOT applied): {len(skip)}")
    for a, why in skip: print(f"     SKIP {a}  :: {why}")
    print("\n  sample own_canonical:", [a for a, _ in plan["own"][:6]])
    print("  sample repoint:", [(a, t) for a, _, t in plan["repoint"][:6]])
    print("  remove:", [a for a, _ in plan["remove"]])

    if not args.apply:
        print("\n(dry-run; pass --apply to write)"); return 0

    def drop_alias(owner_id, alias):
        e = by_id.get(owner_id)
        if e and isinstance(e.get("aliases"), list):
            e["aliases"] = [x for x in e["aliases"] if x != alias]

    for a, owner in plan["remove"]:
        drop_alias(owner, a)
    for a, owner, tgt in plan["repoint"]:
        drop_alias(owner, a)
        te = by_id[tgt]; al = list(te.get("aliases") or [])
        if a not in al: al.append(a)
        te["aliases"] = sorted(set(al))
    for a, owner in plan["own"]:
        drop_alias(owner, a)
        # org_id left null on purpose: derive_model_lineage_fields fills it from
        # the id prefix and the de-orphan pass auto-creates the community org row,
        # so we never dangle an org FK by hard-coding a maybe-missing org.
        entries.append({
            "id": a, "display_name": a.split("/", 1)[1], "org_id": None,
            "family": None, "architecture": None, "params_billions": None,
            "parents": [], "open_weights": None, "release_date": None,
            "input_modalities": None, "output_modalities": None,
            "tags": [], "aliases": [], "metadata": "{}",
            "resolution_source": "inferred", "review_status": "draft",
        })
    if isinstance(doc, dict): doc["entries"] = entries
    CORE.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=10_000))
    print(f"\nAPPLIED: own={len(plan['own'])} repoint={len(plan['repoint'])} remove={len(plan['remove'])}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
