#!/usr/bin/env python3
"""Find aliases that may fold GENUINELY DISTINCT models together.

A recall-oriented, deterministic candidate generator — it never decides; it
surfaces aliases whose identity tokens disagree with their owner canonical so a
judge (the alias-sweep workflow) can rule on each. Conservative on the axes that
have produced false positives before (size sets are compared as sets, so MoE
active-param tags like `235b-a22b` and version tokens like `1.5` don't trip it).

Categories (a candidate may match several; `reasons` lists all):
  ORPHAN_PREFIX  alias is `unknown/<x>` but its owner id is NOT unknown/<x>
                 (self-fold residue: an org-less form glued onto a real model)
  ID_COLLISION   alias normalizes to a DIFFERENT canonical's id (the alias names
                 another real model in the registry)
  SIZE_DISJOINT  alias and owner each carry size tokens and the sets are disjoint
                 (e.g. `...-1.3b` aliased onto a `...-6.7b` canonical)
  CROSS_ORG      alias carries an `org/` prefix whose developer differs from the
                 owner's org_id (possible cross-uploader / cross-vendor fold)

Output: JSON {generated_from, counts, candidates:[{alias, owner_id, owner_org,
reasons, owner_norm, alias_norm}]} — one row per suspicious (alias, owner).

Usage:
    LOCAL_MODE=true uv run python scripts/scan_suspicious_aliases.py [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "seed" / "models" / "core.yaml"
DEFAULT_OUT = ROOT  / "curation" / "suspicious_aliases.json"

_SEP = re.compile(r"[/_.:\-]+")
_SIZE = re.compile(r"(?<![a-z0-9])(\d+(?:\.\d+)?)\s*b(?![a-z0-9])")


def norm(s: str) -> str:
    return _SEP.sub("-", s.lower()).strip("-")


def sizes(s: str) -> set[str]:
    """Size tokens like `7b`, `1.3b`, `70b` (NOT `a22b` MoE active-param, NOT a
    bare version number). Normalised so `6.7b`/`6-7b` compare equal."""
    out: set[str] = set()
    for m in _SIZE.finditer(s.lower()):
        out.add(m.group(1).rstrip("0").rstrip(".") if "." in m.group(1) else m.group(1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    doc = yaml.safe_load(CORE.read_text())
    entries = doc["entries"] if isinstance(doc, dict) else doc
    entries = [e for e in entries if isinstance(e, dict) and isinstance(e.get("id"), str)]

    id_set = {e["id"] for e in entries}
    # normalized id -> canonical id (for ID_COLLISION). Last wins; collisions
    # among canonical ids themselves are a separate gate's concern.
    norm_to_id: dict[str, str] = {}
    for e in entries:
        norm_to_id[norm(e["id"])] = e["id"]

    # developer map for CROSS_ORG (same source the resolver/seed use)
    import sys

    pkg = ROOT / "packages" / "eval-entity-resolver" / "src"
    if str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))
    from eval_entity_resolver.strategies.fuzzy import _ORG_ALIASES  # type: ignore

    hf_to_dev = {k.lower(): v for k, v in _ORG_ALIASES.items()}

    candidates: list[dict] = []
    for e in entries:
        oid = e["id"]
        oorg = e.get("org_id")
        onorm = norm(oid)
        osz = sizes(oid)
        for a in (e.get("aliases") or []):
            if not isinstance(a, str):
                continue
            anorm = norm(a)
            reasons: list[str] = []

            # ORPHAN_PREFIX is only interesting when the org-less form names
            # something OTHER than the owner. An `unknown/<name>` whose name-part
            # matches (or is contained in) the owner's name-part is just a stale
            # pre-org-resolution spelling of the SAME model — not a fold.
            if a.startswith("unknown/") and not oid.startswith("unknown/"):
                acore = norm(a.split("/", 1)[1])
                ocore = norm(oid.split("/", 1)[1]) if "/" in oid else onorm
                if acore != ocore and acore not in ocore and ocore not in acore:
                    reasons.append("ORPHAN_PREFIX")

            hit = norm_to_id.get(anorm)
            if hit is not None and hit != oid:
                reasons.append("ID_COLLISION")

            asz = sizes(a)
            if osz and asz and osz.isdisjoint(asz):
                reasons.append("SIZE_DISJOINT")

            if "/" in a:
                apfx = a.split("/", 1)[0]
                adev = hf_to_dev.get(apfx.lower(), apfx.lower())
                if oorg and apfx.lower() != "unknown" and adev != str(oorg).lower():
                    reasons.append("CROSS_ORG")

            if reasons:
                candidates.append({
                    "alias": a,
                    "owner_id": oid,
                    "owner_org": oorg,
                    "reasons": reasons,
                    "owner_norm": onorm,
                    "alias_norm": anorm,
                })

    counts: dict[str, int] = {}
    for c in candidates:
        for r in c["reasons"]:
            counts[r] = counts.get(r, 0) + 1
    counts["TOTAL_candidates"] = len(candidates)

    out = {
        "generated_from": "seed/models/core.yaml",
        "counts": counts,
        "candidates": sorted(candidates, key=lambda c: (c["owner_id"], c["alias"])),
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"candidates: {len(candidates)}  ->  {args.out}")
    for k, v in sorted(counts.items()):
        print(f"   {k:20} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
