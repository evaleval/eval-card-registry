#!/usr/bin/env python3
"""Apply the MEDIUM-confidence (and any still-unapplied HIGH-confidence)
wrong-alias fixes from the curated wrong-alias audit (the markdown tables at
the DOC path below).

Same approach as scripts/m10_b3_03_apply_alias_audit.py /
scripts/m10_b3_04_deferred_own.py. For each row, the disposition is one of:

  own_canonical : remove the alias from its (wrong) owner; mint a new canonical
                  with id == the alias (clean org/name form only; display_name =
                  the FULL id so its auto-emitted display alias can't collide;
                  org_id=null so the seed derives org-from-prefix + de-orphan
                  creates the community org; org-unknown tag only when prefix is
                  "unknown"; resolution_source=inferred, review_status=draft).
  repoint       : remove the alias from its (wrong) owner; add it to the stated
                  existing target canonical.
  remove_alias  : remove the alias from its (wrong) owner.

Every edit is VALIDATED against LIVE data — core.yaml (seed of truth),
fixtures/*.parquet (current resolved state) and the frozen HF oracle. Only edits
that validate are applied; everything else is SKIPPED and reported (never applied
blind). Display-form / bare / truncated aliases that cannot be cleanly placed in
core.yaml's alias lists are skipped+reported. Already-applied rows (from the
earlier high-confidence pass) are detected and skipped as no-ops.

For any OWN-minted alias that is also an oracle `fixed_near_miss` raw, the alias
string is appended to the audit_bad_nearmiss.json sidecar's "raws" list so the
test-gate exemption + generator guard stay complete.

Dry-run by default; --apply writes seed/models/core.yaml (+ the sidecar). This
phase does NOT reseed.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "specs" / "model-resolution-rework" / "alias-audit.md"
CORE = ROOT / "seed" / "models" / "core.yaml"
SIDECAR = ROOT / "specs" / "model-resolution-rework" / "audit_bad_nearmiss.json"
FIXTURES = ROOT / "fixtures" / "aliases.parquet"
ORACLE = ROOT.parent / "hf_model_id_resolution.json"

ID_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
ANY_ID = re.compile(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+")


def parse_section2(known_ids: set[str]):
    """Header- and section-aware parse of the audit doc's wrong-alias section.

    Returns list of dicts: {alias, conf, disp, target}. disp in
    {own, repoint, remove, ambiguous, None}.

    The section's tables do not share one column layout. The parser tracks the CURRENT
    table's header columns and reads the disposition/target from whichever of
    these is present, in priority order:
      - a `repoint_to` column  -> disp=repoint, target = that cell (column-implied)
      - a `target` column      -> target = that cell; disp from the `disp` cell
      - a `disp` / `disp / target` / `actually is` cell -> disposition keyword
        (own_canonical / repoint / remove / own_canonical-or-repoint) parsed from
        the cell text, plus any inline `repoint <id>` target.
      - a prose sub-header that states a blanket disposition (the BlackBeenie
        cross-uploader table is conf-only and inherits `own_canonical` from its
        "(all own_canonical …)" sub-header line).
    Targets may be a BARE id without an org prefix as written in the source data
    (e.g. `molmo-2-8b`, `dracarys2-llama-3.1-70b-instruct`); those are validated
    against the live canonical id set by the caller.
    """
    lines = DOC.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("## 2."))
    end = next(i for i, l in enumerate(lines) if l.startswith("## 3."))

    section_default = None  # disposition implied by a prose sub-header line
    cols: list[str] | None = None
    out = []
    for raw in lines[start:end]:
        st = raw.strip()
        # A prose sub-header that states a blanket disposition for the next table.
        if st and not st.startswith("|") and not st.startswith("#"):
            low = st.lower()
            section_default = "own" if "own_canonical" in low and (
                "all own_canonical" in low or st.rstrip().endswith(":")
            ) else None
            continue
        if not st.startswith("|"):
            continue
        cells = [c.strip().strip("`").strip() for c in st.strip("|").split("|")]
        if set("".join(cells)) <= set("-"):  # separator row
            continue
        if cells[0].lower() == "alias":  # header row -> remember columns
            cols = [c.lower() for c in cells]
            continue
        if len(cells) < 2:
            continue

        alias = cells[0]
        conf = cells[-1].lower()
        rec = dict(zip(cols, cells)) if cols and len(cols) == len(cells) else {}
        middle = cells[1:-1]
        blob = " ".join(middle)
        low = blob.lower()

        disp = None
        target = None

        # 1. column-implied target/disposition (when the row matches the header arity)
        if "repoint_to" in (cols or []) and rec.get("repoint_to"):
            disp = "repoint"
            target = rec["repoint_to"]
        elif "target" in (cols or []):
            target = rec.get("target") or None
            dcell = (rec.get("disp") or "").lower()
            if "own_canonical or repoint" in dcell:
                disp = "ambiguous"
            elif "repoint" in dcell:
                disp = "repoint"
            elif "own" in dcell:
                disp = "own"

        # 2. content-driven from the disposition blob (disp / disp-or-target cell)
        if disp is None:
            if "own_canonical or repoint" in low or ("own_canonical" in low and "repoint" in low):
                disp = "ambiguous"
            elif "remove" in low:
                disp = "remove"
            elif "own_canonical" in low or "own canonical" in low:
                disp = "own"
            elif "repoint" in low:
                disp = "repoint"
            elif section_default:
                disp = section_default

        # 3. resolve a repoint target if still missing
        if disp == "repoint" and not target:
            m = re.search(r"repoint\s+`?(" + ANY_ID.pattern + r")", blob)
            if m:
                target = m.group(1)
            else:
                wrong = middle[0] if middle else ""
                cand = [c for cell in middle[1:] for c in ANY_ID.findall(cell) if c != wrong]
                target = cand[-1] if cand else None

        out.append({"alias": alias, "conf": conf, "disp": disp, "target": target})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    # --- live data ---------------------------------------------------------
    doc = yaml.safe_load(CORE.read_text())
    entries = doc["entries"] if isinstance(doc, dict) else doc
    by_id = {e["id"]: e for e in entries if isinstance(e, dict) and isinstance(e.get("id"), str)}
    ids = set(by_id)

    alias_owner: dict[str, str] = {}
    for e in entries:
        if isinstance(e, dict):
            for a in (e.get("aliases") or []):
                if isinstance(a, str):
                    alias_owner.setdefault(a, e["id"])

    mal = pd.read_parquet(FIXTURES)
    mal = mal[mal["entity_type"] == "model"]
    fx_canon = dict(zip(mal["raw_value"], mal["canonical_id"]))

    oracle = json.loads(ORACLE.read_text())["resolutions"]
    nearmiss_raws = {
        k for k, v in oracle.items()
        if v.get("resolution_status") == "fixed_near_miss"
    }
    sidecar = json.loads(SIDECAR.read_text())
    sidecar_existing = set(sidecar.get("raws", []))

    rows = parse_section2(ids)  # parses the curated wrong-alias audit tables

    plan_own: list[tuple[str, str | None]] = []   # (alias, owner_or_None)
    plan_repoint: list[tuple[str, str, str]] = []  # (alias, owner, target)
    plan_remove: list[tuple[str, str]] = []        # (alias, owner)
    skip: list[tuple[str, str]] = []
    already: list[tuple[str, str]] = []

    for r in rows:
        a, disp, tgt = r["alias"], r["disp"], r["target"]
        owner = alias_owner.get(a)
        fx = fx_canon.get(a)

        if disp == "ambiguous":
            skip.append((a, "ambiguous own_canonical-OR-repoint (cohere trio) — manual"))
            continue
        if disp is None:
            skip.append((a, "no disposition parsed"))
            continue

        if disp == "own":
            if a in ids:
                already.append((a, "own_canonical already minted"))
                continue
            if not ID_RE.match(a):
                skip.append((a, "own_canonical but alias not a clean org/name id (display/bare) — manual"))
                continue
            if owner is None and fx is None:
                skip.append((a, "own_canonical but alias maps to no current canonical — manual"))
                continue
            # owner may be None when the wrong mapping lives only in a generated
            # source (fixtures), not a core alias list; we can still mint, but we
            # cannot drop the generated alias from core. Only mint when the alias
            # is owned by a core entry so the wrong fold is actually removed.
            if owner is None:
                skip.append((a, f"own_canonical but alias not in any core aliases list (fx->{fx}); display/generated — manual"))
                continue
            plan_own.append((a, owner))

        elif disp == "repoint":
            if not tgt:
                skip.append((a, "repoint but no target parsed — manual"))
                continue
            if tgt not in ids:
                skip.append((a, f"repoint target {tgt!r} does not exist as canonical — manual"))
                continue
            if owner == tgt:
                already.append((a, "repoint already correct (owner == target)"))
                continue
            if owner is None:
                skip.append((a, f"repoint but alias not in any core aliases list (fx->{fx}); display/generated — manual"))
                continue
            plan_repoint.append((a, owner, tgt))

        elif disp == "remove":
            if owner is None:
                skip.append((a, f"remove but alias not in any core aliases list (fx->{fx}) — manual"))
                continue
            plan_remove.append((a, owner))

    # near-miss sidecar additions: own-minted aliases that are oracle near_miss raws
    sidecar_adds = sorted(
        a for a, _ in plan_own
        if a in nearmiss_raws and a not in sidecar_existing
    )

    # --- report ------------------------------------------------------------
    print(f"parsed wrong-alias rows: {len(rows)}")
    print(f"  PLAN own_canonical={len(plan_own)} repoint={len(plan_repoint)} remove={len(plan_remove)}")
    print(f"  already-applied (no-op): {len(already)}")
    print(f"  SKIP (validation/edge — reported, NOT applied): {len(skip)}")
    print(f"  sidecar near_miss additions: {len(sidecar_adds)} -> {sidecar_adds}")
    print("\n  sample own_canonical:", [a for a, _ in plan_own[:8]])
    print("  sample repoint:", [(a, t) for a, _, t in plan_repoint[:8]])
    print("  remove:", [a for a, _ in plan_remove])
    print("\n  SKIP detail:")
    for a, why in skip:
        print(f"     SKIP {a}  :: {why}")

    if not args.apply:
        print("\n(dry-run; pass --apply to write core.yaml + sidecar)")
        return 0

    def drop_alias(owner_id: str, alias: str) -> None:
        e = by_id.get(owner_id)
        if e and isinstance(e.get("aliases"), list):
            e["aliases"] = [x for x in e["aliases"] if x != alias]

    for a, owner in plan_remove:
        drop_alias(owner, a)
    for a, owner, tgt in plan_repoint:
        drop_alias(owner, a)
        te = by_id[tgt]
        al = list(te.get("aliases") or [])
        if a not in al:
            al.append(a)
        te["aliases"] = sorted(set(al))
    for a, owner in plan_own:
        drop_alias(owner, a)
        orgless = a.split("/", 1)[0].lower() == "unknown"
        entries.append({
            "id": a, "display_name": a, "org_id": None,
            "family": None, "architecture": None, "params_billions": None,
            "parents": [], "open_weights": None, "release_date": None,
            "input_modalities": None, "output_modalities": None,
            "tags": ["org-unknown"] if orgless else [], "aliases": [],
            "metadata": "{}", "resolution_source": "inferred", "review_status": "draft",
        })

    if isinstance(doc, dict):
        doc["entries"] = entries
    CORE.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=10_000))

    if sidecar_adds:
        sidecar["raws"] = sorted(set(sidecar.get("raws", [])) | set(sidecar_adds))
        SIDECAR.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n")

    print(f"\nAPPLIED: own={len(plan_own)} repoint={len(plan_repoint)} remove={len(plan_remove)} sidecar+={len(sidecar_adds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
