#!/usr/bin/env python3
"""Convert bare canonical ids into "/"-prefixed form.

Convert every BARE canonical id (no "/") in seed/models/core.yaml into a
"/"-prefixed form. Per bare id, in priority order:

  (i)  ORACLE re-key: if the bare slug normalizes to a REAL oracle HF repo
       name-part AND that repo's developer org agrees with the entry's own
       org_id, re-key the entry to the real repo id (folds into the oracle re-key).
  (ii) DEV-ORG prefix: else, if a developer is UNAMBIGUOUS — the entry already
       carries a non-null org_id (resolved upstream by models.dev's developer
       derivation / the curated org map) that is a real canonical_orgs row —
       re-key to `{org_id}/{slug}`.
  (iii) UNKNOWN route: else (org_id is null), re-key to `unknown/{slug}`, ensure
       tags include `org-unknown`, and surface the entry to
       curation/org_unknown_review.json.

NEVER auto-guesses a vendor from a name: path (ii) fires ONLY when an org
was ALREADY resolved upstream (non-null org_id), never from the slug text.

DETERMINISTIC. VALIDATES every edit against live data (fixtures + core.yaml +
oracle) and emits ONLY edits that validate; everything else is reported as a
skip with a reason. On re-key, the old bare id is preserved as an alias (so
existing resolution keeps working) and any parent edge that referenced the old
bare id is rewritten to the new id.

Dry-run by default; `--apply` writes core.yaml + org_unknown_review.json.

Usage:
    LOCAL_MODE=true uv run python scripts/m10_fin_bareids.py            # dry-run
    LOCAL_MODE=true uv run python scripts/m10_fin_bareids.py --apply    # write
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "seed" / "models" / "core.yaml"
FIXTURES = ROOT / "fixtures"
ORGS_PARQUET = FIXTURES / "canonical_orgs.parquet"
ORACLE = Path("/Users/jchim/projects/evaleval/hf_model_id_resolution.json")
REVIEW = ROOT  / "curation" / "org_unknown_review.json"


def normalize(s: str) -> str:
    """Lowercase + collapse separators (-/_./:) to single dashes.
    Mirrors services.hub_stats.normalize / the resolver's normalized match."""
    return re.sub(r"[/_.:\-]+", "-", s.lower()).strip("-")


# ---------------------------------------------------------------------------
# Live-data loaders
# ---------------------------------------------------------------------------
def load_oracle_repo_index() -> dict[str, dict[str, str]]:
    """name-norm -> {hf-org-lower: real_repo_id} over every real oracle repo
    (fixed_hf_model_id). Used by path (i) to test if a bare slug maps to a real
    repo under an agreeing org."""
    data = json.loads(ORACLE.read_text())["resolutions"]
    real_ids: set[str] = set()
    for meta in data.values():
        fid = meta.get("fixed_hf_model_id")
        if isinstance(fid, str) and "/" in fid:
            real_ids.add(fid)
    idx: dict[str, dict[str, str]] = defaultdict(dict)
    for rid in real_ids:
        org, name = rid.split("/", 1)
        idx[normalize(name)][org.lower()] = rid
    return idx


def load_org_ids() -> set[str]:
    return set(pd.read_parquet(ORGS_PARQUET)["id"].astype(str))


def build_hf_to_dev(curated_org_ids: set[str]) -> dict[str, str]:
    """HF-org-lowercase -> curated developer slug, single-sourced from
    strategies/fuzzy._ORG_ALIASES (the same map generate_hf_oracle_seed.py uses
    for the org-agreement check)."""
    import sys

    pkg = ROOT / "packages" / "eval-entity-resolver" / "src"
    if str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))
    from eval_entity_resolver.strategies.fuzzy import _ORG_ALIASES  # type: ignore

    return {k.lower(): v for k, v in _ORG_ALIASES.items()}


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    doc = yaml.safe_load(CORE.read_text())
    entries = doc["entries"] if isinstance(doc, dict) else doc
    by_id = {e["id"]: e for e in entries if isinstance(e, dict) and isinstance(e.get("id"), str)}
    all_ids = set(by_id)
    all_ids_lc = {i.lower() for i in all_ids}

    # every alias currently in use (to keep the no-dup-alias seed gate green)
    # plus its owning canonical id(s), so a re-key target that is already an
    # alias of a DIFFERENT canonical is rejected (re-keying onto it would mint a
    # second canonical carrying an alias the seed CLI treats as a duplicate).
    alias_in_use: set[str] = set()
    alias_owner: dict[str, set[str]] = defaultdict(set)
    for e in entries:
        if isinstance(e, dict):
            owner = e.get("id")
            for a in (e.get("aliases") or []):
                if isinstance(a, str):
                    alias_in_use.add(a)
                    if isinstance(owner, str):
                        alias_owner[a].add(owner)
    # case-insensitive view of the alias namespace (the seed gate is case-fold)
    alias_owner_lc: dict[str, set[str]] = defaultdict(set)
    for a, owners in alias_owner.items():
        alias_owner_lc[a.lower()] |= owners

    oracle_idx = load_oracle_repo_index()
    org_ids = load_org_ids()
    hf_to_dev = build_hf_to_dev(org_ids)

    bare = [
        e for e in entries
        if isinstance(e, dict) and isinstance(e.get("id"), str) and "/" not in e["id"]
    ]

    # planned re-keys: (entry, old_id, new_id, path, extra)
    plan_i: list[tuple] = []
    plan_ii: list[tuple] = []
    plan_iii: list[tuple] = []
    skips: list[tuple[str, str]] = []

    # track new ids reserved within this run so two bare ids can't collide
    reserved: set[str] = set()

    def reserve_ok(new_id: str, old_id: str) -> str | None:
        """Return a skip-reason if new_id collides with an existing canonical id
        (exact or case-insensitive), one already reserved this run, or an alias
        already owned by a different canonical; else None. old_id is the entry's
        current id, exempted so re-keying onto an entry's own alias is allowed."""
        if new_id in all_ids or new_id in reserved:
            return f"target id already exists: {new_id}"
        if new_id.lower() in all_ids_lc or new_id.lower() in {r.lower() for r in reserved}:
            return f"target id case-collides with an existing canonical: {new_id}"
        other_owners = (alias_owner.get(new_id, set()) | alias_owner_lc.get(new_id.lower(), set())) - {old_id}
        if other_owners:
            return f"target id is already an alias of {sorted(other_owners)[0]}: {new_id}"
        return None

    for e in sorted(bare, key=lambda x: x["id"]):
        old = e["id"]
        slug = old  # bare ids are already lowercase-kebab slugs
        nslug = normalize(slug)
        entry_org = e.get("org_id")

        # ---- path (i): oracle re-key under an agreeing dev-org ----
        repos = oracle_idx.get(nslug)
        if repos:
            # org-agreement: the entry's org_id must map from one of the repos'
            # HF orgs (hf_to_dev), or equal it case-insensitively. Only fires
            # when EXACTLY ONE repo agrees (no ambiguity).
            agreeing = []
            for hf_org, rid in repos.items():
                dev = hf_to_dev.get(hf_org, hf_org)
                if entry_org is not None and (
                    dev == entry_org or hf_org == str(entry_org).lower()
                ):
                    agreeing.append(rid)
            if len(agreeing) == 1:
                new_id = agreeing[0]
                why = reserve_ok(new_id, old)
                if why:
                    skips.append((old, f"path-i {why}"))
                    continue
                reserved.add(new_id)
                plan_i.append((e, old, new_id, "i", None))
                continue
            # repo(s) exist but no single agreeing org -> do NOT guess; fall
            # through to (ii)/(iii).

        # ---- path (ii): unambiguous dev-org already resolved upstream ----
        if entry_org is not None:
            if entry_org not in org_ids:
                skips.append((old, f"path-ii org_id '{entry_org}' not a real canonical_orgs row"))
                continue
            new_id = f"{entry_org}/{slug}"
            why = reserve_ok(new_id, old)
            if why:
                skips.append((old, f"path-ii {why}"))
                continue
            reserved.add(new_id)
            plan_ii.append((e, old, new_id, "ii", entry_org))
            continue

        # ---- path (iii): no recoverable org -> unknown/ + review ----
        new_id = f"unknown/{slug}"
        why = reserve_ok(new_id, old)
        if why:
            skips.append((old, f"path-iii {why}"))
            continue
        reserved.add(new_id)
        plan_iii.append((e, old, new_id, "iii", None))

    planned = plan_i + plan_ii + plan_iii

    # ---- defensive edge-rewrite map (old bare id -> new id) ----
    rekey = {old: new for (_, old, new, _, _) in planned}
    edge_rewrites = 0
    for e in entries:
        if isinstance(e, dict):
            for p in (e.get("parents") or []):
                if isinstance(p, dict) and p.get("id") in rekey:
                    edge_rewrites += 1

    # ---- validation: alias preservation must not create a duplicate alias ----
    # The old bare id becomes an alias on the re-keyed entry. If that string is
    # already an alias elsewhere, the seed CLI would abort on a dup alias.
    for (_, old, _new, path, _x) in list(planned):
        if old in alias_in_use:
            skips.append((old, f"path-{path} old id already used as an alias elsewhere"))
            planned = [t for t in planned if t[1] != old]
            plan_i = [t for t in plan_i if t[1] != old]
            plan_ii = [t for t in plan_ii if t[1] != old]
            plan_iii = [t for t in plan_iii if t[1] != old]

    # ---- report ----
    print(f"bare ids in core.yaml: {len(bare)}")
    print(f"  path (i)   oracle re-key (agreeing dev-org) : {len(plan_i)}")
    print(f"  path (ii)  dev-org prefix (org_id non-null) : {len(plan_ii)}")
    print(f"  path (iii) unknown/ + org_unknown_review    : {len(plan_iii)}")
    print(f"  skipped (validation)                        : {len(skips)}")
    print(f"  parent edge refs to rewrite                 : {edge_rewrites}")
    print()
    print("examples:")
    for (_, old, new, path, _x) in (plan_i[:3] + plan_ii[:4] + plan_iii[:3]):
        print(f"   [{path}] {old}  ->  {new}")
    if skips:
        print("\nskips:")
        for old, why in skips[:20]:
            print(f"   SKIP {old} :: {why}")

    if not args.apply:
        print("\n(dry-run — no files written)")
        return 0

    # ---- apply ----
    for (e, old, new, path, _x) in planned:
        e["id"] = new
        e["display_name"] = new  # full id => unique; no bare-name alias collision
        # preserve old bare id as an alias so existing resolution keeps working
        aliases = e.get("aliases") or []
        if old not in aliases:
            aliases.append(old)
        e["aliases"] = aliases
        if path == "iii":
            tags = e.get("tags") or []
            if "org-unknown" not in tags:
                tags.append("org-unknown")
            e["tags"] = tags
            e["org_id"] = None
    # rewrite parent edges
    for e in entries:
        if isinstance(e, dict):
            for p in (e.get("parents") or []):
                if isinstance(p, dict) and p.get("id") in rekey:
                    p["id"] = rekey[p["id"]]

    if isinstance(doc, dict):
        doc["entries"] = entries
    CORE.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=10_000))

    # update org_unknown_review.json (additive, idempotent by minted_canonical_id)
    review = json.loads(REVIEW.read_text())
    existing_min = {r.get("minted_canonical_id") for r in review.get("entries", [])}
    added = 0
    for (e, old, new, path, _x) in plan_iii:
        if new in existing_min:
            continue
        review["entries"].append({
            "raw_value": old,
            "minted_canonical_id": new,
            "inferred_base": None,
            "proposed_org": None,
            "rationale": "AC-1b bare-id conversion: no recoverable dev-org (org_id was null)",
            "status": "unreviewed",
        })
        added += 1
    review["count"] = len(review["entries"])
    REVIEW.write_text(json.dumps(review, indent=2) + "\n")

    print(f"\nAPPLIED {len(planned)} re-keys "
          f"(i={len(plan_i)}, ii={len(plan_ii)}, iii={len(plan_iii)}); "
          f"edge rewrites={edge_rewrites}; review additions={added}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
