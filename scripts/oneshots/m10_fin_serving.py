#!/usr/bin/env python3
"""Demote serving / moving-pointer leaves to aliases on their base.

Demote serving / moving-pointer LEAVES (per specs/entity-modeling.md: a moving pointer
is an ALIAS on the family/base root, never its own canonical leaf):

  (a) pure moving-pointer leaf — a `-latest` / `-exp` / non-real `-preview` leaf
      that carries a `variant axis=version` parent edge to an EXISTING base
      canonical (the version edge already declares "same release as base"):
      delete the leaf, fold its id + display_name spellings onto the base as
      aliases.
  (b) pure serving-SKU leaf — a free/fast/throughput/tput SKU over an EXISTING
      base with the SAME weights, carrying a single concrete serving platform in
      metadata.alias_platforms/providers: delete the leaf, fold its spellings
      (id, display_name, the underlying serving spellings) onto the base AND
      migrate its `alias_platforms` so the base records the inference_platform.

EXCLUDED by design (NOT touched):
  - name-"Turbo"/"Fast" real products (gpt-4-turbo, qwen-turbo, grok-*-fast,
    Imagen-4-Fast) — distinct models;
  - `-tput`/FP8 = quantization (handled by the quant track, not here);
  - any leaf that is ITSELF a real oracle HF repo id (a real distinct release
    that merely spells "-exp"/"-preview" in its name);
  - any `-latest`/`-exp` WITHOUT an unambiguous existing base canonical (the
    no-parent-edge API pointers like openai/gpt-latest, claude-opus-latest):
    reported, never guessed.

This is a DETERMINISTIC re-derivation: it enumerates candidates from the live
fixtures, classifies each, and VALIDATES every planned fold against live data
(fixtures + core.yaml + oracle) before keeping it. Only validated folds are
applied; everything else is reported.

Validations per fold (a fold that fails ANY is dropped to skip+reported):
  * leaf exists in core.yaml (the flat source of truth) and NOT in any
    generated source (else the merge would re-add it);
  * base exists as a canonical in fixtures AND in core.yaml;
  * leaf is NOT an EEE oracle raw and NOT a real oracle HF repo id;
  * no OTHER model has a parent edge pointing at the leaf (no dangling parent);
  * none of the alias spellings to be added collide with a DIFFERENT canonical
    id or an alias already owned by a DIFFERENT canonical (the seed CLI aborts
    on duplicate aliases);
  * (b) only: the leaf carries exactly one serving platform and it is a real
    canonical_inference_platforms id.

Dry-run by default. `--apply` writes seed/models/core.yaml. No reseed here.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "seed" / "models" / "core.yaml"
FIXTURES = ROOT / "fixtures"
SOURCES_GLOB = str(ROOT / "seed" / "models" / "sources" / "*.generated.yaml")
ORACLE = ROOT.parent / "hf_model_id_resolution.json"

# --- candidate enumeration patterns (case-insensitive, on the name-part) ------
POINTER_SUFFIX = re.compile(r"-(latest|exp|preview)$", re.I)
SKU_TOKEN = re.compile(r"[-:](high-)?(free|fast|throughput|tput)\b", re.I)
# name-"Turbo"/"Fast" real products + quant tokens -> never serving-SKU folds.
SKU_EXCLUDE = re.compile(r"(turbo|grok|imagen|-tput\b|fp8)", re.I)


def _name(cid: str) -> str:
    return cid.split("/", 1)[1] if "/" in cid else cid


def _md(e) -> dict:
    # `e` may be a dict (core.yaml entry) or a pandas Series (fixtures row) —
    # both support .get on the "metadata" key.
    m = e.get("metadata") if hasattr(e, "get") else None
    if isinstance(m, str):
        try:
            return json.loads(m) or {}
        except Exception:
            return {}
    return m if isinstance(m, dict) else {}


def _parents(value) -> list[dict]:
    if isinstance(value, str):
        try:
            return json.loads(value) or []
        except Exception:
            return []
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except (ValueError, TypeError):
        pass
    return [e for e in list(value) if isinstance(e, dict)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    # ---- live data -----------------------------------------------------------
    models = pd.read_parquet(FIXTURES / "canonical_models.parquet")
    fx_ids = set(models["id"].astype(str))
    fx_by_id = {str(r["id"]): r for _, r in models.iterrows()}

    platforms = set(
        pd.read_parquet(FIXTURES / "canonical_inference_platforms.parquet")["id"].astype(str)
    )

    adf = pd.read_parquet(FIXTURES / "aliases.parquet")
    adf = adf[adf["entity_type"] == "model"]
    alias_owner: dict[str, set] = {}
    for _, row in adf.iterrows():
        alias_owner.setdefault(str(row["raw_value"]), set()).add(str(row["canonical_id"]))

    ora = json.loads(ORACLE.read_text())["resolutions"]
    eee_raws = set(ora.keys())
    real_hf = {
        r["fixed_hf_model_id"]
        for r in ora.values()
        if r.get("hf_check_status") == "found_exact" and r.get("fixed_hf_model_id")
    }

    doc = yaml.safe_load(CORE.read_text())
    entries = doc["entries"] if isinstance(doc, dict) else doc
    core_by_id = {e["id"]: e for e in entries if isinstance(e, dict) and isinstance(e.get("id"), str)}
    core_ids = set(core_by_id)

    source_ids: set[str] = set()
    for f in glob.glob(SOURCES_GLOB):
        sd = yaml.safe_load(Path(f).read_text())
        se = sd["entries"] if isinstance(sd, dict) else sd
        for e in se or []:
            if isinstance(e, dict) and isinstance(e.get("id"), str):
                source_ids.add(e["id"])

    # child -> parent-leaf reference index (for the no-dangling guard)
    children_of: dict[str, list[str]] = {}
    for cid in fx_ids:
        for p in _parents(fx_by_id[cid]["parents"]):
            t = p.get("id")
            if isinstance(t, str):
                children_of.setdefault(t, []).append(cid)

    # ---- enumerate + classify candidates ------------------------------------
    plans = []   # (leaf, base, kind, spellings:set, platform_map:dict)
    skips = []   # (leaf, reason)

    for cid in sorted(fx_ids):
        name = _name(cid)
        is_pointer = bool(POINTER_SUFFIX.search(name))
        is_sku = bool(SKU_TOKEN.search(name)) and not SKU_EXCLUDE.search(name)
        if not (is_pointer or is_sku):
            continue

        leaf = fx_by_id[cid]
        # leaf that is itself a real release (real HF repo) is NOT a pointer/SKU fold
        if cid in real_hf:
            skips.append((cid, "leaf is itself a real oracle HF repo id (distinct release)"))
            continue

        kind = None
        base = None
        spellings: set[str] = set()
        platform_map: dict[str, str] = {}

        if is_pointer:
            # (a): require a `variant axis=version` parent edge to an existing base.
            pedges = _parents(leaf["parents"])
            version_parents = [
                p.get("id") for p in pedges
                if p.get("relationship") == "variant" and p.get("axis") == "version"
                and isinstance(p.get("id"), str)
            ]
            if not version_parents:
                skips.append((cid, "moving-pointer leaf without a version-edge base (ambiguous base — not guessed)"))
                continue
            if len(version_parents) > 1:
                skips.append((cid, f"multiple version-edge parents {version_parents} (ambiguous)"))
                continue
            base = version_parents[0]
            kind = "pointer"
            # preserve any serving-platform provenance the pointer leaf carried
            # (e.g. moonshotai-cn) onto the base — losing it on delete would drop
            # a real platform→spelling mapping. Only keep platform ids that are
            # real canonical inference_platforms.
            lap = (_md(leaf).get("alias_platforms") or {})
            platform_map = {k: v for k, v in lap.items() if v in platforms}
            spellings |= set(platform_map)
        else:
            # (b): serving SKU over a base. Require exactly one concrete platform
            # and resolve the base via the leaf's underlying spelling.
            md = _md(leaf)
            ap_map = md.get("alias_platforms") or {}
            providers = md.get("providers") or []
            plats = set(ap_map.values()) | set(providers)
            if len(plats) != 1:
                skips.append((cid, f"serving-SKU platform not single/concrete: {sorted(plats)}"))
                continue
            platform = next(iter(plats))
            if platform not in platforms:
                skips.append((cid, f"serving-SKU platform {platform!r} not a canonical inference_platform"))
                continue
            # derive base id by stripping the SKU token from the name-part and
            # matching against an existing canonical (org-prefixed).
            org = cid.split("/", 1)[0] if "/" in cid else None
            base_name = SKU_TOKEN.sub("", name).rstrip("-:")
            cand_bases = []
            if org:
                # try the exact-cased base, plus the version-dot form (m2-5 -> M2.5)
                cand_bases += [f"{org}/{base_name}"]
            # also: normalized match against any canonical (cosmetic dot/dash + case)
            def norm(s: str) -> str:
                s = s.lower()
                s = re.sub(r"(\d)[-_](\d)", r"\1.\2", s)  # m2-5 -> m2.5
                return re.sub(r"[-_/]+", "", s)
            want = norm(f"{org}/{base_name}") if org else norm(base_name)
            matches = [b for b in fx_ids if b != cid and norm(b) == want]
            for b in cand_bases:
                if b in fx_ids and b not in matches:
                    matches.append(b)
            if len(matches) != 1:
                skips.append((cid, f"serving-SKU base ambiguous/missing for {base_name!r}: {matches[:5]}"))
                continue
            base = matches[0]
            kind = "sku"
            platform_map = {k: platform for k in ap_map}  # migrate the underlying spellings
            spellings |= set(ap_map)

        # ---- shared validations for a kept fold ------------------------------
        spellings |= {cid, leaf["display_name"]}
        spellings = {s for s in spellings if isinstance(s, str) and s}

        bad = None
        if cid in eee_raws:
            bad = "leaf is an EEE oracle raw (removing it would change coverage)"
        elif cid not in core_ids:
            bad = "leaf not in core.yaml source of truth"
        elif cid in source_ids:
            bad = "leaf also defined in a generated source (merge would re-add it)"
        elif base not in fx_ids:
            bad = f"base {base!r} not a canonical in fixtures"
        elif base not in core_ids:
            bad = f"base {base!r} not in core.yaml"
        elif [c for c in children_of.get(cid, []) if c != cid]:
            bad = f"other models parent on the leaf (would dangle): {children_of[cid][:5]}"
        else:
            for s in spellings:
                if s in fx_ids and s != cid and s != base:
                    bad = f"alias spelling {s!r} is a different canonical id"
                    break
                owners = alias_owner.get(s, set()) - {cid, base}
                if owners:
                    bad = f"alias spelling {s!r} owned by other canonical(s) {sorted(owners)}"
                    break
        if bad:
            skips.append((cid, bad))
            continue

        plans.append((cid, base, kind, spellings, platform_map))

    # ---- report --------------------------------------------------------------
    n_a = sum(1 for p in plans if p[2] == "pointer")
    n_b = sum(1 for p in plans if p[2] == "sku")
    print(f"PLANNED folds: {len(plans)}  (a/pointer={n_a}, b/serving-SKU={n_b})")
    for leaf, base, kind, spellings, pmap in plans:
        extra = f"  +platform {sorted(set(pmap.values()))}" if pmap else ""
        print(f"  FOLD[{kind}] {leaf}  ->  {base}{extra}")
        print(f"        add aliases: {sorted(spellings - {base})}")
    print(f"\nSKIPPED candidates: {len(skips)}")
    for leaf, reason in skips:
        print(f"  SKIP {leaf} :: {reason}")

    if not args.apply:
        print("\n(dry-run — no files written; no reseed)")
        return 0

    # ---- apply: edit core.yaml ----------------------------------------------
    fold_leaves = {leaf for leaf, *_ in plans}
    for leaf, base, kind, spellings, pmap in plans:
        be = core_by_id[base]
        al = set(be.get("aliases") or [])
        al |= (spellings - {base})
        be["aliases"] = sorted(al)
        if pmap:
            md = _md(be)
            apm = md.get("alias_platforms") or {}
            apm.update(pmap)
            md["alias_platforms"] = apm
            be["metadata"] = json.dumps(md, sort_keys=True)

    new_entries = [e for e in entries if not (isinstance(e, dict) and e.get("id") in fold_leaves)]
    # guard: removing the leaves must not dangle a parent edge anywhere
    dang = [
        (e["id"], p["id"]) for e in new_entries if isinstance(e, dict)
        for p in _parents(e.get("parents")) if isinstance(p, dict) and p.get("id") in fold_leaves
    ]
    if dang:
        print("ABORT dangling parent edges after removal:", dang)
        return 2

    if isinstance(doc, dict):
        doc["entries"] = new_entries
    else:
        doc = new_entries
    CORE.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=10_000))
    print(f"\nAPPLIED {len(plans)} folds to core.yaml (removed {len(fold_leaves)} leaves).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
