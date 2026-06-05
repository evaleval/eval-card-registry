#!/usr/bin/env python3
"""Fold models.dev-minted duplicate canonicals into their real HF id.

Generalizes the two hand-folds already in core.yaml (Qwen/QwQ-32B,
LiquidAI/LFM2-24B-A2B).

For every `resolution_source: models_dev` canonical in seed/models/core.yaml,
decide whether a *real HF id* already exists for the SAME model. Authority:
  - the frozen HF oracle (hf_model_id_resolution.json): an entry with
    resolution_status in {fixed_exact, fixed_near_miss} carries a
    `fixed_hf_model_id` = a real HF repo id; and
  - the HF-true canonicals already present in core.yaml (resolution_source: hf,
    or ids that ARE oracle fixed ids).

A fold is proposed ONLY on a confident match:
  - exact     : the mint id (or one of its aliases) equals an HF target id;
  - alias     : the mint id / a mint alias is already declared as an alias of an
                HF target (the registry already links them);
  - normalized: normalized-name equality, CORROBORATED BY ORG AGREEMENT after the
                curated dev-org remap (meta-llama->meta, qwen->alibaba, ...);
  - fuzzy     : org agreement + normalized-name equality only AFTER stripping the
                brand prefix from the mint name (the models.dev key carries the
                brand, e.g. alibaba/qwen-qwq-32b vs real Qwen/QwQ-32B).

NO FALSE MERGES: a name-only match across DIFFERENT developers is never folded.
When org agreement cannot be established, the fold is NOT proposed.

--apply (NOT run by the dry-run pass): given a list of confirmed mint ids, deletes each
mint entry, merges providers/release_date/open_weights onto the HF target,
appends the mint id + display_name as aliases on the target, and rewrites any
parents[].id edge pointing at a folded id to the HF target.

Usage:
    LOCAL_MODE=true uv run python scripts/fold_modelsdev_dupes.py            # dry run
    LOCAL_MODE=true uv run python scripts/fold_modelsdev_dupes.py --json     # dry run, JSON out
    LOCAL_MODE=true uv run python scripts/fold_modelsdev_dupes.py --apply --confirm <id> [<id> ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import yaml

from eval_entity_resolver.normalization import normalize as nz
from eval_entity_resolver.strategies.fuzzy import _ORG_ALIASES

REGISTRY_ROOT = Path(__file__).resolve().parents[1]
EVALEVAL_ROOT = REGISTRY_ROOT.parent
ORACLE = EVALEVAL_ROOT / "hf_model_id_resolution.json"
CORE_YAML = REGISTRY_ROOT / "seed" / "models" / "core.yaml"
ORGS_YAML = REGISTRY_ROOT / "seed" / "orgs.yaml"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_core() -> tuple[dict, list[dict]]:
    doc = yaml.safe_load(CORE_YAML.read_text())
    if isinstance(doc, dict):
        return doc, doc.get("entries", []) or []
    return {"entries": doc}, doc


def build_hf_to_dev() -> dict[str, str]:
    """HF-org-lowercase -> curated developer slug (single-sourced, same rule the
    generators use). Curated seed/orgs.yaml hf_org wins on conflict."""
    hf_to_dev = {k.lower(): v for k, v in _ORG_ALIASES.items()}
    if ORGS_YAML.exists():
        for e in yaml.safe_load(ORGS_YAML.read_text()) or []:
            if isinstance(e, dict) and isinstance(e.get("hf_org"), str) and isinstance(e.get("id"), str):
                if e["hf_org"].strip():
                    hf_to_dev[e["hf_org"].lower()] = e["id"]
    return hf_to_dev


def dev_org_of_prefix(prefix: str, hf_to_dev: dict[str, str]) -> str:
    return hf_to_dev.get(prefix.lower(), prefix)


# ---------------------------------------------------------------------------
# Name normalization helpers (brand-prefix stripping for the fuzzy tier)
# ---------------------------------------------------------------------------
def brand_tokens_for(dev_org: str, hf_to_dev: dict[str, str]) -> set[str]:
    """Brand tokens that a models.dev key may glue onto a model name for this
    developer. Includes the dev slug itself plus every HF alias that maps TO it
    (e.g. dev `alibaba` -> {alibaba, qwen, alibaba-nlp, ...}; dev `meta` ->
    {meta, meta-llama, facebook}). Normalized to single tokens (no separators)."""
    toks: set[str] = set()
    d = dev_org.lower()
    toks.add(nz(d).replace(" ", ""))
    for hf_alias, dev in hf_to_dev.items():
        if dev.lower() == d:
            toks.add(nz(hf_alias).replace(" ", ""))
    # common brand spelling variants not in the org map
    extra = {
        "alibaba": {"qwen"},
        "meta": {"llama"},
        "minimax": {"minimax"},
        "google": {"gemini", "gemma"},
        "deepseek": {"deepseek"},
    }
    toks |= extra.get(d, set())
    return {t for t in toks if t}


def name_norm(name: str) -> str:
    """Normalized name with separators collapsed AND removed (one token)."""
    return nz(name).replace(" ", "")


def strip_brand_prefix(norm_name_tok: str, brands: set[str]) -> set[str]:
    """Return candidate name tokens with a leading brand token removed.
    Always includes the original. Strips repeatedly (qwen-qwen-... defensive)."""
    out = {norm_name_tok}
    changed = True
    cur = norm_name_tok
    while changed:
        changed = False
        for b in sorted(brands, key=len, reverse=True):
            if b and cur.startswith(b) and len(cur) > len(b):
                cur = cur[len(b):]
                out.add(cur)
                changed = True
                break
    return out


# ---------------------------------------------------------------------------
# Build HF target authority
# ---------------------------------------------------------------------------
def build_hf_targets(entries: list[dict], hf_to_dev: dict[str, str]):
    """Returns:
      - hf_ids: set of all real-HF canonical ids (entries that are hf-source or
        oracle-fixed) — used for exact id match.
      - alias_to_hf: every alias/display/id string declared on an HF entry ->
        that HF id (for the 'alias' tier).
      - by_org_name: (dev_org, name_token) -> hf_id  for normalized matching.
      - hf_entry_by_id: id -> entry dict (so --apply can merge onto it).
    """
    oracle = json.loads(ORACLE.read_text())["resolutions"]
    fixed_ids: set[str] = set()
    for v in oracle.values():
        if v.get("resolution_status") in ("fixed_exact", "fixed_near_miss"):
            fx = v.get("fixed_hf_model_id")
            if isinstance(fx, str) and "/" in fx:
                fixed_ids.add(fx)

    hf_entry_by_id: dict[str, dict] = {}
    hf_ids: set[str] = set(fixed_ids)
    for e in entries:
        if not isinstance(e, dict):
            continue
        cid = e.get("id")
        if not isinstance(cid, str):
            continue
        is_hf = e.get("resolution_source") == "hf" or cid in fixed_ids
        if is_hf:
            hf_ids.add(cid)
            hf_entry_by_id[cid] = e

    alias_to_hf: dict[str, str] = {}
    by_org_name: dict[tuple[str, str], str] = {}

    def index_target(cid: str, entry: Optional[dict]):
        if "/" not in cid:
            return
        org, name = cid.split("/", 1)
        dev = dev_org_of_prefix(org, hf_to_dev)
        by_org_name.setdefault((dev, name_norm(name)), cid)
        # alias strings (exact, case-sensitive registry linkage)
        alias_to_hf.setdefault(cid, cid)
        if entry is not None:
            dn = entry.get("display_name")
            if isinstance(dn, str):
                alias_to_hf.setdefault(dn, cid)
            for a in entry.get("aliases") or []:
                if isinstance(a, str):
                    alias_to_hf.setdefault(a, cid)

    # entries first (so a present entry's aliases are indexed), then bare oracle ids
    for cid, e in hf_entry_by_id.items():
        index_target(cid, e)
    for cid in fixed_ids:
        if cid not in hf_entry_by_id:
            index_target(cid, None)

    return hf_ids, alias_to_hf, by_org_name, hf_entry_by_id


# ---------------------------------------------------------------------------
# Match decision for one mint
# ---------------------------------------------------------------------------
def decide_fold(mint: dict, hf_ids, alias_to_hf, by_org_name, hf_to_dev):
    """Return a fold dict or None. Confident-match tiers only."""
    cid = mint.get("id")
    if not isinstance(cid, str):
        return None
    mint_org = mint.get("org_id")
    mint_org = mint_org if isinstance(mint_org, str) and mint_org else None
    # org from the id prefix, remapped, as a fallback when org_id is unset
    prefix_org = None
    if "/" in cid:
        prefix_org = dev_org_of_prefix(cid.split("/", 1)[0], hf_to_dev)
    eff_org = mint_org or prefix_org

    mint_strings = [cid]
    dn = mint.get("display_name")
    if isinstance(dn, str):
        mint_strings.append(dn)
    for a in mint.get("aliases") or []:
        if isinstance(a, str):
            mint_strings.append(a)

    # --- tier: exact id equality (mint id or alias == an HF id) -------------
    for s in mint_strings:
        if s in hf_ids and s != cid:
            return _mk(mint, s, "exact", eff_org, hf_to_dev,
                       f"mint string {s!r} is itself a real HF id")

    # --- tier: alias linkage (mint string already an alias of an HF entry) --
    for s in mint_strings:
        tgt = alias_to_hf.get(s)
        if tgt and tgt != cid and tgt in hf_ids:
            return _mk(mint, tgt, "alias", eff_org, hf_to_dev,
                       f"mint string {s!r} already declared on HF target {tgt!r}")

    if eff_org is None:
        return None  # cannot establish org agreement -> never fold (no false merge)

    # candidate names from the mint id + aliases (name part only)
    cand_names: set[str] = set()
    for s in mint_strings:
        nm = s.split("/", 1)[1] if "/" in s else s
        cand_names.add(name_norm(nm))

    # --- tier: normalized-name equality + ORG AGREEMENT --------------------
    for nm in cand_names:
        tgt = by_org_name.get((eff_org, nm))
        if tgt and tgt != cid:
            return _mk(mint, tgt, "normalized", eff_org, hf_to_dev,
                       f"org={eff_org} + normalized name {nm!r} == HF target {tgt!r}")

    # --- tier: fuzzy (brand-prefix-stripped) + ORG AGREEMENT ---------------
    brands = brand_tokens_for(eff_org, hf_to_dev)
    stripped: set[str] = set()
    for nm in cand_names:
        stripped |= strip_brand_prefix(nm, brands)
    stripped -= cand_names  # only the genuinely-stripped variants
    for nm in stripped:
        tgt = by_org_name.get((eff_org, nm))
        if tgt and tgt != cid:
            return _mk(mint, tgt, "fuzzy", eff_org, hf_to_dev,
                       f"org={eff_org} + brand-stripped name {nm!r} == HF target {tgt!r}")

    return None


def _mk(mint, hf_target, match_type, mint_dev_org, hf_to_dev, evidence):
    hf_org = (hf_to_dev.get(hf_target.split("/", 1)[0].lower(), hf_target.split("/", 1)[0])
              if "/" in hf_target else hf_target)
    return {
        "mint_id": mint["id"],
        "hf_target": hf_target,
        "match_type": match_type,
        "mint_org": mint_dev_org or "",
        "hf_org": hf_org,
        "org_agreement": (mint_dev_org or "").lower() == (hf_org or "").lower(),
        "evidence": evidence,
    }


# ---------------------------------------------------------------------------
# parents edge detection
# ---------------------------------------------------------------------------
def parent_refs(entries: list[dict]) -> dict[str, list[str]]:
    """id -> list of entry ids that reference it as a parents[].id edge."""
    refs: dict[str, list[str]] = {}
    for e in entries:
        if not isinstance(e, dict):
            continue
        for p in e.get("parents") or []:
            if isinstance(p, dict) and isinstance(p.get("id"), str):
                refs.setdefault(p["id"], []).append(e.get("id"))
    return refs


# ---------------------------------------------------------------------------
# --apply
# ---------------------------------------------------------------------------
def apply_folds(doc: dict, entries: list[dict], confirm_ids: set[str], folds: list[dict]):
    """Delete each confirmed mint; merge metadata onto the HF target; append the
    mint id + display_name as aliases; rewrite parents edges. Mutates entries."""
    fold_by_mint = {f["mint_id"]: f for f in folds if f["mint_id"] in confirm_ids}
    by_id = {e["id"]: e for e in entries if isinstance(e, dict) and isinstance(e.get("id"), str)}

    rename: dict[str, str] = {}
    for mint_id, f in fold_by_mint.items():
        tgt = f["hf_target"]
        mint = by_id.get(mint_id)
        target = by_id.get(tgt)
        if mint is None or target is None:
            print(f"  SKIP {mint_id}: mint or target missing in entries", file=sys.stderr)
            continue
        rename[mint_id] = tgt

        # merge metadata: providers (union), release_date (fill), open_weights (fill)
        tmd = _load_md(target)
        mmd = _load_md(mint)
        tp = list(tmd.get("providers") or [])
        for p in mmd.get("providers") or []:
            if p not in tp:
                tp.append(p)
        if tp:
            tmd["providers"] = tp
        target["metadata"] = json.dumps(tmd)
        if not target.get("release_date") and mint.get("release_date"):
            target["release_date"] = mint["release_date"]
        if target.get("open_weights") is None and mint.get("open_weights") is not None:
            target["open_weights"] = mint["open_weights"]

        # append mint id + display_name as aliases
        aliases = list(target.get("aliases") or [])
        seen = {a.lower() for a in aliases if isinstance(a, str)}
        for extra in [mint_id, mint.get("display_name"), *(mint.get("aliases") or [])]:
            if isinstance(extra, str) and extra and extra != tgt and extra.lower() not in seen:
                aliases.append(extra)
                seen.add(extra.lower())
        target["aliases"] = aliases

    # delete folded mints
    kept = [e for e in entries if not (isinstance(e, dict) and e.get("id") in rename)]

    # rewrite parents edges pointing at a folded id
    for e in kept:
        if not isinstance(e, dict):
            continue
        for p in e.get("parents") or []:
            if isinstance(p, dict) and isinstance(p.get("id"), str) and p["id"] in rename:
                p["id"] = rename[p["id"]]

    if isinstance(doc, dict):
        doc["entries"] = kept
    CORE_YAML.write_text(
        yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=10_000)
    )
    print(f"applied {len(rename)} folds; core.yaml now {len(kept)} entries")


def _load_md(entry: dict) -> dict:
    md = entry.get("metadata")
    if isinstance(md, str):
        try:
            return json.loads(md) or {}
        except Exception:
            return {}
    if isinstance(md, dict):
        return md
    return {}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--confirm", nargs="*", default=[])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    doc, entries = load_core()
    hf_to_dev = build_hf_to_dev()
    mints = [e for e in entries if isinstance(e, dict) and e.get("resolution_source") == "models_dev"]
    hf_ids, alias_to_hf, by_org_name, _hf_by_id = build_hf_targets(entries, hf_to_dev)

    folds = []
    for m in mints:
        f = decide_fold(m, hf_ids, alias_to_hf, by_org_name, hf_to_dev)
        if f is not None:
            folds.append(f)

    # parent-edge detection
    refs = parent_refs(entries)
    fold_ids = {f["mint_id"] for f in folds}
    for f in folds:
        f["referenced_as_parent"] = f["mint_id"] in refs
    parent_referenced = sorted(fid for fid in fold_ids if fid in refs)

    if args.apply:
        confirm = set(args.confirm)
        if not confirm:
            print("--apply requires --confirm <id> [...]", file=sys.stderr)
            sys.exit(2)
        apply_folds(doc, entries, confirm, folds)
        return

    report = {
        "candidates_examined": len(mints),
        "fold_count": len(folds),
        "no_fold_count": len(mints) - len(folds),
        "folds": folds,
        "parent_referenced_fold_ids": parent_referenced,
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"candidates_examined: {report['candidates_examined']}")
        print(f"fold_count:          {report['fold_count']}")
        print(f"no_fold_count:       {report['no_fold_count']}")
        print(f"parent_referenced:   {parent_referenced}")
        from collections import Counter
        print("by match_type:", dict(Counter(f["match_type"] for f in folds)))
        for f in folds:
            print(f"  {f['mint_id']}  ->  {f['hf_target']}  [{f['match_type']}] "
                  f"org {f['mint_org']}=={f['hf_org']}? {f['org_agreement']} "
                  f"parent_ref={f['referenced_as_parent']}  :: {f['evidence']}")


if __name__ == "__main__":
    main()
