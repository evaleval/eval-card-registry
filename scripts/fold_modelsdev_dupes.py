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
# Org-aware fold decision now lives in the resolver package (single source of
# truth shared with scripts/refresh_from_modelsdev.py and the gate). Re-exported
# below so existing importers (tests/test_gate_invariants.py) keep working.
from eval_entity_resolver.fold import (  # noqa: F401
    brand_tokens_for,
    build_hf_index,
    decide_fold,
    dev_org_of_prefix,
    name_norm,
    strip_brand_prefix,
)

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
    """HF-org-lowercase -> curated developer slug via the SINGLE shared builder
    `fold.build_curated_org_map` (`_ORG_ALIASES` UNION every curated org's id /
    hf_org / ALIASES). Reading the alias tier (not just hf_org) is what folds
    minimaxai->minimax, EnnoAi->Enno-Ai, etc. — so the dedup/shadow predicate
    here agrees with the generators + resolver + gate (no divergent weaker map)."""
    from eval_entity_resolver.fold import build_curated_org_map

    orgs = yaml.safe_load(ORGS_YAML.read_text()) or [] if ORGS_YAML.exists() else []
    return build_curated_org_map(orgs)


# ---------------------------------------------------------------------------
# Build HF target authority (thin wrapper: read the frozen oracle's fixed ids,
# delegate to the shared eval_entity_resolver.fold.build_hf_index).
# ---------------------------------------------------------------------------
def build_hf_targets(entries: list[dict], hf_to_dev: dict[str, str]):
    """Returns (hf_ids, alias_to_hf, by_org_name, hf_entry_by_id) — see
    eval_entity_resolver.fold.build_hf_index. Adds the frozen oracle's
    fixed_exact/near_miss HF ids as extra real-HF targets."""
    oracle = json.loads(ORACLE.read_text())["resolutions"]
    fixed_ids = frozenset(
        v["fixed_hf_model_id"]
        for v in oracle.values()
        if v.get("resolution_status") in ("fixed_exact", "fixed_near_miss")
        and isinstance(v.get("fixed_hf_model_id"), str) and "/" in v["fixed_hf_model_id"]
    )
    return build_hf_index(entries, hf_to_dev, fixed_ids)


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
