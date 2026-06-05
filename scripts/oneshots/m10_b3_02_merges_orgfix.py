#!/usr/bin/env python3
"""Merge under-spec dupes + revert the multi-size-split over-reach + baichuan lab fix.

A. REVERT the multi-size-split over-reach: drop the global family/bare Ollama
   aliases that wrongly mapped a generic name onto one size leaf (asymmetry: gemma-3-4b-it
   IS-A gemma-3, but `gemma-3` is not gemma-3-4b-it). Keep only `:Nb` size tags.
B. MERGE the 6 ID-based under-spec dupes (survivor = HF-true id; lossy id ->
   alias; migrate aliases/display/alias_platforms; rewire parent edges).
C. BAICHUAN as a curated lab: repoint every baichuan/baichuan-inc model to
   org_id=baichuan, drop the generated baichuan-inc community row. (The curated
   lab row in orgs.yaml + the _ORG_ALIASES flip are applied separately by Edit.)

Dry-run by default; --apply writes core.yaml + orgs.generated.yaml.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "seed" / "models" / "core.yaml"
ORGS_GEN = ROOT / "seed" / "orgs.generated.yaml"

REVERT = {  # leaf -> aliases to remove (global/family tags wrongly attached)
    "google/gemma-3-4b-it": ["gemma3", "gemma-3", "google/gemma3"],
    "openai/gpt-oss-20b": ["gpt-oss"],
}
MERGES = [  # (lossy id, survivor HF-true id) — all ID-based, not display-based
    ("ai2/olmo-3-1-32b-instruct", "allenai/Olmo-3.1-32B-Instruct"),
    ("cohere/AyaExpanse-32B", "CohereLabs/aya-expanse-32b"),
    ("cohere/AyaExpanse-8B", "CohereLabs/aya-expanse-8b"),
    ("meta/llama-3.1-nemotron-70b-instruct", "nvidia/Llama-3.1-Nemotron-70B-Instruct"),
    ("meta/llama-3.1-nemotron-70b-instruct-hf", "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"),
    ("alibaba/qwen3-235b-a22b-instruct-2507-tput", "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"),
]
ORG_REPOINT_VALUES = {"baichuan", "baichuan-inc"}
ORG_TARGET = "baichuan"  # the curated lab


def _md(e):
    m = e.get("metadata")
    if isinstance(m, str):
        try: return json.loads(m) or {}
        except Exception: return {}
    return m or {}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    doc = yaml.safe_load(CORE.read_text())
    entries = doc["entries"] if isinstance(doc, dict) else doc
    by = {e["id"]: e for e in entries if isinstance(e, dict) and isinstance(e.get("id"), str)}
    log = []

    # A. revert the multi-size-split over-reach
    for leaf, drop in REVERT.items():
        e = by.get(leaf)
        if not e: continue
        al = [a for a in (e.get("aliases") or []) if a not in drop]
        if len(al) != len(e.get("aliases") or []):
            log.append(f"REVERT {leaf}: drop aliases {drop}")
        e["aliases"] = al
        md = _md(e); ap_map = md.get("alias_platforms") or {}
        for d in drop: ap_map.pop(d, None)
        md["alias_platforms"] = ap_map; e["metadata"] = json.dumps(md, sort_keys=True)

    # B. merges
    rename = {}
    for lossy, surv in MERGES:
        el, es = by.get(lossy), by.get(surv)
        if el is None or es is None:
            log.append(f"SKIP merge {lossy} -> {surv} (missing: lossy={el is not None} surv={es is not None})")
            continue
        rename[lossy] = surv
        al = set(es.get("aliases") or [])
        for extra in [lossy, el.get("display_name"), *(el.get("aliases") or [])]:
            if isinstance(extra, str) and extra and extra != surv:
                al.add(extra)
        es["aliases"] = sorted(al)
        # migrate alias_platforms + fill release/open_weights if survivor lacks
        msd, mld = _md(es), _md(el)
        ap_s = msd.get("alias_platforms") or {}; ap_s.update(mld.get("alias_platforms") or {})
        if ap_s: msd["alias_platforms"] = ap_s
        es["metadata"] = json.dumps(msd, sort_keys=True)
        if not es.get("release_date") and el.get("release_date"): es["release_date"] = el["release_date"]
        if es.get("open_weights") is None and el.get("open_weights") is not None: es["open_weights"] = el["open_weights"]
        log.append(f"MERGE {lossy} -> {surv} (survivor org_id={es.get('org_id')})")

    new_entries = []
    for e in entries:
        if isinstance(e, dict) and e.get("id") in rename:
            continue  # delete lossy
        if isinstance(e, dict):
            for p in (e.get("parents") or []):
                if isinstance(p, dict) and p.get("id") in rename:
                    p["id"] = rename[p["id"]]
            # C. baichuan org repoint
            if e.get("org_id") in ORG_REPOINT_VALUES and e["org_id"] != ORG_TARGET:
                log.append(f"ORG {e['id']}: {e['org_id']} -> {ORG_TARGET}")
                e["org_id"] = ORG_TARGET
        new_entries.append(e)

    # guard: no dangling parent to a deleted lossy id
    dang = [(e["id"], p["id"]) for e in new_entries if isinstance(e, dict)
            for p in (e.get("parents") or []) if isinstance(p, dict) and p.get("id") in rename]
    if dang:
        print("ABORT dangling:", dang); return 2

    # remove generated baichuan-inc community row
    og = yaml.safe_load(ORGS_GEN.read_text()) if ORGS_GEN.exists() else []
    og_list = og if isinstance(og, list) else (og or [])
    kept_orgs = [o for o in og_list if not (isinstance(o, dict) and o.get("id") == "baichuan-inc")]
    if len(kept_orgs) != len(og_list):
        log.append("drop generated org row baichuan-inc (superseded by curated lab)")

    print(f"{len(log)} changes:")
    for l in log: print("  ", l)
    if args.apply:
        if isinstance(doc, dict): doc["entries"] = new_entries
        else: doc = new_entries
        CORE.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=10_000))
        ORGS_GEN.write_text(
            (ORGS_GEN.read_text().split("\n- ", 1)[0].rstrip() + "\n")
            + yaml.safe_dump(kept_orgs, sort_keys=False, allow_unicode=True, width=10_000)
        )
        print("APPLIED.")
    else:
        print("(dry-run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
