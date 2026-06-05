#!/usr/bin/env python3
"""Clean up malformed (display-form) canonical IDs — ids that contain spaces /
colons / '+' and resolve to themselves on a junk node instead of the real model,
silently fragmenting comparability.

Three operations:
  MERGES  — a real model wearing a junk id, folded into its existing clean
            canonical (junk id preserved as an alias so the EEE string keeps
            resolving; parent edges rewritten; junk entry dropped).
  QWEN    — mint a properly-formatted generic Qwen/Qwen2.5 group node, fold the
            junk `alibaba/Qwen 2.5` into it, and link Qwen2.5-72B-Instruct to it
            via variant/size (shared model_family_id).
  REKEYS  — agent/scaffold systems mis-attributed to a real vendor (openai/meta)
            moved to unknown/ (org unresolved; still their own distinct entry).

Agent systems already under unknown/ and applied-compute/* are left as-is.
DETERMINISTIC. Dry-run by default; --apply writes core.yaml.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "seed" / "models" / "core.yaml"

# junk canonical -> existing clean canonical to fold into
MERGES = {
    "anthropic/Opus 4.1": "anthropic/claude-opus-4.1",
    "anthropic/Sonnet 4.5": "anthropic/claude-sonnet-4.5",
    "amazon/amazon.nova-lite-v1:0": "amazon/nova-lite-v1",
    "amazon/amazon.nova-micro-v1:0": "amazon/nova-micro-v1",
    "amazon/amazon.nova-pro-v1:0": "amazon/nova-pro-v1",
    "amazon/nova-premier-v1:0": "amazon/nova-premier-v1",
    "meta/llama-3-3+": "meta-llama/llama-3.3",
}
# agent systems mis-prefixed to a real vendor -> unknown/ (preserve old as alias)
REKEYS = {
    "openai/Lingma Agent + Lingma SWE-GPT 72b (v0918)": "unknown/Lingma Agent + Lingma SWE-GPT 72b (v0918)",
    "openai/Lingma Agent + Lingma SWE-GPT 72b (v0925)": "unknown/Lingma Agent + Lingma SWE-GPT 72b (v0925)",
    "openai/Lingma Agent + Lingma SWE-GPT 7b (v0918)": "unknown/Lingma Agent + Lingma SWE-GPT 7b (v0918)",
    "openai/Lingma Agent + Lingma SWE-GPT 7b (v0925)": "unknown/Lingma Agent + Lingma SWE-GPT 7b (v0925)",
    "meta/RAG + SWE-Llama 13B": "unknown/RAG + SWE-Llama 13B",
}
QWEN_GENERIC = "Qwen/Qwen2.5"
QWEN_JUNK = "alibaba/Qwen 2.5"
QWEN_SIZE_CHILD = "Qwen/Qwen2.5-72B-Instruct"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    doc = yaml.safe_load(CORE.read_text())
    entries = doc["entries"] if isinstance(doc, dict) else doc
    by_id = {e["id"]: e for e in entries if isinstance(e, dict) and isinstance(e.get("id"), str)}

    errors: list[str] = []
    actions: list[str] = []

    def merge(junk: str, clean: str) -> None:
        j, c = by_id.get(junk), by_id.get(clean)
        if j is None:
            errors.append(f"merge source missing: {junk}")
            return
        if c is None:
            errors.append(f"merge target missing: {clean}")
            return
        cal = c.get("aliases") or []
        for a in [junk, *(j.get("aliases") or [])]:
            if a not in cal and a != clean:
                cal.append(a)
        c["aliases"] = cal
        entries.remove(j)
        by_id.pop(junk, None)
        actions.append(f"MERGE {junk!r} -> {clean}")

    # rewrite-edge helper used after id removals/renames
    def rewrite_edges(idmap: dict) -> int:
        n = 0
        for e in entries:
            if not isinstance(e, dict):
                continue
            for p in (e.get("parents") or []):
                if isinstance(p, dict) and p.get("id") in idmap:
                    p["id"] = idmap[p["id"]]
                    n += 1
        return n

    for junk, clean in MERGES.items():
        merge(junk, clean)

    # QWEN: mint generic, fold junk, link 72B-Instruct
    if QWEN_GENERIC in by_id:
        errors.append(f"qwen generic already exists: {QWEN_GENERIC}")
    elif by_id.get(QWEN_JUNK) is None:
        errors.append(f"qwen junk missing: {QWEN_JUNK}")
    elif QWEN_SIZE_CHILD not in by_id:
        errors.append(f"qwen size child missing: {QWEN_SIZE_CHILD}")
    else:
        jq = by_id[QWEN_JUNK]
        new_aliases = [a for a in (jq.get("aliases") or []) if a != QWEN_GENERIC]
        if QWEN_JUNK not in new_aliases:
            new_aliases.append(QWEN_JUNK)
        entries.append({
            "id": QWEN_GENERIC, "display_name": QWEN_GENERIC, "org_id": None,
            "family": None, "architecture": None, "params_billions": None,
            "parents": [], "open_weights": True, "release_date": None,
            "input_modalities": None, "output_modalities": None, "tags": [],
            "metadata": "{}", "review_status": "reviewed", "aliases": new_aliases,
        })
        entries.remove(jq)
        by_id.pop(QWEN_JUNK, None)
        by_id[QWEN_GENERIC] = entries[-1]
        child = by_id[QWEN_SIZE_CHILD]
        cp = child.get("parents") or []
        if not any(isinstance(p, dict) and p.get("id") == QWEN_GENERIC for p in cp):
            cp.append({"id": QWEN_GENERIC, "relationship": "variant", "axis": "size"})
            child["parents"] = cp
        actions.append(f"MINT {QWEN_GENERIC} (fold {QWEN_JUNK!r}); {QWEN_SIZE_CHILD} +variant/size -> {QWEN_GENERIC}")

    # REKEYS: rename id, preserve old as alias
    for old, new in REKEYS.items():
        e = by_id.get(old)
        if e is None:
            errors.append(f"rekey source missing: {old}")
            continue
        if new in by_id:
            errors.append(f"rekey target exists: {new}")
            continue
        e["id"] = new
        e["display_name"] = new
        al = e.get("aliases") or []
        if old not in al:
            al.append(old)
        e["aliases"] = al
        by_id[new] = e
        by_id.pop(old, None)
        actions.append(f"REKEY {old!r} -> {new}")

    fixed = rewrite_edges({**MERGES, **REKEYS})

    for a in actions:
        print(f"   {a}")
    print(f"   (parent edges rewritten: {fixed})")
    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors:
            print(f"   ! {e}")
        print("ABORT.")
        return 1
    if not args.apply:
        print(f"\n[{len(actions)} actions] (dry-run)")
        return 0
    if isinstance(doc, dict):
        doc["entries"] = entries
    CORE.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=10_000))
    print(f"\nAPPLIED {len(actions)} actions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
