#!/usr/bin/env python3
"""Repoint alias-sweep cases where the CORRECT canonical already exists.

Each alias below currently folds onto the wrong canonical, while the model it
actually names is already its own canonical in the registry. Move the alias to
the correct target (verified to exist). These are the unambiguous subset of the
sweep findings — no minting, no judgment; the target is a real distinct model
already present.

DETERMINISTIC. Dry-run by default; --apply writes core.yaml. Validates that each
target exists, the alias is currently present, and no duplicate alias results.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "seed" / "models" / "core.yaml"

# alias -> correct existing canonical it should resolve to
REPOINTS = {
    "unknown/starling-lm-7b-alpha": "berkeley-nest/Starling-LM-7B-alpha",   # LM, not the reward model
    "pku-alignment/beaver-7b-v1-0-cost": "PKU-Alignment/beaver-7b-v1.0-cost",  # versioned cost model exists
    "pku-alignment/beaver-7b-v2-0-cost": "PKU-Alignment/beaver-7b-v2.0-cost",
    "unknown/gemini-1-5-flash-8b": "google/gemini-1.5-flash-8b",            # distinct 8B variant
    "unknown/claude-3-5-sonnet-oct": "anthropic/claude-3.5-sonnet",         # 3.5, not 3-sonnet
    "unknown/internvl-2-8b": "OpenGVLab/internvl2-8b",                      # InternVL2, not 2.5
    "unknown/gemini-2-5-flash-no-thinking": "google/gemini-2.5-flash",      # non-thinking flash
    "moonshot.kimi-k2": "moonshotai/Kimi-K2-Instruct",                      # Kimi K2, not the thinking variant
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    doc = yaml.safe_load(CORE.read_text())
    entries = doc["entries"] if isinstance(doc, dict) else doc
    by_id = {e["id"]: e for e in entries if isinstance(e, dict) and isinstance(e.get("id"), str)}

    errors: list[str] = []
    actions: list[str] = []
    for alias, target in REPOINTS.items():
        if target not in by_id:
            errors.append(f"target missing: {target} (for {alias})")
            continue
        owners = [e for e in entries if isinstance(e, dict) and alias in (e.get("aliases") or [])]
        if not owners:
            errors.append(f"alias not found on any canonical: {alias}")
            continue
        if len(owners) > 1:
            errors.append(f"alias on >1 canonical: {alias} -> {[o['id'] for o in owners]}")
            continue
        cur = owners[0]
        if cur["id"] == target:
            actions.append(f"{alias}: already on {target} (no-op)")
            continue
        cur["aliases"] = [a for a in cur["aliases"] if a != alias]
        tgt = by_id[target]
        tal = tgt.get("aliases") or []
        if alias not in tal:
            tal.append(alias)
        tgt["aliases"] = tal
        actions.append(f"{alias}: {cur['id']} -> {target}")

    print(f"repoints: {len(actions)}")
    for a in actions:
        print(f"   {a}")
    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors:
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
