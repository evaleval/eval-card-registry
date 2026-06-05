#!/usr/bin/env python3
"""Apply batch 2 of the alias-sweep 'unsure' triage (user-confirmed).

- oasst-rm: repoint the size/version aliases onto their existing per-size
  canonicals and link each to the generic root (shared model_family_id).
- drop baichuan-7b-ocr (wrong fold onto baichuan2-7b; the only Baichuan 7B OCR
  repo is the medical BaichuanMed-OCR-7B, exact match unconfirmed) and
  internvl-3-0 (generic version, no sizeless node — leave unresolved).
- gemma2 -> gemma-2-9b and the baichuan training-stage labels (vanilla/sft/
  cpt-sft) are KEPT folded (no separate HF repos); benign re-spellings untouched.

DROPS verified not EEE-raw, so coverage is unaffected. Dry-run by default.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "seed" / "models" / "core.yaml"

REPOINTS = {
    "openassistant/oasst-rm-2-1-pythia-1-4b-epoch-2-5": "OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5",
    "unknown/oasst-rm-2-1-pythia-1-4b": "OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5",
    "openassistant/oasst-rm-2-pythia-6-9b-epoch-1": "OpenAssistant/oasst-rm-2-pythia-6.9b-epoch-1",
}
DROPS = ["unknown/baichuan-7b-ocr", "unknown/internvl-3-0"]
# child -> (parent, rel, axis) family links for the per-size oasst canonicals
EDGES = {
    "OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5": ("OpenAssistant/oasst-rm-pythia", "variant", "size"),
    "OpenAssistant/oasst-rm-2-pythia-6.9b-epoch-1": ("OpenAssistant/oasst-rm-pythia", "variant", "size"),
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

    def owner_of(alias: str):
        ows = [e for e in entries if isinstance(e, dict) and alias in (e.get("aliases") or [])]
        if len(ows) != 1:
            errors.append(f"alias on {len(ows)} canonicals (need 1): {alias}")
            return None
        return ows[0]

    for alias, target in REPOINTS.items():
        if target not in by_id:
            errors.append(f"repoint target missing: {target}")
            continue
        ow = owner_of(alias)
        if ow is None:
            continue
        ow["aliases"] = [a for a in ow["aliases"] if a != alias]
        tal = by_id[target].get("aliases") or []
        if alias not in tal:
            tal.append(alias)
        by_id[target]["aliases"] = tal
        actions.append(f"REPOINT {alias}: {ow['id']} -> {target}")

    for alias in DROPS:
        ow = owner_of(alias)
        if ow is None:
            continue
        ow["aliases"] = [a for a in ow["aliases"] if a != alias]
        actions.append(f"DROP {alias} (off {ow['id']})")

    for child, (pid, rel, axis) in EDGES.items():
        c = by_id.get(child)
        if c is None:
            errors.append(f"edge child missing: {child}")
            continue
        if pid not in by_id:
            errors.append(f"edge parent missing: {pid}")
            continue
        pars = c.get("parents") or []
        if any(isinstance(p, dict) and p.get("id") == pid for p in pars):
            continue
        edge = {"id": pid, "relationship": rel}
        if axis:
            edge["axis"] = axis
        pars.append(edge)
        c["parents"] = pars
        actions.append(f"{child}  +{rel}/{axis} -> {pid}")

    for a in actions:
        print(f"   {a}")
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
