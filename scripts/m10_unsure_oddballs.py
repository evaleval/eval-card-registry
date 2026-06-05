#!/usr/bin/env python3
"""Final 'unsure' oddball: model Nemotron-Nano-12B-v2-VL, which NVIDIA ships ONLY
in quantized form (no unquantized base repo exists). Model it as a precisionless
group anchor with the real per-precision repos as `quantized` leaves sharing its
model_group_id; the precision-unspecified eval string resolves to the anchor.

(gpt-3-curie-ned stays on openai/curie-001 per review — no change.)

DETERMINISTIC. Dry-run by default; --apply writes core.yaml.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "seed" / "models" / "core.yaml"

ANCHOR = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL"
# (aliases, {id, ow, parent})
MINTS = [
    # precisionless group anchor — the eval string (no precision) resolves here
    (["unknown/nemotron-nano-v2-vl"], {"id": ANCHOR, "ow": True}),
    # real per-precision repos as quantized leaves (share the anchor's group)
    ([], {"id": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16", "ow": True, "parent": (ANCHOR, "quantized")}),
    ([], {"id": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8", "ow": True, "parent": (ANCHOR, "quantized")}),
    ([], {"id": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD", "ow": True, "parent": (ANCHOR, "quantized")}),
]


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

    for aliases, spec in MINTS:
        nid = spec["id"]
        if nid in by_id:
            errors.append(f"mint id already exists: {nid}")
            continue
        ok = True
        for a in aliases:
            ow = owner_of(a)
            if ow is None:
                ok = False
                break
            ow["aliases"] = [x for x in ow["aliases"] if x != a]
        if not ok:
            continue
        parents = []
        if spec.get("parent"):
            pid, rel = spec["parent"]
            if pid not in by_id:
                errors.append(f"parent missing: {pid}")
                continue
            parents = [{"id": pid, "relationship": rel}]
        row = {
            "id": nid, "display_name": nid, "org_id": None, "family": None,
            "architecture": None, "params_billions": None, "parents": parents,
            "open_weights": spec["ow"], "release_date": None, "input_modalities": None,
            "output_modalities": None, "tags": [], "metadata": "{}",
            "review_status": "reviewed", "aliases": list(aliases),
        }
        entries.append(row)
        by_id[nid] = row
        actions.append(f"MINT {nid}  aliases={aliases}{' quantized->'+spec['parent'][0] if spec.get('parent') else ''}")

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
