#!/usr/bin/env python3
"""Targeted id/identity fixes (review T4):

1. SPLIT whisper-large-v3-turbo (809M distilled) out of openai/whisper-large-v3
   (1.5B) — different model/generation, must not share a canonical/group.
2. REKEY liuhaotian/llava-1.6-vicuna-7b -> liuhaotian/llava-v1.6-vicuna-7b (the
   real HF repo casing; keep old as alias; rewrite parent edges).
3. Drop the malformed `hf:nvidia` org and repoint its model to the real `nvidia`
   org (the `hf:` prefix is stripped by the generator now; this clears the stale
   row + FK baked into the seed).

DETERMINISTIC. Dry-run by default; --apply writes core.yaml + orgs.generated.yaml.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "seed" / "models" / "core.yaml"
ORGS_GEN = ROOT / "seed" / "orgs.generated.yaml"

WHISPER_BASE = "openai/whisper-large-v3"
WHISPER_TURBO = "openai/whisper-large-v3-turbo"
TURBO_ALIASES = ["whisper-large-v3-turbo"]
LLAVA_OLD = "liuhaotian/llava-1.6-vicuna-7b"
LLAVA_NEW = "liuhaotian/llava-v1.6-vicuna-7b"
HF_NVIDIA_ORG = "hf:nvidia"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    doc = yaml.safe_load(CORE.read_text())
    entries = doc["entries"] if isinstance(doc, dict) else doc
    by_id = {e["id"]: e for e in entries if isinstance(e, dict) and isinstance(e.get("id"), str)}
    errors, actions = [], []

    # 1. whisper turbo split
    base = by_id.get(WHISPER_BASE)
    if base is None:
        errors.append(f"whisper base missing: {WHISPER_BASE}")
    elif WHISPER_TURBO in by_id:
        errors.append(f"turbo id already exists: {WHISPER_TURBO}")
    else:
        moved = [a for a in (base.get("aliases") or []) if a in TURBO_ALIASES]
        base["aliases"] = [a for a in (base.get("aliases") or []) if a not in TURBO_ALIASES]
        entries.append({
            "id": WHISPER_TURBO, "display_name": WHISPER_TURBO, "org_id": "openai",
            "family": None, "architecture": None, "params_billions": None, "parents": [],
            "open_weights": True, "release_date": None, "input_modalities": None,
            "output_modalities": None, "tags": ["open-weight"], "metadata": "{}",
            "review_status": "reviewed", "aliases": moved,
        })
        by_id[WHISPER_TURBO] = entries[-1]
        actions.append(f"SPLIT mint {WHISPER_TURBO} (aliases {moved})")

    # 2. llava casing rekey
    lv = by_id.get(LLAVA_OLD)
    if lv is None:
        errors.append(f"llava old id missing: {LLAVA_OLD}")
    elif LLAVA_NEW in by_id:
        errors.append(f"llava new id already exists: {LLAVA_NEW}")
    else:
        lv["id"] = LLAVA_NEW
        lv["display_name"] = LLAVA_NEW
        al = lv.get("aliases") or []
        if LLAVA_OLD not in al:
            al.append(LLAVA_OLD)
        lv["aliases"] = al
        by_id[LLAVA_NEW] = lv
        by_id.pop(LLAVA_OLD, None)
        n = 0
        for e in entries:
            for p in (e.get("parents") or []):
                if isinstance(p, dict) and p.get("id") == LLAVA_OLD:
                    p["id"] = LLAVA_NEW
                    n += 1
        actions.append(f"REKEY {LLAVA_OLD} -> {LLAVA_NEW} ({n} edges)")

    # 3. hf:nvidia org-fix
    n_org = 0
    for e in entries:
        if isinstance(e, dict) and e.get("org_id") == HF_NVIDIA_ORG:
            e["org_id"] = "nvidia"
            n_org += 1
    actions.append(f"org-fix hf:nvidia -> nvidia ({n_org} models)")

    odoc = yaml.safe_load(ORGS_GEN.read_text())
    oentries = odoc["entries"] if isinstance(odoc, dict) else odoc
    before = len(oentries)
    oentries = [o for o in oentries if not (isinstance(o, dict) and o.get("id") == HF_NVIDIA_ORG)]
    removed_org = before - len(oentries)
    actions.append(f"drop hf:nvidia org row ({removed_org})")

    for a in actions:
        print(f"   {a}")
    if errors:
        print("ERRORS:", errors)
        return 1
    if not args.apply:
        print("\n(dry-run)")
        return 0
    if isinstance(doc, dict):
        doc["entries"] = entries
    CORE.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=10_000))
    if isinstance(odoc, dict):
        odoc["entries"] = oentries
    else:
        odoc = oentries
    ORGS_GEN.write_text(yaml.safe_dump(odoc, sort_keys=False, allow_unicode=True, width=10_000))
    print("\nAPPLIED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
