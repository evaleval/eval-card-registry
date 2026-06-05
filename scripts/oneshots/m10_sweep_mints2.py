#!/usr/bin/env python3
"""Apply the grounded tail of the alias-sweep wrong-folds (batch 2).

Mints, repoints, and drops decided after HF-API grounding of each case. Every
minted id was verified on HF or is a known closed-weight product; every repoint
target is an existing canonical; the two drops are obscure community artifacts
with no stable canonical (if either appears in an EEE sync, ingestion auto-creates
its own draft entity).

Minted rows leave org_id null so the seed's org-from-prefix rule derives the
developer and de-orphans any missing community org row (setting it explicitly
would risk a dangling FK).

DETERMINISTIC. Dry-run by default; --apply writes core.yaml.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "seed" / "models" / "core.yaml"

# alias(es) -> mint spec (open_weights: True on-HF, False closed-product)
MINTS = [
    (["otter-9b", "unknown/otter-9b"], {"id": "luodian/OTTER-9B-LA-InContext", "ow": True}),
    (["unknown/skywork-prm-1-5b"], {"id": "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B", "ow": True}),
    (["unknown/lfm-3b"], {"id": "LiquidAI/lfm-3b", "ow": False}),
    (["unknown/lfm-7b"], {"id": "LiquidAI/lfm-7b", "ow": False}),
    (["unknown/gpt-3"], {"id": "openai/gpt-3", "ow": False}),
    (["unknown/gpt-3-1-3b-babbage-002"], {"id": "openai/babbage-002", "ow": False}),
    # a real (off-HF) EEE eval artifact — a distinct Tulu-2-7B reward-model
    # checkpoint, not base Tulu-2; mint its own canonical like its .json siblings
    # (dropping it would no_match a string that is already in the corpus).
    (["ai2/tulu-2-7b-rm-v0-nectar-binarized-3-8m-check"],
     {"id": "allenai/tulu-2-7b-rm-v0-nectar-binarized-3.8m-check", "ow": None}),
]
REPOINTS = {
    # text-curie-001 (InstructGPT curie) is already folded onto openai/curie-001;
    # keep this alias consistent with that existing handling rather than minting.
    "unknown/instructgpt-curie-v1": "openai/curie-001",
    "moonshotai/k2": "moonshotai/Kimi-K2-Instruct",
    "unknown/llava-video": "lmms-lab/llava-next-video-7b",
    "unknown/llava-next-v-7b": "liuhaotian/llava-1.6-vicuna-7b",
}
DROPS = [
    "unknown/starling-lm-alpha-8x7b-moe-gptq",  # not an EEE raw; obscure community quant
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

    def detach(alias: str, e: dict) -> None:
        e["aliases"] = [a for a in (e.get("aliases") or []) if a != alias]

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
            detach(a, ow)
        if not ok:
            continue
        row = {
            "id": nid, "display_name": nid, "org_id": None, "family": None,
            "architecture": None, "params_billions": None, "parents": [],
            "open_weights": spec["ow"], "release_date": None, "input_modalities": None,
            "output_modalities": None, "tags": [], "metadata": "{}",
            "review_status": "reviewed", "aliases": list(aliases),
        }
        entries.append(row)
        by_id[nid] = row
        actions.append(f"MINT {nid}  (aliases {aliases}, open_weights={spec['ow']})")

    for alias, target in REPOINTS.items():
        if target not in by_id:
            errors.append(f"repoint target missing: {target}")
            continue
        ow = owner_of(alias)
        if ow is None:
            continue
        detach(alias, ow)
        tal = by_id[target].get("aliases") or []
        if alias not in tal:
            tal.append(alias)
        by_id[target]["aliases"] = tal
        actions.append(f"REPOINT {alias}: {ow['id']} -> {target}")

    for alias in DROPS:
        ow = owner_of(alias)
        if ow is None:
            continue
        detach(alias, ow)
        actions.append(f"DROP {alias} (off {ow['id']})")

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
