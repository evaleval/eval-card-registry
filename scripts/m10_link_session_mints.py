#!/usr/bin/env python3
"""Add the relationship edges that earlier session mints were missing, so models
that belong to a size family or have a base in the registry are LINKED (size
siblings end up sharing model_family_id; reward/derived models keep a lineage
edge). Idempotent: skips an edge already present.

Only links where a genuine same-line sibling/base exists. Different-base
look-alikes (OTTER-9B MPT vs 7B LLaMA, Nemotron-VL-8B vs 12B, MiniCPM-Llama3-V
vs MiniCPM-V) are intentionally left unlinked.

DETERMINISTIC. Dry-run by default; --apply writes core.yaml.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "seed" / "models" / "core.yaml"

# child id -> (parent id, relationship, axis-or-None)
EDGES = {
    "LiquidAI/lfm-3b": ("LiquidAI/lfm-40b", "variant", "size"),
    "LiquidAI/lfm-7b": ("LiquidAI/lfm-40b", "variant", "size"),
    "liuhaotian/llava-v1.6-vicuna-13b": ("liuhaotian/llava-1.6-vicuna-7b", "variant", "size"),
    "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B": ("Skywork/skywork-prm-7b", "variant", "size"),
    "bigscience/bloomz-560m": ("bigscience/bloomz", "variant", "size"),
    "bigscience/bloomz-1b1": ("bigscience/bloomz", "variant", "size"),
    "bigscience/bloomz-1b7": ("bigscience/bloomz", "variant", "size"),
    "allenai/tulu-2-7b-rm-v0-nectar-binarized-3.8m-check": ("allenai/tulu-2-7b", "finetune", None),
    "speakleash/Bielik-11B-v2.3-Instruct": ("speakleash/Bielik-11B-v2", "merge", None),
    # Skywork reward size family: link each existing per-size canonical to the
    # sizeless generic root so they share model_family_id.
    "Skywork/Skywork-Reward-V2-Qwen3-0.6B": ("Skywork/skywork-reward-v2-qwen3", "variant", "size"),
    "Skywork/Skywork-Reward-V2-Qwen3-1.7B": ("Skywork/skywork-reward-v2-qwen3", "variant", "size"),
    "Skywork/Skywork-Reward-V2-Qwen3-4B": ("Skywork/skywork-reward-v2-qwen3", "variant", "size"),
    "Skywork/Skywork-Reward-V2-Qwen3-8B": ("Skywork/skywork-reward-v2-qwen3", "variant", "size"),
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
    for child, (pid, rel, axis) in EDGES.items():
        c = by_id.get(child)
        if c is None:
            errors.append(f"child missing: {child}")
            continue
        if pid not in by_id:
            errors.append(f"parent missing: {pid} (for {child})")
            continue
        pars = c.get("parents") or []
        if any(isinstance(p, dict) and p.get("id") == pid for p in pars):
            actions.append(f"{child}: edge to {pid} already present (skip)")
            continue
        edge = {"id": pid, "relationship": rel}
        if axis:
            edge["axis"] = axis
        pars.append(edge)
        c["parents"] = pars
        actions.append(f"{child}  +{rel}{('/'+axis) if axis else ''} -> {pid}")

    for a in actions:
        print(f"   {a}")
    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors:
            print(f"   ! {e}")
        print("ABORT.")
        return 1
    if not args.apply:
        print(f"\n[{len(actions)} edges] (dry-run)")
        return 0
    if isinstance(doc, dict):
        doc["entries"] = entries
    CORE.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=10_000))
    print(f"\nAPPLIED {len(actions)} edges.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
