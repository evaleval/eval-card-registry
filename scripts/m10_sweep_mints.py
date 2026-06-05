#!/usr/bin/env python3
"""Apply the high-confidence alias-sweep wrong-folds: mint the correct canonical
(real HF id verified via the HF API), repoint to an existing canonical, or drop
the alias when the named model is confirmed not to exist.

Every minted id and every repoint target was verified against HF / the existing
canonical set — no fabricated ids. The uncertain tail (closed-weight products,
ambiguous variants) is handled separately after discussion.

DETERMINISTIC. Dry-run by default; --apply writes core.yaml. Validates: each
mint id is new, each repoint/parent target exists, each alias is currently
present on exactly one canonical, no duplicate alias/id results.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "seed" / "models" / "core.yaml"

# alias -> mint spec. The alias moves off its wrong owner onto the new canonical.
MINTS = {
    "unknown/bloomz-560m": {"id": "bigscience/bloomz-560m", "org": "bigscience",
        "parent": ("bigscience/bloom-560m", "finetune")},
    "unknown/bloomz-1-1b": {"id": "bigscience/bloomz-1b1", "org": "bigscience",
        "parent": ("bigscience/bloom-1b1", "finetune")},
    "unknown/bloomz-1-7b": {"id": "bigscience/bloomz-1b7", "org": "bigscience",
        "parent": ("bigscience/bloom-1b7", "finetune")},
    "unknown/bloomz": {"id": "bigscience/bloomz", "org": "bigscience",
        "parent": ("bigscience/bloom-176b", "finetune")},
    "unknown/llava-1-6-vicuna-13b": {"id": "liuhaotian/llava-v1.6-vicuna-13b", "org": "liuhaotian"},
    "unknown/llama-3-1-nemotron-nano-vl-8b-v1": {"id": "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1", "org": "nvidia"},
    # org left null: the namespace isn't a curated/existing org, so the seed's
    # org-from-prefix rule derives it and de-orphans the community row (setting
    # it explicitly here would bypass that and dangle the FK).
    "unknown/dall-e-mini": {"id": "dalle-mini/dalle-mini", "org": None},
}
# two aliases name the same minted MiniCPM model
MINICPM_ALIASES = ["unknown/minicpm-llama3-v-2-5", "unknown/minicpm-llama3-v2-5"]
MINICPM = {"id": "openbmb/MiniCPM-Llama3-V-2_5", "org": "openbmb"}

# alias -> existing canonical to repoint onto
REPOINTS = {
    "unknown/gpt-3-davinci-002": "openai/text-davinci-002",
    "unknown/gpt-3-davinci-003": "openai/text-davinci-003",
    "unknown/instructgpt-davinci-v2": "openai/text-davinci-002",
}
# alias -> drop (named model confirmed not to exist on HF: wrong size)
DROPS = ["unknown/internvl2-5-72b", "unknown/videollama3-8b"]


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

    def mint(spec: dict, aliases: list[str]) -> None:
        nid = spec["id"]
        if nid in by_id:
            errors.append(f"mint id already exists: {nid}")
            return
        for a in aliases:
            ow = owner_of(a)
            if ow is None:
                return
            detach(a, ow)
        parents = []
        if spec.get("parent"):
            pid, rel = spec["parent"]
            if pid not in by_id:
                errors.append(f"parent missing: {pid}")
                return
            parents = [{"id": pid, "relationship": rel}]
        row = {
            "id": nid, "display_name": nid, "org_id": spec["org"], "family": None,
            "architecture": None, "params_billions": None, "parents": parents,
            "open_weights": None, "release_date": None, "input_modalities": None,
            "output_modalities": None, "tags": [], "metadata": "{}",
            "review_status": "reviewed", "aliases": list(aliases),
        }
        entries.append(row)
        by_id[nid] = row
        actions.append(f"MINT {nid}  (aliases {aliases})")

    # mints
    for alias, spec in MINTS.items():
        mint(spec, [alias])
    mint(MINICPM, MINICPM_ALIASES)

    # repoints
    for alias, target in REPOINTS.items():
        if target not in by_id:
            errors.append(f"repoint target missing: {target}")
            continue
        ow = owner_of(alias)
        if ow is None:
            continue
        if ow["id"] == target:
            actions.append(f"REPOINT {alias}: already on {target}")
            continue
        detach(alias, ow)
        tal = by_id[target].get("aliases") or []
        if alias not in tal:
            tal.append(alias)
        by_id[target]["aliases"] = tal
        actions.append(f"REPOINT {alias}: {ow['id']} -> {target}")

    # drops
    for alias in DROPS:
        ow = owner_of(alias)
        if ow is None:
            continue
        detach(alias, ow)
        actions.append(f"DROP {alias} (off {ow['id']}; named model not on HF)")

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
