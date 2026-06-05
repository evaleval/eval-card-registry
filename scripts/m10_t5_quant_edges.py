#!/usr/bin/env python3
"""Add the missing `quantized` parent edge for same-vendor quant variants whose
base canonical exists, so _walk_group folds them into the base's model_group_id
(a precision variant is the same model at the API level). Org-conditional folding
still applies — these are all first-party (same org as the base). Third-party
quants (different org, e.g. an nvidia nvfp4 of a Moonshot model) are intentionally
NOT folded and not touched here.

DETERMINISTIC. Dry-run by default; --apply writes core.yaml.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "seed" / "models" / "core.yaml"

# quant variant -> base (same vendor; both verified canonicals)
EDGES = {
    "deepseek/deepseek-v4-flash-6bit": "deepseek/deepseek-v4-flash",
    "meta/llama-2-7b-chat-fp16": "meta/llama-2-7b-chat",
    "xiaomi/mimo-v2-5-pro-6bit": "xiaomi/mimo-v2-5-pro",
    "zai-org/GLM-4.5-Air-FP8": "zai-org/GLM-4.5-Air",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    doc = yaml.safe_load(CORE.read_text())
    entries = doc["entries"] if isinstance(doc, dict) else doc
    by_id = {e["id"]: e for e in entries if isinstance(e, dict) and isinstance(e.get("id"), str)}
    errors, actions = [], []
    for child, base in EDGES.items():
        c = by_id.get(child)
        if c is None or base not in by_id:
            errors.append(f"missing: {child} / {base}")
            continue
        pars = c.get("parents") or []
        if any(isinstance(p, dict) and p.get("id") == base for p in pars):
            continue
        pars.append({"id": base, "relationship": "quantized"})
        c["parents"] = pars
        actions.append(f"{child}  +quantized -> {base}")
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
    print("\nAPPLIED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
