#!/usr/bin/env python3
"""Split the multi-size Ollama packed shells into the
per-size HF leaves that already exist. Deletes the packed shell, re-attaches its
`:Nb` aliases (with inference_platform=ollama-cloud) onto the matching real leaf.
Dry-run by default; --apply writes seed/models/core.yaml."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import yaml

CORE = Path(__file__).resolve().parents[1] / "seed" / "models" / "core.yaml"
PLATFORM = "ollama-cloud"

# alias-spelling -> target real leaf  (explicit :Nb tags + documented bare default)
ATTACH = {
    # gpt-oss: bare `gpt-oss` = Ollama default 20b (matches the shell display).
    "gpt-oss:120b": "openai/gpt-oss-120b",
    "gpt-oss:20b": "openai/gpt-oss-20b",
    "gpt-oss": "openai/gpt-oss-20b",
    # gemma3: explicit sizes only; bare gemma3/gemma-3/google/gemma3 are
    # size-ambiguous -> intentionally DROPPED (not attached to a guessed size).
    "gemma3:4b": "google/gemma-3-4b-it",
    "gemma3:12b": "google/gemma-3-12b-it",
    "gemma3:27b": "google/gemma-3-27b-it",
}
DROP_ALIASES = {"gemma3", "gemma-3", "google/gemma3"}   # ambiguous bare tags
DELETE_SHELLS = {"openai/gpt-oss", "google/gemma-3"}
# the shell display strings that should also re-attach (gpt-oss shell display = gpt-oss:20b)
SHELL_DISPLAY_AS_ALIAS = {"openai/gpt-oss": "gpt-oss:20b", "google/gemma-3": "gemma3:27b"}
# a -latest child whose parent edge points at a deleted shell -> drop that edge
DROP_PARENT_EDGE_TO_SHELL = {"google/gemma3-latest"}


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

    # sanity: targets exist
    for tgt in set(ATTACH.values()):
        if tgt not in by:
            print(f"ABORT: target leaf {tgt} missing", file=sys.stderr); return 2

    plan = []
    # 1. attach aliases onto targets
    for alias, tgt in ATTACH.items():
        e = by[tgt]
        al = list(e.get("aliases") or [])
        if alias != tgt and alias not in al:
            al.append(alias)
            plan.append(f"attach alias {alias!r} -> {tgt}  [{PLATFORM}]")
        e["aliases"] = sorted(set(al))
        md = _md(e); ap_map = md.get("alias_platforms") or {}
        ap_map[alias] = PLATFORM
        md["alias_platforms"] = ap_map
        e["metadata"] = json.dumps(md, sort_keys=True)
    for alias in DROP_ALIASES:
        plan.append(f"DROP ambiguous bare alias {alias!r} (not attached to a size)")

    # 2. drop dangling parent edges on -latest children
    for cid in DROP_PARENT_EDGE_TO_SHELL:
        e = by.get(cid)
        if not e: continue
        kept = [p for p in (e.get("parents") or [])
                if not (isinstance(p, dict) and p.get("id") in DELETE_SHELLS)]
        if len(kept) != len(e.get("parents") or []):
            plan.append(f"drop dangling parent edge on {cid} (was -> deleted shell)")
        e["parents"] = kept

    # 3. delete shells
    new_entries = [e for e in entries
                   if not (isinstance(e, dict) and e.get("id") in DELETE_SHELLS)]
    for s in DELETE_SHELLS:
        plan.append(f"DELETE packed shell {s}")

    # guard: no surviving entry references a deleted shell as a parent
    dangling = []
    for e in new_entries:
        if not isinstance(e, dict): continue
        for p in (e.get("parents") or []):
            if isinstance(p, dict) and p.get("id") in DELETE_SHELLS:
                dangling.append((e["id"], p["id"]))
    if dangling:
        print(f"ABORT: {len(dangling)} dangling parent edges remain: {dangling}", file=sys.stderr)
        return 2

    print(f"plan ({len(plan)} ops):")
    for p in plan: print("   ", p)
    if args.apply:
        if isinstance(doc, dict): doc["entries"] = new_entries
        else: doc = new_entries
        CORE.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=10_000))
        print("APPLIED.")
    else:
        print("(dry-run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
