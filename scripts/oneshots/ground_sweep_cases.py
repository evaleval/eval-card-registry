#!/usr/bin/env python3
"""Enrich the alias-sweep confirmed/unsure cases with closed-world facts so a
human can triage keep-vs-fix reliably (the per-case judgment the sweep agents
got wrong hinged on facts they didn't have).

For each {alias, owner_id} it reports:
  - alias_in_oracle : is the alias a real EEE submission (so it MUST resolve)?
  - alias_org       : the org prefix on the alias, if any
  - body            : the alias name-part (after last '/')
  - body_matches    : every canonical id whose name-part normalizes to the same
                      body (regardless of org) — reveals whether a distinct model
                      under the aliased org exists, and whether the right target
                      is already in the registry
  - owner_params    : owner canonical's params_billions / family (size checks)

Reads curation/alias_sweep_result.json.
Usage: LOCAL_MODE=true uv run python scripts/ground_sweep_cases.py [--unsure]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "seed" / "models" / "core.yaml"
SWEEP = ROOT  / "curation" / "alias_sweep_result.json"
ORACLE = Path("/Users/jchim/projects/evaleval/hf_model_id_resolution.json")

_SEP = re.compile(r"[/_.:\-]+")


def norm(s: str) -> str:
    return _SEP.sub("-", s.lower()).strip("-")


def body(s: str) -> str:
    return norm(s.split("/")[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unsure", action="store_true", help="ground the unsure list instead of confirmed")
    args = ap.parse_args()

    doc = yaml.safe_load(CORE.read_text())
    entries = [e for e in (doc["entries"] if isinstance(doc, dict) else doc)
               if isinstance(e, dict) and isinstance(e.get("id"), str)]

    # body-norm -> canonical ids (and which canonicals carry it as an alias body)
    body_to_ids: dict[str, set[str]] = defaultdict(set)
    meta: dict[str, dict] = {}
    for e in entries:
        meta[e["id"]] = e
        body_to_ids[body(e["id"])].add(e["id"])
        for a in (e.get("aliases") or []):
            if isinstance(a, str):
                body_to_ids[body(a)].add(e["id"])

    oracle = json.loads(ORACLE.read_text())["resolutions"]
    oracle_ci = {k.lower() for k in oracle}

    data = json.loads(SWEEP.read_text())["result"]
    cases = data["unsure"] if args.unsure else data["confirmed_wrong_fold"]

    for v in cases:
        alias = v["alias"]
        owner = v["owner_id"]
        b = body(alias)
        matches = sorted(body_to_ids.get(b, set()))
        aorg = alias.split("/", 1)[0] if "/" in alias else "(none)"
        ow = meta.get(owner, {})
        in_oracle = alias.lower() in oracle_ci
        # owner body match count = is the body unique to owner, or shared?
        only_owner = matches == [owner]
        print(f"\n• {alias}")
        print(f"    owner      : {owner}  (params={ow.get('params_billions')}, family={ow.get('family')})")
        print(f"    agent      : {v.get('proposed_action')} (c={v.get('confidence')}) {('-> '+v['repoint_target']) if v.get('repoint_target') else ''}")
        print(f"    eee_raw    : {in_oracle}    alias_org: {aorg}")
        print(f"    body       : {b!r}  body_unique_to_owner={only_owner}")
        if not only_owner:
            print(f"    body_matches: {matches}")
    print(f"\n[{len(cases)} cases]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
