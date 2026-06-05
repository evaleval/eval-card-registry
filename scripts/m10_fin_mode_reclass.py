#!/usr/bin/env python3
"""Reclassify mode->training_stage edges.

The buried-alias promotion minted `-instruct`/`-chat`/`-it`/`-sft` variants as
`variant axis=mode` edges, but those are a TRAINING STAGE (instruction/chat tuning),
not a runtime mode. This relabels every such edge `axis: mode -> training_stage`.

Scope is edge-relabel ONLY — no id changes, no org changes, no mints. We touch a
`parents` edge iff:
  - relationship == variant AND axis == mode, AND
  - the CHILD id's name-part carries a training-stage token (-instruct/-chat/-it/-sft),
    delimited by start/end or one of [-_/.].

Genuine runtime modes (reasoning / thinking / non-thinking / base / ...) carry no
such token and are left untouched. Each planned edit is validated against the live
core.yaml (edge still present, still axis=mode, not already training_stage) before
it is applied; anything that does not validate is reported as skipped, not applied.

Dry-run by default; --apply writes seed/models/core.yaml.

Follow-up (NOT done here, deliberately): once this lands, `mode` should be added to
`_is_identity_edge` in store/queries.py (one-line) so thinking variants group with
their base. That is a separate change — do not make it in this script.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

CORE = Path(__file__).resolve().parents[1] / "seed" / "models" / "core.yaml"

# A training-stage token in the CHILD id's name-part, delimited so we never match
# an "it"/"chat" substring buried inside another word.
TRAINING_STAGE_TOKEN = re.compile(
    r"(?:^|[-_/.])(instruct|chat|it|sft)(?:$|[-_/.])", re.IGNORECASE
)


def name_part(model_id: str) -> str:
    return model_id.split("/", 1)[-1]


def has_training_stage_token(model_id: str) -> bool:
    return bool(TRAINING_STAGE_TOKEN.search(name_part(model_id)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="write core.yaml (default: dry-run)")
    args = ap.parse_args()

    doc = yaml.safe_load(CORE.read_text())
    entries = doc["entries"] if isinstance(doc, dict) else doc

    planned: list[tuple[str, str]] = []   # (child_id, parent_id)
    skipped: list[tuple[str, str, str]] = []  # (child_id, parent_id, reason)

    total_mode_edges = 0
    for e in entries:
        if not isinstance(e, dict):
            continue
        child_id = e.get("id")
        if not isinstance(child_id, str):
            continue
        for p in (e.get("parents") or []):
            if not isinstance(p, dict):
                continue
            if p.get("relationship") != "variant":
                continue
            axis = p.get("axis")
            parent_id = p.get("id")
            if axis == "mode":
                total_mode_edges += 1
                if has_training_stage_token(child_id):
                    planned.append((child_id, parent_id))
                else:
                    # genuine runtime mode (reasoning/thinking/base/...) — leave it
                    continue
            elif axis == "training_stage" and has_training_stage_token(child_id):
                # already correctly labelled (e.g. a re-run) — report as skip
                skipped.append((child_id, parent_id, "already axis=training_stage"))

    print(f"mode-axis variant edges total: {total_mode_edges}")
    print(f"  -> relabel mode->training_stage (child has training-stage token): {len(planned)}")
    print(f"  -> left as genuine mode (no token): {total_mode_edges - len(planned)}")
    print(f"already training_stage (skip): {len(skipped)}")

    print("\nexamples (planned):")
    for child_id, parent_id in planned[:12]:
        tok = TRAINING_STAGE_TOKEN.search(name_part(child_id)).group(1).lower()
        print(f"   [{tok:>8}] {child_id}  --variant/mode->training_stage-->  {parent_id}")
    if skipped:
        print("\nexamples (skipped):")
        for child_id, parent_id, reason in skipped[:12]:
            print(f"   SKIP {child_id} -> {parent_id} :: {reason}")

    if not args.apply:
        print("\n(dry-run — no file written; pass --apply to write core.yaml)")
        return 0

    # Apply: re-walk and validate each edge against the live structure before mutating.
    applied = 0
    planned_set = set(planned)
    for e in entries:
        if not isinstance(e, dict):
            continue
        child_id = e.get("id")
        for p in (e.get("parents") or []):
            if not isinstance(p, dict):
                continue
            parent_id = p.get("id")
            if (child_id, parent_id) not in planned_set:
                continue
            # validate live: still a variant/mode edge with the token
            if (
                p.get("relationship") == "variant"
                and p.get("axis") == "mode"
                and has_training_stage_token(child_id)
            ):
                p["axis"] = "training_stage"
                applied += 1

    if isinstance(doc, dict):
        doc["entries"] = entries
    CORE.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=10_000))
    print(f"\nAPPLIED {applied} edge relabels (mode -> training_stage)")
    if applied != len(planned):
        print(f"WARNING: applied {applied} != planned {len(planned)} — re-run dry-run to inspect")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
