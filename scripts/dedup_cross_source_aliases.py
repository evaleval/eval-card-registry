#!/usr/bin/env python3
"""
Deterministic CROSS-SOURCE alias reconciliation — the finalize step of the
generator-driven seed layer.

The seed loader fails fast when two canonicals declare the same alias (the owner
would be nondeterministic). Per-source generators each clean their own output,
but a contested surface form can still be claimed across DIFFERENT sources (e.g.
hf_oracle's base `meta-llama/Llama-3.2-1B` carrying the instruct variant's id, or
two community re-hosts sharing a bare name `gemma-3-1b-it`). This pass resolves
every such collision ONCE, deterministically, applying the project rule
("the more-specific / id-owning entity owns the name") so a clean regen always
produces a seedable, collision-free source set.

Ownership priority for a contested form (highest first):
  1. the canonical whose ID == the form (an id always beats an alias);
  2. curated core over a generated source (core is authoritative);
  3. the canonical whose normalized NAME == the form (the natural owner);
  4. lexicographically-first id (deterministic tie-break).
The owner keeps the form; it is stripped from every other canonical's aliases /
alias_platforms, and a losing display_name is re-derived from the id tail. core
is NEVER stripped (a core-vs-core collision is surfaced, not auto-resolved).

Reproducible + idempotent: re-running on a deduped tree is a no-op. Run as the
last step of scripts/regenerate_sources.sh (and the cron) before seeding.

Usage:
    LOCAL_MODE=true uv run python scripts/dedup_cross_source_aliases.py
    ... --dry-run    # report strips, write nothing
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from eval_entity_resolver.normalization import normalize as _nz

REPO = Path(__file__).resolve().parent.parent
CORE = REPO / "seed" / "models" / "core.yaml"
SOURCES = [
    REPO / "seed" / "models" / "sources" / name
    for name in (
        "hf_oracle.generated.yaml",
        "models_dev.generated.yaml",
        "hub_stats.generated.yaml",
        "models_dev_catalog.generated.yaml",
        "tier3_inferred.generated.yaml",
    )
]


def _load(path: Path):
    if not path.exists():
        return None, []
    doc = yaml.safe_load(path.read_text())
    entries = (doc.get("entries") if isinstance(doc, dict) else doc) or []
    return doc, [e for e in entries if isinstance(e, dict)]


def _name_part(cid: str) -> str:
    return cid.split("/", 1)[-1] if "/" in cid else cid


def build_owner_map(core_entries, source_entries):
    """Return {surface_form -> owner_id} over every id/display_name/alias across
    core + sources, using the ownership priority above. `source_entries` is a
    flat list of (entry, is_core=False); core entries are is_core=True."""
    all_entries = [(e, True) for e in core_entries] + [(e, False) for e in source_entries]
    ids = {e["id"] for e, _ in all_entries if isinstance(e.get("id"), str)}

    # form -> list of candidate owners with their priority key
    cands: dict[str, list[tuple]] = defaultdict(list)

    def consider(form: str, e: dict, is_core: bool):
        if not isinstance(form, str) or not form:
            return
        cid = e["id"]
        is_id_owner = form == cid
        is_natural = _nz(form) == _nz(_name_part(cid))
        # priority key: lower is better
        key = (
            0 if is_id_owner else 1,
            0 if is_core else 1,
            0 if is_natural else 1,
            cid,
        )
        cands[form].append((key, cid))

    for e, is_core in all_entries:
        cid = e.get("id")
        if not isinstance(cid, str):
            continue
        consider(cid, e, is_core)
        dn = e.get("display_name")
        if isinstance(dn, str):
            consider(dn, e, is_core)
        for a in e.get("aliases") or []:
            if isinstance(a, str):
                consider(a, e, is_core)

    owner: dict[str, str] = {}
    for form, lst in cands.items():
        owner[form] = min(lst)[1]   # min by priority key; [1] is the owner id
    return owner, ids


def reconcile(dry_run: bool = False) -> int:
    _core_doc, core_entries = _load(CORE)
    loaded = [(p, *_load(p)) for p in SOURCES]
    source_entries = [e for _p, _doc, ents in loaded for e in ents]

    owner, _ids = build_owner_map(core_entries, source_entries)

    # Surface core-vs-core contention (we never strip core) for visibility.
    core_ids = {e["id"] for e in core_entries}
    strips: list[str] = []

    def clean_entry(e: dict) -> bool:
        """Strip forms this entry does not own. Returns True if mutated."""
        cid = e["id"]
        mutated = False
        kept = []
        for a in e.get("aliases") or []:
            if isinstance(a, str) and a != cid and owner.get(a, cid) != cid:
                strips.append(f"{cid}: drop alias {a!r} (owned by {owner.get(a)!r})")
                mutated = True
                continue
            kept.append(a)
        if mutated:
            e["aliases"] = kept
        ap = e.get("alias_platforms")
        if isinstance(ap, dict):
            new_ap = {k: v for k, v in ap.items() if k == cid or owner.get(k, cid) == cid}
            if len(new_ap) != len(ap):
                mutated = True
                e["alias_platforms"] = new_ap
        dn = e.get("display_name")
        if isinstance(dn, str) and dn and owner.get(dn, cid) != cid:
            e["display_name"] = _name_part(cid)
            strips.append(f"{cid}: re-derive display_name (was {dn!r}, owned by {owner.get(dn)!r})")
            mutated = True
        return mutated

    # Strip SOURCES only (core is authoritative).
    changed_paths = []
    for path, doc, ents in loaded:
        any_changed = False
        for e in ents:
            if clean_entry(e):
                any_changed = True
        if any_changed and not dry_run:
            out = ents if not isinstance(doc, dict) else {**doc, "entries": ents}
            header = ""
            text = path.read_text()
            if text.startswith("#"):
                header = "\n".join(
                    ln for ln in text.splitlines() if ln.startswith("#")
                ) + "\n"
            path.write_text(header + yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=200))
            changed_paths.append(path.name)

    # Report.
    for s in strips:
        print(f"[dedup] {s}", file=sys.stderr)
    print(
        f"[dedup] {'(dry-run) ' if dry_run else ''}stripped {len(strips)} contested "
        f"form(s) across {len(changed_paths)} source file(s): {changed_paths}",
        file=sys.stderr,
    )
    # Sanity: a core entry losing a form it declares is a CURATION conflict.
    core_conflicts = []
    for e in core_entries:
        cid = e["id"]
        for a in (e.get("aliases") or []):
            if isinstance(a, str) and a != cid and owner.get(a) != cid and owner.get(a) in core_ids:
                core_conflicts.append((cid, a, owner.get(a)))
    if core_conflicts:
        print(f"[dedup] WARNING: {len(core_conflicts)} core-vs-core alias conflict(s) "
              f"(NOT auto-stripped — fix curation): {core_conflicts[:5]}", file=sys.stderr)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="report strips; write nothing")
    args = p.parse_args()
    return reconcile(dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
