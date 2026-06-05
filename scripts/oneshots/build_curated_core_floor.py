#!/usr/bin/env python3
"""ONE-SHOT (model-resolution-rework): restore the CURATED FLOOR to core.yaml
PLUS the alias bridges the generators dropped, to enrichments/aliases.yaml.

The un-consolidation (commit 73557fa) emptied core.yaml, but the adversarial
re-review proved that over-shrank: a curated set of oracle canonicals are truly
gone — closed-API (open_weights=False) and hand-curated (resolution_source=NA)
entities that NO HF/models.dev/hub-stats generator reproduces. The spec always
meant "minimal CURATED overrides", not empty.

WHAT COUNTS AS "REPRODUCED" — the RESOLVER, nothing else. The spec's acceptance
criterion is reproducing the oracle's *resolution outcomes*. An old-core entry
is "reproduced" iff resolving its id against the GENERATOR-ONLY registry yields
an IDENTITY match (exact/normalized) to a source canonical. That is operationally
exactly "the generators already cover this model under a (possibly better-spelled)
id". The floor is the no-match remainder: the genuinely un-regeneratable curated
set. (An earlier "id-claim guard" that also dropped entries whose id-leaf merely
coincided with a source leaf was WRONG — it killed genuinely-curated canonicals
like `microsoft/phi-4-mini` and `xiaomi/mimo-v2-flash` because a DIFFERENT-org
source happened to share the leaf. Removed.)

ALIAS PORTAGE — preserve the outcomes the generators dropped. When an old-core
entry is reproduced (and thus dropped), its display_name / aliases that the
generators do NOT carry would otherwise stop resolving (a regression: e.g. the
old-core `NousResearch/Hermes-3-Llama-3.1-70B` carried display "Hermes 3
Llama-3.1 70B" which is the only bridge for the bare EEE id `hermes-3-llama-3.1-
70b`). So for each reproduced-dropped entry we PORT its forms whose normalized
key is CURRENTLY UNCLAIMED (no generator canonical resolves it) onto the target
canonical, as enrichment aliases. Gap-filling only: never overrides an existing
resolution, never collides (unclaimed by construction, deduped by norm).

  floor          = old-core entries whose id does NOT exact/normalized-resolve
                   (+ parent-edge repoint, + alias hygiene)
  enrichments    = {reproduced entry forms that were unclaimed} -> target canonical

Org FKs for the floor are minted separately by the universal org reconcile
(refresh_from_modelsdev --reconcile-orgs over _ALL_MODEL_SOURCES incl. core);
this script does not touch orgs. Deterministic + reproducible.

Usage: see scripts/oneshots/restore_curated_floor.sh (wires the full sequence).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from eval_entity_resolver.normalization import normalize as _nz
from eval_entity_resolver.resolver import Resolver

REPO = Path(__file__).resolve().parents[2]
CORE = REPO / "seed" / "models" / "core.yaml"
ENRICH = REPO / "seed" / "models" / "enrichments" / "aliases.yaml"
FIXTURES = REPO / "fixtures"
MODELS_PARQUET = FIXTURES / "canonical_models.parquet"

# "Reproduced" = the generators already resolve this raw to a SAME-MODEL source
# canonical, so the resolution outcome is preserved and the entry can be dropped.
# exact/normalized are identity matches. FUZZY counts ONLY when the LEAF is
# preserved (same model name modulo org/case/sep) — the org-equivalent dup case
# (`Alibaba-NLP/x` ~ `alibaba/x`, `MiniMaxAI/y` ~ `minimax/y`) we must collapse.
# A leaf-CHANGING fuzzy match is a COARSENING across a semantic token
# (`qwen3-235b-a22b-2507` -> `...-Instruct-2507`; `mistral-7b-instruct` -> `...-v0.3`)
# or a quant fold — it conflates a DISTINCT release, so the entry stays in the
# floor and its curated lineage (the version/training_stage parent edge) survives.


def _entries(doc):
    if isinstance(doc, dict):
        return doc.get("entries") or []
    return doc or []


def _norm(s: str) -> str:
    return _nz(s).replace(" ", "")


def _leafkey(cid: str) -> str:
    return _norm(cid.split("/", 1)[1] if "/" in cid else cid)


def _reproduced(raw_id: str, res) -> bool:
    if not res.canonical_id:
        return False
    if res.strategy in ("exact", "normalized"):
        return True
    if res.strategy == "fuzzy":
        return _leafkey(raw_id) == _leafkey(res.canonical_id)
    return False


def _parse_parents(p):
    if p is None:
        return []
    if isinstance(p, str):
        try:
            return json.loads(p) or []
        except Exception:
            return []
    return p or []


def main() -> int:
    old_core_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/old_core.yaml")
    old_doc = yaml.safe_load(old_core_path.read_text())
    old_entries = [e for e in _entries(old_doc) if isinstance(e, dict) and e.get("id")]

    # Resolver loaded from the GENERATOR-ONLY fixtures (core empty when seeded).
    resolver = Resolver.from_parquet(FIXTURES)
    gen = pd.read_parquet(MODELS_PARQUET)
    gen_ids = set(gen["id"].astype(str))

    # source_claims: normalized forms a generator canonical owns (id, id-leaf,
    # display). Used by alias hygiene on kept floor entries.
    source_claims: set[str] = set()
    for _, r in gen.iterrows():
        cid = str(r["id"])
        source_claims.add(_norm(cid))
        if "/" in cid:
            source_claims.add(_norm(cid.split("/", 1)[1]))
        dn = r.get("display_name")
        if isinstance(dn, str) and dn:
            source_claims.add(_norm(dn))

    def _resolves(raw: str) -> bool:
        return _reproduced(raw, resolver.resolve(raw, "model"))

    floor = []
    ports: dict[str, str] = {}      # normalized form -> target canonical_id
    port_display: dict[str, str] = {}  # normalized form -> the raw form to emit
    repro = 0
    for e in old_entries:
        cid = str(e["id"])
        res = resolver.resolve(cid, "model")
        if _reproduced(cid, res):
            repro += 1
            tgt = res.canonical_id
            # PORT: forms this entry carried that the generators dropped. A form
            # is ported iff it does NOT already resolve — gap-filling. We test
            # ACTUAL resolution (not membership in source_claims): source_claims
            # holds each source's bare id-LEAF norm too, but the seed does NOT
            # register the bare leaf as an alias, so a leaf being "claimed" does
            # not mean the form resolves (e.g. `Hermes 3 Llama-3.1 70B` /
            # `hermes-3-llama-3.1-70b` — leaf of `NousResearch/Hermes-3-Llama-3.1-
            # 70B` — is unresolvable until bridged). An unresolvable form cannot
            # collide with any existing alias, so porting it is collision-safe.
            forms = [cid, e.get("display_name")] + list(e.get("aliases") or [])
            for f in forms:
                if not isinstance(f, str) or not f:
                    continue
                nf = _norm(f)
                if nf in ports or _resolves(f):
                    continue
                ports[nf] = tgt
                port_display[nf] = f
            continue
        floor.append(e)

    # ----- KEPT floor entries: parent-edge repoint + alias hygiene -----------
    repointed = dropped_edges = 0
    for e in floor:
        ps = _parse_parents(e.get("parents"))
        if ps:
            new_ps = []
            for p in ps:
                pid = p.get("id") if isinstance(p, dict) else None
                if not pid or pid in gen_ids or pid in {x["id"] for x in floor}:
                    new_ps.append(p)
                    continue
                r = resolver.resolve(pid, "model")
                if _reproduced(pid, r):
                    p = {**p, "id": r.canonical_id}
                    repointed += 1
                    new_ps.append(p)
                else:
                    dropped_edges += 1  # parent gone -> drop the edge
            if isinstance(e.get("parents"), str):
                e["parents"] = json.dumps(new_ps)
            else:
                e["parents"] = new_ps

    stripped = redisplayed = 0
    for e in floor:
        cid = str(e["id"])
        own = {_norm(cid)}
        if "/" in cid:
            own.add(_norm(cid.split("/", 1)[1]))
        kept = []
        for a in e.get("aliases") or []:
            if isinstance(a, str) and _norm(a) in source_claims and _norm(a) not in own:
                stripped += 1
                continue
            kept.append(a)
        e["aliases"] = kept
        dn = e.get("display_name")
        if isinstance(dn, str) and _norm(dn) in source_claims and _norm(dn) not in own:
            e["display_name"] = cid.split("/", 1)[-1]
            redisplayed += 1

    # ----- WRITE core.yaml ---------------------------------------------------
    cur_doc = yaml.safe_load(CORE.read_text()) if CORE.exists() else {}
    skip_ids = cur_doc.get("skip_ids", []) if isinstance(cur_doc, dict) else []
    skip_src = cur_doc.get("skip_source_ids", []) if isinstance(cur_doc, dict) else []
    floor.sort(key=lambda e: str(e["id"]))
    out = {"skip_ids": skip_ids, "skip_source_ids": skip_src, "entries": floor}
    header = (
        "# Curated model canonicals — the MINIMAL override layer (generators are the\n"
        "# bulk source of truth under sources/*.generated.yaml). This floor holds the\n"
        "# entities NO generator reproduces (resolver exact/normalized): closed-API\n"
        "# (open_weights=False), hand-curated (resolution_source=NA), curated judgment\n"
        "# calls. Derived by scripts/oneshots/build_curated_core_floor.py. May use the\n"
        "# {skip_ids, skip_source_ids, entries} shape.\n"
    )
    CORE.write_text(header + yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=200))

    # ----- WRITE enrichments/aliases.yaml (alias bridges) --------------------
    # Group ported forms by target canonical -> {id, aliases:[...]}.
    by_target: dict[str, list[str]] = {}
    for nf, tgt in sorted(ports.items()):
        by_target.setdefault(tgt, []).append(port_display[nf])
    enrich_entries = [
        {"id": tgt, "aliases": sorted(set(als))}
        for tgt, als in sorted(by_target.items())
    ]
    ENRICH.parent.mkdir(parents=True, exist_ok=True)
    enrich_header = (
        "# Alias bridges — forms the curated oracle carried that the generators do\n"
        "# NOT, attached to the canonical the entry now resolves to. Gap-filling only\n"
        "# (each form was UNCLAIMED by any generator canonical), so these never\n"
        "# override or collide. Loader UNIONs these aliases onto the matching\n"
        "# canonical. Derived by scripts/oneshots/build_curated_core_floor.py.\n"
    )
    ENRICH.write_text(enrich_header + yaml.safe_dump(enrich_entries, sort_keys=False, allow_unicode=True, width=200))

    from collections import Counter
    src = Counter(str(e.get("resolution_source")) for e in floor)
    closed = sum(1 for e in floor if e.get("open_weights") is False)
    rev = sum(1 for e in floor if e.get("review_status") == "reviewed")
    print(f"reproduced & dropped: {repro}")
    print(f"alias bridges: {len(ports)} form(s) -> {len(enrich_entries)} target canonical(s)")
    print(f"floor parent edges: repointed {repointed}, dropped {dropped_edges}")
    print(f"floor alias hygiene: stripped {stripped}, re-derived {redisplayed} display(s)")
    print(f"curated floor: {len(floor)} entries (of {len(old_entries)} old-core)")
    print(f"  by resolution_source: {dict(src)}")
    print(f"  reviewed={rev}, open_weights=False(closed-API)={closed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
