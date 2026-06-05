#!/usr/bin/env python3
"""ONE-SHOT (model-resolution-rework): deconflict the curated floor against the
generator sources.

After build_curated_core_floor.py + the org-aware resolver fold, a residual set
of floor entries still duplicate a source canonical. Three classes, each fixed by
a deterministic pass — all holding to GENERATORS-ARE-THE-SOURCE-OF-TRUTH and
REAL-HF-ID-WINS:

  (A) SHADOW mints — a core (floor) models_dev mint that decide_fold()s onto a
      real HF id (`zai/z-ai-glm-5` -> real `zai-org/GLM-5`). The real HF source
      wins -> DROP the floor mint, PORT its id as an alias bridge to the target.
      Mirrors test_no_minted_modelsdev_canonical_shadows_real_hf_id.
  (B) SLUG dups — a floor REAL-HF entry (`xai-org/grok-1`) duplicated by a
      non-real SLUG in a generated source (`xai/grok-1` in tier3). The floor's
      real id is correct; the source slug is the stale dup -> add the slug to
      core `skip_source_ids` (drops it from sources at load). Mirrors
      test_no_real_hf_id_duplicated_by_slug.
  (C) EXACT collisions — the same literal alias/display string declared by a
      floor entry and a source canonical. Source wins -> drop the floor entry
      (authority tiebreak when both are floor). Iterates (re-seeding) until the
      seed reports none, so cascades from a drop are caught.

Finally (D) repoints/drops parent edges in surviving floor entries that point at
a now-dropped id (via the resolver on the final clean fixtures).

A pure source<->source collision is a GENERATOR bug, not a floor concern — it is
reported and left for the generator layer. No resolution outcome is lost
silently: dropped ids/aliases stay covered by the surviving source canonical (or
a ported bridge), and the coverage gate (every EEE id resolves) is the backstop.

Usage:  LOCAL_MODE=true uv run python scripts/oneshots/deconflict_floor.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
CORE = REPO / "seed" / "models" / "core.yaml"
ENRICH = REPO / "seed" / "models" / "enrichments" / "aliases.yaml"
ENRICH_PARENTS = REPO / "seed" / "models" / "enrichments" / "parents.yaml"
ORGS_YAML = REPO / "seed" / "orgs.yaml"
ORACLE = REPO / "curation" / "oracle_snapshot" / "canonical_models.parquet"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

_COLL_RE = re.compile(
    r"'(?P<rv>[^']+)' \(model[^)]*\): declared by both '(?P<a>[^']+)' and '(?P<b>[^']+)'"
)


# --------------------------------------------------------------------------
# core.yaml + sources I/O
# --------------------------------------------------------------------------
def _load_core() -> dict:
    doc = yaml.safe_load(CORE.read_text()) or {}
    if isinstance(doc, list):
        doc = {"entries": doc}
    doc.setdefault("entries", [])
    doc.setdefault("skip_ids", [])
    doc.setdefault("skip_source_ids", [])
    return doc


def _write_core(doc: dict) -> None:
    header = "\n".join(ln for ln in CORE.read_text().splitlines() if ln.startswith("#"))
    CORE.write_text((header + "\n" if header else "")
                    + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=200))


def _source_entries() -> list[dict]:
    out = []
    for p in sorted((REPO / "seed" / "models" / "sources").glob("*.generated.yaml")):
        d = yaml.safe_load(p.read_text())
        out.extend(d.get("entries") if isinstance(d, dict) else d or [])
    return [e for e in out if isinstance(e, dict) and e.get("id")]


def _append_bridges(bridges: dict[str, list[str]]) -> None:
    """bridges: {target_canonical_id: [alias forms]} -> UNION into enrichments."""
    existing = []
    if ENRICH.exists():
        existing = yaml.safe_load(ENRICH.read_text()) or []
    by_id: dict[str, set] = {}
    for e in existing:
        if isinstance(e, dict) and e.get("id"):
            by_id.setdefault(e["id"], set()).update(e.get("aliases") or [])
    for tgt, als in bridges.items():
        by_id.setdefault(tgt, set()).update(als)
    out = [{"id": t, "aliases": sorted(a)} for t, a in sorted(by_id.items())]
    header = "\n".join(ln for ln in (ENRICH.read_text().splitlines() if ENRICH.exists() else []) if ln.startswith("#"))
    ENRICH.write_text((header + "\n" if header else "")
                      + yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=200))


# --------------------------------------------------------------------------
# (A) shadow mints + (B) slug dups — mirror the gates EXACTLY, reading the
# SEEDED fixtures (canonical_models + the materialised alias table), so we
# catch precisely what test_no_minted_modelsdev_canonical_shadows_real_hf_id
# and test_no_real_hf_id_duplicated_by_slug flag (incl. dupes SURFACED by the
# floor's portage bridges, which only exist on the post-seed alias surface).
# Requires a CLEAN seed first (fixtures present). Returns #changes.
# --------------------------------------------------------------------------
def _fold_dedup_from_fixtures() -> int:
    import pandas as pd
    import fold_modelsdev_dupes as fold

    fixtures = REPO / "fixtures"
    if not (fixtures / "canonical_models.parquet").exists():
        return 0
    mdf = pd.read_parquet(fixtures / "canonical_models.parquet")

    # alias surface per canonical — exactly the gate's _model_aliases_by_canonical
    from collections import defaultdict
    alias_map: dict[str, list[str]] = defaultdict(list)
    apath = fixtures / "aliases.parquet"
    if apath.exists():
        adf = pd.read_parquet(apath)
        adf = adf[adf["entity_type"] == "model"]
        for row in adf.itertuples():
            cid = getattr(row, "canonical_id", None)
            rv = getattr(row, "raw_value", None)
            if isinstance(cid, str) and isinstance(rv, str) and rv:
                alias_map[cid].append(rv)

    def _g(row, k):
        v = getattr(row, k, None)
        return v if isinstance(v, str) else None

    entries = [{
        "id": str(r.id), "org_id": _g(r, "org_id"), "display_name": _g(r, "display_name"),
        "resolution_source": _g(r, "resolution_source"), "metadata": _g(r, "metadata"),
        "aliases": alias_map.get(str(r.id), []),
    } for r in mdf.itertuples()]

    core = _load_core()
    core_ids = {str(e["id"]) for e in core["entries"] if isinstance(e, dict) and e.get("id")}
    src_ids = {str(e["id"]) for e in _source_entries()}

    hf_to_dev = fold.build_hf_to_dev()
    hf_ids, alias_to_hf, by_org_name, _ = fold.build_hf_targets(entries, hf_to_dev)

    drop_ids: set[str] = set()
    skip_src: set[str] = set()
    bridges: dict[str, list[str]] = {}
    fold_map: dict[str, str] = {}  # dropped/skipped id -> surviving target
    alias_of = {e["id"]: e for e in entries}

    def _route(cid: str, tgt: str):
        if cid in core_ids:
            drop_ids.add(cid)
        elif cid in src_ids:
            skip_src.add(cid)
        else:
            return
        fold_map[cid] = tgt
        # COMPLETE fold: bridge EVERY form the folded entry carried (id + all its
        # materialised aliases + display) onto the target, so no raw that used to
        # hit `cid` (or one of its aliases, e.g. `cohere/command-r-plus` on
        # `cohere/cohere-command-r-plus`) regresses to no_match.
        forms = {cid, *(alias_of.get(cid, {}).get("aliases") or [])}
        dn = alias_of.get(cid, {}).get("display_name")
        if dn:
            forms.add(dn)
        bridges.setdefault(tgt, []).extend(forms)

    # ---- (A) SHADOW: models_dev mint folding to a real HF id ----
    for e in entries:
        if e["resolution_source"] != "models_dev":
            continue
        f = fold.decide_fold(e, hf_ids, alias_to_hf, by_org_name, hf_to_dev)
        if f is not None and f["hf_target"] != e["id"]:
            _route(e["id"], f["hf_target"])

    # ---- (B) SLUG: a real-HF id duplicated by a non-real folded/slug variant ----
    # gate ORACLE_PATH: tracked in-repo (curation/), local-dev fallback to root
    oracle_json = REPO / "curation" / "hf_model_id_resolution.json"
    if not oracle_json.exists():
        oracle_json = REPO.parent / "hf_model_id_resolution.json"
    fixed = set()
    if oracle_json.exists():
        for v in json.loads(oracle_json.read_text())["resolutions"].values():
            fx = v.get("fixed_hf_model_id")
            if isinstance(fx, str) and "/" in fx:
                fixed.add(fx)

    def _fold_org(org):
        return hf_to_dev.get(org.lower(), org)

    def _ndup(s):
        s = s.lower()
        s = re.sub(r"([a-z])[-_ /]+(\d)", r"\1\2", s)
        s = re.sub(r"(\d)\.(\d)(?![bmkt])", r"\1-\2", s)
        return re.sub(r"[-_ /]+", "-", s)

    def _is_real(e: dict) -> bool:
        cid = e["id"]
        if cid in fixed or e["resolution_source"] == "hf":
            return True
        md = e["metadata"]
        if isinstance(md, str):
            try:
                return json.loads(md).get("hf_id") == cid
            except Exception:
                return False
        return False

    clusters: dict[tuple, dict] = defaultdict(lambda: {"real": [], "nonreal": []})
    for e in entries:
        cid = e["id"]
        if "/" not in cid:
            continue
        org, name = cid.split("/", 1)
        clusters[(_fold_org(org), _ndup(name))]["real" if _is_real(e) else "nonreal"].append(cid)
    for c in clusters.values():
        if c["real"] and c["nonreal"]:
            tgt = sorted(c["real"])[0]  # the real-HF survivor
            for cid in c["nonreal"]:
                _route(cid, tgt)

    if drop_ids:
        core["entries"] = [e for e in core["entries"]
                           if not (isinstance(e, dict) and str(e.get("id")) in drop_ids)]
    if skip_src:
        core["skip_source_ids"] = sorted(set(core.get("skip_source_ids", [])) | skip_src)
    if drop_ids or skip_src:
        _write_core(core)
    if bridges:
        _append_bridges({t: sorted(set(a)) for t, a in bridges.items()})
    if fold_map:
        # COMPLETE the fold: repoint every parent edge that pointed at a folded id
        # (across ALL sources + core) to the surviving target, so dropping/skipping
        # the dup can't orphan a child's parent FK.
        _repoint_parents(fold_map)
    if drop_ids or skip_src:
        print(f"  (A/B) fold-dedup: dropped {len(drop_ids)} core mint(s), skip_source_ids += "
              f"{len(skip_src)} source dup(s) (all forms bridged, parents repointed). "
              f"e.g. {sorted(drop_ids | skip_src)[:6]}")
    return len(drop_ids) + len(skip_src)


def _repoint_parents(fold_map: dict[str, str]) -> None:
    """Repoint parent edges {id: X} -> {id: fold_map[X]} across every model source
    file AND core.yaml (in place, preserving header + dict shape)."""
    files = [CORE] + sorted((REPO / "seed" / "models" / "sources").glob("*.generated.yaml"))
    for path in files:
        doc = yaml.safe_load(path.read_text())
        is_dict = isinstance(doc, dict)
        entries = doc.get("entries", []) if is_dict else (doc or [])
        changed = False
        for e in entries:
            if not isinstance(e, dict):
                continue
            ps = e.get("parents")
            as_str = isinstance(ps, str)
            if as_str:
                try:
                    ps = json.loads(ps)
                except Exception:
                    continue
            if not ps:
                continue
            new = []
            for p in ps:
                pid = p.get("id") if isinstance(p, dict) else None
                if pid in fold_map and fold_map[pid] != pid:
                    new.append({**p, "id": fold_map[pid]})
                    changed = True
                else:
                    new.append(p)
            if changed:
                e["parents"] = json.dumps(new) if as_str else new
        if changed:
            header = "\n".join(ln for ln in path.read_text().splitlines() if ln.startswith("#"))
            out = {**doc, "entries": entries} if is_dict else entries
            path.write_text((header + "\n" if header else "")
                            + yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=200))


# --------------------------------------------------------------------------
# (C) exact alias collisions — re-seed and parse the authoritative report
# --------------------------------------------------------------------------
def _seed_collisions() -> list[tuple[str, str, str]]:
    env = {**os.environ, "LOCAL_MODE": "true", "COLUMNS": "3000"}
    for f in (REPO / "fixtures").glob("*.parquet"):
        f.unlink()
    out = subprocess.run(
        ["uv", "run", "eval-card-registry", "seed", "--local"],
        cwd=REPO, env=env, capture_output=True, text=True,
    )
    blob = (out.stdout + out.stderr).replace("\n", " ")
    return [(m.group("rv"), m.group("a"), m.group("b")) for m in _COLL_RE.finditer(blob)]


def _authority(e: dict) -> tuple:
    """Higher tuple wins a collision: (reviewed, hand-curated NA-source). A
    reviewed + NA-source floor entry (1,1) must beat a draft models_dev/inferred
    source (0,0) — so the curated id survives and the source yields."""
    rs = e.get("resolution_source")
    return (1 if e.get("review_status") == "reviewed" else 0, 1 if rs in (None, "NA", "") else 0)


def _leaf_key(cid: str) -> str:
    """Separator/case-collapsed model-NAME key (drops the org prefix)."""
    from eval_entity_resolver.normalization import normalize as _nz
    leaf = cid.split("/", 1)[1] if "/" in cid else cid
    return _nz(leaf).replace(" ", "")


def _same_model(a: str, b: str) -> bool:
    """Same model = same model-name leaf. This is SAFE here despite ignoring the
    developer because _same_model is only ever asked about a pair that ALREADY
    COLLIDES on an exact alias string (the seed flagged it). Two genuinely
    different-developer models that merely share a name do NOT share an exact
    alias, so they never reach this decision — they surface (if at all) as a
    stray-alias collision (different leaf) and take the strip branch instead.
    (A stricter org-aware test was tried but is unsafe: an id-level collision
    can't be resolved by stripping — only by folding — so requiring org
    agreement loops; that tightening is deferred until the strip path handles
    id-collisions.)"""
    return _leaf_key(a) == _leaf_key(b)


def _strip_forms(pairs: dict) -> int:
    """Remove specific stray alias/display strings from a canonical's entry,
    across core.yaml + every source file (in place). `pairs` = {cid: {forms}}.
    Used when two DISTINCT models collide on a stray alias: the alias is removed
    from the entry it does not name, leaving the rightful owner's claim intact."""
    if not pairs:
        return 0
    n = 0
    files = [CORE] + sorted((REPO / "seed" / "models" / "sources").glob("*.generated.yaml"))
    for path in files:
        doc = yaml.safe_load(path.read_text())
        is_dict = isinstance(doc, dict)
        entries = doc.get("entries", []) if is_dict else (doc or [])
        changed = False
        for e in entries:
            if not isinstance(e, dict):
                continue
            cid = str(e.get("id"))
            rm = pairs.get(cid)
            if not rm:
                continue
            kept = [a for a in (e.get("aliases") or []) if a not in rm]
            if len(kept) != len(e.get("aliases") or []):
                e["aliases"] = kept
                changed = True
                n += 1
            if e.get("display_name") in rm:
                e["display_name"] = cid.split("/", 1)[-1]  # re-derive from id leaf
                changed = True
                n += 1
        if changed:
            header = "\n".join(ln for ln in path.read_text().splitlines() if ln.startswith("#"))
            out = {**doc, "entries": entries} if is_dict else entries
            path.write_text((header + "\n" if header else "")
                            + yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=200))
    return n


def _entry_meta() -> dict:
    """{id: {authority, layer ('core'|'source'), forms}} over core + every source.
    `forms` = id + display_name + aliases (for bridging a yielded entry)."""
    meta: dict[str, dict] = {}

    def _add(e, layer):
        cid = str(e["id"])
        forms = {cid}
        dn = e.get("display_name")
        if isinstance(dn, str) and dn:
            forms.add(dn)
        forms.update(a for a in (e.get("aliases") or []) if isinstance(a, str) and a)
        # core wins layer label if an id appears in both (curated overrides source)
        if cid not in meta or layer == "core":
            meta[cid] = {"authority": _authority(e), "layer": layer, "forms": forms}
        else:
            meta[cid]["forms"] |= forms
    for e in _load_core()["entries"]:
        if isinstance(e, dict) and e.get("id"):
            _add(e, "core")
    for e in _source_entries():
        _add(e, "source")
    return meta


def _resolve_exact_collisions() -> int:
    """Resolve the seed's exact alias-string collisions: the HIGHER-authority
    canonical keeps the contended name; the LOWER-authority one YIELDS (a core
    entry is dropped, a source entry is skip_source_ids'd) and ALL its forms are
    bridged onto the winner so nothing it used to resolve regresses. This is a
    COMPLETE fold, and it never sacrifices a reviewed/NA curated floor entry to a
    draft source (the over-drop the review caught)."""
    MAX_PASSES = 10
    total = 0
    for _pass in range(1, MAX_PASSES + 1):
        collisions = _seed_collisions()
        if not collisions:
            print(f"  (C) pass {_pass}: 0 exact collisions — clean.")
            return total
        meta = _entry_meta()
        core = _load_core()
        drop_core: set[str] = set()
        skip_src: set[str] = set()
        bridges: dict[str, list[str]] = {}
        strip: dict[str, set] = {}  # cid -> {forms to remove} (distinct-model bad-alias)
        unknown = []
        for rv, a, b in collisions:
            if a not in meta or b not in meta:
                unknown.append((rv, a, b))
                continue
            # SAME model (id-leaf folds equal) -> a genuine duplicate: fold the
            # lower-authority one into the higher (drop core / skip source) and
            # bridge all its forms. DIFFERENT models that merely share the exact
            # string `rv` (a stray/bad alias, e.g. a 3.1 alias mistakenly on a 3.0
            # canonical) must NOT be merged — that conflates distinct releases
            # (the `allenai/Olmo-3.1-32B-Think` vs `olmo-3-32b-think` case). Strip
            # `rv` from the entry it does NOT name (host-stripped leaf mismatch),
            # else from the lower-authority one; the rightful owner keeps it.
            same_model = _same_model(a, b)
            # authority winner/loser (tie -> keep source, else keep `a`)
            if meta[a]["authority"] > meta[b]["authority"]:
                hi, lo = a, b
            elif meta[b]["authority"] > meta[a]["authority"]:
                hi, lo = b, a
            elif meta[b]["layer"] == "source" and meta[a]["layer"] == "core":
                hi, lo = b, a
            else:
                hi, lo = a, b
            if same_model:
                if meta[lo]["layer"] == "core":
                    drop_core.add(lo)
                else:
                    skip_src.add(lo)
                bridges.setdefault(hi, []).extend(meta[lo]["forms"])
            else:
                # strip rv from whichever entry it does NOT name (else the loser)
                rv_key = _leaf_key("x/" + rv.split("/", 1)[-1])
                wrong = (a if _leaf_key(a) != rv_key and _leaf_key(b) == rv_key
                         else b if _leaf_key(b) != rv_key and _leaf_key(a) == rv_key
                         else lo)
                strip.setdefault(wrong, set()).add(rv)
        if not (drop_core or skip_src or strip):
            print(f"  (C) pass {_pass}: {len(collisions)} collision(s) but none resolvable "
                  f"(ids not found in core/sources):")
            for rv, a, b in unknown[:20]:
                print(f"        {rv!r}: {a} | {b}")
            raise SystemExit(2)
        if drop_core:
            core["entries"] = [e for e in core["entries"]
                               if not (isinstance(e, dict) and str(e.get("id")) in drop_core)]
        if skip_src:
            core["skip_source_ids"] = sorted(set(core.get("skip_source_ids", [])) | skip_src)
        _write_core(core)
        yielded = drop_core | skip_src
        _append_bridges({t: sorted(set(fs)) for t, fs in bridges.items() if t not in yielded})
        n_strip = _strip_forms({c: f for c, f in strip.items() if c not in yielded})
        total += len(yielded)
        print(f"  (C) pass {_pass}: {len(collisions)} collision(s) -> dropped {len(drop_core)} core, "
              f"skipped {len(skip_src)} source, stripped {n_strip} stray alias(es) (folds bridged)")
    raise SystemExit("still colliding after max passes")


# --------------------------------------------------------------------------
# (D) parent-edge cleanup — repoint/drop edges to dropped ids
# --------------------------------------------------------------------------
def _clean_parents() -> int:
    import pandas as pd
    from eval_entity_resolver.resolver import Resolver

    fixtures = REPO / "fixtures"
    if not (fixtures / "canonical_models.parquet").exists():
        return 0
    canon = set(pd.read_parquet(fixtures / "canonical_models.parquet")["id"].astype(str))
    resolver = Resolver.from_parquet(fixtures)
    core = _load_core()
    repointed = dropped = 0
    for e in core["entries"]:
        if not isinstance(e, dict):
            continue
        ps = e.get("parents")
        if isinstance(ps, str):
            try:
                ps = json.loads(ps)
            except Exception:
                ps = []
        if not ps:
            continue
        new = []
        changed = False
        for p in ps:
            pid = p.get("id") if isinstance(p, dict) else None
            if not pid or pid in canon:
                new.append(p)
                continue
            r = resolver.resolve(pid, "model")
            if r.strategy in ("exact", "normalized", "fuzzy") and r.canonical_id:
                new.append({**p, "id": r.canonical_id})
                repointed += 1
            else:
                dropped += 1
            changed = True
        if changed:
            e["parents"] = json.dumps(new) if isinstance(e.get("parents"), str) else new
    if repointed or dropped:
        _write_core(core)
        print(f"  (D) parents: repointed {repointed}, dropped {dropped} dangling edge(s)")
    return repointed + dropped


def _replay_oracle() -> int:
    """(E) REPLAY the Phase-0 behavioral oracle: any oracle alias raw (or oracle
    canonical id) that no longer resolves against the final fixtures is bridged
    to the CURRENT home of its oracle model (where the oracle canonical now
    resolves). This guarantees 'never regress already-seen resolution' for the
    full 7196-canonical / 28900-alias snapshot — the forms the floor build's
    portage + the deconflict folds didn't already cover (e.g. dated/display
    spellings of closed-API models the generators don't carry). Gap-filling only:
    each ported form was no_match, so the bridge can't collide."""
    import pandas as pd
    from eval_entity_resolver.resolver import Resolver

    fixtures = REPO / "fixtures"
    if not (fixtures / "canonical_models.parquet").exists():
        return 0
    r = Resolver.from_parquet(fixtures)
    snap = REPO / "curation" / "oracle_snapshot"
    oc = pd.read_parquet(snap / "canonical_models.parquet")
    oa = pd.read_parquet(snap / "aliases.parquet")
    oa = oa[(oa["entity_type"] == "model") & (oa["status"] != "rejected")]

    home: dict[str, str] = {}

    def _home(ocanon: str):
        if ocanon not in home:
            home[ocanon] = r.resolve(ocanon, "model").canonical_id
        return home[ocanon]

    bridges: dict[str, set] = {}
    orphan = 0
    # alias raws: bridge to the current home of the oracle canonical they targeted
    for rv, ocanon in zip(oa["raw_value"], oa["canonical_id"]):
        if not isinstance(rv, str) or not rv:
            continue
        if r.resolve(rv, "model").canonical_id is not None:
            continue
        t = _home(str(ocanon)) if isinstance(ocanon, str) else None
        if t:
            bridges.setdefault(t, set()).add(rv)
        else:
            orphan += 1
    # oracle canonical ids themselves (id is its own raw): bridge the id to its
    # home only when the home is a DIFFERENT surviving canonical (else it already
    # resolves or is genuinely gone — the canonicals gate guards that).
    for oid in oc["id"]:
        oid = str(oid)
        if r.resolve(oid, "model").canonical_id is not None:
            continue
        t = _home(oid)
        if t and t != oid:
            bridges.setdefault(t, set()).add(oid)
        else:
            orphan += 1
    if bridges:
        _append_bridges({t: sorted(a) for t, a in bridges.items()})
        print(f"  (E) oracle replay: bridged {sum(len(a) for a in bridges.values())} "
              f"no_match oracle form(s) to their model's current home; {orphan} truly-orphaned (no home).")
    else:
        print(f"  (E) oracle replay: nothing to bridge; {orphan} truly-orphaned.")
    return sum(len(a) for a in bridges.values())


def _append_parents(by_id_parents: dict) -> None:
    """UNION typed parent edges into enrichments/parents.yaml, keyed by the
    surviving canonical id (deduped by parent edge id)."""
    existing = yaml.safe_load(ENRICH_PARENTS.read_text()) or [] if ENRICH_PARENTS.exists() else []
    by_id: dict[str, dict] = {}
    for e in existing:
        if isinstance(e, dict) and e.get("id"):
            by_id[e["id"]] = e
    for tid, plist in by_id_parents.items():
        ent = by_id.setdefault(tid, {"id": tid})
        cur = ent.get("parents") or []
        seen = {p["id"] for p in cur if isinstance(p, dict) and p.get("id")}
        for p in plist:
            if p["id"] not in seen:
                cur.append(p)
                seen.add(p["id"])
        ent["parents"] = cur
    out = [by_id[k] for k in sorted(by_id)]
    header = (
        "# Curated typed-edge graph (parents) the generators cannot reproduce —\n"
        "# the Phase-0 oracle's lineage (variant axes incl. training_stage/version,\n"
        "# finetune/merge/quantized/adapter) for models whose source baseModels are\n"
        "# absent from the frozen inputs. Repointed to surviving canonicals + UNIONed\n"
        "# onto them at seed (cli.py _merge_into). Derived by deconflict_floor.py.\n"
    )
    ENRICH_PARENTS.write_text(header + yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=200))


def _replay_oracle_edges() -> int:
    """(E2) PRESERVE THE TYPED EDGE GRAPH. For every Phase-0 oracle canonical,
    re-attach its typed parent edges (relationship + axis: variant/finetune/
    quantized/merge/adapter, every axis) onto the canonical it now resolves to,
    with each PARENT target repointed to its surviving canonical. These edges
    live only in the oracle (the source baseModels are missing from the frozen
    inputs, and variant-axis edges like training_stage are name-derived), so a
    generator can't reproduce them — they are curated lineage, the floor pattern
    for edges. UNIONed onto the canonical's own parents (additive; a generator
    edge to the same parent keeps the oracle's typed rel/axis as the contract)."""
    import json as _json
    import pandas as pd
    from eval_entity_resolver.resolver import Resolver

    fixtures = REPO / "fixtures"
    if not (fixtures / "canonical_models.parquet").exists():
        return 0
    r = Resolver.from_parquet(fixtures)
    oc = pd.read_parquet(REPO / "curation" / "oracle_snapshot" / "canonical_models.parquet")

    home: dict[str, str] = {}

    def _home(i: str):
        if i not in home:
            home[i] = r.resolve(i, "model").canonical_id
        return home[i]

    def _edges(v):
        if isinstance(v, str):
            try:
                return _json.loads(v) or []
            except Exception:
                return []
        return v if isinstance(v, list) else []

    by_t: dict[str, dict] = {}
    dropped = 0
    for _, row in oc.iterrows():
        edges = _edges(row.get("parents"))
        if not edges:
            continue
        t = _home(str(row["id"]))
        if not t:
            continue  # entry gone (canonicals gate guards that)
        for e in edges:
            if not isinstance(e, dict) or not e.get("id"):
                continue
            tp = _home(str(e["id"]))
            if not tp or tp == t:  # parent gone, or self-edge
                dropped += 0 if tp == t else 1
                continue
            edge = {"id": tp, "relationship": e.get("relationship")}
            if e.get("axis"):
                edge["axis"] = e.get("axis")
            by_t.setdefault(t, {})[(tp, e.get("relationship"), e.get("axis"))] = edge
    if by_t:
        _append_parents({t: list(d.values()) for t, d in by_t.items()})
        n_edges = sum(len(d) for d in by_t.values())
        print(f"  (E2) oracle edge replay: restored {n_edges} typed parent edge(s) onto "
              f"{len(by_t)} canonical(s); {dropped} edge(s) dropped (parent gone).")
    return len(by_t)


def main() -> int:
    print("deconflict floor vs sources:")
    # (C) FIRST resolve exact alias collisions so the seed succeeds (fixtures
    #     needed by the fixtures-based fold-dedup below).
    n_exact = _resolve_exact_collisions()
    # (A/B) Then fold-dedup against the SEEDED fixtures (mirrors the gates). A
    #       change edits core/skip_source_ids + bridges -> reseed and re-run
    #       both passes until stable (a bridge can surface a fresh dupe; an
    #       exact collision can reappear). Bounded.
    n_fold = 0
    for _ in range(6):
        f = _fold_dedup_from_fixtures()
        n_fold += f
        if not f:
            break
        if _resolve_exact_collisions() == 0:
            continue
    # (D) repoint/drop parent edges to dropped ids, then confirm still clean.
    n_par = _clean_parents()
    if n_par:
        leftover = _seed_collisions()
        if leftover:
            print(f"  WARNING: {len(leftover)} collision(s) reappeared after parent cleanup")
            return 1
    # (E) replay the Phase-0 oracle so no already-seen resolution regresses. May
    #     add bridges -> reseed once and confirm still collision-free.
    n_replay = _replay_oracle()
    if n_replay:
        leftover = _seed_collisions()
        if leftover:
            print(f"  WARNING: {len(leftover)} collision(s) after oracle replay (bridges collided)")
            return 1
    # (E2) preserve the typed EDGE graph (parents) the generators can't reproduce.
    n_edges = _replay_oracle_edges()
    if n_edges:
        leftover = _seed_collisions()
        if leftover:
            print(f"  WARNING: {len(leftover)} collision(s) after edge replay")
            return 1
    print(f"deconflict done: exact-drops {n_exact}, fold-dedup {n_fold}, parent-fixes {n_par}, "
          f"oracle-replay {n_replay}, edge-replay {n_edges}; core now {len(_load_core()['entries'])} entries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
