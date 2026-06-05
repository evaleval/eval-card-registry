"""Un-fold canonical_id to the real HF repo id + canonicalize org identity.

ONE rule everywhere: canonical_id = the real HF repo id (a consumer can build
`huggingface.co/{id}`); org_id = the canonical PARENT and carries all developer
grouping. id and org are decoupled.

Operates on the seed YAML SOURCE files (the source of truth; fixtures are a
build artifact). Idempotent — safe to re-run. Dry-run by default; pass --apply
to write.

Transforms, per model entry:
  1. UN-FOLD: a folded id (`alibaba/Qwen2-7B`) whose real HF id is present as an
     alias (`Qwen/Qwen2-7B`) is swapped: id := real HF id, old folded id demoted
     to alias. Detected via the old folding map (alias folds to the current id).
     API/non-HF models (`alibaba/qwen-flash`) have no such alias -> left as-is.
  2. CASING: a community org prefix in the wrong case (`prithivmlmods/X`) ->
     true HF casing (`prithivMLmods/X`), old id demoted to alias.
  3. MALFORMED: parse artifacts (`Gemini-3-Flash(12/...`) -> clean `{org}/{name}`.
  4. org_id canonicalization (all entries): new-org folds (facebook->meta,
     ibm-granite->ibm, ...) + community true-casing. The id is NOT folded.
Then rewrites `parents[].id` / `lineage_origin_model_org_id` references, rebuilds
`orgs.generated.yaml` (every referenced community org, true-cased, kind=community
-> no dangling FK, case-splits merged) and the curated `orgs.yaml` edits
(add ByteDance, fold ibm-granite into ibm).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SEED = ROOT / "seed"
MODELS_SRC = SEED / "models" / "sources"
CORE = SEED / "models" / "core.yaml"
ORGS_YAML = SEED / "orgs.yaml"
ORGS_GEN = SEED / "orgs.generated.yaml"
ORACLE = Path("/Users/jchim/projects/evaleval/hf_model_id_resolution.json")

MODEL_FILES = [
    MODELS_SRC / "hf_oracle.generated.yaml",
    MODELS_SRC / "hub_stats.generated.yaml",
    MODELS_SRC / "models_dev.generated.yaml",
    MODELS_SRC / "models_dev_catalog.generated.yaml",
    MODELS_SRC / "tier3_inferred.generated.yaml",
    CORE,
]

# New-org folds (org_id ONLY; the id keeps its real HF prefix). Keys lowercase.
NEW_FOLDS = {
    "facebook": "meta",
    "mistral": "mistralai",
    "mosaicml": "databricks",
    "databricks-mosaic-research": "databricks",
    "alibaba-aidc": "alibaba",
    "alibaba-nlp": "alibaba",
    "aws-prototyping": "amazon",
    "ibm-research": "ibm",
    "ibm-granite": "ibm",
    "bytedance-seed": "bytedance",
    # Separator/case variants of one community org that the case-insensitive
    # gate misses (differ by a separator, not just case) -> fold to the
    # hf-sourced / higher-population variant.
    "prime-intellect": "PrimeIntellect",
    "ennoai": "Enno-Ai",
    "contextual-ai": "ContextualAI",
    "abacus-ai": "abacusai",
    "sarvam-ai": "sarvam",
}


def ndup(s: str) -> str:
    """Size-preserving normalized NAME key for duplicate detection. Collapses
    letter->digit separators (`qwen-2` == `qwen2`) and VERSION dots
    (`2.5` == `2-5`), but KEEPS a decimal before a size unit so `opt-1.3b`
    (1.3B) never collides with `opt-13b` (13B)."""
    s = s.lower()
    s = re.sub(r"([a-z])[-_ /]+(\d)", r"\1\2", s)          # qwen-2 -> qwen2
    s = re.sub(r"(\d)\.(\d)(?![bmkt])", r"\1-\2", s)        # 2.5 -> 2-5; keep 1.3b
    s = re.sub(r"[-_ /]+", "-", s)                          # collapse separators
    return s

# Malformed org_ids (parse artifacts) -> (true id-prefix, canonical org_id).
MALFORMED = {
    "Gemini-3-Flash(12": ("google", "google"),
    "Gemini-3-Pro(11": ("google", "google"),
    "Seed-OSS-36B-Base(w": ("ByteDance-Seed", "bytedance"),
}


def load_entries(path: Path):
    """Returns (doc, entries). Handles a flat list OR core.yaml's nested
    `{skip_ids, skip_source_ids, entries}` shape. `entries` is a live reference
    into `doc`, so mutating it then writing `doc` persists the change."""
    if not path.exists():
        return None, []
    doc = yaml.safe_load(path.read_text()) or []
    if isinstance(doc, list):
        return doc, doc
    if isinstance(doc, dict) and isinstance(doc.get("entries"), list):
        return doc, doc["entries"]
    return doc, []


def build_hf_to_dev(curated_orgs):
    """Real-HF-org-lowercase -> canonical slug (the OLD folding map: what
    produced the folded ids). _ORG_ALIASES + curated hf_org."""
    sys.path.insert(0, str(ROOT / "packages" / "eval-entity-resolver" / "src"))
    from eval_entity_resolver.strategies.fuzzy import _ORG_ALIASES

    m = {k.lower(): v for k, v in _ORG_ALIASES.items()}
    for o in curated_orgs:
        hf = o.get("hf_org")
        oid = o.get("id")
        if isinstance(hf, str) and hf.strip() and isinstance(oid, str):
            m[hf.lower()] = oid
    return m


def build_true_case():
    """lowercased HF org -> authoritative HF casing (MOST COMMON spelling).
    Sourced from the frozen oracle PLUS hub-stats `metadata.hf_id` (real HF ids
    for ~3.8k models) so community orgs the oracle never saw (Tencent, BAAI,
    Nexusflow, ...) still get their true casing."""
    counts = defaultdict(Counter)
    for meta in json.loads(ORACLE.read_text())["resolutions"].values():
        fx = meta.get("fixed_hf_model_id")
        if isinstance(fx, str) and "/" in fx:
            counts[fx.split("/", 1)[0].lower()][fx.split("/", 1)[0]] += 1
    hub = MODELS_SRC / "hub_stats.generated.yaml"
    if hub.exists():
        for e in yaml.safe_load(hub.read_text()) or []:
            md = e.get("metadata") if isinstance(e, dict) else None
            if isinstance(md, str):
                try:
                    hfid = json.loads(md).get("hf_id")
                except Exception:
                    hfid = None
                if isinstance(hfid, str) and "/" in hfid:
                    counts[hfid.split("/", 1)[0].lower()][hfid.split("/", 1)[0]] += 1
    return {lo: c.most_common(1)[0][0] for lo, c in counts.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    curated_orgs = [
        e for e in (yaml.safe_load(ORGS_YAML.read_text()) or []) if isinstance(e, dict)
    ]
    hf_to_dev = build_hf_to_dev(curated_orgs)
    true_case = build_true_case()
    curated_ids = {o["id"] for o in curated_orgs if isinstance(o.get("id"), str)}

    def old_fold(hf_id: str) -> str:
        if "/" not in hf_id:
            return hf_id
        org, name = hf_id.split("/", 1)
        return f"{hf_to_dev.get(org.lower(), org)}/{name}"

    from eval_card_registry.services.hub_stats import normalize as _nz

    # Authoritative un-fold map from the frozen oracle: the folded id that
    # generate_hf_oracle_seed minted (old_fold(fixed)) -> the real HF id.
    oracle_unfold = {}
    oracle_fixed = set()  # the authoritative real HF ids (keeper preference in dup-merge)
    for meta in json.loads(ORACLE.read_text())["resolutions"].values():
        fx = meta.get("fixed_hf_model_id")
        if isinstance(fx, str) and "/" in fx:
            oracle_fixed.add(fx)
            folded = old_fold(fx)
            if folded != fx:
                oracle_unfold[folded] = fx
    # Normalized index so a models.dev / slugified representation of the SAME
    # folded model (`zai/glm-4-5`) un-folds to the same real HF id as the HF
    # entry (`zai/GLM-4.5` -> `zai-org/GLM-4.5`), preserving the seed merge.
    oracle_unfold_norm = {}
    for k, v in oracle_unfold.items():
        oracle_unfold_norm.setdefault(_nz(k), v)
    # Canonical parent slugs that orgs fold INTO (guards the normalized match
    # to only fold-org prefixes, never unrelated community models).
    fold_slugs = {v for k, v in hf_to_dev.items() if k != v.lower()}

    curated_lower = {c.lower(): c for c in curated_ids}

    def canon_org(org_id):
        if not isinstance(org_id, str) or not org_id:
            return org_id
        lo = org_id.lower()
        if lo in NEW_FOLDS:
            return NEW_FOLDS[lo]
        # A real HF org folds to its curated parent slug (NousResearch ->
        # nous-research, HuggingFaceH4 -> huggingface, qwen -> alibaba).
        if lo in hf_to_dev:
            return hf_to_dev[lo]
        # A curated org keeps its authored slug (e.g. `eleutherai`, not the HF
        # casing `EleutherAI`) — the id prefix carries the real HF casing, the
        # org_id stays the canonical developer slug.
        if lo in curated_lower:
            return curated_lower[lo]
        if lo in true_case and true_case[lo] != org_id:
            return true_case[lo]
        return org_id

    def find_real_hf_alias(cid, aliases):
        cands = [
            a
            for a in aliases
            if isinstance(a, str) and "/" in a and a != cid and old_fold(a) == cid
        ]
        if not cands:
            return None
        # prefer the alias whose org matches the authoritative HF casing
        for a in cands:
            org = a.split("/", 1)[0]
            if true_case.get(org.lower()) == org:
                return a
        return cands[0]

    id_rename = {}
    stats = Counter()
    samples = defaultdict(list)
    all_files = []

    for path in MODEL_FILES:
        doc, entries = load_entries(path)
        if doc is None:
            continue
        all_files.append((path, doc, entries))

    # Pass 1a — DISCOVERY: build the folded-id -> real-HF-id unfold map so EVERY
    # representation of a model (across source files) collapses to the SAME real
    # HF id (no split -> no alias collision). Three authoritative sources:
    all_ids = {
        e["id"]
        for _p, _d, entries in all_files
        for e in entries
        if isinstance(e, dict) and isinstance(e.get("id"), str)
    }
    unfold_map = {}
    # (a) a real HF id already present as an entry id -> its folded form maps back
    #     (catches a folded `zai/GLM-4.5` when the real `zai-org/GLM-4.5` exists).
    #     Only TRUE-CASED entry ids seed the map, so a mis-cased duplicate
    #     (`bytedance-seed/...` vs `ByteDance-Seed/...`) can't win first-seen.
    for rid in all_ids:
        if "/" not in rid:
            continue
        org = rid.split("/", 1)[0]
        if true_case.get(org.lower(), org) != org:
            continue
        folded = old_fold(rid)
        if folded != rid:
            unfold_map.setdefault(folded, rid)
    # (b) oracle fixed ids (the real id may not be present as an entry).
    for folded, real in oracle_unfold.items():
        unfold_map.setdefault(folded, real)
    # (c) the real HF id recorded in `metadata.hf_id` (hub-stats) — authoritative
    #     for the non-oracle tail the casing pass otherwise left mis-cased
    #     (e.g. `qwen/qwen2.5-coder-3b` -> `Qwen/Qwen2.5-Coder-3B`,
    #     `baichuan/baichuan2-13b-base` -> `baichuan-inc/Baichuan2-13B-Base`).
    #     Runs BEFORE the alias backstop: metadata.hf_id is the verified real id,
    #     so it must win over an alias that may only carry a lowercase form.
    for _p, _d, entries in all_files:
        for e in entries:
            if not isinstance(e, dict) or not isinstance(e.get("id"), str):
                continue
            cid = e["id"]
            if cid in unfold_map:
                continue
            md = e.get("metadata")
            if isinstance(md, str):
                try:
                    hfid = json.loads(md).get("hf_id")
                except Exception:
                    hfid = None
                if isinstance(hfid, str) and "/" in hfid and hfid != cid:
                    unfold_map[cid] = hfid
    # (d) the real HF id present only as an alias on a folded entry (backstop).
    for _p, _d, entries in all_files:
        for e in entries:
            if not isinstance(e, dict) or not isinstance(e.get("id"), str):
                continue
            cid = e["id"]
            if cid in unfold_map or "/" not in cid:
                continue
            aliases = [a for a in (e.get("aliases") or []) if isinstance(a, str)]
            real = find_real_hf_alias(cid, aliases)
            if real:
                unfold_map[cid] = real
    # normalized variants (slugified models.dev forms of a fold-org id).
    unfold_norm = {}
    for folded, real in unfold_map.items():
        if folded.split("/", 1)[0] in fold_slugs:
            unfold_norm.setdefault(_nz(folded), real)

    def resolve_new_id(cid):
        prefix = cid.split("/", 1)[0] if "/" in cid else ""
        if prefix in MALFORMED:
            return f"{MALFORMED[prefix][0]}/{cid.split('/', 1)[1]}"
        if cid in unfold_map:
            return unfold_map[cid]
        if "/" in cid and prefix in fold_slugs and _nz(cid) in unfold_norm:
            return unfold_norm[_nz(cid)]
        lp = prefix.lower()
        if "/" in cid and lp in true_case and true_case[lp] != prefix:
            return f"{true_case[lp]}/{cid.split('/', 1)[1]}"
        return cid

    # Pass 1b — APPLY uniformly to every entry.
    for _p, _d, entries in all_files:
        for e in entries:
            if not isinstance(e, dict) or not isinstance(e.get("id"), str):
                continue
            cid = e["id"]
            prefix = cid.split("/", 1)[0] if "/" in cid else ""
            new_id = resolve_new_id(cid)
            aliases = [a for a in (e.get("aliases") or []) if isinstance(a, str)]
            old_org = e.get("org_id")
            if prefix in MALFORMED:
                new_org = MALFORMED[prefix][1]
            elif new_id != cid and "/" in new_id:
                # Renamed to the real HF id -> derive org from the corrected
                # prefix (canon_org folds the HF org back to the parent), so a
                # metadata.hf_id fix like baichuan/x -> baichuan-inc/x gets the
                # right org instead of the stale one.
                new_org = canon_org(new_id.split("/", 1)[0])
            else:
                new_org = canon_org(old_org)
            if new_org != old_org:
                stats["org_repoint"] += 1
                if len(samples["org_repoint"]) < 8:
                    samples["org_repoint"].append(f"{cid}: {old_org} -> {new_org}")
            if new_id != cid:
                id_rename[cid] = new_id
                cat = (
                    "malformed"
                    if prefix in MALFORMED
                    else "unfold"
                    if (cid in unfold_map or (prefix in fold_slugs and _nz(cid) in unfold_norm))
                    else "casing"
                )
                stats[cat] += 1
                if len(samples[cat]) < 8:
                    samples[cat].append(f"{cid} -> {new_id}")
                aliases = [a for a in aliases if a != new_id]
                if cid not in aliases:
                    aliases.append(cid)
                e["aliases"] = aliases
            e["id"] = new_id
            if old_org is not None:
                e["org_id"] = new_org

    # Pass 1c — DUP-MERGE: collapse same-model duplicate canonicals the resolver
    # normalize misses (models.dev dashed slug `alibaba/qwen-2-5-7b-instruct` vs
    # real `Qwen/Qwen2.5-7B-Instruct`). Cluster the now-final ids by (canonical
    # org, size-preserving name); keep the real-HF member, re-point the rest.
    # EXCLUDE skip_ids: a skipped entry is dropped at seed and exists only to
    # route its aliases (e.g. the `minimax/minimax-m2` family) — merging it
    # breaks that routing.
    _skipped_dup = set()
    for _p, doc, _e in all_files:
        if isinstance(doc, dict):
            for key in ("skip_ids", "skip_source_ids"):
                for x in doc.get(key) or []:
                    if isinstance(x, str):
                        _skipped_dup.add(x)
    realness: dict = {}
    for _p, _d, entries in all_files:
        for e in entries:
            if not isinstance(e, dict) or not isinstance(e.get("id"), str):
                continue
            cid = e["id"]
            if cid in _skipped_dup:
                continue
            h = None
            md = e.get("metadata")
            if isinstance(md, str):
                try:
                    h = json.loads(md).get("hf_id")
                except Exception:
                    h = None
            real = e.get("resolution_source") == "hf" or (isinstance(h, str) and h == cid)
            realness[cid] = realness.get(cid, False) or real
    # Families with a hand-curated skip share a (canon_org, ndup) key — don't
    # blanket-merge those; the curation owns their structure (e.g. minimax keeps
    # M2 / M2.7 as DISTINCT models via skipped `MiniMaxAI/minimax-m2[.7]` dups,
    # and blanket-merging drops their EEE resolutions).
    skip_keys = set()
    for x in _skipped_dup:
        if "/" in x:
            so, sn = x.split("/", 1)
            skip_keys.add((canon_org(so), ndup(sn)))
    clusters: dict = defaultdict(set)
    for cid in {
        e["id"] for _p, _d, entries in all_files for e in entries
        if isinstance(e, dict) and isinstance(e.get("id"), str)
    }:
        if "/" not in cid or cid in _skipped_dup:
            continue
        org, name = cid.split("/", 1)
        clusters[(canon_org(org), ndup(name))].add(cid)
    dup_rename: dict = {}
    for key, members in clusters.items():
        if len(members) < 2 or key in skip_keys:
            continue
        # Merge dups of the SAME model. Keeper = the single real member (oracle
        # fixed id / hf-sourced) if present, else the lexicographically-smallest
        # (0-real API/spelling dups: `openai/gpt-4-1-...` <= `...gpt-4.1-...`).
        # SKIP >=2-real clusters (distinct real repos that normalize alike, e.g.
        # NAPS v-0.1.0 vs v0.1.0).
        reals = sorted(c for c in members if c in oracle_fixed or realness.get(c))
        if len(reals) >= 2:
            continue
        # Keeper = the single real member; else (0-real API/spelling dups) prefer
        # the version-DOTTED spelling (`gpt-4.1` over `gpt-4-1`) — the natural
        # canonical — then lexicographic for ties.
        keeper = reals[0] if reals else sorted(members, key=lambda c: (-c.count("."), c))[0]
        for c in members:
            if c != keeper:
                dup_rename[c] = keeper
    dup_demoted: set = set()  # ids demoted to aliases by the dup-merge (alias-drop must keep these)
    if dup_rename:
        for _p, _d, entries in all_files:
            for e in entries:
                if not isinstance(e, dict) or not isinstance(e.get("id"), str):
                    continue
                cid = e["id"]
                if cid in dup_rename:
                    new = dup_rename[cid]
                    al = [a for a in (e.get("aliases") or []) if isinstance(a, str) and a != new]
                    if cid not in al:
                        al.append(cid)
                    dup_demoted.add(cid)
                    e["aliases"] = al
                    e["id"] = new
                    # The merged-away dup contributes ONLY aliases to the keeper:
                    # strip identity scalars so a dup's `inferred`/draft state
                    # can't pollute the keeper (which owns the real identity).
                    e.pop("resolution_source", None)
                    e.pop("review_status", None)
                    id_rename[cid] = new
                    stats["dup_merged"] += 1
                    if len(samples["dup_merged"]) < 10:
                        samples["dup_merged"].append(f"{cid} -> {new}")

    # Final community-org case-dedup: if both `Tencent` (an HF repo) and
    # `tencent` (a bare API slug) end up referenced, collapse to ONE casing —
    # prefer the spelling with uppercase (real HF orgs are TitleCase).
    refs = Counter()
    for _p, _d, entries in all_files:
        for e in entries:
            if isinstance(e, dict) and isinstance(e.get("org_id"), str):
                refs[e["org_id"]] += 1
    by_lower = defaultdict(list)
    for org in refs:
        by_lower[org.lower()].append(org)
    recase = {}
    for lo, variants in by_lower.items():
        if len(variants) < 2:
            continue
        winner = max(variants, key=lambda s: (sum(c.isupper() for c in s), refs[s]))
        for v in variants:
            if v != winner:
                recase[v] = winner
    if recase:
        for _p, _d, entries in all_files:
            for e in entries:
                if not isinstance(e, dict):
                    continue
                if e.get("org_id") in recase:
                    e["org_id"] = recase[e["org_id"]]
                    stats["org_recased"] += 1
                if e.get("lineage_origin_model_org_id") in recase:
                    e["lineage_origin_model_org_id"] = recase[e["lineage_origin_model_org_id"]]

    # Drop aliases that resolve to a DIFFERENT canonical's final id: post-un-fold,
    # a curated bare-name node may alias what is now a real repo's identity
    # (e.g. `cohere/command-a-reasoning` aliasing `CohereLabs/command-a-reasoning-08-2025`).
    # An alias may not equal another canonical's id (nondeterministic owner).
    # EXCLUDE skip_ids/skip_source_ids: a SKIPPED entry is dropped at seed and is
    # NOT a canonical, so an alias routing that skipped id to a base (e.g.
    # `anthropic/claude-2` -> claude-2.0, `minimax/minimax-m2` -> a base) is
    # legitimate and must survive.
    _skipped = set()
    for _p, doc, _e in all_files:
        if isinstance(doc, dict):
            for key in ("skip_ids", "skip_source_ids"):
                for x in doc.get(key) or []:
                    if isinstance(x, str):
                        _skipped.add(x)
    final_ids = {
        e["id"]
        for _p, _d, entries in all_files
        for e in entries
        if isinstance(e, dict) and isinstance(e.get("id"), str)
    } - _skipped
    for _p, _d, entries in all_files:
        for e in entries:
            if not isinstance(e, dict) or not isinstance(e.get("id"), str):
                continue
            eid = e["id"]
            al = e.get("aliases")
            if not isinstance(al, list):
                continue
            kept = []
            for a in al:
                if isinstance(a, str) and a not in dup_demoted:
                    # A dup-demoted id is the OLD id of THIS merged model — keep
                    # it so its EEE raw resolves to the keeper (don't let
                    # resolve_new_id re-resolve it to a different canonical and
                    # drop it, which broke the minimax-m2 family).
                    r = resolve_new_id(a)
                    if r in final_ids and r != eid:
                        stats["alias_dropped"] += 1
                        continue
                kept.append(a)
            e["aliases"] = kept

    # Rewrite core.yaml's skip_ids / skip_source_ids exclusion lists through the
    # same un-fold (they reference folded ids that the seed compares against).
    for path, doc, _entries in all_files:
        if isinstance(doc, dict):
            for key in ("skip_ids", "skip_source_ids"):
                lst = doc.get(key)
                if isinstance(lst, list):
                    doc[key] = [
                        resolve_new_id(x) if isinstance(x, str) else x for x in lst
                    ]

    # Pass 2: rewrite parents[].id + lineage_origin_model_org_id references.
    for path, doc, entries in all_files:
        for e in entries:
            if not isinstance(e, dict):
                continue
            for edge in e.get("parents") or []:
                if isinstance(edge, dict) and isinstance(edge.get("id"), str):
                    # Resolve through the FULL un-fold (not just id_rename): a
                    # parent may reference a folded id that was never a top-level
                    # entry (e.g. `eleutherai/pythia-1b`), which the seed would
                    # otherwise materialise into a phantom lowercase canonical.
                    # Then chain the dup-merge rename so a parent pointing at a
                    # merged-away dup follows it to the keeper.
                    nid = resolve_new_id(edge["id"])
                    nid = dup_rename.get(nid, nid)
                    if nid != edge["id"]:
                        edge["id"] = nid
                        stats["parent_ref_rewritten"] += 1
            lo = e.get("lineage_origin_model_org_id")
            if isinstance(lo, str):
                nlo = canon_org(lo)
                if nlo != lo:
                    e["lineage_origin_model_org_id"] = nlo
                    stats["lineage_org_repoint"] += 1

    # Pass 3: collect every org_id referenced post-transform -> rebuild community rows.
    referenced = set()
    for path, doc, entries in all_files:
        for e in entries:
            if isinstance(e, dict) and isinstance(e.get("org_id"), str):
                referenced.add(e["org_id"])
    community = sorted(referenced - curated_ids)
    stats["community_orgs"] = len(community)
    # case-split / dangling sanity: any community org whose true casing differs?
    miscased = [c for c in community if true_case.get(c.lower(), c) != c]
    stats["community_miscased_remaining"] = len(miscased)

    print("=== transform stats (dry-run) ===" if not args.apply else "=== APPLIED ===")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print(f"  id renames total: {len(id_rename)}")
    for cat in ("unfold", "casing", "malformed", "org_repoint"):
        if samples[cat]:
            print(f"\n  sample {cat}:")
            for s in samples[cat]:
                print(f"    {s}")
    if miscased:
        print(f"\n  WARN miscased community orgs remaining: {miscased[:10]}")

    if not args.apply:
        print("\n(dry-run — pass --apply to write)")
        return

    # Write model files.
    for path, doc, entries in all_files:
        path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=4096))
    # Rebuild orgs.generated.yaml (community rows, true-cased, kind=community).
    gen_rows = [
        {
            "id": c,
            "display_name": c,
            "hf_org": c,
            "kind": "community",
            "tags": "[]",
            "metadata": "{}",
            "review_status": "reviewed",
        }
        for c in community
    ]
    ORGS_GEN.write_text(
        "# AUTO-GENERATED by scripts/canonicalize_orgs_and_unfold.py — DO NOT HAND-EDIT.\n"
        + yaml.safe_dump(gen_rows, sort_keys=False, allow_unicode=True, width=4096)
    )
    print(f"\nwrote {len(all_files)} model files + orgs.generated.yaml ({len(gen_rows)} community orgs)")


if __name__ == "__main__":
    main()
