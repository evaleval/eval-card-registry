"""Seed-time normalize-collision fold.

Different sources mint the SAME model under different separator spellings
(`google/gemini-1.5-pro` vs `google/gemini-1-5-pro`, `openai/gpt-5.2` vs the
venice relabel `openai/gpt-52`, `anthropic/claude-opus-4.5` vs `claude-opus4-5`).
Each becomes its own canonical → its own page. This pass folds the losers of a
collision group into one winner so there is ONE canonical (one page) per model:
the loser id + its aliases move onto the winner, and parent edges that pointed at
a loser are repointed to the winner.

Guards against FALSE merges — a separator difference that changes a parameter
SIZE's value (`opt-1.3b` is 1.3B params, `opt-13b` is 13B — different models). A
curated `never_fold` list covers any structural look-alikes the size guard can't
catch; a `prefer` map can pin the winner spelling for a collision key.
"""
import json as _json
import re
from collections import defaultdict

_SEP_RE = re.compile(r"[-_./]+")
_BSIZE_RE = re.compile(r"(\d+(?:\.\d+)?)b\b")
# Curated → curated/HF beat generators; dotted version spelling beats joined.
_SRC_RANK = {None: 3, "NA": 3, "hf": 3, "models_dev": 2, "inferred": 1}
# Placeholder org prefixes — a draft id whose org couldn't be resolved. Stripped
# for collision keying so `unknown/perplexity-sonar-reasoning` buckets with the
# real `perplexity/sonar-reasoning`, and never wins the fold over a real org.
_PLACEHOLDER_ORGS = frozenset({"unknown", "none", "na", ""})


def _strip_placeholder(s: str) -> str:
    if "/" in s:
        org, rest = s.split("/", 1)
        if org.lower() in _PLACEHOLDER_ORGS:
            return rest
    return s


def _has_real_org(s: str) -> bool:
    return "/" in s and s.split("/", 1)[0].lower() not in _PLACEHOLDER_ORGS


def collision_key(s: str) -> str:
    """Separator-, case- and placeholder-org-agnostic key (strips a leading
    unknown//none/ prefix, then - _ . /), so every spelling of the same name —
    including org-less and unresolved-org drafts — collapses into one bucket."""
    return _SEP_RE.sub("", _strip_placeholder(str(s)).lower())


def _bsizes(s: str):
    return tuple(sorted(float(m) for m in _BSIZE_RE.findall(str(s).lower())))


def _winner_rank(e: dict):
    id_ = e["id"]
    name = id_.split("/")[-1]
    return (
        _has_real_org(id_),                          # a real org prefix beats unknown//bare
        _SRC_RANK.get(e.get("resolution_source"), 0),
        "." in name,                                 # prefer a dotted version spelling
        -len(id_),                                   # then the shorter id
        id_,                                         # deterministic final tiebreak
    )


def _edges(v):
    if isinstance(v, str):
        try:
            return _json.loads(v) or []
        except Exception:
            return []
    return list(v) if isinstance(v, list) else []


def fold_collisions(entries: list[dict], never_fold=(), prefer=None, force_merge=None):
    """Fold normalize-colliding canonical model entries into one winner each.

    `force_merge` is an explicit {loser_id -> winner_id} map for duplicates that
    do NOT share a collision key (cross-spelling repos the resolver missed, e.g.
    `deepseek-v2-lite-chat` -> `deepseek-ai/DeepSeek-V2-Lite-Chat`); these fold
    through the same alias/parent-transfer path.

    Returns (surviving_entries, remap) where remap maps every folded loser id to
    its winner id. Pure function over the entry dicts (mutates winners in place).
    """
    prefer = prefer or {}
    never = [frozenset(p) for p in never_fold]
    by_id = {e["id"]: e for e in entries}

    groups: dict[str, list[str]] = defaultdict(list)
    for e in entries:
        groups[collision_key(e["id"])].append(e["id"])

    # Seed with the explicit cross-spelling merges (only those whose winner exists).
    remap: dict[str, str] = {
        l: w for l, w in (force_merge or {}).items() if l in by_id and w in by_id and l != w
    }
    for key, gids in groups.items():
        if len(gids) < 2:
            continue
        if sum(1 for i in gids if by_id[i].get("resolution_source") == "hf") > 1:
            # Two+ real HF repos that merely normalize-collide are DISTINCT
            # uploads (e.g. naps-...-v-0.1.0 vs ...-v0.1.0) — keep them separate.
            continue
        if len({_bsizes(i) for i in gids}) > 1:          # size-conflict guard
            continue
        if any(nf <= set(gids) for nf in never):          # curated never-fold
            continue
        if key in prefer and prefer[key] in gids:
            winner = prefer[key]
        else:
            winner = max(gids, key=lambda i: _winner_rank(by_id[i]))
        for lid in gids:
            if lid != winner and lid not in remap:
                remap[lid] = winner
    if not remap:
        return entries, {}

    surviving = [e for e in entries if e["id"] not in remap]
    surv_by_id = {e["id"]: e for e in surviving}

    # Move each loser's id + aliases onto its winner; backfill missing scalars.
    for lid, wid in remap.items():
        le, we = by_id[lid], surv_by_id[wid]
        aliases = list(we.get("aliases") or [])
        for a in [lid, *(le.get("aliases") or [])]:
            if a and a != wid and a not in aliases:
                aliases.append(a)
        we["aliases"] = aliases
        # Union the loser's parent edges into the winner — they are the same
        # model, so the winner inherits the loser's lineage (the repoint pass
        # below repoints folded parent ids, drops self-edges, and dedupes by id).
        le_edges = _edges(le.get("parents"))
        if le_edges:
            raw = we.get("parents")
            combined = _edges(raw) + le_edges
            we["parents"] = _json.dumps(combined) if isinstance(raw, str) else combined
        for f in ("release_date", "params_billions", "open_weights", "family", "architecture"):
            if we.get(f) in (None, "") and le.get(f) not in (None, ""):
                we[f] = le[f]

    # Repoint parent edges loser→winner, dropping self-edges and duplicates.
    for e in surviving:
        raw = e.get("parents")
        edges = _edges(raw)
        if not edges:
            continue
        out, seen, changed = [], set(), False
        for ed in edges:
            if not isinstance(ed, dict):
                out.append(ed)
                continue
            pid = ed.get("id")
            if pid in remap:
                pid = remap[pid]
                ed = {**ed, "id": pid}
                changed = True
            if pid == e["id"] or pid in seen:   # self-edge / duplicate after repoint
                changed = True
                continue
            seen.add(pid)
            out.append(ed)
        if changed:
            e["parents"] = _json.dumps(out) if isinstance(raw, str) else out

    return surviving, remap
