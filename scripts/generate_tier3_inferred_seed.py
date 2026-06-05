#!/usr/bin/env python3
"""
Generate the Tier-3 name-based-inference seed.

OFFLINE, deterministic, re-runnable. Mirrors the HF-oracle generator
(`generate_hf_oracle_seed.py`): the resolution gate is RESOLVE-based (it calls
`Resolver.resolve()`, not auto-create), so the residual no_match tail must be
SEEDED so every EEE id resolves to a non-null canonical.

What it does
------------
1. Re-runs `Resolver.resolve()` over all 6,720 EEE ids (the keys of
   `hf_model_id_resolution.json`) against the CURRENT fixtures + seed YAML and
   finds the ids that STILL return no_match (the residual tail after the
   Tier-1/Tier-2 sources + curation).

2. Mints a stable canonical for each residual id:
   - **org present** (`org/name`): canonical id = `{org}/{name}` with the
     two-tier org rule (big-dev namespace remap; HF/community org casing
     preserved verbatim otherwise). model-name casing preserved. The
     org goes through the org resolver/`hf_to_dev` so we never mint a name-only
     id that could collide across orgs.
   - **org-less** (no `/`, or a free-text label): a stable lowercase slug,
     `org_id = None`, `tags: [org-unknown]`, surfaced to
     `org_unknown_review.json` — NEVER auto-guess an org.

3. Adds the raw EEE id as an exact alias of the minted canonical so
   `resolve(raw)` hits it.

4. BASE INFERENCE (inferable bucket only): from the name tokens, detect a base
   family + (optionally) a derivation marker, build a candidate base id, and
   look it up against the CURRENT alias/canonical universe. Emit a typed parent
   edge (`finetune`, or `variant` with an axis) ONLY when that base
   alias-confirms to an existing canonical. NEVER invent an edge. The
   inferred-base org also lets us set the org for an org-less id ONLY when the
   raw id literally carries the org as a path prefix — otherwise org stays None.

5. Writes `seed/models/sources/tier3_inferred.generated.yaml`
   (`resolution_source: inferred`, `review_status: draft`).

6. Writes `specs/model-resolution-rework/org_unknown_review.json` — the
   org-null mints + a *proposed* org (from an inferred base, if any), NOT
   auto-applied.

Usage:
    LOCAL_MODE=true uv run python scripts/generate_tier3_inferred_seed.py
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import yaml

from eval_entity_resolver.resolver import Resolver
from eval_entity_resolver.strategies.fuzzy import _ORG_ALIASES

REGISTRY_ROOT = Path(__file__).resolve().parents[1]            # eval-card-registry/
EVALEVAL_ROOT = REGISTRY_ROOT.parent                           # evaleval/
ORACLE = EVALEVAL_ROOT / "hf_model_id_resolution.json"

SEED = REGISTRY_ROOT / "seed"
FIXTURES = REGISTRY_ROOT / "fixtures"
MODELS_DIR = SEED / "models"
SOURCES_DIR = MODELS_DIR / "sources"
TIER3_YAML = SOURCES_DIR / "tier3_inferred.generated.yaml"
ORGS_YAML = SEED / "orgs.yaml"
SPEC_DIR = REGISTRY_ROOT / "specs" / "model-resolution-rework"
REVIEW_JSON = SPEC_DIR / "org_unknown_review.json"


# --- Tier-3 lexicon ---------------------------------------------------------
# Base family tokens -> a canonical "stem" used to build the candidate base id
# for alias confirmation. Mapping a token to its canonical stem is conservative;
# the actual base must STILL alias-confirm against the registry before any edge
# is emitted.
BASE_FAMILY_TOKENS = [
    "tinyllama", "openhermes", "openchat", "mixtral",
    "llama", "mistral", "qwen", "gemma", "yi", "phi", "gpt", "claude",
    "gemini", "falcon", "bloom", "deepseek", "baichuan", "cohere", "command",
    "neural", "solar", "nous", "zephyr", "orca", "dolphin",
]

# Derivation markers: presence => finetune/merge (no axis). Order doesn't
# matter; we only use them as a boolean "this is a derivation" signal.
DERIVATION_MARKERS = [
    "dpo", "sft", "ft", "lora", "qlora", "merge", "slerp", "orpo", "kto",
    "uncensored", "abliterated", "ablated", "dare", "ties", "finetune",
    "instruct", "chat", "it", "base", "reasoning", "extended", "special",
]


def _clean_resolver() -> Resolver:
    """Resolver from fixtures with this generator's own prior output filtered
    out (`canonical_models.resolution_source == 'inferred'` rows AND their alias
    rows). Re-runnable: a re-run resolves against the SAME Tier-1/2 + curation
    universe regardless of stale inferred rows left in fixtures by a previous
    seed. If fixtures are already clean (no inferred rows) this is a no-op
    filter."""
    import pandas as pd
    from eval_entity_resolver import AliasStore, CanonicalStore

    cs = CanonicalStore.from_parquet(str(FIXTURES))
    models = cs._tables.get("model")
    inferred_ids: set[str] = set()
    if models is not None and not models.empty and "resolution_source" in models:
        mask = models["resolution_source"] == "inferred"
        inferred_ids = set(models.loc[mask, "id"].astype(str))
        cs._tables["model"] = models.loc[~mask].reset_index(drop=True)
        cs._index.clear()

    aliases_df = pd.read_parquet(FIXTURES / "aliases.parquet")
    if inferred_ids:
        aliases_df = aliases_df[
            ~aliases_df["canonical_id"].astype(str).isin(inferred_ids)
        ].reset_index(drop=True)
    alias_store = AliasStore(aliases_df, read_only=True)
    return Resolver(alias_store, canonical_store=cs)


def _load_curated_orgs() -> list[dict]:
    if not ORGS_YAML.exists():
        return []
    with open(ORGS_YAML) as f:
        return [e for e in (yaml.safe_load(f) or []) if isinstance(e, dict)]


def build_hf_to_dev(curated_orgs: list[dict]) -> dict[str, str]:
    """Two-tier org map: HF-org-lowercase -> curated developer slug.
    Single-sourced from `_ORG_ALIASES` + `seed/orgs.yaml:hf_org`."""
    hf_to_dev: dict[str, str] = {k.lower(): v for k, v in _ORG_ALIASES.items()}
    for row in curated_orgs:
        hf_org = row.get("hf_org")
        org_id = row.get("id")
        if isinstance(hf_org, str) and hf_org.strip() and isinstance(org_id, str):
            hf_to_dev[hf_org.lower()] = org_id
    return hf_to_dev


def canon_id_for_org_present(raw: str, hf_to_dev: dict[str, str]) -> tuple[str, str]:
    """`org/name` -> (canonical_id, org_id). canonical_id keeps the real
    `org/name` verbatim (org never folded into the id); org_id = the curated
    parent if the org maps to one, else the org verbatim."""
    org_part, name_part = raw.split("/", 1)
    org_id = hf_to_dev.get(org_part.lower(), org_part)
    return f"{org_part}/{name_part}", org_id


def _slug(value: str) -> str:
    """Stable lowercase slug for an org-less mint id (keeps the leading
    `inferred/` namespace out — these have no org). Collapses separators."""
    s = re.sub(r"[^\w\s\-./]", "", value.lower().strip())
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-/")
    return s or "model"


def _tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[\s\-_/.:]+", name.lower()) if t]


def detect_base_token(name: str) -> Optional[str]:
    toks = _tokens(name)
    tokset = set(toks)
    for fam in BASE_FAMILY_TOKENS:
        if fam in tokset:
            return fam
        # token may carry an attached version digit, e.g. `llama3`, `qwen2`.
        for t in toks:
            if t.startswith(fam) and t[len(fam):][:1].isdigit():
                return fam
    return None


def has_derivation_marker(name: str) -> bool:
    tokset = set(_tokens(name))
    return any(m in tokset for m in DERIVATION_MARKERS)


def main() -> None:
    oracle = json.loads(ORACLE.read_text())["resolutions"]
    curated_orgs = _load_curated_orgs()
    hf_to_dev = build_hf_to_dev(curated_orgs)

    # core.yaml `skip_ids` are intentionally-dropped canonical ids — a tier3
    # mint colliding with one would be silently removed by the seed loader and
    # the raw would stay no_match. Read them so we can suffix-disambiguate.
    core_skip_ids: set[str] = set()
    core_doc = yaml.safe_load((MODELS_DIR / "core.yaml").read_text()) or {}
    if isinstance(core_doc, dict):
        core_skip_ids = set(core_doc.get("skip_ids") or [])

    # Build a resolver from fixtures but with this generator's OWN prior output
    # (resolution_source == "inferred") filtered out, so a re-run sees the same
    # residual as a clean (Tier-1/2 + curation only) build. Without this, stale
    # inferred rows from a previous run/seed pollute the fixtures and the
    # residual collapses to ~0 (non-deterministic, non-re-runnable). Follows the
    # "read the source of truth, not the build artifact" rule.
    r = _clean_resolver()

    # --- alias-confirmation index: normalized base candidate -> canonical id.
    # Built from the resolver's own alias/canonical universe so an inferred base
    # only yields an edge when it alias-confirms to something that exists.
    from eval_card_registry.services.hub_stats import normalize as _nz

    def _confirm_base(candidate: str) -> Optional[str]:
        """Resolve a candidate base id through the live resolver. Accept only an
        EXACT or NORMALIZED match (no fuzzy) so we never invent a cross-version
        / cross-family edge."""
        res = r.resolve(candidate, "model")
        if res.canonical_id and res.strategy in ("exact", "normalized"):
            return res.canonical_id
        return None

    # --- find the CURRENT no_match residual over all 6,720 EEE ids. ----------
    residual: list[str] = []
    for raw in oracle:
        if r.resolve(raw, "model").canonical_id is None:
            residual.append(raw)

    minted: list[dict] = []
    minted_by_id: dict[str, dict] = {}
    review: list[dict] = []
    buckets = Counter()

    for raw in residual:
        has_org = "/" in raw and all(p.strip() for p in raw.split("/", 1))

        # Base inference from the model-name part (right of `/`) or whole id.
        name_for_infer = raw.split("/", 1)[1] if has_org else raw
        base_tok = detect_base_token(name_for_infer)
        deriv = has_derivation_marker(name_for_infer)

        # Try to alias-confirm a base. Candidate base id:
        #   org-present: `{org}/{base-stem}` AND bare `{base-stem}` (org-aware
        #               first so we never merge name-only across orgs).
        # We only attempt confirmation when a base token is present.
        parent_edge: Optional[dict] = None
        confirmed_base: Optional[str] = None
        if base_tok:
            # Build a small set of conservative candidate base ids from the
            # name: progressively drop trailing derivation tokens to land on a
            # base the registry knows. e.g. `llama-3.1-8b-instruct-dpo` ->
            # try `llama-3.1-8b-instruct`, `llama-3.1-8b`, `llama-3.1`...
            name_toks = _tokens(name_for_infer)
            # locate the base token start
            try:
                start = next(
                    i for i, t in enumerate(name_toks)
                    if t == base_tok or (t.startswith(base_tok) and t[len(base_tok):][:1].isdigit())
                )
            except StopIteration:
                start = 0
            tail = name_toks[start:]
            candidates: list[str] = []
            for end in range(len(tail), 0, -1):
                stem = "-".join(tail[:end])
                if has_org:
                    org_slug = raw.split("/", 1)[0]
                    dev = hf_to_dev.get(org_slug.lower(), org_slug)
                    candidates.append(f"{dev}/{stem}")
                candidates.append(stem)
            # de-dup preserving order; skip a candidate identical to raw itself.
            raw_name_nz = _nz(name_for_infer)
            seen = set()
            for cand in candidates:
                if cand in seen or _nz(cand) == _nz(raw):
                    continue
                seen.add(cand)
                hit = _confirm_base(cand)
                if not hit:
                    continue
                # Reject a "base" that is really the SAME model identity: same
                # full id (modulo separators) OR same model-NAME part as raw
                # (e.g. raw `01-ai/yi-lightning` confirming a bare org-less
                # `yi-lightning`). That would be a self-edge, not a base.
                hit_name = hit.split("/", 1)[1] if "/" in hit else hit
                if _nz(hit) == _nz(raw) or _nz(hit_name) == raw_name_nz:
                    continue
                confirmed_base = hit
                break
            if confirmed_base:
                # Derivation marker present => finetune (new release, no axis).
                # Otherwise it's a variant of a known base along the family
                # version line; emit a plain finetune edge unless the only
                # difference is a recognized variant axis. We stay conservative:
                # mark `finetune` (community uploads are overwhelmingly
                # finetunes/merges of the confirmed base).
                parent_edge = {"id": confirmed_base, "relationship": "finetune"}

        # --- mint id + org -------------------------------------------------
        if has_org:
            cid, org_id = canon_id_for_org_present(raw, hf_to_dev)
            bucket = "inferable-base" if confirmed_base else "opaque"
        else:
            # org-less: stable slug, org_id None, flagged for review. NEVER
            # auto-guess the org even if a base alias-confirms.
            cid = _slug(raw)
            org_id = None
            bucket = "org-less"

        # id-collision guard (CASE-INSENSITIVE). Two distinct raw ids can mint to
        # the SAME canonical (modulo case) only when they are the same model
        # under different casing — `Dracarys2-72B-Instruct` vs
        # `dracarys2-72b-instruct`, `Quazim0t0/ODB-14B-sce` vs `…ODB-14b-sce`.
        # That is a legitimate same-identity merge (NOT a name-only cross-org
        # merge — for org-present ids the org part is identical). Fold the raw
        # as an extra alias on the existing mint (keeping the first-seen casing
        # as the canonical id) rather than minting a case-variant duplicate
        # (which would violate the dedup gate) or dropping it (no_match).
        def _fold_dup(target_cid: str) -> bool:
            existing = minted_by_id.get(target_cid.lower())
            if existing is None:
                return False
            if raw not in existing["aliases"]:
                existing["aliases"].append(raw)
            return True

        if _fold_dup(cid):
            continue
        # If the id collides with an existing canonical in the registry (a
        # genuine cross-entity clash), or with a core `skip_ids` entry (would be
        # silently dropped by the seed loader), suffix-disambiguate so we never
        # name-only-merge onto an unrelated entity or vanish.
        if cid in core_skip_ids or r.resolve(cid, "model").canonical_id is not None:
            cid = f"{cid}-inferred"
            if _fold_dup(cid):
                continue

        # display_name = the raw EEE id verbatim. We deliberately do NOT use the
        # bare model-name part: the seed loop promotes display_name to a global
        # alias, and a bare name (e.g. `granite-3.1-2b-base`, `yi-lightning`)
        # frequently already belongs to a different canonical — that would be an
        # ambiguous alias collision (and a name-only cross-org merge risk). Using
        # the full raw id keeps the only emitted aliases = {cid, raw}, both
        # org-qualified.
        entry: dict = {
            "id": cid,
            "display_name": raw,
            "resolution_source": "inferred",
            "review_status": "draft",
            "resolution_granularity": "variant",
            "metadata": "{}",
            "aliases": [raw],
        }
        if org_id is not None:
            entry["org_id"] = org_id
        else:
            entry["tags"] = ["org-unknown"]
        if parent_edge is not None:
            entry["parents"] = [parent_edge]

        minted.append(entry)
        minted_by_id[cid.lower()] = entry   # case-insensitive dedup key
        buckets[bucket] += 1

        if org_id is None:
            proposed = (
                confirmed_base.split("/", 1)[0]
                if confirmed_base and "/" in confirmed_base else None
            )
            review.append({
                "raw_value": raw,
                "minted_canonical_id": cid,
                "inferred_base": confirmed_base,
                "proposed_org": proposed,
                "rationale": (
                    "inferred base org (review before applying)" if proposed
                    else "no recognizable org/base token"
                ),
                "status": "unreviewed",
            })

    # --- write outputs -------------------------------------------------------
    header = (
        "# AUTO-GENERATED by scripts/generate_tier3_inferred_seed.py — DO NOT HAND-EDIT.\n"
        "# Tier-3 name-based inference. Mints a stable canonical for every\n"
        "# EEE id that still no_match-es after Tier 1/2 + curation, so the\n"
        "# resolve-based gate returns non-null. resolution_source=inferred,\n"
        "# review_status=draft. Base edges are alias-confirmed only (NO invented\n"
        "# edges); org-less ids carry org_id=None + tags:[org-unknown] and are\n"
        "# surfaced to org_unknown_review.json (never auto-guessed).\n"
    )
    minted.sort(key=lambda e: e["id"])
    with open(TIER3_YAML, "w") as f:
        f.write(header)
        yaml.safe_dump(minted, f, sort_keys=False, allow_unicode=True, default_flow_style=False)

    REVIEW_JSON.write_text(json.dumps(
        {
            "_note": (
                "Org-less Tier-3 mints (org_id=None + tags:[org-unknown]). "
                "proposed_org is a SUGGESTION from an inferred base only; NOT "
                "auto-applied. A reviewer / EEE upstream fix must confirm."
            ),
            "count": len(review),
            "entries": sorted(review, key=lambda e: e["raw_value"]),
        },
        indent=1,
    ))

    print("Tier-3 residual (current no_match over 6,720):", len(residual))
    print("minted:", len(minted))
    print("buckets:", dict(buckets))
    print("org-less review entries:", len(review))
    print("wrote", TIER3_YAML.relative_to(REGISTRY_ROOT))
    print("wrote", REVIEW_JSON.relative_to(REGISTRY_ROOT))


if __name__ == "__main__":
    main()
