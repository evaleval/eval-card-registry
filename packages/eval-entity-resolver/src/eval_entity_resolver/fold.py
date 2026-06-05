"""Org-aware fold decision: does a minted off-HF / dev-org-slug canonical refer
to the SAME model as a real HF repo already in the registry?

This is the single source of truth for the cross-source same-model dedup used by
BOTH the generator (scripts/refresh_from_modelsdev.py reconciliation, so a mint
DEFERS to the real HF id at generate time) AND the gate
(tests/test_gate_invariants.py via scripts/fold_modelsdev_dupes.py, which verifies
none survive). Keeping it here means the two can never drift.

A match is "confident" only with ORG AGREEMENT (after the curated two-tier dev-org
remap: meta-llama->meta, Qwen->alibaba, ...): a group with no resolvable org, or
whose name matches only under a DIFFERENT developer, NEVER folds (no cross-vendor
false merge). Tiers, strongest first: exact id == HF id; mint string already an
alias of an HF entry; normalized-name (all separators removed) + org agreement;
brand-prefix-stripped name + org agreement (so `qwen-qwq-32b` -> `QwQ-32B`).

`name_norm` removes ALL separators to ONE token, so models.dev's mangled
spellings collapse onto HF casing: `qwen-2-5-14b-instruct` == `Qwen2.5-14B-Instruct`.
"""
from __future__ import annotations

import re
from typing import Optional

from eval_entity_resolver.normalization import normalize as _nz


# Curated HF-namespace -> developer-org remap. The SINGLE owner of this map
# (moved here from strategies/fuzzy.py so both the resolver and the seed
# generators consume one source). HF namespaces that are alternate spellings of
# one developer fold to that developer's slug (org_id only — a canonical_id keeps
# its real HF repo prefix). Lowercased keys.
_ORG_ALIASES: dict[str, str] = {
    "deepseek-ai": "deepseek",
    "cohereforai": "cohere",
    "cohere-labs": "cohere",
    # HF renamed the Cohere org `CohereForAI` -> `CohereLabs` (no hyphen);
    # both are the same lab, canonical `cohere`.
    "coherelabs": "cohere",
    # HF's SmolLM team `HuggingFaceTB` is part of Hugging Face.
    "huggingfacetb": "huggingface",
    # Baichuan is a curated lab; its HF namespace `baichuan-inc` folds into it.
    "baichuan-inc": "baichuan",
    # HF `MiniMaxAI` / `SarvamAI` namespaces -> the lab slug we already use.
    "minimaxai": "minimax",
    "sarvamai": "sarvam",
    "tii-uae": "tiiuae",
    "meta-llama": "meta",
    "mistral-ai": "mistralai",
    "nvidia-nemo": "nvidia",
    # Zhipu/Z.ai → zai. `THUDM` is the legacy HF org for the GLM/ChatGLM
    # family (Tsinghua/Zhipu); HF now publishes under `zai-org`.
    "zhipu": "zai",
    "zhipu-ai": "zai",
    "z-ai": "zai",
    "zai-org": "zai",
    "thudm": "zai",
    # Moonshot → moonshotai
    "moonshot": "moonshotai",
    "moonshot-ai": "moonshotai",
    # Qwen models live under canonical org `alibaba` (Alibaba Cloud).
    # HF uploads use the `Qwen/` namespace (e.g. Qwen/Qwen2-VL-7B-Instruct).
    # The reverse mapping (alibaba → qwen) was rejected because
    # `alibaba__mineru2-pipeline` is a non-Qwen entry; this direction has
    # no analogous collision since every `qwen/<X>` upstream id we've seen
    # corresponds to an Alibaba/Qwen-family model.
    "qwen": "alibaba",
    # Alternate HF namespaces of a known developer fold to the one parent org
    # (org_id only — the canonical_id keeps the real HF repo prefix). These
    # consolidate the developer in downstream listings.
    "facebook": "meta",        # Meta's pre-Llama HF org (OPT, BART, ...)
    "mistral": "mistralai",
    "mosaicml": "databricks",  # MosaicML (MPT) acquired by Databricks
    "databricks-mosaic-research": "databricks",
    "alibaba-aidc": "alibaba",
    "alibaba-nlp": "alibaba",
    "aws-prototyping": "amazon",
    "ibm-research": "ibm",
    "ibm-granite": "ibm",      # Granite folded into IBM (curation decision)
    "bytedance-seed": "bytedance",
}


def build_curated_org_map(orgs_yaml_entries: list[dict]) -> dict[str, str]:
    """The SINGLE curated HF-namespace -> developer-org map every generator and
    the resolver should use: `_ORG_ALIASES` UNION every curated org's id +
    `hf_org` + each entry in its `aliases`, keyed lowercase -> curated id. The
    curated seed (orgs.yaml) wins over `_ORG_ALIASES` on conflict.

    Every generator and the resolver build their fold map here so a curated
    alias added to orgs.yaml automatically reaches every consumer (no drift)."""
    m: dict[str, str] = {k.lower(): v for k, v in _ORG_ALIASES.items()}
    for e in orgs_yaml_entries or []:
        if not isinstance(e, dict):
            continue
        oid = e.get("id")
        if not isinstance(oid, str) or not oid:
            continue
        m[oid.lower()] = oid
        hf_org = e.get("hf_org")
        if isinstance(hf_org, str) and hf_org.strip():
            m[hf_org.lower()] = oid
        for a in (e.get("aliases") or []):
            if isinstance(a, str) and a.strip():
                m[a.lower()] = oid
    return m


def build_org_dev_map_from_store(org_records, org_alias_pairs) -> dict[str, str]:
    """Same dev-org map as `build_curated_org_map`, for STORE-backed callers that
    don't have orgs.yaml at hand (the live resolution_service; seed-time lineage
    derivation; the deployed Space reads from the HF dataset, not seed files).
    `canonical_orgs` has no `aliases` column, so the alias tier lives as org rows
    in the alias table — feed it via `org_alias_pairs` ((raw, canonical_id) of
    entity_type=org). Identical result to build_curated_org_map over orgs.yaml, so
    every consumer folds orgs the same way (single source, no drift)."""
    m = build_curated_org_map([
        {"id": r.get("id"), "hf_org": r.get("hf_org")} for r in (org_records or [])
    ])
    for raw, cid in org_alias_pairs or []:
        if isinstance(raw, str) and raw and isinstance(cid, str) and cid:
            m[raw.lower()] = cid
    return m


def _norm_org_key(org: str) -> str:
    """Separator/case-insensitive org key (one token) for community-casing folds."""
    return re.sub(r"[^a-z0-9]", "", org.lower())


def build_community_casing(org_prefixes: list[str]) -> dict[str, str]:
    """Map a separator/case-insensitive org key -> the authoritative HF-true
    casing, derived from real-HF repo org prefixes (the hf_oracle / hub_stats
    sources). Lets every generator snap a community org (no curated id) to ONE
    canonical spelling so `Sao10K`/`sao10k`/`sao10K` collapse to one
    canonical_orgs row. Deterministic tie-break (sorted) when a key has >1 real
    spelling — so a refresh never flips a previously-chosen casing."""
    by_key: dict[str, set[str]] = {}
    for p in org_prefixes:
        if isinstance(p, str) and p.strip():
            by_key.setdefault(_norm_org_key(p), set()).add(p)
    return {k: sorted(v)[0] for k, v in by_key.items() if v}


def canonicalize_org(
    prefix: str,
    curated_map: dict[str, str],
    community_casing: Optional[dict[str, str]] = None,
    distinct_allowlist: Optional[set[str]] = None,
) -> str:
    """Canonical org id for an HF org spelling: (1) curated developer id when the
    prefix folds to one; (2) else the authoritative HF-true community casing
    (unless the spelling is on the distinct-org allowlist, which keeps verified
    separate uploaders apart); (3) else the prefix verbatim. The single org
    canonicalizer all generators + the reconcile call."""
    if not prefix:
        return prefix
    curated = curated_map.get(prefix.lower())
    if curated is not None:
        return curated
    if distinct_allowlist and prefix in distinct_allowlist:
        return prefix
    if community_casing:
        cased = community_casing.get(_norm_org_key(prefix))
        if cased is not None:
            return cased
    return prefix


def dev_org_of_prefix(prefix: str, hf_to_dev: dict[str, str]) -> str:
    """Remap an id-prefix org to its curated developer slug (else itself)."""
    return hf_to_dev.get(prefix.lower(), prefix)


def name_norm(name: str) -> str:
    """Normalized name with separators collapsed AND removed (one token), so
    `Qwen2.5-14B-Instruct` and `qwen-2-5-14b-instruct` map to the same token."""
    return _nz(name).replace(" ", "")


def brand_tokens_for(dev_org: str, hf_to_dev: dict[str, str]) -> set[str]:
    """Brand tokens a models.dev key may glue onto a model name for this
    developer: the dev slug + every HF alias mapping to it (alibaba -> {alibaba,
    qwen, ...}; meta -> {meta, meta-llama, facebook}), plus a few spelling
    variants not in the org map. Normalized to single tokens."""
    toks: set[str] = set()
    d = dev_org.lower()
    toks.add(name_norm(d))
    for hf_alias, dev in hf_to_dev.items():
        if dev.lower() == d:
            toks.add(name_norm(hf_alias))
    extra = {
        "alibaba": {"qwen"},
        "meta": {"llama"},
        "minimax": {"minimax"},
        "google": {"gemini", "gemma"},
        "deepseek": {"deepseek"},
    }
    toks |= extra.get(d, set())
    return {t for t in toks if t}


def strip_brand_prefix(norm_name_tok: str, brands: set[str]) -> set[str]:
    """Candidate name tokens with a leading brand token removed (always includes
    the original). Strips repeatedly (defensive against `qwen-qwen-...`)."""
    out = {norm_name_tok}
    cur = norm_name_tok
    changed = True
    while changed:
        changed = False
        for b in sorted(brands, key=len, reverse=True):
            if b and cur.startswith(b) and len(cur) > len(b):
                cur = cur[len(b):]
                out.add(cur)
                changed = True
                break
    return out


def build_hf_index(
    entries: list[dict],
    hf_to_dev: dict[str, str],
    fixed_ids: frozenset[str] = frozenset(),
):
    """Build the HF target authority from registry `entries` (+ any extra
    real-HF `fixed_ids`, e.g. from the frozen oracle):
      - hf_ids: every real-HF canonical id (resolution_source == 'hf' OR in
        fixed_ids) — for exact id match.
      - alias_to_hf: every id/display/alias string on an HF entry -> that HF id.
      - by_org_name: (dev_org, name_norm) -> hf_id — for org-aware normalized match.
      - hf_entry_by_id: id -> entry (so callers can merge onto it).
    """
    hf_entry_by_id: dict[str, dict] = {}
    hf_ids: set[str] = set(fixed_ids)
    for e in entries:
        if not isinstance(e, dict):
            continue
        cid = e.get("id")
        if not isinstance(cid, str):
            continue
        if e.get("resolution_source") == "hf" or cid in fixed_ids:
            hf_ids.add(cid)
            hf_entry_by_id[cid] = e

    alias_to_hf: dict[str, str] = {}
    by_org_name: dict[tuple[str, str], str] = {}

    def index_target(cid: str, entry: Optional[dict]) -> None:
        if "/" not in cid:
            return
        org, name = cid.split("/", 1)
        dev = dev_org_of_prefix(org, hf_to_dev)
        by_org_name.setdefault((dev, name_norm(name)), cid)
        alias_to_hf.setdefault(cid, cid)
        if entry is not None:
            dn = entry.get("display_name")
            if isinstance(dn, str):
                alias_to_hf.setdefault(dn, cid)
            for a in entry.get("aliases") or []:
                if isinstance(a, str):
                    alias_to_hf.setdefault(a, cid)

    for cid, e in hf_entry_by_id.items():
        index_target(cid, e)
    for cid in fixed_ids:
        if cid not in hf_entry_by_id:
            index_target(cid, None)

    return hf_ids, alias_to_hf, by_org_name, hf_entry_by_id


def decide_fold(mint: dict, hf_ids, alias_to_hf, by_org_name, hf_to_dev) -> Optional[dict]:
    """Return a fold dict (mint_id, hf_target, match_type, org agreement,
    evidence) when `mint` confidently refers to the same model as a real HF id;
    else None (never a cross-developer false merge)."""
    cid = mint.get("id")
    if not isinstance(cid, str):
        return None
    mint_org = mint.get("org_id")
    mint_org = mint_org if isinstance(mint_org, str) and mint_org else None
    prefix_org = dev_org_of_prefix(cid.split("/", 1)[0], hf_to_dev) if "/" in cid else None
    eff_org = mint_org or prefix_org

    mint_strings = [cid]
    dn = mint.get("display_name")
    if isinstance(dn, str):
        mint_strings.append(dn)
    for a in mint.get("aliases") or []:
        if isinstance(a, str):
            mint_strings.append(a)

    # exact id equality
    for s in mint_strings:
        if s in hf_ids and s != cid:
            return _mk(mint, s, "exact", eff_org, hf_to_dev, f"mint string {s!r} is a real HF id")
    # alias linkage (mint string already an alias of an HF entry). A mint string
    # can be a generic BARE name (e.g. `gemma-3-4b-it`) that DISTINCT developers
    # legitimately both carry (google's gemma AND unsloth's re-upload). Such a
    # shared alias must NOT link the mint across developers, so require org
    # agreement (after the dev remap) when BOTH the mint's effective org and the
    # target's developer are known and they DISAGREE — skip that match instead of
    # false-merging unsloth/gemma-3-4b-it into google/gemma-3-4b-it. (A full
    # org/model HF-id match is handled by the exact tier above, which is
    # unambiguous and stays unguarded.)
    for s in mint_strings:
        tgt = alias_to_hf.get(s)
        if tgt and tgt != cid and tgt in hf_ids:
            tgt_org = dev_org_of_prefix(tgt.split("/", 1)[0], hf_to_dev) if "/" in tgt else None
            if eff_org and tgt_org and eff_org.lower() != tgt_org.lower():
                continue
            return _mk(mint, tgt, "alias", eff_org, hf_to_dev, f"mint string {s!r} declared on {tgt!r}")

    if eff_org is None:
        return None  # no org agreement possible -> never fold

    cand_names = {(s.split("/", 1)[1] if "/" in s else s) for s in mint_strings}
    cand_names = {name_norm(n) for n in cand_names}

    # normalized-name equality + org agreement
    for nm in cand_names:
        tgt = by_org_name.get((eff_org, nm))
        if tgt and tgt != cid:
            return _mk(mint, tgt, "normalized", eff_org, hf_to_dev,
                       f"org={eff_org} + name {nm!r} == {tgt!r}")
    # fuzzy: brand-prefix-stripped name + org agreement
    brands = brand_tokens_for(eff_org, hf_to_dev)
    stripped: set[str] = set()
    for nm in cand_names:
        stripped |= strip_brand_prefix(nm, brands)
    for nm in stripped - cand_names:
        tgt = by_org_name.get((eff_org, nm))
        if tgt and tgt != cid:
            return _mk(mint, tgt, "fuzzy", eff_org, hf_to_dev,
                       f"org={eff_org} + brand-stripped {nm!r} == {tgt!r}")
    return None


def _mk(mint, hf_target, match_type, mint_dev_org, hf_to_dev, evidence) -> dict:
    hf_org = (hf_to_dev.get(hf_target.split("/", 1)[0].lower(), hf_target.split("/", 1)[0])
              if "/" in hf_target else hf_target)
    return {
        "mint_id": mint["id"],
        "hf_target": hf_target,
        "match_type": match_type,
        "mint_org": mint_dev_org or "",
        "hf_org": hf_org,
        "org_agreement": (mint_dev_org or "").lower() == (hf_org or "").lower(),
        "evidence": evidence,
    }
