#!/usr/bin/env python3
"""
Generate seed/models/sources/models_dev.generated.yaml from models.dev.

This script writes ONE data source — pure models.dev output, no curated
overlays. The seed CLI loader applies `seed/models/core.yaml` (curated
canonicals) and `seed/models/enrichments/aliases.yaml` (optional alias
additions) at load time.

models.dev is strong on hosted-API model catalogs (Anthropic, OpenAI, xAI,
Google Gemini), weaker on open-weight families released directly to
HuggingFace (Meta Llama, Mistral / Mixtral, Qwen open weights, Gemma, Phi,
Yi, OLMo, Falcon, Granite, etc.). Curated entries in `core.yaml` cover
those and win at load time on id collision.

The right policy: prioritize correct expected coverage of what EEE actually
contains over the bounds of any single upstream catalog. When a refresh PR
introduces an unexpected drop or a too-coarse family, prefer adding/keeping
a `core.yaml` entry over chasing the upstream catalog.

This script fetches https://models.dev/api.json, filters to known
model-author providers (labs that release their own models, not re-hosting
inference providers), collapses models to family granularity, and writes
the generated YAML. Curated entries in core.yaml are NOT merged here — the
output is pure data-source.

Usage:
    python scripts/refresh_from_modelsdev.py              # fetch + write
    python scripts/refresh_from_modelsdev.py --no-fetch   # use /tmp cache
    python scripts/refresh_from_modelsdev.py --dry-run    # diff vs current

Re-running this is safe: it overwrites the generated YAML. The seed CLI
(`uv run eval-card-registry seed --local`) is idempotent over the result.

Source: https://models.dev (MIT, (c) 2025 models.dev)
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml

# Resolver lives in the workspace package; this script runs from the repo
# root via `uv run`, so the import resolves through pyproject's path dep.
from eval_entity_resolver.display import humanize_model_slug

# Strip trailing date suffixes and `-latest` to collapse a model down to its
# major-version family slug. Mirrors the resolver's fuzzy stem so per-snapshot
# entries from models.dev (e.g. `gpt-4o-2024-05-13`, `claude-opus-4-5-20251101`)
# fold into a single canonical with the dated snapshots as aliases.
_FAMILY_DATE_RES = [
    re.compile(r"-\d{8}$"),                  # YYYYMMDD: -20251101
    re.compile(r"-\d{4}-\d{2}-\d{2}$"),      # YYYY-MM-DD: -2024-05-13
    re.compile(r"-preview-\d{2}-\d{2}$"),    # -preview-05-06 (Google Gemini preview snapshots)
    re.compile(r"-preview-\d{4}-\d{2}-\d{2}$"),  # -preview-2024-05-13 (rare)
    re.compile(r"-preview$"),                # bare -preview
    re.compile(r"-latest$"),                 # -latest hosting tag
    re.compile(r"-v\d+(\.\d+)*$", re.IGNORECASE),  # version suffix: -v0.3, -v1, -v1.0.0
]
# Legacy alias names kept for any external callers (tests etc.)
_FAMILY_LATEST_RE = _FAMILY_DATE_RES[-1]
_FAMILY_PREVIEW_RE = _FAMILY_DATE_RES[-2]

# Training-stage suffixes — matches the resolver's _STRIP_SUFFIXES. Stripped
# from canonical so base / instruct / chat / it variants share one entry.
_FAMILY_STAGE_SUFFIXES = ("-instruct", "-chat", "-it", "-base")

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED_PATH = REPO_ROOT / "seed" / "models" / "sources" / "models_dev.generated.yaml"
ORGS_SEED_PATH = REPO_ROOT / "seed" / "orgs.yaml"
CACHE_PATH = Path("/tmp/modelsdev_api.json")
SOURCE_URL = "https://models.dev/api.json"

# Map models.dev provider slug -> our canonical_orgs.id.
# Most match by name; a few need translation. Providers not listed here are
# skipped (most are inference re-hosts, gateways, regional duplicates).
PROVIDER_TO_ORG: dict[str, str] = {
    "anthropic": "anthropic",
    "openai": "openai",
    "google": "google",
    "xai": "xai",
    "cohere": "cohere",
    "mistral": "mistralai",
    "deepseek": "deepseek",
    "alibaba": "alibaba",
    "moonshotai": "moonshotai",
    "stepfun": "stepfun",
    "minimax": "minimax",
    "zai": "zai",
    "inception": "inception",
    "upstage": "upstage",
    "perplexity": "perplexity",
    "nvidia": "nvidia",
    # Add more here as we extend the allowlist; each must have a matching
    # entry in seed/orgs.yaml or the validator will fail.
}

# --- inference_platforms single-source --------------------------------------
# Path to the curated 137-platform catalog (source of truth). The same file
# seeds seed/inference_platforms.yaml; we read its `models_dev_provider` field
# to build PROVIDER_TO_INFERENCE_PLATFORM so the host-token→platform map stays
# byte-identical between seed generation here and runtime capture in fuzzy.py
# (single-source mandate — DO NOT hand-maintain a parallel dict).
INFERENCE_PLATFORMS_JSON = (
    REPO_ROOT  / "curation" / "inference_platforms.proposed.json"
)


def _load_provider_to_inference_platform(
    path: Path = INFERENCE_PLATFORMS_JSON,
) -> dict[str, str]:
    """Build {models.dev provider slug -> inference_platforms.id} from the
    curated catalog. Every one of the 137 platforms declares exactly one
    `models_dev_provider`; this maps all of them (including the ~122 the old
    PROVIDER_TO_ORG gate discarded)."""
    data = json.loads(path.read_text())
    mapping: dict[str, str] = {}
    for plat in data.get("platforms", []):
        prov = plat.get("models_dev_provider")
        pid = plat.get("id")
        if not prov or not pid:
            continue
        mapping[prov] = pid
    return mapping


# Module-level singleton. Falls back to an empty dict if the spec file is not
# present (e.g. after the spec dir is removed post-integration — at that point
# the map should be sourced from seed/inference_platforms.yaml instead).
try:
    PROVIDER_TO_INFERENCE_PLATFORM: dict[str, str] = _load_provider_to_inference_platform()
except FileNotFoundError:  # pragma: no cover - integration-time fallback
    PROVIDER_TO_INFERENCE_PLATFORM = {}


# --- Author-lab classification + org inference -----------------------------

# Author-lab provider slugs, sourced from the SAME curated catalog (kind ==
# "author_lab"). These are the providers whose spelling can anchor a group's
# authorship (but only when their org matches the family-implied org — a lab
# can re-host others' models too).
def _load_strict_author(path: Path = INFERENCE_PLATFORMS_JSON) -> set[str]:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:  # pragma: no cover
        return set()
    return {
        p["models_dev_provider"]
        for p in data.get("platforms", [])
        if p.get("kind") == "author_lab" and p.get("models_dev_provider")
    }


STRICT_AUTHOR: set[str] = _load_strict_author()

# Author-lab provider id -> HF-style org slug (used when that provider's
# spelling anchors a group).
AUTHOR_PROV_ORG: dict[str, str | None] = {
    "anthropic": "anthropic", "openai": "openai", "google": "google",
    "mistral": "mistralai", "cohere": "cohere", "zai": "zai-org",
    "zhipuai": "zai-org", "alibaba": "qwen", "deepseek": "deepseek-ai",
    "llama": "meta-llama", "minimax": "minimaxai", "moonshotai": "moonshotai",
    "nvidia": "nvidia", "xai": "xai", "xiaomi": "xiaomi",
    "stepfun": "stepfun-ai", "stepfun-ai": "stepfun-ai", "upstage": "upstage",
    "venice": None, "perplexity": "perplexity", "perplexity-agent": "perplexity",
    "nova": "amazon", "sarvam": "sarvam-ai", "inception": "inceptionai",
    "poolside": "poolside", "morph": "morph", "v0": "vercel",
    "lucidquery": None, "inceptron": None,
}

# org inference from family / name tokens (HF-style slugs).
ORG_BY_FAMILY_PREFIX: dict[str, str] = {
    "claude": "anthropic", "gpt": "openai", "o-": "openai", "o": "openai", "gpt-": "openai",
    "gemini": "google", "gemma": "google", "imagen": "google", "learnlm": "google",
    "qwen": "qwen", "qwen3": "qwen", "qwen3.": "qwen",
    "llama": "meta-llama",
    "glm": "zai-org", "glm-": "zai-org",
    "deepseek": "deepseek-ai",
    "minimax": "minimaxai", "mimo": "xiaomi",
    "kimi": "moonshotai",
    "grok": "xai",
    "mistral": "mistralai", "ministral": "mistralai", "devstral": "mistralai",
    "codestral": "mistralai", "mistral-": "mistralai", "mixtral": "mistralai",
    "phi": "microsoft",
    "nemotron": "nvidia", "nova": "amazon", "command": "cohere",
    "command-r": "cohere", "command-a": "cohere",
    "ernie": "baidu", "hunyuan": "tencent", "seed": "bytedance", "doubao": "bytedance",
    "flux": "black-forest-labs", "voyage": "voyageai", "ling": "inclusionai",
    "gpt-oss": "openai",
}


def org_from_family(fam: str | None) -> str | None:
    """Infer an HF-style org slug from a models.dev `family` token."""
    if not fam:
        return None
    f = fam.lower()
    base = f.split("-")[0]
    for key in (f, base):
        if key in ORG_BY_FAMILY_PREFIX:
            return ORG_BY_FAMILY_PREFIX[key]
    for pref, org in ORG_BY_FAMILY_PREFIX.items():
        if f.startswith(pref):
            return org
    return None


# Map HF-style org slugs onto the registry's CURATED developer-org slugs
# (org identity & casing model: curated dev org ids keep their authored slug;
# HF namespaces are recorded as aliases and RESOLVE to the dev org). Built once
# from seed/orgs.yaml alias index, with a few explicit overrides for HF slugs
# that aren't already aliased.
_ORG_SLUG_OVERRIDES: dict[str, str] = {
    "meta-llama": "meta",
    "qwen": "alibaba",
    "deepseek-ai": "deepseek",
    "zai-org": "zai",
}


def _build_org_alias_index() -> dict[str, str]:
    """{lowercased org id / hf_org / alias -> curated org id} from seed/orgs.yaml.

    Includes `hf_org` so a model id's HF namespace prefix (`Qwen/`→alibaba,
    `meta-llama/`→meta) is recognised as a KNOWN curated org straight from the
    catalog."""
    if not ORGS_SEED_PATH.exists():
        return {}
    data = yaml.safe_load(ORGS_SEED_PATH.read_text()) or []
    out: dict[str, str] = {}
    for e in data:
        oid = e.get("id")
        if not oid:
            continue
        out[oid.lower()] = oid
        if e.get("hf_org"):
            out.setdefault(str(e["hf_org"]).lower(), oid)
        for a in (e.get("aliases") or []):
            out.setdefault(str(a).lower(), oid)
    return out


_DEV_ALIAS_INDEX: dict[str, str] | None = None


def _dev_alias_index() -> dict[str, str]:
    """Module-cached org alias index (orgs.yaml parsed once, not per group)."""
    global _DEV_ALIAS_INDEX
    if _DEV_ALIAS_INDEX is None:
        _DEV_ALIAS_INDEX = _build_org_alias_index()
    return _DEV_ALIAS_INDEX


def normalize_org_slug(hf_org: str | None, alias_index: dict[str, str]) -> str | None:
    """Map an HF-style org slug to the registry's curated org id when one
    exists; else return the HF slug unchanged (a new HF-derived org row will be
    auto-created with HF casing). Returns None for None in."""
    if not hf_org:
        return None
    if hf_org in _ORG_SLUG_OVERRIDES:
        return _ORG_SLUG_OVERRIDES[hf_org]
    mapped = alias_index.get(hf_org.lower())
    return mapped if mapped else hf_org


# --- Developer (org) derivation: PREFIX-authoritative ----------------------
# The model id's org prefix is the developer. The model NAME only says what a
# model is DERIVED FROM (base lineage), not who MADE it, so name-matching is
# used ONLY for bare ids / serving-hosted ids (no genuine prefix), NEVER to
# override a prefix (`3rd-Degree-Burn/Llama-...` is by 3rd-Degree-Burn, NOT
# meta; `nvidia/llama-nemotron` is nvidia's, NOT meta). Serving hosts are
# stripped; a curated-prefix model whose name disagrees (a possible re-host,
# e.g. nvidia/whisper) is FLAGGED for curation, not auto-flipped.
# Serving / gateway platforms (where a model is SERVED, not DEVELOPED) — a model
# id prefixed with one of these is stripped and the developer taken from the
# name. Single-sourced from the curated inference_platforms catalog: every
# models.dev provider that is NOT an author_lab (those re-host others' models —
# fireworks, together, volcengine, fal-ai, openrouter, ...), plus host
# scaffolding tokens. (nvidia IS an author_lab, so it stays out and its genuine
# re-hosts are handled by explicit curation, not stripped here.)
_SERVING_HOSTS = {
    "accounts", "clarifai", "route", "orcarouter", "workers-ai", "openrouter",
    "stealth", "hf", "cf", "@cf",
    # serving-brand id namespaces (not top-level providers, but appear as id
    # prefixes): ByteDance's Volcano Engine cloud, fal's image-serving, etc.
    "volcengine", "fal-ai", "fal", "kilo", "kilo-auto",
} | ({p.lower() for p in PROVIDER_TO_INFERENCE_PLATFORM} - {a.lower() for a in STRICT_AUTHOR})
# model NAME starts with TOKEN -> HF-style developer slug (normalize_org_slug
# maps to the curated org). Longest-prefix-first.
_NAME_VENDOR_MAP: list[tuple[str, str]] = sorted([
    ("meta-llama", "meta-llama"), ("codellama", "meta-llama"), ("llama", "meta-llama"),
    ("ministral", "mistralai"), ("mixtral", "mistralai"), ("pixtral", "mistralai"),
    ("codestral", "mistralai"), ("devstral", "mistralai"), ("magistral", "mistralai"),
    ("mistral", "mistralai"),
    ("qwen", "qwen"), ("qwq", "qwen"), ("qvq", "qwen"),
    ("gpt-oss", "openai"), ("gpt", "openai"), ("o1", "openai"), ("o3", "openai"),
    ("o4", "openai"), ("whisper", "openai"), ("chatgpt", "openai"), ("codex", "openai"),
    ("dall-e", "openai"), ("text-embedding", "openai"),
    ("claude", "anthropic"),
    ("gemini", "google"), ("paligemma", "google"), ("gemma", "google"),
    ("imagen", "google"), ("learnlm", "google"),
    ("deepseek", "deepseek-ai"), ("grok", "xai"), ("chatglm", "zai-org"), ("glm", "zai-org"),
    ("kimi", "moonshotai"), ("moonshot", "moonshotai"), ("minimax", "minimax"),
    ("phi", "microsoft"), ("nemotron", "nvidia"), ("nvlm", "nvidia"), ("mimo", "xiaomi"),
    ("hunyuan", "tencent"), ("ernie", "baidu"), ("doubao", "bytedance"), ("seed", "bytedance"),
    ("command", "cohere"), ("aya", "cohere"), ("nova", "amazon"), ("titan", "amazon"),
    ("solar", "upstage"), ("jamba", "ai21"), ("jurassic", "ai21"), ("sonar", "perplexity"),
    ("hermes", "nousresearch"), ("granite", "ibm"), ("flux", "black-forest-labs"),
    ("voyage", "voyageai"), ("cogito", "deepcogito"), ("falcon", "tiiuae"),
    ("olmo", "allenai"), ("tulu", "allenai"), ("bge", "baai"), ("inflection", "inflection"),
], key=lambda kv: -len(kv[0]))


def developer_from_name(name: str | None) -> str | None:
    """HF-style developer slug from a LEADING vendor token in the model name.
    Leading-token only, so a derivative's base token mid-name can't hijack it.
    For BARE / serving-hosted ids only — never to override a real prefix."""
    if not name:
        return None
    s = re.sub(r"^[a-z]+\.", "", str(name).strip().lower())
    s = s.split("/")[-1]
    s = re.sub(r"[_\s]+", "-", s)
    for tok, dev in _NAME_VENDOR_MAP:
        if re.match(re.escape(tok) + r"([0-9._:\-]|$)", s):
            return dev
    return None


def _derive_group_org(recs: list[dict], alias_index: dict[str, str]):
    """Developer org for a models.dev underlying group (prefix-authoritative).

    Returns (hf_org_slug | None, rehost_review | None). `rehost_review` is the
    disagreeing name-vendor when a CURATED-org prefix's model name points to a
    different vendor (a possible re-host to curate, e.g. nvidia/whisper)."""
    prefix_orgs: list[str] = []
    name_orgs: list[str] = []
    for r in recs:
        raw = (r.get("raw") or "").lstrip("~")
        if "/" in raw and raw.split("/")[0].lower() not in _SERVING_HOSTS:
            prefix_orgs.append(raw.split("/")[0])          # uploader OR curated (raw spelling)
        else:                                              # bare or serving-hosted
            no = developer_from_name(r.get("name"))
            if no:
                name_orgs.append(no)
    if prefix_orgs:
        # Take the prefix VERBATIM (most common spelling present). We do NOT
        # reconcile spelling variants (e.g. 'TheDrummer 2' vs 'thedrummer') — we
        # have no authoritative basis to assert they're the same uploader, so
        # picking/cleaning one would be an arbitrary, unverified identity claim.
        # Curated orgs.yaml is the place to assert such equivalences explicitly.
        low = Counter(p.lower() for p in prefix_orgs).most_common(1)[0][0]
        org = next(p for p in prefix_orgs if p.lower() == low)
        rehost = None
        if alias_index.get(org.lower()) and name_orgs:    # curated prefix + name signal
            nd = Counter(name_orgs).most_common(1)[0][0]
            if normalize_org_slug(nd, alias_index) != normalize_org_slug(org, alias_index):
                rehost = nd
        return org, rehost
    if name_orgs:
        return Counter(name_orgs).most_common(1)[0][0], None
    return None, None


# --- Dedup key + head-pick --------------------------------------------------
_DATE8_RE = re.compile(r"^\d{8}$")
_DATE6_RE = re.compile(r"^\d{6}$")
_NUM_TOKEN_RE = re.compile(r"^\d+[a-z]?$")


def normalize_modelsdev_id(raw: str) -> str:
    """Normalize a models.dev model id to an underlying-model spelling — strips
    provider/host/region scaffolding, drops serving variants, unifies
    separators."""
    s = raw.strip()
    s = s.lstrip("~")  # openrouter '~' latest marker
    s = re.sub(r"^accounts/[^/]+/models/", "", s)
    s = re.sub(r"^hf:", "", s)
    s = re.sub(r"^@cf/", "", s)
    s = re.sub(r"^clarifai/[^/]+/models/", "", s)
    s = re.sub(r"^route/", "", s)
    s = re.sub(r"^orcarouter/", "", s)
    s = s.replace("--", "/")  # sap style
    s = re.sub(r"^databricks-", "", s)
    s = re.sub(r"^azure-", "", s)
    s = re.sub(r"^aws-", "", s)
    s = re.sub(r"^openai-", "", s)
    s = re.sub(r"^anthropic-", "", s)
    s = re.sub(r"^ai21-", "ai21/", s)
    s = re.sub(r"^stealth/", "", s)
    s = _strip_host_region_prefixes(s)
    if "/" in s:
        s = s.split("/")[-1]
    s = _strip_host_region_prefixes(s)
    s = s.lower()
    s = s.replace("@default", "")
    s = re.sub(r"@(\d{8})$", r"-\1", s)
    s = re.sub(r"@.*$", "", s)
    s = _BEDROCK_VER_RE.sub("", s)
    prev = None
    while prev != s:
        prev = s
        s = _VARIANT_SUFFIX_RE.sub("", s)
    s = re.sub(r"(\d)\.(\d)", r"\1-\2", s)
    s = re.sub(r"[_\s]+", "-", s)
    s = re.sub(r"\(.*$", "", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


_HOST_PREFIXES = ["amazon.", "anthropic.", "qwen.", "meta.", "google.", "cohere.",
                  "ai21.", "deepseek.", "mistral."]
_REGION_PREFIXES = ["us.", "eu.", "jp.", "au.", "apac.", "global."]
_VARIANT_SUFFIX_RE = re.compile(
    r"(:.*$)"
    r"|(-thinking$)|(-think$)|(-reasoner$)|(-reasoning$)"
    r"|(-turbo$)"
    r"|(-tee$)|(-fp8$)|(-bf16$)|(-int8$)|(-awq$)|(-gptq$)"
    r"|(-fast$)|(-precision$)|(-free$)"
)
_BEDROCK_VER_RE = re.compile(r"-v\d+:\d+$")


def _strip_host_region_prefixes(s: str) -> str:
    changed = True
    while changed:
        changed = False
        for rp in _REGION_PREFIXES:
            if s.startswith(rp):
                s = s[len(rp):]
                changed = True
        for hp in _HOST_PREFIXES:
            if s.startswith(hp):
                s = s[len(hp):]
                changed = True
    return s


def canon_key_ordered(norm: str) -> str:
    """Underlying-model key (order-preserving)."""
    s = re.sub(r"-(latest|old|new)$", "", norm)
    s = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", s)
    s = re.sub(r"-\d{2}-\d{2}$", "", s)
    toks = [t for t in s.split("-") if not _DATE8_RE.match(t) and not _DATE6_RE.match(t)]
    if toks and re.match(r"^v\d+$", toks[-1]):
        toks.pop()
    return "-".join(toks)


def safe_sig(key: str) -> str:
    """Token-multiset signature for the merge pass. Words sorted, version
    numbers order-preserved (so `gpt-4-5` != `gpt-5-4`)."""
    toks = key.split("-")
    words = sorted(t for t in toks if not _NUM_TOKEN_RE.match(t))
    nums = [t for t in toks if _NUM_TOKEN_RE.match(t)]
    return "|".join(words) + "#" + "-".join(nums)


def build_underlying_groups(api_json: dict) -> dict[str, list[dict]]:
    """Group all (provider, model) records across the catalog into underlying
    groups keyed by a canonical root via a union-find merge pass. Returns
    {root_key -> [records]} where each record is
    {provider, raw, norm, key, family, release, name, open_weights}."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for prov, pdata in api_json.items():
        for mid, mr in (pdata.get("models") or {}).items():
            n = normalize_modelsdev_id(mid)
            k = canon_key_ordered(n)
            if not k:
                continue
            groups[k].append(dict(
                provider=prov, raw=mid, norm=n, key=k,
                family=mr.get("family"), release=mr.get("release_date"),
                name=mr.get("name"), open_weights=mr.get("open_weights"),
                record=mr,
            ))

    # Union-find: merge ordered keys sharing the same token-multiset signature,
    # but only when their families don't CONFLICT (guards anagram merges).
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    for k in groups:
        find(k)
    mset: dict[str, list[str]] = defaultdict(list)
    for k in groups:
        mset[safe_sig(k)].append(k)
    for _ms, ks in mset.items():
        if len(ks) > 1:
            base = ks[0]
            for o in ks[1:]:
                fam_a = {r["family"] for r in groups[base] if r["family"]}
                fam_b = {r["family"] for r in groups[o] if r["family"]}
                if fam_a & fam_b or not fam_a or not fam_b:
                    union(o, base)

    merged: dict[str, list[dict]] = defaultdict(list)
    for k, recs in groups.items():
        merged[find(k)].extend(recs)
    return merged


def pick_underlying(root: str, recs: list[dict]) -> dict:
    """Choose the authoritative (org, display_name, release, open_weights) and
    head spelling for an underlying group. Returns a dict with a
    `head_spelling` (the cleanest spelling to mint from).

    Org returned is HF-style; callers normalize via normalize_org_slug() to the
    curated dev org."""
    fams = [r["family"] for r in recs if r["family"]]
    fam_org = org_from_family(Counter(fams).most_common(1)[0][0]) if fams else None
    author_recs = [r for r in recs if r["provider"] in STRICT_AUTHOR]
    true_author_recs = [
        r for r in author_recs
        if fam_org is None or AUTHOR_PROV_ORG.get(r["provider"]) == fam_org
    ]
    # Developer org: PREFIX-authoritative (the id's namespace is the developer);
    # name only for bare/serving ids; never name-override a prefix. Re-host
    # disagreements (curated prefix vs name) are flagged for curation, not flipped.
    org, rehost_review = _derive_group_org(recs, _dev_alias_index())
    has_author = bool(true_author_recs)  # gates family-tree vs single mint downstream

    if true_author_recs:
        disp = true_author_recs[0]["name"] or root
        head_spelling = true_author_recs[0]["raw"]
    else:
        names = [r["name"] for r in recs if r["name"]]
        disp = Counter(names).most_common(1)[0][0] if names else root
        # Head spelling for a re-host-only group: the cleanest normalised
        # spelling = the one whose normalized form is shortest / equals root.
        head_spelling = _cleanest_spelling(root, recs)

    rels = [r["release"] for r in recs if r["release"]]
    release = min(rels) if rels else None
    ow = any(r["open_weights"] for r in recs)
    return {
        "author_org": org,
        "display_name": disp,
        "release_date": release,
        "open_weights": ow,
        "has_author_lab_entry": has_author,
        "head_spelling": head_spelling,
        "rehost_review": rehost_review,
    }


def _cleanest_spelling(root: str, recs: list[dict]) -> str:
    """Pick the cleanest raw spelling to mint a canonical from when no author
    lab anchors the group: prefer a record whose normalized form equals the
    canon root, else the shortest normalized form; tie-break alphabetically by
    raw for determinism."""
    exact = [r for r in recs if r["norm"] == root]
    pool = exact or recs
    return sorted(pool, key=lambda r: (len(r["norm"]), r["raw"]))[0]["raw"]


def _fetch(use_cache: bool) -> dict:
    if use_cache and CACHE_PATH.exists():
        print(f"[refresh] using cache: {CACHE_PATH}", file=sys.stderr)
        return json.loads(CACHE_PATH.read_text())
    print(f"[refresh] fetching {SOURCE_URL}", file=sys.stderr)
    # Send an identifiable User-Agent — models.dev's CDN rejects the default
    # Python-urllib UA with a 403, and a generic UA is a good citizen anyway
    # since it lets the data source see who's hitting the API.
    req = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "evalcard-registry-refresh/1.0 (+https://github.com/evaleval/evalcard-registry)"},
    )
    with urllib.request.urlopen(req) as r:
        raw = r.read().decode()
    CACHE_PATH.write_text(raw)
    return json.loads(raw)


def _slugify(value: str) -> str:
    """Lowercase + collapse whitespace/underscores to single hyphens.
    Preserves dots (for version readability), hyphens, slashes. Drops
    parens, brackets, and other display-only punctuation that would
    otherwise leak into canonical ids."""
    s = value.strip().lower()
    # Drop punctuation that's display-only (e.g. "Claude 4.5 (latest)")
    s = re.sub(r"[()\[\]{}]", "", s)
    s = re.sub(r"[\s_]+", "-", s)   # spaces/underscores → hyphen
    s = re.sub(r"-+", "-", s)        # collapse multiple hyphens
    return s.strip("-")


def _family_for(model: dict) -> str:
    """Group key for a model record (= the lineage canonical slug).

    Prefer the `name` field over `id` because models.dev's `id` slugs
    sometimes mangle separators (e.g. Alibaba's `qwen2-5-14b-instruct` for
    what HF calls `Qwen2.5-14B-Instruct`). The `name` field carries the
    lab's spelling with dots intact.

    Strip date suffixes (snapshot collapse), `-latest`/`-preview` markers,
    and training-stage suffixes (`-instruct`, `-chat`, `-it`, `-base`) so
    base + instruct + chat variants share one canonical. This mirrors the
    resolver's fuzzy stem; doing it here means the seed is consistent with
    the resolution rule.

    We do NOT use models.dev's `family` field — it's too coarse (groups all
    Claude Opus major versions 4.0/4.5/4.7 under one slug). The pipeline's
    `family_slug` works at "Opus 4.5" granularity.
    """
    raw = _slugify(model.get("name") or model.get("id", ""))
    # Loop until no suffix matches — date / preview / latest / training-stage
    # patterns can stack (e.g. "-preview-05-06-instruct").
    for _ in range(5):
        before = raw
        for pat in _FAMILY_DATE_RES:
            raw = pat.sub("", raw)
        for suffix in sorted(_FAMILY_STAGE_SUFFIXES, key=len, reverse=True):
            if raw.endswith(suffix):
                raw = raw[: -len(suffix)]
                break
        if raw == before:
            break
    return raw


# --- Suffix → (relationship, axis) classification --------------------------
# Used to translate the diff between a snapshot's id and its family slug
# into a typed `parents` edge. The classification table uses the same enum
# values as curated core.yaml entries so models.dev refresh and curated
# entries land in identical shapes.
#
# Axis semantics (closed enum):
#   version  — dated snapshot / vN marker: same API identity, different release.
#   training_stage — base / instruct / chat / it: a post-training stage of the
#                    same pretrained model (was previously folded under `mode`).
#   tier     — branded sibling (haiku/sonnet/opus, mini/nano, flash/pro,
#              small/medium/large): a DIFFERENT product in the same family that
#              makes NO disclosed-scale claim. Never emit a `size` edge for these.
#   size     — a genuinely-disclosed scale: an open-weight name size token
#              (7b/70b/405b/8x7b) or a MoE active-param token (a16b). See
#              `_is_size_token`. NEVER assert for a branded tier.
#   mode     — runtime/decoding mode that is not a training stage (thinking,
#              reasoning, guard, …).
#   modality / domain — unchanged.
_TOKEN_CLASSIFICATIONS: dict[str, tuple[str, str | None]] = {
    # Training-stage post-training axis (was `mode`).
    "instruct": ("variant", "training_stage"),
    "it": ("variant", "training_stage"),
    "chat": ("variant", "training_stage"),
    "base": ("variant", "training_stage"),
    "pt": ("variant", "training_stage"),
    "sft": ("variant", "training_stage"),
    # Branded tiers — sibling products, NO scale claim (axis=tier).
    "haiku": ("variant", "tier"),
    "sonnet": ("variant", "tier"),
    "opus": ("variant", "tier"),
    "mini": ("variant", "tier"),
    "nano": ("variant", "tier"),
    "micro": ("variant", "tier"),
    "flash": ("variant", "tier"),
    "pro": ("variant", "tier"),
    "small": ("variant", "tier"),
    "medium": ("variant", "tier"),
    "large": ("variant", "tier"),
    "lite": ("variant", "tier"),
    # Runtime / decoding modes (NOT training stages).
    "thinking": ("variant", "mode"),
    "reasoning": ("variant", "mode"),
    "nothink": ("variant", "mode"),
    "guard": ("variant", "mode"),
    "safeguard": ("variant", "mode"),
    "moderation": ("variant", "mode"),
    # Modality / domain.
    "vision": ("variant", "modality"),
    "vl": ("variant", "modality"),
    "coder": ("variant", "domain"),
    "code": ("variant", "domain"),
    "math": ("variant", "domain"),
    # Precision / serving quantization.
    "turbo": ("quantized", None),
    "fp8": ("quantized", None),
    "fp16": ("quantized", None),
    "bf16": ("quantized", None),
    "int4": ("quantized", None),
    "int8": ("quantized", None),
    "awq": ("quantized", None),
    "gptq": ("quantized", None),
    "gguf": ("quantized", None),
}
# The set of branded-tier tokens — used to GUARD against ever emitting a size
# edge for a tier (branded tiers never carry a scale claim).
_TIER_TOKENS = frozenset(
    k for k, (_rel, axis) in _TOKEN_CLASSIFICATIONS.items() if axis == "tier"
)
_VERSION_RE = re.compile(r"^v\d+(\.\d+)*$", re.IGNORECASE)
_DATE_8_RE = re.compile(r"^\d{8}$")
_DATE_4_RE = re.compile(r"^\d{4}$")
_DATE_3_RE = re.compile(r"^\d{3}$")
_MOE_ACTIVE_RE = re.compile(r"^a\d+b$", re.IGNORECASE)
# Disclosed open-weight scale tokens: 7b, 70b, 405b, 1.5b (dot already slugged
# to dash so we also accept a leading number-dash-number — handled in the
# size-aware classifier), and MoE expert tokens like 8x7b / 8x22b.
_SIZE_TOKEN_RE = re.compile(r"^\d+(\.\d+)?b$", re.IGNORECASE)
_MOE_EXPERT_RE = re.compile(r"^\d+x\d+(\.\d+)?b$", re.IGNORECASE)


def _is_size_token(token: str) -> bool:
    """True iff `token` is a GENUINELY-DISCLOSED scale token from an open-weight
    family name: a bare param count (`7b`, `70b`, `405b`, `1.5b`), a MoE
    expert spec (`8x7b`, `8x22b`), or a MoE active-param spec (`a16b`).

    This is the ONLY route to a `size` edge from models.dev (which carries no
    params field). A branded tier token is NEVER a size token (guarded by
    classification order in `_classify_token`)."""
    t = token.lower()
    return bool(_SIZE_TOKEN_RE.match(t) or _MOE_EXPERT_RE.match(t) or _MOE_ACTIVE_RE.match(t))


def _classify_token(token: str) -> tuple[str, str | None] | None:
    t = token.lower()
    # Branded tiers and named tokens take priority over the size regex so a
    # tier is never mis-read as a scale claim.
    if t in _TOKEN_CLASSIFICATIONS:
        return _TOKEN_CLASSIFICATIONS[t]
    if _VERSION_RE.match(t) or _DATE_8_RE.match(t) or _DATE_4_RE.match(t) or _DATE_3_RE.match(t):
        return ("variant", "version")
    # Disclosed scale: open-weight size token / MoE spec. Tier tokens already
    # returned above, so a `size` axis here always reflects a real scale.
    if _is_size_token(t):
        return ("variant", "size")
    return None


def _classify_suffix_segments(suffix: str) -> list[tuple[str, str | None, str]]:
    """Greedy left-to-right parse of a hyphen/dot-separated suffix.

    Returns a list of (relationship, axis, token) segments. When a single
    token can't be classified, falls back to a single (variant, version, suffix)
    segment so we always emit at least one parent edge — consumer can refine
    via curated core.yaml entries that override on collision.
    """
    if not suffix:
        return []
    # Normalize separators within the suffix for token splitting (mirrors
    # what _slugify does to ids — but apply also to dot for v0.1 → v0-1).
    norm = re.sub(r"[._]+", "-", suffix.lower()).strip("-")
    if not norm:
        return []
    tokens = norm.split("-")
    segments: list[tuple[str, str | None, str]] = []
    i = 0
    while i < len(tokens):
        # YYYY-MM-DD across 3 tokens
        if i + 3 <= len(tokens):
            window = "-".join(tokens[i:i + 3])
            if re.match(r"^\d{4}-\d{2}-\d{2}$", window):
                segments.append(("variant", "version", window))
                i += 3
                continue
        # vN-N across 2 tokens (slugified v0.3 → v0-3)
        if i + 2 <= len(tokens):
            window = "-".join(tokens[i:i + 2])
            if re.match(r"^v\d+-\d+$", window) or re.match(r"^\d{4}-\d{2}$", window):
                segments.append(("variant", "version", window))
                i += 2
                continue
        cls = _classify_token(tokens[i])
        if cls is None:
            # Unknown token mid-suffix — bail out and emit the rest as a
            # single version segment so at least the outer parent edge is set.
            tail = "-".join(tokens[i:])
            segments.append(("variant", "version", tail))
            return segments
        relationship, axis = cls
        segments.append((relationship, axis, tokens[i]))
        i += 1
    return segments


def _build_family_entries(org_id: str, family_slug: str, models: list[dict]) -> list[dict]:
    """Emit canonical entries for a family.

    Returns a list:
      [0]   family root canonical (parents=[])
      [1..] one child per snapshot/variant whose slugified id != family_slug,
            each with a typed `parents` edge. Compound suffixes (e.g.
            `mistral-7b-instruct-v0-3`) materialize their intermediate
            canonicals so models.dev output matches the post-promotion
            shape of core.yaml — this matters because the seed loader's
            parents-merge is union-by-id, so disagreement on the parent
            id between source and core would produce a spurious second edge.
    """
    family_canonical_id = f"{org_id}/{family_slug}"

    # ---- Family root display_name + aggregated metadata ----
    # Prefer the lab's preferred name (vendor casing like `GPT-4o`); fall
    # back to our humanizer when models.dev didn't supply a name.
    display_name = ""
    for m in models:
        if _slugify(m.get("id", "")) == family_slug:
            display_name = m.get("name") or humanize_model_slug(family_slug)
            break
    if not display_name:
        display_name = humanize_model_slug(family_slug)

    open_weights = any(m.get("open_weights") for m in models)
    release_dates = sorted({m["release_date"] for m in models if m.get("release_date")})
    release_date = release_dates[0] if release_dates else None
    snapshot_ids = sorted({_slugify(m["id"]) for m in models if m.get("id")})

    # Aggregate modalities across all snapshots in the family (union).
    # Per-snapshot modalities still flow through to leaf children below;
    # the family root surfaces the superset so any snapshot's modality is
    # represented at the parent identity.
    family_input_modalities: set[str] = set()
    family_output_modalities: set[str] = set()
    for m in models:
        mods = m.get("modalities") or {}
        for v in (mods.get("input") or []):
            if isinstance(v, str) and v.strip():
                family_input_modalities.add(v.strip())
        for v in (mods.get("output") or []):
            if isinstance(v, str) and v.strip():
                family_output_modalities.add(v.strip())
    family_input_modalities_list = sorted(family_input_modalities) or None
    family_output_modalities_list = sorted(family_output_modalities) or None

    metadata: dict = {"snapshots": snapshot_ids}
    if release_dates:
        metadata["release_dates"] = release_dates
    if any(m.get("knowledge") for m in models):
        metadata["knowledge_cutoffs"] = sorted({
            m["knowledge"] for m in models if m.get("knowledge")
        })

    # Family-root aliases — surface forms of the family slug only. Snapshot
    # ids no longer go on the root; they're emitted as separate canonicals.
    root_aliases = [f"{org_id}/{family_slug}"] if f"{org_id}/{family_slug}" != family_canonical_id else []

    family_root_entry = {
        "id": family_canonical_id,
        "display_name": display_name,
        "org_id": org_id,
        "family": family_slug,
        "architecture": None,
        "params_billions": None,
        "parents": [],
        "open_weights": open_weights,
        "release_date": release_date,
        "input_modalities": family_input_modalities_list,
        "output_modalities": family_output_modalities_list,
        "tags": ["open-weight"] if open_weights else [],
        "aliases": root_aliases,
        "metadata": json.dumps(metadata, sort_keys=True),
        "review_status": "reviewed",
    }

    # ---- Child entries: one per snapshot/variant whose id != family_slug ----
    out_entries: list[dict] = [family_root_entry]
    seen_ids: dict[str, dict] = {family_canonical_id: family_root_entry}

    # `family_slug` may carry dots from the lab's display name
    # (`qwen2.5-7b`); slugified ids use dashes (`qwen2-5-7b-instruct`).
    # Compare on the dashed form, but build chain canonical ids from the
    # dotted form so children inherit the lab's preferred spelling.
    family_slug_dashed = re.sub(r"\.", "-", family_slug)

    for m in models:
        snap_dashed = _slugify(m.get("id", ""))
        if not snap_dashed:
            continue
        # If the model's id already matches the family slug (in either form),
        # it IS the family root — no child entry needed.
        if snap_dashed == family_slug_dashed or snap_dashed == family_slug:
            continue
        # Snapshot ids that don't share the family-slug prefix are unusual
        # (mirror entries, etc.) — skip rather than emit a malformed child.
        if not snap_dashed.startswith(family_slug_dashed + "-"):
            continue
        suffix = snap_dashed[len(family_slug_dashed) + 1:]
        segments = _classify_suffix_segments(suffix)
        if not segments:
            continue

        # Walk the chain, materializing intermediates as anchor entries.
        current_id = family_canonical_id
        for idx, (relationship, axis, token) in enumerate(segments):
            new_id = f"{current_id}-{token}"
            is_leaf = (idx == len(segments) - 1)
            parent_edge = {"id": current_id, "relationship": relationship}
            if axis:
                parent_edge["axis"] = axis

            if new_id in seen_ids:
                # Intermediate already emitted — walk through.
                current_id = new_id
                continue

            # Build the entry. Leaf gets the source model's metadata + the
            # original models.dev id as an alias (so dashed-form raw values
            # like `qwen2-5-7b-instruct` resolve via exact match even when
            # the canonical uses the dotted spelling). Intermediates are
            # anchor-only — humanized name, no release_date, no aliases.
            child_aliases: list[str] = []
            child_release: str | None = None
            child_open_weights = open_weights
            child_input_modalities: list[str] | None = None
            child_output_modalities: list[str] | None = None
            if is_leaf:
                child_aliases = sorted({snap_dashed, f"{org_id}/{snap_dashed}"})
                if m.get("release_date"):
                    child_release = m["release_date"]
                child_open_weights = bool(m.get("open_weights")) or open_weights
                # Per-snapshot modalities — narrower than the family aggregate.
                mods = m.get("modalities") or {}
                _ci = sorted({v.strip() for v in (mods.get("input") or []) if isinstance(v, str) and v.strip()})
                _co = sorted({v.strip() for v in (mods.get("output") or []) if isinstance(v, str) and v.strip()})
                child_input_modalities = _ci or None
                child_output_modalities = _co or None

            entry = {
                "id": new_id,
                "display_name": (m.get("name") or humanize_model_slug(new_id)) if is_leaf else humanize_model_slug(new_id),
                "org_id": org_id,
                "family": family_slug,
                "architecture": None,
                "params_billions": None,
                "parents": [parent_edge],
                "open_weights": child_open_weights,
                "release_date": child_release,
                "input_modalities": child_input_modalities,
                "output_modalities": child_output_modalities,
                "tags": ["open-weight"] if child_open_weights else [],
                "aliases": child_aliases,
                "metadata": "{}",
                "review_status": "reviewed",
            }
            seen_ids[new_id] = entry
            out_entries.append(entry)
            current_id = new_id

    return out_entries


def _provider_alias_forms(raw: str, org_id: str | None) -> list[str]:
    """Surface forms a provider's raw spelling should resolve through.

    Emits clean, resolvable forms only:
      - the raw spelling AS-IS when it carries no host/account scaffolding
        (no leading `@cf/`, no `accounts/...`, no embedded slash) — a provider's
        own bare spelling is worth an exact alias;
      - the models.dev-normalized form (host/region/account scaffolding stripped),
        which is what the resolver sees post-host-capture;
      - the org-prefixed form of the normalized slug (last path segment), so both
        bare and `org/`-prefixed spellings resolve.

    Gnarly multi-segment host-scaffolded raws (`@cf/qwen/qwen3-30b-a3b-fp8`,
    `workers-ai/@cf/...`) are NOT emitted verbatim — only their normalized form
    is, so the alias list stays clean and free of double-prefix ids."""
    forms: set[str] = set()
    if not raw:
        return []
    # Keep the raw spelling only when it's a clean single-token id.
    if "/" not in raw and not raw.startswith("@") and not raw.startswith("~"):
        forms.add(raw)
        slug_raw = _slugify(raw)
        if slug_raw:
            forms.add(slug_raw)
    # Always emit the normalized form + its org-prefixed variant.
    norm = normalize_modelsdev_id(raw)
    slug = _slugify(norm)
    if slug:
        leaf = slug.rsplit("/", 1)[-1]
        forms.add(leaf)
        if org_id:
            forms.add(f"{org_id}/{leaf}")
    return sorted(f for f in forms if f and f.count("/") <= 1)


def _attach_provider_aliases(
    entries: list[dict],
    group_recs: list[dict],
    org_id: str | None,
) -> None:
    """Union every provider spelling in the underlying group onto the matching
    emitted entry as a provider-tagged alias.

    Each models.dev record is routed to the entry whose canonical/alias set
    already contains its slugified family/leaf spelling; the spelling is added
    with its `inference_platform` (from PROVIDER_TO_INFERENCE_PLATFORM). Tagged
    aliases are accumulated under entry['alias_platforms'] (a {alias->platform}
    map) which the writer flattens into the alias list while preserving the
    platform provenance in metadata. Plain aliases (no platform) still go on
    entry['aliases']."""
    # Index entries by every id/alias surface form -> entry.
    by_form: dict[str, dict] = {}
    for e in entries:
        by_form[e["id"]] = e
        for a in e.get("aliases", []):
            by_form.setdefault(a, e)
    # Fallback target: the family-root entry (parents == []), else the first.
    root = next((e for e in entries if not e.get("parents")), entries[0] if entries else None)

    for r in group_recs:
        platform = PROVIDER_TO_INFERENCE_PLATFORM.get(r["provider"])
        raw = r["raw"]
        # Route via the NORMALIZED leaf slug (host/account scaffolding stripped)
        # so a host-prefixed mirror still lands on the right entry. Fall to the
        # raw form, the org-prefixed slug, then the family-root entry.
        norm_slug = _slugify(normalize_modelsdev_id(raw)).rsplit("/", 1)[-1]
        target = (
            by_form.get(raw)
            or by_form.get(norm_slug)
            or (by_form.get(f"{org_id}/{norm_slug}") if org_id else None)
            or root
        )
        if target is None:
            continue
        ap = target.setdefault("alias_platforms", {})
        for form in _provider_alias_forms(raw, org_id):
            if form == target["id"]:
                continue
            # Record the platform provenance; a form seen from multiple
            # providers keeps the first non-null platform.
            if form not in ap or (ap.get(form) is None and platform):
                ap[form] = platform


# ---------------------------------------------------------------------------
# Mint-decision rule. Before minting an off-HF {org}/{slug} canonical we
# ask: is this underlying group already a real HF repo? The authority is the
# frozen HF oracle (hf_model_id_resolution.json). We DEFER (no mint; the
# canonical IS the real HF id) only on a normalized-identity match CORROBORATED
# BY ORG AGREEMENT after the curated two-tier dev-org remap — never a loose
# name-only match across different developers. Default to MINT when unsure.
# ---------------------------------------------------------------------------

# evaleval/hf_model_id_resolution.json (one level above the registry repo root).
HF_ORACLE_JSON = REPO_ROOT.parent / "hf_model_id_resolution.json"

_HF_AUTHORITY: dict[str, dict[str, str]] | None = None


def _build_hf_authority(
    oracle_path: Path = HF_ORACLE_JSON,
    alias_index: dict[str, str] | None = None,
) -> dict[str, dict[str, str]]:
    """Build the "is this on HF" authority from the frozen oracle.

    Returns {dev_org: {normalized_name: real_hf_model_id}} over every oracle
    entry with resolution_status in {fixed_exact, fixed_near_miss} carrying a
    fixed_hf_model_id. The HF org of each repo is remapped through the SAME
    curated two-tier dev-org map used everywhere else (`_build_org_alias_index`
    / orgs.yaml hf_org + _ORG_ALIASES), so the bucket key is the developer org
    (`Qwen`->`alibaba`, `meta-llama`->`meta`, ...). The name is normalized via
    the resolver's `normalize` (case + all separators collapsed to a space)."""
    from eval_entity_resolver.normalization import normalize as _norm

    ai = alias_index if alias_index is not None else _dev_alias_index()
    out: dict[str, dict[str, str]] = defaultdict(dict)
    if not oracle_path.exists():
        return out
    oracle = json.loads(oracle_path.read_text()).get("resolutions", {})
    for _raw, meta in oracle.items():
        if meta.get("resolution_status") not in ("fixed_exact", "fixed_near_miss"):
            continue
        fixed = meta.get("fixed_hf_model_id")
        if not isinstance(fixed, str) or "/" not in fixed:
            continue
        hf_org, hf_name = fixed.split("/", 1)
        dev_org = ai.get(hf_org.lower(), hf_org.lower())
        out[dev_org].setdefault(_norm(hf_name), fixed)
    return out


def _hf_authority() -> dict[str, dict[str, str]]:
    global _HF_AUTHORITY
    if _HF_AUTHORITY is None:
        _HF_AUTHORITY = _build_hf_authority()
    return _HF_AUTHORITY


def _candidate_name_norms(
    spellings: list[str], dev_org: str | None, alias_index: dict[str, str]
) -> set[str]:
    """Normalized NAME forms a models.dev group's spellings should match HF on.

    For each spelling we take its leaf (post org/host strip), normalize it, AND
    — because a models.dev key can carry the developer's brand as a prefix
    (`qwen-qwq-32b` for `Qwen/QwQ-32B`) — also emit the form with a leading
    brand token stripped when that token is a curated alias of THIS group's dev
    org. This collapses `qwen-qwq-32b`->`qwq-32b` without a bespoke fuzzy
    matcher: it reuses the same org alias index used for org resolution."""
    from eval_entity_resolver.normalization import normalize as _norm

    norms: set[str] = set()
    for sp in spellings:
        if not sp:
            continue
        leaf = sp.rsplit("/", 1)[-1]
        n = _norm(leaf)
        if not n:
            continue
        norms.add(n)
        # Strip a leading brand token that is a curated alias of the dev org.
        toks = n.split(" ")
        if dev_org and len(toks) > 1:
            stripped = " ".join(toks[1:])
            if alias_index.get(toks[0]) == dev_org and stripped:
                norms.add(stripped)
    return norms


def _hf_defer_target(
    candidate_id: str,
    org_id: str | None,
    spellings: list[str],
    alias_index: dict[str, str],
    authority: dict[str, dict[str, str]] | None = None,
) -> str | None:
    """Decide DEFER vs MINT for a models.dev underlying group.

    Returns the real HF id to defer to when the group resolves to an HF repo
    with org agreement (after dev-org remap); returns None to MINT otherwise.

    Confident == normalized-identity match WITHIN THE SAME dev-org bucket. A
    group with no org, or whose normalized names match only under a DIFFERENT
    developer, always MINTS (no cross-developer false merges)."""
    if org_id is None:
        return None
    auth = authority if authority is not None else _hf_authority()
    bucket = auth.get(org_id)
    if not bucket:
        return None
    forms = list(spellings)
    if candidate_id:
        forms.append(candidate_id)
    for name_norm in _candidate_name_norms(forms, org_id, alias_index):
        hit = bucket.get(name_norm)
        if hit is not None:
            return hit
    return None


def _hf_deferred_entry(
    hf_id: str,
    org_id: str | None,
    head: dict,
    group_recs: list[dict],
    mint_id: str,
    display_name: str,
) -> dict:
    """Build an HF-deferred record: canonical id IS the real HF repo id, with
    models.dev metadata (providers / open_weights / release_date) merged on and
    the models.dev spellings (mint id + display name) added as aliases. Mirrors
    the hand-folds for Qwen/QwQ-32B and LiquidAI/LFM2-24B-A2B (no new mint)."""
    aliases = []
    for a in (mint_id, display_name):
        if a and a != hf_id and a not in aliases:
            aliases.append(a)
    return {
        "id": hf_id,
        "display_name": head["display_name"] or humanize_model_slug(hf_id.split("/", 1)[-1]),
        "org_id": org_id,
        "family": None,
        "architecture": None,
        "params_billions": None,
        "parents": [],
        "open_weights": head["open_weights"],
        "release_date": head["release_date"],
        "input_modalities": None,
        "output_modalities": None,
        "tags": ["open-weight"] if head["open_weights"] else [],
        "aliases": aliases,
        "metadata": json.dumps(
            {
                "underlying_key": head.get("root_key", hf_id),
                "providers": sorted({r["provider"] for r in group_recs}),
                "hf_deferred": True,
            },
            sort_keys=True,
        ),
        # The canonical is the HF id; this record only enriches it, so it must
        # never override the HF source's reviewed status — it's an enrichment.
        "review_status": "reviewed",
        "resolution_source": "models_dev",
    }


def _mint_rehost_entry(
    root_key: str,
    org_id: str | None,
    head: dict,
    group_recs: list[dict],
) -> list[dict]:
    """Mint a single canonical for a re-host-only / closed-API group that has
    no author-lab family tree. The canonical id is `{org}/{Model-Name}` when an
    org is known, else a bare slug (org-less,
    flagged for curation). Returns [entry] (provider aliases attached by the
    caller via _attach_provider_aliases)."""
    head_raw = head["head_spelling"] or root_key
    # Mint slug from the head spelling, but FIRST run it through the models.dev
    # normalizer so provider/host/account scaffolding (`@cf/`, `accounts/.../`,
    # `org/`) is stripped — otherwise a mirror spelling leaks slashes into the
    # canonical id. Fall back to the canon root key when normalization empties.
    slug = _slugify(normalize_modelsdev_id(head_raw)) or _slugify(root_key)
    # Defensive: never let a multi-segment spelling produce a 2-slash id.
    slug = slug.rsplit("/", 1)[-1]
    canonical_id = f"{org_id}/{slug}" if org_id else slug
    display_name = head["display_name"] or humanize_model_slug(slug)
    open_weights = head["open_weights"]
    release_date = head["release_date"]
    tags = ["open-weight"] if open_weights else []
    if org_id is None:
        tags = tags + ["org-unknown"]
    entry = {
        "id": canonical_id,
        "display_name": display_name,
        "org_id": org_id,
        "family": slug,
        "architecture": None,
        "params_billions": None,
        "parents": [],
        "open_weights": open_weights,
        "release_date": release_date,
        "input_modalities": None,
        "output_modalities": None,
        "tags": tags,
        "aliases": [],
        "metadata": json.dumps(
            {"underlying_key": root_key, "providers": sorted({r["provider"] for r in group_recs})},
            sort_keys=True,
        ),
        # Re-host-only / minted-from-models.dev groups are NOT author-confirmed.
        "review_status": "draft" if not head["has_author_lab_entry"] else "reviewed",
        "resolution_source": "models_dev",
    }
    return [entry]


def _mint_or_defer_rehost(
    root_key: str,
    org_id: str | None,
    head: dict,
    group_recs: list[dict],
    alias_index: dict[str, str],
) -> list[dict]:
    """Mint-decision wrapper for the re-host path: if the underlying group
    already resolves to a real HF repo (normalized-identity match with
    dev-org agreement against the frozen oracle), DEFER — emit an HF-deferred
    record keyed by the real HF id with the models.dev spellings as aliases.
    Otherwise MINT the off-HF {org}/{slug} canonical exactly as before."""
    # The prospective mint id + display, mirroring _mint_rehost_entry's slug.
    head_raw = head["head_spelling"] or root_key
    slug = (_slugify(normalize_modelsdev_id(head_raw)) or _slugify(root_key)).rsplit("/", 1)[-1]
    mint_id = f"{org_id}/{slug}" if org_id else slug
    display_name = head["display_name"] or humanize_model_slug(slug)

    # Candidate spellings to check against HF: the mint id, the head spelling,
    # every raw provider spelling, and the display name (org/host scaffolding is
    # stripped to the leaf inside _candidate_name_norms).
    spellings = [mint_id, head_raw, display_name, slug]
    spellings += [r["raw"] for r in group_recs if r.get("raw")]

    hf_id = _hf_defer_target(mint_id, org_id, spellings, alias_index)
    if hf_id is not None:
        head = {**head, "root_key": root_key}
        return [_hf_deferred_entry(hf_id, org_id, head, group_recs, mint_id, display_name)]
    return _mint_rehost_entry(root_key, org_id, head, group_recs)


def _generate_models(api_json: dict, known_org_ids: set[str]) -> tuple[list[dict], list[str]]:
    """Provider-preserving group -> mint -> alias over the FULL models.dev
    catalog. Every provider that maps to an inference_platform is
    processed (no author-only gate); each underlying group yields one canonical
    family (author-lab tree when the author lab is present, else a minted
    re-host canonical), and every provider spelling in the group is aliased in
    carrying its inference_platform.

    Returns (entries, skipped_no_org). A non-empty skipped_no_org is a hard
    error (an author-lab provider mapped to an org missing from seed/orgs.yaml).
    """
    out: list[dict] = []
    skipped_providers: list[str] = []
    skipped_no_org: list[str] = []
    alias_index = _build_org_alias_index()

    # 1. Dedup the whole catalog into underlying groups.
    groups = build_underlying_groups(api_json)

    for root_key, recs in sorted(groups.items()):
        # Drop records whose provider isn't a known inference_platform (none
        # today — all 137 map — but keep the guard for forward-compat).
        recs = [r for r in recs if r["provider"] in PROVIDER_TO_INFERENCE_PLATFORM]
        if not recs:
            for r0 in recs:
                skipped_providers.append(r0["provider"])
            continue

        head = pick_underlying(root_key, recs)
        hf_org = head["author_org"]
        org_id = normalize_org_slug(hf_org, alias_index)

        # If a provider in the curated PROVIDER_TO_ORG allowlist authored this
        # group, prefer ITS curated org id (validated against seed/orgs.yaml)
        # over the reference scripts' HF-style slug — the two maps can diverge
        # (e.g. reference says `inceptionai`, PROVIDER_TO_ORG says `inception`).
        curated_author_recs = [
            r for r in recs
            if r["provider"] in PROVIDER_TO_ORG
            and "/" not in r["raw"]
            and head["has_author_lab_entry"]
            # only treat as author when its curated org agrees with the family-org
            and (
                org_id is None
                or PROVIDER_TO_ORG[r["provider"]] == org_id
                or normalize_org_slug(AUTHOR_PROV_ORG.get(r["provider"]), alias_index) == org_id
            )
        ]
        if curated_author_recs:
            org_id = PROVIDER_TO_ORG[curated_author_recs[0]["provider"]]

        # Records belonging to the author lab (their provider's org matches the
        # group org) drive the family tree; the rest are re-host aliases.
        author_recs = curated_author_recs

        if head["has_author_lab_entry"] and author_recs and org_id:
            if org_id not in known_org_ids:
                skipped_no_org.append(
                    f"{author_recs[0]['provider']} -> {org_id} (group {root_key})"
                )
                continue
            # Build the author-lab family tree from the author records, grouped
            # by family slug (a group may span a couple of stage/size siblings).
            by_family: dict[str, list[dict]] = defaultdict(list)
            for r in author_recs:
                by_family[_family_for(r["record"])].append(r["record"])
            group_entries: list[dict] = []
            for family_slug, models in sorted(by_family.items()):
                if not family_slug:
                    continue
                group_entries.extend(_build_family_entries(org_id, family_slug, models))
            if not group_entries:
                group_entries = _mint_or_defer_rehost(root_key, org_id, head, recs, alias_index)
        else:
            # Re-host-only / closed-API group with no usable author tree: mint
            # UNLESS this group is already a real HF repo (defer instead).
            group_entries = _mint_or_defer_rehost(root_key, org_id, head, recs, alias_index)

        # 2. Alias every provider spelling in the group into the entries.
        _attach_provider_aliases(group_entries, recs, org_id)
        out.extend(group_entries)

    if skipped_providers:
        print(
            f"[refresh] skipped {len(skipped_providers)} provider records not in "
            f"PROVIDER_TO_INFERENCE_PLATFORM",
            file=sys.stderr,
        )
    # Dedup entries by id (a model that appears in two dedup groups, e.g. via
    # different snapshots, would otherwise emit twice). Merge aliases on collide.
    return _dedup_entries(out), skipped_no_org


def _dedup_entries(entries: list[dict]) -> list[dict]:
    """Collapse entries that share a canonical id (can happen when two
    underlying groups mint the same family root). Union aliases / tags /
    alias_platforms; prefer reviewed over draft; keep first non-null scalars."""
    by_id: dict[str, dict] = {}
    for e in entries:
        cur = by_id.get(e["id"])
        if cur is None:
            by_id[e["id"]] = e
            continue
        # Union list/dict fields.
        cur["aliases"] = sorted(set(cur.get("aliases", [])) | set(e.get("aliases", [])))
        cur["tags"] = sorted(set(cur.get("tags", [])) | set(e.get("tags", [])))
        ap = cur.setdefault("alias_platforms", {})
        for k, v in (e.get("alias_platforms") or {}).items():
            if k not in ap or (ap.get(k) is None and v):
                ap[k] = v
        # open_weights any(True); reviewed wins.
        cur["open_weights"] = bool(cur.get("open_weights")) or bool(e.get("open_weights"))
        if e.get("review_status") == "reviewed":
            cur["review_status"] = "reviewed"
        # Keep earliest release_date.
        rd = [d for d in (cur.get("release_date"), e.get("release_date")) if d]
        cur["release_date"] = min(rd) if rd else None
    return sorted(by_id.values(), key=lambda e: e["id"])


def _load_known_org_ids() -> set[str]:
    if not ORGS_SEED_PATH.exists():
        return set()
    data = yaml.safe_load(ORGS_SEED_PATH.read_text()) or []
    return {e["id"] for e in data if "id" in e}


_HEADER = """# Generated from models.dev (https://models.dev) — DO NOT EDIT BY HAND.
# To update: edit seed/models/core.yaml (curated canonicals win at load
# time), then run `python scripts/refresh_from_modelsdev.py` to regenerate
# this file.
#
# Source: https://models.dev/api.json (MIT, (c) 2025 models.dev)
# Last refresh date is in git history — see
# `git log -1 -- seed/models/sources/models_dev.generated.yaml`.
#
# This file is one data source among potentially several under
# `seed/models/sources/`. It contains pure models.dev output — no curated
# overlays. The seed CLI loader merges sources → core → enrichments at
# load time (field-level merge: aliases / tags UNION).
#
# Each entry collapses all snapshots / dated variants of a model family
# into one canonical id (`<org>/<family-slug>`). The resolver's fuzzy stem
# step (in eval_entity_resolver/strategies/fuzzy.py) strips date suffixes,
# thinking budgets, hosting provider tags, etc., so per-snapshot raw IDs
# resolve to this family canonical without needing per-snapshot entries.
#
# `aliases` lists snapshot IDs we observed in models.dev for this family.
"""


def _finalize_entries(entries: list[dict]) -> list[dict]:
    """Flatten the in-progress `alias_platforms` map into the persisted shape:
    union its keys into `aliases`, and fold the {alias -> inference_platform}
    provenance into metadata['alias_platforms'] so the loader can wire the
    platform FK per alias. Drops the working `alias_platforms` key. Returns a
    new list; does not mutate inputs in place beyond the working key."""
    finalized: list[dict] = []
    for e in entries:
        ap = e.pop("alias_platforms", None) or {}
        aliases = set(e.get("aliases", []))
        aliases.update(ap.keys())
        # Remove self-id from aliases.
        aliases.discard(e["id"])
        e["aliases"] = sorted(aliases)
        if ap:
            meta = json.loads(e.get("metadata") or "{}")
            # Only non-null platform tags carry FK provenance.
            meta["alias_platforms"] = {k: v for k, v in sorted(ap.items()) if v}
            e["metadata"] = json.dumps(meta, sort_keys=True)
        finalized.append(e)
    return finalized


def _write_yaml(entries: list[dict], path: Path) -> str:
    body = yaml.safe_dump(_finalize_entries(entries), sort_keys=False, allow_unicode=True, width=200)
    return _HEADER + "\n" + body


# ---------------------------------------------------------------------------
# Full-catalog split/dedup + org de-orphan. The daily cron regenerates
# models_dev_catalog.generated.yaml directly so it never goes stale, and HF
# source-of-truth wins every collision: an HF-present model becomes an
# ALIAS-ONLY enrichment onto the existing HF-cased canonical (no lowercase twin
# minted); only genuinely models.dev-only (not-on-HF) models are minted fresh.
# ---------------------------------------------------------------------------

CATALOG_OUT_PATH = REPO_ROOT / "seed" / "models" / "sources" / "models_dev_catalog.generated.yaml"
HF_ORACLE_PATH = REPO_ROOT / "seed" / "models" / "sources" / "hf_oracle.generated.yaml"
HUB_STATS_PATH = REPO_ROOT / "seed" / "models" / "sources" / "hub_stats.generated.yaml"
CORE_PATH = REPO_ROOT / "seed" / "models" / "core.yaml"
ENRICH_ALIASES_PATH = REPO_ROOT / "seed" / "models" / "enrichments" / "aliases.yaml"
ORGS_GENERATED_PATH = REPO_ROOT / "seed" / "orgs.generated.yaml"

# All EXISTING model sources whose id+alias surface forms the catalog must not
# clash with. HF/curated WIN id+casing. models_dev.generated.yaml (the
# re-cased pure source) is included so the catalog stays purely additive.
_CATALOG_EXISTING_SOURCES = (
    HF_ORACLE_PATH, SEED_PATH, HUB_STATS_PATH, CORE_PATH, ENRICH_ALIASES_PATH,
)

_CATALOG_HEADER = """# AUTO-GENERATED by scripts/refresh_from_modelsdev.py (catalog split) — DO NOT HAND-EDIT.
# models.dev full-catalog seed. Two record kinds:
#   * fresh canonical mints: models.dev-only (not-on-HF) models — closed-API
#     families (Claude/GPT/Gemini/Grok) + the re-host/community tail. No HF
#     collision, so HF source-of-truth is not violated.
#   * alias-only enrichments {id, aliases}: a models.dev model that IS HF-present
#     (already a canonical). The existing HF-cased canonical wins; only the
#     provider-spelling aliases (carrying inference_platform in
#     metadata.alias_platforms) union onto it. No duplicate canonical is minted.
# The re-cased seed/models/sources/models_dev.generated.yaml is left intact;
# this file is purely additive. Regenerated by the daily refresh-models cron.
"""


def _catalog_load_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    d = yaml.safe_load(path.read_text())
    if isinstance(d, dict):
        d = d.get("entries", [])
    return d or []


def regenerate_catalog(full: list[dict]) -> None:
    """Split the finalized full models.dev catalog (`full`) against the existing
    canonical universe and write models_dev_catalog.generated.yaml + reconcile
    HF-derived community orgs into orgs.generated.yaml. `full` is the output of
    `_finalize_entries(_generate_models(...))`. Dedup/steal-guard semantics:
    HF wins; mint only models.dev-only."""
    # Use the resolver's normalize (collapses case + all separators + digit-dots)
    # to mirror the seed validator's normalized_match steal-guard.
    from eval_entity_resolver.normalization import normalize as _norm

    existing_form_to_cid: dict[str, str] = {}
    existing_exact: dict[str, str] = {}
    existing_norm: dict[str, str] = {}

    def _add_form(form: str, cid: str) -> None:
        if form:
            existing_form_to_cid.setdefault(form.lower(), cid)

    def _add_exact(form: str, cid: str) -> None:
        if form:
            existing_exact.setdefault(form, cid)
            existing_norm.setdefault(_norm(form), cid)

    for path in _CATALOG_EXISTING_SOURCES:
        for e in _catalog_load_list(path):
            cid = e.get("id")
            if not cid:
                continue
            _add_form(cid, cid)
            _add_exact(cid, cid)
            dn = e.get("display_name")
            if dn:
                _add_exact(dn, cid)
            for a in (e.get("aliases") or []):
                _add_form(a, cid)
                _add_exact(a, cid)

    def _steals(form: str, cid: str) -> bool:
        owner = existing_exact.get(form)
        if owner is not None and owner != cid:
            return True
        nowner = existing_norm.get(_norm(form))
        return nowner is not None and nowner != cid

    fresh: list[dict] = []
    enrich: list[dict] = []
    fresh_seen_lc: dict[str, dict] = {}
    fresh_form_owner: dict[str, str] = {}

    def _enrich_target(cid: str, aliases: list[str], ap: dict | None) -> None:
        keep = sorted({a for a in aliases if a and a != cid and not _steals(a, cid)})
        rec: dict = {"id": cid}
        if keep:
            rec["aliases"] = keep
        if ap:
            ap2 = {k: v for k, v in ap.items() if k != cid and not _steals(k, cid)}
            if ap2:
                rec["metadata"] = json.dumps({"alias_platforms": ap2}, sort_keys=True)
        if keep or rec.get("metadata"):
            enrich.append(rec)

    def _forms_of(e: dict) -> list[str]:
        forms = [e["id"]]
        if e.get("display_name"):
            forms.append(e["display_name"])
        forms.extend(a for a in (e.get("aliases") or []) if a)
        return forms

    for e in full:
        cid = e["id"]
        cid_low = cid.lower()
        meta = json.loads(e.get("metadata") or "{}")
        ap = meta.get("alias_platforms") or {}

        cased = existing_form_to_cid.get(cid_low)
        if cased is not None:
            _enrich_target(cased, e.get("aliases", []), ap)
            continue
        owner_exact = (
            existing_exact.get(cid)
            or existing_norm.get(_norm(cid))
            or (existing_exact.get(e.get("display_name")) if e.get("display_name") else None)
        )
        if owner_exact is not None:
            _enrich_target(owner_exact, [cid] + list(e.get("aliases", [])), ap)
            continue
        if cid_low in fresh_seen_lc:
            prior = fresh_seen_lc[cid_low]
            cur = set(prior.get("aliases", []))
            for a in (a for a in e.get("aliases", []) if a):
                if a == prior["id"] or existing_exact.get(a) is not None:
                    continue
                owner = fresh_form_owner.get(a)
                if owner is not None and owner != prior["id"]:
                    continue
                cur.add(a)
                fresh_form_owner[a] = prior["id"]
            prior["aliases"] = sorted(cur)
            continue
        peer = fresh_form_owner.get(cid)
        if peer is not None and peer != cid:
            continue

        clean: list[str] = []
        for a in e.get("aliases", []):
            if not a or a == cid or _steals(a, cid):
                continue
            owner = fresh_form_owner.get(a)
            if owner is not None and owner != cid:
                continue
            clean.append(a)
        e["aliases"] = sorted(set(clean))
        dn = e.get("display_name")

        def _claimed(form: str) -> bool:
            return existing_exact.get(form) is not None or (
                fresh_form_owner.get(form) not in (None, cid))

        if dn and _claimed(dn):
            cand = cid.split("/", 1)[-1]
            e["display_name"] = cand if not _claimed(cand) else cid
        for form in _forms_of(e):
            fresh_form_owner.setdefault(form, cid)
            existing_exact.setdefault(form, cid)
            existing_norm.setdefault(_norm(form), cid)
        fresh_seen_lc[cid_low] = e
        if ap:
            ap2 = {k: v for k, v in ap.items() if k in e["aliases"]}
            if ap2:
                meta["alias_platforms"] = ap2
            else:
                meta.pop("alias_platforms", None)
            e["metadata"] = json.dumps(meta, sort_keys=True)
        fresh.append(e)

    out_entries = fresh + enrich
    body = yaml.safe_dump(out_entries, sort_keys=False, allow_unicode=True, width=200)
    CATALOG_OUT_PATH.write_text(_CATALOG_HEADER + "\n" + body)

    # --- Org reconciliation (two-tier rule) --------------------------------
    curated_org_ids = {e["id"] for e in _catalog_load_list(ORGS_SEED_PATH) if "id" in e}
    gen_orgs = _catalog_load_list(ORGS_GENERATED_PATH)
    gen_org_ids = {e["id"] for e in gen_orgs if "id" in e}
    referenced = {e.get("org_id") for e in fresh if e.get("org_id")}
    missing = sorted(referenced - curated_org_ids - gen_org_ids)
    if missing:
        for oid in missing:
            gen_orgs.append({
                "id": oid, "display_name": oid, "hf_org": oid,
                "kind": "community", "tags": "[]", "metadata": "{}",
                "review_status": "reviewed",
            })
        gen_header = (
            ORGS_GENERATED_PATH.read_text().split("\n- ", 1)[0].rstrip()
            if ORGS_GENERATED_PATH.exists() else ""
        )
        if not gen_header.startswith("#"):
            gen_header = "# AUTO-GENERATED — HF-derived community orgs."
        ORGS_GENERATED_PATH.write_text(
            gen_header + "\n"
            + yaml.safe_dump(gen_orgs, sort_keys=False, allow_unicode=True, width=200)
        )
        print(f"[refresh] catalog: reconciled {len(missing)} missing community org(s): {missing}", file=sys.stderr)
    print(
        f"[refresh] catalog: {len(fresh)} fresh mint(s) (not-on-HF), "
        f"{len(enrich)} alias-only enrichment(s) (HF-present) -> {CATALOG_OUT_PATH}",
        file=sys.stderr,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--no-fetch", action="store_true", help="use cached /tmp/modelsdev_api.json")
    p.add_argument("--dry-run", action="store_true", help=f"print diff vs current {SEED_PATH}; don't write")
    p.add_argument(
        "--preview-out",
        type=Path,
        default=None,
        help="write to this PREVIEW path instead of the committed generated YAML "
        "(for inspection; leaves the committed file untouched)",
    )
    p.add_argument(
        "--catalog",
        action="store_true",
        help="ONLY (re)generate the models_dev_catalog.generated.yaml "
        "split (de-orphan) + reconcile orgs.generated.yaml; does NOT rewrite "
        "models_dev.generated.yaml. The cron runs this as a second step after "
        "the source write so the catalog splits against the settled re-cased "
        "models_dev.",
    )
    args = p.parse_args()

    api = _fetch(use_cache=args.no_fetch)
    known_orgs = _load_known_org_ids()
    if not known_orgs:
        print(f"[refresh] ERROR: {ORGS_SEED_PATH} not found or empty. Seed orgs first.", file=sys.stderr)
        return 1

    # --catalog: skip the models_dev source rewrite entirely; only split the
    # full author-lab catalog against the EXISTING on-disk sources.
    if args.catalog:
        generated, skipped_no_org = _generate_models(api, known_orgs)
        if skipped_no_org:
            print(f"[refresh] ERROR: {len(skipped_no_org)} provider(s) -> unknown org_id", file=sys.stderr)
            return 1
        regenerate_catalog(_finalize_entries(generated))
        return 0

    generated, skipped_no_org = _generate_models(api, known_orgs)
    if skipped_no_org:
        print(
            f"[refresh] ERROR: {len(skipped_no_org)} provider(s) mapped to unknown org_id "
            f"(must exist in {ORGS_SEED_PATH}):",
            file=sys.stderr,
        )
        for entry in skipped_no_org:
            print(f"  - {entry}", file=sys.stderr)
        print(
            "[refresh] Add the missing orgs to seed/orgs.yaml or fix the "
            "PROVIDER_TO_ORG mapping in this script, then re-run.",
            file=sys.stderr,
        )
        return 1
    out_path = args.preview_out or SEED_PATH
    new_text = _write_yaml(generated, out_path)

    if args.preview_out is not None and not args.dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(new_text)
        print(
            f"[refresh] PREVIEW: wrote {len(generated)} model entries to {out_path} "
            f"(committed {SEED_PATH} untouched)",
            file=sys.stderr,
        )
        return 0

    if args.dry_run:
        if SEED_PATH.exists():
            old = SEED_PATH.read_text()
            if old == new_text:
                print("[refresh] no changes")
            else:
                import difflib
                diff = difflib.unified_diff(
                    old.splitlines(keepends=True),
                    new_text.splitlines(keepends=True),
                    fromfile=str(SEED_PATH),
                    tofile=f"{SEED_PATH} (generated)",
                )
                sys.stdout.writelines(diff)
        else:
            print(new_text)
        return 0

    SEED_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEED_PATH.write_text(new_text)
    print(
        f"[refresh] wrote {len(generated)} model entries to {SEED_PATH}",
        file=sys.stderr,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
