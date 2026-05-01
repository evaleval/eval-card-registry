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
from collections import defaultdict
from pathlib import Path

import yaml

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


def _build_family_entry(org_id: str, family_slug: str, models: list[dict]) -> dict:
    """Collapse a list of snapshot/variant model records into one canonical entry.

    Aliases include every snapshot id we saw under this family + the bare
    family slug. The fuzzy resolver further strips dates/thinking budgets, so
    snapshot variants beyond what models.dev lists also resolve correctly.
    """
    canonical_id = f"{org_id}/{family_slug}"

    # Pick the most descriptive display name: prefer entries whose id matches
    # the family slug exactly (the "canonical" snapshot for that family).
    # Fall back to a humanized slug rather than picking an arbitrary snapshot
    # name (which would produce e.g. "Qwen3 235B-A22B" for the qwen family).
    display_name = ""
    for m in models:
        if _slugify(m.get("id", "")) == family_slug:
            display_name = m.get("name") or family_slug
            break
    if not display_name:
        display_name = " ".join(p.capitalize() for p in family_slug.replace("_", "-").split("-"))

    # Aliases: every snapshot id (e.g. claude-opus-4-5-20251101), each
    # listed both bare AND with the org prefix. EEE corpus contains both
    # forms (`claude-opus-4-5-20251101` from API logs and
    # `anthropic/claude-opus-4-5-20251101` from HF-format ids), so both
    # need exact-match coverage. Display name is added by the seed CLI.
    snapshot_ids = sorted({_slugify(m["id"]) for m in models if m.get("id")})
    aliases: list[str] = []
    for snap in snapshot_ids:
        if snap != family_slug:
            aliases.append(snap)
        # Always include the org-prefixed form (covers HF-format raw values
        # like `anthropic/claude-3-5-haiku-20241022` that have a different
        # word order than the lab's display name).
        prefixed = f"{org_id}/{snap}"
        if prefixed != canonical_id:
            aliases.append(prefixed)

    # Extract release dates / open_weights flag from any snapshot that has them
    open_weights = any(m.get("open_weights") for m in models)
    release_dates = sorted({m["release_date"] for m in models if m.get("release_date")})

    metadata: dict = {"snapshots": snapshot_ids}
    if release_dates:
        metadata["release_dates"] = release_dates
    if any(m.get("knowledge") for m in models):
        metadata["knowledge_cutoffs"] = sorted({
            m["knowledge"] for m in models if m.get("knowledge")
        })

    return {
        "id": canonical_id,
        "display_name": display_name,
        "org_id": org_id,
        "family": family_slug,
        "architecture": None,
        "params_billions": None,
        "parent_model_id": None,
        "tags": ["open-weight"] if open_weights else [],
        "aliases": aliases,
        "metadata": json.dumps(metadata, sort_keys=True),
        "review_status": "reviewed",
    }


def _generate_models(api_json: dict, known_org_ids: set[str]) -> tuple[list[dict], list[str]]:
    """Walk providers, filter to known authors, group by family, emit entries.

    Returns (entries, skipped_no_org). The caller should treat a non-empty
    skipped_no_org as a hard error — silently dropping a provider's models
    because of a missing org row would quietly degrade coverage on the next
    automated CI refresh.
    """
    out: list[dict] = []
    skipped_providers: list[str] = []
    skipped_no_org: list[str] = []

    for provider_slug, provider_data in api_json.items():
        if provider_slug not in PROVIDER_TO_ORG:
            skipped_providers.append(provider_slug)
            continue
        org_id = PROVIDER_TO_ORG[provider_slug]
        if org_id not in known_org_ids:
            skipped_no_org.append(f"{provider_slug} -> {org_id}")
            continue

        # Group provider's models by family. Skip mirror entries (model id
        # like `nvidia/meta/llama-3.3-70b-instruct` is NVIDIA hosting a Meta
        # release, not an NVIDIA original) — those create false canonicals
        # that compete with the authoring lab's entry.
        by_family: dict[str, list[dict]] = defaultdict(list)
        for model in (provider_data.get("models") or {}).values():
            model_id = model.get("id", "")
            if "/" in model_id:
                continue
            by_family[_family_for(model)].append(model)

        for family_slug, models in sorted(by_family.items()):
            if not family_slug:
                continue
            out.append(_build_family_entry(org_id, family_slug, models))

    if skipped_providers:
        print(
            f"[refresh] skipped {len(skipped_providers)} providers not in PROVIDER_TO_ORG "
            f"(inference re-hosts/gateways/duplicates)",
            file=sys.stderr,
        )
    return sorted(out, key=lambda e: e["id"]), skipped_no_org


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


def _write_yaml(entries: list[dict], path: Path) -> str:
    body = yaml.safe_dump(entries, sort_keys=False, allow_unicode=True, width=200)
    return _HEADER + "\n" + body


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--no-fetch", action="store_true", help="use cached /tmp/modelsdev_api.json")
    p.add_argument("--dry-run", action="store_true", help=f"print diff vs current {SEED_PATH}; don't write")
    args = p.parse_args()

    api = _fetch(use_cache=args.no_fetch)
    known_orgs = _load_known_org_ids()
    if not known_orgs:
        print(f"[refresh] ERROR: {ORGS_SEED_PATH} not found or empty. Seed orgs first.", file=sys.stderr)
        return 1

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
    new_text = _write_yaml(generated, SEED_PATH)

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
