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
# into a typed `parents` edge. The classification table mirrors the
# one-shot promotion pass that split buried aliases into their own
# canonicals — keeping the same enum values here means models.dev
# refresh and curated core.yaml entries land in identical shapes.
_TOKEN_CLASSIFICATIONS: dict[str, tuple[str, str | None]] = {
    "instruct": ("variant", "mode"),
    "it": ("variant", "mode"),
    "chat": ("variant", "mode"),
    "base": ("variant", "mode"),
    "thinking": ("variant", "mode"),
    "reasoning": ("variant", "mode"),
    "nothink": ("variant", "mode"),
    "guard": ("variant", "mode"),
    "safeguard": ("variant", "mode"),
    "moderation": ("variant", "mode"),
    "vision": ("variant", "modality"),
    "vl": ("variant", "modality"),
    "coder": ("variant", "domain"),
    "code": ("variant", "domain"),
    "math": ("variant", "domain"),
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
_VERSION_RE = re.compile(r"^v\d+(\.\d+)*$", re.IGNORECASE)
_DATE_8_RE = re.compile(r"^\d{8}$")
_DATE_4_RE = re.compile(r"^\d{4}$")
_DATE_3_RE = re.compile(r"^\d{3}$")
_MOE_ACTIVE_RE = re.compile(r"^a\d+b$", re.IGNORECASE)


def _classify_token(token: str) -> tuple[str, str | None] | None:
    t = token.lower()
    if t in _TOKEN_CLASSIFICATIONS:
        return _TOKEN_CLASSIFICATIONS[t]
    if _VERSION_RE.match(t) or _DATE_8_RE.match(t) or _DATE_4_RE.match(t) or _DATE_3_RE.match(t):
        return ("variant", "version")
    if _MOE_ACTIVE_RE.match(t):
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
    display_name = ""
    for m in models:
        if _slugify(m.get("id", "")) == family_slug:
            display_name = m.get("name") or family_slug
            break
    if not display_name:
        display_name = " ".join(p.capitalize() for p in family_slug.replace("_", "-").split("-"))

    open_weights = any(m.get("open_weights") for m in models)
    release_dates = sorted({m["release_date"] for m in models if m.get("release_date")})
    release_date = release_dates[0] if release_dates else None
    snapshot_ids = sorted({_slugify(m["id"]) for m in models if m.get("id")})

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
            if is_leaf:
                child_aliases = sorted({snap_dashed, f"{org_id}/{snap_dashed}"})
                if m.get("release_date"):
                    child_release = m["release_date"]
                child_open_weights = bool(m.get("open_weights")) or open_weights

            entry = {
                "id": new_id,
                "display_name": (m.get("name") or _humanize(new_id.split("/", 1)[-1])) if is_leaf else _humanize(new_id.split("/", 1)[-1]),
                "org_id": org_id,
                "family": family_slug,
                "architecture": None,
                "params_billions": None,
                "parents": [parent_edge],
                "open_weights": child_open_weights,
                "release_date": child_release,
                "tags": ["open-weight"] if child_open_weights else [],
                "aliases": child_aliases,
                "metadata": "{}",
                "review_status": "reviewed",
            }
            seen_ids[new_id] = entry
            out_entries.append(entry)
            current_id = new_id

    return out_entries


def _humanize(slug: str) -> str:
    parts = slug.replace("_", "-").split("-")
    return " ".join(p.capitalize() if p[:1].isalpha() else p for p in parts)


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
            out.extend(_build_family_entries(org_id, family_slug, models))

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
