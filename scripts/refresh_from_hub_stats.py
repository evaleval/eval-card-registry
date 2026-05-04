#!/usr/bin/env python3
"""
Generate seed/models/sources/hub_stats.generated.yaml from
HuggingFace's `cfahlgren1/hub-stats` dataset.

Scope: BACKFILL ONLY. For HF ids (`org/name`) already aliased to one of
our canonicals, enrich the existing canonical with hub-stats metadata —
`release_date` from `createdAt`, `params_billions` approximated from
the safetensors total, license from cardData, useful tags from the
tag list. The seed loader's field-level merge fills in missing scalars
without overriding curated values in `seed/models/core.yaml`.

Lineage descendant pre-load (community quants/finetunes whose
`baseModels` chain points at our covered models) is intentionally
NOT done here — initial scoping showed ~177k descendants of our
4k HF ids, mostly LoRA adapters. The on-demand enrichment at draft
creation handles those when EEE actually encounters them.

Re-run efficiency: an etag-watermark in
`seed/models/sources/hub_stats.state.json` records the parquet's
content ETag and the candidate HF ids checked against it. When the
upstream parquet hasn't republished AND the candidate set is unchanged,
the script exits without querying — most cron cycles are no-ops.

Output:
    seed/models/sources/hub_stats.generated.yaml — enrichment entries
    that merge into existing canonicals at seed time.
    seed/models/sources/hub_stats.state.json     — etag watermark.

Usage:
    python scripts/refresh_from_hub_stats.py             # full run
    python scripts/refresh_from_hub_stats.py --dry-run   # preview only
    python scripts/refresh_from_hub_stats.py --force     # ignore etag
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import duckdb
import httpx
import yaml

from eval_entity_resolver.display import humanize_model_slug

# Shared hub-stats helpers live in the package so the runtime resolver
# (live lookup at draft creation) and this bulk refresh script stay
# consistent on row-shape parsing.
from eval_card_registry.services.hub_stats import (
    PARQUET_URL,
    QUERY_COLUMNS,
    approx_params_billions as _approx_params_billions,
    coerce_date as _coerce_date,
    enrich_draft_from_row,
    extract_license as _extract_license,
    filter_useful_tags as _filter_useful_tags,
    hf_id_to_canonical,
    normalize as _normalize,
    slugify as _slugify,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ORGS_PATH = REPO_ROOT / "seed" / "orgs.yaml"
MODELS_OUT_PATH = REPO_ROOT / "seed" / "models" / "sources" / "hub_stats.generated.yaml"
STATE_PATH = REPO_ROOT / "seed" / "models" / "sources" / "hub_stats.state.json"
CORE_PATH = REPO_ROOT / "seed" / "models" / "core.yaml"
MODELS_DEV_GENERATED = REPO_ROOT / "seed" / "models" / "sources" / "models_dev.generated.yaml"


# ---------------------------------------------------------------------------
# Etag watermark — short-circuits the daily cron on no-op cycles.
# ---------------------------------------------------------------------------

def fetch_parquet_etag() -> Optional[str]:
    """HEAD the hub-stats parquet and return its content ETag (a stable
    SHA-256 of the file body). Returns None on any HTTP/network failure
    so the caller can fall back to the unconditional re-fetch path."""
    try:
        with httpx.Client(follow_redirects=True, timeout=30.0) as c:
            r = c.head(PARQUET_URL)
            r.raise_for_status()
            etag = r.headers.get("etag")
            return etag.strip('"') if etag else None
    except Exception as e:
        print(f"[refresh] WARN: parquet HEAD failed ({e}); skipping etag check", file=sys.stderr)
        return None


def load_state() -> dict:
    """Read the watermark state file. Returns an empty dict on missing,
    unreadable, or schema-invalid input — degrades to "no state known,
    re-check everything" so a corrupted/deleted file never makes the
    script behave worse than today's unconditional behaviour."""
    if not STATE_PATH.exists():
        return {}
    try:
        data = json.loads(STATE_PATH.read_text())
        if not isinstance(data, dict):
            return {}
        if not isinstance(data.get("rows_checked_at_etag"), list):
            return {}
        return data
    except (OSError, ValueError):
        return {}


def write_state(etag: str, rows_checked: list[str]) -> None:
    """Write the watermark. Sorted for diff-clean commits. Called LAST,
    after the YAML write succeeds — a crash before this line just means
    next cron run redoes the work, never that we mark rows checked
    without their data having landed in the YAML."""
    payload = {
        "parquet_etag": etag,
        "rows_checked_at_etag": sorted(set(rows_checked)),
    }
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Loading the existing registry's index — both org aliases and canonical
# model HF aliases. We use these to gate which hub-stats rows we look up.
# ---------------------------------------------------------------------------

def load_org_alias_to_canonical() -> dict[str, str]:
    """Map normalized HF org alias → our canonical org id.
    Example entry: `'meta-llama' → 'meta'`, `'qwen' → 'alibaba'`."""
    with open(ORGS_PATH) as f:
        orgs = yaml.safe_load(f) or []
    out: dict[str, str] = {}
    for o in orgs:
        cid = o.get("id")
        if not cid:
            continue
        out[_normalize(cid)] = cid
        if o.get("hf_org"):
            out[_normalize(o["hf_org"])] = cid
        for a in (o.get("aliases") or []):
            if isinstance(a, str):
                out[_normalize(a)] = cid
    return out


def load_existing_canonical_aliases() -> dict[str, str]:
    """Walk core.yaml + models_dev.generated.yaml and map every
    HF-shaped surface form (any string containing `/`) to the canonical
    id it belongs to. Keyed by normalized form so case/separator drift
    doesn't matter at lookup time."""
    out: dict[str, str] = {}

    def _ingest(path: Path) -> None:
        if not path.exists():
            return
        raw = yaml.safe_load(path.read_text()) or []
        entries = raw.get("entries") if isinstance(raw, dict) else raw
        for e in entries or []:
            cid = e.get("id")
            if not cid:
                continue
            out.setdefault(_normalize(cid), cid)
            for a in (e.get("aliases") or []):
                if isinstance(a, str) and "/" in a:
                    out.setdefault(_normalize(a), cid)

    _ingest(CORE_PATH)
    _ingest(MODELS_DEV_GENERATED)
    return out


def query_hub_stats(
    con: duckdb.DuckDBPyConnection,
    candidate_hf_ids: set[str],
) -> list[dict]:
    """Fetch hub-stats rows whose `id` is in `candidate_hf_ids`. Single
    pass — bounded by the size of our HF-aliased canonical set."""
    if not candidate_hf_ids:
        return []
    quoted = ", ".join(f"'{i.replace(chr(39), chr(39)*2)}'" for i in candidate_hf_ids)
    sql = f"""
        SELECT {QUERY_COLUMNS}
        FROM read_parquet('{PARQUET_URL}')
        WHERE id IN ({quoted})
    """
    cursor = con.execute(sql)
    cols = [d[0] for d in cursor.description]
    rows = cursor.fetchall()
    return [dict(zip(cols, r)) for r in rows]


def build_entry(
    row: dict,
    org_alias_map: dict[str, str],
    aliases_to_canonical: dict[str, str],
) -> Optional[dict]:
    """Construct a seed entry from a hub-stats row. Returns None when
    the row's HF id isn't in our existing aliases — backfill only
    operates on canonicals we already cover.

    All hub-stats-derived data fields (release_date, params_billions,
    open_weights, tags, metadata, parents, lineage_origin_org_id) come
    from `enrich_draft_from_row` so this script and the live-lookup
    path stay byte-identical on extraction. The seed loader's
    `_merge_into` unions parents by id across sources — generated
    parents from hub-stats compose with any curated parents in
    core.yaml rather than overriding them.
    """
    hf_id = row["id"]
    canonical_id, org_id = hf_id_to_canonical(hf_id, org_alias_map)
    norm_canon = _normalize(canonical_id)
    if norm_canon not in aliases_to_canonical:
        return None
    # Use the registry's actual canonical id (may differ in dot/dash from
    # our slugify of the HF id — e.g., `meta/llama-3.1-8b` vs `meta/llama-3-1-8b`).
    canonical_id = aliases_to_canonical[norm_canon]

    aliases = sorted({hf_id, _slugify(hf_id)})

    enrichment = enrich_draft_from_row(row, aliases_to_canonical, org_alias_map)

    # Decode tags from JSON-encoded string (helper output) back to a YAML
    # list. Loader accepts either form; list-form keeps generated YAML
    # diffs reviewable.
    if "tags" in enrichment:
        try:
            enrichment["tags"] = json.loads(enrichment["tags"])
        except (ValueError, TypeError):
            pass

    # Preserve the existing YAML key order so the diff after this change
    # is dominated by NEW fields (parents / lineage_origin_org_id) rather
    # than wholesale reformatting.
    entry: dict = {
        "id": canonical_id,
        "display_name": humanize_model_slug(hf_id),
        "org_id": org_id,
    }
    if "release_date" in enrichment:
        entry["release_date"] = enrichment["release_date"]
    if "tags" in enrichment:
        entry["tags"] = enrichment["tags"]
    entry["aliases"] = [a for a in aliases if a != canonical_id]
    if "metadata" in enrichment:
        entry["metadata"] = enrichment["metadata"]
    entry["review_status"] = "reviewed"
    if "params_billions" in enrichment:
        entry["params_billions"] = enrichment["params_billions"]
    if "open_weights" in enrichment:
        entry["open_weights"] = enrichment["open_weights"]
    if "parents" in enrichment:
        entry["parents"] = json.loads(enrichment["parents"])
    if "lineage_origin_org_id" in enrichment:
        entry["lineage_origin_org_id"] = enrichment["lineage_origin_org_id"]
    return entry


# ---------------------------------------------------------------------------
# Output writing.
# ---------------------------------------------------------------------------

_HEADER = """# Generated from cfahlgren1/hub-stats — DO NOT EDIT BY HAND.
# To update: run `python scripts/refresh_from_hub_stats.py`.
#
# Source: https://huggingface.co/datasets/cfahlgren1/hub-stats (Apache-2.0)
# Last refresh date is in git history.
#
# Backfill-only: every entry here MERGES into an existing canonical_models
# row at seed time. Fills in release_date, params_billions, license, tags,
# etc. via the seed loader's field-level merge (curated values in
# core.yaml take precedence on conflict).
#
# Lineage descendant pre-load (community quants/finetunes of our covered
# models) is deferred — see deferred task. EEE drafts get on-demand
# enrichment via the live hub-stats lookup at draft creation (Phase 2).
"""


def write_models_yaml(entries: list[dict]) -> None:
    MODELS_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(entries, sort_keys=False, allow_unicode=True, width=200)
    MODELS_OUT_PATH.write_text(_HEADER + "\n" + body)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="print summary; don't write files")
    p.add_argument("--force", action="store_true", help="ignore the etag watermark; re-fetch unconditionally")
    args = p.parse_args()

    org_alias_map = load_org_alias_to_canonical()
    aliases_to_canonical = load_existing_canonical_aliases()
    if not org_alias_map:
        print("[refresh] ERROR: seed/orgs.yaml is empty or missing.", file=sys.stderr)
        return 1
    if not aliases_to_canonical:
        print("[refresh] ERROR: no canonical models found in seed/. Seed first.", file=sys.stderr)
        return 1

    # Initial candidates: every HF-shaped alias on a known canonical.
    # We pass the original (non-normalized) alias to DuckDB since the
    # parquet `id` column carries case-sensitive original strings.
    initial: set[str] = set()
    for path in (CORE_PATH, MODELS_DEV_GENERATED):
        if not path.exists():
            continue
        raw = yaml.safe_load(path.read_text()) or []
        entries = raw.get("entries") if isinstance(raw, dict) else raw
        for e in entries or []:
            for a in (e.get("aliases") or []):
                if isinstance(a, str) and "/" in a:
                    initial.add(a)
            cid = e.get("id")
            if isinstance(cid, str) and "/" in cid:
                initial.add(cid)
    print(f"[refresh] HF-id candidates to look up: {len(initial)}", file=sys.stderr)

    # Etag short-circuit: if the parquet hasn't republished AND we've
    # already checked every current candidate against this etag AND the
    # YAML is still on disk, exit without doing any DuckDB work. Etag
    # fetch failure → fall through to unconditional re-fetch (degrades
    # to pre-watermark behaviour). The YAML-existence guard catches the
    # case where someone deleted the YAML but the state file survived;
    # without it we'd silently leave hub-stats enrichment missing from
    # the seed merge.
    current_etag = fetch_parquet_etag()
    state = load_state()
    can_short_circuit = (
        not args.force
        and not args.dry_run
        and current_etag is not None
        and state.get("parquet_etag") == current_etag
        and MODELS_OUT_PATH.exists()
        and initial.issubset(set(state.get("rows_checked_at_etag", [])))
    )
    if can_short_circuit:
        print(
            f"[refresh] no-op: parquet etag unchanged ({current_etag[:12]}...) "
            f"and all {len(initial)} candidates already checked",
            file=sys.stderr,
        )
        return 0

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    # Authenticate parquet fetches when HF_TOKEN is in the environment.
    # Unauth limit is 500 requests / 5min; one DuckDB read_parquet streams
    # the file via many range requests and can brush that ceiling on its
    # own. With auth the limit is ~30k / 5min.
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        escaped = hf_token.replace("'", "''")
        con.execute(
            f"CREATE SECRET hf_auth (TYPE HTTP, BEARER_TOKEN '{escaped}', "
            f"SCOPE 'https://huggingface.co');"
        )
    rows = query_hub_stats(con, initial)
    print(f"[refresh] hub-stats rows fetched: {len(rows)}", file=sys.stderr)

    entries: list[dict] = []
    for row in rows:
        e = build_entry(row, org_alias_map, aliases_to_canonical)
        if e is not None:
            entries.append(e)
    entries.sort(key=lambda e: e["id"])
    print(f"[refresh] enrichment entries: {len(entries)}", file=sys.stderr)

    if args.dry_run:
        print("[refresh] [dry-run] no files written")
        if entries:
            print("\nFirst 5 entries:")
            for e in entries[:5]:
                print(f"  {e['id']}  release_date={e.get('release_date')}  params={e.get('params_billions')}")
        return 0

    write_models_yaml(entries)
    print(f"[refresh] wrote {MODELS_OUT_PATH}", file=sys.stderr)

    # State write LAST: if anything above crashed, next run sees old
    # state and redoes the work — safe. Skip when etag fetch failed so
    # we don't pin a bogus watermark; next run will re-attempt cleanly.
    if current_etag is not None:
        write_state(current_etag, sorted(initial))
        print(f"[refresh] wrote {STATE_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
