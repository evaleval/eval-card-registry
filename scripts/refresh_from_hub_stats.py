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
creation (Phase 2, separate task) handles those when EEE actually
encounters them.

Output:
    seed/models/sources/hub_stats.generated.yaml — enrichment entries
    that merge into existing canonicals at seed time.

Usage:
    python scripts/refresh_from_hub_stats.py             # full run
    python scripts/refresh_from_hub_stats.py --dry-run   # preview only
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import duckdb
import yaml

# Shared hub-stats helpers live in the package so the runtime resolver
# (live lookup at draft creation) and this bulk refresh script stay
# consistent on row-shape parsing.
from eval_card_registry.services.hub_stats import (
    PARQUET_URL,
    QUERY_COLUMNS,
    approx_params_billions as _approx_params_billions,
    coerce_date as _coerce_date,
    extract_license as _extract_license,
    filter_useful_tags as _filter_useful_tags,
    hf_id_to_canonical,
    normalize as _normalize,
    slugify as _slugify,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ORGS_PATH = REPO_ROOT / "seed" / "orgs.yaml"
MODELS_OUT_PATH = REPO_ROOT / "seed" / "models" / "sources" / "hub_stats.generated.yaml"
CORE_PATH = REPO_ROOT / "seed" / "models" / "core.yaml"
MODELS_DEV_GENERATED = REPO_ROOT / "seed" / "models" / "sources" / "models_dev.generated.yaml"


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
    operates on canonicals we already cover."""
    hf_id = row["id"]
    canonical_id, org_id = hf_id_to_canonical(hf_id, org_alias_map)
    norm_canon = _normalize(canonical_id)
    if norm_canon not in aliases_to_canonical:
        return None
    # Use the registry's actual canonical id (may differ in dot/dash from
    # our slugify of the HF id — e.g., `meta/llama-3.1-8b` vs `meta/llama-3-1-8b`).
    canonical_id = aliases_to_canonical[norm_canon]

    aliases = sorted({hf_id, _slugify(hf_id)})

    metadata = {
        "source": "hub_stats",
        "hf_id": hf_id,
    }
    if row.get("downloadsAllTime") is not None:
        metadata["downloads_all_time"] = int(row["downloadsAllTime"])
    if row.get("likes") is not None:
        metadata["likes"] = int(row["likes"])
    if row.get("library_name"):
        metadata["library_name"] = row["library_name"]
    if row.get("pipeline_tag"):
        metadata["pipeline_tag"] = row["pipeline_tag"]
    license_str = _extract_license(row.get("cardData"))
    if license_str:
        metadata["license"] = license_str

    entry: dict = {
        "id": canonical_id,
        "display_name": hf_id.split("/", 1)[-1] if "/" in hf_id else hf_id,
        "org_id": org_id,
        "release_date": _coerce_date(row.get("createdAt")),
        "tags": _filter_useful_tags(row.get("tags")),
        "aliases": [a for a in aliases if a != canonical_id],
        "metadata": json.dumps(metadata, sort_keys=True),
        "review_status": "reviewed",
    }
    params = _approx_params_billions(row.get("safetensors"))
    if params is not None:
        entry["params_billions"] = params
    # Anything in hub-stats with downloadable artifacts is open weights.
    # Mirrors the inference in services/hub_stats.enrich_draft_from_row
    # so live + bulk paths agree.
    from eval_card_registry.services.hub_stats import has_downloadable_weights
    if has_downloadable_weights(row):
        entry["open_weights"] = True
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

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
