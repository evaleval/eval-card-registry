#!/usr/bin/env python3
"""
Build the local hub-stats id INDEX consulted by the read-only resolve path.

The registry resolves raw model strings to canonical entities. For an HF model
id that was never minted into the registry, the read-only API still wants to
CONFIRM that the exact id is a real HF repo (returning a `hub_stats_index`
confirmation in the standard ResolveResponse shape) — without minting or
persisting anything. This script materializes that index: one DuckDB `COPY`
over cfahlgren1/hub-stats, filtered to repos with downloadable weights.

Output:
    fixtures/hub_stats_index.parquet  (single file; published to the dataset's
    `hub_stats_index/part-0.parquet` subdir by the refresh-hub-stats-index cron)

Columns (mirror store/schemas.py `hub_stats_index`):
    id (HF-true), id_norm, release_date, pipeline_tag, params_billions,
    open_weights (always true), downloads

`id_norm` is computed in SQL to mirror `services.hub_stats.normalize` EXACTLY:
    lower(id) -> collapse runs of [/_.:-] to a single '-' -> trim leading/
    trailing '-'.

Reuses services/hub_stats.py PARQUET_URL / resolve_parquet_source and the
httpfs + HF_TOKEN CREATE SECRET pattern from scripts/freeze_hub_stats_cache.py.

Usage:
    HF_TOKEN=... uv run python scripts/build_hub_stats_index.py        # live pull
    HUB_STATS_LOCAL_PARQUET=path uv run python scripts/build_hub_stats_index.py --limit 50  # offline test
    uv run python scripts/build_hub_stats_index.py --source path/to.parquet --limit 50
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import duckdb

from eval_card_registry.services.hub_stats import PARQUET_URL, resolve_parquet_source

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "fixtures" / "hub_stats_index.parquet"

# A full pull must not shrink the live index — abort below this floor without
# writing. The full safetensors/gguf-filtered set is well over 500k rows.
ROW_FLOOR = 500_000

# id_norm mirrors services.hub_stats.normalize:
#   re.sub(r"[/_.:\-]+", "-", s.lower()).strip("-")
# DuckDB equivalent: lower -> regexp_replace collapses any run of separator
# chars [/_.:-] (mixed runs included) to a single dash -> trim leading/trailing
# dashes. Equivalence is asserted by tests/test_hub_stats_index_confirm.py.
_ID_NORM_SQL = "trim(regexp_replace(lower(id), '[/_.:-]+', '-', 'g'), '-')"

# The hub-stats models table is published by the HF datasets-server as MULTIPLE
# parquet shards (.../parquet/models/train/0.parquet, /1.parquet, ...). Reading
# only shard 0 would cover a fraction of the id space and silently starve the
# index, so we read ALL shards. The listing endpoint is the URL minus the shard
# filename.
_LISTING_URL = PARQUET_URL.rsplit("/", 1)[0]


def _live_shard_urls(token: str | None) -> list[str]:
    """All parquet shard URLs for the hub-stats models/train split, via the HF
    datasets-server parquet listing API. Falls back to the single committed
    PARQUET_URL on any error so the build still runs (the row-floor guard then
    catches an under-covered pull)."""
    import json
    import urllib.request

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        req = urllib.request.Request(_LISTING_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as r:
            payload = json.loads(r.read())
    except Exception as e:  # noqa: BLE001 — best-effort; fall back to one shard
        print(f"[index] WARN: shard listing failed ({e}); using single shard", file=sys.stderr)
        return [PARQUET_URL]
    # The API returns a JSON array of shard URLs (older shapes nest under a key).
    urls = payload if isinstance(payload, list) else (
        payload.get("parquet_files") or payload.get("urls") or []
    )
    shards = [u for u in urls if isinstance(u, str) and u.endswith(".parquet")]
    if not shards:
        print("[index] WARN: empty shard listing; using single shard", file=sys.stderr)
        return [PARQUET_URL]
    print(f"[index] {len(shards)} shard(s) enumerated", file=sys.stderr)
    return shards


def build_query(sources: list[str], limit: int | None) -> str:
    src_list = ", ".join("'" + s.replace("'", "''") + "'" for s in sources)
    body = f"""
    SELECT
        id,
        {_ID_NORM_SQL} AS id_norm,
        substr(CAST(createdAt AS VARCHAR), 1, 10) AS release_date,
        pipeline_tag,
        CAST(safetensors.total AS DOUBLE) / 1e9 AS params_billions,
        CAST(downloads AS BIGINT) AS downloads,
        true AS open_weights
    FROM read_parquet([{src_list}])
    WHERE safetensors IS NOT NULL OR gguf IS NOT NULL
"""
    if limit is not None:
        body = f"{body}\n    LIMIT {int(limit)}"
    return body


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=OUT_PATH)
    p.add_argument(
        "--source",
        type=str,
        default=None,
        help="Override parquet source (offline path). Defaults to "
        "$HUB_STATS_LOCAL_PARQUET if set, else the live hub-stats URL.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap rows (for offline testing only; bypasses the row floor).",
    )
    args = p.parse_args()

    src = args.source or resolve_parquet_source(PARQUET_URL)
    is_live = src == PARQUET_URL
    tok = os.environ.get("HF_TOKEN")

    con = duckdb.connect()
    if is_live:
        con.execute("INSTALL httpfs; LOAD httpfs;")
        if tok:
            con.execute(
                f"CREATE SECRET hf_auth (TYPE HTTP, "
                f"BEARER_TOKEN '{tok.replace(chr(39), chr(39) * 2)}', "
                f"SCOPE 'https://huggingface.co');"
            )
        sources = _live_shard_urls(tok)  # ALL shards, not just train/0.parquet
    else:
        sources = [src]  # offline: a single local parquet
    print(f"[index] reading {len(sources)} source(s); first: {sources[0]}", file=sys.stderr)

    query = build_query(sources, args.limit)

    # Count first so the floor check can abort WITHOUT writing a partial index.
    n = con.execute(f"SELECT count(*) FROM ({query})").fetchone()[0]
    print(f"[index] rows matched: {n}", file=sys.stderr)
    if args.limit is None and n < ROW_FLOOR:
        print(
            f"[index] ERROR: {n} rows < floor {ROW_FLOOR} — a partial pull must "
            f"not shrink the live index. Aborting (no file written).",
            file=sys.stderr,
        )
        return 1
    if n == 0:
        print("[index] ERROR: 0 rows matched — aborting (no file written).", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY ({query}) TO '{args.out}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    written = con.execute(f"SELECT count(*) FROM read_parquet('{args.out}')").fetchone()[0]
    print(f"[index] wrote {written} rows -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
