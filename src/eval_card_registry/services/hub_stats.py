"""
Shared hub-stats integration: helpers for normalizing HF metadata, plus
a `HubStatsClient` that wraps a DuckDB connection for single-id lookups
against the `cfahlgren1/hub-stats` parquet.

Used by:
  - `scripts/refresh_from_hub_stats.py` — bulk backfill of existing
    canonicals' metadata
  - `services/resolution_service.py` — on-demand enrichment of model
    drafts at auto-create time (Phase 2)

Both paths share the same row-shape parsing so behavior stays consistent
between the bulk pre-load and the live lookup.
"""
from __future__ import annotations

import json
import re
import threading
from datetime import datetime, date
from typing import Optional

PARQUET_URL = (
    "https://huggingface.co/api/datasets/cfahlgren1/hub-stats/parquet/models/train/0.parquet"
)

# Columns the queries fetch. Centralized so the bulk and lookup paths
# parse the same row shape.
QUERY_COLUMNS = (
    "id, author, createdAt, lastModified, downloads, downloadsAllTime, "
    "likes, trendingScore, tags, cardData, safetensors, baseModels, "
    "pipeline_tag, library_name"
)


# ---------------------------------------------------------------------------
# Slug + normalization helpers (mirror the resolver/promotion script style)
# ---------------------------------------------------------------------------

def normalize(s: str) -> str:
    """Lowercase + collapse separators (-/_./:) to single dashes."""
    return re.sub(r"[/_.:\-]+", "-", s.lower()).strip("-")


def slugify(value: str) -> str:
    """Lowercase, drop display-only punctuation, collapse whitespace and
    underscores to dashes. Preserves dots so canonical ids like
    `claude-opus-4.5` stay legible."""
    s = value.strip().lower()
    s = re.sub(r"[()\[\]{}]", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-")


def hf_id_to_canonical(
    hf_id: str,
    org_alias_map: dict[str, str],
) -> tuple[str, str]:
    """Convert an HF id like `'meta-llama/Llama-3.1-70B'` into our
    canonical form `('meta/llama-3.1-70b', 'meta')`. The HF org segment
    maps via `org_alias_map`; unknown orgs slug-normalize their author
    name and become their own org id (auto-org).

    Ids without `/` use `unknown` as the org placeholder — matches the
    resolver's convention for raw values without a dev prefix.
    """
    if "/" not in hf_id:
        return f"unknown/{slugify(hf_id)}", "unknown"
    org_part, name_part = hf_id.split("/", 1)
    org_norm = normalize(org_part)
    org_id = org_alias_map.get(org_norm, org_norm)
    name_slug = slugify(name_part)
    return f"{org_id}/{name_slug}", org_id


# ---------------------------------------------------------------------------
# Field extraction from a single hub-stats row
# ---------------------------------------------------------------------------

def coerce_date(value) -> Optional[str]:
    """Normalize createdAt to YYYY-MM-DD (drops time portion)."""
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.strftime("%Y-%m-%d") if hasattr(value, "strftime") else None
    if isinstance(value, str):
        return value[:10] if len(value) >= 10 else value
    return None


def has_downloadable_weights(row: dict) -> bool:
    """An HF model row in hub-stats has open weights iff it carries
    safetensors and/or gguf data — the formats users actually download.
    Models without either are typically inference-only API endpoints
    (rare in the HF index) or deleted/empty repos."""
    if row.get("safetensors"):
        return True
    gguf = row.get("gguf")
    return bool(gguf)


def approx_params_billions(safetensors) -> Optional[float]:
    """Estimate parameter count (in billions) from safetensors metadata.
    Total bytes / 2 assumes BF16 (= 2 bytes/param) as the dominant dtype.
    Approximate; consumers needing per-dtype precision should derive from
    the safetensors struct directly."""
    if safetensors is None:
        return None
    if isinstance(safetensors, dict):
        total = safetensors.get("total")
    else:
        total = getattr(safetensors, "total", None)
    if not total:
        return None
    return round(total / 2 / 1e9, 2) if total > 0 else None


def extract_license(card_data) -> Optional[str]:
    if not card_data:
        return None
    if isinstance(card_data, str):
        try:
            card_data = json.loads(card_data)
        except (ValueError, TypeError):
            return None
    if isinstance(card_data, dict):
        lic = card_data.get("license")
        return str(lic) if lic else None
    return None


def filter_useful_tags(raw_tags) -> list[str]:
    """Pick only tags worth surfacing on the canonical: language codes,
    eval-results, license markers, format markers. Drop noisy
    library/region/pipeline tags."""
    if not raw_tags:
        return []
    tags = list(raw_tags) if not isinstance(raw_tags, str) else [raw_tags]
    keep: list[str] = []
    for t in tags:
        t = str(t)
        if t == "eval-results":
            keep.append(t)
        elif t.startswith("license:"):
            keep.append(t)
        elif len(t) <= 3 and t.isalpha():  # language codes (en, zh, fr)
            keep.append(t)
        elif t in ("safetensors", "gguf"):
            keep.append(t)
    return sorted(set(keep))


def extract_base_models(base_models) -> list[dict]:
    """Decode the `baseModels` struct into a list of typed parent edges.
    Returns `[{id, relationship}, ...]` — caller resolves each id to our
    canonical via the alias index. Empty list when no baseModels."""
    if base_models is None:
        return []
    if isinstance(base_models, dict):
        relation = base_models.get("relation")
        models_list = base_models.get("models") or []
    else:
        relation = getattr(base_models, "relation", None)
        models_list = getattr(base_models, "models", None) or []
    if not relation:
        return []
    out: list[dict] = []
    for m in models_list:
        base_hf = m.get("id") if isinstance(m, dict) else getattr(m, "id", None)
        if base_hf:
            out.append({"id": base_hf, "relationship": relation})
    return out


# ---------------------------------------------------------------------------
# Live lookup client
# ---------------------------------------------------------------------------

class HubStatsClient:
    """Single-id lookup against the hub-stats parquet via DuckDB.

    Materializes the remote parquet into a local DuckDB table on first
    use, then serves all subsequent lookups against the local copy.
    Trade-off: one bulk fetch (large but a single HTTP transaction)
    instead of N per-id range-fetch queries (small but rate-limit-prone
    when N is in the thousands — `huggingface.co` 429s aggressively on
    the parquet API even with auth, and DuckDB's range-fetch pattern
    counts as multiple HTTP HEAD/GET requests per logical query).

    On bulk-fetch failure, falls back to per-id remote queries (older
    behaviour) so the client degrades gracefully rather than going
    silent. Per-id fallback retains the same in-process cache.

    Failure mode: any DuckDB error (network down, parquet schema drift,
    etc.) returns None from `lookup()` and is logged. Callers must
    handle None as "no enrichment data available" — never raise."""

    def __init__(self, parquet_url: str = PARQUET_URL) -> None:
        self.parquet_url = parquet_url
        self._con = None
        self._local_table_ready: bool = False
        self._local_table_failed: bool = False
        self._cache: dict[str, Optional[dict]] = {}
        self._lock = threading.Lock()

    def _ensure_con(self):
        if self._con is not None:
            return self._con
        # Import lazily so processes that never call lookup() don't pay
        # the duckdb import cost.
        import os
        import duckdb
        con = duckdb.connect()
        con.execute("INSTALL httpfs; LOAD httpfs;")
        # Authenticate parquet fetches when HF_TOKEN is in the environment
        # (typical on the deployed Space). Unauth limit is 500 req/5min;
        # one DuckDB read_parquet against the remote file streams via
        # several range requests and a sync that auto-creates many drafts
        # can brush that ceiling. With auth the ceiling is ~30k/5min.
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            escaped = hf_token.replace("'", "''")
            con.execute(
                f"CREATE SECRET hf_auth (TYPE HTTP, BEARER_TOKEN '{escaped}', "
                f"SCOPE 'https://huggingface.co');"
            )
        self._con = con
        return con

    def _ensure_local_table(self, con) -> bool:
        """Materialize the remote parquet into a local DuckDB table. Returns
        True when the table is queryable, False when the bulk fetch failed
        (caller should fall through to per-id remote query). Idempotent.

        Caller must pass an already-prepared connection so this method
        doesn't double-count `_ensure_con` invocations under tests that
        patch the latter as a call counter."""
        if self._local_table_ready:
            return True
        if self._local_table_failed:
            return False
        try:
            import logging
            log = logging.getLogger(__name__)
            log.info(
                "hub-stats: bulk-loading remote parquet into local table "
                "(one-time cost; avoids per-id rate limits)..."
            )
            # `id` is included implicitly via QUERY_COLUMNS. Materializing
            # only the columns we need keeps the local table small enough
            # to live in memory comfortably.
            con.execute(
                f"CREATE TABLE hub_stats AS "
                f"SELECT {QUERY_COLUMNS} FROM read_parquet('{self.parquet_url}')"
            )
            # Indexed lookup — DuckDB handles a single-column equality
            # filter on a string column efficiently without an explicit
            # index, but adding one explicitly costs ~ms and makes the
            # plan unambiguous to anyone reading EXPLAIN later.
            con.execute("CREATE INDEX hub_stats_id_idx ON hub_stats(id)")
            row_count = con.execute("SELECT COUNT(*) FROM hub_stats").fetchone()[0]
            log.info("hub-stats: local table loaded (%d rows)", row_count)
            self._local_table_ready = True
            return True
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "hub-stats: bulk load failed (%s: %s); falling back to "
                "per-id remote queries (rate-limit-prone)",
                type(exc).__name__, exc,
            )
            self._local_table_failed = True
            return False

    def close(self) -> None:
        if self._con is not None:
            try:
                self._con.close()
            except Exception:
                pass
            self._con = None
            self._local_table_ready = False
            self._local_table_failed = False

    def lookup(self, hf_id: str) -> Optional[dict]:
        """Query hub-stats for one HF id. Returns the row as a dict, or
        None if not found / on any error. Threadsafe + cached.

        Prefers the locally-materialized table (one bulk fetch upfront);
        falls back to a per-id remote query if the bulk fetch failed."""
        with self._lock:
            if hf_id in self._cache:
                return self._cache[hf_id]
        try:
            con = self._ensure_con()
            use_local = self._ensure_local_table(con)
            escaped = hf_id.replace("'", "''")
            if use_local:
                sql = f"SELECT * FROM hub_stats WHERE id = '{escaped}' LIMIT 1"
            else:
                sql = (
                    f"SELECT {QUERY_COLUMNS} "
                    f"FROM read_parquet('{self.parquet_url}') "
                    f"WHERE id = '{escaped}' LIMIT 1"
                )
            cursor = con.execute(sql)
            cols = [d[0] for d in cursor.description]
            row = cursor.fetchone()
            result = dict(zip(cols, row)) if row else None
        except Exception as e:
            # Log-and-continue: live lookup is best-effort enrichment,
            # never blocks draft creation.
            import logging
            logging.getLogger(__name__).warning(
                "hub-stats lookup failed for %r: %s", hf_id, e
            )
            result = None
        with self._lock:
            self._cache[hf_id] = result
        return result


# ---------------------------------------------------------------------------
# High-level enrichment: take a hub-stats row + the registry's alias index
# and return a dict of fields ready to merge into a draft canonical_models row.
# ---------------------------------------------------------------------------

def enrich_draft_from_row(
    row: dict,
    aliases_to_canonical: dict[str, str],
    org_alias_map: dict[str, str],
) -> dict:
    """Convert one hub-stats row into a partial canonical_models dict
    suitable for merging into an auto-created draft. Computes:
      - release_date (from createdAt)
      - params_billions (approx from safetensors)
      - tags (filtered list)
      - parents (resolved against our canonical alias index when possible)
      - lineage_origin_org_id (= upstream lab when a parent edge resolves)
      - metadata (license, downloads, etc.)

    Caller decides what to actually write — none of these fields are
    forced. Returns an empty dict if the row carries no usable info.
    """
    out: dict = {}

    release = coerce_date(row.get("createdAt"))
    if release:
        out["release_date"] = release

    params = approx_params_billions(row.get("safetensors"))
    if params is not None:
        out["params_billions"] = params

    # Anything we can pull from hub-stats with downloadable artifacts is
    # by definition open weights. Closed-API models aren't on HF.
    if has_downloadable_weights(row):
        out["open_weights"] = True

    useful_tags = filter_useful_tags(row.get("tags"))
    if useful_tags:
        out["tags"] = json.dumps(useful_tags)

    # Resolve baseModels parents to OUR canonical ids when possible.
    # Drop edges we can't resolve — a parents edge pointing at an HF id
    # we don't track would dangle.
    parents: list[dict] = []
    lineage_origin_org_id: Optional[str] = None
    for edge in extract_base_models(row.get("baseModels")):
        base_hf = edge["id"]
        n = normalize(base_hf)
        parent_canonical = aliases_to_canonical.get(n)
        if parent_canonical is None:
            # Try slugifying via org map (covers HF ids that aren't yet
            # in our aliases but whose org we know).
            slugified, _ = hf_id_to_canonical(base_hf, org_alias_map)
            parent_canonical = aliases_to_canonical.get(normalize(slugified))
        if parent_canonical:
            parents.append({"id": parent_canonical, "relationship": edge["relationship"]})
            # First resolved non-variant edge sets the lineage origin
            # (matches `derive_model_lineage_fields` semantics).
            if lineage_origin_org_id is None and edge["relationship"] != "variant":
                if "/" in parent_canonical:
                    lineage_origin_org_id = parent_canonical.split("/", 1)[0]
    if parents:
        out["parents"] = json.dumps(parents)
    if lineage_origin_org_id:
        out["lineage_origin_org_id"] = lineage_origin_org_id

    # Stash extra hub-stats context in metadata so consumers can find it
    # without re-querying.
    metadata: dict = {"source": "hub_stats", "hf_id": row.get("id")}
    if row.get("downloadsAllTime") is not None:
        metadata["downloads_all_time"] = int(row["downloadsAllTime"])
    if row.get("likes") is not None:
        metadata["likes"] = int(row["likes"])
    if row.get("library_name"):
        metadata["library_name"] = row["library_name"]
    if row.get("pipeline_tag"):
        metadata["pipeline_tag"] = row["pipeline_tag"]
    license_str = extract_license(row.get("cardData"))
    if license_str:
        metadata["license"] = license_str
    if len(metadata) > 2:  # has more than just source + hf_id
        out["metadata"] = json.dumps(metadata, sort_keys=True)

    return out
