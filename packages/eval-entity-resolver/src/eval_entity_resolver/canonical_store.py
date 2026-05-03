"""Wraps the canonical entity tables (models / benchmarks / metrics /
harnesses / orgs) for the resolver to enrich its results with metadata
beyond just the matched canonical_id.

The resolver package is intentionally small — alias matching is its core
job. But callers consistently want richer return values: the matched
entity's `review_status`, parent edges, model-specific lineage fields,
quantized-chain root collapse. Putting that lookup logic here means any
caller of the bare `Resolver` gets the same response shape as the HTTP
API, without duplicating logic in the service wrapper.

Structure mirrors `AliasStore`: lazy loading from parquet/HF, an empty
fallback when the underlying file is missing, and read-only lookup
methods. Writes are out of scope — entity creation is a service-side
concern (the resolver doesn't auto-draft anything)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Per-entity-type parquet filenames (matches the eval-card-registry
# fixtures layout / HF Dataset config naming).
_TABLES = {
    "model": "canonical_models",
    "benchmark": "canonical_benchmarks",
    "metric": "canonical_metrics",
    "harness": "eval_harnesses",
    "org": "canonical_orgs",
}

# The `parent_*` column for each entity type that carries the
# in-family parent id (used for non-model types). Models use the typed
# `parents` JSON list instead — see `decode_parents`.
_PARENT_FIELD = {
    "benchmark": "parent_benchmark_id",
    "org": "parent_org_id",
}


# ---------------------------------------------------------------------------
# Helpers (pure, exported for reuse — service wrapper imports these)
# ---------------------------------------------------------------------------

def _is_na(value) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _na_to_none(value):
    return None if _is_na(value) else value


def decode_parents(value) -> list[dict]:
    """Decode `canonical_models.parents` (JSON-encoded list-of-edges) to a
    Python list. Tolerant of NA/NaN, None, empty strings, and pre-decoded
    lists. Returns [] for any unparseable input."""
    if _is_na(value) or value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s or s in ("[]", "null"):
            return []
        try:
            decoded = json.loads(s)
            return list(decoded) if isinstance(decoded, list) else []
        except (ValueError, TypeError):
            return []
    return []


def variant_parent_id(parents: list[dict]) -> Optional[str]:
    """Return the id of the first `variant` edge in a parents list, or
    None. The ResolutionResult `parent_canonical_id` field exposes the
    family / variant hierarchy that callers historically read off
    `parent_model_id`."""
    for edge in parents:
        if isinstance(edge, dict) and edge.get("relationship") == "variant":
            pid = edge.get("id")
            if pid:
                return pid
    return None


# ---------------------------------------------------------------------------
# CanonicalStore
# ---------------------------------------------------------------------------

class CanonicalStore:
    """Read-only access to the canonical entity tables. Holds one
    DataFrame per entity type; provides `lookup(entity_type, id)` for
    O(1) row retrieval. Empty tables are valid — `lookup` just returns
    None."""

    def __init__(
        self,
        models_df: Optional[pd.DataFrame] = None,
        benchmarks_df: Optional[pd.DataFrame] = None,
        metrics_df: Optional[pd.DataFrame] = None,
        harnesses_df: Optional[pd.DataFrame] = None,
        orgs_df: Optional[pd.DataFrame] = None,
    ) -> None:
        self._tables: dict[str, pd.DataFrame] = {
            "model": models_df if models_df is not None else pd.DataFrame(),
            "benchmark": benchmarks_df if benchmarks_df is not None else pd.DataFrame(),
            "metric": metrics_df if metrics_df is not None else pd.DataFrame(),
            "harness": harnesses_df if harnesses_df is not None else pd.DataFrame(),
            "org": orgs_df if orgs_df is not None else pd.DataFrame(),
        }
        # Per-table id-indexed cache for O(1) lookups
        self._index: dict[str, dict[str, dict]] = {}

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_parquet(cls, path: str | Path) -> "CanonicalStore":
        """Load all five canonical tables from `<path>/<table>.parquet`.
        Missing files become empty tables — matches the AliasStore
        fallback so a partial fixtures directory still works."""
        p = Path(path)
        kwargs: dict[str, pd.DataFrame] = {}
        for entity_type, fname in _TABLES.items():
            file = p / f"{fname}.parquet"
            if not file.exists():
                logger.info(
                    "CanonicalStore.from_parquet: %s not found; using empty table",
                    file,
                )
                continue
            try:
                df = pd.read_parquet(file)
            except (OSError, ValueError) as exc:
                logger.warning(
                    "CanonicalStore.from_parquet: failed to read %s (%s: %s); "
                    "falling back to empty table",
                    file, type(exc).__name__, exc,
                )
                continue
            kwargs[f"{entity_type}s_df" if entity_type != "harness" else "harnesses_df"] = df
        return cls(**kwargs)

    @classmethod
    def from_hf(cls, repo_id: str) -> "CanonicalStore":
        """Load all five canonical tables from a HF Dataset repo. Each
        table lives at `<table>/part-0.parquet`. Missing tables fall
        back to empty (matches AliasStore's behavior)."""
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import (
            EntryNotFoundError,
            HfHubHTTPError,
            RepositoryNotFoundError,
        )

        kwargs: dict[str, pd.DataFrame] = {}
        for entity_type, fname in _TABLES.items():
            try:
                local = hf_hub_download(
                    repo_id=repo_id,
                    filename=f"{fname}/part-0.parquet",
                    repo_type="dataset",
                )
                df = pd.read_parquet(local)
            except (
                RepositoryNotFoundError,
                EntryNotFoundError,
                HfHubHTTPError,
                FileNotFoundError,
                OSError,
                ValueError,
            ) as exc:
                logger.warning(
                    "CanonicalStore.from_hf: failed to load %s from %r (%s: %s); "
                    "using empty table",
                    fname, repo_id, type(exc).__name__, exc,
                )
                continue
            kwargs[f"{entity_type}s_df" if entity_type != "harness" else "harnesses_df"] = df
        return cls(**kwargs)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def _ensure_index(self, entity_type: str) -> dict[str, dict]:
        if entity_type in self._index:
            return self._index[entity_type]
        df = self._tables.get(entity_type)
        idx: dict[str, dict] = {}
        if df is not None and not df.empty and "id" in df.columns:
            for _, row in df.iterrows():
                cid = row["id"]
                if isinstance(cid, str):
                    idx[cid] = {k: _na_to_none(v) for k, v in row.items()}
        self._index[entity_type] = idx
        return idx

    def lookup(self, entity_type: str, canonical_id: str) -> Optional[dict]:
        """Return the canonical row as a dict (with NaN coerced to None),
        or None when the id isn't present. O(1)."""
        if not canonical_id:
            return None
        return self._ensure_index(entity_type).get(canonical_id)

    # ------------------------------------------------------------------
    # Enrichment — used by `Resolver` to populate the rich response
    # fields. Pure functions of (entity, optional root entity); no
    # access to any state outside what's passed in.
    # ------------------------------------------------------------------

    def parent_canonical_id(
        self, entity_type: str, entity: Optional[dict]
    ) -> Optional[str]:
        """Family/variant parent id. For models: the first `variant` edge
        in the typed parents list. For benchmarks/orgs: the
        `parent_*_id` scalar column."""
        if not entity:
            return None
        if entity_type == "model":
            return variant_parent_id(decode_parents(entity.get("parents")))
        field = _PARENT_FIELD.get(entity_type)
        if not field:
            return None
        return _na_to_none(entity.get(field))

    def model_metadata_fields(
        self, matched_canonical_id: str, matched_entity: Optional[dict]
    ) -> dict:
        """Compute the model-specific response fields. When the matched
        canonical has `root_model_id` set, the returned `canonical_id`
        is the identity root (= unquantized base for quantization
        chains); otherwise it's the matched leaf itself.
        `resolved_leaf_id` always carries the originally-matched id so
        callers wanting leaf-level data can opt in.

        All metadata fields (`open_weights`, `release_date`,
        `params_billions`, etc.) are sourced from the *returned*
        canonical's row — keeping the response internally consistent.
        Quantization preserves identity, so root metadata describes the
        same model the response identifies."""
        if not matched_entity:
            return {
                "canonical_id": matched_canonical_id,
                "resolved_leaf_id": matched_canonical_id,
                "root_model_id": None,
                "lineage_origin_org_id": None,
                "parents": None,
                "open_weights": None,
                "release_date": None,
                "params_billions": None,
            }

        leaf_root = _na_to_none(matched_entity.get("root_model_id"))
        parents_decoded = decode_parents(matched_entity.get("parents")) or None

        if leaf_root:
            root_entity = self.lookup("model", leaf_root) or {}
            return {
                "canonical_id": leaf_root,
                "resolved_leaf_id": matched_canonical_id,
                "root_model_id": leaf_root,
                "lineage_origin_org_id": _na_to_none(root_entity.get("lineage_origin_org_id")),
                "parents": parents_decoded,
                "open_weights": _na_to_none(root_entity.get("open_weights")),
                "release_date": _na_to_none(root_entity.get("release_date")),
                "params_billions": _na_to_none(root_entity.get("params_billions")),
            }

        return {
            "canonical_id": matched_canonical_id,
            "resolved_leaf_id": matched_canonical_id,
            "root_model_id": None,
            "lineage_origin_org_id": _na_to_none(matched_entity.get("lineage_origin_org_id")),
            "parents": parents_decoded,
            "open_weights": _na_to_none(matched_entity.get("open_weights")),
            "release_date": _na_to_none(matched_entity.get("release_date")),
            "params_billions": _na_to_none(matched_entity.get("params_billions")),
        }
