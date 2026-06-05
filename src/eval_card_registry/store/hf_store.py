"""
In-memory store backed by HF Dataset parquet configs (or local fixtures in LOCAL_MODE).

All tables are loaded into memory on startup. Writes update memory immediately.
HF Hub push happens only at end of sync run (call push_to_hub()).
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from eval_card_registry.config import settings
from eval_card_registry.store import schemas


def _local_mode() -> bool:
    """Check LOCAL_MODE at call time, not import time, so CLI --local flag works."""
    return os.environ.get("LOCAL_MODE", "").lower() in ("true", "1") or settings.local_mode


def _fixtures_path() -> Path:
    return Path(os.environ.get("FIXTURES_PATH", settings.fixtures_path))


TABLE_NAMES = [
    "canonical_orgs",
    "canonical_models",
    "canonical_benchmarks",
    "canonical_families",
    "canonical_composites",
    "canonical_metrics",
    "eval_harnesses",
    "canonical_inference_platforms",
    "aliases",
    "resolution_log",
    "eval_results",
    "sync_runs",
]


def _reconcile_schema(table: str, df: pd.DataFrame) -> pd.DataFrame:
    """Bring a freshly-loaded parquet into line with the current schema:
      * add missing columns (NA-filled, correct dtype)
      * drop columns no longer in the schema
      * apply one-shot data migrations (e.g. legacy `parent_model_id`
        scalar → typed `parents` JSON list).

    The data migration shims here are intentionally narrow and one-way: the
    parquet file gets re-saved with the new shape on the next sync, after
    which the shim becomes a no-op. Keep them around at least one release
    cycle so HF-deployed read-only Spaces survive a schema rollout.
    """
    import json as _json

    expected = schemas._SCHEMAS[table]
    df = df.copy()

    # canonical_models: legacy `parent_model_id` (scalar) → `parents` (JSON list)
    if table == "canonical_models" and "parent_model_id" in df.columns and "parents" not in df.columns:
        def _to_parents(legacy):
            try:
                if pd.isna(legacy):
                    return None
            except (TypeError, ValueError):
                pass
            if legacy is None or legacy == "":
                return None
            return _json.dumps([{"id": str(legacy), "relationship": "variant", "axis": "size"}])
        df["parents"] = df["parent_model_id"].map(_to_parents).astype(pd.StringDtype())

    # Add any other missing columns as NA with the schema's dtype.
    # Use `None` (not `pd.NA`) as the fill value: pandas 3.x rejects pd.NA
    # when constructing a float64 Series, but None coerces correctly to
    # both nullable-string (NA) and float64 (NaN).
    for col, dtype in expected.items():
        if col not in df.columns:
            df[col] = pd.Series([None] * len(df), dtype=dtype)

    # Drop columns not in the schema (e.g. legacy `parent_model_id` after the
    # migration above). Order columns to match the schema for deterministic
    # parquet output.
    keep = [c for c in expected if c in df.columns]
    return df[keep]

# Tables needed for query-only (read-only) mode
QUERY_TABLE_NAMES = [
    "canonical_orgs",
    "canonical_models",
    "canonical_benchmarks",
    "canonical_families",
    "canonical_composites",
    "canonical_metrics",
    "eval_harnesses",
    "canonical_inference_platforms",
    "aliases",
]


class RegistryStore:
    """Holds all tables in memory. Single instance per process."""

    def __init__(self) -> None:
        self._tables: dict[str, pd.DataFrame] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self, tables: list[str] | None = None) -> None:
        """Load tables from HF Hub or local fixtures.

        Args:
            tables: Specific table names to load. If None, loads all tables.
        """
        names = tables or TABLE_NAMES
        if _local_mode():
            self._load_from_fixtures(_fixtures_path(), names)
        else:
            self._load_from_hf(settings.hf_dataset_repo, names)
        self._loaded = True

    def _load_from_fixtures(self, path: Path, names: list[str]) -> None:
        for table in names:
            p = path / f"{table}.parquet"
            if p.exists():
                self._tables[table] = _reconcile_schema(table, pd.read_parquet(p))
            else:
                self._tables[table] = schemas.empty(table)

    def _load_from_hf(self, repo_id: str, names: list[str]) -> None:
        from huggingface_hub import hf_hub_download

        for table in names:
            try:
                local = hf_hub_download(
                    repo_id=repo_id,
                    filename=f"{table}/part-0.parquet",
                    repo_type="dataset",
                    token=settings.hf_token or None,
                )
                self._tables[table] = _reconcile_schema(table, pd.read_parquet(local))
            except Exception:
                self._tables[table] = schemas.empty(table)

    def push_to_hub(self) -> None:
        """Push all in-memory tables to HF Hub. Called at end of sync run only."""
        if _local_mode():
            self._flush_to_fixtures(_fixtures_path())
            return

        import tempfile
        from huggingface_hub import HfApi

        api = HfApi(token=settings.hf_token or None)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            for table, df in self._tables.items():
                p = tmp / f"{table}.parquet"
                df.to_parquet(p, index=False)
                api.upload_file(
                    path_or_fileobj=str(p),
                    path_in_repo=f"{table}/part-0.parquet",
                    repo_id=settings.hf_dataset_repo,
                    repo_type="dataset",
                    token=settings.hf_token or None,
                )

    def _flush_to_fixtures(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        for table, df in self._tables.items():
            df.to_parquet(path / f"{table}.parquet", index=False)

    # ------------------------------------------------------------------
    # Table access
    # ------------------------------------------------------------------

    def table(self, name: str) -> pd.DataFrame:
        return self._tables[name]

    def has_table(self, name: str) -> bool:
        return name in self._tables

    def set_table(self, name: str, df: pd.DataFrame) -> None:
        self._tables[name] = df

    @property
    def loaded(self) -> bool:
        return self._loaded


# Module-level singleton
_store: Optional[RegistryStore] = None


def get_store() -> RegistryStore:
    global _store
    if _store is None:
        _store = RegistryStore()
    return _store
