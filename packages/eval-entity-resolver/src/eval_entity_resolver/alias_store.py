from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


_SCHEMA = {
    "id": pd.StringDtype(),
    "raw_value": pd.StringDtype(),
    "entity_type": pd.StringDtype(),
    "canonical_id": pd.StringDtype(),
    "source_config": pd.StringDtype(),
    "source_field": pd.StringDtype(),
    "status": pd.StringDtype(),
    "strategy": pd.StringDtype(),
    "confidence": "float64",
    "notes": pd.StringDtype(),
    "created_at": pd.StringDtype(),
    "updated_at": pd.StringDtype(),
}


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in _SCHEMA.items()})


class AliasStore:
    """Wraps the aliases table. Loaded into memory; writes are in-memory only."""

    def __init__(self, df: pd.DataFrame, read_only: bool = False) -> None:
        self._df = df.copy()
        self.read_only = read_only
        # Per-entity_type caches — built lazily on first access
        self._normalized_cache: dict[str, dict[str, str]] = {}
        self._candidates_cache: dict[str, list[tuple[str, str]]] = {}
        self._lookup_index: dict[tuple[str, str, Optional[str]], str] | None = None

    def _ensure_lookup_index(self) -> None:
        """Build a dict index for O(1) exact lookups."""
        if self._lookup_index is not None:
            return
        self._lookup_index = {}
        df = self._df[self._df["status"] != "rejected"]
        for _, row in df.iterrows():
            key = (row["raw_value"], row["entity_type"], row.get("source_config"))
            self._lookup_index[key] = row["canonical_id"]

    def _invalidate_caches(self) -> None:
        self._normalized_cache.clear()
        self._candidates_cache.clear()
        self._lookup_index = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_parquet(cls, path: str | Path, read_only: bool = False) -> "AliasStore":
        p = Path(path) / "aliases.parquet"
        if p.exists():
            df = pd.read_parquet(p)
        else:
            df = _empty_df()
        return cls(df, read_only=read_only)

    @classmethod
    def from_hf(cls, repo_id: str, read_only: bool = False) -> "AliasStore":
        from huggingface_hub import hf_hub_download

        try:
            local = hf_hub_download(
                repo_id=repo_id,
                filename="aliases/part-0.parquet",
                repo_type="dataset",
            )
            df = pd.read_parquet(local)
        except Exception:
            df = _empty_df()
        return cls(df, read_only=read_only)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(
        self,
        raw_value: str,
        entity_type: str,
        source_config: Optional[str],
    ) -> Optional[str]:
        """Return canonical_id for first non-rejected match. Config-scoped before global."""
        self._ensure_lookup_index()
        # Config-scoped
        if source_config:
            result = self._lookup_index.get((raw_value, entity_type, source_config))
            if result is not None:
                return result
        # Global
        return self._lookup_index.get((raw_value, entity_type, None))

    # ------------------------------------------------------------------
    # Writes (in-memory only; caller is responsible for persistence)
    # ------------------------------------------------------------------

    def add_alias(
        self,
        raw_value: str,
        entity_type: str,
        canonical_id: str,
        source_config: Optional[str],
        source_field: Optional[str],
        status: str,
        strategy: str,
        confidence: float,
    ) -> None:
        if self.read_only:
            raise RuntimeError("AliasStore is read-only")
        now = datetime.now(timezone.utc).isoformat()
        row = {
            "id": str(uuid.uuid4()),
            "raw_value": raw_value,
            "entity_type": entity_type,
            "canonical_id": canonical_id,
            "source_config": source_config,
            "source_field": source_field,
            "status": status,
            "strategy": strategy,
            "confidence": confidence,
            "notes": None,
            "created_at": now,
            "updated_at": now,
        }
        self._df = pd.concat([self._df, pd.DataFrame([row])], ignore_index=True)
        self._invalidate_caches()

    def update_alias(
        self,
        raw_value: str,
        entity_type: str,
        source_config: Optional[str],
        canonical_id: str,
        status: str,
        strategy: str,
        confidence: float,
    ) -> None:
        """Upsert: update existing alias row or add new one."""
        if self.read_only:
            raise RuntimeError("AliasStore is read-only")
        df = self._df
        mask = (df["raw_value"] == raw_value) & (df["entity_type"] == entity_type)
        if source_config:
            mask = mask & (df["source_config"] == source_config)
        else:
            mask = mask & df["source_config"].isna()
        if mask.any():
            now = datetime.now(timezone.utc).isoformat()
            self._df.loc[mask, "canonical_id"] = canonical_id
            self._df.loc[mask, "status"] = status
            self._df.loc[mask, "strategy"] = strategy
            self._df.loc[mask, "confidence"] = confidence
            self._df.loc[mask, "updated_at"] = now
            self._invalidate_caches()
        else:
            self.add_alias(raw_value, entity_type, canonical_id, source_config, None, status, strategy, confidence)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        return self._df.copy()

    def get_normalized_lookup(self, entity_type: str) -> dict[str, str]:
        """Return {normalized_raw_value: canonical_id} for use by strategies. Cached per entity_type."""
        if entity_type in self._normalized_cache:
            return self._normalized_cache[entity_type]

        from eval_entity_resolver.normalization import normalize

        df = self._df[(self._df["entity_type"] == entity_type) & (self._df["status"] != "rejected")]
        result = {}
        for _, row in df.iterrows():
            result[normalize(row["raw_value"])] = row["canonical_id"]
        self._normalized_cache[entity_type] = result
        return result

    def get_all_for_type(self, entity_type: str) -> list[tuple[str, str]]:
        """Return [(raw_value, canonical_id)] for all non-rejected aliases of entity_type. Cached."""
        if entity_type in self._candidates_cache:
            return self._candidates_cache[entity_type]

        df = self._df[(self._df["entity_type"] == entity_type) & (self._df["status"] != "rejected")]
        result = list(zip(df["raw_value"].tolist(), df["canonical_id"].tolist()))
        self._candidates_cache[entity_type] = result
        return result
