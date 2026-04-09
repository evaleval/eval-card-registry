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


_TABLE_NAMES = [
    "canonical_models",
    "canonical_benchmarks",
    "canonical_metrics",
    "eval_harnesses",
    "aliases",
    "resolution_log",
    "eval_results",
    "sync_runs",
]


class RegistryStore:
    """Holds all tables in memory. Single instance per process."""

    def __init__(self) -> None:
        self._tables: dict[str, pd.DataFrame] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load all tables from HF Hub or local fixtures."""
        if _local_mode():
            self._load_from_fixtures(_fixtures_path())
        else:
            self._load_from_hf(settings.hf_dataset_repo)
        self._loaded = True

    def _load_from_fixtures(self, path: Path) -> None:
        for table in _TABLE_NAMES:
            p = path / f"{table}.parquet"
            if p.exists():
                self._tables[table] = pd.read_parquet(p)
            else:
                self._tables[table] = schemas.empty(table)

    def _load_from_hf(self, repo_id: str) -> None:
        from huggingface_hub import hf_hub_download

        for table in _TABLE_NAMES:
            try:
                local = hf_hub_download(
                    repo_id=repo_id,
                    filename=f"{table}/part-0.parquet",
                    repo_type="dataset",
                    token=settings.hf_token or None,
                )
                self._tables[table] = pd.read_parquet(local)
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
