"""
Read/write helpers for entity tables, aliases, and logs.
All operate on in-memory DataFrames from RegistryStore.

Performance note: write helpers that are called in tight loops (add_alias,
upsert_eval_result, append_resolution_log) accumulate rows in a pending
buffer on the store.  Call flush_pending(store) once at the end of a sync
to apply all buffered writes in a single pd.concat per table.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from eval_card_registry.store.hf_store import RegistryStore
from eval_card_registry.store import schemas


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_na(value) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _source_config_key(value) -> Optional[str]:
    """Normalize nullable source_config values for alias-index keys."""
    return None if _is_na(value) else value


def _row_to_dict(row: pd.Series) -> dict:
    """Convert a Series to dict, coercing pandas NA/NaN/NaT to None for JSON.
    Uses Series.to_dict() so numpy scalars are unboxed to Python types."""
    return {k: (None if _is_na(v) else v) for k, v in row.to_dict().items()}


def _records(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to list-of-dicts, coercing NA/NaN to None for JSON."""
    if df.empty:
        return []
    return df.astype(object).mask(df.isna(), None).to_dict(orient="records")


# ------------------------------------------------------------------
# Pending-row buffer  (avoids O(n²) pd.concat-per-row)
# ------------------------------------------------------------------

def _get_pending(store: RegistryStore, table: str) -> list[dict]:
    """Return the pending-row list for *table*, creating it if needed."""
    if not hasattr(store, "_pending"):
        store._pending = {}
    return store._pending.setdefault(table, [])


def flush_pending(store: RegistryStore) -> None:
    """Concat all buffered rows into their respective tables in one shot."""
    pending = getattr(store, "_pending", {})
    for table, rows in pending.items():
        if not rows:
            continue
        df = store.table(table)
        new_df = pd.DataFrame(rows)
        df = pd.concat([df, new_df], ignore_index=True)
        store.set_table(table, df)
    store._pending = {}


# ------------------------------------------------------------------
# Generic entity helpers
# ------------------------------------------------------------------

def get_entity(store: RegistryStore, table: str, entity_id: str) -> Optional[dict]:
    df = store.table(table)
    row = df[df["id"] == entity_id]
    if row.empty:
        # Check pending rows too
        for pending_row in _get_pending(store, table):
            if pending_row.get("id") == entity_id:
                return pending_row
        return None
    return _row_to_dict(row.iloc[0])


def list_entities(
    store: RegistryStore,
    table: str,
    search: Optional[str] = None,
    review_status: Optional[str] = None,
    **filters: Any,
) -> list[dict]:
    df = store.table(table)
    if search:
        mask = df["id"].str.contains(search, case=False, na=False)
        if "display_name" in df.columns:
            mask = mask | df["display_name"].str.contains(search, case=False, na=False)
        df = df[mask]
    if review_status:
        df = df[df["review_status"] == review_status]
    for col, val in filters.items():
        if col in df.columns and val is not None:
            df = df[df[col] == val]
    return _records(df)


def upsert_entity(store: RegistryStore, table: str, data: dict, buffered: bool = False) -> dict:
    """Insert or update an entity row. `data` must contain `id`.
    If buffered=True, new rows go to the pending buffer (flushed by flush_pending).
    """
    df = store.table(table)
    entity_id = data["id"]
    now = _now()
    existing = df[df["id"] == entity_id]

    if existing.empty:
        # Check pending too
        pending = _get_pending(store, table)
        for p in pending:
            if p.get("id") == entity_id:
                p.update({k: v for k, v in data.items() if k != "id"})
                p["updated_at"] = now
                return p
        row = {**data, "created_at": now, "updated_at": now}
        if buffered:
            pending.append(row)
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            store.set_table(table, df)
        return row
    else:
        for col, val in data.items():
            if col != "id" and col in df.columns:
                df.loc[df["id"] == entity_id, col] = val
        df.loc[df["id"] == entity_id, "updated_at"] = now
        store.set_table(table, df)
        return _row_to_dict(df[df["id"] == entity_id].iloc[0])


# ------------------------------------------------------------------
# Alias helpers
# ------------------------------------------------------------------

# In-memory index for fast alias lookups during sync.
# Key: (entity_type, raw_value, source_config_or_None) → dict
_alias_index: dict[tuple, dict] = {}


def _rebuild_alias_index(store: RegistryStore) -> None:
    """Rebuild the in-memory alias index from the aliases table."""
    global _alias_index
    _alias_index = {}
    df = store.table("aliases")
    for _, row in df.iterrows():
        if row.get("status") != "rejected":
            row_dict = _row_to_dict(row)
            key = (
                row_dict["entity_type"],
                row_dict["raw_value"],
                _source_config_key(row_dict.get("source_config")),
            )
            _alias_index[key] = row_dict
    # Also index pending aliases
    for pending_row in _get_pending(store, "aliases"):
        if pending_row.get("status") != "rejected":
            key = (
                pending_row["entity_type"],
                pending_row["raw_value"],
                _source_config_key(pending_row.get("source_config")),
            )
            _alias_index[key] = pending_row


def get_alias(
    store: RegistryStore,
    raw_value: str,
    entity_type: str,
    source_config: Optional[str],
) -> Optional[dict]:
    source_config = _source_config_key(source_config)
    # Fast path: use index if available
    if _alias_index:
        if source_config:
            scoped = _alias_index.get((entity_type, raw_value, source_config))
            if scoped:
                return scoped
        global_ = _alias_index.get((entity_type, raw_value, None))
        if global_:
            return global_
        return None

    # Slow path: scan DataFrame
    df = store.table("aliases")
    mask = (
        (df["raw_value"] == raw_value)
        & (df["entity_type"] == entity_type)
        & (df["status"] != "rejected")
    )
    if source_config:
        scoped = df[mask & (df["source_config"] == source_config)]
        if not scoped.empty:
            return _row_to_dict(scoped.iloc[0])
    global_ = df[mask & df["source_config"].isna()]
    if not global_.empty:
        return _row_to_dict(global_.iloc[0])
    return None


def add_alias(store: RegistryStore, data: dict, buffered: bool = False) -> dict:
    """
    Insert a new alias row. Enforces uniqueness on (entity_type, raw_value, source_config).
    Raises ValueError if a non-rejected alias already exists for that key.

    If buffered=True, the row is added to the pending buffer (flushed by flush_pending).
    Otherwise it is written immediately to the DataFrame.
    """
    raw_value = data["raw_value"]
    entity_type = data["entity_type"]
    source_config = _source_config_key(data.get("source_config"))
    key = (entity_type, raw_value, source_config)

    # Check uniqueness via index if available
    if _alias_index and key in _alias_index:
        raise ValueError(
            f"Alias already exists for ({entity_type!r}, {raw_value!r}, source_config={source_config!r}). "
            "Use update_alias() to modify an existing alias."
        )

    # Check DataFrame
    df = store.table("aliases")
    mask = (
        (df["raw_value"] == raw_value)
        & (df["entity_type"] == entity_type)
        & (df["status"] != "rejected")
    )
    if source_config is not None:
        mask = mask & (df["source_config"] == source_config)
    else:
        mask = mask & df["source_config"].isna()
    if mask.any():
        raise ValueError(
            f"Alias already exists for ({entity_type!r}, {raw_value!r}, source_config={source_config!r}). "
            "Use update_alias() to modify an existing alias."
        )

    # Check pending buffer
    for p in _get_pending(store, "aliases"):
        if (p["entity_type"] == entity_type and p["raw_value"] == raw_value
                and _source_config_key(p.get("source_config")) == source_config
                and p.get("status") != "rejected"):
            raise ValueError(
                f"Alias already exists for ({entity_type!r}, {raw_value!r}, source_config={source_config!r}). "
                "Use update_alias() to modify an existing alias."
            )

    now = _now()
    row = {
        **data,
        "source_config": source_config,
        "id": str(uuid.uuid4()),
        "created_at": now,
        "updated_at": now,
    }

    if buffered:
        _get_pending(store, "aliases").append(row)
    else:
        df = store.table("aliases")  # re-read in case it changed
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        store.set_table("aliases", df)

    # Update index only if it has already been built. If it is empty, get_alias
    # should keep using the DataFrame/pending slow path instead of a partial index.
    if _alias_index and row.get("status") != "rejected":
        _alias_index[key] = row
    return row


def update_alias(store: RegistryStore, alias_id: str, updates: dict) -> Optional[dict]:
    df = store.table("aliases")
    if not (df["id"] == alias_id).any():
        return None
    for col, val in updates.items():
        if col in df.columns:
            df.loc[df["id"] == alias_id, col] = val
    df.loc[df["id"] == alias_id, "updated_at"] = _now()
    store.set_table("aliases", df)
    updated = _row_to_dict(df[df["id"] == alias_id].iloc[0])
    # Keep the in-memory index in sync if it was built — otherwise a follow-up
    # add_alias() / get_alias() would see stale canonical data for this key.
    if _alias_index:
        key = (
            updated["entity_type"],
            updated["raw_value"],
            _source_config_key(updated.get("source_config")),
        )
        if updated.get("status") != "rejected":
            _alias_index[key] = updated
        else:
            _alias_index.pop(key, None)
    return updated


# ------------------------------------------------------------------
# Eval results (mapping table: one row per EEE evaluation result)
# ------------------------------------------------------------------

def _eval_result_id(evaluation_id: str, result_index: int) -> str:
    """Deterministic ID from evaluation_id + result_index."""
    key = f"{evaluation_id}:{result_index}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# Track IDs already in pending buffer to detect upsert-vs-insert
_pending_result_ids: set[str] = set()


def upsert_eval_result(store: RegistryStore, data: dict) -> dict:
    """Insert or update an eval_results row. Uses deterministic ID from evaluation_id + result_index."""
    row_id = _eval_result_id(data["evaluation_id"], data["result_index"])
    now = _now()

    # Check if already in pending buffer
    if row_id in _pending_result_ids:
        pending = _get_pending(store, "eval_results")
        for p in pending:
            if p["id"] == row_id:
                for col, val in data.items():
                    if col != "id":
                        p[col] = val
                p["updated_at"] = now
                return p

    # Check committed table
    df = store.table("eval_results")
    existing = df[df["id"] == row_id]
    if not existing.empty:
        for col, val in data.items():
            if col != "id" and col in df.columns:
                df.loc[df["id"] == row_id, col] = val
        df.loc[df["id"] == row_id, "updated_at"] = now
        store.set_table("eval_results", df)
        return _row_to_dict(df[df["id"] == row_id].iloc[0])

    # New row — buffer it
    row = {**data, "id": row_id, "created_at": now, "updated_at": now}
    _get_pending(store, "eval_results").append(row)
    _pending_result_ids.add(row_id)
    return row


def get_eval_results(
    store: RegistryStore,
    model_id: Optional[str] = None,
    benchmark_id: Optional[str] = None,
    source_config: Optional[str] = None,
) -> list[dict]:
    """Query eval_results with optional filters."""
    df = store.table("eval_results")
    if model_id:
        df = df[df["model_id"] == model_id]
    if benchmark_id:
        df = df[df["benchmark_id"] == benchmark_id]
    if source_config:
        df = df[df["source_config"] == source_config]
    return _records(df)


# ------------------------------------------------------------------
# Resolution log
# ------------------------------------------------------------------

def append_resolution_log(store: RegistryStore, entry: dict) -> None:
    row = {**entry, "id": str(uuid.uuid4()), "timestamp": _now()}
    _get_pending(store, "resolution_log").append(row)


# ------------------------------------------------------------------
# Sync runs
# ------------------------------------------------------------------

def start_sync_run(
    store: RegistryStore, source_config: str, rerun: bool
) -> str:
    run_id = str(uuid.uuid4())
    df = store.table("sync_runs")
    row = {
        "id": run_id,
        "source_config": source_config,
        "started_at": _now(),
        "completed_at": None,
        "status": "running",
        "rerun": rerun,
        "entities_created": 0,
        "entities_updated": 0,
        "aliases_created": 0,
        "aliases_updated": 0,
        "errors": json.dumps([]),
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    store.set_table("sync_runs", df)
    return run_id


def finish_sync_run(store: RegistryStore, run_id: str, counts: dict, errors: list) -> None:
    df = store.table("sync_runs")
    df.loc[df["id"] == run_id, "completed_at"] = _now()
    df.loc[df["id"] == run_id, "status"] = "failed" if errors else "completed"
    for col, val in counts.items():
        if col in df.columns:
            df.loc[df["id"] == run_id, col] = val
    df.loc[df["id"] == run_id, "errors"] = json.dumps(errors)
    store.set_table("sync_runs", df)
