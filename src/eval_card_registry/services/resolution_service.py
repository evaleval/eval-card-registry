"""
resolution_service: wraps the eval-entity-resolver package.

Responsibilities:
- Call the resolver
- Auto-create draft canonical entities when resolver returns no_match
- Write aliases for every resolution (add on first resolve, update on rerun)
- Append to the resolution log
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Optional

from eval_entity_resolver import AliasStore, Resolver, ResolverConfig, ResolutionResult

from eval_card_registry.config import settings
from eval_card_registry.store.hf_store import RegistryStore
from eval_card_registry.store import queries


# Map entity_type to table name
_ENTITY_TABLE = {
    "model": "canonical_models",
    "benchmark": "canonical_benchmarks",
    "metric": "canonical_metrics",
    "harness": "eval_harnesses",
    "org": "canonical_orgs",
}

_PARENT_FIELD = {
    "model": "parent_model_id",
    "benchmark": "parent_benchmark_id",
    "org": "parent_org_id",
}


def _slugify(value: str) -> str:
    """
    Produce a lowercase slug for auto-created entity IDs.
    Falls back to a UUID-derived ID if the input reduces to nothing (e.g. all punctuation).
    """
    slug = value.lower().strip()
    slug = re.sub(r"[^\w\s\-/]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = slug.strip("-")  # trim leading/trailing dashes
    if not slug:
        slug = f"auto-{str(uuid.uuid4())[:8]}"
    return slug


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_alias_store(registry_store: RegistryStore) -> AliasStore:
    """Build an AliasStore from the registry's in-memory aliases table."""
    aliases_df = registry_store.table("aliases")
    return AliasStore(aliases_df, read_only=True)


def _no_match_result() -> dict:
    return {
        "canonical_id": None,
        "strategy": "no_match",
        "confidence": 0.0,
        "created_new": False,
        "review_status": None,
        "parent_canonical_id": None,
    }


def _match_result(
    canonical_id: str,
    strategy: str,
    confidence: float,
    review_status: Optional[str],
    created_new: bool = False,
    parent_canonical_id: Optional[str] = None,
) -> dict:
    return {
        "canonical_id": canonical_id,
        "strategy": strategy,
        "confidence": confidence,
        "created_new": created_new,
        "review_status": review_status,
        "parent_canonical_id": parent_canonical_id,
    }


def _parent_canonical_id(entity_type: str, entity: Optional[dict]) -> Optional[str]:
    if not entity:
        return None
    field = _PARENT_FIELD.get(entity_type)
    if not field:
        return None
    value = entity.get(field)
    if queries._is_na(value):
        return None
    return value or None


class ResolutionService:
    def __init__(self, registry_store: RegistryStore) -> None:
        self.store = registry_store
        self._resolver: Optional[Resolver] = None
        # Cache: (raw_value, entity_type, source_config) → resolve result dict.
        # Avoids re-running the full strategy chain for duplicate strings
        # (e.g. "Accuracy" appears in every record).
        self._resolve_cache: dict[tuple[str, str, Optional[str]], dict] = {}

    def _get_resolver(self) -> Resolver:
        if self._resolver is None:
            alias_store = _build_alias_store(self.store)
            config = ResolverConfig(threshold=settings.resolver_auto_merge_threshold)
            self._resolver = Resolver(alias_store, config)
        return self._resolver

    def invalidate_resolver(self) -> None:
        """Call after alias or entity changes to force resolver rebuild."""
        self._resolver = None

    def resolve(
        self,
        raw_value: str,
        entity_type: str,
        source_config: Optional[str],
        source_field: Optional[str],
        sync_run_id: Optional[str] = None,
        rerun: bool = False,
    ) -> dict:
        """
        Resolve a raw value to a canonical entity. Returns a dict with:
        - canonical_id, strategy, confidence, created_new, review_status
        """
        if not raw_value or not raw_value.strip():
            return _no_match_result()

        # Fast path: return cached result for duplicate (raw_value, entity_type, source_config)
        cache_key = (raw_value, entity_type, source_config)
        if not rerun and cache_key in self._resolve_cache:
            return self._resolve_cache[cache_key]

        # Read-only mode: resolve only, no side effects on entity data
        if settings.read_only:
            resolver = self._get_resolver()
            result: ResolutionResult = resolver.resolve(raw_value, entity_type, source_config)
            if result.canonical_id is not None:
                entity = queries.get_entity(self.store, _ENTITY_TABLE[entity_type], result.canonical_id)
                result_dict = _match_result(
                    result.canonical_id,
                    result.strategy,
                    result.confidence,
                    entity.get("review_status") if entity else None,
                    parent_canonical_id=_parent_canonical_id(entity_type, entity),
                )
            else:
                result_dict = _no_match_result()
            self._resolve_cache[cache_key] = result_dict
            return result_dict

        # Check if alias already exists (skip resolver on rerun=False)
        if not rerun:
            existing = queries.get_alias(self.store, raw_value, entity_type, source_config)
            if existing:
                entity = queries.get_entity(self.store, _ENTITY_TABLE[entity_type], existing["canonical_id"])
                result_dict = _match_result(
                    existing["canonical_id"],
                    existing["strategy"],
                    existing["confidence"],
                    entity.get("review_status") if entity else None,
                    parent_canonical_id=_parent_canonical_id(entity_type, entity),
                )
                self._resolve_cache[cache_key] = result_dict
                return result_dict

        resolver = self._get_resolver()
        result = resolver.resolve(raw_value, entity_type, source_config)

        created_new = False
        alias_status = "auto"

        if result.canonical_id is not None:
            # Match found above threshold
            canonical_id = result.canonical_id
            alias_status = "auto"
        else:
            # No match — auto-create draft entity
            canonical_id = self._auto_create_entity(entity_type, raw_value)
            alias_status = "uncertain"
            created_new = True

        strategy_used = result.strategy if result.canonical_id else "auto_draft"

        # Write alias (buffered during sync for performance)
        alias_data = {
            "raw_value": raw_value,
            "entity_type": entity_type,
            "canonical_id": canonical_id,
            "source_config": source_config,
            "source_field": source_field,
            "status": alias_status,
            "strategy": strategy_used,
            "confidence": result.confidence,
            "notes": None,
        }
        if rerun:
            existing_alias_id = self._find_alias_id(raw_value, entity_type, source_config)
            if existing_alias_id:
                queries.update_alias(
                    self.store,
                    alias_id=existing_alias_id,
                    updates={
                        "canonical_id": canonical_id,
                        "status": alias_status,
                        "strategy": strategy_used,
                        "confidence": result.confidence,
                    },
                )
            else:
                try:
                    queries.add_alias(self.store, alias_data, buffered=True)
                except ValueError:
                    pass
        else:
            try:
                queries.add_alias(self.store, alias_data, buffered=True)
            except ValueError:
                pass  # alias already exists (from prior resolution in this run)

        # Log
        if sync_run_id:
            queries.append_resolution_log(
                self.store,
                {
                    "sync_run_id": sync_run_id,
                    "raw_value": raw_value,
                    "entity_type": entity_type,
                    "source_config": source_config,
                    "strategy": strategy_used,
                    "confidence": result.confidence,
                    "canonical_id": canonical_id,
                    "created_new": created_new,
                },
            )

        # Only invalidate when a new entity was created — its alias could
        # help future fuzzy matches.  Scoped aliases for existing entities
        # don't affect lookups for other raw values.
        if created_new:
            self.invalidate_resolver()

        entity = queries.get_entity(self.store, _ENTITY_TABLE[entity_type], canonical_id)
        result_dict = _match_result(
            canonical_id,
            strategy_used,
            result.confidence,
            entity.get("review_status") if entity else "draft",
            created_new=created_new,
            parent_canonical_id=_parent_canonical_id(entity_type, entity),
        )
        self._resolve_cache[cache_key] = result_dict
        return result_dict

    def _auto_create_entity(self, entity_type: str, raw_value: str) -> str:
        table = _ENTITY_TABLE[entity_type]
        candidate_id = _slugify(raw_value)
        # Ensure uniqueness
        df = self.store.table(table)
        if (df["id"] == candidate_id).any():
            candidate_id = f"{candidate_id}-{str(uuid.uuid4())[:8]}"

        now = _now()
        base = {
            "id": candidate_id,
            "display_name": raw_value,
            "metadata": "{}",
            "review_status": "draft",
            "created_at": now,
            "updated_at": now,
        }
        if entity_type == "model":
            base.update({
                "developer": None,
                "org_id": self._resolve_model_org_id(raw_value),
                "family": None,
                "architecture": None,
                "params_billions": None,
                "parent_model_id": None,
                "tags": "[]",
            })
        elif entity_type == "benchmark":
            base.update({"description": None, "dataset_repo": None, "parent_benchmark_id": None, "tags": "[]"})
        elif entity_type == "metric":
            base.update({"score_type": None, "lower_is_better": False, "min_score": None, "max_score": None})
        elif entity_type == "harness":
            base.update({"version": None, "fork_url": None})
        elif entity_type == "org":
            base.update({"parent_org_id": None, "website": None, "hf_org": None, "tags": "[]"})

        queries.upsert_entity(self.store, table, base, buffered=True)
        return candidate_id

    def _resolve_model_org_id(self, raw_value: str) -> Optional[str]:
        if "/" not in raw_value:
            return None
        raw_org = raw_value.split("/", 1)[0].strip()
        if not raw_org:
            return None
        result = self._get_resolver().resolve(raw_org, "org", None)
        return result.canonical_id

    def _find_alias_id(
        self,
        raw_value: str,
        entity_type: str,
        source_config: Optional[str],
    ) -> Optional[str]:
        df = self.store.table("aliases")
        mask = (df["raw_value"] == raw_value) & (df["entity_type"] == entity_type)
        if source_config:
            mask = mask & (df["source_config"] == source_config)
        else:
            mask = mask & df["source_config"].isna()
        rows = df[mask]
        return rows.iloc[0]["id"] if not rows.empty else None
