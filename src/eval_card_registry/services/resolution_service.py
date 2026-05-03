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
    "benchmark": "parent_benchmark_id",
    "org": "parent_org_id",
}


def _decode_parents(value) -> list[dict]:
    """Decode `canonical_models.parents` (JSON-encoded list-of-edges) to a
    Python list. Tolerant of NaN, None, empty strings, and pre-decoded lists."""
    if value is None or queries._is_na(value):
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


def _variant_parent_id(parents: list[dict]) -> Optional[str]:
    """Return the id of the first `variant` edge in a parents list, or None.
    The ResolveResponse `parent_canonical_id` field exposes the family /
    variant hierarchy that callers historically read off `parent_model_id`."""
    for edge in parents:
        if isinstance(edge, dict) and edge.get("relationship") == "variant":
            pid = edge.get("id")
            if pid:
                return pid
    return None


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
        "resolved_leaf_id": None,
        "root_model_id": None,
        "lineage_origin_org_id": None,
        "parents": None,
        "open_weights": None,
        "release_date": None,
        "params_billions": None,
    }


def _na_to_none(value):
    return None if queries._is_na(value) else value


def _model_response_fields(
    store: RegistryStore,
    matched_canonical_id: str,
    matched_entity: Optional[dict],
) -> dict:
    """Compute model-specific response fields. When the matched canonical
    has `root_model_id` set, the returned `canonical_id` is the root (=
    identity-root collapsing for quantization chains); otherwise it's the
    matched leaf itself. `resolved_leaf_id` always carries the original
    match so callers wanting leaf-level can opt in."""
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
    parents_decoded = queries.decode_parents(matched_entity.get("parents")) or None

    if leaf_root:
        # Resolve to root by default. Source the metadata fields from the
        # root entity since `canonical_id` is the root — keeps the response
        # internally consistent (the canonical_id and its metadata refer
        # to the same model). Quantization preserves identity, so root's
        # release_date / params / open_weights all describe the same
        # model the caller is being told about. Caller wanting leaf-
        # specific info reads `resolved_leaf_id` and does a follow-up GET.
        root_entity = queries.get_entity(store, _ENTITY_TABLE["model"], leaf_root) or {}
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


def _match_result(
    canonical_id: str,
    strategy: str,
    confidence: float,
    review_status: Optional[str],
    created_new: bool = False,
    parent_canonical_id: Optional[str] = None,
    resolved_leaf_id: Optional[str] = None,
    root_model_id: Optional[str] = None,
    lineage_origin_org_id: Optional[str] = None,
    parents: Optional[list] = None,
    open_weights: Optional[bool] = None,
    release_date: Optional[str] = None,
    params_billions: Optional[float] = None,
) -> dict:
    return {
        "canonical_id": canonical_id,
        "strategy": strategy,
        "confidence": confidence,
        "created_new": created_new,
        "review_status": review_status,
        "parent_canonical_id": parent_canonical_id,
        "resolved_leaf_id": resolved_leaf_id,
        "root_model_id": root_model_id,
        "lineage_origin_org_id": lineage_origin_org_id,
        "parents": parents,
        "open_weights": open_weights,
        "release_date": release_date,
        "params_billions": params_billions,
    }


def _build_match(
    store: RegistryStore,
    entity_type: str,
    matched_canonical_id: str,
    matched_entity: Optional[dict],
    strategy: str,
    confidence: float,
    *,
    created_new: bool = False,
    review_status_override: Optional[str] = None,
) -> dict:
    """Construct a resolve response dict, applying model-specific
    root-collapsing when relevant. Non-model entity types pass through
    with `canonical_id = matched_canonical_id` and the new fields NULL."""
    review_status = review_status_override if review_status_override is not None else (
        matched_entity.get("review_status") if matched_entity else None
    )
    if entity_type == "model":
        fields = _model_response_fields(store, matched_canonical_id, matched_entity)
        # Re-fetch the entity at the response canonical (root may differ from matched leaf)
        # so review_status reflects the returned canonical when caller surfaces it.
        if review_status_override is None and fields["canonical_id"] != matched_canonical_id:
            root_entity = queries.get_entity(store, _ENTITY_TABLE["model"], fields["canonical_id"])
            review_status = (root_entity or {}).get("review_status")
        return _match_result(
            canonical_id=fields["canonical_id"],
            strategy=strategy,
            confidence=confidence,
            review_status=review_status,
            created_new=created_new,
            parent_canonical_id=_parent_canonical_id("model", matched_entity),
            resolved_leaf_id=fields["resolved_leaf_id"],
            root_model_id=fields["root_model_id"],
            lineage_origin_org_id=fields["lineage_origin_org_id"],
            parents=fields["parents"],
            open_weights=fields["open_weights"],
            release_date=fields["release_date"],
            params_billions=fields["params_billions"],
        )
    return _match_result(
        canonical_id=matched_canonical_id,
        strategy=strategy,
        confidence=confidence,
        review_status=review_status,
        created_new=created_new,
        parent_canonical_id=_parent_canonical_id(entity_type, matched_entity),
    )


def _parent_canonical_id(entity_type: str, entity: Optional[dict]) -> Optional[str]:
    if not entity:
        return None
    if entity_type == "model":
        # Models store a typed parents list. Surface the variant edge here so
        # the API contract stays stable for callers reading the family parent.
        return _variant_parent_id(_decode_parents(entity.get("parents")))
    field = _PARENT_FIELD.get(entity_type)
    if not field:
        return None
    value = entity.get(field)
    if queries._is_na(value):
        return None
    return value or None


class ResolutionService:
    def __init__(self, registry_store: RegistryStore) -> None:
        import threading
        self.store = registry_store
        self._resolver: Optional[Resolver] = None
        # Cache: (raw_value, entity_type, source_config) → resolve result dict.
        # Avoids re-running the full strategy chain for duplicate strings
        # (e.g. "Accuracy" appears in every record).
        self._resolve_cache: dict[tuple[str, str, Optional[str]], dict] = {}
        # Hub-stats live-lookup state (built lazily on first use). The
        # indices snapshot the aliases / orgs tables; both get invalidated
        # by `invalidate_resolver()` whenever a new entity is auto-created
        # so subsequent lookups can resolve baseModels against the just-
        # added canonical. Lock guards the lazy build under FastAPI's
        # threadpool executor.
        self._hub_stats_client = None
        self._hub_stats_indices: Optional[tuple[dict[str, str], dict[str, str]]] = None
        self._hub_stats_indices_lock = threading.Lock()

    def _get_resolver(self) -> Resolver:
        if self._resolver is None:
            alias_store = _build_alias_store(self.store)
            config = ResolverConfig(threshold=settings.resolver_auto_merge_threshold)
            self._resolver = Resolver(alias_store, config)
        return self._resolver

    def invalidate_resolver(self) -> None:
        """Call after alias or entity changes to force resolver rebuild.
        Also clears the hub-stats indices cache so subsequent live lookups
        can resolve `baseModels` parents against just-added canonicals
        (e.g. when EEE sync creates a parent draft, then sees a child
        whose baseModels references that parent in the same run)."""
        self._resolver = None
        with self._hub_stats_indices_lock:
            self._hub_stats_indices = None

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
                result_dict = _build_match(
                    self.store, entity_type, result.canonical_id, entity,
                    result.strategy, result.confidence,
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
                result_dict = _build_match(
                    self.store, entity_type, existing["canonical_id"], entity,
                    existing["strategy"], existing["confidence"],
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
        result_dict = _build_match(
            self.store, entity_type, canonical_id, entity,
            strategy_used, result.confidence,
            created_new=created_new,
            review_status_override=(entity.get("review_status") if entity else "draft"),
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
        # Hub-stats live enrichment: when a model raw value looks like an
        # HF id, query hub-stats for release_date / params / parents /
        # lineage_origin_org_id and merge into the base draft. Best-effort
        # — `enrichment` is `{}` on lookup miss or any error.
        enrichment: dict = {}
        if entity_type == "model" and self._looks_like_hf_id(raw_value):
            enrichment = self._lookup_hub_stats(raw_value) or {}
        if entity_type == "model":
            base.update({
                "developer": None,
                "org_id": self._resolve_model_org_id(raw_value),
                "family": None,
                "architecture": None,
                "params_billions": None,
                "parents": "[]",
                "root_model_id": None,
                "lineage_origin_org_id": None,
                "open_weights": None,
                "tags": "[]",
            })
            # Apply hub-stats enrichment last so its non-empty values
            # override the defaults we just set. The enrichment dict
            # only contains keys hub-stats actually had data for; other
            # defaults (None / "[]") survive.
            for k, v in enrichment.items():
                if v is not None:
                    base[k] = v
        elif entity_type == "benchmark":
            base.update({"description": None, "dataset_repo": None, "parent_benchmark_id": None, "tags": "[]"})
        elif entity_type == "metric":
            base.update({"score_type": None, "lower_is_better": False, "min_score": None, "max_score": None})
        elif entity_type == "harness":
            base.update({"version": None, "fork_url": None})
        elif entity_type == "org":
            base.update({
                "parent_org_id": None,
                "website": None,
                "hf_org": None,
                "kind": "unknown",
                "tags": "[]",
            })

        queries.upsert_entity(self.store, table, base, buffered=True)
        return candidate_id

    @staticmethod
    def _looks_like_hf_id(raw_value: str) -> bool:
        """HF id heuristic: contains a single `/` with non-empty parts on
        both sides. Conservative — won't trigger hub-stats lookups for
        bare model names or paths with multiple slashes (which are likely
        malformed)."""
        if not raw_value or raw_value.count("/") != 1:
            return False
        org, name = raw_value.split("/", 1)
        return bool(org.strip()) and bool(name.strip())

    def _lookup_hub_stats(self, hf_id: str) -> Optional[dict]:
        """Query hub-stats live for `hf_id` and return a partial draft
        dict (release_date, params_billions, parents, lineage_origin_org_id,
        tags, metadata) ready to merge. Returns None on miss or any error.
        Uses the `aliases` table to resolve baseModels parents to our
        canonical ids, and `canonical_orgs` HF aliases to map authors."""
        if not settings.hub_stats_lookup_enabled:
            return None
        try:
            client = self._get_hub_stats_client()
            row = client.lookup(hf_id)
        except Exception:
            return None
        if row is None:
            return None
        from eval_card_registry.services import hub_stats as _hs
        try:
            aliases_to_canonical, org_alias_map = self._build_hub_stats_indices()
            return _hs.enrich_draft_from_row(row, aliases_to_canonical, org_alias_map)
        except Exception:
            return None

    def _get_hub_stats_client(self):
        """Lazy-init the hub-stats client. Reused across lookups."""
        if self._hub_stats_client is None:
            from eval_card_registry.services.hub_stats import HubStatsClient
            self._hub_stats_client = HubStatsClient()
        return self._hub_stats_client

    def _build_hub_stats_indices(self) -> tuple[dict[str, str], dict[str, str]]:
        """Cache + return the indices `enrich_draft_from_row` needs:
        - normalized canonical-alias → canonical_id (so baseModels parents
          can resolve to our registry's ids)
        - normalized HF org alias → canonical org_id (so author-org
          mapping picks the right slug)
        Built lazily, cached until `invalidate_resolver()` clears it.
        Lock-guarded so two concurrent threads (FastAPI threadpool) don't
        race the lazy build."""
        # Fast path: check without taking the lock to avoid the contention
        # cost on the hot path where the cache is already populated.
        cached = self._hub_stats_indices
        if cached is not None:
            return cached
        with self._hub_stats_indices_lock:
            # Double-check after acquiring — another thread may have built it.
            if self._hub_stats_indices is not None:
                return self._hub_stats_indices
            from eval_card_registry.services.hub_stats import normalize as _hsnorm

            aliases_df = self.store.table("aliases")
            models_df = self.store.table("canonical_models")
            orgs_df = self.store.table("canonical_orgs")

            a2c: dict[str, str] = {}
            # Aliases: only model-typed and HF-shaped (containing `/`)
            for _, row in aliases_df.iterrows():
                if row.get("entity_type") != "model":
                    continue
                raw = row.get("raw_value")
                cid = row.get("canonical_id")
                if isinstance(raw, str) and "/" in raw and isinstance(cid, str):
                    a2c.setdefault(_hsnorm(raw), cid)
            # Canonical ids themselves
            for _, row in models_df.iterrows():
                cid = row.get("id")
                if isinstance(cid, str):
                    a2c.setdefault(_hsnorm(cid), cid)

            org_map: dict[str, str] = {}
            for _, row in orgs_df.iterrows():
                cid = row.get("id")
                if not isinstance(cid, str):
                    continue
                org_map[_hsnorm(cid)] = cid
                hf_org = row.get("hf_org")
                if isinstance(hf_org, str):
                    org_map[_hsnorm(hf_org)] = cid

            self._hub_stats_indices = (a2c, org_map)
            return self._hub_stats_indices

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
