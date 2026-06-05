"""The bare resolver. Matches a raw value to a canonical id via the
strategy chain (exact → normalized → fuzzy → no_match), and — when
given a `CanonicalStore` — enriches the result with the matched
canonical's metadata, parent edges, model-specific lineage fields,
and quantized-chain root collapse.

The enrichment matches the HTTP API's response shape exactly. Callers
using the resolver standalone get the same `ResolutionResult` they'd
get back from `POST /api/v1/resolve`."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from eval_entity_resolver.alias_store import AliasStore
from eval_entity_resolver.canonical_store import CanonicalStore
from eval_entity_resolver.models import ResolutionResult, ResolverConfig
from eval_entity_resolver.strategies.exact import exact_match
from eval_entity_resolver.strategies.normalized import normalized_match
from eval_entity_resolver.strategies.fuzzy import fuzzy_match


class Resolver:
    def __init__(
        self,
        store: AliasStore,
        config: Optional[ResolverConfig] = None,
        canonical_store: Optional[CanonicalStore] = None,
    ) -> None:
        """`store` is required (alias matching is the resolver's core job).
        `canonical_store` is optional — when provided, results are
        enriched with parent / lineage / metadata fields. Without it,
        only the basic match fields (canonical_id, strategy, confidence)
        are populated."""
        self.store = store
        self.config = config or ResolverConfig()
        self.canonical_store = canonical_store

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        config: Optional[ResolverConfig] = None,
    ) -> "Resolver":
        """Load both alias and canonical stores from a parquet directory
        (e.g. `./fixtures/`) and return a fully-enriching resolver. This
        is the recommended convenience for callers who want the same
        response shape as the HTTP API."""
        return cls(
            AliasStore.from_parquet(path),
            config=config,
            canonical_store=CanonicalStore.from_parquet(path),
        )

    @classmethod
    def from_hf(
        cls,
        repo_id: str,
        config: Optional[ResolverConfig] = None,
    ) -> "Resolver":
        """Load both stores from a HF Dataset repo and return a
        fully-enriching resolver."""
        return cls(
            AliasStore.from_hf(repo_id),
            config=config,
            canonical_store=CanonicalStore.from_hf(repo_id),
        )

    def resolve(
        self,
        raw_value: str,
        entity_type: str,
        source_config: Optional[str] = None,
    ) -> ResolutionResult:
        # 1. Exact
        canonical_id = exact_match(raw_value, entity_type, source_config, self.store)
        if canonical_id is not None:
            return self._enrich(raw_value, entity_type, source_config, canonical_id, "exact", 1.0)

        # 2. Normalized (confidence 0.95 — only return if above threshold)
        _NORMALIZED_CONFIDENCE = 0.95
        if _NORMALIZED_CONFIDENCE >= self.config.threshold:
            canonical_id = normalized_match(raw_value, entity_type, self.store, source_config)
            if canonical_id is not None:
                return self._enrich(
                    raw_value, entity_type, source_config,
                    canonical_id, "normalized", _NORMALIZED_CONFIDENCE,
                )

        # 3. Fuzzy
        canonical_id, confidence, inferred_platform = fuzzy_match(
            raw_value, entity_type, self.config.threshold, self.store, source_config
        )
        if canonical_id is not None:
            result = self._enrich(
                raw_value, entity_type, source_config,
                canonical_id, "fuzzy", confidence,
            )
            # Thread the captured inference_platform onto the result. This is
            # the per-run platform read off an EXPLICIT host token in the raw
            # id (a `together/`-prefix or `-bedrock`-suffix), which WINS — an
            # explicit host token in the id is the strongest per-run platform
            # fact. Only set it when a token was actually present (None
            # otherwise), so non-host ids leave the field untouched.
            if inferred_platform is not None:
                result.inference_platform = inferred_platform
            return result

        # 4. No match
        return ResolutionResult(
            raw_value=raw_value,
            entity_type=entity_type,
            source_config=source_config,
            canonical_id=None,
            strategy="no_match",
            confidence=0.0,
        )

    # ------------------------------------------------------------------
    # Enrichment (no-op when no canonical_store is attached)
    # ------------------------------------------------------------------

    def build_result(
        self,
        raw_value: str,
        entity_type: str,
        source_config: Optional[str],
        canonical_id: str,
        strategy: str,
        confidence: float,
    ) -> ResolutionResult:
        """Construct an enriched `ResolutionResult` for a canonical_id
        the caller already knows — useful for callers that bypass the
        strategy chain (e.g. an alias-table cache hit, an auto-created
        draft) but want the same rich response shape. Identical to the
        enrichment that happens inside `resolve()`."""
        return self._enrich(raw_value, entity_type, source_config, canonical_id, strategy, confidence)

    def _enrich(
        self,
        raw_value: str,
        entity_type: str,
        source_config: Optional[str],
        matched_canonical_id: str,
        strategy: str,
        confidence: float,
    ) -> ResolutionResult:
        """Look up the matched canonical's row and populate the rich
        response fields. When no canonical_store is attached, the rich
        fields stay None and the result has just the basic match info."""
        if self.canonical_store is None:
            return ResolutionResult(
                raw_value=raw_value,
                entity_type=entity_type,
                source_config=source_config,
                canonical_id=matched_canonical_id,
                strategy=strategy,
                confidence=confidence,
            )

        cs = self.canonical_store
        matched_entity = cs.lookup(entity_type, matched_canonical_id)
        review_status = (matched_entity or {}).get("review_status") if matched_entity else None

        if entity_type == "model":
            fields = cs.model_metadata_fields(matched_canonical_id, matched_entity)
            # If the response collapses to a different canonical (root),
            # surface THAT canonical's review_status — keeps the response
            # internally consistent.
            if fields["canonical_id"] != matched_canonical_id:
                root_entity = cs.lookup("model", fields["canonical_id"])
                if root_entity:
                    review_status = root_entity.get("review_status") or review_status
            return ResolutionResult(
                raw_value=raw_value,
                entity_type=entity_type,
                source_config=source_config,
                canonical_id=fields["canonical_id"],
                strategy=strategy,
                confidence=confidence,
                review_status=review_status,
                parent_canonical_id=cs.parent_canonical_id("model", matched_entity),
                resolved_leaf_id=fields["resolved_leaf_id"],
                root_model_id=fields["root_model_id"],
                lineage_origin_org_id=fields["lineage_origin_org_id"],
                # Extended lineage / provenance fields. None-safe .get so a
                # store predating these keys still works.
                model_group_id=fields.get("model_group_id"),
                model_family_id=fields.get("model_family_id"),
                lineage_origin_model_id=fields.get("lineage_origin_model_id"),
                lineage_origin_model_org_id=fields.get("lineage_origin_model_org_id"),
                inference_platform=fields.get("inference_platform"),
                resolution_source=fields.get("resolution_source"),
                resolution_granularity=fields.get("resolution_granularity"),
                parents=fields["parents"],
                open_weights=fields["open_weights"],
                release_date=fields["release_date"],
                params_billions=fields["params_billions"],
                ancestry=cs.compute_ancestry("model", fields["canonical_id"], matched_entity),
                resolution_detail=cs.resolution_detail(
                    "model", fields["canonical_id"], matched_entity=matched_entity
                ),
            )

        # Benchmark: fill in hierarchy-alignment fields (family_key,
        # category) by walking canonical_families. composite_keys stays
        # empty here — see CanonicalStore.benchmark_family_enrichment for
        # why composite computation belongs in the producer.
        if entity_type == "benchmark":
            fam = cs.benchmark_family_enrichment(matched_canonical_id)
            return ResolutionResult(
                raw_value=raw_value,
                entity_type=entity_type,
                source_config=source_config,
                canonical_id=matched_canonical_id,
                strategy=strategy,
                confidence=confidence,
                review_status=review_status,
                parent_canonical_id=cs.parent_canonical_id(entity_type, matched_entity),
                family_key=fam["family_key"],
                category=fam["category"],
                composite_keys=fam["composite_keys"],
                ancestry=cs.compute_ancestry("benchmark", matched_canonical_id, matched_entity),
                resolution_detail=cs.resolution_detail(
                    "benchmark", matched_canonical_id,
                    raw_value=raw_value, matched_entity=matched_entity,
                ),
            )

        # Other non-model types (metric, harness, org, family, composite):
        # parent_canonical_id + review_status, plus ancestry/detail (family
        # carries a composite parent; the rest are roots with empty detail).
        return ResolutionResult(
            raw_value=raw_value,
            entity_type=entity_type,
            source_config=source_config,
            canonical_id=matched_canonical_id,
            strategy=strategy,
            confidence=confidence,
            review_status=review_status,
            parent_canonical_id=cs.parent_canonical_id(entity_type, matched_entity),
            ancestry=cs.compute_ancestry(entity_type, matched_canonical_id, matched_entity),
            resolution_detail=cs.resolution_detail(
                entity_type, matched_canonical_id,
                raw_value=raw_value, matched_entity=matched_entity,
            ),
        )
