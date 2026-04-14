from typing import Optional

from eval_entity_resolver.alias_store import AliasStore
from eval_entity_resolver.models import ResolutionResult, ResolverConfig
from eval_entity_resolver.strategies.exact import exact_match
from eval_entity_resolver.strategies.normalized import normalized_match
from eval_entity_resolver.strategies.fuzzy import fuzzy_match


class Resolver:
    def __init__(self, store: AliasStore, config: Optional[ResolverConfig] = None) -> None:
        self.store = store
        self.config = config or ResolverConfig()

    def resolve(
        self,
        raw_value: str,
        entity_type: str,
        source_config: Optional[str] = None,
    ) -> ResolutionResult:
        # 1. Exact
        canonical_id = exact_match(raw_value, entity_type, source_config, self.store)
        if canonical_id is not None:
            return ResolutionResult(
                raw_value=raw_value,
                entity_type=entity_type,
                source_config=source_config,
                canonical_id=canonical_id,
                strategy="exact",
                confidence=1.0,
            )

        # 2. Normalized (confidence 0.95 — only return if above threshold)
        _NORMALIZED_CONFIDENCE = 0.95
        if _NORMALIZED_CONFIDENCE >= self.config.threshold:
            canonical_id = normalized_match(raw_value, entity_type, self.store, source_config)
            if canonical_id is not None:
                return ResolutionResult(
                    raw_value=raw_value,
                    entity_type=entity_type,
                    source_config=source_config,
                    canonical_id=canonical_id,
                    strategy="normalized",
                    confidence=_NORMALIZED_CONFIDENCE,
                )

        # 3. Fuzzy
        canonical_id, confidence = fuzzy_match(
            raw_value, entity_type, self.config.threshold, self.store, source_config
        )
        if canonical_id is not None:
            return ResolutionResult(
                raw_value=raw_value,
                entity_type=entity_type,
                source_config=source_config,
                canonical_id=canonical_id,
                strategy="fuzzy",
                confidence=confidence,
            )

        # 4. No match
        return ResolutionResult(
            raw_value=raw_value,
            entity_type=entity_type,
            source_config=source_config,
            canonical_id=None,
            strategy="no_match",
            confidence=0.0,
        )
