from dataclasses import dataclass, field
from typing import Literal, Optional


ResolutionStrategy = Literal["exact", "normalized", "fuzzy", "no_match"]
EntityType = Literal["model", "benchmark", "metric", "harness"]


@dataclass
class ResolutionResult:
    raw_value: str
    entity_type: EntityType
    source_config: Optional[str]
    canonical_id: Optional[str]
    strategy: ResolutionStrategy
    confidence: float


@dataclass
class ResolverConfig:
    threshold: float = 0.85
