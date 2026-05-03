from dataclasses import dataclass
from typing import Literal, Optional


ResolutionStrategy = Literal["exact", "normalized", "fuzzy", "no_match"]
EntityType = Literal["model", "benchmark", "metric", "harness", "org"]


@dataclass
class ResolutionResult:
    """Outcome of one `Resolver.resolve` call.

    Core matching fields (always populated):
      - `raw_value`, `entity_type`, `source_config`: echo of the inputs
      - `canonical_id`: the matched canonical (None on no_match). For
        models with a `root_model_id` set, this is the IDENTITY ROOT —
        i.e. the unquantized base — so callers reasoning about
        same-identity quantizations get one canonical instead of N.
      - `strategy`: which matcher fired (or "no_match")
      - `confidence`: 0.0–1.0

    Enrichment fields (populated when the `Resolver` is constructed
    with a `CanonicalStore`; otherwise None):
      - `review_status`: review state of the matched canonical
      - `parent_canonical_id`: family/variant parent (for models, the
        `variant` edge in `parents`; for benchmarks/orgs, the
        `parent_*_id` scalar column).
      - `resolved_leaf_id`: the originally-matched canonical before
        any root-collapse. Equals `canonical_id` when no quantized
        chain. Models only.
      - `root_model_id`: identity root via quantized-only walk. NULL
        when the matched leaf IS the root. Models only.
      - `lineage_origin_org_id`: deepest non-variant ancestor's
        org_id. Models only.
      - `parents`: full typed-edge list of the matched leaf. Models only.
      - `open_weights`: True/False/None. Models only.
      - `release_date`: YYYY-MM or YYYY-MM-DD. Models only.
      - `params_billions`: approximate parameter count. Models only.
    """
    raw_value: str
    entity_type: EntityType
    source_config: Optional[str]
    canonical_id: Optional[str]
    strategy: ResolutionStrategy
    confidence: float
    # Enrichment fields — None when no CanonicalStore is attached.
    review_status: Optional[str] = None
    parent_canonical_id: Optional[str] = None
    resolved_leaf_id: Optional[str] = None
    root_model_id: Optional[str] = None
    lineage_origin_org_id: Optional[str] = None
    parents: Optional[list[dict]] = None
    open_weights: Optional[bool] = None
    release_date: Optional[str] = None
    params_billions: Optional[float] = None


@dataclass
class ResolverConfig:
    threshold: float = 0.85
