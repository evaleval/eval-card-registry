from dataclasses import dataclass
from typing import Literal, Optional


ResolutionStrategy = Literal["exact", "normalized", "fuzzy", "no_match"]
# `composite` and `family` are first-class resolvable entity types
# (they resolve against canonical_composites / canonical_families).
# `slice`/subset is deliberately NOT a type — it stays a parent-only
# alias-fold (see specs/entity-modeling.md); a slice match is surfaced
# as resolution detail, not as its own entity.
EntityType = Literal[
    "model", "benchmark", "metric", "harness", "org", "composite", "family"
]


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
        when the matched leaf IS the root. Models only. DEPRECATED
        output alias — equals `model_group_id`; drop once the producer
        is live.
      - `model_group_id`: identity-group root (fold {version, quantized,
        mode}); the rename target of `root_model_id`. Models only.
      - `model_family_id`: family-release root (fold the versioned
        release line). Models only.
      - `lineage_origin_model_id`: deepest non-variant ancestor's id
        (what it was built from). Models only.
      - `lineage_origin_org_id`: deepest non-variant ancestor's
        org_id. Models only. DEPRECATED output alias — equals
        `lineage_origin_model_org_id`.
      - `lineage_origin_model_org_id`: deepest non-variant ancestor's
        org_id; the rename target of `lineage_origin_org_id`. Models only.
      - `inference_platform`: serving platform (FK→inference_platforms.id).
        Models only.
      - `resolution_source`: enum {hf|models_dev|curated|inferred|none}.
      - `resolution_granularity`: enum {variant|group|family}.
      - `parents`: full typed-edge list of the matched leaf. Models only.
      - `open_weights`: True/False/None. Models only.
      - `release_date`: YYYY-MM or YYYY-MM-DD. Models only.
      - `params_billions`: approximate parameter count. Models only.
      - `family_key`: canonical_families.id this benchmark belongs to.
        Defaults to the benchmark's own id for singleton families
        (when no curated multi-benchmark family covers it). Benchmarks
        only.
      - `composite_keys`: canonical_composites.id values where this
        benchmark appears (via the composite's source_configs ↔ EEE
        folders chain). Benchmarks only; empty list when none.
      - `category`: curated single-valued category from the family
        (general / agentic / reasoning / knowledge / multimodal /
        tool-use / math / security / factuality / reward-modelling /
        safety / code / instruction-following / other). Benchmarks
        only; None when no category curated.
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
    # Extended lineage / provenance fields (all Optional[str]=None).
    # `model_group_id` / `lineage_origin_model_org_id` are the rename targets
    # of the deprecated `root_model_id` / `lineage_origin_org_id` (kept above
    # for compat).
    model_group_id: Optional[str] = None
    model_family_id: Optional[str] = None
    lineage_origin_model_id: Optional[str] = None
    lineage_origin_model_org_id: Optional[str] = None
    inference_platform: Optional[str] = None
    resolution_source: Optional[str] = None
    resolution_granularity: Optional[str] = None
    parents: Optional[list[dict]] = None
    open_weights: Optional[bool] = None
    release_date: Optional[str] = None
    params_billions: Optional[float] = None
    # Benchmark-only enrichment.
    family_key: Optional[str] = None
    composite_keys: Optional[list[str]] = None
    category: Optional[str] = None
    # --- Hierarchy contract (type-agnostic ancestry + typed detail) ---
    # `ancestry`: ordered list of `{canonical_id, level}` from the matched
    # entity's IMMEDIATE PARENT up to the root. `[]` when self is a root.
    #   model     -> e.g. [{group}, {family}]
    #   benchmark -> e.g. [{family}, {composite}]
    #   family    -> e.g. [{composite}]
    #   composite/metric/harness/org -> [] (roots)
    # Computed by `CanonicalStore.compute_ancestry` from the existing
    # graph tables (model group/family walk; benchmark→family via
    # canonical_families.benchmark_ids; family→composite via
    # canonical_families.composite_keys / canonical_composites.family_id).
    ancestry: Optional[list[dict]] = None
    # `resolution_detail`: typed sub-object keyed by entity_type.
    #   model     -> {"granularity": variant|group|family}
    #   benchmark -> {"level": composite|family|benchmark|slice,
    #                 "matched_subset": str|None}
    #   composite|family|metric|harness|org -> {} (reserved)
    resolution_detail: Optional[dict] = None


@dataclass
class ResolverConfig:
    threshold: float = 0.85
