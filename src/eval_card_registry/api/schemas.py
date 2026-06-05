from typing import Any, Literal, Optional
from pydantic import BaseModel


EntityType = Literal[
    "benchmark", "model", "metric", "harness", "org", "composite", "family"
]
ReviewStatus = Literal["draft", "reviewed"]
AliasStatus = Literal["auto", "uncertain", "confirmed", "rejected"]
ParentRelationship = Literal["variant", "finetune", "quantized", "merge", "adapter"]
ParentAxis = Literal["size", "mode", "modality", "domain", "version"]
OrgKind = Literal["lab", "community", "individual", "unknown"]


class ParentEdge(BaseModel):
    id: str
    relationship: ParentRelationship
    axis: Optional[ParentAxis] = None


# --- Resolve ---

class ResolveRequest(BaseModel):
    raw_value: str
    entity_type: EntityType
    source_config: Optional[str] = None
    source_field: Optional[str] = None


# --- D1 lean resolve contract (post independent review) ---
# The HTTP resolve response is a TYPE-AGNOSTIC CORE (identical shape for
# all entity types) + an ordered `ancestry` chain + a typed
# `resolution_detail` sub-object. Type-specific ENTITY structure
# (group/family/lineage/params for models; family_key/composite_keys/
# category for benchmarks; members for families/composites) lives on the
# entity GET endpoints — never on resolve. The in-process
# `ResolutionResult` stays the rich union (producer path-dep); the route
# projects it down to this lean shape.

AncestryLevel = Literal["group", "family", "composite", "benchmark", "slice"]


class AncestryEntry(BaseModel):
    """One hop in the ancestry chain — a parent canonical and the level it
    sits at, ordered from the matched entity's immediate parent up to the
    root."""
    canonical_id: str
    level: AncestryLevel


class ModelResolutionDetail(BaseModel):
    """Type-specific resolution detail for a model match."""
    granularity: Optional[str] = None  # variant | group | family


class BenchmarkResolutionDetail(BaseModel):
    """Type-specific resolution detail for a benchmark match. `level=slice`
    + `matched_subset` is how a subset / alias-fold match (e.g. an MMLU
    subject folded onto the `mmlu` parent) is surfaced without minting a
    slice entity — a subset is a parent-only alias-fold, never its own canonical."""
    level: Optional[str] = None  # composite | family | benchmark | slice
    matched_subset: Optional[str] = None


class ResolveResponse(BaseModel):
    # Echo of the raw input.
    raw_value: Optional[str] = None
    # Echo of the requested entity type — tells the caller which entity
    # endpoint(s) to follow for detail.
    entity_type: Optional[str] = None
    # The matched canonical id (None on no_match). For models this is the
    # exact matched leaf; group/family membership is carried in `ancestry`.
    canonical_id: Optional[str] = None
    strategy: str
    confidence: float
    created_new: bool
    resolution_source: Optional[str] = None
    review_status: Optional[str] = None
    # Ordered hierarchy chain from the matched entity's IMMEDIATE PARENT up
    # to the root. `[]` when self is a root. Replaces the single
    # `parent_canonical_id`: the only genuinely type-agnostic way to carry
    # a 1-hop model edge AND a multi-level benchmark tree.
    #   model     -> e.g. [{group}, {family}]
    #   benchmark -> e.g. [{family}, {composite}]
    ancestry: list[AncestryEntry] = []
    # Typed resolution detail, schema selected by `entity_type`. `{}` for
    # composite / family / metric / harness / org (reserved).
    resolution_detail: dict[str, Any] = {}


# --- Entities ---

class ModelCreate(BaseModel):
    id: str
    display_name: str
    developer: Optional[str] = None
    org_id: Optional[str] = None
    family: Optional[str] = None
    architecture: Optional[str] = None
    params_billions: Optional[float] = None
    parents: list[ParentEdge] = []
    model_group_id: Optional[str] = None
    lineage_origin_model_org_id: Optional[str] = None
    open_weights: Optional[bool] = None
    release_date: Optional[str] = None
    tags: list[str] = []
    metadata: dict[str, Any] = {}
    review_status: str = "draft"


class ModelPatch(BaseModel):
    display_name: Optional[str] = None
    developer: Optional[str] = None
    org_id: Optional[str] = None
    family: Optional[str] = None
    architecture: Optional[str] = None
    params_billions: Optional[float] = None
    parents: Optional[list[ParentEdge]] = None
    model_group_id: Optional[str] = None
    lineage_origin_model_org_id: Optional[str] = None
    open_weights: Optional[bool] = None
    release_date: Optional[str] = None
    tags: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None
    review_status: Optional[str] = None


class BenchmarkCreate(BaseModel):
    id: str
    display_name: str
    description: Optional[str] = None
    dataset_repo: Optional[str] = None
    parent_benchmark_id: Optional[str] = None
    tags: list[str] = []
    metadata: dict[str, Any] = {}
    review_status: str = "draft"


class BenchmarkPatch(BaseModel):
    display_name: Optional[str] = None
    description: Optional[str] = None
    dataset_repo: Optional[str] = None
    parent_benchmark_id: Optional[str] = None
    tags: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None
    review_status: Optional[str] = None


class MetricCreate(BaseModel):
    id: str
    display_name: str
    score_type: Optional[str] = None
    lower_is_better: bool = False
    min_score: Optional[float] = None
    max_score: Optional[float] = None
    metadata: dict[str, Any] = {}
    review_status: str = "draft"


class MetricPatch(BaseModel):
    display_name: Optional[str] = None
    score_type: Optional[str] = None
    lower_is_better: Optional[bool] = None
    min_score: Optional[float] = None
    max_score: Optional[float] = None
    metadata: Optional[dict[str, Any]] = None
    review_status: Optional[str] = None


class HarnessCreate(BaseModel):
    id: str
    display_name: str
    version: Optional[str] = None
    fork_url: Optional[str] = None
    metadata: dict[str, Any] = {}
    review_status: str = "draft"


class HarnessPatch(BaseModel):
    display_name: Optional[str] = None
    version: Optional[str] = None
    fork_url: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    review_status: Optional[str] = None


# --- Orgs ---

class OrgCreate(BaseModel):
    id: str
    display_name: str
    parent_org_id: Optional[str] = None
    website: Optional[str] = None
    hf_org: Optional[str] = None
    kind: OrgKind = "unknown"
    tags: list[str] = []
    metadata: dict[str, Any] = {}
    review_status: str = "draft"


class OrgPatch(BaseModel):
    display_name: Optional[str] = None
    parent_org_id: Optional[str] = None
    website: Optional[str] = None
    hf_org: Optional[str] = None
    kind: Optional[OrgKind] = None
    tags: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None
    review_status: Optional[str] = None


# --- Aliases ---

class AliasPatch(BaseModel):
    canonical_id: Optional[str] = None
    status: Optional[str] = None
    notes: Optional[str] = None
