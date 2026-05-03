from typing import Any, Literal, Optional
from pydantic import BaseModel


EntityType = Literal["benchmark", "model", "metric", "harness", "org"]
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


class ResolveResponse(BaseModel):
    # `canonical_id` is what callers should use by default. For models, this
    # is the IDENTITY ROOT — the unquantized base — when the matched leaf
    # has a `root_model_id` set; otherwise it's the matched leaf itself.
    # The split lets callers reason about same-identity quantizations
    # without conflating finetunes/merges (those land on their own leaf).
    canonical_id: Optional[str]
    strategy: str
    confidence: float
    created_new: bool
    review_status: Optional[str]
    # Family / variant parent (the curated hierarchy edge — preserved for
    # backwards compatibility with callers reading the "family parent" id).
    parent_canonical_id: Optional[str] = None
    # The actual canonical that the raw value matched, before any
    # root-collapsing. Equals `canonical_id` when there's no quantized chain.
    # Models only.
    resolved_leaf_id: Optional[str] = None
    # Identity root for the matched canonical (NULL when self IS the root).
    # Models only.
    root_model_id: Optional[str] = None
    # Org id of the deepest non-variant ancestor — captures upstream lab
    # for finetunes/quants of someone else's weights. Models only.
    lineage_origin_org_id: Optional[str] = None
    # Full parents edge list of the matched leaf. Models only.
    parents: Optional[list[ParentEdge]] = None
    # Whether the resolved model has downloadable weights. NULL when
    # unknown. Lets callers filter "open weights only" without a follow-up
    # GET. Models only.
    open_weights: Optional[bool] = None


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
    root_model_id: Optional[str] = None
    lineage_origin_org_id: Optional[str] = None
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
    root_model_id: Optional[str] = None
    lineage_origin_org_id: Optional[str] = None
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
