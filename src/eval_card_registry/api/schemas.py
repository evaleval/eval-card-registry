from typing import Any, Literal, Optional
from pydantic import BaseModel


EntityType = Literal["benchmark", "model", "metric", "harness", "org"]
ReviewStatus = Literal["draft", "reviewed"]
AliasStatus = Literal["auto", "uncertain", "confirmed", "rejected"]


# --- Resolve ---

class ResolveRequest(BaseModel):
    raw_value: str
    entity_type: EntityType
    source_config: Optional[str] = None
    source_field: Optional[str] = None


class ResolveResponse(BaseModel):
    canonical_id: Optional[str]
    strategy: str
    confidence: float
    created_new: bool
    review_status: Optional[str]
    parent_canonical_id: Optional[str] = None


# --- Entities ---

class ModelCreate(BaseModel):
    id: str
    display_name: str
    developer: Optional[str] = None
    org_id: Optional[str] = None
    family: Optional[str] = None
    architecture: Optional[str] = None
    params_billions: Optional[float] = None
    parent_model_id: Optional[str] = None
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
    parent_model_id: Optional[str] = None
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
    tags: list[str] = []
    metadata: dict[str, Any] = {}
    review_status: str = "draft"


class OrgPatch(BaseModel):
    display_name: Optional[str] = None
    parent_org_id: Optional[str] = None
    website: Optional[str] = None
    hf_org: Optional[str] = None
    tags: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None
    review_status: Optional[str] = None


# --- Aliases ---

class AliasPatch(BaseModel):
    canonical_id: Optional[str] = None
    status: Optional[str] = None
    notes: Optional[str] = None
