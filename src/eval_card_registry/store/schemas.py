"""Empty DataFrame schemas for each table."""
import pandas as pd


_SCHEMAS: dict[str, dict] = {
    "canonical_orgs": {
        "id": pd.StringDtype(),
        "display_name": pd.StringDtype(),
        "parent_org_id": pd.StringDtype(),
        "website": pd.StringDtype(),
        "hf_org": pd.StringDtype(),
        # Org category — `lab` (curated first-party), `community` (multi-model
        # HF org), `individual` (single-person account), `unknown` (auto-
        # created, undecided). Consumers wanting "real labs only" filter on
        # `lab` or on `review_status = reviewed`.
        "kind": pd.StringDtype(),
        "tags": pd.StringDtype(),     # JSON-encoded list
        "metadata": pd.StringDtype(), # JSON-encoded dict
        "review_status": pd.StringDtype(),
        "created_at": pd.StringDtype(),
        "updated_at": pd.StringDtype(),
    },
    "canonical_models": {
        "id": pd.StringDtype(),
        "display_name": pd.StringDtype(),
        "developer": pd.StringDtype(),
        "org_id": pd.StringDtype(),
        "family": pd.StringDtype(),
        "architecture": pd.StringDtype(),
        "params_billions": "float64",
        # JSON-encoded list of parent edges:
        #   [{"id": "...", "relationship": "variant|finetune|quantized|merge|adapter",
        #     "axis": "size|mode|modality|domain|version"}]
        # `relationship: quantized` covers all inference-precision variants
        # (-turbo, -fp8, -int4, -awq, -gguf, etc.) — matches hub-stats's
        # baseModels.relation values and keeps the source/registry encoding
        # consistent. Quants are NOT a sub-axis of `variant`.
        # `axis` is optional and only meaningful for `relationship: variant`.
        # Multi-element lists support merges (multiple parents share the
        # `merge` relationship) and the rare case of a model that's both a
        # family variant AND a finetune of an external base.
        "parents": pd.StringDtype(),
        # Identity root: walk `parents` up following only `quantized` edges.
        # NULL when self has no quantized ancestor (i.e. self IS the identity
        # root). Resolver default-returns this when set; callers wanting the
        # leaf get the un-collapsed canonical id in `resolved_leaf_id`.
        "root_model_id": pd.StringDtype(),
        # Denormalized: `org_id` of the deepest non-`variant` ancestor in
        # `parents`. For Meta-originated models = self.org_id. For
        # finetunes/quants of someone else's weights = the upstream lab's
        # org_id. Recomputed on every refresh; treat as a cache.
        "lineage_origin_org_id": pd.StringDtype(),
        # Open vs closed weights. NULL when unknown. Populated:
        #   - models.dev refresh   → from `open_weights` field directly
        #   - hub-stats refresh    → True iff safetensors/gguf data present
        #   - live hub-stats lookup → same inference as bulk refresh
        # Closed-API models (Anthropic / OpenAI / Google) get False from
        # models.dev. Hand-curated core.yaml entries may set explicitly.
        "open_weights": pd.BooleanDtype(),
        # ISO date (YYYY-MM-DD or YYYY-MM) of the family's earliest known
        # snapshot. Populated automatically by scripts/refresh_from_modelsdev.py
        # from models.dev. Per-snapshot dates remain in metadata.release_dates.
        "release_date": pd.StringDtype(),
        "tags": pd.StringDtype(),     # JSON-encoded list
        "metadata": pd.StringDtype(), # JSON-encoded dict
        "review_status": pd.StringDtype(),
        "created_at": pd.StringDtype(),
        "updated_at": pd.StringDtype(),
    },
    "canonical_benchmarks": {
        "id": pd.StringDtype(),
        "display_name": pd.StringDtype(),
        "description": pd.StringDtype(),
        "dataset_repo": pd.StringDtype(),
        "parent_benchmark_id": pd.StringDtype(),
        "tags": pd.StringDtype(),
        "metadata": pd.StringDtype(),
        "review_status": pd.StringDtype(),
        "created_at": pd.StringDtype(),
        "updated_at": pd.StringDtype(),
    },
    "canonical_metrics": {
        "id": pd.StringDtype(),
        "display_name": pd.StringDtype(),
        "score_type": pd.StringDtype(),
        "lower_is_better": "bool",
        "min_score": "float64",
        "max_score": "float64",
        "metadata": pd.StringDtype(),
        "review_status": pd.StringDtype(),
        "created_at": pd.StringDtype(),
        "updated_at": pd.StringDtype(),
    },
    "eval_harnesses": {
        "id": pd.StringDtype(),
        "display_name": pd.StringDtype(),
        "version": pd.StringDtype(),
        "fork_url": pd.StringDtype(),
        "metadata": pd.StringDtype(),
        "review_status": pd.StringDtype(),
        "created_at": pd.StringDtype(),
        "updated_at": pd.StringDtype(),
    },
    "aliases": {
        "id": pd.StringDtype(),
        "raw_value": pd.StringDtype(),
        "entity_type": pd.StringDtype(),
        "canonical_id": pd.StringDtype(),
        "source_config": pd.StringDtype(),
        "source_field": pd.StringDtype(),
        "status": pd.StringDtype(),
        "strategy": pd.StringDtype(),
        "confidence": "float64",
        "notes": pd.StringDtype(),
        "created_at": pd.StringDtype(),
        "updated_at": pd.StringDtype(),
    },
    "resolution_log": {
        "id": pd.StringDtype(),
        "sync_run_id": pd.StringDtype(),
        "raw_value": pd.StringDtype(),
        "entity_type": pd.StringDtype(),
        "source_config": pd.StringDtype(),
        "strategy": pd.StringDtype(),
        "confidence": "float64",
        "canonical_id": pd.StringDtype(),
        "created_new": "bool",
        "timestamp": pd.StringDtype(),
    },
    "eval_results": {
        "id": pd.StringDtype(),            # hash(evaluation_id + result_index)
        "evaluation_id": pd.StringDtype(), # from EEE record
        "result_index": "Int64",           # position in evaluation_results array
        "source_config": pd.StringDtype(),
        "model_id": pd.StringDtype(),      # resolved canonical ID
        "harness_id": pd.StringDtype(),    # resolved canonical ID (nullable)
        "benchmark_id": pd.StringDtype(),  # resolved canonical ID
        "parent_benchmark_id": pd.StringDtype(),  # nullable
        "metric_id": pd.StringDtype(),     # resolved canonical ID (nullable)
        "benchmark_card_id": pd.StringDtype(),  # FK to auto-benchmarkcard output (nullable)
        "score": "float64",
        "score_details": pd.StringDtype(), # JSON-encoded dict
        "created_at": pd.StringDtype(),
        "updated_at": pd.StringDtype(),
    },
    "sync_runs": {
        "id": pd.StringDtype(),
        "source_config": pd.StringDtype(),
        "started_at": pd.StringDtype(),
        "completed_at": pd.StringDtype(),
        "status": pd.StringDtype(),
        "rerun": "bool",
        "entities_created": "Int64",
        "entities_updated": "Int64",
        "aliases_created": "Int64",
        "aliases_updated": "Int64",
        "errors": pd.StringDtype(),  # JSON-encoded list
    },
}


def empty(table: str) -> pd.DataFrame:
    schema = _SCHEMAS[table]
    return pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in schema.items()})
